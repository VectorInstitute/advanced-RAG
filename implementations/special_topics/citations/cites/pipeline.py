"""Every step of the citation comparison, for a cached reference and a live run alike.

:class:`Pipeline` is what the notebook and ``build_reference.py`` call. The two
model-produced steps, answering and judging, go through
:func:`aieng.advanced_rag.runs.load_or_run`: with ``live`` off you read the
reference result built on the on-prem ``Qwen3_6-35B-A3B``, labelled with what
rebuilding it on your models would cost, and with ``live`` on the step runs on
your generation and judging models. Steps that call no chat model (retrieval,
the shuffled control, gold scoring, and the local MiniCheck checker) go
through :class:`aieng.advanced_rag.cache.TieredResultCache` with every input in
the key.

Every request goes through one governed client per model: a shared response
cache, so a re-run costs nothing, and a cap of :data:`RPM` requests per minute.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import random
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from aieng.advanced_rag.cache import CacheKey, JsonValue, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.evaluation.citations import (
    INLINE_TEMPLATE,
    QUESTION_CLAIM,
    QUOTE_TEMPLATE,
    SUPPORT_PROMPT,
    Citation,
    CitedAnswer,
    GoldScore,
    JudgedAnswer,
    LLMSupportChecker,
    MiniCheckChecker,
    Statement,
    SupportVerdict,
    generate_cited,
    judge_citations,
    parse_inline,
    score_against_gold,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.models import ModelChoice, ModelSet, choose_models
from aieng.advanced_rag.pipeline.generate import GENERATION_TEMPLATE, ChatCompleter
from aieng.advanced_rag.provenance import Provenance
from aieng.advanced_rag.retrieval.chunkers import sentence_spans
from aieng.advanced_rag.runs import Step, StepResult, load_or_run

from . import data, explore


RI_ID = "citations"
#: The on-prem model that answered and judged the reference run.
REFERENCE_MODEL = "Qwen3_6-35B-A3B"
#: Passages placed in every answer prompt, RI #1's evidence budget.
K = 5
MAX_CHARS = 1200
#: Requests per minute for each model; other jobs share the on-prem models.
RPM = 20
#: Questions a live run answers: a stratified subset of the reference set.
LIVE_SUBSET = 40
EMBEDDER = "BAAI/bge-small-en-v1.5"
MINICHECK = "lytang/MiniCheck-Flan-T5-Large"
SHUFFLE_SEED = 0
WORKERS = 4

Style = Literal["answer", "inline", "quotes"]
Which = Literal["all", "subset"]
#: The three citation styles over the top ``K`` retrieved chunks.
STYLES: tuple[Style, ...] = ("answer", "inline", "quotes")
#: Arms whose answers a model writes: the styles, and quotes over the whole agreement.
GENERATED: tuple[str, ...] = (*STYLES, "fulldoc")
#: Arms built from the inline arm's statements with no chat call: two controls
#: (citations moved to another shown passage, or all pointed at the top-ranked
#: one) and citations attached after the fact by MiniCheck.
DERIVED: tuple[str, ...] = ("shuffled", "rank1", "posthoc")
ARMS: tuple[str, ...] = (*GENERATED, *DERIVED)
TEMPLATES: dict[str, str] = {
    "answer": GENERATION_TEMPLATE,
    "inline": INLINE_TEMPLATE,
    "quotes": QUOTE_TEMPLATE,
    "fulldoc": QUOTE_TEMPLATE,
}
#: Support probability at or above which MiniCheck attaches a passage.
POSTHOC_THRESHOLD = 0.5

T = TypeVar("T")
R = TypeVar("R")


def reference_models() -> ModelSet:
    """Return the model set the reference run used."""
    return choose_models("onprem")


def digest_of(*parts: Any) -> str:
    """Return a short digest of prompts or settings, for cache keys."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def parallel_map(function: Callable[[T], R], items: Sequence[T], workers: int | None = None) -> list[R]:
    """Map in threads, each carrying the caller's context so metering follows."""
    with ThreadPoolExecutor(max_workers=workers or WORKERS) as pool:
        futures = [pool.submit(contextvars.copy_context().run, function, item) for item in items]
        return [future.result() for future in futures]


def response_cache_path() -> Path:
    """Return the SQLite file this topic's responses are cached in."""
    return Path(cache_root()) / "responses" / "citations.sqlite"


class Clients:
    """One governed client per model, sharing one response cache and a cap per model."""

    def __init__(self, env_path: Path | None = None, rpm: int | None = None) -> None:
        self._env_path = env_path
        self._rpm = RPM if rpm is None else rpm
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cache: Any = None

    def _client(self, model: str) -> Any:
        # Imported here so a cached run needs neither the client nor a key.
        from aieng.advanced_rag.graphrag.proxy import RateLimiter, ResponseCache, govern  # noqa: PLC0415
        from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        with self._lock:
            if self._cache is None:
                self._cache = ResponseCache(response_cache_path())
            if model not in self._clients:
                client = OpenAI(
                    api_key=resolve_api_key(self._env_path),
                    base_url=resolve_base_url(self._env_path),
                    max_retries=0,
                    timeout=300,
                )
                self._clients[model] = govern(client, cache=self._cache, limiter=RateLimiter(self._rpm))
            return self._clients[model]

    def completer(self, model: str, max_tokens: int) -> ChatCompleter:
        """Return a prompt-to-text callable for one model."""
        return ChatCompleter(self._client(model), model, max_tokens=max_tokens)


class NotCachedError(LookupError):
    """A step that calls no chat model has no cached result, and running it is off."""


@dataclass
class Pipeline:
    """The comparison, step by step, for one model set.

    Parameters
    ----------
    models : ModelSet
        ``generation`` answers in every arm and ``judging`` checks support.
    live : bool, optional
        Run a model step when no result for these models is cached. Pass the
        notebook's ``RUN_EXPENSIVE``.
    local_checker : bool, optional
        Run MiniCheck on answers it has not scored. It downloads about 3 GB of
        weights once and runs on the CPU.
    cache : TieredResultCache, optional
        The artifact cache.
    confirm : bool, optional
        Passed to ``load_or_run``.
    show : bool, optional
        Print ``load_or_run``'s labels.
    env_path : Path or None, optional
        A ``.env`` holding the proxy key, when the environment has none.
    """

    models: ModelSet
    live: bool = False
    local_checker: bool = False
    cache: TieredResultCache = field(default_factory=TieredResultCache)
    confirm: bool = False
    show: bool = True
    env_path: Path | None = None
    _clients: Clients | None = None
    _memo: dict[str, Any] = field(default_factory=dict)

    @property
    def clients(self) -> Clients:
        """Return the proxy clients, built on first use so a cached run needs no key."""
        if self._clients is None:
            self._clients = Clients(self.env_path)
        return self._clients

    # ------------------------------------------------------------------ data

    def _dataset(self) -> tuple[list[data.Item], dict[str, str]]:
        if "dataset" not in self._memo:
            self._memo["dataset"] = data.load()
        return cast(tuple[list[data.Item], dict[str, str]], self._memo["dataset"])

    def items(self, which: Which = "all") -> list[data.Item]:
        """Return the questions of the full set or of the live subset."""
        items = self._dataset()[0]
        return items if which == "all" else data.subset(items, LIVE_SUBSET)

    def item(self, query_id: str) -> data.Item:
        """Return one question by identifier."""
        return next(i for i in self._dataset()[0] if i.query_id == query_id)

    def texts(self) -> dict[str, str]:
        """Return each document's text."""
        return self._dataset()[1]

    def dataset_config(self) -> dict[str, JsonValue]:
        """Return the inputs every key names: split, sample, chunking, and budget."""
        return {
            "split": contract_nli.CONTRACT_NLI_TEST.member_sha256,
            "documents": data.DOCUMENTS,
            "nulls_per_document": data.NULLS_PER_DOCUMENT,
            "seed": data.SEED,
            "question": data.QUESTION,
            "chunk_chars": data.CHUNK_CHARS,
            "overlap_chars": data.OVERLAP_CHARS,
            "k": K,
            "max_chars": MAX_CHARS,
        }

    # ------------------------------------------------------------------ retrieval

    def rankings_key(self) -> CacheKey:
        """Return the key of the dense rankings."""
        return CacheKey(RI_ID, "rankings", "dense", {**self.dataset_config(), "embedder": EMBEDDER, "version": 1})

    def rankings(self) -> dict[str, list[list[Any]]]:
        """Return each question's top ``K`` chunks as ``[chunk_id, start, end]``.

        Retrieval is scoped to the question's own agreement, as in RI #1. It
        embeds 24 agreements with the local ``bge-small-en-v1.5`` in under a
        minute, so it runs when no cached ranking exists.
        """
        if "rankings" not in self._memo:

            def compute() -> JsonValue:
                # sentence-transformers loads only when rankings are rebuilt.
                from aieng.advanced_rag.retrieval import DenseRetriever  # noqa: PLC0415
                from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

                retriever = DenseRetriever(SentenceTransformerEmbedder(EMBEDDER))
                retriever.index(data.chunks(self.texts()))
                ranked: dict[str, JsonValue] = {}
                for item in self.items():
                    hits = retriever.retrieve(item.question, K, scope=item.doc_id)
                    ranked[item.query_id] = [[h.chunk_id, h.provenance.start, h.provenance.end] for h in hits]
                return ranked

            self._memo["rankings"] = self.cache.load_or_compute(self.rankings_key(), compute).value
        return cast(dict[str, list[list[Any]]], self._memo["rankings"])

    def evidence(self, query_id: str, arm: str = "inline") -> list[Evidence]:
        """Return the passages an arm's answer prompt shows, in rank order.

        The ``fulldoc`` arm shows the whole agreement as one source.
        """
        doc_id = query_id.split("::", maxsplit=1)[0]
        text = self.texts()[doc_id]
        if arm.startswith("fulldoc"):
            provenance = Provenance(doc_id=doc_id, chunk_id=f"{doc_id}::full", start=0, end=len(text))
            return [Evidence(text=text, provenance=provenance, retriever="none")]
        return [
            Evidence(
                text=text[start:end],
                provenance=Provenance(doc_id=doc_id, chunk_id=chunk_id, start=start, end=end),
                retriever="dense",
            )
            for chunk_id, start, end in self.rankings()[query_id]
        ]

    # ------------------------------------------------------------------ answers

    def _run(self, step: Step, compute: Callable[[ModelChoice], Any]) -> StepResult:
        return load_or_run(
            step, compute, models=self.models, live=self.live, confirm=self.confirm, cache=self.cache, show=self.show
        )

    def answer_step(self, arm: str, which: Which) -> Step:
        """Return the step that answers every question in one generated arm."""
        if arm in explore.VARIANTS:
            variant: dict[str, JsonValue] = {
                **self.dataset_config(),
                "which": which,
                "template": digest_of(explore.TEMPLATES[arm]),
                "max_tokens": 1200,
                "version": 1,
            }
            base = explore.BASES.get(arm)
            upstream = {"answers": self.answers_digest(base, which)} if base else {}
            return Step(RI_ID, "cited-answers-explore", f"explore/{arm}", variant, role="generation", upstream=upstream)
        style = "quotes" if arm == "fulldoc" else arm
        config: dict[str, JsonValue] = {
            **self.dataset_config(),
            "which": which,
            "style": style,
            "template": digest_of(TEMPLATES[arm]),
            "max_tokens": 700,
            "version": 1,
        }
        if arm == "fulldoc":
            config["context"] = "document"
        return Step(RI_ID, "cited-answers", f"generate_cited/{arm}", config, role="generation")

    def answer_result(self, arm: str, which: Which = "all") -> StepResult:
        """Return one generated arm's answers, from the cache or a live run."""
        memo = f"answers/{arm}/{which}"
        if memo not in self._memo and arm in explore.VARIANTS:
            self._memo[memo] = self._run(self.answer_step(arm, which), self._explore_compute(arm, which))
        if memo not in self._memo:
            items = self.items(which)
            style = cast(Style, "quotes" if arm == "fulldoc" else arm)
            budget, max_chars = (None, None) if arm == "fulldoc" else (K, MAX_CHARS)

            def compute(choice: ModelChoice) -> JsonValue:
                complete = self.clients.completer(choice.model, max_tokens=700)

                def one(item: data.Item) -> JsonValue:
                    evidence = self.evidence(item.query_id, arm)
                    answer = generate_cited(
                        item.query_id,
                        item.question,
                        evidence,
                        complete,
                        style=style,
                        budget=budget,
                        max_chars=max_chars,
                    )
                    return cast(JsonValue, answer.to_dict())

                return {"answers": cast(JsonValue, parallel_map(one, items))}

            self._memo[memo] = self._run(self.answer_step(arm, which), compute)
        return cast(StepResult, self._memo[memo])

    def _explore_compute(self, arm: str, which: Which) -> Callable[[ModelChoice], JsonValue]:
        """Return the compute function of one variant arm."""
        items = self.items(which)
        base = {a.query_id: a for a in self.answers(explore.BASES[arm], which)} if arm in explore.BASES else {}

        def compute(choice: ModelChoice) -> JsonValue:
            complete = self.clients.completer(choice.model, max_tokens=1200)

            def one(item: data.Item) -> JsonValue:
                document = self.evidence(item.query_id, "fulldoc")[0]
                if arm == "fulldoc-sentinel":
                    answer = explore.sentinel(item.query_id, item.question, document, complete)
                elif arm == "fulldoc-extract":
                    answer = explore.extract_then_answer(item.query_id, item.question, document, complete)
                elif arm == "fulldoc-check":
                    answer = explore.self_check(base[item.query_id], item.question, complete)
                else:
                    answer = explore.standalone(base[item.query_id], item.question, complete)
                return cast(JsonValue, answer.to_dict())

            return {"answers": cast(JsonValue, parallel_map(one, items))}

        return compute

    def answers(self, arm: str, which: Which = "all") -> list[CitedAnswer]:
        """Return one arm's answers; the derived arms start from the inline arm's."""
        if arm == "shuffled":
            return shuffle_citations(self.answers("inline", which), self.evidence, SHUFFLE_SEED)
        if arm == "rank1":
            return cite_rank1(self.answers("inline", which), self.evidence)
        if arm == "posthoc":
            return self.posthoc(which)
        value = self.answer_result(arm, which).value
        return [CitedAnswer.from_dict(a) for a in value["answers"]]

    def answers_digest(self, arm: str, which: Which) -> str:
        """Return the digest of the answers an arm's judging step reads."""
        if arm in DERIVED:
            settings: dict[str, tuple[Any, ...]] = {
                "shuffled": ("shuffle", SHUFFLE_SEED),
                "rank1": ("rank1",),
                "posthoc": ("posthoc", MINICHECK, POSTHOC_THRESHOLD),
            }
            return digest_of(self.answer_result("inline", which).digest, *settings[arm])
        return str(self.answer_result(arm, which).digest)

    def posthoc_key(self, which: Which) -> CacheKey:
        """Return the key of the citations MiniCheck attached to inline statements."""
        config: dict[str, JsonValue] = {
            "answers": self.answer_result("inline", which).digest,
            "model": MINICHECK,
            "threshold": POSTHOC_THRESHOLD,
            "version": 1,
        }
        return CacheKey(RI_ID, "cited-answers", "posthoc/minicheck", config)

    def posthoc(self, which: Which = "all") -> list[CitedAnswer]:
        """Return the inline statements with citations chosen by MiniCheck.

        Raises
        ------
        NotCachedError
            When none are cached and ``local_checker`` is off.
        """
        memo = f"posthoc/{which}"
        if memo not in self._memo:
            key = self.posthoc_key(which)
            found = self.cache.get(key)
            if found is None and not self.local_checker:
                raise NotCachedError("no post-hoc citations cached; set RUN_LOCAL_CHECKER = True to attach them")

            def compute() -> JsonValue:
                checker = MiniCheckChecker(MINICHECK)
                moved = attribute_posthoc(self.answers("inline", which), self.evidence, checker, POSTHOC_THRESHOLD)
                return {"answers": [a.to_dict() for a in moved]}

            value = found.value if found is not None else self.cache.load_or_compute(key, compute).value
            self._memo[memo] = [CitedAnswer.from_dict(a) for a in cast(dict[str, Any], value)["answers"]]
        return cast(list[CitedAnswer], self._memo[memo])

    # ------------------------------------------------------------------ judging

    def judge_step(self, arm: str, which: Which) -> Step:
        """Return the step that judges one arm's citations with the judging model."""
        config: dict[str, JsonValue] = {"which": which, "prompt": digest_of(SUPPORT_PROMPT), "version": 1}
        return Step(
            RI_ID,
            "citation-judgements",
            f"judge_citations/{arm}",
            config,
            role="judging",
            upstream={"answers": self.answers_digest(arm, which)},
        )

    def judged(self, arm: str, which: Which = "all") -> list[JudgedAnswer]:
        """Return ALCE verdicts for one arm from the judging model."""
        memo = f"judged/{arm}/{which}"
        if memo not in self._memo:
            answered = [a for a in self.answers(arm, which) if not a.abstained]

            def compute(choice: ModelChoice) -> JsonValue:
                checker = LLMSupportChecker(self.clients.completer(choice.model, max_tokens=200), choice.model)

                def one(answer: CitedAnswer) -> JsonValue:
                    question = self.item(answer.query_id).question
                    return cast(JsonValue, judge_citations(answer, checker, question=question).to_dict())

                return {"judged": cast(JsonValue, parallel_map(one, answered))}

            result = self._run(self.judge_step(arm, which), compute)
            self._memo[f"{memo}/result"] = result
            self._memo[memo] = [JudgedAnswer.from_dict(j) for j in result.value["judged"]]
        return cast(list[JudgedAnswer], self._memo[memo])

    def judge_result(self, arm: str, which: Which = "all") -> StepResult:
        """Return the judging step's result, with its provenance, for reports."""
        self.judged(arm, which)
        return cast(StepResult, self._memo[f"judged/{arm}/{which}/result"])

    def minicheck_key(self, arm: str, which: Which, with_question: bool = False) -> CacheKey:
        """Return the key of MiniCheck's verdicts on one arm."""
        config: dict[str, JsonValue] = {
            "answers": self.answers_digest(arm, which),
            "model": MINICHECK,
            "threshold": 0.5,
            "version": 1,
        }
        if with_question:
            config["question_in_claim"] = digest_of(QUESTION_CLAIM)
        return CacheKey(RI_ID, "citation-judgements-minicheck", f"minicheck/{arm}", config)

    def minichecked(self, arm: str, which: Which = "all", with_question: bool = False) -> list[JudgedAnswer]:
        """Return ALCE verdicts for one arm from the local MiniCheck checker.

        With ``with_question`` on, the claim MiniCheck reads is the question
        followed by the statement.

        Raises
        ------
        NotCachedError
            When no verdicts are cached for these answers and
            ``local_checker`` is off.
        """
        memo = f"minicheck/{arm}/{which}/{with_question}"
        if memo not in self._memo:
            key = self.minicheck_key(arm, which, with_question)
            found = self.cache.get(key)
            if found is None and not self.local_checker:
                raise NotCachedError(
                    f"no MiniCheck verdicts for the {arm} answers; set RUN_LOCAL_CHECKER = True to score them on the CPU"
                )

            def compute() -> JsonValue:
                answered = [a for a in self.answers(arm, which) if not a.abstained]
                checker = MiniCheckChecker(MINICHECK, question_in_claim=with_question)
                score_in_batches(checker, answered, self.item)
                return {
                    "judged": [
                        judge_citations(a, checker, question=self.item(a.query_id).question).to_dict() for a in answered
                    ]
                }

            value = found.value if found is not None else self.cache.load_or_compute(key, compute).value
            self._memo[memo] = [JudgedAnswer.from_dict(j) for j in cast(dict[str, Any], value)["judged"]]
        return cast(list[JudgedAnswer], self._memo[memo])

    # ------------------------------------------------------------------ gold

    def gold(self, arm: str, which: Which = "all") -> list[GoldScore]:
        """Return gold-span scores for one arm's answered, labelled questions."""
        scores: list[GoldScore] = []
        for answer in self.answers(arm, which):
            item = self.item(answer.query_id)
            if answer.abstained or item.is_null:
                continue
            shown = self.evidence(item.query_id, arm)
            scores.append(score_against_gold(answer, item.gold, doc_id=item.doc_id, shown=shown))
        return scores


def shuffle_citations(
    answers: Sequence[CitedAnswer], evidence: Callable[[str], list[Evidence]], seed: int
) -> list[CitedAnswer]:
    """Point every citation at another passage the model was shown.

    This is the control: the statements stay as written and each citation moves
    to a different source from the same prompt, chosen with ``seed``. A scorer
    that cannot tell these answers from the originals is not measuring
    citations. A statement whose question showed one passage keeps its
    citation, because there is nowhere else to point.
    """
    rng = random.Random(seed)
    moved: list[CitedAnswer] = []
    for answer in answers:
        if answer.abstained:
            moved.append(answer)
            continue
        shown = evidence(answer.query_id)
        statements: list[Statement] = []
        for statement in answer.statements:
            cited = []
            for citation in statement.citations:
                others = [n for n in range(1, len(shown) + 1) if n != citation.source]
                number = rng.choice(others) if others else citation.source
                cited.append(number)
            markers = "".join(f"[{n}]" for n in cited)
            rebuilt = parse_inline(answer.query_id, f"{statement.text} {markers}".strip(), shown)
            statements.extend(rebuilt.statements[:1] or [Statement(statement.text)])
        moved.append(CitedAnswer(answer.query_id, tuple(statements), "shuffled", raw=answer.raw))
    return moved


def cite_rank1(answers: Sequence[CitedAnswer], evidence: Callable[[str], list[Evidence]]) -> list[CitedAnswer]:
    """Point every statement at the top-ranked passage, a control that needs no model.

    If this scores as well as the model's own citations, the citations carry no
    information beyond the retriever's ranking.
    """
    cited: list[CitedAnswer] = []
    for answer in answers:
        if answer.abstained:
            cited.append(answer)
            continue
        shown = evidence(answer.query_id)
        statements = [parse_inline(answer.query_id, f"{s.text} [1]", shown).statements[0] for s in answer.statements]
        cited.append(CitedAnswer(answer.query_id, tuple(statements), "rank1", raw=answer.raw))
    return cited


def attribute_posthoc(
    answers: Sequence[CitedAnswer],
    evidence: Callable[[str], list[Evidence]],
    checker: MiniCheckChecker,
    threshold: float = POSTHOC_THRESHOLD,
) -> list[CitedAnswer]:
    """Attach citations after generation: the best passage, then its best sentence.

    The generator's own markers are discarded. For each statement MiniCheck
    scores every shown passage; the best one at or above ``threshold`` is kept,
    and within it the sentence MiniCheck scores highest becomes the citation, so
    the citation is sentence-sized like a quote. A statement no passage
    supports stays uncited.
    """
    pairs = [
        (passage.text, statement.text)
        for answer in answers
        if not answer.abstained
        for passage in evidence(answer.query_id)
        for statement in answer.statements
    ]
    checker.probabilities(pairs)
    moved: list[CitedAnswer] = []
    for answer in answers:
        if answer.abstained:
            moved.append(answer)
            continue
        shown = evidence(answer.query_id)
        statements: list[Statement] = []
        for statement in answer.statements:
            scores = checker.probabilities([(p.text, statement.text) for p in shown])
            best = max(range(len(shown)), key=lambda n: scores[n]) if shown else None
            if best is None or scores[best] < threshold:
                statements.append(Statement(statement.text))
                continue
            passage = shown[best]
            spans = sentence_spans(passage.text) or [(0, len(passage.text))]
            within = checker.probabilities([(passage.text[a:b], statement.text) for a, b in spans])
            start, end = spans[max(range(len(spans)), key=lambda n: within[n])]
            offset = passage.provenance.start or 0
            citation = Citation(
                source=best + 1,
                doc_id=passage.doc_id,
                chunk_id=passage.chunk_id,
                text=passage.text[start:end].strip(),
                start=offset + start,
                end=offset + end,
            )
            statements.append(Statement(statement.text, (citation,)))
        moved.append(CitedAnswer(answer.query_id, tuple(statements), "posthoc", raw=answer.raw))
    return moved


class _Recorder:
    """Records every pair a judging pass asks for, answering ``none``."""

    name = "recorder"

    def __init__(self, checker: MiniCheckChecker) -> None:
        self.checker = checker
        self.pairs: list[tuple[str, str]] = []

    def __call__(self, premise: str, claim: str, *, question: str = "") -> SupportVerdict:
        self.pairs.append((premise, self.checker.claim_for(claim, question)))
        return SupportVerdict("none")


def score_in_batches(
    checker: MiniCheckChecker, answers: Sequence[CitedAnswer], item: Callable[[str], data.Item] | None = None
) -> None:
    """Score the joint and single-citation pairs in batches before judging.

    With every verdict ``none``, a judging pass asks for exactly the joint and
    single-citation pairs, which are almost all the pairs a real pass needs.
    Scoring them in batches first fills the checker's memo, so the real pass
    runs batched except for the few leave-one-out checks.
    """
    recorder = _Recorder(checker)
    for answer in answers:
        judge_citations(answer, recorder, question=item(answer.query_id).question if item else "")
    checker.probabilities(recorder.pairs)


__all__ = [
    "ARMS",
    "DERIVED",
    "GENERATED",
    "K",
    "LIVE_SUBSET",
    "MINICHECK",
    "REFERENCE_MODEL",
    "RI_ID",
    "RPM",
    "STYLES",
    "Clients",
    "NotCachedError",
    "Pipeline",
    "reference_models",
    "attribute_posthoc",
    "cite_rank1",
    "score_in_batches",
    "shuffle_citations",
]
