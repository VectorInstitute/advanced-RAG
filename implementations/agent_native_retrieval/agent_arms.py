"""RI #5's arms on MuSiQue, each one model-produced step behind ``load_or_run``.

Three kinds of arm share one corpus and one question set:

* Agent arms (``C0``, ``C5``, ``C7``): an ADK agent run by
  :func:`~aieng.advanced_rag.agents.session_loop.run_session` with an
  :class:`~aieng.advanced_rag.agents.corpus_tools.ArmConfig`, up to 30 model calls.
* ``grep-iterative``: RI #4's ``dense-iterative`` pipeline with grep in place of
  dense retrieval. The decomposition, step, and answer prompts are RI #4's own,
  so the only change is the retriever.

Every model response is cached in ``<cache root>/agent_native/responses.sqlite``:
an agent's through ADK's response-cache plugin, ``grep-iterative``'s and the
judge's through a governed OpenAI client. A re-run costs nothing.
Each finished question is also cached on its own, keyed by arm, model, corpus,
and question, so an interrupted run resumes and keeps the seconds measured on
its first, cold pass.
"""

from __future__ import annotations

import contextvars
import json
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag import metering
from aieng.advanced_rag.agents.corpus_tools import (
    STANDARD_ARMS,
    ArmConfig,
    Corpus,
    CorpusDocument,
    CorpusToolServer,
    DenseSearch,
    GrepBackend,
    RegexGrep,
    digest,
)
from aieng.advanced_rag.agents.records import DEFAULT_INSTRUCTION, MeasuredLimiter
from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora.musique import MusiqueQuestion, Passage, load_musique
from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1
from aieng.advanced_rag.graphrag.proxy import CallLedger, ResponseCache, govern
from aieng.advanced_rag.metering import CallUsage
from aieng.advanced_rag.models import (
    DEFAULT_MODELS,
    REFERENCE_MODELS,
    ModelChoice,
    ModelSet,
    family_of,
    request_params,
)
from aieng.advanced_rag.runs import Step, StepResult, load_or_run


RI_ID = "ri5"
ENCODER = "BAAI/bge-small-en-v1.5"
REFERENCE_MODEL = REFERENCE_MODELS.generation.model
#: The reference runs' request rate, under the proxy's per-model cap so other
#: jobs on the key still fit.
REFERENCE_RPM = 40
#: Concurrent questions. The proxy's throughput collapses above 8.
WORKERS = 8
#: What every agent arm's system prompt calls the MuSiQue corpus.
MUSIQUE_DESCRIPTION = "a corpus of Wikipedia passages"
#: Questions on a mid-tier or frontier model in the hands-on notebook time.
LIVE_CAP_EXPENSIVE = 10
#: Questions per hop count on a cheap-tier model.
LIVE_PER_HOP_CHEAP = 10
AGENT_MAX_TOKENS = 1024
ARM_NAMES = ("C0", "C5", "C7", "grep-iterative", "C0-top2", "C7-top5-full")


def work_root() -> Path:
    """Return where RI #5's caches and embeddings live."""
    return Path(cache_root()) / "agent_native"


# ---------------------------------------------------------------------------
# Corpus and tools


def musique_passages(size: int = 11_656) -> list[Passage]:
    """Return the 11,656 packaged passages or the 139,416-passage pool.

    The pool opens with the packaged passages under the same identifiers, so a
    question's supporting passages are the same at both sizes.
    """
    if size == 11_656:
        passages: list[Passage] = load_musique()[0]
        return passages
    if size == 139_416:
        from aieng.advanced_rag.corpora.musique import load_pool  # noqa: PLC0415

        pool: list[Passage] = load_pool()
        return pool
    raise ValueError(f"no MuSiQue corpus of {size} passages; use 11656 or 139416")


def musique_corpus(size: int = 11_656) -> Corpus:
    """Return a MuSiQue corpus with each title on line 1 of its passage."""
    return Corpus.from_passages(musique_passages(size), f"musique-{size}")


def dense_search(corpus: Corpus, *, device: str = "cpu") -> DenseSearch:
    """Return dense search over the corpus with local ``bge-small-en-v1.5``.

    The passage matrix is cached. For the packaged corpus, RI #4's matrix is
    reused when present, because it embeds the same title-then-text strings in
    the same order, which makes ``C0``'s search the same ranking as RI #4's
    ``dense`` arm.
    """
    from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
        SentenceTransformerEmbedder,
    )

    embedder = SentenceTransformerEmbedder(ENCODER, device=device)
    # MuSiQue matrices are named by size, as RI #4 names its own; any other
    # corpus by its name, so two corpora of one size never share a file.
    stem = str(len(corpus)) if corpus.name.startswith("musique-") else corpus.name
    cache = work_root() / f"dense-{stem}.npy"
    shared = Path(cache_root()) / "graphrag" / "musique" / f"dense-{len(corpus)}.npy"
    if not cache.is_file() and shared.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shared, cache)
    return DenseSearch(corpus, embedder, cache=cache)


def tool_server(corpus: Corpus, search: DenseSearch | None = None, grep: GrepBackend | None = None) -> CorpusToolServer:
    """Return the tool server, with the regex grep backend unless given another."""
    return CorpusToolServer(corpus, grep=grep or RegexGrep(corpus), search=search)


# ---------------------------------------------------------------------------
# The governed client


@dataclass
class Client:
    """One governed client and what it shares across arms."""

    client: Any
    limiter: MeasuredLimiter
    ledger: CallLedger
    #: The response cache both call paths share. ADK's keys are tagged, so
    #: they never collide with the OpenAI client's.
    cache: ResponseCache


def governed_client(*, rpm: int = REFERENCE_RPM, cache_path: Path | None = None) -> Client:
    """Return an OpenAI-compatible client that caches, throttles, and meters.

    On first use the default response cache is seeded from RI #4's MuSiQue cache, when
    present. ``grep-iterative`` sends RI #4's decomposition prompt unchanged, so
    on the reference model it reads back the same decompositions
    ``dense-iterative`` used, and the comparison changes only the retriever.
    """
    from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    path = cache_path or work_root() / "responses.sqlite"
    seed = Path(cache_root()) / "graphrag" / "musique" / "responses-test.sqlite"
    if cache_path is None and not path.exists() and seed.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed, path)
    limiter = MeasuredLimiter(per_minute=rpm)
    ledger = CallLedger(RI_ID)
    cache = ResponseCache(path)
    client = govern(
        OpenAI(base_url=resolve_base_url(), api_key=resolve_api_key(), max_retries=0, timeout=600.0),
        cache=cache,
        limiter=limiter,
        ledger=ledger,
        max_retries=16,
    )
    return Client(client, limiter, ledger, cache)


Completion = Callable[[str], tuple[str, int, int]]


def completer(client: Client, model: str, max_tokens: int = 256) -> Completion:
    """Return a one-prompt completion function, built the way RI #4 builds it.

    It returns the text with the prompt and completion tokens of the call, so a
    pipeline can total its own cost under concurrency.
    """
    params = request_params(model, max_tokens=max_tokens, temperature=0.0)
    params.pop("extra_body", None)

    def complete(prompt: str) -> tuple[str, int, int]:
        response = client.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **params
        )
        usage = metering.usage_of(response)
        text = str(response.choices[0].message.content or "")
        return text, usage.prompt_tokens, usage.completion_tokens

    return complete


# ---------------------------------------------------------------------------
# grep-iterative: RI #4's dense-iterative with the retriever swapped


# These three prompts are RI #4's, character for character
# (graphrag_technical_support/musique_arms.py); a test checks it.
DECOMPOSE = """Break this multi-hop question into the single-hop questions needed to answer it, in order.
A later question may refer to an earlier answer as #1, #2, and so on. Use as few steps as the
question needs, at most four.

Return only JSON: {{"steps": ["...", "..."]}}

Question: {question}"""

STEP = """Answer the question using only the passages. Reply with a short answer of a few words,
or "unknown" if the passages do not say.

Passages:
{passages}

Question: {question}
Answer:"""

ANSWER = """Answer the question using only the passages. Reply with the answer only, a short phrase
of a few words, and no explanation. If the passages do not contain it, reply "unknown".

Passages:
{passages}

Question: {question}
Answer:"""

ANSWER_PASSAGES = 5
RANK_DEPTH = 10
#: Names the step-retrieval rule in the cache key; change it when the rule changes.
GREP_RULE = "rarest-span-then-overlap/2"

_STOP = frozenset(  # noqa: SIM905 - a word list reads best as one string
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "does",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    ]
)
_STOP_SPANS = frozenset({"What", "Who", "When", "Where", "Which", "How", "The", "In", "Is", "Was", "Did", "Why"})
_SPAN = re.compile(r"(?:[A-Z][\w'’.-]*(?:\s+(?:of|the|de|and|for|in)?\s*[A-Z][\w'’.-]*)*)")


def parse_steps(raw: str, question: str) -> list[str]:
    """Return the decomposition steps, or the question itself when none parse.

    The same rule as RI #4's ``musique_arms.parse_steps``.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            steps = json.loads(candidate).get("steps")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(steps, list):
            cleaned = [str(s).strip() for s in steps if str(s).strip()]
            if cleaned:
                return cleaned[:4]
    return [question]


def substitute(step: str, answers: Sequence[str]) -> str:
    """Replace ``#n`` in a step with the n-th earlier answer when there is one."""

    def fill(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return answers[index] if 0 <= index < len(answers) else match.group(0)

    return re.sub(r"#(\d+)", fill, step)


def interleave(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Merge ranked lists round-robin, skipping repeats, until ``k`` are chosen."""
    merged: list[str] = []
    seen: set[str] = set()
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in seen:
                seen.add(ranking[depth])
                merged.append(ranking[depth])
                if len(merged) == k:
                    return merged
    return merged


def spans(text: str) -> list[str]:
    """Return the capitalized spans and quoted strings of a question."""
    found = [re.sub(r"['’]s$", "", s.strip(" .,?")).strip(" '’") for s in _SPAN.findall(text)]
    found += re.findall(r'"([^"]{3,})"', text)
    return [s for s in dict.fromkeys(found) if len(s) >= 3 and s not in _STOP_SPANS]


def content_words(text: str) -> list[str]:
    """Return a question's lower-cased content words, longest first."""
    words = {w for w in re.findall(r"[\w'’-]+", text.lower()) if len(w) > 3 and w not in _STOP}
    return sorted(words, key=lambda w: (-len(w), w))


class GrepIterative:
    """Decompose, grep for each step's rarest span, answer the step, substitute.

    The step retrieval is literal: every candidate span (each earlier answer the
    step names, then its capitalized spans) is grepped, the span with the fewest
    hits above zero wins, and its hits are ordered by how many of the step's
    content words each passage holds. Content words are the fallback when no
    span matches.

    Parameters
    ----------
    corpus : Corpus
        The passages.
    complete : callable
        One prompt in; the model's text, prompt tokens, and completion tokens out.
    step_k : int, optional
        Passages a step keeps, as RI #4's arm.
    shown : int, optional
        Passages of a step shown to the model, as RI #4's arm.
    """

    name = "grep-iterative"

    def __init__(self, corpus: Corpus, complete: Completion, *, step_k: int = 5, shown: int = 3) -> None:
        self.corpus = corpus
        self.grep = RegexGrep(corpus)
        self.complete = complete
        self.step_k = step_k
        self.shown = shown

    def text_of(self, doc_id: str) -> str:
        """Return a passage as RI #4 shows it, title then text."""
        found = self.corpus.get(doc_id)
        return found.text if found is not None else ""

    def retrieve(self, step: str, bridges: Sequence[str]) -> tuple[list[str], str]:
        """Return a step's passages and the span that found them."""
        candidates = [b for b in bridges if b and b.lower() != "unknown" and len(b) >= 2]
        candidates += spans(step)
        best: tuple[int, int, str, list[str]] | None = None
        for candidate in dict.fromkeys(candidates):
            found = self.grep.grep(re.escape(candidate), max_results=5000)
            if found.total_hits and (best is None or (found.total_hits, -len(candidate)) < best[:2]):
                best = (found.total_hits, -len(candidate), candidate, [m.doc_id for m in found.matches])
        if best is None:
            for word in content_words(step):
                found = self.grep.grep(r"\b" + re.escape(word) + r"\b", max_results=5000)
                if found.total_hits and (best is None or found.total_hits < best[0]):
                    best = (found.total_hits, 0, word, [m.doc_id for m in found.matches])
        if best is None:
            return [], ""
        words = content_words(step)

        def overlap(doc_id: str) -> int:
            text = self.text_of(doc_id).lower()
            return sum(1 for w in words if w in text)

        ranked = sorted(best[3], key=lambda d: -overlap(d))
        return ranked[: self.step_k], best[2]

    def run(self, question: str, waited: Callable[[], float] = lambda: 0.0) -> dict[str, Any]:
        """Answer one question and return the ranking, what was shown, and the steps.

        ``waited`` returns the rate-limit wait so far for this question, which
        comes off the reported seconds.
        """
        started = time.perf_counter()
        tokens = [0, 0]

        def ask(prompt: str) -> str:
            text, prompt_tokens, completion_tokens = self.complete(prompt)
            tokens[0] += prompt_tokens
            tokens[1] += completion_tokens
            return text

        steps = parse_steps(ask(DECOMPOSE.format(question=question)), question)
        answers: list[str] = []
        rankings: list[list[str]] = []
        shown_ids: list[str] = []
        record = []
        for step in steps:
            asked = substitute(step, answers)
            bridges = [answers[int(n) - 1] for n in re.findall(r"#(\d+)", step) if 0 < int(n) <= len(answers)]
            ids, span = self.retrieve(asked, bridges)
            rankings.append(ids)
            shown = ids[: self.shown]
            shown_ids.extend(shown)
            passages = "\n\n".join(self.text_of(i) for i in shown)
            reply = ask(STEP.format(passages=passages, question=asked))
            answers.append(reply.strip().splitlines()[0].strip() if reply.strip() else "unknown")
            record.append({"step": asked, "span": span, "hits": ids, "answer": answers[-1]})
        ranked = interleave(rankings, RANK_DEPTH)
        top = ranked[:ANSWER_PASSAGES]
        passages = "\n\n".join(f"[{n}] {self.text_of(i)}" for n, i in enumerate(top, start=1))
        text = ask(ANSWER.format(passages=passages, question=question)).strip()
        return {
            "answer": text.splitlines()[0].strip() if text else "",
            "ranked": ranked,
            "read_ids": list(dict.fromkeys([*shown_ids, *top])),
            "steps": record,
            "model_calls": 2 + len(steps),
            "tool_calls": 0,
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "exhausted": False,
            # Wall time includes rate-limit waits; grep-iterative's calls are
            # sequential, so the waits are subtracted through the limiter.
            "seconds": round(time.perf_counter() - started - waited(), 3),
        }


# ---------------------------------------------------------------------------
# Running an arm over questions


#: Names the agent loop in the cache keys. Runs from the earlier
#: chat-completions loop sent other tool declarations, so they never hit.
AGENT_LOOP = "adk"


def prompt_digest(arm: ArmConfig, corpus: Corpus, description: str = MUSIQUE_DESCRIPTION) -> str:
    """Return a digest of everything the model is told under one arm."""
    if arm.name == "grep-iterative":
        return str(digest([DECOMPOSE, STEP, ANSWER, GREP_RULE]))
    session = CorpusToolServer(corpus, search=lambda q, k: []).session(arm)
    from aieng.advanced_rag.agents.toolset import tool_declarations  # noqa: PLC0415

    return str(digest([AGENT_LOOP, DEFAULT_INSTRUCTION, description, tool_declarations(session.tools)]))


def agent_run(
    question: str,
    session: Any,
    *,
    model: str,
    client: Client,
    description: str = MUSIQUE_DESCRIPTION,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Run one question on ADK and return its :class:`AgentRun` record.

    The arm's call budget, forced final answer, read-first guard, and 60,000-token
    prompt guard come from :func:`run_session`.
    """
    from aieng.advanced_rag.agents.session_loop import run_session  # noqa: PLC0415

    run = run_session(
        question,
        session,
        model=model,
        corpus_description=description,
        instruction=instruction,
        max_tokens=AGENT_MAX_TOKENS,
        cache=client.cache,
        limiter=client.limiter,
    )
    record: dict[str, Any] = run.to_dict()
    return record


GREP_ITERATIVE = ArmConfig(
    "grep-iterative",
    ("grep",),
    "RI #4's dense-iterative with grep in place of dense retrieval",
    max_model_calls=6,
    read_before_answer=False,
)


#: Follow-up variants of the standard arms. ``C0-top2`` starves dense search to
#: test whether ``C5``'s lead returns when search loses hops; ``C7-top5-full``
#: shows ``C7`` five full passages per search instead of twenty 40-token
#: previews, to test whether its loss on Gemini is previews it never opens.
VARIANT_ARMS: Mapping[str, ArmConfig] = {
    "C0-top2": ArmConfig(
        "C0-top2",
        ("search",),
        "search with k fixed at 2 and 512-token previews",
        search_k=2,
        read_before_answer=False,
    ),
    "C7-top5-full": ArmConfig(
        "C7-top5-full",
        ("search", "grep", "open"),
        "search at k 5 with 512-token previews, then grep inside what search returned, then open",
        search_k=5,
        search_preview_tokens=512,
        grep_scope="searched",
    ),
}


def arm_config(name: str) -> ArmConfig:
    """Return an arm by name."""
    if name == "grep-iterative":
        return GREP_ITERATIVE
    if name in VARIANT_ARMS:
        return VARIANT_ARMS[name]
    return STANDARD_ARMS[name]


def _question_key(arm: ArmConfig, model: str, corpus: Corpus, question: MusiqueQuestion, prompts: str) -> CacheKey:
    return CacheKey(
        RI_ID,
        "agent-question",
        arm.name,
        {
            "arm": arm.to_dict(),
            "model": model,
            "corpus": corpus.name,
            "question_id": question.question_id,
            "question": digest(question.question),
            "prompts": prompts,
        },
    )


def _replay_usage(model: str, run: Mapping[str, Any]) -> None:
    """Report a cached question's calls to the meter as cache hits."""
    calls = max(int(run.get("model_calls", 0)), 1)
    prompt = int(run.get("prompt_tokens", 0))
    completion = int(run.get("completion_tokens", 0))
    for index in range(calls):
        last = index == calls - 1
        metering.report_cache_hit(
            model=model,
            usage=CallUsage(
                prompt // calls + (prompt % calls if last else 0),
                completion // calls + (completion % calls if last else 0),
            ),
        )


def score(run: Mapping[str, Any], question: MusiqueQuestion) -> dict[str, Any]:
    """Return a run with the question's identifiers and its EM and F1."""
    golds = (question.answer, *question.answer_aliases)
    return {
        **run,
        "question_id": question.question_id,
        "hops": question.hops,
        "em": exact_match(run.get("answer", ""), golds),
        "f1": token_f1(run.get("answer", ""), golds),
    }


def run_questions(
    arm: ArmConfig,
    questions: Sequence[MusiqueQuestion],
    *,
    model: str,
    corpus: Corpus,
    client: Client,
    search: DenseSearch | None = None,
    workers: int = WORKERS,
    cache: TieredResultCache | None = None,
    progress: Callable[[int, int], None] | None = None,
    grep: GrepBackend | None = None,
    description: str = MUSIQUE_DESCRIPTION,
) -> list[dict[str, Any]]:
    """Run one arm over questions on one model, one cached record per question.

    Returns
    -------
    list of dict
        One scored record per question, in question order: the
        :class:`~aieng.advanced_rag.agents.records.AgentRun` fields (or the
        ``grep-iterative`` record), ``question_id``, ``hops``, ``em``, ``f1``.
    """
    store = cache or TieredResultCache()
    prompts = prompt_digest(arm, corpus, description)
    server = tool_server(corpus, search, grep)
    pipeline = GrepIterative(corpus, completer(client, model)) if arm.name == "grep-iterative" else None

    def one(question: MusiqueQuestion) -> dict[str, Any]:
        key = _question_key(arm, model, corpus, question, prompts)
        cached = store.get(key)
        if cached is not None and isinstance(cached.value, dict):
            _replay_usage(model, cached.value)
            return score(cached.value, question)
        if pipeline is not None:
            mark = client.limiter.waited()
            record: dict[str, Any] = pipeline.run(question.question, lambda: client.limiter.waited() - mark)
        else:
            record = agent_run(
                question.question, server.session(arm), model=model, client=client, description=description
            )
        store.load_or_compute(key, lambda: record, force_refresh=True)
        return score(record, question)

    done = [0]
    results: list[dict[str, Any] | None] = [None] * len(questions)

    def task(index: int) -> None:
        results[index] = one(questions[index])
        done[0] += 1
        if progress is not None:
            progress(done[0], len(questions))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Each task gets its own copy of the context, so the meter that
        # load_or_run binds sees the calls made on worker threads.
        futures = [pool.submit(contextvars.copy_context().run, task, i) for i in range(len(questions))]
        for future in futures:
            future.result()
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Steps


def question_digest(questions: Sequence[MusiqueQuestion]) -> str:
    """Return a digest of a question list's identifiers, in order."""
    return str(digest([q.question_id for q in questions]))


def arm_step(
    arm: ArmConfig,
    corpus: Corpus,
    questions: Sequence[MusiqueQuestion],
    label: str,
    description: str = MUSIQUE_DESCRIPTION,
) -> Step:
    """Return the step for one arm over one question set."""
    return Step(
        RI_ID,
        "agent-runs",
        f"{arm.name}/{label}",
        {
            "arm": arm.to_dict(),
            "corpus": corpus.name,
            "questions": question_digest(questions),
            "question_count": len(questions),
            "prompts": prompt_digest(arm, corpus, description),
            "max_tokens": AGENT_MAX_TOKENS,
        },
        reference_model=REFERENCE_MODEL,
        # Agents call through ADK's GovernedLlmClient and grep-iterative through
        # governed_client, so a re-run on cached responses needs no --confirm.
        guarded=True,
    )


def run_arm(
    arm_name: str,
    questions: Sequence[MusiqueQuestion],
    *,
    corpus: Corpus,
    label: str,
    models: ModelSet | None = None,
    live: bool = False,
    confirm: bool = False,
    search: DenseSearch | None | Callable[[], DenseSearch] = None,
    grep: GrepBackend | None = None,
    description: str = MUSIQUE_DESCRIPTION,
    client: Client | None = None,
    rpm: int = REFERENCE_RPM,
    workers: int = WORKERS,
    show: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> StepResult:
    """Return one arm's runs: yours, the labelled reference, or a live run.

    Parameters
    ----------
    arm_name : str
        ``C0``, ``C5``, ``C7``, or ``grep-iterative``.
    questions : sequence of MusiqueQuestion
        The questions, in order.
    corpus : Corpus
        The passages.
    label : str
        Names the question set in the cache key, such as ``shared-300``.
    models : ModelSet or None, optional
        Your models; the generation model runs the arm.
    live : bool, optional
        Pass ``RUN_EXPENSIVE``.
    confirm : bool, optional
        Run even when the estimate is over 10% of the budget left.
    search : DenseSearch, callable, or None, optional
        The dense search, or a function that builds it, needed by arms with
        ``search``. A function is called only when the step runs live.
    client : Client or None, optional
        A governed client to share across arms.

    Returns
    -------
    StepResult
        ``value`` is ``{"runs": [...]}``.
    """
    arm = arm_config(arm_name)
    step = arm_step(arm, corpus, questions, label, description)

    def compute(choice: ModelChoice) -> dict[str, Any]:
        used = client or governed_client(rpm=rpm)
        backend = search
        if "search" not in arm.tools:
            backend = None
        elif callable(search) and not isinstance(search, DenseSearch):
            backend = search()
        runs = run_questions(
            arm,
            questions,
            model=choice.model,
            corpus=corpus,
            client=used,
            search=backend,
            workers=workers,
            progress=progress,
            grep=grep,
            description=description,
        )
        return {"model": choice.model, "runs": runs}

    return load_or_run(step, compute, models=models, live=live, confirm=confirm, show=show)


def judge_runs(
    runs: Sequence[Mapping[str, Any]],
    questions: Sequence[MusiqueQuestion],
    *,
    corpus: Corpus,
    answered_by: str,
    label: str,
    models: ModelSet | None = None,
    live: bool = False,
    client: Client | None = None,
    show: bool = True,
) -> StepResult:
    """Judge one arm's answers with the judging model, as a step of its own.

    The judge reads the question, the reference answer, the passages the run
    read, and the answer, and returns whether the answer is correct. The step
    key names the answering model, so judgments of your runs and of the
    reference runs never share a key.

    Returns
    -------
    StepResult
        ``value`` is ``{"judge": model, "verdicts": {question_id: bool}}``.
    """
    from aieng.advanced_rag.evaluation.answers import judge_answer  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: PLC0415
    from aieng.advanced_rag.provenance import Provenance  # noqa: PLC0415

    by_id = {q.question_id: q for q in questions}
    step = Step(
        RI_ID,
        "agent-judgments",
        f"judge/{label}",
        {
            "answers": digest([(r["question_id"], r.get("answer", "")) for r in runs]),
            "answered_by": answered_by,
            "corpus": corpus.name,
        },
        role="judging",
        reference_model=REFERENCE_MODELS.judging.model,
    )

    def compute(choice: ModelChoice) -> dict[str, Any]:
        used = client or governed_client()
        complete = completer(used, choice.model, max_tokens=300)
        verdicts: dict[str, bool] = {}
        for run in runs:
            question = by_id[run["question_id"]]
            evidence = []
            for doc_id in (run.get("read_ids") or [])[:8]:
                doc = corpus.get(doc_id)
                if doc is not None:
                    evidence.append(Evidence(text=doc.text, provenance=Provenance(doc_id=doc_id)))
            try:
                verdict = judge_answer(
                    question.question_id,
                    question.question,
                    run.get("answer", "") or "(no answer)",
                    question.answer,
                    evidence,
                    lambda prompt: complete(prompt)[0],
                )
                verdicts[question.question_id] = verdict.correct
            except ValueError:
                verdicts[question.question_id] = False
        return {"judge": choice.model, "verdicts": verdicts}

    return load_or_run(step, compute, models=models, live=live, show=show)


#: Questions per hop count in the scale ladder.
LADDER_PER_HOP = 40


def ladder_questions(questions: Sequence[MusiqueQuestion]) -> list[MusiqueQuestion]:
    """Return the scale-ladder subset: the first 40 questions of each hop count."""
    counts: dict[int, int] = {}
    picked = []
    for question in questions:
        if counts.get(question.hops, 0) < LADDER_PER_HOP:
            counts[question.hops] = counts.get(question.hops, 0) + 1
            picked.append(question)
    return picked


@dataclass(frozen=True)
class Question:
    """A question over any corpus, with a reference answer and optional gold documents.

    It has the fields :func:`run_questions` reads from a MuSiQue question, so
    the same runner serves a folder of your own files.
    """

    question_id: str
    question: str
    answer: str = ""
    answer_aliases: tuple[str, ...] = ()
    hops: int = 1
    supporting_ids: tuple[str, ...] = ()


#: What the agent arms' system prompt calls the ServerFault corpus.
SERVERFAULT_DESCRIPTION = "a corpus of ServerFault questions, each with its accepted answer"


def serverfault_set() -> tuple[Corpus, list[Question]]:
    """Return the 10,000-question ServerFault slice and the 30 error-string questions.

    Each document is a post's title, question, and accepted answer. A question
    has no reference answer: it names an exact error string, and its gold
    documents are the posts that contain it.
    """
    from aieng.advanced_rag.corpora import serverfault  # noqa: PLC0415
    from question_sets import load_serverfault_questions  # noqa: PLC0415

    documents = serverfault.documents(serverfault.fetch_slice())
    corpus = Corpus((CorpusDocument(d.doc_id, d.title, d.text) for d in documents), "serverfault-10k")
    questions = [
        Question(q.query_id, q.question, supporting_ids=q.gold_doc_ids) for q in load_serverfault_questions(documents)
    ]
    return corpus, questions


#: Answers that count as abstaining on a near-miss null question.
ABSTAIN = frozenset({"", "unknown", "not stated", "none", "n/a"})


def abstained(answer: str) -> bool:
    """Return whether a short answer declines to answer."""
    text = answer.strip().strip(".").lower()
    return text in ABSTAIN or text.startswith(("unknown", "the corpus does not", "the documents do not"))


def musique_set(name: str) -> list[Question]:
    """Return the paraphrase control or the near-miss nulls as runnable questions.

    ``paraphrase`` keeps each rewrite's reference answer and supporting passages.
    ``nulls`` has no answer and no supporting passages, because the corpus does
    not answer them; abstaining is the right result.
    """
    from question_sets import NEAR_MISS_NULLS, PARAPHRASE_CONTROL, load_jsonl  # noqa: PLC0415

    path = {"paraphrase": PARAPHRASE_CONTROL, "nulls": NEAR_MISS_NULLS}[name]
    return [
        Question(
            row["query_id"],
            row["question"],
            row.get("answer") or "",
            tuple(row.get("answer_aliases") or ()),
            int(row["hops"]),
            tuple(row.get("supporting_ids") or ()),
        )
        for row in load_jsonl(path)
    ]


def live_questions(questions: Sequence[MusiqueQuestion], model: str) -> list[MusiqueQuestion]:
    """Return the hands-on subset: 10 per hop count on the cheap tier, else 10 in all.

    The cheap tier is each family's default generation model and every on-prem
    model. Any other model, mid tier or frontier, is capped at 10 questions.
    """
    family = family_of(model)
    cheap = family == "onprem" or (family is not None and DEFAULT_MODELS[family]["generation"] == model)
    by_hop: dict[int, list[MusiqueQuestion]] = {}
    for question in questions:
        by_hop.setdefault(question.hops, []).append(question)
    if cheap:
        return [q for hops in sorted(by_hop) for q in by_hop[hops][:LIVE_PER_HOP_CHEAP]]
    picked: list[MusiqueQuestion] = []
    for depth in range(LIVE_CAP_EXPENSIVE):
        for hops in sorted(by_hop):
            if depth < len(by_hop[hops]) and len(picked) < LIVE_CAP_EXPENSIVE:
                picked.append(by_hop[hops][depth])
    return sorted(picked, key=lambda q: (q.hops, questions.index(q)))


__all__ = [
    "ANSWER",
    "ARM_NAMES",
    "VARIANT_ARMS",
    "DECOMPOSE",
    "ENCODER",
    "GREP_ITERATIVE",
    "MUSIQUE_DESCRIPTION",
    "REFERENCE_MODEL",
    "REFERENCE_RPM",
    "RI_ID",
    "STEP",
    "Client",
    "GrepIterative",
    "Question",
    "abstained",
    "agent_run",
    "AGENT_LOOP",
    "musique_set",
    "SERVERFAULT_DESCRIPTION",
    "serverfault_set",
    "arm_config",
    "arm_step",
    "dense_search",
    "governed_client",
    "judge_runs",
    "ladder_questions",
    "live_questions",
    "musique_corpus",
    "run_arm",
    "run_questions",
    "score",
    "spans",
    "tool_server",
    "work_root",
]
