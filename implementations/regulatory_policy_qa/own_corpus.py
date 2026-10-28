"""Retrieval levers and answering with abstention on a team's own question set.

``07_levers_and_answers_on_your_corpus.ipynb`` calls two functions after
:mod:`byoc` has read the folder and returned the question set that notebook 04
built. :func:`score_levers` ranks BM25, dense, SPLADE, a hybrid, and a reranker
on the set, each against the BM25 baseline with a paired bootstrap interval.
:func:`answer_questions` runs RI #1's single-pass answering configuration on
every answerable and null question and judges the answerable ones.

Both results go through the shared result cache. With ``RUN_EXPENSIVE`` off they
read what is cached, and for the stand-in contracts that is the reference result
in ``outputs/baseline_cache``. Rebuild the stand-in's reference results from this
folder, on the on-prem models:

    python own_corpus.py --env-file <env_file>

where ``<env_file>`` is a proxy ``.env`` fallback, needed only when onboarding
has not exported a key and a response is not cached.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import byoc
from aieng.advanced_rag import spend
from aieng.advanced_rag.batch import ProxyBatch
from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.evaluation.answers import AnswerCase, judge_answers, judge_prompt
from aieng.advanced_rag.evaluation.metrics import score_null_detection
from aieng.advanced_rag.evaluation.multihop import paired_bootstrap
from aieng.advanced_rag.evaluation.synthetic import Document, SyntheticSet
from aieng.advanced_rag.evaluation.synthetic.validation import (
    CORPUS_SCOPE,
    DENSE_MODEL,
    RERANKER,
    build_arms,
    chunks_for,
    mean_scores,
    queries_for,
    score_arms,
)
from aieng.advanced_rag.graphrag.proxy import request_key
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.runs import StepResult


RI_ID = byoc.RI_ID
PRODUCER = "regulatory_policy_qa/07_levers_and_answers_on_your_corpus"

#: The arms scored, at 1,200-character chunks. BM25 is the baseline every
#: other arm is compared with.
BASELINE_ARM = "bm25@1200"
LEVERS: tuple[str, ...] = (
    "bm25@1200",
    "dense@1200",
    "splade@1200",
    "hybrid-rrf-0.2@1200",
    "dense+rerank@1200",
)
#: The SPLADE model, notebook 02's first stage.
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
METRICS = ("ndcg@10", "recall@3", "mrr")

#: RI #1's recommended answering configuration: one retrieval, a model-free
#: grade, at most one generation call, and the generator's own abstention.
TOP_K = 10
EVIDENCE_BUDGET = 5
GENERATION_TOKENS = 512
JUDGE_TOKENS = 400

#: The models the stand-in's reference answers were built with.
REFERENCE_MODELS: ModelSet = choose_models("onprem")


def _cache(cache: TieredResultCache | None) -> TieredResultCache:
    return cache or TieredResultCache(global_dir=byoc.BASELINE)


def _indexed(generated: SyntheticSet, documents: Sequence[Document]) -> list[Document]:
    indexed = set(generated.meta["documents"]["indexed"])
    return [d for d in documents if d.doc_id in indexed]


def _corpus_wide(generated: SyntheticSet) -> bool:
    return bool(generated.meta.get("config", {}).get("scope") == "corpus")


# -- retrieval levers ----------------------------------------------------------


def _lever_arms(documents: Sequence[Document], corpus_wide: bool) -> dict[str, Any]:
    """Return the scored arms: four of RI #1's harness arms and SPLADE."""
    from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
        CachedQueryEmbedder,
        SentenceTransformerEmbedder,
    )
    from aieng.advanced_rag.retrieval.representations import SpladeRetriever  # noqa: PLC0415

    embedder = CachedQueryEmbedder(SentenceTransformerEmbedder(DENSE_MODEL))
    arms = {
        name: arm for name, arm in build_arms(documents, embedder, corpus_wide=corpus_wide).items() if name in LEVERS
    }
    chunks = chunks_for(documents, 1200, 200)
    splade = SpladeRetriever()
    splade.index([replace(c, doc_id=CORPUS_SCOPE) for c in chunks] if corpus_wide else chunks)
    arms["splade@1200"] = (splade.search, chunks)
    return {name: arms[name] for name in LEVERS}


def _versus(scores: dict[str, dict[str, dict[str, float]]], baseline: str) -> dict[str, Any]:
    """Return each arm's paired difference from the baseline, per metric."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        base = scores[metric][baseline]
        order = sorted(base)
        out[metric] = {}
        for arm, values in scores[metric].items():
            if arm == baseline:
                continue
            result = paired_bootstrap([values[q] for q in order], [base[q] for q in order], samples=2000)
            out[metric][arm] = {
                "mean_difference": round(result.mean_difference, 4),
                "low": round(result.low, 4),
                "high": round(result.high, 4),
            }
    return out


def score_levers(
    result: StepResult,
    generated: SyntheticSet,
    documents: Sequence[Document],
    *,
    live: bool,
    cache: TieredResultCache | None = None,
) -> dict[str, Any]:
    """Score the levers on the set's answerable questions, cached or computed locally.

    Every arm runs on this machine: the embedder, SPLADE, and the reranker are
    local checkpoints, so scoring makes no model call and costs nothing.

    Parameters
    ----------
    result : StepResult
        The question set's step result, from :func:`byoc.generate`.
    generated : SyntheticSet
        The set itself.
    documents : sequence of Document
        From :func:`byoc.read_corpus`.
    live : bool
        The notebook's ``RUN_EXPENSIVE``.
    cache : TieredResultCache or None, optional
        Defaults to the shipped tier in ``outputs/baseline_cache``.

    Returns
    -------
    dict
        ``mean`` (metric to arm to score), ``versus_baseline`` (metric to arm
        to the paired difference and its 95% interval), and the counts.

    Raises
    ------
    FileNotFoundError
        With ``live`` off, when no scores for this set are cached.
    """
    cache = _cache(cache)
    corpus_wide = _corpus_wide(generated)
    key = CacheKey(
        RI_ID,
        "own-corpus-levers",
        PRODUCER,
        {
            "set": result.digest,
            "corpus_wide": corpus_wide,
            "arms": list(LEVERS),
            "embedder": DENSE_MODEL,
            "reranker": RERANKER,
            "splade": SPLADE_MODEL,
        },
    )
    cached = cache.get(key)
    if cached is not None:
        print(f"levers: cached scores for the set {result.digest[:12]}")
        return dict(cached.value)
    if not live:
        raise FileNotFoundError("no cached lever scores for this set; set RUN_EXPENSIVE = True to score it")

    def compute() -> dict[str, Any]:
        plain = generated.variant("plain")
        indexed = _indexed(generated, documents)
        arms = _lever_arms(indexed, corpus_wide)
        scores = score_arms(arms, queries_for(plain, corpus_wide=corpus_wide), plain.qrels)
        return {
            "arms": list(arms),
            "baseline": BASELINE_ARM,
            "questions": len(plain.answerable),
            "documents_indexed": len(indexed),
            "corpus_wide": corpus_wide,
            "mean": mean_scores(scores),
            "versus_baseline": _versus(scores, BASELINE_ARM),
        }

    print(f"levers: scoring the set {result.digest[:12]} locally, with no model call and no cost")
    return dict(cache.load_or_compute(key, compute).value)


# -- answering with abstention --------------------------------------------------


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _send(
    label: str,
    batch: ProxyBatch,
    prompts: Sequence[str],
    params: dict[str, Any],
    *,
    confirm: bool,
) -> None:
    """Print a step's estimate, then send what the response cache does not hold.

    The estimate takes every reply at its full completion cap, so it is an
    upper bound. A paid batch over the confirmation threshold stops with
    ``ConfirmationRequiredError`` until ``confirm`` is true.
    """
    missing = [
        p
        for p in prompts
        if batch.cache is None or batch.cache.entry(request_key(batch.model, _messages(p), params)) is None
    ]
    dollars = batch.estimate_usd(missing, params) if missing else 0.0
    price = "price unknown" if dollars is None else f"est. ${dollars:,.2f} at most"
    print(f"{label}: {len(prompts)} calls to {batch.model}, {len(prompts) - len(missing)} cached, {price}", flush=True)
    if missing:
        batch.complete_batch(missing, confirm=confirm, **params)


def _retriever(name: str, documents: Sequence[Document]) -> Any:
    """Return an indexed retriever the answering spine can call."""
    from aieng.advanced_rag.retrieval.sparse import BM25Retriever, HybridRetriever  # noqa: PLC0415

    chunks = chunks_for(documents, 1200, 200)
    bm25 = BM25Retriever()
    bm25.index(chunks)
    if name == "bm25":
        return bm25
    if name == "hybrid":
        from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
        from aieng.advanced_rag.retrieval.retrievers import DenseRetriever  # noqa: PLC0415

        dense = DenseRetriever(SentenceTransformerEmbedder(DENSE_MODEL))
        dense.index(chunks)
        return HybridRetriever(dense, bm25, dense_weight=0.2)
    raise ValueError(f"unknown retriever {name!r}; expected 'bm25' or 'hybrid'")


def _answer_key(result: StepResult, models: ModelSet, retriever: str) -> CacheKey:
    return CacheKey(
        RI_ID,
        "own-corpus-answers",
        PRODUCER,
        {
            "set": result.digest,
            "retriever": retriever,
            "generation": models.generation.model,
            "judging": models.judging.model,
            "top_k": TOP_K,
            "evidence_budget": EVIDENCE_BUDGET,
            "policy": "single-shot",
            "grader": "heuristic",
        },
    )


def answer_questions(
    result: StepResult,
    generated: SyntheticSet,
    documents: Sequence[Document],
    models: ModelSet,
    *,
    live: bool,
    confirm: bool = False,
    retriever: str = "bm25",
    cache: TieredResultCache | None = None,
    env_path: Path | None = None,
) -> dict[str, Any]:
    """Answer every question in the set or abstain, then judge the answerable ones.

    The configuration is RI #1's recommended single pass: one retrieval of
    ``TOP_K`` passages, a model-free grade, ``EVIDENCE_BUDGET`` passages in the
    prompt, and one generation call, which abstains with "not stated" when the
    evidence does not answer. Judging uses the judging model, with correctness
    and grounding as separate verdicts.

    Each live step prints its call count and an upper-bound estimate before it
    sends anything, and a paid step over the confirmation threshold stops until
    ``confirm`` is true.

    Parameters
    ----------
    result, generated, documents
        As for :func:`score_levers`.
    models : ModelSet
        From ``choose_models``.
    live : bool
        The notebook's ``RUN_EXPENSIVE``.
    confirm : bool, optional
        Send a paid step whose estimate needs confirmation.
    retriever : {"bm25", "hybrid"}, optional
        The first stage: BM25, or BM25 fused with dense at a dense weight of 0.2.
    cache : TieredResultCache or None, optional
        Defaults to the shipped tier in ``outputs/baseline_cache``.
    env_path : Path or None, optional
        A ``.env`` for the key resolver.

    Returns
    -------
    dict
        The null-detection report, answerable correctness and grounding, the
        balanced decision accuracy, the calls made, and one row per question.

    Raises
    ------
    FileNotFoundError
        With ``live`` off, when neither your result nor the stand-in's
        reference result is cached.
    ConfirmationRequiredError
        Before a paid step whose estimate needs ``confirm=True``.
    """
    cache = _cache(cache)
    key = _answer_key(result, models, retriever)
    cached = cache.get(key)
    if cached is None and not live:
        fallback = cache.get(_answer_key(result, REFERENCE_MODELS, retriever))
        if fallback is not None:
            print(
                f"answers: reference result, built with {REFERENCE_MODELS.generation.model} on Vector's "
                "cluster at $0 (set RUN_EXPENSIVE = True to answer with your models)"
            )
            return dict(fallback.value)
        raise FileNotFoundError("no cached answers for this set; set RUN_EXPENSIVE = True to run them")
    if cached is not None:
        print(f"answers: cached result for the set {result.digest[:12]}")
        return dict(cached.value)

    def compute() -> dict[str, Any]:
        from aieng.advanced_rag.pipeline.answering import SingleShotPolicy  # noqa: PLC0415
        from aieng.advanced_rag.pipeline.grading import HeuristicGrader  # noqa: PLC0415
        from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig  # noqa: PLC0415

        plain = generated.variant("plain")
        search = _retriever(retriever, _indexed(generated, documents))
        responses = cache_root() / "own-corpus" / "responses.sqlite"
        generation = {"max_tokens": GENERATION_TOKENS, "temperature": 0.0}
        judging = {"max_tokens": JUDGE_TOKENS, "temperature": 0.0}

        def spine(complete: Any) -> Any:
            return AdvancedRagSpine(
                search,
                HeuristicGrader(),
                complete,
                config=SpineConfig(top_k=TOP_K, evidence_budget=EVIDENCE_BUDGET, max_attempts=1),
                policy=SingleShotPolicy(),
            )

        # The single pass's prompt depends only on retrieval, so the prompts are
        # collected first, estimated, and sent as one batch. The second pass reads
        # every reply from the response cache.
        prompts: list[str] = []

        def collect(prompt: str) -> str:
            prompts.append(prompt)
            return ""

        collector = spine(collect)
        for q in plain.questions:
            collector.answer(q.query_id, q.question, q.scope)

        with (
            ProxyBatch(models.generation.model, cache=responses, env_path=env_path) as gen,
            ProxyBatch(models.judging.model, cache=responses, env_path=env_path) as judge,
        ):
            _send("generation", gen, prompts, generation, confirm=confirm)
            answering = spine(gen.completer(**generation))
            results = {q.query_id: answering.answer(q.query_id, q.question, q.scope) for q in plain.questions}
            cases = [
                AnswerCase(
                    q.query_id,
                    q.question,
                    results[q.query_id].answer.text,
                    q.reference,
                    list(results[q.query_id].evidence),
                )
                for q in plain.answerable
            ]
            _send(
                "judging",
                judge,
                [judge_prompt(c.question, c.answer_text, c.reference, c.evidence) for c in cases],
                judging,
                confirm=confirm,
            )
            verdicts = dict(
                zip(
                    [c.query_id for c in cases],
                    judge_answers(cases, judge.completer(**judging)),
                    strict=True,
                )
            )
            calls = {
                "generation": {"model": gen.model, "calls": len(prompts)},
                "judging": {"model": judge.model, "calls": len(cases)},
            }

        abstained = {qid: r.answer.abstained for qid, r in results.items()}
        nulls = [q.query_id for q in plain.nulls]
        judged = [v for v in verdicts.values() if v is not None]
        answerable = len(plain.answerable)
        answered = sum(not abstained[q.query_id] for q in plain.answerable)
        null_abstained = sum(abstained[qid] for qid in nulls)
        return {
            "retriever": retriever,
            "models": {"generation": models.generation.model, "judging": models.judging.model},
            "questions": len(plain.questions),
            "answerable": answerable,
            "null": len(nulls),
            "null_detection": score_null_detection(abstained, nulls).to_dict(),
            "answerable_abstained": answerable - answered,
            "judged": len(judged),
            "correct": sum(v.correct for v in judged),
            "grounded": sum(v.grounded for v in judged),
            "balanced_decision_accuracy": round(
                ((answered / answerable if answerable else 0.0) + (null_abstained / len(nulls) if nulls else 0.0)) / 2,
                4,
            ),
            "calls": calls,
            "rows": [
                {
                    "query_id": q.query_id,
                    "kind": q.kind,
                    "question": q.question,
                    "answer": results[q.query_id].answer.text,
                    "abstained": abstained[q.query_id],
                    "correct": None if verdicts.get(q.query_id) is None else verdicts[q.query_id].correct,
                    "grounded": None if verdicts.get(q.query_id) is None else verdicts[q.query_id].grounded,
                }
                for q in plain.questions
            ],
        }

    return dict(cache.load_or_compute(key, compute).value)


def main() -> int:
    """Build the stand-in's reference scores and answers into the shipped tier."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    writer = TieredResultCache(local_dir=byoc.BASELINE)
    documents, demo = byoc.read_corpus(None)
    result, generated = byoc.generate(documents, byoc.DEMO_CONFIG, REFERENCE_MODELS, live=False, demo=demo)
    levers = score_levers(result, generated, documents, live=True, cache=writer)
    logging.info("nDCG@10 %s", levers["mean"]["ndcg@10"])
    answers = answer_questions(
        result, generated, documents, REFERENCE_MODELS, live=True, cache=writer, env_path=args.env_file
    )
    logging.info(
        "null detection %s, correct %s of %s", answers["null_detection"], answers["correct"], answers["judged"]
    )
    print(spend.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
