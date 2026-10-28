"""Run the dense control through the shared retrieval and answering harness."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_LICENCE,
    MUSIQUE_REVISION,
    MusiqueQuestion,
    Passage,
    fetch_musique,
    load_musique,
)
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.proxy import resolve_base_url
from synthesis_lab.dense_baseline import DensePassageRetriever, answer_baseline, rank_questions, score_by_hops
from synthesis_lab.runtime import (
    DEFAULT_CACHE,
    CachedProxyCompletion,
    embeddings,
    write_json,
)


def run_baseline(
    cache_dir: Path,
    output_dir: Path,
    *,
    answer: bool = False,
    answer_model: str | None = None,
    models: ModelSet | None = None,
) -> dict[str, Any]:
    """Load and rank the pinned corpus, then optionally score cited answers."""
    models = choose_models(generation=answer_model) if answer_model is not None else (models or choose_models())
    cache_dir.mkdir(parents=True, exist_ok=True)
    passages, questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    docs, queries, embedding_record = embeddings(cache_dir, passages, questions)
    started = time.perf_counter()
    runs = rank_questions(passages, questions, docs, queries, k=20)
    ranking_seconds = time.perf_counter() - started
    report = {
        "method": "dense-bge-small",
        "corpus_revision": MUSIQUE_REVISION,
        "corpus_passages": len(passages),
        "questions": len(questions),
        "embeddings": embedding_record,
        "ranking_seconds": ranking_seconds,
        "metrics": score_by_hops(questions, runs, [5, 10, 20]),
        "retrieval_model_calls": 0,
        "latency_note": "Query embeddings are batched; batch time per query is throughput, not single-query latency.",
        "attribution": MUSIQUE_LICENCE.attribution,
    }
    write_json(output_dir / "retrieval.json", report)
    write_json(output_dir / "rankings.json", runs)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    if answer:
        spend.session()
        _answer_questions(
            passages, questions, runs, cache_dir, output_dir, answer_model=models.generation.model, models=models
        )
    return report


def _answer_questions(
    passages: list[Passage],
    questions: list[MusiqueQuestion],
    runs: dict[str, list[str]],
    cache_dir: Path,
    output_dir: Path,
    *,
    answer_model: str,
    models: ModelSet | None = None,
) -> None:
    """Keep answer scoring, traces, and usage alongside the retrieval report."""
    completion = CachedProxyCompletion(cache_dir, answer_model)
    by_text = {q.question: runs[q.question_id] for q in questions}
    retriever = DensePassageRetriever(passages, by_text)
    meter = UsageMeter()
    arm = answer_baseline(questions, retriever, completion, meter)
    predictions = {r.query_id: "" if r.abstained else r.answer.text for r in arm.results}
    answer_report = {
        "method": "dense-single-pass",
        "retrieval": "retrieval.json",
        "answer_model": answer_model,
        "model_roles": (models or choose_models()).as_dict(),
        "base_url": resolve_base_url(),
        "evidence_budget": 5,
        "max_chars": None,
        "temperature": 0,
        "thinking": False,
        "require_citations": True,
        "workers": 1,
        "metrics": score_by_hops(questions, runs, [5], predictions),
        "harness": arm.to_dict(),
        "usage": meter.to_dict(),
        "attribution": MUSIQUE_LICENCE.attribution,
    }
    write_json(output_dir / "answers.json", answer_report)
    per_question_usage = meter.per_question()
    write_json(
        output_dir / "traces.json",
        [
            {
                "question_id": r.query_id,
                "result": r.to_dict(),
                "raw_answer": r.answer.raw,
                "completion_requests": [request],
            }
            for r, request in zip(arm.results, completion.requests, strict=True)
        ],
    )
    write_json(
        output_dir / "predictions.json",
        [
            {
                "question_id": r.query_id,
                "answer": r.answer.text,
                "abstained": r.abstained,
                "citation_ids": list(r.answer.chunk_ids),
                "usage": per_question_usage[r.query_id].to_dict(),
            }
            for r in arm.results
        ],
    )
    print(json.dumps(answer_report["metrics"], indent=2), flush=True)


def main() -> None:
    """Score the full pinned corpus and optionally run the answering harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE.parent / "ri2-experiments/dense-baseline")
    parser.add_argument("--answer", action="store_true", help="Enable Vector proxy generation")
    parser.add_argument("--answer-model", help="Override the selected generation role")
    args = parser.parse_args()
    if args.answer:
        models = choose_models(generation=args.answer_model, show=True)
        try:
            run_baseline(args.cache_dir, args.output_dir, answer=True, models=models)
        finally:
            print(spend.report())
    else:
        run_baseline(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
