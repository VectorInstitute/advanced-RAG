"""Inspect saved decomposition scores, or opt into a bounded Vector proxy run."""

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
    fetch_musique,
    load_musique,
)
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.pipeline.answering import Answerer
from aieng.advanced_rag.proxy import resolve_base_url
from synthesis_lab.decomposition import DecompositionAnswerer
from synthesis_lab.dense_baseline import score_by_hops
from synthesis_lab.reference import reference_dense_predictions
from synthesis_lab.runtime import (
    DEFAULT_CACHE,
    LAB_DIR,
    MODEL,
    MODEL_REVISION,
    CachedProxyCompletion,
    embeddings,
    local_embedder,
    write_json,
)


DEFAULT_LIVE_OUTPUT = DEFAULT_CACHE.parent / "decomposition/live.json"


def select_questions(questions: list[MusiqueQuestion], per_hop: int) -> list[MusiqueQuestion]:
    """Select a deterministic balanced smoke slice, or the full split with zero."""
    if per_hop < 0:
        raise ValueError("per_hop must be non-negative")
    if per_hop == 0:
        return questions
    return [q for hops in (2, 3, 4) for q in [q for q in questions if q.hops == hops][:per_hop]]


def matched_dense_control(questions: list[MusiqueQuestion]) -> dict[str, Any]:
    """Read and score controls before making any new model calls."""
    baseline_dir = LAB_DIR / "outputs/reference/dense_baseline"
    runs = json.loads((baseline_dir / "rankings.json").read_text())
    # The predictions hold answer text, so they ship in RI #2's reference release.
    predictions = {
        row["question_id"]: "" if row["abstained"] else row["answer"] for row in reference_dense_predictions()
    }
    return score_by_hops(
        questions,
        {q.question_id: runs[q.question_id] for q in questions},
        [5, 10, 20],
        {q.question_id: predictions[q.question_id] for q in questions},
    )


def run_live(cache_dir: Path, output: Path, per_hop: int, *, models: ModelSet | None = None) -> dict[str, Any]:
    """Run the Answerer on compute with local embeddings and per-question usage."""
    models = models or choose_models()
    spend.session()
    passages, all_questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    questions = select_questions(all_questions, per_hop)
    control = matched_dense_control(questions)
    embedder = local_embedder(cache_dir)
    docs, _, embedding_record = embeddings(cache_dir, passages, all_questions, embedder=embedder)
    complete = CachedProxyCompletion(cache_dir, models.generation.model)
    plan = CachedProxyCompletion(cache_dir, models.extraction.model)
    implementation = DecompositionAnswerer(passages, docs, embedder.encode_queries, plan, complete)
    answerer: Answerer = implementation
    meter = UsageMeter()
    report: dict[str, Any] = {
        "method": "dense-decomposition-round-robin",
        "corpus_revision": MUSIQUE_REVISION,
        "corpus_passages": len(passages),
        "embedding_model": MODEL,
        "embedding_revision": MODEL_REVISION,
        "embeddings": embedding_record,
        "answer_model": models.generation.model,
        "model_roles": models.as_dict(),
        "base_url": resolve_base_url(),
        "thinking": False,
        "temperature": 0,
        "max_tokens_per_completion": 256,
        "max_subquestions": 4,
        "retrieval_depth_per_query": 20,
        "evidence_budget": 5,
        "original_question_anchor": True,
        "pooling": "rank-first round robin, subquestions then original, unique IDs",
        "workers": 1,
        "max_attempts": 1,
        "selection": "full split" if per_hop == 0 else f"first {per_hop} questions per hop count in pinned file order",
        "purpose": "full benchmark"
        if per_hop == 0
        else "balanced integration slice; not a population quality estimate",
        "selected_questions": len(questions),
        "completed": False,
        "attribution": MUSIQUE_LICENCE.attribution,
        "records": [],
    }
    runs: dict[str, list[str]] = {}
    predictions: dict[str, str] = {}
    for q in questions:
        started = time.perf_counter()
        request_start = len(complete.requests)
        plan_start = len(plan.requests)
        with meter.question(q.question_id):
            result = answerer.answer(q.question_id, q.question)
        runs[q.question_id] = result.retrieved_ids
        predictions[q.question_id] = "" if result.abstained else result.answer.text
        report["records"].append(
            {
                "question_id": q.question_id,
                "hops": q.hops,
                **implementation.traces[q.question_id],
                "completion_requests": [*plan.requests[plan_start:], *complete.requests[request_start:]],
                "usage": meter.per_question()[q.question_id].to_dict(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        report["usage"] = meter.to_dict()
        write_json(output, report)
        print(f"Completed {len(runs)}/{len(questions)}: {q.question_id}", flush=True)
    report["metrics"] = score_by_hops(questions, runs, [5, 10, 20], predictions)
    report["matched_dense_baseline"] = control
    report["dense_control_note"] = (
        "Saved Qwen reference control; use run_benchmark.py for fresh matched model-role comparisons."
    )
    report["completed"] = True
    write_json(output, report)
    return report


def main() -> None:
    """Show saved scores by default; gate corpus, embedding, and proxy execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output", type=Path, help="Defaults to the local cache for live runs, the reference worked example otherwise"
    )
    parser.add_argument("--per-hop", type=int, default=10, help="Questions per hop count; 0 runs all 1,000")
    args = parser.parse_args()
    if args.per_hop < 0:
        parser.error("--per-hop must be non-negative")
    output = args.output or DEFAULT_LIVE_OUTPUT
    if args.live:
        models = choose_models(show=True)
        try:
            report = run_live(args.cache_dir, output, args.per_hop, models=models)
        finally:
            print(spend.report())
    elif args.output is not None:
        report = json.loads(output.read_text())
    else:
        from synthesis_lab.reference import decomposition_example  # noqa: PLC0415

        report = decomposition_example(per_hop=args.per_hop).value["report"]
    print(
        json.dumps(
            {key: report[key] for key in ("completed", "selection", "metrics", "matched_dense_baseline")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
