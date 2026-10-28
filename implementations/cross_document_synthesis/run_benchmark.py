"""Run matched RI #2 methods with durable records and replayable scores."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.agents.tools import ToolFunction
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url
from aieng.advanced_rag.spend import SpendLimitReached
from synthesis_lab.corpora import Corpus, Question, load_corpus, load_rewrites
from synthesis_lab.decomposition import DECOMPOSITION_TEMPLATE, DecompositionAnswerer
from synthesis_lab.dense_baseline import ANSWER_TEMPLATE
from synthesis_lab.evaluation import summarize
from synthesis_lab.ircot import IRCOT_INSTRUCTION, IRCoTAnswerer
from synthesis_lab.methods import DenseAnswerer
from synthesis_lab.runtime import (
    DEFAULT_CACHE,
    LAB_DIR,
    MODEL,
    MODEL_REVISION,
    CachedProxyCompletion,
    write_json,
)
from synthesis_lab.vectors import corpus_vectors


def experiment_spec(
    corpus: Corpus, method: str, *, response_cache: Path | None = None, models: ModelSet | None = None
) -> dict[str, Any]:
    """Identify input text, executable methods, and shared prompts."""
    models = models or choose_models()
    source_root = LAB_DIR.parents[1] / "aieng-advanced-rag/aieng/advanced_rag"
    paths = [
        *sorted((LAB_DIR / "synthesis_lab").glob("*.py")),
        Path(__file__),
        *(source_root / name for name in ("models.py", "proxy.py", "spend.py", "metering.py")),
        *sorted((source_root / "pipeline").glob("*.py")),
        *sorted((source_root / "agents").glob("*.py")),
    ]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    spec: dict[str, Any] = {
        "schema_version": 1,
        "corpus": corpus.name,
        "provenance": corpus.provenance,
        "method": method,
        "questions": [{"id": q.question_id, "text": q.question} for q in corpus.questions],
        "model": models.generation.model,
        "model_roles": models.as_dict(),
        "base_url": resolve_base_url(),
        "embedding": [MODEL, MODEL_REVISION],
        "source_sha256": source_hash,
        "temperature": 0,
        "thinking": False,
        "answer_prompt_sha256": hashlib.sha256(ANSWER_TEMPLATE.encode()).hexdigest(),
        "planning_prompt_sha256": hashlib.sha256(
            (IRCOT_INSTRUCTION if method == "ircot" else DECOMPOSITION_TEMPLATE).encode()
        ).hexdigest(),
        "bounds": {
            "retrieval_depth": 20,
            "generation_passages": 5,
            "ircot_rounds": 4,
            "ircot_exposed_per_round": 5,
            "ircot_model_calls": 10,
            "corrective_attempts": 2,
        },
        "corrective_thresholds": {"sufficient": 0.6, "insufficient": 0.0, "tuned_on_musique": False},
    }

    if response_cache is not None:
        spec["response_cache_namespace"] = cache_namespace(response_cache)
    return spec


def cache_namespace(response_cache: Path) -> str:
    """Name a response cache by its path under the local cache root.

    The name is part of the resume identity and enters committed reports, so it
    must not carry a home directory or host layout. A cache outside the root is
    named by its final directory.
    """
    resolved = response_cache.resolve()
    root = DEFAULT_CACHE.parent.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else resolved.name


def open_report(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Reject incompatible resumption and duplicate/unselected result records."""
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if path.exists():
        report: dict[str, Any] = json.loads(path.read_text())
        if report["spec_sha256"] != digest or report["spec"] != spec:
            raise ValueError("report configuration differs; choose another output path")
        ids = [r["question_id"] for r in report["records"]]
        if len(ids) != len(set(ids)) or not set(ids) <= {q["id"] for q in spec["questions"]}:
            raise ValueError("invalid report coverage")
        return report
    return {
        "spec": spec,
        "spec_sha256": digest,
        "completed": False,
        "records": [],
        "cost_note": "Usage is actual execution usage. Cache hits are not cold-run cost; original service cost is unknown unless recorded separately.",
    }


async def run(  # noqa: PLR0915 - metered execution and checkpoint lifecycle
    corpus: Corpus,
    method: str,
    output: Path,
    cache: Path,
    *,
    workers: int = 1,
    response_cache: Path | None = None,
    models: ModelSet | None = None,
) -> dict[str, Any]:
    """Execute or resume a matched run, retaining failures with zero credit."""
    if workers < 1:
        raise ValueError("workers must be positive")
    models = models or choose_models()
    spec = experiment_spec(corpus, method, response_cache=response_cache, models=models)
    report = open_report(output, spec)
    if report["completed"]:
        report["scores"] = summarize(corpus.questions, report["records"])
        return report
    spend.session()
    docs, encode, manifest = corpus_vectors(corpus, cache)
    report["embeddings"] = manifest
    report["workers"] = workers
    meter = UsageMeter()
    completion_cache = response_cache if response_cache is not None else DEFAULT_CACHE
    complete = CachedProxyCompletion(completion_cache, models.generation.model)
    plan = (
        CachedProxyCompletion(completion_cache, models.extraction.model)
        if method in {"decomposition", "corrective"}
        else complete
    )
    if method == "ircot":
        from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: PLC0415 - optional extra

        def loop_factory(tools: Sequence[ToolFunction]) -> AgentLoop:
            return AgentLoop(
                tools,
                model=models.extraction.model,
                base_url=resolve_base_url(),
                api_key=resolve_api_key(),
                instruction=IRCOT_INSTRUCTION,
                max_model_calls=10,
                max_tokens=768,
                enable_thinking=False,
            )

        answerer: Any = IRCoTAnswerer(corpus.passages, docs, encode, complete, loop_factory)
    elif method == "decomposition":
        answerer = DecompositionAnswerer(corpus.passages, docs, encode, plan, complete)
    elif method in {"dense", "corrective", "graded-single-pass"}:
        answerer = DenseAnswerer(
            corpus.passages,
            docs,
            encode,
            complete,
            corrective=method == "corrective",
            graded_control=method == "graded-single-pass",
            grade=CachedProxyCompletion(completion_cache, models.judging.model)
            if method in {"corrective", "graded-single-pass"}
            else None,
            rewrite=plan,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    done = {r["question_id"] for r in report["records"]}
    semaphore = asyncio.Semaphore(workers)
    started = time.perf_counter()

    async def one(q: Question) -> None:
        async with semaphore:
            begin = time.perf_counter()
            with meter.question(q.question_id):
                try:
                    # The reference decomposer keeps its last plan for tracing.
                    # Give each concurrent question a private instance.
                    implementation: Any = (
                        DecompositionAnswerer(corpus.passages, docs, encode, plan, complete)
                        if method == "decomposition"
                        else answerer
                    )
                    result = (
                        await implementation.aanswer(q.question_id, q.question)
                        if method == "ircot"
                        else await asyncio.to_thread(implementation.answer, q.question_id, q.question)
                    )
                    trace = implementation.traces[q.question_id]
                    row = {
                        "question_id": q.question_id,
                        "answer": result.answer.text,
                        "abstained": result.abstained,
                        "retrieved_ids": trace.get("retrieved_ids", result.retrieved_ids),
                        "generation_ids": trace["generation_ids"],
                        "citation_ids": list(result.answer.chunk_ids),
                        "dropped_citations": list(result.answer.dropped_citations),
                        "stop_reason": trace.get("stop_reason", result.answer.reason or "answered"),
                        "failure": None,
                        "trace": trace,
                        "result": result.to_dict(),
                    }
                except SpendLimitReached:
                    report["interrupted"] = {"reason": "spend_limit", "question_id": q.question_id}
                    write_json(output, report)
                    raise
                except Exception as error:
                    row = {
                        "question_id": q.question_id,
                        "answer": "",
                        "abstained": True,
                        "retrieved_ids": [],
                        "generation_ids": [],
                        "citation_ids": [],
                        "stop_reason": "execution_failure",
                        "failure": {"type": type(error).__name__, "message": str(error)[:500]},
                    }
            row["usage"] = meter.per_question()[q.question_id].to_dict()
            row["elapsed_seconds"] = time.perf_counter() - begin
            report["records"].append(row)
            write_json(output, report)
            print(
                f"{corpus.name}/{method}: {len(report['records'])}/{len(corpus.questions)} {q.question_id} {row['stop_reason']}",
                flush=True,
            )

    await asyncio.gather(*(one(q) for q in corpus.questions if q.question_id not in done))
    order = {q.question_id: i for i, q in enumerate(corpus.questions)}
    report["records"].sort(key=lambda row: order[row["question_id"]])
    report["scores"] = summarize(corpus.questions, report["records"])
    report["coverage_complete"] = len(report["records"]) == len(corpus.questions)
    report["completed"] = report["coverage_complete"]
    report["execution_success"] = not any(row["failure"] for row in report["records"])
    report["execution_wall_seconds"] = report.get("execution_wall_seconds", 0) + time.perf_counter() - started
    write_json(output, report)
    return report


def select_shared_300(corpus: Corpus) -> Corpus:
    """Return the corpus limited to the shared 300 MuSiQue questions, in corpus order.

    RI #4, RI #5, and the special topics draw the same 150 two-hop, 75 three-hop,
    and 75 four-hop questions with ``stratified_test_set`` and its fixed seed.
    """
    if corpus.name != "musique":
        raise ValueError("the shared 300 questions are a MuSiQue selection")
    from aieng.advanced_rag.corpora.musique import load_musique, stratified_test_set  # noqa: PLC0415

    _, raw_questions = load_musique()
    wanted = {q.question_id for q in stratified_test_set(raw_questions)}
    selected = tuple(q for q in corpus.questions if q.question_id in wanted)
    if len(selected) != len(wanted):
        raise ValueError(f"expected {len(wanted)} shared questions, found {len(selected)}")
    return replace(corpus, questions=selected)


def main() -> None:
    """Require explicit live execution; saved reports can be inspected offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--corpus", choices=("musique", "nq", "multihop-rag"), default="musique")
    parser.add_argument(
        "--method", choices=("dense", "decomposition", "ircot", "corrective", "graded-single-pass"), default="dense"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE.parent / "ri2-complete")
    parser.add_argument("--per-category", type=int, default=10, help="First N per category; 0 selects all")
    parser.add_argument(
        "--shared-300",
        action="store_true",
        help="Select the bootcamp's shared 300 MuSiQue questions (stratified_test_set); ignores --per-category",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--response-cache-dir",
        type=Path,
        help="Isolated generation/grading cache; namespace is part of resume identity",
    )
    parser.add_argument("--rewrites", type=Path)
    parser.add_argument("--article-ruling", type=Path)
    args = parser.parse_args()
    if args.per_category < 0 or args.workers < 1:
        parser.error("selection must be non-negative and workers positive")
    if not args.live:
        report = json.loads(args.output.read_text())
    else:
        corpus = load_corpus(
            args.corpus,
            DEFAULT_CACHE if args.corpus == "musique" else args.cache_dir / args.corpus,
            ruling=args.article_ruling,
        )
        if args.rewrites:
            corpus = load_rewrites(args.rewrites, corpus)
        if args.shared_300:
            corpus = select_shared_300(corpus)
        elif args.per_category:
            counts: dict[str, int] = {}
            selected = []
            for q in corpus.questions:
                # Keep both rewrite variants together when slicing original IDs.
                key = q.category + "/" + q.wording
                counts[key] = counts.get(key, 0) + 1
                if counts[key] <= args.per_category:
                    selected.append(q)
            corpus = replace(corpus, questions=tuple(selected))
        models = choose_models(show=True)
        try:
            report = asyncio.run(
                run(
                    corpus,
                    args.method,
                    args.output,
                    args.cache_dir,
                    workers=args.workers,
                    response_cache=args.response_cache_dir,
                    models=models,
                )
            )
        finally:
            print(spend.report())
    print(json.dumps({k: report.get(k) for k in ("completed", "coverage_complete", "scores")}, indent=2))


if __name__ == "__main__":
    main()
