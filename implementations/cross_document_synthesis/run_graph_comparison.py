"""Consume a verified RI #4 index and score its original-passage retrieval arm."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.metering import UsageMeter, current_question
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url
from aieng.advanced_rag.spend import SpendLimitReached
from run_benchmark import experiment_spec, open_report
from synthesis_lab.corpora import Corpus, Question, load_corpus
from synthesis_lab.evaluation import summarize
from synthesis_lab.graph import GraphAnswerer, PendingGraphArtifact, resolve_index
from synthesis_lab.graph_release import MANIFEST, hippo_precompute_files, release_entry
from synthesis_lab.runtime import (
    DEFAULT_CACHE,
    CachedProxyCompletion,
    local_embedder,
    write_json,
)


@dataclass(frozen=True)
class GraphSession:
    """A live graph ranking function and its asynchronous storage cleanup."""

    rank: Callable[[str, int], Awaitable[list[str]]]
    close: Callable[[], Awaitable[None]]


def mapped_passages(ids: Sequence[str], mapping: dict[str, str], known: set[str], k: int) -> list[str]:
    """Deduplicate chunks by original passage before spending the passage budget."""
    mapped = [mapping.get(pid, pid) for pid in ids]
    if not set(mapped) <= known:
        raise ValueError("graph chunk mapping contains unknown original passages")
    return list(dict.fromkeys(mapped))[:k]


async def ranker(directory: Path, metadata: dict[str, Any], *, models: ModelSet | None = None) -> GraphSession:
    """Query a prebuilt index through its actual shared retrieval adapter."""
    models = models or choose_models()
    if metadata["backend"] == "atlas-rag":
        return await hippo_ranker(directory, metadata, models=models)
    if metadata["backend"] != "lightrag":
        raise PendingGraphArtifact(
            f"RI #4 loading adapter for {metadata['backend']} is not supplied; backend substitution is disabled"
        )
    required = {
        "kv_store_doc_status.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
    }
    if any(not (directory / name).is_file() for name in required):
        raise PendingGraphArtifact("LightRAG handoff lacks prebuilt graph/vector stores")
    statuses = json.loads((directory / "kv_store_doc_status.json").read_text())
    if len(statuses) != len(metadata["passage_ids"]) or any(
        row.get("status") != "processed" for row in statuses.values()
    ):
        raise ValueError("LightRAG index does not contain all processed corpus passages")
    from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.evidence import to_runs  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, async_llm  # noqa: PLC0415
    from lightrag.utils import EmbeddingFunc  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415 - optional live path
    from synthesis_lab.runtime import MODEL, MODEL_REVISION  # noqa: PLC0415

    settings = metadata["settings"]
    if (
        settings.get("embedding_model") != MODEL
        or settings.get("query_embedding_revision", settings.get("embedding_revision")) != MODEL_REVISION
        or settings.get("mode") != "mix"
    ):
        raise ValueError("graph encoder or retrieval settings do not match the declared comparison")
    embedder = local_embedder(DEFAULT_CACHE)

    async def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(embedder.encode(texts), dtype=np.float32)

    client = AsyncOpenAI(base_url=resolve_base_url(), api_key=resolve_api_key(), timeout=90, max_retries=0)
    llm = async_llm(
        client,
        model=models.extraction.model,
        cache=ResponseCache(DEFAULT_CACHE.parent / "ri2-complete/graph-responses.sqlite"),
        limiter=RateLimiter(per_minute=60),
        ledger=CallLedger("ri2-graph-query"),
    )
    provider_errors: dict[int, list[Exception]] = {}

    async def observed_llm(*args: Any, **kwargs: Any) -> str:
        try:
            return str(await llm(*args, **kwargs))
        except Exception as error:
            provider_errors.setdefault(id(current_question()), []).append(error)
            raise

    retriever = await LightRAGRetriever.build(
        str(directory), observed_llm, EmbeddingFunc(embedding_dim=384, max_token_size=512, func=embed), mode="mix"
    )
    mapping = metadata.get("retriever_id_map", {})

    async def rank(question: str, k: int) -> list[str]:
        owner = id(current_question())
        try:
            evidence = await retriever.aretrieve(question, 40)
            errors = provider_errors.get(owner, [])
            if errors:
                budget_error = next((e for e in errors if isinstance(e, SpendLimitReached)), None)
                if budget_error is not None:
                    raise budget_error
                raise RuntimeError("LightRAG retrieval provider failed") from errors[0]
        finally:
            provider_errors.pop(owner, None)
        return mapped_passages(to_runs(evidence), mapping, set(metadata["passage_ids"]), k)

    async def close() -> None:
        await retriever._rag.finalize_storages()
        await client.close()

    return GraphSession(rank, close)


async def hippo_ranker(directory: Path, metadata: dict[str, Any], *, models: ModelSet | None = None) -> GraphSession:
    """Load the atlas-rag variant with its precomputed entity/relation indexes."""
    from aieng.advanced_rag.graphrag.adapters.hipporag2 import HippoRAG2Retriever  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.evidence import to_runs  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, govern  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415
    from synthesis_lab.runtime import MODEL, MODEL_REVISION  # noqa: PLC0415

    models = models or choose_models()
    settings = metadata["settings"]
    if settings.get("implementation") != "atlas-rag" or settings.get("encoder") != MODEL:
        raise ValueError("HippoRAG requires the released atlas-rag configuration")
    if any(not (directory / "precompute" / name).is_file() for name in hippo_precompute_files()):
        raise PendingGraphArtifact("HippoRAG lacks prebuilt vector stores; rebuilding is disabled")
    client = govern(
        OpenAI(base_url=resolve_base_url(), api_key=resolve_api_key(), timeout=90, max_retries=0),
        cache=ResponseCache(DEFAULT_CACHE.parent / "ri2-complete/graph-responses.sqlite"),
        limiter=RateLimiter(per_minute=60),
        ledger=CallLedger("ri2-hipporag-query"),
    )
    retriever = await asyncio.to_thread(
        HippoRAG2Retriever.build,
        kg_dir=str(directory),
        llm_client=client,
        llm_model=models.extraction.model,
        encoder_model=MODEL,
        encoder_revision=MODEL_REVISION,
        keyword="musique",
        include_concept=False,
        include_events=False,
    )
    known = set(metadata["passage_ids"])

    async def rank(question: str, k: int) -> list[str]:
        target = current_question()
        query_client = client.copy()
        create = client.chat.completions.create
        failures: list[Exception] = []

        def bound_create(**kwargs: Any) -> Any:
            try:
                if target is None:
                    return create(**kwargs)
                with target.bind():
                    return create(**kwargs)
            except Exception as error:
                failures.append(error)
                raise

        query_client.chat.completions.create = bound_create
        query_retriever = retriever.with_client(query_client)
        evidence = await asyncio.to_thread(query_retriever.retrieve, question, k)
        # Upstream catches provider exceptions and substitutes unfiltered facts.
        # Preserve those failures rather than score the fallback as a normal run.
        if failures:
            budget_error = next((e for e in failures if isinstance(e, SpendLimitReached)), None)
            if budget_error is not None:
                raise budget_error
            raise RuntimeError("atlas-rag retrieval provider failed") from failures[0]
        return mapped_passages(to_runs(evidence), metadata["retriever_id_map"], known, k)

    async def close() -> None:
        await asyncio.to_thread(client.close)

    return GraphSession(rank, close)


async def run(
    handoff: Path | None,
    output: Path,
    *,
    per_hop: int = 10,
    backend: str | None = None,
    artifact_manifest: Path | None = None,
    workers: int = 1,
    allow_incomplete_index: bool = False,
    models: ModelSet | None = None,
) -> dict[str, Any]:
    """Verify the index before opening a model client; never build missing indexes."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if (handoff is None) == (backend is None):
        raise ValueError("choose exactly one of a release backend or a handoff")
    descriptor = json.loads(handoff.read_text()) if handoff is not None else release_entry(str(backend))
    key = CacheKey(**descriptor["key"])
    corpus = load_corpus("musique", DEFAULT_CACHE)
    cache = TieredResultCache()
    if artifact_manifest is not None:
        cache.fetch_manifest(artifact_manifest)
    elif descriptor.get("artifact_manifest") and handoff is not None:
        cache.fetch_manifest((handoff.parent / descriptor["artifact_manifest"]).resolve())
    work = DEFAULT_CACHE.parent / "ri2-complete/graph" / key.digest
    directory, metadata = resolve_index(cache, key, corpus, work, allow_incomplete=allow_incomplete_index)
    if per_hop:
        corpus = replace(
            corpus,
            questions=tuple(
                q
                for category in ("2-hop", "3-hop", "4-hop")
                for q in [q for q in corpus.questions if q.category == category][:per_hop]
            ),
        )
    graph_sources = sorted(
        (Path(__file__).resolve().parents[2] / "aieng-advanced-rag/aieng/advanced_rag/graphrag").rglob("*.py")
    )
    models = models or choose_models()
    spec = experiment_spec(corpus, metadata.get("producer_backend", metadata["backend"]), models=models) | {
        "retrieval_source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in graph_sources)).hexdigest(),
        "workers": workers,
        "implementation": metadata["settings"].get("implementation", metadata["backend"]),
        "artifact_key": key.digest,
        "handoff_sha256": hashlib.sha256((handoff or MANIFEST).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "producer": metadata,
        "graph_settings": metadata["settings"],
        "index_coverage": metadata.get("index_coverage", {"complete": True}),
    }
    report = open_report(output, spec)
    if report["completed"]:
        return report
    spend.session()
    report["produced_by"] = {"model": models.generation.model, "model_roles": models.as_dict()}
    report["evidence_cohorts"] = evidence_cohorts(
        corpus, set(metadata.get("index_coverage", {}).get("missing_passage_ids", []))
    )
    report["index_build_cost"] = metadata.get("index_build_cost")
    report["index_cost_status"] = "reported" if report["index_build_cost"] is not None else "unavailable"
    # Query-time storage writes must never change the verified extracted index.
    with tempfile.TemporaryDirectory(prefix="ri2-graph-query-", dir=work.parent) as temporary:
        query_directory = Path(temporary) / "index"
        shutil.copytree(directory, query_directory)
        session = await ranker(query_directory, metadata, models=models)
        try:
            return await score_queries(corpus, session.rank, report, output, workers=workers, models=models)
        finally:
            await session.close()


def evidence_cohorts(corpus: Corpus, missing_ids: set[str]) -> dict[str, list[str]]:
    """Partition scorer annotations; gold supports never enter the answerer."""
    return {
        "no_directly_missing_support": [
            q.question_id for q in corpus.questions if not missing_ids.intersection(q.supporting_ids)
        ],
        "directly_missing_support": [
            q.question_id for q in corpus.questions if missing_ids.intersection(q.supporting_ids)
        ],
    }


async def score_queries(
    corpus: Corpus,
    rank: Callable[[str, int], Awaitable[list[str]]],
    report: dict[str, Any],
    output: Path,
    *,
    workers: int = 1,
    models: ModelSet | None = None,
) -> dict[str, Any]:
    """Record every attempted question and preserve failures and spent usage."""
    models = models or choose_models()
    answerer = GraphAnswerer(corpus.passages, rank, CachedProxyCompletion(DEFAULT_CACHE, models.generation.model))
    meter = UsageMeter()
    done = {row["question_id"] for row in report["records"]}
    gate = asyncio.Semaphore(workers)
    saved_wall = float(report.get("execution_wall_seconds") or 0.0)
    started = time.perf_counter()

    async def one(q: Question) -> None:
        async with gate:
            begin = time.perf_counter()
            with meter.question(q.question_id):
                try:
                    result = await answerer.aanswer(q.question_id, q.question)
                    trace = answerer.traces[q.question_id]
                    row = {
                        "question_id": q.question_id,
                        "answer": result.answer.text,
                        "abstained": result.abstained,
                        "retrieved_ids": result.retrieved_ids,
                        "generation_ids": trace["generation_ids"],
                        "citation_ids": list(result.answer.chunk_ids),
                        "dropped_citations": list(result.answer.dropped_citations),
                        "failure": None,
                        "stop_reason": trace["stop_reason"],
                        "trace": trace,
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
                        "failure": {"type": type(error).__name__, "message": str(error)[:500]},
                        "stop_reason": "execution_failure",
                    }
            row["usage"] = meter.per_question()[q.question_id].to_dict()
            row["elapsed_seconds"] = time.perf_counter() - begin
            report["records"].append(row)
            report["execution_wall_seconds"] = saved_wall + time.perf_counter() - started
            write_json(output, report)

    await asyncio.gather(*(one(q) for q in corpus.questions if q.question_id not in done))
    order = {q.question_id: i for i, q in enumerate(corpus.questions)}
    report["records"].sort(key=lambda row: order[row["question_id"]])
    report["workers"] = workers
    report["execution_wall_seconds"] = saved_wall + time.perf_counter() - started
    report["scores"] = summarize(corpus.questions, report["records"])
    report["coverage_complete"] = True
    report["completed"] = True
    report["execution_success"] = not any(row["failure"] for row in report["records"])
    write_json(output, report)
    return report


def main() -> None:
    """Inspect saved results or opt into a verified graph run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--handoff", type=Path)
    source.add_argument("--backend", choices=("hipporag2", "lightrag"))
    parser.add_argument("--artifact-manifest", type=Path, help="Checksum-pinned local or release download manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-hop", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-incomplete-index",
        action="store_true",
        help="Explore the pinned LightRAG release despite missing chunks; record coverage and separate affected questions",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.per_hop < 0 or (args.live and args.handoff is None and args.backend is None):
        parser.error("live graph runs need --backend or --handoff and non-negative --per-hop")
    try:
        if args.live:
            models = choose_models(show=True)
            try:
                report = asyncio.run(
                    run(
                        args.handoff,
                        args.output,
                        per_hop=args.per_hop,
                        backend=args.backend,
                        artifact_manifest=args.artifact_manifest,
                        workers=args.workers,
                        allow_incomplete_index=args.allow_incomplete_index,
                        models=models,
                    )
                )
            finally:
                print(spend.report())
        else:
            report = json.loads(args.output.read_text())
    except PendingGraphArtifact as error:
        parser.exit(2, f"Pending graph handoff: {error}\n")
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()
