#!/usr/bin/env python3
"""Build the LightRAG and HippoRAG 2 indexes over the MuSiQue passages.

Both methods read the same passages, in the same order, through the same proxy
model, request budget, and response cache, so the comparison is between methods
and not between models or limits. Every model call is counted at the client and
the report says what the index cost.

LightRAG runs at its defaults: 300-token chunks, one gleaning pass, and its own
extraction prompt. Each passage is inserted with its corpus identifier as the
``file_path``, so a retrieved chunk reports the passage it came from.

HippoRAG 2 runs through ``atlas-rag``, which implements the method without the
``hipporag`` package's vLLM and torch pins. It is atlas-rag's variant, not the
paper's: the extraction prompt and the graph build are atlas-rag's. Only the
entity-relation stage runs, which is the phrase-and-passage graph HippoRAG 2
retrieves over. Atlas-rag's event stages and concept generation stay off, as does
every option that would add calls per passage.

A partial run is safe. The response cache turns a repeat into hits, and a build
that finishes with documents unprocessed raises instead of returning an index
that quietly lacks them. LightRAG writes a document's status to disk at once but
holds its chunks, vectors, and graph in memory until the next flush, so a build
killed between flushes leaves documents marked processed that the index lacks.
On start the LightRAG build finds those documents, removes their records, and
indexes them again. When it finishes, it fails if any processed document lacks
a chunk, a vector, or the graph links its extraction listed. Only one build can
use a working directory at a time.

The build model is separate from the model that answers questions. It comes
from ``--model``, then ``ADVANCED_RAG_EXTRACTION_MODEL``, then the deprecated
``LLM_MODEL``, then :data:`REFERENCE_EXTRACTION_MODEL`, the model the reference
indexes were built with. ``run_musique_test.py`` restores the indexes by the
build model and answers on its own model, so answering on another model still
loads the reference indexes.

Examples
--------
Measure on a sample and project the full run, then build everything::

    $ python build_indexes.py lightrag --passages 100
    $ python build_indexes.py hipporag2 --passages 100
    $ python build_indexes.py lightrag

Build on another model, into its own cache key::

    $ python build_indexes.py hipporag2 --model gpt-6-luna
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import signal
import sys
import time
import warnings
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from aieng.advanced_rag.cache import CacheKey, TieredResultCache, store_directory
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora.musique import MUSIQUE_REVISION, Passage, load_musique
from aieng.advanced_rag.graphrag.lightrag_index import incomplete_documents, index_problems
from aieng.advanced_rag.graphrag.proxy import (
    CallLedger,
    RateLimiter,
    ResponseCache,
    async_llm,
    govern,
)
from aieng.advanced_rag.models import ROLE_ENV, endpoint_of
from aieng.advanced_rag.proxy import KEY_NAMES, MissingProxyKeyError, resolve_api_key
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder


# The default names the Vector Institute proxy. Set LLM_BASE_URL to use another
# OpenAI-compatible endpoint.
BASE_URL = os.environ.get("LLM_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
#: The model the reference MuSiQue indexes were built with. Their cache keys name
#: it, so it stays fixed whatever model a participant answers with.
REFERENCE_EXTRACTION_MODEL = "Qwen3_6-35B-A3B"
#: Read for one release, with a DeprecationWarning naming the replacement.
DEPRECATED_MODEL_ENV = "LLM_MODEL"
ENCODER = "BAAI/bge-small-en-v1.5"
RI_ID = "ri4"
# Stay under the endpoint's per-model request cap, which is 90 a minute for the
# default model.
DEFAULT_RPM = 80
DEFAULT_WORKERS = 64
# Atlas-rag waits for the slowest call in a batch before it starts the next, so a
# small batch leaves the request budget idle. A large one amortises that wait.
DEFAULT_BATCH = 512
FLUSH_INTERVAL_SECONDS = 180.0
LIGHTRAG_CHUNK_TOKENS = 300
LIGHTRAG_CHUNK_OVERLAP = 50
HIPPORAG_KEYWORD = "musique"


def build_model(cli: str | None = None) -> str:
    """Return the model to build with, from the flag, the environment, or the reference.

    ``LLM_MODEL`` used to set both the build model and the answering model, so a
    participant who set it to answer on a paid model lost the reference index.
    It is still read here, for the build only, with a warning.
    """
    if cli:
        return cli
    chosen = os.environ.get(ROLE_ENV["extraction"], "").strip()
    if chosen:
        return chosen
    legacy = os.environ.get(DEPRECATED_MODEL_ENV, "").strip()
    if legacy:
        warnings.warn(
            f"{DEPRECATED_MODEL_ENV} is deprecated: set {ROLE_ENV['extraction']} or "
            f"--model for the build model and {ROLE_ENV['generation']} for the "
            "answering model",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return REFERENCE_EXTRACTION_MODEL


def work_root() -> Path:
    """Return the directory the builds and their response caches live under."""
    return Path(cache_root()) / "graphrag" / "musique"


def passage_text(passage: Passage) -> str:
    """Return the text a method indexes: the article title, then the passage."""
    return f"{passage.title}\n{passage.text}"


def select(passages: Sequence[Passage], count: int | None) -> list[Passage]:
    """Return the first ``count`` passages, or all of them."""
    return list(passages if count is None else passages[:count])


def api_key() -> str:
    """Return the API key, never printing it.

    ``LLM_API_KEY`` comes first, for a run against another endpoint, then the
    proxy key through :func:`aieng.advanced_rag.proxy.resolve_api_key`: the
    environment, then the repository ``.env``.
    """
    try:
        key: str = resolve_api_key(names=("LLM_API_KEY", *KEY_NAMES))
    except MissingProxyKeyError as error:
        raise SystemExit(str(error)) from error
    return key


def report(
    method: str,
    ledger: CallLedger,
    *,
    model: str,
    passages: int,
    seconds: float,
    total: int,
    rpm: int,
) -> dict[str, Any]:
    """Return the cost of a build and a projection to the full corpus."""
    cost = ledger.to_dict()
    cold = cost["calls"] + cost["cache_hits"]
    scale = total / passages
    projection = {
        "calls": round(cold * scale),
        "total_tokens": round(cost["total_tokens"] * scale),
        "hours_at_rpm": round(cold * scale / rpm / 60, 1),
    }
    return {
        "produced_by": {
            "model": model,
            "endpoint": endpoint_of(model),
            "producer": "graphrag_technical_support/build_indexes.py",
        },
        "method": method,
        "model": model,
        "passages": passages,
        "wall_seconds": round(seconds, 1),
        "cost": cost,
        "calls_per_passage": round(cold / passages, 2),
        "tokens_per_passage": round(cost["total_tokens"] / passages, 1) if cost["calls"] else None,
        "projected_for_all_passages": projection if passages < total else None,
    }


def embedder_function(embedder: SentenceTransformerEmbedder) -> Any:
    """Return LightRAG's embedding function over the local encoder."""
    from lightrag.utils import EmbeddingFunc  # noqa: PLC0415

    async def embed(texts: Sequence[str]) -> Any:
        return np.array(embedder.encode(list(texts)), dtype=np.float32)

    return EmbeddingFunc(embedding_dim=embedder.dimension(), max_token_size=512, func=embed)


def document_statuses(working_dir: Path) -> dict[str, int]:
    """Count LightRAG's per-document statuses from its status store."""
    path = working_dir / "kv_store_doc_status.json"
    counts: dict[str, int] = {}
    if path.is_file():
        for record in json.loads(path.read_text()).values():
            counts[str(record.get("status"))] = counts.get(str(record.get("status")), 0) + 1
    return counts


def known_document_ids(working_dir: Path) -> set[str]:
    """Return the ids LightRAG holds, after dropping duplicate-name records."""
    path = working_dir / "kv_store_doc_status.json"
    if not path.is_file():
        return set()
    records = json.loads(path.read_text())
    kept = {
        key: record
        for key, record in records.items()
        if not str(record.get("error_msg", "")).startswith("File name already exists")
    }
    if len(kept) != len(records):
        path.write_text(json.dumps(kept))
    return set(kept)


async def forget_documents(rag: Any, doc_ids: Sequence[str]) -> None:
    """Remove the records that make LightRAG treat ``doc_ids`` as indexed.

    Deletes each document's chunks and chunk vectors, its extraction anchors, and
    then its stored text and status, and flushes, so the next insert indexes it
    from the start. Graph nodes and edges the document did reach stay: re-merging
    the same extraction leaves their sources and descriptions as they were.

    ``LightRAG.adelete_by_doc_id`` is not used. For an entity other documents
    share, it rebuilds the description from LightRAG's extraction cache, which this
    build turns off, so it would drop what those other documents contributed.
    """
    if not doc_ids:
        return
    ids = list(doc_ids)
    records = await rag.doc_status.get_by_ids(ids)
    chunk_ids = sorted({c for r in records if r for c in r.get("chunks_list") or []})
    if chunk_ids:
        await rag.chunks_vdb.delete(chunk_ids)
        await rag.text_chunks.delete(chunk_ids)
    await rag.full_entities.delete(ids)
    await rag.full_relations.delete(ids)
    await rag._flush_storages([rag.chunks_vdb, rag.text_chunks, rag.full_entities, rag.full_relations])  # noqa: SLF001
    # The status goes last, so a run stopped before here still finds the
    # document incomplete and repeats this.
    await rag.full_docs.delete(ids)
    await rag.doc_status.delete(ids)
    await rag._flush_storages([rag.full_docs, rag.doc_status])  # noqa: SLF001


@contextlib.contextmanager
def exclusive(directory: Path) -> Iterator[None]:
    """Hold a lock on ``directory`` so a second build cannot write it at the same time.

    The lock file sits beside the directory, so the stored artifact never holds it.

    Raises
    ------
    SystemExit
        If another process holds the lock.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    path = directory.with_name(f"{directory.name}.lock")
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"another build is writing {directory}") from error
        yield


def throttle_flushes(rag: Any, interval: float) -> Callable[[], Awaitable[None]]:
    """Flush LightRAG's storages at most every ``interval`` seconds.

    LightRAG rewrites every JSON, vector, and graph file after each document, so
    the cost of a flush grows with the index and the build slows as it fills. A
    skipped flush loses nothing that the response cache does not already hold, but
    the status store writes at once, so a process killed before the next flush
    leaves documents marked processed without their chunks or graph.
    :func:`incomplete_documents` finds those on the next start.

    Returns
    -------
    Callable
        A coroutine function that forces a flush.
    """
    original = rag._flush_storages  # noqa: SLF001
    last = [time.monotonic()]

    async def gated(storages: list[Any]) -> None:
        full = len(storages) == len(rag._index_storages())  # noqa: SLF001
        if full and time.monotonic() - last[0] < interval:
            return
        await original(storages)
        if full:
            last[0] = time.monotonic()

    async def flush_all() -> None:
        await original(rag._index_storages())  # noqa: SLF001

    rag._flush_storages = gated  # noqa: SLF001
    return flush_all


async def build_lightrag(
    chosen: Sequence[Passage], directory: Path, args: argparse.Namespace
) -> tuple[CallLedger, dict[str, int]]:
    """Index the passages with LightRAG and return its cost and document statuses."""
    from lightrag import LightRAG  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    cache = ResponseCache(work_root() / "responses-lightrag.sqlite")
    ledger = CallLedger("lightrag")
    llm = async_llm(
        AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        args.model,
        cache=cache,
        limiter=RateLimiter(per_minute=args.rpm),
        ledger=ledger,
        max_retries=16,
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Before LightRAG loads its status store, which it would write back over the
    # cleaned file.
    known = known_document_ids(directory)
    wanted = {p.passage_id for p in chosen}
    stale = sorted(set(incomplete_documents(directory)) & wanted)
    rag = LightRAG(
        working_dir=str(directory),
        llm_model_func=llm,
        llm_model_name=args.model,
        embedding_func=embedder_function(SentenceTransformerEmbedder(ENCODER)),
        chunk_token_size=LIGHTRAG_CHUNK_TOKENS,
        chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
        llm_model_max_async=args.workers,
        max_parallel_insert=args.parallel_insert or max(2, args.workers // 4),
        # A stalled embedding call must not abort a batch.
        default_embedding_timeout=600,
        # The response cache above already stores every call, so LightRAG's own copy
        # only adds a 45 MB file that it rewrites on each flush.
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    if stale:
        print(f"re-indexing {len(stale)} documents marked processed but incomplete: {stale}", flush=True)
        await forget_documents(rag, stale)
        known -= set(stale)
    flush_all = throttle_flushes(rag, FLUSH_INTERVAL_SECONDS)
    # SIGTERM cancels the build like Ctrl-C does, so the flush below still runs.
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is not None:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        fresh = [p for p in chosen if p.passage_id not in known]
        if fresh:
            ids = [p.passage_id for p in fresh]
            await rag.ainsert([passage_text(p) for p in fresh], ids=ids, file_paths=ids)
        # Re-inserting a known id would add a failed duplicate record, so retries go
        # through the pipeline, which picks up every pending and failed document.
        await rag.apipeline_process_enqueue_documents()
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        await flush_all()
        await rag.finalize_storages()
    return ledger, document_statuses(directory)


def build_hipporag2(chosen: Sequence[Passage], directory: Path, args: argparse.Namespace) -> CallLedger:
    """Index the passages with atlas-rag's HippoRAG 2 pipeline, entity-relation only."""
    from atlas_rag.kg_construction.triple_config import ProcessingConfig  # noqa: PLC0415
    from atlas_rag.kg_construction.triple_extraction import (  # noqa: PLC0415
        KnowledgeGraphExtractor,
    )
    from atlas_rag.llm_generator import LLMGenerator  # noqa: PLC0415
    from atlas_rag.llm_generator.format.validate_json_schema import (  # noqa: PLC0415
        ATLAS_SCHEMA,
    )
    from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (  # noqa: PLC0415
        TRIPLE_INSTRUCTIONS,
    )
    from atlas_rag.vectorstore import create_embeddings_and_index  # noqa: PLC0415
    from atlas_rag.vectorstore.embedding_model import SentenceEmbedding  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    data.mkdir(exist_ok=True)
    with (data / f"{HIPPORAG_KEYWORD}.jsonl").open("w", encoding="utf-8") as handle:
        for passage in chosen:
            record = {
                "id": passage.passage_id,
                "text": passage_text(passage),
                "metadata": {"lang": "en"},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # One stage only: entity-relation triples, atlas-rag's own prompt and schema.
    prompt = directory / "prompt.json"
    schema = directory / "schema.json"
    prompt.write_text(
        json.dumps(
            {
                "en": {
                    "system": TRIPLE_INSTRUCTIONS["en"]["system"],
                    "entity_relation": TRIPLE_INSTRUCTIONS["en"]["entity_relation"],
                }
            }
        )
    )
    schema.write_text(json.dumps({"entity_relation": ATLAS_SCHEMA["entity_relation"]}))

    cache = ResponseCache(work_root() / "responses-hipporag2.sqlite")
    ledger = CallLedger("hipporag2")
    client = govern(
        OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        cache=cache,
        limiter=RateLimiter(per_minute=args.rpm),
        ledger=ledger,
        max_retries=16,
    )
    generator = LLMGenerator(client=client, model_name=args.model, max_workers=args.workers)
    config = ProcessingConfig(
        model_path=args.model,
        data_directory=str(data),
        filename_pattern=HIPPORAG_KEYWORD,
        output_directory=str(directory),
        batch_size_triple=args.batch,
        max_workers=args.workers,
        include_concept=False,
        triple_extraction_prompt_path=str(prompt),
        triple_extraction_schema_path=str(schema),
    )
    extractor = KnowledgeGraphExtractor(model=generator, config=config)
    extractor.run_extraction()
    extractor.convert_json_to_csv()
    extractor.convert_to_graphml()

    encoder = SentenceEmbedding(SentenceTransformer(ENCODER, trust_remote_code=True))
    create_embeddings_and_index(
        sentence_encoder=encoder,
        model_name=ENCODER,
        working_directory=str(directory),
        keyword=HIPPORAG_KEYWORD,
        include_concept=False,
        include_events=False,
        normalize_embeddings=True,
    )
    return ledger


def cache_key(method: str, count: int, total: int, model: str = REFERENCE_EXTRACTION_MODEL) -> CacheKey:
    """Return the artifact key for a build, naming everything that changes it.

    ``model`` defaults to the reference build model, so a reader restores the
    reference index unless it names another build.
    """
    config: dict[str, Any] = {
        "method": method,
        "model": model,
        "encoder": ENCODER,
        "passages": count,
        "corpus_passages": total,
        "dataset_revision": MUSIQUE_REVISION,
    }
    if method == "lightrag":
        config.update(chunk_tokens=LIGHTRAG_CHUNK_TOKENS, overlap=LIGHTRAG_CHUNK_OVERLAP, gleaning="default")
    else:
        config.update(
            keyword=HIPPORAG_KEYWORD,
            stages="entity_relation",
            include_concept=False,
            include_events=False,
        )
    return CacheKey(ri_id=RI_ID, artifact_type="graph-index", producer_id=method, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    """Build one index, report its cost, and store it in the artifact cache."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("method", choices=("lightrag", "hipporag2"))
    parser.add_argument("--passages", type=int, default=None, help="first N passages only")
    parser.add_argument(
        "--model",
        default=None,
        help=f"build model (default: {ROLE_ENV['extraction']}, then {REFERENCE_EXTRACTION_MODEL})",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    parser.add_argument(
        "--parallel-insert",
        type=int,
        default=None,
        help="LightRAG documents processed at once (default: workers // 4)",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="HippoRAG 2 batch size")
    parser.add_argument("--no-store", action="store_true", help="skip the artifact cache")
    parser.add_argument("--out", type=Path, default=None, help="write the cost report here")
    args = parser.parse_args(argv)
    args.model = build_model(args.model)

    passages, _ = load_musique()
    chosen = select(passages, args.passages)
    suffix = f"-{len(chosen)}" if args.passages else ""
    directory = work_root() / f"{args.method}{suffix}"
    began = time.perf_counter()
    statuses: dict[str, int] | None = None
    if args.method == "lightrag":
        with exclusive(directory):
            ledger, statuses = asyncio.run(build_lightrag(chosen, directory, args))
        processed = statuses.get("processed", 0)
        if processed != len(chosen):
            raise SystemExit(
                f"LightRAG processed {processed} of {len(chosen)} passages: {statuses}. "
                "Run it again: the response cache makes the retry cheap."
            )
        problems = index_problems(directory, {p.passage_id for p in chosen})
        if problems:
            raise SystemExit(
                "LightRAG's stores disagree:\n  " + "\n  ".join(problems[:20]) + "\nRun it again: the start of "
                "the build re-indexes incomplete documents."
            )
    else:
        ledger = build_hipporag2(chosen, directory, args)
    cost = report(
        args.method,
        ledger,
        model=args.model,
        passages=len(chosen),
        seconds=time.perf_counter() - began,
        total=len(passages),
        rpm=args.rpm,
    )
    if statuses is not None:
        cost["document_statuses"] = statuses
    (directory / "index_cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps(cost, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")
    if not args.no_store:
        key = cache_key(args.method, len(chosen), len(passages), args.model)
        # Replace any stored copy: a resumed or repaired build keeps its key but
        # changes the directory.
        stored = store_directory(TieredResultCache(), key, directory, force_refresh=True)
        print(f"stored: {stored.metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
