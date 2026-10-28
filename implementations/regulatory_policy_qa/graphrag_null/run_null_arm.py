#!/usr/bin/env python3
"""Measure null-answer detection with dense and GraphRAG evidence, and controls.

    python run_null_arm.py --documents 20

RI #1's original claim was that naive GraphRAG collapses on "not stated"
questions. This runner measures that claim and replays cached evidence at new
operating points. Unpacked score-only bundles automatically forbid indexing
and retrieval; ``--score-only`` applies the same rule to any other store.

Every arm goes through the same grading, generation, and abstention path, so
the only thing that varies is where the evidence came from. The arms are dense
retrieval scoped to the contract a question belongs to, LightRAG over a
per-contract graph, a shuffled control with the same candidate pool and no
ranking signal, an unscoped dense control that prices the retrieval scope on
its own, and the always-answer and always-abstain floors.

Three phases, each skipped when its output already exists:

1. ``index``    build one LightRAG store per contract, metering what it costs.
2. ``retrieve`` produce one evidence table per arm and store it.
3. ``generate`` replay each table through the pipeline and score it.

Reasoning is disabled on every model call. Embeddings are local
``BAAI/bge-small-en-v1.5``; nothing here calls a hosted embedding API.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "aieng-advanced-rag"))
sys.path.insert(0, str(HERE))

from aieng.advanced_rag import spend  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.batch import ProxyBatch  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.corpora import cuad  # noqa: E402
from aieng.advanced_rag.corpora.contract_nli import (  # noqa: E402
    CONTRACT_NLI_LICENCE,
    build_slice,
    fetch_contract_nli,
    load_contract_nli,
)
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: E402
from aieng.advanced_rag.models import (  # noqa: E402 - use the source checkout after bootstrap
    REFERENCE_MODELS,
    choose_models,
)
from aieng.advanced_rag.pipeline.grading import (  # noqa: E402
    GradingThresholds,
    LLMGrader,
)
from aieng.advanced_rag.pipeline.spine import (  # noqa: E402
    AdvancedRagSpine,
    SpineConfig,
    evidence_from_hits,
)
from aieng.advanced_rag.provenance import Provenance  # noqa: E402
from aieng.advanced_rag.proxy import resolve_base_url  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.retrieval.corpus import Query, Slice  # noqa: E402
from aieng.advanced_rag.retrieval.embedders import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from aieng.advanced_rag.retrieval.retrievers import (  # noqa: E402
    DenseRetriever,
    ShuffledRetriever,
)
from export_store import MANIFEST_NAME, unpacked_mismatches  # noqa: E402
from null_lab import (  # noqa: E402
    DEFAULT_MODEL,
    Meter,
    MeteredCompleter,
    QueryOutcome,
    answerable_cost,
    constant_outcomes,
    evidence_covers_spans,
    load_api_key,
    load_evidence,
    null_report,
    proxy_batch,
    save_evidence,
    table_retrieve,
)


EMBED_MODEL = "BAAI/bge-small-en-v1.5"

#: Arms whose evidence comes from a retriever. The controls are derived from
#: whichever of these ran, so they always score the same question set.
RETRIEVAL_ARMS = ("dense", "lightrag-mix", "shuffled", "dense-unscoped", "no-retrieval")

#: LightRAG chunking, matched to the harness slice so the comparison is about
#: where evidence comes from rather than how wide a passage is.
LIGHTRAG_CHUNK_TOKENS = 300
LIGHTRAG_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Paths:
    """Where a run reads and writes.

    Parameters
    ----------
    store : Path
        Working directory for LightRAG stores, the corpus download, and the
        response cache. Not committed: it holds corpus text and vectors.
    out : Path
        Committed output directory.
    env : Path
        ``.env`` holding the proxy key.
    """

    store: Path
    out: Path
    env: Path
    corpus_name: str = "contract-nli"

    @property
    def corpus(self) -> Path:
        """Return the directory the pinned ContractNLI split is cached in."""
        return self.store / "corpus"

    @property
    def evidence(self) -> Path:
        """Return the directory retrieval tables are stored in."""
        return self.store / "evidence"

    @property
    def cache(self) -> Path:
        """Return the response cache file."""
        return self.store / "proxy_cache.sqlite"

    @property
    def index_report(self) -> Path:
        """Return the file holding per-contract LightRAG indexing cost."""
        return self.out / "lightrag_index_cost.json"


def connection(paths: Paths, rpm: int) -> dict[str, Any]:
    """Return the proxy settings for a live batch, starting the spend session."""
    spend.session()
    return {
        "rpm": rpm,
        "timeout": 600.0,
        "api_key": load_api_key(paths.env),
        "base_url": resolve_base_url(paths.env),
    }


def lightrag_key(model: str) -> Any:
    """Return the cache key function the existing LightRAG stores were built with."""

    def key(messages: list[dict[str, Any]], params: Mapping[str, Any]) -> str:
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": params["max_tokens"],
            "temperature": 0.0,
            "thinking": False,
        }
        digest: str = ResponseCache.key(request)
        return digest

    return key


def lightrag_batch(paths: Paths, rpm: int) -> ProxyBatch:
    """Return the proxy connection LightRAG's model function sends through.

    The key and endpoint are read when the first call the cache cannot answer
    is sent, so a fully cached run needs neither.
    """
    return ProxyBatch(
        DEFAULT_MODEL,
        cache=ResponseCache(paths.cache),
        key=lightrag_key(DEFAULT_MODEL),
        payload="text",
        **connection(paths, rpm),
    )


def corpus_slice(paths: Paths, documents: int) -> tuple[Slice, dict[str, list[str]]]:
    """Fetch and chunk the selected pinned contract test split.

    Documents are taken in file order, which is stable across runs, so the
    slice is not selected on anything the measurement is about.

    Parameters
    ----------
    paths : Paths
        Where to cache the download.
    documents : int
        Contracts to take.

    Returns
    -------
    tuple
        The prepared slice, and the gold span texts keyed by query identifier.
    """
    if paths.corpus_name == "cuad":
        path = cuad.fetch_cuad_test(dest=paths.corpus)
        prepared = cuad.build_slice(path, max_documents=documents)
    else:
        path = fetch_contract_nli(dest=paths.corpus)
        prepared = build_slice(path, max_documents=documents)
    spans = {query.query_id: [answer.text for answer in query.answers] for query in prepared.queries}
    return prepared, spans


def document_texts(paths: Paths, documents: int) -> list[tuple[str, str]]:
    """Return ``(doc_id, text)`` for the first ``documents`` contracts."""
    if paths.corpus_name == "cuad":
        path = cuad.fetch_cuad_test(dest=paths.corpus)
        return [(doc_id, text) for doc_id, text, _ in list(cuad.load_cuad(path))[:documents]]
    path = fetch_contract_nli(dest=paths.corpus)
    texts: list[tuple[str, str]] = []
    for count, (doc_id, text, _) in enumerate(load_contract_nli(path)):
        if count >= documents:
            break
        texts.append((doc_id, text))
    return texts


def build_lightrag(
    working_dir: Path,
    workspace: str,
    llm: Any,
    embedder: SentenceTransformerEmbedder,
    max_async: int,
) -> Any:
    """Construct a LightRAG instance over a local encoder and the proxy model.

    ``workspace`` is not decoration. LightRAG keeps its default storages in
    process-global state, so two instances built in one process share
    everything unless their workspaces differ, and a per-contract graph would
    silently accumulate every contract indexed before it.
    """
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: Sequence[str]) -> Any:
        return np.array(embedder.encode(list(texts)), dtype=np.float32)

    return LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        llm_model_func=llm,
        llm_model_name=DEFAULT_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=embedder.dimension(), max_token_size=512, func=embed),
        chunk_token_size=LIGHTRAG_CHUNK_TOKENS,
        chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
        llm_model_max_async=max_async,
    )


def async_completer(batch: ProxyBatch, meter: Meter) -> Any:
    """Return the metered model function LightRAG calls.

    LightRAG builds its own requests, so the accounting happens here and
    anything the library chooses to do inside its own code path is counted.
    ``batch``, from :func:`lightrag_batch`, answers from the response cache
    first, switches reasoning off, and retries transient errors with its full
    budget, because LightRAG treats one failed extraction call as a failed
    document: it writes no graph, marks the document ``failed`` in its status
    store, and carries on.
    """

    async def llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        for drop in ("hashing_kv", "keyword_extraction", "mode", "enable_cot"):
            kwargs.pop(drop, None)
        max_tokens = int(kwargs.pop("max_tokens", 4000))

        outcome = await batch.acreate(messages, max_tokens=max_tokens, temperature=0.0)
        text = str(outcome.text or "")
        if outcome.cached:
            meter.record_cache_hit()
            return text
        seconds = outcome.metrics.service_time_s if outcome.metrics is not None else 0.0
        meter.record(outcome.usage.prompt_tokens, outcome.usage.completion_tokens, seconds)
        if not text:
            meter.record_empty()
        return text

    return llm


def document_status(working_dir: Path, workspace: str) -> str:
    """Return LightRAG's own verdict on whether a document indexed.

    A failed extraction leaves a store that opens cleanly and retrieves
    nothing, so the status file is the only place the failure is visible. An
    arm reported from an empty graph would look like a GraphRAG result and
    would be a bug.
    """
    status_file = working_dir / workspace / "kv_store_doc_status.json"
    if not status_file.exists():
        return "missing"
    records = json.loads(status_file.read_text(encoding="utf-8"))
    # Duplicate-attempt receipts describe an insert attempt, not the source graph.
    # Ignore only flagged receipts that point to an existing original record.
    primary = {
        key: row
        for key, row in records.items()
        if not (key.startswith("dup-") and row.get("metadata", {}).get("is_duplicate") is True)
    }
    for key, row in records.items():
        if key not in primary and row.get("metadata", {}).get("original_doc_id") not in primary:
            return "orphan-duplicate"
    states = {str(record.get("status", "unknown")) for record in primary.values()}
    if not states:
        return "empty"
    return "processed" if states == {"processed"} else "/".join(sorted(states))


def require_processed_index(paths: Paths, doc_id: str) -> None:
    """Require a processed graph store before fresh retrieval."""
    state = document_status(paths.store / "lightrag" / doc_id, f"cnli-{doc_id}")
    if state != "processed":
        raise RuntimeError(f"LightRAG index for {doc_id} is {state}; a processed store is required for fresh retrieval")


async def index_documents(
    paths: Paths, documents: int, max_async: int, force: bool, batch: ProxyBatch
) -> dict[str, Any]:
    """Build one LightRAG store per contract and report what indexing cost.

    Parameters
    ----------
    paths : Paths
        Where the stores live.
    documents : int
        Contracts to index.
    max_async : int
        LightRAG's concurrent model calls.
    force : bool
        Rebuild stores that already exist.
    batch : ProxyBatch
        The proxy connection for the indexing model.

    Returns
    -------
    dict
        Per-contract and total indexing cost.

    Raises
    ------
    RuntimeError
        If LightRAG marked any contract as anything but processed.
    """
    if paths.index_report.exists() and not force:
        loaded: dict[str, Any] = json.loads(paths.index_report.read_text(encoding="utf-8"))
        expected = {doc_id for doc_id, _ in document_texts(paths, documents)}
        records = loaded.get("per_document", {})
        if (
            loaded.get("documents") != documents
            or set(records) != expected
            or any(row.get("status") != "processed" for row in records.values())
        ):
            raise RuntimeError("cached LightRAG index report is incomplete or unhealthy; rebuild explicitly")
        for doc_id in expected:
            require_processed_index(paths, doc_id)
        print(f"index: reusing {paths.index_report}", file=sys.stderr)
        return loaded

    from lightrag.kg.shared_storage import initialize_pipeline_status

    texts = document_texts(paths, documents)
    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    meter = Meter()
    llm = async_completer(batch, meter)

    per_document: dict[str, Any] = {}
    began = time.perf_counter()
    for position, (doc_id, text) in enumerate(texts, start=1):
        workspace = f"cnli-{doc_id}"
        working_dir = paths.store / "lightrag" / doc_id
        if force and working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        rag = build_lightrag(working_dir, workspace, llm, embedder, max_async)
        await rag.initialize_storages()
        await initialize_pipeline_status(workspace=workspace)
        meter.phase = f"index:{doc_id}"
        started = time.perf_counter()
        await rag.ainsert([text], file_paths=[doc_id])
        elapsed = time.perf_counter() - started
        await rag.finalize_storages()

        totals: dict[str, Any] = dict(meter.phase_totals(meter.phase))
        totals["doc_chars"] = len(text)
        totals["wall_seconds"] = round(elapsed, 1)
        totals["status"] = document_status(working_dir, workspace)
        per_document[doc_id] = totals
        print(
            f"index {position}/{len(texts)} doc {doc_id}: "
            f"{totals['calls']} calls, {totals['total_tokens']} tokens, "
            f"{totals['wall_seconds']}s, {totals['status']}",
            file=sys.stderr,
            flush=True,
        )

    report = {
        "documents": len(texts),
        "doc_chars": sum(len(text) for _, text in texts),
        "chunk_token_size": LIGHTRAG_CHUNK_TOKENS,
        "chunk_overlap_token_size": LIGHTRAG_CHUNK_OVERLAP,
        "llm_model_max_async": max_async,
        "wall_seconds": round(time.perf_counter() - began, 1),
        "totals": {
            "calls": sum(int(v["calls"]) for v in per_document.values()),
            "prompt_tokens": sum(int(v["prompt_tokens"]) for v in per_document.values()),
            "completion_tokens": sum(int(v["completion_tokens"]) for v in per_document.values()),
            "total_tokens": sum(int(v["total_tokens"]) for v in per_document.values()),
            "cache_hits": sum(int(v["cache_hits"]) for v in per_document.values()),
        },
        "per_document": per_document,
    }
    paths.index_report.parent.mkdir(parents=True, exist_ok=True)
    paths.index_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unhealthy = {
        doc_id: str(totals["status"]) for doc_id, totals in per_document.items() if totals["status"] != "processed"
    }
    if unhealthy:
        raise RuntimeError(f"LightRAG did not index every contract: {unhealthy}")
    return report


async def retrieve_lightrag(  # noqa: PLR0917 - mirrors the retrieval run configuration
    paths: Paths,
    prepared: Slice,
    documents: int,
    max_async: int,
    top_k: int,
    batch: ProxyBatch,
) -> dict[str, Any]:
    """Retrieve with LightRAG for every question and store the evidence table.

    Returns
    -------
    dict
        Query cost for the arm.
    """
    for doc_id in {query.doc_id for query in prepared.queries}:
        require_processed_index(paths, doc_id)

    from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever
    from lightrag.kg.shared_storage import initialize_pipeline_status

    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    meter = Meter(phase="query")
    llm = async_completer(batch, meter)

    by_doc: dict[str, list[Query]] = {}
    for query in prepared.queries:
        by_doc.setdefault(query.doc_id, []).append(query)

    table: dict[str, list[Evidence]] = {}
    began = time.perf_counter()
    for position, (doc_id, queries) in enumerate(sorted(by_doc.items()), start=1):
        workspace = f"cnli-{doc_id}"
        rag = build_lightrag(paths.store / "lightrag" / doc_id, workspace, llm, embedder, max_async)
        await rag.initialize_storages()
        await initialize_pipeline_status(workspace=workspace)
        retriever = LightRAGRetriever(rag, mode="mix")
        gate = asyncio.Semaphore(max_async)

        async def one(query: Query, retriever: Any = retriever, gate: Any = gate) -> None:
            async with gate:
                table[query.query_id] = restore_graph_scope(
                    await retriever.aretrieve(query.question, top_k), query.doc_id
                )

        await asyncio.gather(*(one(query) for query in queries))
        await rag.finalize_storages()
        print(
            f"retrieve {position}/{len(by_doc)} doc {doc_id}: {len(queries)} questions",
            file=sys.stderr,
            flush=True,
        )

    save_evidence(paths.evidence / "lightrag-mix.json", table)
    cost: dict[str, Any] = dict(meter.phase_totals("query"))
    cost["questions"] = len(prepared.queries)
    cost["wall_seconds"] = round(time.perf_counter() - began, 1)
    cost["documents"] = documents
    cost["evidence_sha256"] = hashlib.sha256((paths.evidence / "lightrag-mix.json").read_bytes()).hexdigest()
    write_checkpoint(paths.out / "lightrag_query_cost.json", cost)
    return cost


def restore_graph_scope(evidence: list[Evidence], doc_id: str) -> list[Evidence]:
    """Restore whitespace stripped by LightRAG within a per-document workspace.

    A different source identifier is an error, never a reason to relabel evidence.
    """
    restored = []
    for item in evidence:
        if item.doc_id not in {doc_id, doc_id.strip()}:
            raise ValueError(f"Graph evidence violates contract scope for {doc_id}")
        restored.append(replace(item, provenance=replace(item.provenance, doc_id=doc_id)))
    return restored


def write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete cost checkpoint atomically before scoring starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cached_graph_costs(paths: Paths, prepared: Slice) -> dict[str, Any]:
    """Retain recorded graph costs only when they match this cached population."""
    costs = {}
    expected = {query.doc_id for query in prepared.queries}
    if paths.index_report.exists():
        report = json.loads(paths.index_report.read_text())
        if report.get("documents") == len(expected) and set(report.get("per_document", {})) == expected:
            costs["lightrag_index"] = report
    query_report = paths.out / "lightrag_query_cost.json"
    evidence = paths.evidence / "lightrag-mix.json"
    if query_report.exists() and evidence.exists():
        report = json.loads(query_report.read_text())
        fingerprint = hashlib.sha256(evidence.read_bytes()).hexdigest()
        if (
            report.get("documents") == len(expected)
            and report.get("questions") == len(prepared.queries)
            and report.get("evidence_sha256") == fingerprint
        ):
            costs["lightrag_query"] = report
    return costs


def retrieve_dense(paths: Paths, prepared: Slice, top_k: int) -> None:
    """Retrieve with the harness dense arm, scoped to each question's contract."""
    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    retriever = DenseRetriever(embedder)
    retriever.index(prepared.chunks)
    by_id = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    table = {
        query.query_id: evidence_from_hits(retriever.search(query, k=top_k), by_id, "dense")
        for query in prepared.queries
    }
    save_evidence(paths.evidence / "dense.json", table)


def retrieve_dense_unscoped(paths: Paths, prepared: Slice, top_k: int) -> None:
    """Retrieve densely over every contract at once, ignoring the question's scope.

    The scope control. ContractNLI asks the same 17 hypotheses of every
    contract, so a retriever that is not told which contract a question belongs
    to can answer it from a different one. That is a property of the retrieval
    scope and not of the retrieval method, and without this row a GraphRAG arm
    that loses on scope would be read as losing on graphs.
    """
    from aieng.advanced_rag.retrieval.stores import QdrantStore

    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    chunks = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    store = QdrantStore(collection="dense-unscoped")
    store.add(list(chunks), embedder.encode([chunk.indexed_text for chunk in chunks.values()]))
    questions = [query.question for query in prepared.queries]
    ranked = store.search_batch(embedder.encode_queries(questions), top_k)

    table: dict[str, list[Evidence]] = {}
    for query, hits in zip(prepared.queries, ranked):
        table[query.query_id] = [
            Evidence(
                text=chunks[chunk_id].text,
                provenance=Provenance.from_chunk(chunks[chunk_id]),
                score=score,
                retriever="dense-unscoped",
            )
            for chunk_id, score in hits
        ]
    save_evidence(paths.evidence / "dense-unscoped.json", table)


def retrieve_shuffled(paths: Paths, prepared: Slice, top_k: int) -> None:
    """Retrieve the right contract's chunks in a fixed random order."""
    retriever = ShuffledRetriever()
    retriever.index(prepared.chunks)
    by_id = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    table = {
        query.query_id: evidence_from_hits(retriever.search(query, k=top_k), by_id, "shuffled")
        for query in prepared.queries
    }
    save_evidence(paths.evidence / "shuffled.json", table)


def retrieve_none(paths: Paths, prepared: Slice) -> None:
    """Retrieve nothing, so the pipeline abstains without a model call."""
    save_evidence(
        paths.evidence / "no-retrieval.json",
        {query.query_id: [] for query in prepared.queries},
    )


def run_arm(  # noqa: PLR0917 - keeps one shared config and completer across arms
    name: str,
    prepared: Slice,
    spans: dict[str, list[str]],
    table: dict[str, list[Evidence]],
    completer: MeteredCompleter,
    config: SpineConfig,
    workers: int,
    *,
    grader_completer: MeteredCompleter | None = None,
) -> list[QueryOutcome]:
    """Replay one arm's evidence through the pipeline and record every outcome.

    The grader, the generator, the thresholds, and the abstention paths are the
    same object for every arm. Only ``table`` differs.

    Parameters
    ----------
    name : str
        Arm label.
    prepared : Slice
        The corpus slice.
    spans : dict
        Gold span texts keyed by query identifier.
    table : dict
        Query identifier to ranked evidence.
    completer : MeteredCompleter
        The metered model callable, shared with the grader.
    config : SpineConfig
        The pipeline's shipped defaults.
    workers : int
        Concurrent questions.

    Returns
    -------
    list of QueryOutcome
        One per question, in corpus order.
    """
    query_ids = {(query.doc_id, query.question): query.query_id for query in prepared.queries}
    spine = AdvancedRagSpine(
        retrieve=table_retrieve(table, query_ids),
        grader=LLMGrader(grader_completer or completer, thresholds=config.thresholds),
        complete=completer,
        config=config,
    )

    def answer(query: Query) -> QueryOutcome:
        result = spine.answer(query.query_id, query.question, query.doc_id)
        shown = list(table.get(query.query_id, ()))[: config.evidence_budget]
        final = result.attempts[-1] if result.attempts else None
        return QueryOutcome(
            query_id=query.query_id,
            doc_id=query.doc_id,
            is_null=not query.is_answerable,
            abstained=result.abstained,
            reason=result.answer.reason,
            verdict=str(final.verdict) if final else "not graded",
            score=float(final.score) if final else 0.0,
            retrieved=len(table.get(query.query_id, ())),
            evidence_chars=sum(len(item.text) for item in shown),
            span_reached=visible_span_reached(shown, spans.get(query.query_id, ())),
            answer=result.answer.text,
        )

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = []
        for outcome in pool.map(answer, prepared.queries):
            outcomes.append(outcome)
            if len(outcomes) % 25 == 0 or len(outcomes) == len(prepared.queries):
                print(f"arm {name}: {len(outcomes)}/{len(prepared.queries)} completed", file=sys.stderr, flush=True)
    print(
        f"arm {name}: {len(outcomes)} questions in {time.perf_counter() - began:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    return outcomes


def visible_span_reached(evidence: Sequence[Evidence], spans: Sequence[str]) -> bool:
    """Count a gold span only when it fits in the generator's passage budget."""
    return bool(evidence_covers_spans([replace(item, text=item.text[:1200]) for item in evidence], spans))


def summarise(
    name: str,
    outcomes: Sequence[QueryOutcome],
    qrels_hit: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Return one arm's row: null detection beside what it cost on answerables.

    ``qrels_hit`` is the harness judgment set's own verdict on the same
    passages, available only where an arm's chunk identifiers are the harness's.
    Where it is, the agreement with the span-containment rule is reported, which
    is what licenses using that rule on the arm where identifiers do not carry
    across.
    """
    row = {
        "arm": name,
        "queries": len(outcomes),
        "null_detection": null_report(outcomes).to_dict(),
        "answerable": answerable_cost(outcomes).to_dict(),
        "abstention_reasons": _reason_counts(outcomes),
    }
    if qrels_hit is not None:
        row["span_rule_check"] = _span_rule_check(outcomes, qrels_hit)
    return row


def _span_rule_check(outcomes: Sequence[QueryOutcome], qrels_hit: Mapping[str, bool]) -> dict[str, Any]:
    """Compare span containment with the identifier-based judgment set."""
    answerable = [outcome for outcome in outcomes if not outcome.is_null]
    if not answerable:
        return {"answerable_queries": 0, "agreement": 0.0}
    agree = sum(1 for outcome in answerable if outcome.span_reached == qrels_hit.get(outcome.query_id, False))
    return {
        "answerable_queries": len(answerable),
        "qrels_hit_rate": round(
            sum(1 for o in answerable if qrels_hit.get(o.query_id, False)) / len(answerable),
            4,
        ),
        "span_reach_rate": round(sum(1 for o in answerable if o.span_reached) / len(answerable), 4),
        "agreement": round(agree / len(answerable), 4),
    }


def qrels_hits(prepared: Slice, table: Mapping[str, Sequence[Evidence]], budget: int) -> dict[str, bool]:
    """Return whether each question's top-budget passages hold a judged chunk."""
    hits: dict[str, bool] = {}
    for query in prepared.queries:
        ranked = {
            item.provenance.chunk_id
            for item in list(table.get(query.query_id, ()))[:budget]
            if item.provenance.chunk_id
        }
        hits[query.query_id] = bool(ranked & prepared.qrels.get(query.query_id, set()))
    return hits


def _reason_counts(outcomes: Sequence[QueryOutcome]) -> dict[str, int]:
    """Return how many abstentions each path produced.

    Which of the pipeline's abstention paths fired is the difference between a
    system that read the evidence and declined and one that never generated at
    all, and the two are not the same result.
    """
    counts: dict[str, int] = {}
    for outcome in outcomes:
        if not outcome.abstained:
            continue
        reason = outcome.reason.split(":")[0].strip() or "unstated"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def write_outputs(  # noqa: PLR0917 - records the complete measured run
    paths: Paths,
    prepared: Slice,
    arms: dict[str, list[QueryOutcome]],
    summaries: list[dict[str, Any]],
    costs: dict[str, Any],
    config: SpineConfig,
    model: str = DEFAULT_MODEL,
    model_roles: dict[str, Any] | None = None,
) -> None:
    """Write the per-question records and the summary every memo number cites."""
    paths.out.mkdir(parents=True, exist_ok=True)
    for name, outcomes in arms.items():
        (paths.out / f"outcomes_{name}.json").write_text(
            json.dumps([outcome.to_dict() for outcome in outcomes], indent=1) + "\n",
            encoding="utf-8",
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        **({"model_roles": model_roles} if model_roles is not None else {}),
        "embedding_model": EMBED_MODEL,
        "reasoning": "disabled on every call",
        "corpus": {
            "name": "CUAD test split" if paths.corpus_name == "cuad" else "ContractNLI test split",
            "licence": (cuad.CUAD_LICENCE if paths.corpus_name == "cuad" else CONTRACT_NLI_LICENCE).name,
            "attribution": (cuad.CUAD_LICENCE if paths.corpus_name == "cuad" else CONTRACT_NLI_LICENCE).attribution,
            **prepared.summary(),
        },
        "pipeline": {
            "top_k": config.top_k,
            "evidence_budget": config.evidence_budget,
            "max_attempts": config.max_attempts,
            "require_citations": config.require_citations,
            "sufficient": config.thresholds.sufficient,
            "insufficient": config.thresholds.insufficient,
            "grader": "LLMGrader",
            "rewriter": None,
            "thresholds_note": (
                "measured on ContractNLI; the sweep is in THRESHOLDS.md. insufficient ships at "
                "0.0 because 0.3 caught 20 more of 322 null questions and refused 43 more of 358 "
                "answerable ones. sufficient was flat from 0.2 to 1.0 within sampling error and "
                "stays at 0.6"
            ),
        },
        "cost": costs,
        "arms": summaries,
    }
    (paths.out / "null_arm_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line."""
    default_store = Path.home() / ".cache" / "advanced-rag" / "graphrag-null"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--corpus", choices=("contract-nli", "cuad"), default="contract-nli")
    parser.add_argument("--store", type=Path, default=default_store)
    parser.add_argument("--out", type=Path, help="corpus-specific output directory")
    parser.add_argument("--env", type=Path, default=REPO / ".env")
    parser.add_argument("--model", help="Override the scoring model; offline replay defaults to the reference model")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="score from cached evidence and cached responses only; set by a score-only MANIFEST.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-async", type=int, default=6)
    parser.add_argument(
        "--rpm",
        type=int,
        default=90,
        help="request budget per minute, under the proxy's 120 so a second job fits",
    )
    parser.add_argument("--arms", nargs="+", choices=RETRIEVAL_ARMS, default=list(RETRIEVAL_ARMS))
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-retrieve", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--cache-only", action="store_true", help="score cached evidence and responses without credentials or API calls"
    )
    args = parser.parse_args(argv)
    if args.out is None:
        if args.corpus == "cuad":
            relative = "outputs/cuad_null"
        elif args.documents == 123:
            relative = (
                "outputs/full_contract_nli/dense" if args.arms == ["dense"] else "outputs/full_contract_nli/null_arms"
            )
        else:
            relative = "outputs/graphrag_null/measured_thresholds"
        args.out = HERE.parent / relative
    return args


def validate_table(
    table: Mapping[str, Sequence[Evidence]],
    queries: Sequence[Query],
    arm: str,
    documents: Collection[str] | None = None,
) -> None:
    """Reject incomplete, malformed, or incorrectly scoped caches before scoring them.

    ``documents`` is the slice's document set, which unscoped arms may draw
    from. It defaults to the documents the queries name.
    """
    if set(table) != {q.query_id for q in queries}:
        raise ValueError(f"{arm}: cached evidence IDs do not match the requested corpus slice")
    if arm == "no-retrieval" and any(table.values()):
        raise ValueError("no-retrieval: the control table must hold no evidence")
    documents = set(documents) if documents is not None else {q.doc_id for q in queries}
    for query in queries:
        for item in table[query.query_id]:
            if not isinstance(item, Evidence) or not item.text.strip():
                raise ValueError(f"{arm}: malformed or empty evidence for {query.query_id}")
            if arm in ("dense", "lightrag-mix") and item.provenance.doc_id != query.doc_id:
                raise ValueError(f"{arm}: cached evidence violates contract scope for {query.query_id}")
            if item.provenance.doc_id not in documents:
                raise ValueError(f"{arm}: cached evidence cites a document outside the slice for {query.query_id}")


def estimate(paths: Paths, documents: int) -> None:
    """Print the LightRAG indexing bill before spending it.

    Measured at 5 documents, LightRAG's wall time grew faster than linearly in
    the size of one graph. Per-contract graphs
    keep each one small, so a per-character projection from a measured contract
    is the right shape, but it is a projection and is printed as one.
    """
    texts = document_texts(paths, documents)
    chars = sum(len(text) for _, text in texts)
    # Measured on ContractNLI document 1: 16,632 characters produced 13 chunks,
    # 30 model calls, and 91,542 tokens.
    per_char_calls = 30 / 16_632
    per_char_tokens = 91_542 / 16_632
    print(
        json.dumps(
            {
                "documents": len(texts),
                "corpus_chars": chars,
                "projected_index_calls": round(chars * per_char_calls),
                "projected_index_tokens": round(chars * per_char_tokens),
                "basis": "ContractNLI document 1, measured",
            },
            indent=2,
        )
    )


def preflight(args: argparse.Namespace, paths: Paths) -> None:
    """Reject unsafe score-only combinations before any indexing or model access.

    A store holding ``MANIFEST.json`` is an unpacked bundle. Its files are
    checked against the manifest, and a ``score-only`` bundle turns on
    ``--score-only``. An unpacked bundle never falls back to live indexing or
    retrieval for a missing table unless a ``--force`` flag asks for it.
    """
    if args.documents < 1 or args.workers < 1 or args.rpm < 1:
        raise ValueError("documents, workers, and rpm must be positive")
    manifest_path = paths.store / MANIFEST_NAME
    bundled = manifest_path.is_file()
    if bundled:
        altered = unpacked_mismatches(paths.store)
        if altered:
            raise ValueError(f"unpacked store does not match {MANIFEST_NAME}: {', '.join(altered)}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle") == "score-only":
            args.score_only = True
    missing = [name for name in args.arms if not (paths.evidence / f"{name}.json").is_file()]
    if args.score_only:
        if args.force_index or args.force_retrieve:
            raise ValueError("--score-only cannot be combined with --force-index or --force-retrieve")
        if missing:
            raise FileNotFoundError(f"--score-only requires cached evidence for: {', '.join(missing)}")
        if not paths.cache.is_file():
            raise FileNotFoundError(f"--score-only requires the response cache at {paths.cache}")
    elif bundled and missing and not (args.force_index or args.force_retrieve):
        raise FileNotFoundError(
            f"unpacked store has no evidence for {', '.join(missing)}; pass --force-retrieve to rebuild it"
        )


def scoring_completer(
    paths: Paths,
    args: argparse.Namespace,
    batches: dict[str, ProxyBatch],
    meter: Meter,
    *,
    model: str | None = None,
) -> MeteredCompleter:
    """Construct scoring I/O only after preflight, honoring offline mode.

    ``batches`` holds one proxy connection per model, so callers on the same
    model share its rate limit.
    """
    model = model or args.model
    offline = args.score_only or args.cache_only
    if model not in batches:
        batches[model] = proxy_batch(
            model,
            ResponseCache(paths.cache),
            cache_only=offline,
            **({} if offline else connection(paths, args.rpm)),
        )
    return MeteredCompleter(batches[model], meter=meter)


def cached_tables(paths: Paths, args: argparse.Namespace, prepared: Slice) -> dict[str, Any]:
    """Validate all score-only evidence before client construction."""
    return validated_tables(paths, prepared, args.arms) if args.score_only else {}


def validated_tables(paths: Paths, prepared: Slice, arms: Sequence[str]) -> dict[str, Any]:
    """Validate all populations and scopes before scoring any arm."""
    tables = {}
    for arm in arms:
        tables[arm] = load_evidence(paths.evidence / f"{arm}.json")
        validate_table(tables[arm], prepared.queries, arm, {chunk.doc_id for chunk in prepared.chunks})
    return tables


def retrieve_local_arms(paths: Paths, prepared: Slice, args: argparse.Namespace, top_k: int) -> None:
    """Fill only requested missing local tables without rebuilding graph indexes."""
    retrievers: tuple[tuple[str, Callable[[], None]], ...] = (
        ("dense", lambda: retrieve_dense(paths, prepared, top_k)),
        ("dense-unscoped", lambda: retrieve_dense_unscoped(paths, prepared, top_k)),
        ("shuffled", lambda: retrieve_shuffled(paths, prepared, top_k)),
        ("no-retrieval", lambda: retrieve_none(paths, prepared)),
    )
    for name, retrieve in retrievers:
        if name in args.arms and (args.force_retrieve or not (paths.evidence / f"{name}.json").exists()):
            retrieve()


def build_evidence(
    args: argparse.Namespace, paths: Paths, prepared: Slice, top_k: int, batches: dict[str, ProxyBatch]
) -> dict[str, Any]:
    """Index and retrieve whatever the store lacks, and return what that cost."""
    costs: dict[str, Any] = cached_graph_costs(paths, prepared) if "lightrag-mix" in args.arms else {}
    # Indexing exists to make retrieval possible, and retrieval's output is the
    # evidence table. A store that already carries a table for every requested
    # arm therefore needs no index. The check reads the store, because
    # `index_report` lives under `--out`, and pointing `--out` somewhere new
    # re-indexed 20 contracts, about 1.18M proxy tokens, from a full store.
    tables = {name: paths.evidence / f"{name}.json" for name in args.arms}
    scorable = [name for name, table in tables.items() if table.exists()]
    if scorable:
        print(
            f"evidence: reusing {len(scorable)} of {len(tables)} arms from "
            f"{paths.evidence} ({', '.join(sorted(scorable))})",
            file=sys.stderr,
        )

    def graph_batch() -> ProxyBatch:
        if "lightrag" not in batches:
            batches["lightrag"] = lightrag_batch(paths, args.rpm)
        return batches["lightrag"]

    if "lightrag-mix" in args.arms and (args.force_index or not tables["lightrag-mix"].exists()):
        costs["lightrag_index"] = asyncio.run(
            index_documents(paths, args.documents, args.max_async, args.force_index, graph_batch())
        )
    elif "lightrag-mix" in args.arms:
        print("index: skipped, graph evidence is already cached", file=sys.stderr)

    retrieve_local_arms(paths, prepared, args, top_k)
    if "lightrag-mix" in args.arms and (args.force_retrieve or not (paths.evidence / "lightrag-mix.json").exists()):
        costs["lightrag_query"] = asyncio.run(
            retrieve_lightrag(paths, prepared, args.documents, args.max_async, top_k, graph_batch())
        )
    return costs


def main(argv: Sequence[str] | None = None) -> int:
    """Run the phases and write the report."""
    args = parse_args(argv)
    paths = Paths(store=args.store, out=args.out, env=args.env, corpus_name=args.corpus)
    preflight(args, paths)
    if args.cache_only and not args.score_only:
        raise ValueError("--cache-only requires --score-only; indexing and retrieval must be disabled")
    paths.store.mkdir(parents=True, exist_ok=True)

    if args.estimate_only:
        estimate(paths, args.documents)
        return 0

    models = (
        choose_models(generation=args.model, judging=args.model, extraction=args.model)
        if args.model
        else REFERENCE_MODELS
        if args.score_only or args.cache_only
        else choose_models()
    )
    args.model = models.generation.model
    prepared, spans = corpus_slice(paths, args.documents)
    slice_documents = {chunk.doc_id for chunk in prepared.chunks}
    loaded_tables = cached_tables(paths, args, prepared)
    print(json.dumps(prepared.summary(), indent=2), file=sys.stderr)
    config = SpineConfig(thresholds=GradingThresholds())
    batches: dict[str, ProxyBatch] = {}
    costs: dict[str, Any] = cached_graph_costs(paths, prepared) if "lightrag-mix" in args.arms else {}

    # A score-only run reads every table from the store and answers only from
    # the response cache, so it skips indexing and retrieval.
    if not args.score_only:
        costs.update(build_evidence(args, paths, prepared, config.top_k, batches))

    loaded_tables = loaded_tables or validated_tables(paths, prepared, args.arms)
    meter = Meter()
    completer = scoring_completer(paths, args, batches, meter)
    grader = scoring_completer(paths, args, batches, meter, model=models.judging.model)
    arms: dict[str, list[QueryOutcome]] = {}
    summaries: list[dict[str, Any]] = []
    for name in args.arms:
        table = loaded_tables[name]
        validate_table(table, prepared.queries, name, slice_documents)
        meter.phase = f"pipeline:{name}"
        began = time.perf_counter()
        arms[name] = run_arm(name, prepared, spans, table, completer, config, args.workers, grader_completer=grader)
        # LightRAG chunks a document its own way, so only the harness-chunked
        # arms can be checked against the identifier-based judgment set.
        checkable = name != "lightrag-mix"
        row = summarise(
            name,
            arms[name],
            qrels_hits(prepared, table, config.evidence_budget) if checkable else None,
        )
        row["cost"] = dict(meter.phase_totals(meter.phase))
        row["cost"]["wall_seconds"] = round(time.perf_counter() - began, 1)
        summaries.append(row)

    reference = arms[args.arms[0]]
    for control, flag in (("always-abstain", True), ("always-answer", False)):
        arms[control] = constant_outcomes(reference, flag)
        summaries.append(summarise(control, arms[control]))

    costs["pipeline"] = meter.to_dict()
    write_outputs(
        paths,
        prepared,
        arms,
        summaries,
        costs,
        config,
        model=args.model,
        model_roles=None if args.cache_only or args.score_only else models.as_dict(),
    )
    if not (args.cache_only or args.score_only):
        print(spend.report(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
