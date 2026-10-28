#!/usr/bin/env python3
"""Build the ServerFault knowledge graph by direct extraction.

Reads the 10,000-question slice of the pinned Stack Exchange dump, extracts
entities and relations from each question and its accepted answer with one call
per section, enforces the seven-and-seven schema after the call, and cites every
fact to a document, a section, a block, and a character span. The graph is saved
as ``graph.json`` and ``graph.graphml`` and stored in the artifact cache.

The model calls go through :class:`aieng.advanced_rag.batch.ProxyBatch`, so they
are sent as one infermesh batch held to a request budget, cached as each one
settles so a run resumes, and counted. A run whose every call is cached makes no
call and needs no key. The graph is checked against
its own rules before it is stored, and a graph that breaks one is an error.

The build model comes from ``--model``, then ``ADVANCED_RAG_EXTRACTION_MODEL``,
then the deprecated ``LLM_MODEL``, then :data:`REFERENCE_EXTRACTION_MODEL`.
:func:`cache_key` defaults to the reference model, so the RI #4 notebook loads
the reference graph whatever model answers.

Examples
--------
Measure a sample and project the full run, then build everything::

    $ python build_serverfault_graph.py --documents 50
    $ python build_serverfault_graph.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.batch import ProxyBatch
from aieng.advanced_rag.cache import CacheKey, TieredResultCache, store_directory
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora import serverfault
from aieng.advanced_rag.graphrag import extraction
from aieng.advanced_rag.graphrag.proxy import CallLedger, ResponseCache
from aieng.advanced_rag.models import ROLE_ENV, endpoint_of, request_params
from aieng.advanced_rag.proxy import KEY_NAMES, MissingProxyKeyError, resolve_api_key


# Unset, the endpoint comes from OPENAI_BASE_URL, then the repository .env, then
# the Vector Institute proxy. Set LLM_BASE_URL to use another OpenAI-compatible
# endpoint.
BASE_URL = os.environ.get("LLM_BASE_URL") or None
#: The model the reference graph was built with. Its request limit is separate
#: from Qwen3_6-35B-A3B's, so the build ran alongside the MuSiQue index builds.
REFERENCE_EXTRACTION_MODEL = "Qwen3_8-27B"
#: Read for one release, with a DeprecationWarning naming the replacement.
DEPRECATED_MODEL_ENV = "LLM_MODEL"
DEFAULT_RPM = 50
DEFAULT_WORKERS = 48
MAX_TOKENS = 2000
RI_ID = "ri4"


def build_model(cli: str | None = None) -> str:
    """Return the build model: the flag, the environment, or the reference model."""
    if cli:
        return cli
    chosen = os.environ.get(ROLE_ENV["extraction"], "").strip()
    if chosen:
        return chosen
    legacy = os.environ.get(DEPRECATED_MODEL_ENV, "").strip()
    if legacy:
        warnings.warn(
            f"{DEPRECATED_MODEL_ENV} is deprecated: set {ROLE_ENV['extraction']} or --model for the build model",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return REFERENCE_EXTRACTION_MODEL


def work_root() -> Path:
    """Return the directory the graph and its response cache live under."""
    return Path(cache_root()) / "graphrag" / "serverfault"


def api_key() -> str | None:
    """Return the API key, never printing it, or None when none is set.

    ``LLM_API_KEY`` comes first, for a run against another endpoint, then the
    proxy key through :func:`aieng.advanced_rag.proxy.resolve_api_key`: the
    environment, then the repository ``.env``. Without one, a run the response
    cache fully answers still completes, and the first uncached call raises
    :class:`~aieng.advanced_rag.proxy.MissingProxyKeyError`.
    """
    try:
        key: str = resolve_api_key(names=("LLM_API_KEY", *KEY_NAMES))
    except MissingProxyKeyError:
        return None
    return key


def prompt_digest() -> str:
    """Return a short digest of the prompt and schema, so a change makes a new key."""
    material = json.dumps(
        [
            extraction.SYSTEM,
            extraction.PROMPT,
            extraction.ENTITY_TYPES,
            extraction.RELATION_TYPES,
            extraction.MAX_WINDOW_CHARS,
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def cache_key(documents: int, model: str = REFERENCE_EXTRACTION_MODEL) -> CacheKey:
    """Return the artifact key for a graph, naming everything that changes it.

    ``model`` defaults to the reference build model, so a reader restores the
    reference graph unless it names another build.
    """
    return CacheKey(
        ri_id=RI_ID,
        artifact_type="knowledge-graph",
        producer_id="direct-extraction",
        config={
            "model": model,
            "documents": documents,
            "slice_sha256": serverfault.SLICE_10K_SHA256,
            "dump_md5": serverfault.DUMP_MD5,
            "prompt": prompt_digest(),
            "max_tokens": MAX_TOKENS,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Extract the graph, validate it, report its cost, and store it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--documents", type=int, default=None, help="first N documents only")
    parser.add_argument(
        "--model",
        default=None,
        help=f"build model (default: {ROLE_ENV['extraction']}, then {REFERENCE_EXTRACTION_MODEL})",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    parser.add_argument("--no-store", action="store_true", help="skip the artifact cache")
    parser.add_argument("--out", type=Path, default=None, help="write the summary here")
    args = parser.parse_args(argv)
    model = build_model(args.model)

    payload = serverfault.fetch_slice(serverfault.SLICE_10K_QUESTIONS)
    docs = serverfault.documents(payload, args.documents)
    total = len(serverfault.documents(payload))
    suffix = f"-{len(docs)}" if args.documents else ""
    directory = work_root() / f"graph{suffix}"

    cache = ResponseCache(work_root() / "responses.sqlite")
    ledger_total = CallLedger("serverfault-graph")
    batch = ProxyBatch(
        model,
        cache=cache,
        rpm=args.rpm,
        max_parallel=args.workers,
        max_retries=16,
        timeout=600.0,
        ledger=ledger_total,
        api_key=api_key(),
        base_url=BASE_URL,
    )

    # ProxyBatch adds each family's thinking switch per call, and keeping it out
    # of the request keeps the cache keys of the reference graph's requests
    # unchanged.
    params = request_params(model, max_tokens=MAX_TOKENS, temperature=0.0)
    params.pop("extra_body", None)
    complete = batch.completer(**params)

    def progress(done: int, calls: int) -> None:
        if done % 250 == 0 or done == calls:
            print(f"  {done}/{calls} calls", flush=True)

    began = time.perf_counter()
    with batch:
        graph = extraction.build_graph(
            [(d.doc_id, d.text) for d in docs],
            complete,
            max_workers=args.workers,
            on_progress=progress,
        )
    seconds = time.perf_counter() - began
    problems = graph.validate()
    if problems:
        raise SystemExit(f"{len(problems)} schema problems, first: {problems[0]}")
    graph.save(directory)

    ledger = ledger_total.to_dict()
    cold = ledger["calls"] + ledger["cache_hits"]
    summary: dict[str, Any] = {
        "produced_by": {
            "model": model,
            "endpoint": endpoint_of(model),
            "producer": "graphrag_technical_support/build_serverfault_graph.py",
        },
        "model": model,
        "documents": len(docs),
        "wall_seconds": round(seconds, 1),
        "cost": ledger,
        "calls_per_document": round(cold / max(len(docs), 1), 2),
        "tokens_per_document": round(ledger["total_tokens"] / max(len(docs), 1), 1),
        "graph": graph.stats(),
        "projected_for_all_documents": (
            {
                "calls": round(cold / len(docs) * total),
                "hours_at_rpm": round(cold / len(docs) * total / args.rpm / 60, 1),
            }
            if len(docs) < total
            else None
        ),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.no_store:
        stored = store_directory(TieredResultCache(), cache_key(len(docs), model), directory)
        print(f"stored: {stored.metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
