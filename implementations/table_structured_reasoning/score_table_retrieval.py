#!/usr/bin/env python3
"""Score plain against table-aware chunking on the FinanceBench questions.

Reads the section folders that ``ingest_financebench.py`` wrote, chunks every
filing two ways over the same serialised text, and scores BM25 and a local dense
arm against page-level judgments derived from FinanceBench's evidence pages. The
dense arm is ``BAAI/bge-small-en-v1.5`` run locally, never a paid embedding API.

For each chunking it also reports how many numeric table cells keep their column
header and row label inside one chunk, which does not depend on any question.

The result is cached through the artifact cache under a key that names the pinned
dataset commit, the parser version, the chunk size, and the embedder, so a
re-run is a cache hit and a changed input recomputes.

Examples
--------
Score and print a table, then read it back from the cache::

    $ python score_table_retrieval.py
    $ python score_table_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora.financebench import (
    FINANCEBENCH_COMMIT,
    load_queries,
)
from aieng.advanced_rag.evaluation.metrics import score_retrieval
from aieng.advanced_rag.ingestion import PARSER_VERSION
from aieng.advanced_rag.retrieval.chunkers import all_facts
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from aieng.advanced_rag.retrieval.filings import (
    FilingSlice,
    Strategy,
    build_filing_slice,
)
from aieng.advanced_rag.retrieval.retrievers import (
    DenseRetriever,
    Ranker,
    run_retriever,
)
from aieng.advanced_rag.retrieval.sparse import BM25Retriever
from aieng.advanced_rag.retrieval.tables import attachment_report


RI_ID = "ri3"
EMBEDDER = "BAAI/bge-small-en-v1.5"
STRATEGIES: tuple[Strategy, ...] = ("plain", "table_aware")


def default_root() -> Path:
    """Return where the ingestion script wrote the section folders."""
    return Path(cache_root()) / "ingestion" / "financebench"


def attachment(sliced: FilingSlice) -> dict[str, float | int]:
    """Sum the table-cell attachment counts over every filing in a slice."""
    totals: dict[str, float | int] = {}
    for filing in sliced.filings.values():
        report = attachment_report(filing.chunks, all_facts(filing.items))
        for name, value in report.items():
            if not name.endswith("_rate"):
                totals[name] = totals.get(name, 0) + value
    pairs, cells = totals.get("chunk_cell_pairs", 0), totals.get("scored_cells", 0)
    totals["attachment_rate"] = round(totals["attached_pairs"] / pairs, 4) if pairs else 0.0
    totals["cell_recoverable_rate"] = round(totals["cells_ever_complete"] / cells, 4) if cells else 0.0
    return totals


def score_strategy(strategy: Strategy, folders: dict[str, Path], k: int, max_chars: int) -> dict[str, Any]:
    """Chunk every filing one way and score both retrieval arms."""
    queries = load_queries()
    began = time.perf_counter()
    sliced = build_filing_slice(queries, folders, strategy=strategy, max_chars=max_chars)
    asked = [q.as_query() for q in sliced.queries]
    arms: dict[str, Ranker] = {
        "bm25": BM25Retriever(),
        "dense": DenseRetriever(SentenceTransformerEmbedder(EMBEDDER)),
    }
    scores: dict[str, Any] = {}
    for name, retriever in arms.items():
        retriever.index(sliced.chunks)
        runs = run_retriever(retriever, asked, k=k)
        scores[name] = score_retrieval(runs, sliced.qrels, k=k).to_dict()
    return {
        "strategy": strategy,
        "filings": len(sliced.filings),
        "chunks": len(sliced.chunks),
        "queries_scored": len(sliced.queries),
        "uncovered": sliced.uncovered,
        "attachment": attachment(sliced),
        "retrieval": scores,
        "seconds": round(time.perf_counter() - began, 1),
    }


def run(root: Path, k: int, max_chars: int, force_refresh: bool) -> dict[str, Any]:
    """Score both chunkings, through the artifact cache."""
    folders = {path.parent.name: path.parent for path in sorted(root.glob("*/ingestion.json"))}
    key = CacheKey(
        ri_id=RI_ID,
        artifact_type="table-retrieval-scores",
        producer_id="score_table_retrieval",
        config={
            "dataset_commit": FINANCEBENCH_COMMIT,
            "parser_version": PARSER_VERSION,
            "max_chars": max_chars,
            "k": k,
            "embedder": EMBEDDER,
            "filings": sorted(folders),
        },
    )
    result = TieredResultCache().load_or_compute(
        key,
        lambda: {s: score_strategy(s, folders, k, max_chars) for s in STRATEGIES},
        force_refresh=force_refresh,
    )
    value: dict[str, Any] = dict(result.value)
    value["cache_hit"] = result.cache_hit
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Score, print a summary table, and optionally write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(args.root or default_root(), args.k, args.max_chars, args.force_refresh)
    print(f"cache hit: {result['cache_hit']}")
    for strategy in STRATEGIES:
        row = result[strategy]
        print(
            f"{strategy}: {row['filings']} filings, {row['chunks']} chunks, "
            f"{row['queries_scored']} questions scored, {len(row['uncovered'])} uncovered"
        )
        for arm, report in row["retrieval"].items():
            print(
                f"  {arm}: failure {report['retrieval_failure_rate']}, "
                f"MRR {report['mrr']}, recall@{args.k} {report[f'recall_at_{args.k}']}"
            )
        attached = row["attachment"]
        print(
            f"  cells: {attached['attachment_rate']} of chunk-cell pairs attached, "
            f"{attached['cell_recoverable_rate']} of cells recoverable"
        )
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
