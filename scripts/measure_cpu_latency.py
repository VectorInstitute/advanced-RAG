#!/usr/bin/env python3
"""Re-measure the CPU latency figures under the bootcamp container's CPU budget.

The bootcamp runs in a container with 8 CPUs and no GPU. This script pins the
process to 8 CPUs, hides every GPU, and times the arms the teaching material
quotes latency for: the cross-encoder rerankers, the dense query encoder, and,
when the ``representations`` extra is installed, the SPLADE and ColBERT query
encoders. Pairs per second is the figure to compare against older numbers,
because it does not depend on how many candidates a query reranks.

A scaling check runs first. It times one matrix multiply on 1 thread and on 8 and
reports the ratio, so a result from a CPU that does not scale across threads says
so instead of passing as a container figure.

The output names the CPU budget and the container it stands in for and nothing
about the host.

Examples
--------
Measure and write the report::

    $ python scripts/fetch_test_data.py
    $ python scripts/measure_cpu_latency.py --out latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


CPUS = 8
CONTAINER = "e2-standard-8"

# These have to be set before torch and onnxruntime are imported, because both
# read the CPU affinity and the thread variables once at load.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(CPUS)
os.environ["MKL_NUM_THREADS"] = str(CPUS)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_available = sorted(os.sched_getaffinity(0))
if len(_available) < CPUS:
    sys.exit(f"this measurement needs at least {CPUS} CPUs; {len(_available)} given")
os.sched_setaffinity(0, set(_available[:CPUS]))

import torch  # noqa: E402
from aieng.advanced_rag.corpora.cuad import fetch_cuad_test  # noqa: E402
from aieng.advanced_rag.retrieval.corpus import Query, build_slice  # noqa: E402
from aieng.advanced_rag.retrieval.embedders import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from aieng.advanced_rag.retrieval.rerankers import CrossEncoderReranker  # noqa: E402
from aieng.advanced_rag.retrieval.sparse import BM25Retriever  # noqa: E402


RERANKERS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-v2-m3",
)
DENSE = "BAAI/bge-small-en-v1.5"
Work = list[tuple[Query, list[tuple[str, str]]]]
# A cap, not a count: a contract with fewer chunks reranks fewer, so the report
# gives pairs per query and pairs per second, which is the comparable figure.
CANDIDATES = 30


def scaling_ratio(size: int = 2048, repeats: int = 5) -> float:
    """Return how much faster a matrix multiply runs on 8 threads than on 1."""
    matrix = torch.randn(size, size)
    rates = {}
    for threads in (1, CPUS):
        torch.set_num_threads(threads)
        matrix @ matrix  # warm up
        began = time.perf_counter()
        for _ in range(repeats):
            matrix @ matrix
        rates[threads] = repeats / (time.perf_counter() - began)
    return round(rates[CPUS] / rates[1], 2)


def workload(queries: int) -> Work:
    """Return queries with ``CANDIDATES`` first-stage candidates each."""
    prepared = build_slice(fetch_cuad_test(), max_contracts=20)
    bm25 = BM25Retriever()
    bm25.index(prepared.chunks)
    texts = {chunk.chunk_id: chunk.indexed_text for chunk in prepared.chunks}
    work = []
    for query in prepared.answerable[:queries]:
        hits = bm25.search(query, k=CANDIDATES)
        work.append((query, [(hit.chunk_id, texts[hit.chunk_id]) for hit in hits]))
    return work


def time_rerankers(work: Work) -> list[dict[str, Any]]:
    """Time each reranker over the workload, scoring pairs per second."""
    rows = []
    for model in RERANKERS:
        reranker = CrossEncoderReranker(model)
        reranker.rerank(work[0][0].question, work[0][1], k=10)  # warm up
        reranker.report.seconds = 0.0
        reranker.report.pairs = 0
        reranker.report.queries = 0
        for query, candidates in work:
            reranker.rerank(query.question, candidates, k=10)
        report = reranker.report
        rows.append(
            {
                "arm": model,
                "queries": report.queries,
                "pairs_per_query": round(report.pairs / report.queries, 1),
                "pairs_per_second": round(report.pairs_per_second, 2),
                "seconds_per_query": round(report.seconds / report.queries, 3),
            }
        )
    return rows


def time_query_encoders(work: Work) -> list[dict[str, Any]]:
    """Time query encoding for the dense arm and, when installed, SPLADE and ColBERT."""
    questions = [query.question for query, _ in work]
    arms = {DENSE: SentenceTransformerEmbedder(DENSE).encode_queries}
    try:
        from aieng.advanced_rag.retrieval.representations import (  # noqa: PLC0415
            FastembedColbertEncoder,
            FastembedSparseEncoder,
        )
    except ImportError:
        pass
    else:
        arms["splade query"] = FastembedSparseEncoder().encode_query
        arms["colbert query"] = FastembedColbertEncoder().encode_query
    rows = []
    for name, encode in arms.items():
        # fastembed encodes one query per call and the embedder takes a list, so
        # both are timed per question.
        call = (lambda q, f=encode: f([q])) if name == DENSE else encode
        call(questions[0])  # warm up
        times = []
        for question in questions:
            began = time.perf_counter()
            call(question)
            times.append((time.perf_counter() - began) * 1000)
        rows.append(
            {
                "arm": name,
                "queries": len(times),
                "ms_per_query_mean": round(statistics.mean(times), 1),
                "ms_per_query_p95": round(sorted(times)[int(0.95 * len(times))], 1),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the measurement, print the report, and optionally write it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    ratio = scaling_ratio()
    torch.set_num_threads(CPUS)
    work = workload(args.queries)
    report = {
        "cpu_limit": CPUS,
        "stands_in_for": CONTAINER,
        "gpu": "hidden",
        "thread_scaling_1_to_8": ratio,
        "rerank_candidate_cap": CANDIDATES,
        "rerankers": time_rerankers(work),
        "query_encoders": time_query_encoders(work),
        "not_re_measured": [
            "ColPali and ColModernVBERT (not installed here)",
            "Docling (not installed here)",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
