"""Sweep the corrective loop's grading thresholds and write the report.

This runner measures ``GradingThresholds``. It harvests one trace per query,
replays the whole threshold grid over those traces at no further cost, and
reports what each operating point buys on the null questions next to what it
costs on the answerable ones. The result it produced, and the defaults that
came out of it, are in ``THRESHOLDS.md``.

Run the estimate first. A harvest arm's call count is printed before anything is
launched, and the run refuses to start above ``--max-calls`` without ``--yes``.

Examples
--------
Print what the headline arm would cost and stop::

    python3 run_threshold_sweep.py --corpus contract-nli --documents 40 \
        --estimate-only

Harvest and sweep it::

    python3 run_threshold_sweep.py --corpus contract-nli --documents 40 --yes

Embeddings run locally on ``BAAI/bge-small-en-v1.5``. The proxy is for chat
completions only; it hosts no embedding model of its own, so an embedding call
through it bills an external provider.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.corpora import contract_nli, cuad, multihop_rag
from aieng.advanced_rag.evaluation.sweep import (
    QueryTrace,
    control_rows,
    harvest_traces,
    null_only_sweep,
    sweep,
    threshold_grid,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.models import choose_models
from aieng.advanced_rag.pipeline.grading import (
    HeuristicGrader,
    LLMGrader,
)
from aieng.advanced_rag.pipeline.rewrite import LLMRewriter
from aieng.advanced_rag.pipeline.spine import Retrieve, evidence_from_hits
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url
from aieng.advanced_rag.retrieval.corpus import Query, Slice
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever
from sweep_lab import (
    ProxyClient,
    contract_nli_slice,
    cuad_slice,
    estimate_calls,
    markdown_table,
    scoped_null_queries,
    write_json,
)


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

# The axes swept. Both carry 0.6 and 0.3, the pair that shipped before this
# sweep, so its position on the curve stays visible rather than inferred.
#
# The range is set by where the two graders actually put their scores.
# ``HeuristicGrader`` scores term coverage, which is continuous and lands
# between 0.4 and 0.95 on this corpus, so it needs resolution across that
# whole span. ``LLMGrader`` returns the three values its prompt anchors, 0.0,
# 0.5 and 1.0, and nothing between them, so what matters there is which side
# of 0.5 and 1.0 each threshold falls on. Both axes therefore straddle 0.5 and
# reach 1.0, and ``insufficient`` starts at 0.0, which is the setting under
# which the grader never abstains and only the generator's own "not stated"
# path can.
SUFFICIENT_AXIS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
INSUFFICIENT_AXIS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: Embeddings are local by policy. See ``docs/models-and-spend.md``.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_retrieve(
    chunks: Sequence[Any],
    *,
    device: str,
    top_k: int,
) -> tuple[Retrieve, float]:
    """Index the slice locally and return a thread-safe retrieval callable.

    The proxy is the bottleneck, so the harvest runs many queries at once and
    the encoder is shared between them. A lock around the encode call keeps that
    honest: one model, one forward pass at a time, and no assumption about
    whether sentence-transformers is re-entrant.

    Parameters
    ----------
    chunks : sequence of Chunk
        The chunks to index.
    device : str
        Torch device for the encoder.
    top_k : int
        Passages to request per query.

    Returns
    -------
    tuple
        The retrieval callable, and the seconds indexing took.
    """
    embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL, device=device)
    retriever = DenseRetriever(embedder)
    started = time.perf_counter()
    retriever.index(list(chunks))
    elapsed = time.perf_counter() - started

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lock = threading.Lock()

    def retrieve(question: str, scope: str | None = None, k: int | None = None) -> list[Evidence]:
        query = Query(query_id="", doc_id=scope or "", category="", question=question)
        with lock:
            if scope is None:
                return list(retriever.retrieve(question, top_k if k is None else k, scope=None))
            hits = retriever.search(query, k=top_k if k is None else k)
        evidence: list[Evidence] = evidence_from_hits(hits, by_id, retriever.name)
        return evidence

    return retrieve, elapsed


def harvest_arm(
    queries: Sequence[Query],
    retrieve: Retrieve,
    qrels: dict[str, set[str]],
    client: ProxyClient,
    *,
    grader_name: str,
    attempts: int,
    top_k: int,
    evidence_budget: int,
    max_chunks: int,
    workers: int,
    progress_every: int = 25,
    grader_client: ProxyClient | None = None,
    rewrite_client: ProxyClient | None = None,
) -> list[QueryTrace]:
    """Harvest one grader's traces over one query set, many queries at a time.

    Each worker gets its own completer, so the per-stage call attribution inside
    ``harvest_traces`` is not read across threads. The grader and the rewriter a
    worker uses are built on that worker's completer for the same reason.

    Parameters
    ----------
    queries : sequence of Query
        The queries to harvest.
    retrieve : callable
        Takes ``(question, scope, k)``, returns ranked evidence.
    qrels : dict
        Query identifier to relevant chunk identifiers.
    client : ProxyClient
        Supplies completers and accumulates usage.
    grader_name : str
        ``"heuristic"`` or ``"llm"``.
    attempts : int
        Retrievals recorded per query.
    top_k : int
        Passages kept from each retrieval.
    evidence_budget : int
        Passages placed in the generation prompt.
    max_chunks : int
        Passages the LLM grader scores per attempt.
    workers : int
        Queries harvested at once.
    progress_every : int, optional
        Queries between progress lines.

    Returns
    -------
    list of QueryTrace
        Traces in the order the queries were given.
    """
    done = {"n": 0}
    lock = threading.Lock()
    started = time.perf_counter()

    def one(query: Query) -> QueryTrace:
        complete = client.completer()
        grade = grader_client.completer() if grader_client is not None and grader_name == "llm" else complete
        rewrite = rewrite_client.completer() if rewrite_client is not None else complete
        completions = {id(c): c for c in (complete, grade, rewrite)}
        grader: Any = HeuristicGrader() if grader_name == "heuristic" else LLMGrader(grade, max_chunks=max_chunks)
        trace = harvest_traces(
            [query],
            retrieve,
            grader,
            complete,
            qrels,
            rewriter=LLMRewriter(rewrite),
            top_k=top_k,
            evidence_budget=evidence_budget,
            attempts=attempts,
            count_calls=lambda: sum(c.calls for c in completions.values()),
        )[0]
        with lock:
            done["n"] += 1
            if done["n"] % progress_every == 0:
                rate = done["n"] / (time.perf_counter() - started)
                print(
                    f"  {grader_name}: {done['n']}/{len(queries)} ({rate:.2f} queries/s)",
                    flush=True,
                )
        return trace

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, queries))


def load_arm(
    corpus: str,
    *,
    documents: int | None,
    queries_limit: int | None,
) -> tuple[Slice, list[Query], str]:
    """Load one corpus arm: the indexed slice, the queries, and a label.

    The out-of-domain control has no corpus of its own that may be
    redistributed, so its queries are scoped onto the ContractNLI documents and
    retrieve contract clauses for a question about a news event.
    """
    if corpus == "contract-nli":
        eval_slice = contract_nli_slice(contract_nli.fetch_contract_nli(), documents=documents)
        return eval_slice, eval_slice.queries[:queries_limit], "ContractNLI test"
    if corpus == "cuad":
        eval_slice = cuad_slice(cuad.fetch_cuad_test(), contracts=documents)
        return eval_slice, eval_slice.queries[:queries_limit], "CUAD test"
    if corpus == "multihop-null":
        eval_slice = contract_nli_slice(contract_nli.fetch_contract_nli(), documents=documents)
        doc_ids = sorted({chunk.doc_id for chunk in eval_slice.chunks})
        queries = scoped_null_queries(multihop_rag.fetch_multihop_queries(), doc_ids, limit=queries_limit)
        return eval_slice, queries, "MultiHop-RAG nulls over ContractNLI"
    raise ValueError(f"unknown corpus {corpus!r}")


def report(
    traces_by_grader: dict[str, list[QueryTrace]],
    *,
    corpus: str,
    label: str,
    attempts: Sequence[int],
    evidence_budget: int,
    shape: dict[str, Any],
    usage: dict[str, int],
    output_dir: Path = OUTPUTS,
) -> dict[str, Any]:
    """Score every grid point for every grader arm and write the raw results."""
    payload: dict[str, Any] = {
        "corpus": corpus,
        "label": label,
        "slice": shape,
        "embedding_model": EMBEDDING_MODEL,
        "usage": usage,
        "grid": {
            "sufficient": list(SUFFICIENT_AXIS),
            "insufficient": list(INSUFFICIENT_AXIS),
        },
        "arms": {},
    }
    grid = threshold_grid(SUFFICIENT_AXIS, INSUFFICIENT_AXIS)
    for grader_name, traces in traces_by_grader.items():
        # An all-null control has no answerable side, so balanced accuracy and
        # the cost column are undefined for it and a SweepRow would invent both.
        # It reports the abstention rate on far nulls instead.
        if any(not trace.is_null for trace in traces):
            rows = sweep(
                traces,
                grid,
                attempts=attempts,
                grader=grader_name,
                evidence_budget=evidence_budget,
            )
            controls = control_rows(traces, evidence_budget=evidence_budget)
            payload["arms"][grader_name] = {
                "null_only": False,
                "rows": [row.to_dict() for row in rows],
                "controls": [row.to_dict() for row in controls],
            }
        else:
            payload["arms"][grader_name] = {
                "null_only": True,
                "rows": null_only_sweep(traces, grid, attempts=attempts, grader=grader_name),
                "controls": [],
            }
        write_json(
            output_dir / f"traces_{corpus}_{grader_name}.json",
            [trace.to_dict() for trace in traces],
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Return the runner's command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="contract-nli",
        choices=("contract-nli", "cuad", "multihop-null"),
    )
    parser.add_argument("--documents", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--graders", default="heuristic,llm", help="comma-separated grader arms")
    parser.add_argument("--attempts", default="1,2", help="comma-separated")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--evidence-budget", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--model", help="Use one model for all roles; otherwise use the shared role settings")
    parser.add_argument("--env", type=Path, default=HERE.parents[1] / ".env")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-calls", type=int, default=12000)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--cache-only", action="store_true", help="require complete saved traces; never construct a model client"
    )
    parser.add_argument("--replay-dir", type=Path, help="replay a complete trace set without retrieval or model calls")
    parser.add_argument(
        "--coverage-only", action="store_true", help="audit saved query coverage without scoring or model calls"
    )
    parser.add_argument("--dense-evidence", type=Path, help="optional saved dense table to audit with --coverage-only")
    parser.add_argument("--cache", default=str(OUTPUTS / "responses.sqlite"))
    return parser


def print_tables(
    payload: dict[str, Any],
    traces_by_grader: dict[str, list[QueryTrace]],
    *,
    label: str,
    attempts: Sequence[int],
    evidence_budget: int,
) -> None:
    """Print one table per grader arm, best operating point first."""
    for grader_name, arm in payload["arms"].items():
        print(f"\n### {label}, {grader_name} grader\n")
        if arm["null_only"]:
            best = max(arm["rows"], key=lambda row: row["abstain_rate"])
            worst = min(arm["rows"], key=lambda row: row["abstain_rate"])
            print(
                f"far-null abstention runs {worst['abstain_rate']:.3f} at "
                f"{worst['label']} (attempts {worst['max_attempts']}) to "
                f"{best['abstain_rate']:.3f} at {best['label']} "
                f"(attempts {best['max_attempts']})"
            )
            continue
        traces = traces_by_grader[grader_name]
        rows = sweep(
            traces,
            threshold_grid(SUFFICIENT_AXIS, INSUFFICIENT_AXIS),
            attempts=attempts,
            grader=grader_name,
            evidence_budget=evidence_budget,
        )
        controls = control_rows(traces, evidence_budget=evidence_budget)
        print(markdown_table(rows + controls, by_score=True))


def replay_saved(
    args: argparse.Namespace,
    eval_slice: Slice,
    queries: Sequence[Query],
    graders: Sequence[str],
    *,
    label: str,
    attempts: Sequence[int],
    shape: dict[str, Any],
) -> int:
    """Validate complete saved traces and replay without any model client."""
    if (args.top_k, args.evidence_budget, args.max_chunks) != (10, 5, 5):
        raise ValueError(
            "legacy cached traces require top-k=10, evidence-budget=5, and max-chunks=5; their recorded generations cannot be rescored at a different reader budget"
        )
    expected = {query.query_id for query in queries}
    traces_by_grader = {}
    for name in graders:
        raw = json.loads((args.replay_dir / f"traces_{args.corpus}_{name}.json").read_text())
        if any(type(row.get("is_null")) is not bool for row in raw):
            raise ValueError(f"{name}: cached null labels must be booleans")
        for row in raw:
            if not row.get("attempts"):
                raise ValueError(f"{name}: cached trace has no attempt data")
            for attempt in row["attempts"]:
                if any(type(attempt.get(field)) is not bool for field in ("generated_abstained", "rewrite_available")):
                    raise ValueError(f"{name}: cached generation and rewrite flags must be booleans")
                if any(type(score) not in (int, float) or not 0 <= score <= 1 for score in attempt["scores"]):
                    raise ValueError(f"{name}: cached grades must be finite numbers between zero and one")
        traces = [QueryTrace.from_dict(row) for row in raw]
        if len(traces) != len(expected) or {trace.query_id for trace in traces} != expected:
            raise ValueError(f"{name}: cached trace IDs do not match all {len(expected)} requested queries")
        by_id = {query.query_id: query for query in queries}
        for trace in traces:
            query = by_id[trace.query_id]
            if trace.question != query.question or trace.is_null != (not query.is_answerable):
                raise ValueError(f"{name}: cached question or answerability differs from the pinned corpus")
            if trace.gold_chunk_ids != frozenset(eval_slice.qrels.get(query.query_id, set())):
                raise ValueError(f"{name}: cached gold judgments differ from the requested chunking")
        traces_by_grader[name] = traces
    payload = report(
        traces_by_grader,
        corpus=args.corpus,
        label=label,
        attempts=attempts,
        evidence_budget=args.evidence_budget,
        shape=shape,
        usage={"network_calls": 0},
        output_dir=args.output_dir,
    )
    payload["execution"] = "complete saved traces replayed; zero new model calls"
    payload["cache_configuration_assumption"] = {
        "top_k": 10,
        "evidence_budget": 5,
        "max_chunks": 5,
        "provenance": "historical RI #1 configuration; legacy trace files do not contain a generation-budget or model manifest",
    }
    write_json(args.output_dir / f"sweep_{args.corpus}.json", payload)
    print_tables(payload, traces_by_grader, label=label, attempts=attempts, evidence_budget=args.evidence_budget)
    return 0


def coverage_saved(args: argparse.Namespace, eval_slice: Slice, queries: Sequence[Query]) -> int:
    """Audit available caches without treating partial traces as a measurement."""
    from advanced_lab.common import digest, provenance  # noqa: PLC0415

    if args.replay_dir is None:
        raise ValueError("--coverage-only requires --replay-dir")
    expected = {query.query_id for query in queries}
    coverage = {}
    for name in (value.strip() for value in args.graders.split(",") if value.strip()):
        path = args.replay_dir / f"traces_{args.corpus}_{name}.json"
        rows = json.loads(path.read_text())
        ids = {row["query_id"] for row in rows}
        coverage[name] = {
            "cached_queries": len(ids & expected),
            "missing_queries": len(expected - ids),
            "unexpected_queries": len(ids - expected),
            "duplicate_rows": len(rows) - len(ids),
            "trace_sha256": digest(path),
        }
    dense = None
    if args.dense_evidence is not None:
        ids = set(json.loads(args.dense_evidence.read_text()))
        dense = len(ids & expected)
    payload = {
        **provenance(),
        "model": None,
        "model_roles": None,
        "requested_model": args.model,
        "execution": "cache-only coverage audit; no new measurement",
        "corpus": eval_slice.summary(),
        "threshold_trace_coverage": coverage,
        "existing_dense_evidence_queries": dense,
        "dense_evidence_sha256": digest(args.dense_evidence) if args.dense_evidence else None,
        "full_split_measurement_complete": False,
        "new_model_calls": 0,
        "source_trace_model_provenance": "Legacy trace files have no per-trace model metadata. Replay preserves source outcomes and does not relabel their model.",
        "reason": "Coverage is not quality scoring or cache validation. Complete ID, question, answerability, and qrel validation occurs during replay; no partial full-split quality figures are inferred.",
    }
    if args.corpus == "contract-nli":
        payload["corpus_sha256"] = digest(contract_nli.fetch_contract_nli())
    write_json(args.output_dir / "cache_status.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


def experiment_axes(grader_names: str, attempt_counts: str) -> tuple[list[str], tuple[int, ...]]:
    """Validate the named arms before loading a corpus or constructing clients."""
    graders = [name.strip() for name in grader_names.split(",") if name.strip()]
    if not graders or len(set(graders)) != len(graders) or not set(graders) <= {"heuristic", "llm"}:
        raise ValueError("graders must be a nonempty, unique selection of heuristic,llm")
    attempts = tuple(int(value) for value in attempt_counts.split(","))
    if min(attempts) < 1:
        raise ValueError("attempt counts must be positive")
    return graders, attempts


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, estimate the cost, and run the sweep."""
    args = build_parser().parse_args(argv)
    if args.coverage_only:
        args.cache_only = True
    graders, attempts = experiment_axes(args.graders, args.attempts)
    harvest_attempts = max(attempts)

    eval_slice, queries, label = load_arm(args.corpus, documents=args.documents, queries_limit=args.queries)
    shape = dict(eval_slice.summary())
    if args.output_dir == OUTPUTS:
        if args.corpus == "contract-nli" and shape.get("contracts") == 123:
            args.output_dir = OUTPUTS / "full_contract_nli"
        elif args.cache_only or args.replay_dir is not None:
            args.output_dir = OUTPUTS / "cached_replay"
    shape["harvested_queries"] = len(queries)
    shape["null_queries_harvested"] = sum(1 for query in queries if not query.is_answerable)
    if args.coverage_only:
        return coverage_saved(args, eval_slice, queries)

    total = 0
    print(f"{label}: {len(queries)} queries, {len(eval_slice.chunks)} chunks")
    for grader_name in graders:
        estimate = estimate_calls(
            len(queries),
            attempts=harvest_attempts,
            max_chunks=args.max_chunks,
            llm_grader=grader_name == "llm",
        )
        total += estimate["total"]
        print(f"  {grader_name} arm: {estimate}")
    print(f"  estimated model calls: {total} (upper bound, before cache hits)")

    if args.estimate_only:
        return 0
    if args.cache_only and args.replay_dir is None:
        raise ValueError("--cache-only requires --replay-dir with complete saved traces for the requested slice")
    if args.replay_dir is not None:
        return replay_saved(args, eval_slice, queries, graders, label=label, attempts=attempts, shape=shape)
    if total > args.max_calls and not args.yes:
        print(
            f"refusing: {total} calls is over --max-calls {args.max_calls}. "
            "Cut the grid with --documents or --queries, or pass --yes.",
            file=sys.stderr,
        )
        return 1

    retrieve, index_seconds = build_retrieve(eval_slice.chunks, device=args.device, top_k=args.top_k)
    shape["index_seconds"] = round(index_seconds, 1)
    print(f"  indexed in {index_seconds:.1f}s on {args.device}")

    models = (
        choose_models(generation=args.model, judging=args.model, extraction=args.model)
        if args.model
        else choose_models()
    )
    connection: dict[str, Any] = {
        "api_key": resolve_api_key(args.env),
        "base_url": resolve_base_url(args.env),
        "cache_path": Path(args.cache),
    }
    client = ProxyClient(model=models.generation.model, **connection)
    grade_client = ProxyClient(model=models.judging.model, **connection)
    rewrite_client = ProxyClient(model=models.extraction.model, **connection)
    traces_by_grader: dict[str, list[QueryTrace]] = {}
    for grader_name in graders:
        started = time.perf_counter()
        traces_by_grader[grader_name] = harvest_arm(
            queries,
            retrieve,
            eval_slice.qrels,
            client,
            grader_name=grader_name,
            attempts=harvest_attempts,
            top_k=args.top_k,
            evidence_budget=args.evidence_budget,
            max_chunks=args.max_chunks,
            workers=args.workers,
            grader_client=grade_client,
            rewrite_client=rewrite_client,
        )
        print(
            f"  {grader_name} arm harvested in {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    payload = report(
        traces_by_grader,
        corpus=args.corpus,
        label=label,
        attempts=attempts,
        evidence_budget=args.evidence_budget,
        shape=shape,
        usage={key: sum(c.usage()[key] for c in (client, grade_client, rewrite_client)) for key in client.usage()},
        output_dir=args.output_dir,
    )
    payload["model"] = models.generation.model
    payload["model_roles"] = models.as_dict()
    path = write_json(args.output_dir / f"sweep_{args.corpus}.json", payload)
    print(f"  wrote {path}")

    print_tables(
        payload,
        traces_by_grader,
        label=label,
        attempts=attempts,
        evidence_budget=args.evidence_budget,
    )
    print(f"\nusage: {client.usage()}")
    print(spend.report(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
