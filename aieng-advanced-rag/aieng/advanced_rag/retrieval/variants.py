"""Score chunking variants under one rule, scoped and unscoped.

A chunking variant changes one of two things: the units a retriever ranks, or
the text it embeds for each unit. Hierarchical chunking ranks small children and
hands their parents to the reader. A header or an LLM-written context changes
the embedded text and leaves the spans alone. To compare them fairly, every arm
here is scored on the unit the reader receives, with judgments derived from that
unit's own span, at a fixed depth and again at the character budget the control
reads at that depth.

Scope is the second axis. With ``scoped`` true a query ranks only its own
document's units, which is how RI #1 runs. With ``scoped`` false it ranks the
whole corpus, and the question has to carry whatever identifies the document.
:func:`document_hit_rate` reports how often the right document reached the top
``k`` at all, which separates finding the document from finding the clause.

This module needs numpy, so the package root does not import it. Reach it by
path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..evaluation.metrics import retrieved_chars, score_retrieval
from .corpus import Chunk, Query
from .index import DenseIndex


@dataclass(frozen=True)
class Arm:
    """One chunking variant, ready to embed.

    Parameters
    ----------
    name : str
        Label for the report.
    indexed : sequence of Chunk
        The units the retriever ranks. ``indexed_text`` is what gets embedded.
    read : sequence of Chunk or None, optional
        The units the reader receives, when they differ from ``indexed``.
    read_of : mapping or None, optional
        Indexed-unit identifier to read-unit identifier. Required with ``read``.
    """

    name: str
    indexed: Sequence[Chunk]
    read: Sequence[Chunk] | None = None
    read_of: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Check that a hierarchical arm says how children map to parents."""
        if (self.read is None) != (self.read_of is None):
            raise ValueError("pass read and read_of together")

    @property
    def read_units(self) -> Sequence[Chunk]:
        """Return the units a reader receives."""
        return self.read if self.read is not None else self.indexed


def judgments(units: Sequence[Chunk], queries: Sequence[Query]) -> dict[str, set[str]]:
    """Derive relevance from overlap between each query's gold spans and the units.

    Parameters
    ----------
    units : sequence of Chunk
        The read units of one arm.
    queries : sequence of Query
        The queries to judge.

    Returns
    -------
    dict
        Query identifier to the identifiers of overlapping units in the query's
        own document.
    """
    by_doc: dict[str, list[Chunk]] = {}
    for unit in units:
        by_doc.setdefault(unit.doc_id, []).append(unit)
    qrels: dict[str, set[str]] = {}
    for query in queries:
        relevant: set[str] = set()
        for answer in query.answers:
            for unit in by_doc.get(query.doc_id, ()):
                if unit.start < answer.end and answer.start < unit.end:
                    relevant.add(unit.chunk_id)
        qrels[query.query_id] = relevant
    return qrels


def rank(
    index: DenseIndex,
    query_vectors: npt.NDArray[np.float32],
    queries: Sequence[Query],
    *,
    scoped: bool,
    depth: int,
    read_of: Mapping[str, str] | None = None,
    overfetch: int = 8,
) -> dict[str, list[str]]:
    """Rank read units for each query, inside its document or across the corpus.

    When indexed units map to read units, the ranked children are collapsed to
    their parents in first-hit order, so a cluster of hits inside one parent
    costs one read unit.

    Parameters
    ----------
    index : DenseIndex
        The arm's index over its indexed units.
    query_vectors : numpy.ndarray
        One normalised vector per query, in query order.
    queries : sequence of Query
        The queries; ``doc_id`` sets the scope when ``scoped`` is true.
    scoped : bool
        Rank only the query's own document.
    depth : int
        Read units to return per query.
    read_of : mapping or None, optional
        Indexed-unit identifier to read-unit identifier.
    overfetch : int, optional
        How many indexed units to rank per read unit wanted, when collapsing.

    Returns
    -------
    dict
        Query identifier to ranked read-unit identifiers.
    """
    fetch = depth if read_of is None else depth * overfetch
    runs: dict[str, list[str]] = {}
    for vector, query in zip(query_vectors, queries):
        rows = index.rows_for_doc(query.doc_id) if scoped else None
        ranked = index.rank(vector, rows, fetch)
        if read_of is not None:
            seen: dict[str, None] = {}
            for unit_id in ranked:
                seen.setdefault(read_of[unit_id], None)
                if len(seen) >= depth:
                    break
            ranked = list(seen)
        runs[query.query_id] = ranked[:depth]
    return runs


def document_hit_rate(
    runs: Mapping[str, Sequence[str]],
    doc_of: Mapping[str, str],
    queries: Sequence[Query],
    k: int,
) -> float:
    """Return the share of queries whose top ``k`` holds any unit of their document.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked unit identifiers.
    doc_of : mapping
        Unit identifier to document identifier.
    queries : sequence of Query
        The queries scored.
    k : int
        Depth.

    Returns
    -------
    float
        Rounded to four places.
    """
    if not queries:
        return 0.0
    hits = sum(
        any(doc_of.get(unit) == q.doc_id for unit in runs.get(q.query_id, [])[:k])
        for q in queries
    )
    return round(hits / len(queries), 4)


def nearest_k(
    runs: Mapping[str, Sequence[str]],
    chars_by_id: Mapping[str, int],
    budget: float,
    depth: int,
) -> int:
    """Return the depth whose mean characters read is closest to ``budget``.

    ``evaluation.metrics.budget_k`` returns the smallest depth that reaches the
    budget, which can hand one arm a whole extra chunk when its top three fall a
    few characters short. Between arms with equal unit lengths, the nearest
    depth is the same depth, which is the comparison wanted. Ties go to the
    smaller depth.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked unit identifiers.
    chars_by_id : mapping
        Unit identifier to its length in characters.
    budget : float
        Characters the reader is allowed.
    depth : int
        Largest depth to consider.

    Returns
    -------
    int
        The depth to compare at.
    """
    gaps = [
        (abs(retrieved_chars(runs, chars_by_id, k) - budget), k)
        for k in range(1, depth + 1)
    ]
    return min(gaps)[1]


def score_arm(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    units: Sequence[Chunk],
    queries: Sequence[Query],
    *,
    k: int,
    budgets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Score one arm at depth ``k`` and at named character budgets.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked read-unit identifiers, deep enough for the
        budget depth.
    qrels : mapping
        Judgments from :func:`judgments` over the same read units.
    units : sequence of Chunk
        The read units, for their lengths and documents.
    queries : sequence of Query
        The answerable queries that were ranked.
    k : int
        The fixed depth.
    budgets : mapping or None, optional
        Label to characters the reader is allowed, usually the control's mean
        at a few depths. Each is scored at the depth whose mean is nearest it.
        A tight budget matters when the documents are short: at ``k=10`` a
        scoped retriever can return most of a document, and the failure rate
        stops separating the arms.

    Returns
    -------
    dict
        ``fixed`` and ``budgets`` entries, each with the failure rate, MRR,
        nDCG, the mean characters read, and the document hit rate at that
        depth.
    """
    chars = {u.chunk_id: len(u.text) for u in units}
    doc_of = {u.chunk_id: u.doc_id for u in units}
    depth = max(len(r) for r in runs.values()) if runs else k

    def at(depth_k: int) -> dict[str, Any]:
        cut = {qid: list(r[:depth_k]) for qid, r in runs.items()}
        report = score_retrieval(cut, qrels, k=depth_k).to_dict()
        return {
            "k": depth_k,
            "failure": report["retrieval_failure_rate"],
            "mrr": report["mrr"],
            "ndcg": report[f"ndcg_at_{depth_k}"],
            "queries": report["queries"],
            "chars": round(retrieved_chars(cut, chars, depth_k)),
            "document_hit": document_hit_rate(cut, doc_of, queries, depth_k),
        }

    out: dict[str, Any] = {"fixed": at(k)}
    out["budgets"] = {
        label: at(nearest_k(runs, chars, allowed, depth))
        for label, allowed in (budgets or {}).items()
    }
    out["gold_units_per_query"] = round(
        sum(len(qrels[q.query_id]) for q in queries) / max(len(queries), 1), 2
    )
    return out
