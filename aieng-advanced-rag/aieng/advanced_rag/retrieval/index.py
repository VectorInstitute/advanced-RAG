"""A batched dense index, for sweeps that re-index a corpus many times.

``retrievers.DenseRetriever`` encodes and ranks one query at a time. This takes
vectors that are already computed, holds them in the same Qdrant store, and
ranks a batch of queries in one call, which is what makes a twenty-arm sweep
take minutes instead of hours. ``tests/.../test_index.py`` asserts the two rank
identically.

Rows are sorted by chunk identifier at build time and ties are broken on that
identifier ascending, matching ``DenseRetriever``. CUAD windows repeat
boilerplate, so exact score ties are common, and a different tie-break moves the
fourth decimal of the control for a reason that has nothing to do with the
retriever.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .corpus import Chunk
from .embedders import Embedder
from .stores import QdrantStore


# Inputs take any float width. The index casts them to float32 and stores that.
_FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class IndexCost:
    """What it cost to embed and hold the corpus.

    Parameters
    ----------
    chunks : int
        Chunks indexed.
    dimension : int
        Vector width.
    index_seconds : float
        Wall time of the embedding pass.
    chunks_per_second : float
        Indexing throughput.
    index_bytes : int
        Size of the float32 matrix.
    """

    chunks: int
    dimension: int
    index_seconds: float
    chunks_per_second: float
    index_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary."""
        return {
            "chunks": self.chunks,
            "dimension": self.dimension,
            "index_seconds": round(self.index_seconds, 2),
            "chunks_per_second": round(self.chunks_per_second, 2),
            "index_bytes": self.index_bytes,
            "index_mib": round(self.index_bytes / (1024 * 1024), 3),
        }


@dataclass(frozen=True)
class QueryCost:
    """Single-shot query cost, split into encoding and scoring.

    Split because the two scale differently. Encoding is the model and grows
    with parameter count; scoring is the matrix product and grows with corpus
    size and dimension. A model that is slow to encode is slow on every query
    forever, which is the column that decides whether it belongs in a workshop.

    Parameters
    ----------
    queries : int
        Queries measured.
    median_ms : float
        Median end-to-end milliseconds per query.
    mean_ms : float
        Mean end-to-end milliseconds per query.
    p95_ms : float
        95th percentile end-to-end milliseconds.
    median_encode_ms : float
        Median milliseconds spent encoding the query.
    median_score_ms : float
        Median milliseconds spent ranking.
    """

    queries: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    median_encode_ms: float
    median_score_ms: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary."""
        return {
            "queries": self.queries,
            "median_ms": round(self.median_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "median_encode_ms": round(self.median_encode_ms, 2),
            "median_score_ms": round(self.median_score_ms, 2),
        }


class DenseIndex:
    """A batched view of one Qdrant collection, with rows in chunk-ID order.

    Rows are sorted by chunk identifier once at build time and become the
    collection's point identifiers, so row ``i`` is point ``i`` and a row
    restriction is a filter on point identifiers.

    Parameters
    ----------
    chunk_ids : sequence of str
        One identifier per row of ``vectors``, in any order.
    vectors : numpy.ndarray
        L2-normalised embeddings, one row per chunk.
    doc_ids : sequence of str or None, optional
        The document each chunk belongs to. Required for document-scoped
        search, which is what CUAD needs; ``None`` supports corpus-wide search
        only, which is what a ServerFault-shaped corpus needs.
    store : QdrantStore or None, optional
        An empty store to build in. ``None`` builds an in-process one.

    Raises
    ------
    ValueError
        If the identifier count does not match the vector count.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        vectors: _FloatArray,
        doc_ids: Sequence[str] | None = None,
        *,
        store: QdrantStore | None = None,
    ) -> None:
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunk_ids)} chunk ids against {vectors.shape[0]} vectors",
            )
        order = np.argsort(np.asarray(chunk_ids, dtype=object), kind="stable")
        self.chunk_ids: list[str] = [chunk_ids[i] for i in order]
        self.doc_ids: list[str] | None = (
            [doc_ids[i] for i in order] if doc_ids is not None else None
        )
        matrix: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)[order]
        self._dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        self.store = (
            store
            if store is not None
            else QdrantStore(collection=f"index-{uuid.uuid4().hex[:12]}")
        )
        if len(self.store):
            raise ValueError("DenseIndex needs an empty store")
        self.store.add(self.chunk_ids, matrix, self.doc_ids)
        self._row_of = {cid: row for row, cid in enumerate(self.chunk_ids)}
        buckets: dict[str, list[int]] = {}
        for row, doc_id in enumerate(self.doc_ids or []):
            buckets.setdefault(doc_id, []).append(row)
        self._rows_by_doc = {
            d: np.asarray(r, dtype=np.int64) for d, r in buckets.items()
        }

    @property
    def dimension(self) -> int:
        """Return the vector width."""
        return self._dimension

    @property
    def nbytes(self) -> int:
        """Return the size of the float32 vectors in bytes."""
        return len(self.chunk_ids) * self._dimension * 4

    @property
    def matrix(self) -> NDArray[np.float32]:
        """Return the stored vectors, read back from the store, in row order."""
        return self.store.vectors()

    def rows_for(self, chunk_ids: Sequence[str]) -> NDArray[np.int64]:
        """Return the rows holding these chunks."""
        return np.asarray([self._row_of[cid] for cid in chunk_ids], dtype=np.int64)

    def rows_for_doc(self, doc_id: str | None) -> NDArray[np.int64] | None:
        """Return the rows of one document, or ``None`` for corpus-wide search."""
        if doc_id is None:
            return None
        return self._rows_by_doc.get(str(doc_id), np.asarray([], dtype=np.int64))

    def rank(
        self,
        query_vector: _FloatArray,
        rows: NDArray[np.int64] | None,
        k: int,
    ) -> list[str]:
        """Return the top ``k`` chunk IDs for one query, restricted to ``rows``.

        Parameters
        ----------
        query_vector : numpy.ndarray
            One normalised query embedding.
        rows : numpy.ndarray or None
            Rows to consider, or ``None`` for the whole corpus.
        k : int
            Depth.

        Returns
        -------
        list of str
            Chunk identifiers, ties broken on the identifier ascending.
        """
        within = None if rows is None else [int(r) for r in rows]
        return [key for key, _ in self.store.search(query_vector, k, within=within)]

    def search(
        self,
        query_vectors: _FloatArray,
        doc_ids: Sequence[str | None] | None = None,
        k: int = 10,
    ) -> list[list[str]]:
        """Rank chunks for a batch of queries, in one round trip to the store.

        Parameters
        ----------
        query_vectors : numpy.ndarray
            One row per query.
        doc_ids : sequence or None, optional
            The document to scope each query to. ``None``, or a ``None`` entry,
            searches the whole corpus for that query.
        k : int, optional
            Depth.

        Returns
        -------
        list of list of str
            Ranked chunk identifiers, one list per query.
        """
        scopes: list[str | None] = (
            [None] * len(query_vectors)
            if doc_ids is None
            else [None if d is None else str(d) for d in doc_ids]
        )
        ranked = self.store.search_batch(query_vectors, k, scopes=scopes)
        return [[key for key, _ in hits] for hits in ranked]


def build_index(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    vectors: _FloatArray | None = None,
) -> tuple[DenseIndex, IndexCost]:
    """Embed chunks once, time it, and return the index.

    Parameters
    ----------
    embedder : Embedder
        Encodes ``Chunk.indexed_text`` as documents.
    chunks : sequence of Chunk
        The corpus.
    vectors : numpy.ndarray or None, optional
        Embeddings computed some other way. Late chunking supplies them, since
        its vectors come from pooling a document-level forward pass rather than
        from encoding each chunk alone.

    Returns
    -------
    tuple of (DenseIndex, IndexCost)
        The index and what it cost to build.
    """
    started = time.perf_counter()
    if vectors is None:
        vectors = np.asarray(
            embedder.encode([c.indexed_text for c in chunks]),
            dtype=np.float32,
        )
    elapsed = time.perf_counter() - started
    index = DenseIndex(
        chunk_ids=[c.chunk_id for c in chunks],
        vectors=np.asarray(vectors, dtype=np.float32),
        doc_ids=[c.doc_id for c in chunks],
    )
    return index, IndexCost(
        chunks=len(chunks),
        dimension=index.dimension,
        index_seconds=elapsed,
        chunks_per_second=len(chunks) / elapsed if elapsed else float("nan"),
        index_bytes=index.nbytes,
    )


def measure_queries(
    index: DenseIndex,
    encode_query: Callable[[str], _FloatArray],
    queries: Sequence[tuple[str, str, str | None]],
    *,
    k: int = 10,
    warmup: int = 5,
) -> tuple[dict[str, list[str]], QueryCost]:
    """Run ``(query_id, question, doc_id)`` triples one at a time and time them.

    One query per forward pass, so the latency column is a real single-shot
    number rather than a batched throughput figure divided by the batch size.

    Parameters
    ----------
    index : DenseIndex
        The index to search.
    encode_query : callable
        Takes a question, returns one query vector.
    queries : sequence of tuple
        ``(query_id, question, doc_id)``. A ``None`` document searches the whole
        corpus for that query.
    k : int, optional
        Depth.
    warmup : int, optional
        Queries encoded before timing starts, so a cold first pass does not
        land in the percentiles.

    Returns
    -------
    tuple of (dict, QueryCost)
        Ranked chunk identifiers per query, and the measured cost.
    """
    for _, text, _ in queries[:warmup]:
        encode_query(text)

    runs: dict[str, list[str]] = {}
    totals: list[float] = []
    encodes: list[float] = []
    scores: list[float] = []
    for query_id, text, doc_id in queries:
        started = time.perf_counter()
        vector = encode_query(text)
        encoded = time.perf_counter()
        runs[query_id] = index.rank(vector, index.rows_for_doc(doc_id), k)
        ranked = time.perf_counter()
        encodes.append((encoded - started) * 1000)
        scores.append((ranked - encoded) * 1000)
        totals.append((ranked - started) * 1000)

    ordered = sorted(totals)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return runs, QueryCost(
        queries=len(totals),
        median_ms=median(totals),
        mean_ms=sum(totals) / len(totals),
        p95_ms=p95,
        median_encode_ms=median(encodes),
        median_score_ms=median(scores),
    )
