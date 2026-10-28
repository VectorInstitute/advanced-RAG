"""The numpy index: ranking, tie-breaking, scoping, cost, and agreement.

The last test is the one that keeps the two dense implementations honest. A
sweep uses the matrix index for speed and reports its numbers next to a
DenseRetriever baseline, so the two have to rank identically, tie-breaks
included.
"""

from __future__ import annotations

import numpy as np
import pytest

from aieng.advanced_rag.retrieval.chunkers import fixed_chars
from aieng.advanced_rag.retrieval.corpus import Chunk, Query
from aieng.advanced_rag.retrieval.embedders import HashEmbedder
from aieng.advanced_rag.retrieval.index import (
    DenseIndex,
    IndexCost,
    build_index,
    measure_queries,
)
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever


def unit(*values: float) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def array(*rows: list[float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class TestRanking:
    def test_rank_orders_by_cosine_similarity(self):
        index = DenseIndex(["b", "a", "c"], array(unit(0, 1), unit(1, 0), unit(1, 1)))
        query = np.asarray(unit(1, 0), dtype=np.float32)
        assert index.rank(query, None, 3) == ["a", "c", "b"]

    def test_rows_are_sorted_by_chunk_id(self):
        index = DenseIndex(["z", "a", "m"], np.eye(3, dtype=np.float32))
        assert index.chunk_ids == ["a", "m", "z"]
        # The vector originally paired with "a" must still be paired with "a".
        row = index.rows_for(["a"])[0]
        assert index.matrix[row].tolist() == [0.0, 1.0, 0.0]

    def test_ties_break_on_chunk_id_ascending(self):
        vectors = array(*([unit(1, 0)] * 4))
        index = DenseIndex(["d::0003", "d::0001", "d::0002", "d::0000"], vectors)
        assert index.rank(np.asarray(unit(1, 0), dtype=np.float32), None, 4) == [
            "d::0000",
            "d::0001",
            "d::0002",
            "d::0003",
        ]

    def test_ties_break_on_chunk_id_when_k_is_smaller_than_the_pool(self):
        # The partitioned path, and the one that used to be wrong. Selecting
        # the top k of a large pool leaves the selection in arbitrary order, so
        # sorting it by score alone does not put tied chunks in ID order. The
        # pool here is 954, the size of the 20-contract CUAD slice, because the
        # reordering only shows up once the pool is large enough for numpy to
        # partition rather than sort.
        size = 954
        ids = [f"d::{i:04d}" for i in range(size)]
        index = DenseIndex(ids, np.tile(array(unit(1, 0)), (size, 1)))
        ranked = index.rank(np.asarray(unit(1, 0), dtype=np.float32), None, 10)
        assert ranked == ids[:10]

    def test_ties_break_on_chunk_id_inside_a_scoped_search(self):
        # A scoped search hands over rows in corpus order rather than sorted
        # order, so the row index has to be part of the tie-break too.
        size = 954
        ids = [f"d::{i:04d}" for i in range(size)]
        index = DenseIndex(ids, np.tile(array(unit(1, 0)), (size, 1)), ["d"] * size)
        ranked = index.rank(
            np.asarray(unit(1, 0), dtype=np.float32), index.rows_for_doc("d"), 10
        )
        assert ranked == ids[:10]

    def test_rank_respects_row_scoping(self):
        index = DenseIndex(
            ["a", "b", "c"], array(unit(1, 0), unit(0, 1), unit(1, 0.01))
        )
        query = np.asarray(unit(1, 0), dtype=np.float32)
        rows = index.rows_for(["b", "c"])
        assert index.rank(query, rows, 2) == ["c", "b"]
        assert "a" not in index.rank(query, rows, 2)

    def test_rank_on_empty_scope_returns_nothing(self):
        index = DenseIndex(["a"], array(unit(1, 0)))
        empty = np.asarray([], dtype=np.int64)
        assert index.rank(np.asarray(unit(1, 0), dtype=np.float32), empty, 5) == []

    def test_rank_k_larger_than_corpus_returns_everything(self):
        index = DenseIndex(["a", "b"], array(unit(1, 0), unit(0, 1)))
        query = np.asarray(unit(1, 1), dtype=np.float32)
        assert len(index.rank(query, None, 10)) == 2

    def test_mismatched_ids_and_vectors_raise(self):
        with pytest.raises(ValueError, match="chunk ids"):
            DenseIndex(["a", "b"], np.eye(3, dtype=np.float32))


class TestScopedSearch:
    def _index(self) -> DenseIndex:
        return DenseIndex(
            ["alpha::0", "alpha::1", "beta::0"],
            array(unit(1, 0), unit(0.9, 0.1), unit(0, 1)),
            ["alpha", "alpha", "beta"],
        )

    def test_scoping_never_leaks_across_documents(self):
        index = self._index()
        ranked = index.search(array(unit(0, 1)), ["alpha"], k=10)[0]
        assert ranked and all(cid.startswith("alpha::") for cid in ranked)

    def test_a_none_scope_searches_the_whole_corpus(self):
        index = self._index()
        assert index.search(array(unit(0, 1)), [None], k=1)[0] == ["beta::0"]

    def test_omitting_the_scope_searches_the_whole_corpus(self):
        index = self._index()
        assert index.search(array(unit(0, 1)), k=1)[0] == ["beta::0"]

    def test_an_unknown_document_returns_nothing(self):
        index = self._index()
        assert index.search(array(unit(1, 0)), ["missing"], k=5)[0] == []


class TestBuildIndex:
    def test_reports_measured_cost(self):
        chunks = [Chunk(f"c::{i}", "d", "x" * (i + 1), 0, 1) for i in range(3)]
        index, cost = build_index(HashEmbedder(dimension=2), chunks)
        assert isinstance(cost, IndexCost)
        assert cost.chunks == 3
        assert cost.dimension == 2
        # float32, 3 vectors of 2 dimensions.
        assert cost.index_bytes == 24
        assert cost.chunks_per_second > 0
        # index_mib is the rounded convenience field; index_bytes is exact.
        assert cost.to_dict()["index_mib"] == round(24 / 1024 / 1024, 3)
        assert index.chunk_ids == ["c::0", "c::1", "c::2"]

    def test_supplied_vectors_skip_the_encoder(self):
        # Late chunking pools a document-level forward pass, so its vectors do
        # not come from encoding each chunk alone.
        chunks = [Chunk("c::0", "d", "text", 0, 4)]
        index, cost = build_index(
            HashEmbedder(dimension=2), chunks, vectors=array(unit(1, 0))
        )
        assert index.matrix.tolist() == [[1.0, 0.0]]
        assert cost.chunks == 1


class TestMeasureQueries:
    def test_it_ranks_and_times_every_query(self):
        index = DenseIndex(
            ["alpha::0", "beta::0"], array(unit(1, 0), unit(0, 1)), ["alpha", "beta"]
        )

        def encode_query(text: str) -> np.ndarray:
            return np.asarray(unit(1, 0) if text == "a" else unit(0, 1), np.float32)

        runs, cost = measure_queries(
            index,
            encode_query,
            [("q1", "a", "alpha"), ("q2", "b", None)],
            k=1,
            warmup=0,
        )
        assert runs == {"q1": ["alpha::0"], "q2": ["beta::0"]}
        assert cost.queries == 2
        assert cost.median_ms >= 0.0
        assert set(cost.to_dict()) >= {"median_encode_ms", "median_score_ms", "p95_ms"}


DOCS = {
    "alpha": "Termination for convenience is permitted on thirty days notice. "
    "Payment terms are net thirty. " * 12,
    "beta": "Governing law is the law of Ontario. Assignment requires consent. " * 12,
}
QUESTIONS = ["termination for convenience", "governing law", "payment terms"]


def test_the_matrix_index_ranks_exactly_as_the_reference_retriever() -> None:
    chunks: list[Chunk] = []
    for doc_id, text in DOCS.items():
        chunks.extend(fixed_chars(doc_id, text, chunk_chars=300, overlap_chars=50))

    embedder = HashEmbedder(dimension=128)
    index = DenseIndex(
        [c.chunk_id for c in chunks],
        np.asarray(embedder.encode([c.indexed_text for c in chunks]), dtype=np.float32),
        [c.doc_id for c in chunks],
    )
    reference = DenseRetriever(embedder)
    reference.index(chunks)

    for doc_id in DOCS:
        for question in QUESTIONS:
            query = Query(f"{doc_id}:{question}", doc_id, "cat", question)
            vectors = np.asarray(embedder.encode([question]), dtype=np.float32)
            mine = index.search(vectors, [doc_id], k=5)[0]
            theirs = [hit.chunk_id for hit in reference.search(query, k=5)]
            assert mine == theirs
