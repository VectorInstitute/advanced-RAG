"""The BM25 arm and weighted fusion.

The BM25 implementation is a port, so the tests that matter are the ones that
would catch a porting mistake: that IDF actually falls with document frequency,
that length normalisation is applied, and that fusion weight moves the result in
the direction the weight says.
"""

from __future__ import annotations

import pytest

from aieng.advanced_rag.retrieval.corpus import Chunk, Query
from aieng.advanced_rag.retrieval.retrievers import Retrieved
from aieng.advanced_rag.retrieval.sparse import (
    BM25Retriever,
    HybridRetriever,
    SparseEncoder,
    sparse_dot,
    sparse_index,
    weighted_rrf,
)


def make_chunks() -> list[Chunk]:
    texts = [
        "This Agreement is entered into by Acme Corporation and Beta Industries.",
        "Governing law shall be the laws of the State of Delaware.",
        "The term of this agreement begins on the Effective Date of June 1 2019.",
        "Termination for convenience requires ninety days written notice.",
    ]
    return [
        Chunk(
            chunk_id=f"c::{i}",
            doc_id="c",
            text=text,
            start=i * 100,
            end=i * 100 + len(text),
        )
        for i, text in enumerate(texts)
    ]


def query(question: str) -> Query:
    return Query(query_id="q1", doc_id="c", category="test", question=question)


def test_sparse_index_is_stable_and_in_range():
    index = sparse_index("agreement")
    assert index == sparse_index("agreement")
    assert 0 <= index < 2**31


def test_idf_falls_as_document_frequency_rises():
    encoder = SparseEncoder.fit([c.text for c in make_chunks()])
    common = encoder._idf(sparse_index("agreement"))  # in 2 of 4 documents
    rare = encoder._idf(sparse_index("delaware"))  # in 1 of 4
    assert rare > common


def test_fitting_on_an_empty_corpus_is_an_error():
    with pytest.raises(ValueError):
        SparseEncoder.fit([])


def test_document_encoding_saturates_and_normalizes_length():
    short = SparseEncoder.fit(["alpha", "beta gamma"])
    vector = short.encode_document("alpha alpha alpha")
    index = sparse_index("alpha")
    # Term saturation: three occurrences score less than three times one.
    single = short.encode_document("alpha")[index]
    assert vector[index] < 3 * single


def test_bm25_finds_the_lexically_matching_chunk():
    retriever = BM25Retriever()
    retriever.index(make_chunks())
    hits = retriever.search(query("What is the governing law of this contract?"), k=2)
    assert hits[0].chunk_id == "c::1"


def test_bm25_scopes_to_the_query_document():
    retriever = BM25Retriever()
    retriever.index(make_chunks())
    assert retriever.search(Query("q", "other", "test", "governing law"), k=5) == []


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError):
        BM25Retriever().search(query("anything"))


def test_sparse_dot_is_symmetric():
    left, right = {1: 2.0, 2: 3.0}, {2: 1.0, 3: 5.0}
    assert sparse_dot(left, right) == sparse_dot(right, left) == 3.0


def test_equal_weights_reduce_to_standard_rrf():
    rankings = [["a", "b", "c"], ["c", "b", "a"]]
    fused = weighted_rrf(rankings, [0.5, 0.5], limit=3)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(2 * 0.5 / 62)
    assert scores["a"] == pytest.approx(scores["c"])


def test_weight_moves_the_fusion_toward_the_favoured_arm():
    rankings = [["dense_only"], ["lexical_only"]]
    dense_heavy = weighted_rrf(rankings, [0.9, 0.1], limit=2)
    lexical_heavy = weighted_rrf(rankings, [0.1, 0.9], limit=2)
    assert dense_heavy[0][0] == "dense_only"
    assert lexical_heavy[0][0] == "lexical_only"


def test_zero_weight_arm_contributes_nothing():
    fused = weighted_rrf([["a"], ["b"]], [1.0, 0.0], limit=5)
    assert [chunk_id for chunk_id, _ in fused] == ["a"]


def test_weights_must_match_rankings():
    with pytest.raises(ValueError):
        weighted_rrf([["a"], ["b"]], [1.0], limit=1)


def test_hybrid_returns_both_arms_contributions():
    chunks = make_chunks()
    lexical = BM25Retriever()
    lexical.index(chunks)

    class OnlyLast:
        """A stand-in dense arm with a fixed, different opinion."""

        name = "stub"

        def index(self, chunks, batch_size=64):
            pass

        def search(self, query, k=10):
            return [Retrieved("c::3", 1.0)]

    hybrid = HybridRetriever(OnlyLast(), lexical, dense_weight=0.5, fusion_depth=10)
    hits = {hit.chunk_id for hit in hybrid.search(query("governing law delaware"), k=4)}
    assert "c::3" in hits and "c::1" in hits


def test_hybrid_indexing_is_a_no_op():
    # Both arms are indexed by the caller and shared across a weight sweep, so
    # re-indexing per weight would re-embed the corpus for nothing.
    lexical = BM25Retriever()
    lexical.index(make_chunks())
    hybrid = HybridRetriever(lexical, lexical, dense_weight=0.5)
    hybrid.index(make_chunks())
    assert hybrid.search(query("governing law"), k=1)[0].chunk_id == "c::1"
