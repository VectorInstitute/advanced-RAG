"""The SPLADE and ColBERT retrievers, with fake encoders so no model downloads.

What these tests can catch is the wiring: scope, tie-breaking, the MaxSim
arithmetic, the query path, and the interface. They cannot say the models rank
well. The published-figure guards do that against the real models.
"""

from __future__ import annotations

import numpy as np
import pytest

from aieng.advanced_rag.retrieval.conformance import check_retrieval
from aieng.advanced_rag.retrieval.corpus import Chunk, Query
from aieng.advanced_rag.retrieval.representations import (
    ColbertRetriever,
    SpladeRetriever,
    max_sim,
)


WIDTH = 16


def words(text: str) -> list[str]:
    return [w.strip(".,?").lower() for w in text.split() if w.strip(".,?")]


def slot(word: str) -> int:
    return sum(ord(c) for c in word) % WIDTH


class FakeSparse:
    """One weight per word, so overlap is the dot product."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts):
        self.document_calls += 1
        return [{slot(w): 1.0 for w in words(t)} for t in texts]

    def encode_query(self, text):
        self.query_calls += 1
        return {slot(w): 1.0 for w in words(text)}


class FakeMultiVector:
    """One one-hot token vector per word."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    @staticmethod
    def matrix(text):
        rows = np.zeros((max(len(words(text)), 1), WIDTH), dtype=np.float32)
        for row, word in enumerate(words(text)):
            rows[row, slot(word)] = 1.0
        return rows

    def encode_documents(self, texts):
        return [self.matrix(t) for t in texts]

    def encode_query(self, text):
        self.queries.append(text)
        return self.matrix(text)


def make_chunks(doc_id="c"):
    texts = [
        "Governing law is the law of Delaware.",
        "Termination needs ninety days notice.",
        "The effective date is June first.",
    ]
    return [
        Chunk(
            chunk_id=f"{doc_id}::{i}",
            doc_id=doc_id,
            text=text,
            start=i * 100,
            end=i * 100 + len(text),
        )
        for i, text in enumerate(texts)
    ]


def query(question, doc_id="c"):
    return Query(query_id="q", doc_id=doc_id, category="t", question=question)


@pytest.fixture(params=["splade", "colbert"])
def retriever(request):
    if request.param == "splade":
        return SpladeRetriever(FakeSparse())
    return ColbertRetriever(FakeMultiVector())


def test_each_ranks_the_matching_chunk_first(retriever):
    retriever.index(make_chunks())
    hits = retriever.search(query("governing law Delaware"), k=3)
    assert hits[0].chunk_id == "c::0"
    assert hits[0].score >= hits[-1].score


def test_each_scopes_to_the_query_document(retriever):
    retriever.index(make_chunks("c") + make_chunks("d"))
    hits = retriever.search(query("governing law", doc_id="d"), k=10)
    assert {h.chunk_id.split("::")[0] for h in hits} == {"d"}
    assert retriever.search(query("governing law", doc_id="none")) == []


def test_each_breaks_ties_on_chunk_id(retriever):
    retriever.index(make_chunks())
    hits = retriever.search(query("governing law Delaware"), k=3)
    assert [(-h.score, h.chunk_id) for h in hits] == sorted(
        (-h.score, h.chunk_id) for h in hits
    )
    tied = [h.chunk_id for h in hits if h.score == hits[-1].score]
    assert tied == sorted(tied)


def test_search_before_index_is_an_error(retriever):
    with pytest.raises(RuntimeError, match="index"):
        retriever.search(query("anything"))


def test_each_conforms_to_the_retriever_interface(retriever):
    report = check_retrieval(
        retriever, "governing law Delaware", chunks=make_chunks(), k=2, scope="c"
    )
    assert report.ok, report.summary()


def test_the_encoder_is_built_once_and_chunks_are_encoded_in_one_call():
    encoder = FakeSparse()
    retriever = SpladeRetriever(encoder)
    retriever.index(make_chunks())
    retriever.search(query("law"))
    retriever.search(query("notice"))
    assert encoder.document_calls == 1
    assert encoder.query_calls == 2


def test_colbert_encodes_the_question_through_the_query_path():
    encoder = FakeMultiVector()
    retriever = ColbertRetriever(encoder)
    retriever.index(make_chunks())
    retriever.search(query("governing law"))
    assert encoder.queries == ["governing law"]


def test_max_sim_takes_the_best_document_token_per_query_token():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert max_sim(q, d) == pytest.approx(1.0 + 0.5)


def test_max_sim_is_zero_for_an_empty_side_and_rejects_bad_shapes():
    assert max_sim(np.zeros((0, 2)), np.ones((3, 2))) == 0.0
    with pytest.raises(ValueError, match="2-D"):
        max_sim(np.ones(3), np.ones((3, 3)))
    with pytest.raises(ValueError, match="width"):
        max_sim(np.ones((2, 3)), np.ones((2, 4)))


def test_a_missing_extra_fails_at_index_not_at_import(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no fastembed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match="fastembed"):
        SpladeRetriever().index(make_chunks())
