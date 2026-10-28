"""Tests for the seams between the pipeline and the retrieval harness.

These are the only pipeline tests that import
``aieng.advanced_rag.retrieval`` and ``aieng.advanced_rag.evaluation.metrics``.
That half of the harness is being consolidated onto those paths by separate
work, so until it lands this whole module skips. Everything else in this
package runs without it.

What is checked here is the contract in both directions: chunks and queries go
in with the fields the harness names, and the pipeline's abstentions and ranked
identifiers come out in the shapes ``score_null_detection`` and
``score_retrieval`` read.
"""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.pipeline.grading import HeuristicGrader
from aieng.advanced_rag.pipeline.hierarchical import split_document, to_chunks
from aieng.advanced_rag.pipeline.spine import (
    AdvancedRagSpine,
    SpineConfig,
    abstentions,
    evidence_from_hits,
    retrieval_runs,
)


REASON = "the retrieval half of the harness is not merged yet"

corpus = pytest.importorskip("aieng.advanced_rag.retrieval.corpus", reason=REASON)
retrievers = pytest.importorskip(
    "aieng.advanced_rag.retrieval.retrievers", reason=REASON
)
metrics = pytest.importorskip("aieng.advanced_rag.evaluation.metrics", reason=REASON)


CONTRACT = (
    "ARTICLE I  DEFINITIONS\n"
    + "The term Agreement means this contract between Party A and Party B. " * 10
    + "\n\nARTICLE II  GOVERNING LAW\n"
    + "This Agreement is governed by the laws of the Province of Ontario. " * 10
)
GOVERNING_LAW = "governed by the laws of the Province of Ontario"


class KeywordRetriever(retrievers.ChunkStoreRetriever):
    """Ranks chunks containing a keyword first, so a run is deterministic.

    Inherits the shared interface, so it is passed to the spine as it is: the
    ``__call__`` the spine needs comes from the same mixin every retriever in
    the package uses.
    """

    name = "keyword"

    def __init__(self, chunks, keyword: str) -> None:
        super().__init__()
        self.chunks = list(chunks)
        self.keyword = keyword
        self.seen: list[object] = []
        self.index(self.chunks)

    def search(self, query, k=10):
        self.seen.append(query)
        scoped = [c for c in self.chunks if c.contract_id == query.contract_id]
        ranked = sorted(scoped, key=lambda c: self.keyword not in c.text)
        return [retrievers.Retrieved(chunk.chunk_id, 1.0) for chunk in ranked[:k]]


def hierarchy():
    return split_document(
        "doc1", CONTRACT, parent_chars=700, child_chars=200, overlap_chars=0
    )


def test_hierarchical_children_convert_to_harness_chunks():
    children = list(hierarchy().children)
    chunks = to_chunks(children)
    assert isinstance(chunks[0], corpus.Chunk)
    assert chunks[0].chunk_id == children[0].node_id
    assert chunks[0].contract_id == "doc1"
    assert chunks[0].span == (children[0].start, children[0].end)


def test_children_and_queries_build_a_harness_slice():
    children = list(hierarchy().children)
    chunks = to_chunks(children)
    span = corpus.AnswerSpan(
        text=GOVERNING_LAW,
        start=CONTRACT.index(GOVERNING_LAW),
        end=CONTRACT.index(GOVERNING_LAW) + len(GOVERNING_LAW),
    )
    query = corpus.Query(
        query_id="q1",
        doc_id="doc1",
        category="Governing Law",
        question="Which law governs the agreement?",
        answers=(span,),
    )
    gold = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.start < span.end and span.start < chunk.end
    }
    slice_ = corpus.Slice(chunks=chunks, queries=[query], qrels={"q1": gold})
    assert slice_.summary()["chunks"] == len(chunks)
    assert slice_.answerable == [query]
    assert gold


def test_the_interface_scopes_and_joins_by_chunk_id():
    chunks = to_chunks(list(hierarchy().children))
    retriever = KeywordRetriever(chunks, GOVERNING_LAW)

    evidence = retriever.retrieve("Which law governs the agreement?", 2, scope="doc1")

    assert isinstance(retriever.seen[0], corpus.Query)
    assert retriever.seen[0].contract_id == "doc1"
    assert retriever.seen[0].question == "Which law governs the agreement?"
    assert len(evidence) == 2
    assert GOVERNING_LAW in evidence[0].text
    assert evidence[0].retriever == "keyword"
    assert evidence[0].provenance.doc_id == "doc1"


def test_the_spines_callable_shape_needs_no_adapter():
    # What retiring harness_retrieve rests on: the retriever is the callable the
    # spine takes, at the retriever's own default depth.
    chunks = to_chunks(list(hierarchy().children))
    retriever = KeywordRetriever(chunks, GOVERNING_LAW)

    called = retriever("Which law governs the agreement?", "doc1")

    assert [e.chunk_id for e in called] == [
        e.chunk_id
        for e in retriever.retrieve(
            "Which law governs the agreement?", 10, scope="doc1"
        )
    ]
    assert GOVERNING_LAW in called[0].text


def test_the_callable_shape_forwards_the_depth_it_is_given():
    # The shared mixin used to drop the depth, so SpineConfig.top_k above the
    # mixin's own default of 10 never reached search().
    chunks = to_chunks(list(hierarchy().children))
    retriever = KeywordRetriever(chunks, GOVERNING_LAW)

    asked_for_two = retriever("Which law governs the agreement?", "doc1", 2)

    assert [e.chunk_id for e in asked_for_two] == [
        e.chunk_id
        for e in retriever.retrieve("Which law governs the agreement?", 2, scope="doc1")
    ]
    assert len(retriever("Which law governs the agreement?", "doc1")) > 2


def test_a_hit_on_a_chunk_the_adapter_never_saw_is_dropped():
    chunks = to_chunks(list(hierarchy().children))
    hits = [retrievers.Retrieved("doc1::p999::c999", 1.0)]
    assert evidence_from_hits(hits, {c.chunk_id: c for c in chunks}) == []


def test_the_pipelines_abstentions_score_as_null_detection():
    chunks = to_chunks(list(hierarchy().children))
    retriever = KeywordRetriever(chunks, GOVERNING_LAW)
    spine = AdvancedRagSpine(
        retriever,
        HeuristicGrader(),
        lambda prompt: json.dumps(
            {"answer": "Ontario law governs.", "sources": [1], "not_stated": False}
        ),
        config=SpineConfig(top_k=3, evidence_budget=2, max_attempts=1),
    )

    answered = spine.answer("q1", "Which law governs the agreement?", scope="doc1")
    # A question scoped to a contract that was never indexed retrieves nothing,
    # which is the cheap abstention path: no grading call, no generation call.
    unanswerable = spine.answer("q2", "What is the fee schedule?", scope="absent")

    flags = abstentions([answered, unanswerable])
    assert flags == {"q1": False, "q2": True}

    report = metrics.score_null_detection(flags, truly_null=["q2"])
    assert isinstance(report, metrics.NullReport)
    assert report.abstain_recall == 1.0
    assert report.abstain_precision == 1.0
    assert report.false_answer_rate == 0.0


def test_the_pipelines_runs_score_as_retrieval():
    chunks = to_chunks(list(hierarchy().children))
    start = CONTRACT.index(GOVERNING_LAW)
    gold = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.start < start + len(GOVERNING_LAW) and start < chunk.end
    }
    retriever = KeywordRetriever(chunks, GOVERNING_LAW)
    spine = AdvancedRagSpine(
        retriever,
        HeuristicGrader(),
        lambda prompt: json.dumps({"answer": "Ontario.", "sources": [1]}),
        config=SpineConfig(top_k=5, evidence_budget=2, max_attempts=1),
    )

    result = spine.answer("q1", "Which law governs the agreement?", scope="doc1")
    report = metrics.score_retrieval(retrieval_runs([result]), {"q1": gold}, k=5)

    assert isinstance(report, metrics.RetrievalReport)
    assert report.retrieval_failure_rate == 0.0
    assert report.queries == 1
