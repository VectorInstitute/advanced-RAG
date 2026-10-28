"""Retriever behaviour, including the end-to-end sanity check.

The last test is the one that says the harness measures something: a retriever
with lexical signal has to beat a shuffled one on the same candidate pool, and
the shuffled one has to show a high failure rate. If both scored the same, the
metric would be reporting the size of the pool rather than retrieval quality.
"""

from __future__ import annotations

from aieng.advanced_rag.evaluation.metrics import score_retrieval
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query
from aieng.advanced_rag.retrieval.embedders import HashEmbedder
from aieng.advanced_rag.retrieval.retrievers import (
    DenseRetriever,
    ShuffledRetriever,
    run_retriever,
)


def make_chunks(doc: str, texts: list[str]) -> list[Chunk]:
    chunks, cursor = [], 0
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{doc}::{index:04d}",
                doc_id=doc,
                text=text,
                start=cursor,
                end=cursor + len(text),
            )
        )
        cursor += len(text)
    return chunks


class RecordingEmbedder:
    """A hashed embedder that records which side each text arrived on."""

    name = "recording"

    def __init__(self) -> None:
        self._inner = HashEmbedder(dimension=128)
        self.documents: list[str] = []
        self.queries: list[str] = []

    def dimension(self) -> int:
        return self._inner.dimension()

    def encode(self, texts):
        self.documents.extend(texts)
        return self._inner.encode(texts)

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return self._inner.encode(texts)


class TestDenseRetriever:
    def setup_method(self):
        self.chunks = make_chunks(
            "C1",
            [
                "This agreement is governed by the laws of Delaware.",
                "Payment terms are net thirty days after invoice.",
                "The supplier shall maintain product liability insurance.",
            ],
        )
        self.retriever = DenseRetriever(HashEmbedder(dimension=512))
        self.retriever.index(self.chunks)

    def test_ranks_the_matching_chunk_first(self):
        query = Query("q1", "C1", "Governing Law", "governed by the laws of Delaware")
        assert self.retriever.search(query, k=3)[0].chunk_id == "C1::0000"

    def test_k_bounds_the_result_count(self):
        query = Query("q1", "C1", "Governing Law", "insurance")
        assert len(self.retriever.search(query, k=2)) == 2

    def test_scoped_to_the_query_document(self):
        # Chunks from another document are indexed but must never be returned.
        self.retriever.index(make_chunks("C2", ["governed by the laws of Delaware"]))
        query = Query("q1", "C1", "Governing Law", "governed by the laws of Delaware")
        assert all(
            hit.chunk_id.startswith("C1::") for hit in self.retriever.search(query, k=5)
        )

    def test_unknown_document_returns_nothing(self):
        query = Query("q1", "MISSING", "Governing Law", "anything")
        assert self.retriever.search(query, k=5) == []

    def test_ranking_is_stable_for_tied_scores(self):
        query = Query("q1", "C1", "x", "zzz no overlap at all")
        first = [h.chunk_id for h in self.retriever.search(query, k=3)]
        second = [h.chunk_id for h in self.retriever.search(query, k=3)]
        assert first == second == sorted(first)


class TestQuerySideEncoding:
    def test_queries_go_through_the_query_side_and_chunks_do_not(self):
        # The BGE query prompt only lands on queries because the retriever asks
        # for encode_queries. Without this, an asymmetric model is run wrong.
        embedder = RecordingEmbedder()
        retriever = DenseRetriever(embedder)
        retriever.index(make_chunks("C1", ["a chunk of text"]))
        retriever.search(Query("q", "C1", "cat", "a question"), k=1)
        assert embedder.documents == ["a chunk of text"]
        assert embedder.queries == ["a question"]

    def test_an_embedder_without_a_query_side_still_works(self):
        # The protocol only requires encode; encode_queries is optional.
        class DocumentOnly:
            name = "document-only"

            def dimension(self):
                return 128

            def encode(self, texts):
                return HashEmbedder(dimension=128).encode(texts)

        retriever = DenseRetriever(DocumentOnly())
        retriever.index(make_chunks("C1", ["governed by the laws of Delaware"]))
        hits = retriever.search(Query("q", "C1", "cat", "governing law"), k=1)
        assert hits[0].chunk_id == "C1::0000"

    def test_the_indexed_text_is_what_gets_embedded(self):
        # Contextual retrieval sets embed_text and the span stays the judgment.
        embedder = RecordingEmbedder()
        retriever = DenseRetriever(embedder)
        chunk = make_chunks("C1", ["the clause"])[0]
        retriever.index([chunk.with_embed_text("From the Acme agreement. the clause")])
        assert embedder.documents == ["From the Acme agreement. the clause"]


class TestShuffledRetriever:
    def test_deterministic_across_instances(self):
        chunks = make_chunks("C1", [f"clause {i}" for i in range(20)])
        first, second = ShuffledRetriever(seed=7), ShuffledRetriever(seed=7)
        first.index(chunks)
        second.index(chunks)
        query = Query("q", "C1", "cat", "question")
        assert [h.chunk_id for h in first.search(query, k=5)] == [
            h.chunk_id for h in second.search(query, k=5)
        ]

    def test_it_only_returns_the_query_s_own_document(self):
        control = ShuffledRetriever(seed=0)
        control.index(make_chunks("C1", ["a", "b"]) + make_chunks("C2", ["c", "d"]))
        hits = control.search(Query("q", "C1", "cat", "question"), k=10)
        assert {h.chunk_id for h in hits} == {"C1::0000", "C1::0001"}


class TestHarnessEndToEnd:
    """The sanity check, on a synthetic corpus.

    Twenty documents, forty chunks each, with the gold clause hidden in one of
    them. Retrieving 5 of 40 at random should find it around 12% of the time,
    so a shuffled control must show a failure rate near 0.88 while a retriever
    with real lexical signal must be far below that.
    """

    def _fixture(self):
        chunks, queries, qrels = [], {}, {}
        for doc in range(20):
            doc_id = f"K{doc:02d}"
            texts = [
                f"boilerplate clause {i} concerning delivery schedules"
                for i in range(40)
            ]
            texts[doc % 40] = (
                "This agreement shall be governed by and construed under the "
                "laws of Delaware."
            )
            doc_chunks = make_chunks(doc_id, texts)
            chunks.extend(doc_chunks)

            query_id = f"{doc_id}__gov"
            queries[query_id] = Query(
                query_id,
                doc_id,
                "Governing Law",
                "governed by and construed under the laws of Delaware",
                answers=(AnswerSpan("laws of Delaware", 0, 10),),
            )
            qrels[query_id] = {doc_chunks[doc % 40].chunk_id}
        return chunks, list(queries.values()), qrels

    def test_dense_beats_the_shuffled_control(self):
        chunks, queries, qrels = self._fixture()

        dense = DenseRetriever(HashEmbedder(dimension=512))
        dense.index(chunks)
        control = ShuffledRetriever(seed=0)
        control.index(chunks)

        dense_report = score_retrieval(run_retriever(dense, queries, k=5), qrels, k=5)
        control_report = score_retrieval(
            run_retriever(control, queries, k=5), qrels, k=5
        )

        # The control is near the 1 - 5/40 = 0.875 chance floor.
        assert control_report.retrieval_failure_rate > 0.7
        # Lexical signal finds a verbatim clause every time.
        assert dense_report.retrieval_failure_rate == 0.0
        assert dense_report.mrr > control_report.mrr
