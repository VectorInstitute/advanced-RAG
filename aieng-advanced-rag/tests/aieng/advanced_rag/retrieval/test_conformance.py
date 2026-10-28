"""The conformance kit, run over every retriever and adapter in the package.

This module is thin on purpose. The checks live in
``aieng.advanced_rag.retrieval.conformance`` as library code, so a reference
implementation points the same kit at its own retriever with its own corpus.
What is here is the list of implementations the package ships, the stubs the two
framework adapters need to run without a framework, and the guard that those
stubs previously made impossible. The stubs return the identifiers the installed
libraries return, which is not the identifier space the corpus is scored in, so a
graph adapter that stopped translating them would report recall 0.0 for passages
it had retrieved and a test here would fail.
"""

from __future__ import annotations

import asyncio

import pytest

from aieng.advanced_rag.evaluation.metrics import score_retrieval
from aieng.advanced_rag.graphrag.adapters.dense import DenseRetriever as GraphDense
from aieng.advanced_rag.graphrag.adapters.extracted_graph import (
    ExtractedGraphRetriever,
)
from aieng.advanced_rag.graphrag.adapters.hipporag2 import HippoRAG2Retriever
from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever
from aieng.advanced_rag.graphrag.evidence import to_runs
from aieng.advanced_rag.retrieval.conformance import (
    CHECKS,
    NOT_CHECKED,
    _index,
    acheck_retrieval,
    assert_conforms,
    check_retrieval,
)
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.embedders import HashEmbedder
from aieng.advanced_rag.retrieval.interface import ScopeUnsupported
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever, ShuffledRetriever
from aieng.advanced_rag.retrieval.sparse import BM25Retriever, HybridRetriever


QUESTION = "which law governs the agreement"
TEXTS = [
    "This agreement is governed by the laws of Delaware and nothing else.",
    "Payment terms are net thirty days after the invoice date.",
    "The supplier shall maintain product liability insurance at all times.",
    "Either party may terminate this agreement on ninety days notice.",
]


def make_chunks(doc: str = "C1") -> list[Chunk]:
    chunks, cursor = [], 0
    for index, text in enumerate(TEXTS):
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


#: The identifier space the harness scores in: one id per corpus passage.
CORPUS_IDS = {chunk.chunk_id for chunk in make_chunks()}


def word_encoder(texts):
    """Encode with a tiny deterministic bag of words, so no model is needed."""
    vocabulary = ("governed", "laws", "delaware", "payment", "insurance")
    return [[float(text.lower().count(word)) for word in vocabulary] for text in texts]


class StubUpstreamHippo:
    """``atlas_rag``'s retriever, identifiers included.

    Upstream returns two parallel lists, and each entry of the second one is
    ``text_id_to_node_name[passage_id]``. For a passage node that value is the
    node's ``id`` attribute, which the graph build sets to the passage text, so
    the identifier a caller can score on is the node key and only this map
    recovers it.
    """

    def __init__(self) -> None:
        # Passage node key to node name, built exactly as upstream builds it from
        # the graph: the key is the corpus id, the value is the passage text.
        self.text_id_to_node_name = {
            chunk.chunk_id: chunk.text for chunk in make_chunks()
        }

    def retrieve(self, query, topN=5, **kwargs):  # noqa: N803 - upstream spelling
        chunks = make_chunks()[:topN]
        return (
            [chunk.text for chunk in chunks],
            [self.text_id_to_node_name[chunk.chunk_id] for chunk in chunks],
        )


class StubLightRAG:
    """The ``aquery_data`` payload the adapter reads, with nothing installed.

    The identifiers are LightRAG's own. ``chunk_id`` is a document hash and an
    ordinal, in no corpus's identifier space, and ``file_path`` carries the value
    each document was inserted with, which is the only corpus identifier LightRAG
    reports. One document per passage, which is how a passage-level corpus goes
    in.

    Parameters
    ----------
    file_paths : bool, optional
        ``False`` is the same index built without ``file_paths=``, where LightRAG
        has no corpus identifier to report and writes ``"unknown_source"``.
    """

    def __init__(self, file_paths: bool = True) -> None:
        self.params = []
        self._file_paths = file_paths

    async def aquery_data(self, query, param):
        self.params.append(param)
        return {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "entities": [{"entity_name": "delaware", "reference_id": "1"}],
                "relationships": [],
                "chunks": [
                    {
                        "content": chunk.text,
                        "chunk_id": f"doc-{index:04x}beef-chunk-000",
                        "file_path": chunk.chunk_id
                        if self._file_paths
                        else "unknown_source",
                        "reference_id": f"{index + 1}",
                    }
                    for index, chunk in enumerate(make_chunks())
                ],
            },
        }


def harness_arms():
    """Return every harness retriever, indexed over the same chunks."""
    chunks = make_chunks()
    dense = DenseRetriever(HashEmbedder(dimension=128))
    dense.index(chunks)
    lexical = BM25Retriever()
    lexical.index(chunks)
    control = ShuffledRetriever(seed=0)
    control.index(chunks)
    hybrid = HybridRetriever(dense, lexical, dense_weight=0.5, fusion_depth=10)
    return {"dense": dense, "bm25": lexical, "control": control, "hybrid": hybrid}


@pytest.mark.parametrize("arm", list(harness_arms()))
def test_every_harness_retriever_conforms(arm):
    assert_conforms(harness_arms()[arm], QUESTION, k=2, scope="C1")


def test_the_graph_dense_control_conforms():
    retriever = GraphDense(word_encoder, name="graph-dense")
    assert_conforms(retriever, QUESTION, chunks=make_chunks(), k=2, scope="C1")


def test_the_extracted_graph_retriever_conforms():
    entity = type(
        "E", (), {"name": "delaware", "chunks": ["C1::0000"], "documents": []}
    )
    # Capability checks are off for this one: its ``expand`` walks entity
    # neighbours and returns weights, which is the older of the two meanings of
    # the word and not the offset widening Expandable describes.
    assert_conforms(
        ExtractedGraphRetriever(
            [entity()], [], {chunk.chunk_id: chunk.text for chunk in make_chunks()}
        ),
        "delaware",
        k=2,
        scope="C1",
        capabilities=(),
    )


def test_the_hipporag2_adapter_conforms():
    assert_conforms(HippoRAG2Retriever(StubUpstreamHippo()), QUESTION, k=2, scope="C1")


def test_the_lightrag_adapter_conforms_through_its_async_twin():
    pytest.importorskip("lightrag", reason="the lightrag extra is not installed")
    retriever = LightRAGRetriever(StubLightRAG(), mode="mix")
    report = asyncio.run(acheck_retrieval(retriever, QUESTION, k=2, scope="C1"))
    assert report.ok, report.summary()


def test_the_lightrag_adapter_also_conforms_synchronously():
    pytest.importorskip("lightrag", reason="the lightrag extra is not installed")
    retriever = LightRAGRetriever(StubLightRAG(), mode="mix")
    report = check_retrieval(retriever, QUESTION, k=2, scope="C1")
    assert report.ok, report.summary()


class TestGraphIdentifiersStayScoreable:
    """``to_runs`` has to yield identifiers the corpus recognises.

    This is the guard on the defect both graph adapters shipped with: each
    reported identifiers from its own library's space, which no corpus knows, so
    recall came out 0.0 for an arm that had retrieved every gold passage. Nothing
    raised, so only a check on the identifiers themselves catches it.
    """

    def _lightrag_runs(self, **kwargs):
        retriever = LightRAGRetriever(StubLightRAG(**kwargs), mode="mix")
        return to_runs(asyncio.run(retriever.aretrieve(QUESTION, 4)))

    def test_hipporag2_reports_identifiers_from_the_corpus(self):
        runs = to_runs(HippoRAG2Retriever(StubUpstreamHippo()).retrieve(QUESTION, 4))
        assert runs
        assert set(runs) <= CORPUS_IDS

    def test_lightrag_reports_identifiers_from_the_corpus(self):
        runs = self._lightrag_runs()
        assert runs
        assert set(runs) <= CORPUS_IDS

    @pytest.mark.parametrize("arm", ["hipporag2", "lightrag"])
    def test_recall_is_one_when_the_arm_retrieved_every_gold_passage(self, arm):
        gold = {chunk.chunk_id for chunk in make_chunks()[:2]}
        if arm == "hipporag2":
            runs = to_runs(
                HippoRAG2Retriever(StubUpstreamHippo()).retrieve(QUESTION, 4)
            )
        else:
            runs = self._lightrag_runs()
        report = score_retrieval({"q1": runs}, {"q1": gold}, k=4)
        assert report.recall_at_k == 1.0

    def test_lightrag_keeps_its_own_chunk_id_for_debugging(self):
        retriever = LightRAGRetriever(StubLightRAG(), mode="mix")
        item = asyncio.run(retriever.aretrieve(QUESTION, 1))[0]
        assert item.provenance.extra["lightrag_chunk_id"].startswith("doc-")

    def test_lightrag_reports_its_own_ids_when_nothing_better_exists(self):
        # An index built without ``file_paths=`` carries no corpus identifier, so
        # LightRAG's own chunk id is all there is to report.
        retriever = LightRAGRetriever(StubLightRAG(file_paths=False), mode="mix")
        evidence = asyncio.run(retriever.aretrieve(QUESTION, 4))
        assert to_runs(evidence) == [
            item.provenance.extra["lightrag_chunk_id"] for item in evidence
        ]

    def test_hipporag2_reports_what_upstream_gave_without_a_map(self):
        # A stand-in retriever that already reports identifiers keeps working.
        class NoMap:
            def retrieve(self, query, topN=5):  # noqa: N803 - upstream spelling
                return ["a passage"], ["C1::0002"]

        assert to_runs(HippoRAG2Retriever(NoMap()).retrieve(QUESTION, 1)) == [
            "C1::0002"
        ]

    def test_k_sets_lightrag_chunk_budget_as_well_as_its_entity_budget(self):
        # LightRAG counts ``top_k`` in entities and ``chunk_top_k`` in chunks, so
        # ``k`` only means passages when both are set. Left alone, the chunk
        # budget stays at LightRAG's default of 20.
        pytest.importorskip("lightrag", reason="the lightrag extra is not installed")
        rag = StubLightRAG()
        asyncio.run(LightRAGRetriever(rag, mode="mix").aretrieve(QUESTION, 3))
        assert (rag.params[0].top_k, rag.params[0].chunk_top_k) == (3, 3)


def test_a_scope_raises_rather_than_returning_fewer_than_k():
    # The whole reason ScopeUnsupported exists: a graph adapter that filtered a
    # finished ranking would hand back fewer than k and stop k meaning what it
    # means on the harness arms.
    retriever = GraphDense(word_encoder)
    retriever.index(make_chunks())
    with pytest.raises(ScopeUnsupported):
        retriever.retrieve(QUESTION, 2, scope="C1")


def test_the_kit_reports_a_retriever_that_does_not_conform():
    class TooMany:
        name = "too-many"

        def retrieve(self, question, k=10, *, scope=None, hints=None):
            store = DenseRetriever(HashEmbedder(dimension=32))
            store.index(make_chunks())
            return store.retrieve(question, k + 2, scope="C1")

        async def aretrieve(self, question, k=10, *, scope=None, hints=None):
            return self.retrieve(question, k, scope=scope, hints=hints)

    report = check_retrieval(TooMany(), QUESTION, k=1)
    assert not report.ok
    assert "for k=1" in report.summary()


def test_the_kit_says_what_it_does_not_check():
    # Ordering is deliberately outside the kit: a descending-score rule would
    # rewrite the negative control's published numbers.
    assert any("order" in item for item in NOT_CHECKED)
    assert any("scores descend" in item for item in NOT_CHECKED)
    assert len(CHECKS) >= 10


def test_the_kit_indexes_only_something_that_retrieves():
    # The index step used to gate on `isinstance(retriever, Indexable)`,
    # which is True for a list, so this call reached `list.index(chunks)`.
    holder: list[object] = []
    _index(holder, make_chunks())
    assert holder == []


def test_the_old_kit_entry_point_still_imports():
    from aieng.advanced_rag.retrieval.conformance import run_checks

    assert run_checks is assert_conforms
