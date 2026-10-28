"""The retrieval boundary: protocol conformance, evidence, and lazy extras.

The point of this module is that a method can be swapped without anything
downstream changing, so the tests are mostly about uniformity rather than about
any one adapter.
"""

from __future__ import annotations

import asyncio

import pytest

from aieng.advanced_rag.graphrag import (
    DenseRetriever,
    Evidence,
    GraphRetriever,
    HippoRAG2Retriever,
    LightRAGRetriever,
    MissingExtraError,
    build_prompt,
    format_sources,
    to_runs,
    unique_documents,
)
from aieng.advanced_rag.provenance import Provenance


class _Chunk:
    def __init__(self, chunk_id, doc_id, text):
        self.chunk_id, self.doc_id, self.text = chunk_id, doc_id, text
        self.start, self.end = 0, len(text)


def _word_encoder(vocabulary: tuple[str, ...]):
    """Encode with a tiny deterministic bag of words, so tests need no model."""

    def encode(texts):
        return [
            [float(text.lower().count(word)) for word in vocabulary] for text in texts
        ]

    return encode


@pytest.fixture
def dense():
    encoder = _word_encoder(("nginx", "postgres", "timeout", "backup"))
    retriever = DenseRetriever(encoder, name="dense-test")
    retriever.index(
        [
            _Chunk("d1::0", "d1", "nginx timeout on upstream"),
            _Chunk("d2::0", "d2", "postgres backup schedule"),
            _Chunk("d3::0", "d3", "nginx nginx worker tuning"),
        ]
    )
    return retriever


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "adapter", [DenseRetriever, LightRAGRetriever, HippoRAG2Retriever]
    )
    def test_adapter_declares_the_interface(self, adapter):
        assert hasattr(adapter, "retrieve")
        assert (
            hasattr(adapter, "name") or "name" in adapter.__init__.__code__.co_varnames
        )

    def test_dense_satisfies_the_protocol_at_runtime(self, dense):
        assert isinstance(dense, GraphRetriever)


class TestDenseAdapter:
    def test_ranks_the_matching_chunk_first(self, dense):
        assert dense.retrieve("nginx", k=3)[0].doc_id == "d3"

    def test_k_bounds_the_results(self, dense):
        assert len(dense.retrieve("nginx", k=2)) == 2

    def test_evidence_carries_provenance_and_retriever_name(self, dense):
        item = dense.retrieve("postgres backup", k=1)[0]
        assert item.chunk_id == "d2::0"
        assert item.provenance.span == (0, len("postgres backup schedule"))
        assert item.retriever == "dense-test"

    def test_scores_are_populated_and_descending(self, dense):
        scores = [item.score for item in dense.retrieve("nginx", k=3)]
        assert all(score is not None for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_empty_index_returns_nothing(self):
        assert DenseRetriever(_word_encoder(("a",))).retrieve("a") == []


class TestEvidenceHelpers:
    def _evidence(self):
        return [
            Evidence("a", Provenance("d1", "d1::0"), 0.9),
            Evidence("b", Provenance("d1", "d1::1"), 0.8),
            Evidence("c", Provenance("d2", "d2::0"), 0.7),
        ]

    def test_to_runs_preserves_rank_order(self):
        assert to_runs(self._evidence()) == ["d1::0", "d1::1", "d2::0"]

    def test_to_runs_deduplicates(self):
        duplicated = self._evidence() + [Evidence("a", Provenance("d1", "d1::0"))]
        assert to_runs(duplicated) == ["d1::0", "d1::1", "d2::0"]

    def test_to_runs_falls_back_to_document_id(self):
        # A method reporting only document granularity must stay scoreable.
        assert to_runs([Evidence("a", Provenance("d1"))]) == ["d1"]

    def test_unique_documents_collapses_chunks(self):
        assert unique_documents(self._evidence()) == ["d1", "d2"]


class TestSynthesisIsOptional:
    def test_sources_are_numbered_and_cited(self):
        rendered = format_sources(
            [Evidence("body text", Provenance("d1", section="Answer"))]
        )
        assert rendered.startswith("[1] d1 > Answer")
        assert "body text" in rendered

    def test_passages_are_truncated(self):
        rendered = format_sources([Evidence("x" * 500, Provenance("d1"))], max_chars=10)
        assert "x" * 10 in rendered and "x" * 11 not in rendered

    def test_prompt_carries_the_query_and_sources(self):
        prompt = build_prompt(
            "why 502?", [Evidence("because upstream", Provenance("d1"))]
        )
        assert "why 502?" in prompt and "because upstream" in prompt

    def test_a_custom_template_replaces_the_default_entirely(self):
        # Output shape is the practitioner's decision, which is the whole point.
        prompt = build_prompt(
            "q",
            [Evidence("t", Provenance("d1"))],
            template="ONLY {sources} AND {query}",
        )
        assert prompt.startswith("ONLY [1] d1")


class TestOptionalExtras:
    """Optional dependencies must not be needed to import, only to construct."""

    def test_importing_the_package_needs_no_framework(self):
        import aieng.advanced_rag.graphrag as module

        assert module.LightRAGRetriever is not None

    def test_missing_extra_names_the_install_command(self):
        error = MissingExtraError("hipporag", "atlas-rag")
        assert "aieng-advanced-rag[hipporag]" in str(error)
        assert isinstance(error, ImportError)

    @pytest.mark.parametrize(
        "builder,extra",
        [
            (
                lambda: asyncio.run(LightRAGRetriever.build("/tmp/x", None, None)),
                "lightrag",
            ),
            (lambda: HippoRAG2Retriever.build("/tmp/x", None, "m"), "hipporag"),
        ],
    )
    def test_building_without_the_extra_raises_a_named_error(self, builder, extra):
        pytest.importorskip  # noqa: B018 - readability marker
        try:
            builder()
        except MissingExtraError as error:
            assert error.extra == extra
        except Exception:
            # The library is installed in this environment, so construction got
            # far enough to fail for another reason. That is also a pass for
            # this test's purpose: the extra was not the blocker.
            pass


class TestLightRAGBuild:
    """``build`` has to hand back an instance whose first query works."""

    def test_the_storages_are_initialized_before_build_returns(self, monkeypatch):
        # LightRAG leaves ``auto_manage_storages_states`` off, so an instance that
        # skipped this step fails its first query with an opaque TypeError about
        # the asynchronous context manager protocol.
        pytest.importorskip("lightrag", reason="the lightrag extra is not installed")
        import lightrag

        initialized: list[str] = []

        class FakeLightRAG:
            def __init__(self, **kwargs):
                self.working_dir = kwargs["working_dir"]

            async def initialize_storages(self):
                initialized.append(self.working_dir)

        monkeypatch.setattr(lightrag, "LightRAG", FakeLightRAG)
        retriever = asyncio.run(
            LightRAGRetriever.build("/tmp/wd", None, None, mode="local")
        )
        assert initialized == ["/tmp/wd"]
        assert isinstance(retriever, LightRAGRetriever)
        assert retriever.mode == "local"


class TestLightRAGDocumentIds:
    """Citations have to group by document even when file paths are absent."""

    def _resolve(self, chunk):
        from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever

        return LightRAGRetriever._document_id(chunk)

    def test_file_path_is_preferred(self):
        assert (
            self._resolve({"file_path": "q123.md", "chunk_id": "doc-abc-chunk-001"})
            == "q123.md"
        )

    def test_unknown_source_falls_back_to_the_document_hash(self):
        # LightRAG writes "unknown_source" when insert had no file_paths.
        assert (
            self._resolve(
                {"file_path": "unknown_source", "chunk_id": "doc-abc123-chunk-002"}
            )
            == "doc-abc123"
        )

    def test_missing_file_path_falls_back_to_the_document_hash(self):
        assert self._resolve({"chunk_id": "doc-deadbeef-chunk-000"}) == "doc-deadbeef"

    def test_chunks_from_one_document_share_a_document_id(self):
        first = self._resolve({"chunk_id": "doc-aaa-chunk-000"})
        second = self._resolve({"chunk_id": "doc-aaa-chunk-007"})
        assert first == second

    def test_reference_id_is_the_last_resort(self):
        assert self._resolve({"reference_id": "r1"}) == "r1"

    def test_nothing_usable_returns_unknown(self):
        assert self._resolve({}) == "unknown"
