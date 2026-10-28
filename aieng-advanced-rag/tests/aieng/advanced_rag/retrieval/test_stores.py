"""The Qdrant vector store: ranking, ties, scope, settings, export, and parity.

The parity tests hold the list-based dense retriever the package shipped before
the store, verbatim in its ranking, and assert the Qdrant arm returns the same
chunks in the same order. The server tests run only when ``QDRANT_URL`` names a
running Qdrant server.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.graphrag.evidence import to_runs
from aieng.advanced_rag.retrieval.chunkers import fixed_chars
from aieng.advanced_rag.retrieval.corpus import Chunk, Query
from aieng.advanced_rag.retrieval.embedders import HashEmbedder
from aieng.advanced_rag.retrieval.retrievers import (
    DenseRetriever,
    QdrantRetriever,
    run_retriever,
)
from aieng.advanced_rag.retrieval.stores import (
    ARTIFACT_TYPE,
    QdrantSettings,
    QdrantStore,
    cache_collection,
    connect,
    export_collection,
    fetch_collection,
    is_local,
    load_collection,
)


def unit(*values: float) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    return (vector / np.linalg.norm(vector)).tolist()


class ListReference:
    """The pre-store ``DenseRetriever`` ranking: float64 dot product, ties on id."""

    def __init__(self, embedder: HashEmbedder) -> None:
        self.embedder = embedder
        self._vectors: dict[str, list[float]] = {}
        self._by_doc: dict[str, list[str]] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        vectors = self.embedder.encode([c.indexed_text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._vectors[chunk.chunk_id] = vector
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def search(self, query: Query, k: int) -> list[str]:
        vector = self.embedder.encode([query.question])[0]
        scored = [
            (cid, sum(a * b for a, b in zip(vector, self._vectors[cid])))
            for cid in self._by_doc.get(query.doc_id, [])
        ]
        scored.sort(key=lambda r: (-r[1], r[0]))
        return [cid for cid, _ in scored[:k]]


DOCS = {
    "alpha": "Termination for convenience is permitted on thirty days notice. "
    "Payment terms are net thirty. " * 12,
    "beta": "Governing law is the law of Ontario. Assignment requires consent. " * 12,
    "gamma": "Confidential information must be returned on request. " * 20,
}
QUESTIONS = [
    "termination for convenience",
    "governing law",
    "payment terms",
    "confidential information",
    "nothing in common",
]


def corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in DOCS.items():
        chunks.extend(fixed_chars(doc_id, text, chunk_chars=300, overlap_chars=50))
    return chunks


class TestRanking:
    def test_search_orders_by_dot_product(self):
        store = QdrantStore()
        store.add(["b", "a", "c"], [unit(0, 1), unit(1, 0), unit(1, 1)])
        assert [k for k, _ in store.search(unit(1, 0), 3)] == ["a", "c", "b"]

    def test_ties_break_on_key_even_when_they_run_past_the_fetch(self):
        # Sixty identical vectors and k=10: the first fetch holds k plus a
        # margin, every point in it ties, so the store has to widen the fetch
        # to find the ten smallest keys rather than return whichever ten
        # qdrant-client's scan put first. That scan returns ties in reverse
        # insertion order, so inserting in key order would hide the bug.
        keys = [f"d::{i:04d}" for i in range(60)]
        store = QdrantStore()
        store.add(list(reversed(keys)), [unit(1, 0)] * 60)
        assert [k for k, _ in store.search(unit(1, 0), 10)] == keys[:10]

    def test_ties_on_the_boundary_resolve_on_key(self):
        store = QdrantStore()
        store.add(
            ["z", "y", "x", "w"], [unit(1, 0), unit(1, 1), unit(1, 1), unit(0, 1)]
        )
        assert [k for k, _ in store.search(unit(1, 0), 2)] == ["z", "x"]

    def test_scope_filters_on_document(self):
        store = QdrantStore()
        store.add(
            ["a::0", "a::1", "b::0"],
            [unit(1, 0), unit(0.9, 0.1), unit(1, 0)],
            ["a", "a", "b"],
        )
        assert [k for k, _ in store.search(unit(1, 0), 5, scope="a")] == [
            "a::0",
            "a::1",
        ]
        assert store.search(unit(1, 0), 5, scope="missing") == []

    def test_within_restricts_to_point_identifiers(self):
        store = QdrantStore()
        store.add(["a", "b", "c"], [unit(1, 0), unit(0, 1), unit(1, 0.01)])
        assert [k for k, _ in store.search(unit(1, 0), 2, within=[1, 2])] == ["c", "b"]
        assert store.search(unit(1, 0), 2, within=[]) == []

    def test_a_batch_ranks_each_query_as_a_single_search_would(self):
        store = QdrantStore()
        store.add(["a", "b", "c"], [unit(1, 0), unit(0, 1), unit(1, 1)])
        queries = [unit(1, 0), unit(0, 1)]
        assert store.search_batch(queries, 2) == [store.search(q, 2) for q in queries]

    def test_an_empty_store_and_a_zero_depth_return_nothing(self):
        store = QdrantStore()
        assert store.search(unit(1, 0), 3) == []
        store.add(["a"], [unit(1, 0)])
        assert store.search(unit(1, 0), 0) == []

    def test_repeated_keys_and_a_changed_width_are_refused(self):
        store = QdrantStore()
        store.add(["a"], [unit(1, 0)])
        with pytest.raises(ValueError, match="already in the store"):
            store.add(["a"], [unit(0, 1)])
        with pytest.raises(ValueError, match="width"):
            store.add(["b"], [unit(1, 0, 0)])
        with pytest.raises(ValueError, match="keys against"):
            store.add(["c", "d"], [unit(1, 0)])

    def test_scores_are_float32_dot_products(self):
        store = QdrantStore()
        vector = unit(3, 4)
        store.add(["a"], [vector])
        _, score = store.search(unit(1, 0), 1)[0]
        assert score == pytest.approx(0.6, abs=1e-6)


class TestSettings:
    def test_the_default_is_an_exact_search_that_needs_no_server(self):
        settings = QdrantSettings()
        assert not settings.needs_server
        assert settings.collection_kwargs() == {}
        assert settings.search_params().exact is True

    @pytest.mark.parametrize(
        "settings",
        [
            QdrantSettings(quantization="turbo", bits=2),
            QdrantSettings(quantization="binary", oversampling=4.0, rescore=True),
            QdrantSettings(hnsw_ef=64),
        ],
    )
    def test_in_process_mode_refuses_what_it_would_ignore(self, settings):
        with pytest.raises(ValueError, match="need a Qdrant server"):
            QdrantStore(settings=settings)

    def test_turbo_and_binary_map_to_qdrant_configs(self):
        turbo = QdrantSettings(quantization="turbo", bits=1.5).collection_kwargs()
        assert turbo["quantization_config"].turbo.bits.value == "bits1_5"
        binary = QdrantSettings(
            quantization="binary", bits=2, asymmetric=True
        ).collection_kwargs()
        config = binary["quantization_config"].binary
        assert config.encoding.value == "two_bits"
        assert config.query_encoding.value == "scalar8bits"

    def test_server_settings_lower_the_per_segment_thresholds(self):
        # The defaults apply per segment, so a bootcamp-sized collection would
        # otherwise never build the graph or the quantized copy.
        quantized = QdrantSettings(quantization="turbo", bits=2).collection_kwargs()
        assert quantized["optimizers_config"].indexing_threshold == 1
        assert "hnsw_config" not in quantized
        graph = QdrantSettings(hnsw_ef=16).collection_kwargs()
        assert graph["hnsw_config"].full_scan_threshold == 10

    def test_search_params_carry_oversampling_rescoring_and_ef(self):
        params = QdrantSettings(
            quantization="binary", oversampling=4.0, rescore=True, hnsw_ef=128
        ).search_params()
        assert params.hnsw_ef == 128
        assert params.exact is None
        assert params.quantization.oversampling == 4.0
        assert params.quantization.rescore is True

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"quantization": "turbo", "bits": 3}, "takes 1, 1.5, 2, 4 bits"),
            ({"quantization": "binary", "bits": 4}, "takes 1, 1.5, 2 bits"),
            ({"quantization": "scalar"}, "must be 'binary', 'turbo'"),
            ({"oversampling": 2.0}, "need a quantization"),
            ({"quantization": "turbo", "asymmetric": True}, "binary quantization only"),
            ({"quantization": "binary", "oversampling": 0.5}, "at least 1"),
        ],
    )
    def test_settings_that_cannot_work_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            QdrantSettings(**kwargs)


class TestConnect:
    def test_memory_and_paths_are_in_process(self, tmp_path):
        assert is_local(connect(":memory:"))
        assert is_local(connect(str(tmp_path / "qdrant")))

    def test_no_location_falls_back_to_memory_without_qdrant_url(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        assert is_local(connect(None))


class TestExport:
    def build(self) -> QdrantStore:
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(40, 8)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        store = QdrantStore()
        store.add(
            [f"d{i % 4}::{i:03d}" for i in range(40)],
            matrix,
            [f"d{i % 4}" for i in range(40)],
        )
        return store

    def test_a_collection_round_trips_through_files_into_another_mode(self, tmp_path):
        store = self.build()
        export_collection(store, tmp_path / "export", metadata={"model": "m"})
        on_disk = QdrantStore(str(tmp_path / "qdrant"))
        loaded = load_collection(tmp_path / "export", on_disk)

        assert loaded.metadata == {"model": "m"}
        assert on_disk.keys == store.keys and on_disk.doc_ids == store.doc_ids
        np.testing.assert_array_equal(on_disk.vectors(), store.vectors())
        query = store.vectors()[3]
        assert on_disk.search(query, 7, scope="d3") == store.search(
            query, 7, scope="d3"
        )

    def test_the_export_is_plain_files(self, tmp_path):
        export_collection(self.build(), tmp_path / "export")
        assert sorted(p.name for p in (tmp_path / "export").iterdir()) == [
            "collection.json",
            "points.jsonl",
            "vectors.npy",
        ]

    def test_exporting_over_files_and_loading_twice_are_refused(self, tmp_path):
        store = self.build()
        export_collection(store, tmp_path / "export")
        with pytest.raises(FileExistsError):
            export_collection(store, tmp_path / "export")
        with pytest.raises(ValueError, match="empty store"):
            load_collection(tmp_path / "export", store)

    def test_a_collection_round_trips_through_the_artifact_cache(self, tmp_path):
        cache = TieredResultCache(
            global_dir=tmp_path / "global", local_dir=tmp_path / "local"
        )
        key = CacheKey("ri1", ARTIFACT_TYPE, "test", {"corpus": "x"})
        store = self.build()
        assert fetch_collection(cache, key, QdrantStore()) is None
        cache_collection(cache, key, store)
        restored = QdrantStore()
        assert fetch_collection(cache, key, restored) is not None
        query = store.vectors()[0]
        assert restored.search(query, 5) == store.search(query, 5)


class CountingEmbedder(HashEmbedder):
    """Counts document encodes, so a cache hit can be told from a rebuild."""

    def __init__(self) -> None:
        super().__init__(dimension=64)
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        return super().encode(texts)


class TestRetrieverPersistence:
    def test_load_or_index_embeds_once_and_then_loads(self, tmp_path):
        cache = TieredResultCache(
            global_dir=tmp_path / "global", local_dir=tmp_path / "local"
        )
        key = CacheKey("ri1", ARTIFACT_TYPE, "dense", {"corpus": "docs"})
        chunks = corpus()

        first = QdrantRetriever(CountingEmbedder())
        assert first.load_or_index(cache, key, chunks) is False
        embedder = CountingEmbedder()
        second = QdrantRetriever(embedder)
        assert second.load_or_index(cache, key, chunks) is True

        assert embedder.encoded == 0
        for question in QUESTIONS:
            assert to_runs(second.retrieve(question, 5)) == to_runs(
                first.retrieve(question, 5)
            )
            assert second.retrieve(question, 1)[0].text

    def test_loading_refuses_other_chunks_or_another_embedder(self, tmp_path):
        retriever = QdrantRetriever(HashEmbedder(dimension=64))
        retriever.index(corpus())
        retriever.save(tmp_path / "export")

        with pytest.raises(ValueError, match="other chunks"):
            QdrantRetriever(HashEmbedder(dimension=64)).load(
                tmp_path / "export", corpus()[:-1]
            )

        class Other(HashEmbedder):
            name = "other"

        with pytest.raises(ValueError, match="embedder"):
            QdrantRetriever(Other(dimension=64)).load(tmp_path / "export", corpus())


class TestParityWithTheListRetriever:
    def test_rankings_match_the_pre_store_retriever_exactly(self):
        chunks = corpus()
        embedder = HashEmbedder(dimension=128)
        reference = ListReference(embedder)
        reference.index(chunks)
        retriever = DenseRetriever(embedder)
        retriever.index(chunks)

        queries = [
            Query(f"{doc}:{n}", doc, "cat", question)
            for doc in DOCS
            for n, question in enumerate(QUESTIONS)
        ]
        assert run_retriever(retriever, queries, k=6) == {
            q.query_id: reference.search(q, 6) for q in queries
        }

    def test_the_repeated_boilerplate_ties_resolve_as_before(self):
        # Repeated clauses tie exactly on score, which is the CUAD case the
        # chunk-identifier tie-break exists for. Thirty copies run the ties past
        # the store's first fetch.
        clause = "Confidential information must be returned on request."
        chunks = [
            Chunk(
                f"gamma::{i:04d}",
                "gamma",
                clause if i % 3 else f"{clause} Notice.",
                i * 60,
                i * 60 + 55,
            )
            for i in range(30)
        ]
        embedder = HashEmbedder(dimension=128)
        reference = ListReference(embedder)
        reference.index(chunks)
        retriever = DenseRetriever(embedder)
        retriever.index(chunks)
        query = Query("q", "gamma", "cat", "confidential information")
        scores = [h.score for h in retriever.search(query, 10)]
        assert len(set(scores)) < len(scores), "the corpus no longer produces ties"
        assert [h.chunk_id for h in retriever.search(query, 10)] == reference.search(
            query, 10
        )

    def test_an_unscoped_retrieve_ranks_every_document_at_once(self):
        retriever = DenseRetriever(HashEmbedder(dimension=128))
        retriever.index(corpus())
        hits = retriever.retrieve("governing law", 3)
        assert {h.doc_id for h in hits} == {"beta"}
        scores = [h.score for h in retriever.retrieve("payment terms", 10)]
        assert scores == sorted(scores, reverse=True)

    def test_the_retriever_is_the_dense_arm_under_its_old_name(self):
        assert DenseRetriever is QdrantRetriever
        assert QdrantRetriever.name == "dense"


SERVER = os.environ.get("QDRANT_URL")


@pytest.mark.integration_test
@pytest.mark.skipif(not SERVER, reason="QDRANT_URL does not name a Qdrant server")
class TestServer:
    def test_exact_server_search_matches_in_process_search(self):
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(500, 32)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        keys = [f"d{i % 5}::{i:04d}" for i in range(500)]
        docs = [f"d{i % 5}" for i in range(500)]
        local, remote = QdrantStore(), QdrantStore(SERVER, collection="test-exact")
        local.add(keys, matrix, docs)
        remote.add(keys, matrix, docs)
        queries = matrix[:10] + 0.1 * rng.normal(size=(10, 32)).astype(np.float32)
        assert [[k for k, _ in r] for r in remote.search_batch(queries, 10)] == [
            [k for k, _ in r] for r in local.search_batch(queries, 10)
        ]

    def test_quantization_without_rescoring_changes_the_ranking(self):
        rng = np.random.default_rng(2)
        matrix = rng.normal(size=(2000, 64)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        keys = [f"p{i:05d}" for i in range(2000)]
        exact = QdrantStore()
        exact.add(keys, matrix)
        binary = QdrantStore(
            SERVER,
            collection="test-binary",
            settings=QdrantSettings(quantization="binary", rescore=False),
        )
        binary.add(keys, matrix)
        queries = matrix[:20]
        agreement = np.mean(
            [
                len({k for k, _ in a} & {k for k, _ in b}) / 10
                for a, b in zip(
                    exact.search_batch(queries, 10), binary.search_batch(queries, 10)
                )
            ]
        )
        # Measured 0.175 on random 64-dimension vectors: a quantized copy that
        # was never built would agree 1.0.
        assert agreement < 0.9


def test_a_path_store_survives_its_directory(tmp_path: Path):
    # A path is a convenience for one machine, never the shipping format.
    store = QdrantStore(str(tmp_path / "db"))
    store.add(["a", "b"], [unit(1, 0), unit(0, 1)])
    assert (tmp_path / "db").is_dir()
    assert [k for k, _ in store.search(unit(0, 1), 1)] == ["b"]
