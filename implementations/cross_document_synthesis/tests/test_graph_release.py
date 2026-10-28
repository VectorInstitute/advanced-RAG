"""Native RI4 release integrity, passage mapping, and backend selection."""

import asyncio
import hashlib
import json
import sys
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import pytest
import run_graph_comparison as runner
from aieng.advanced_rag.cache import CacheKey, TieredResultCache, store_directory
from aieng.advanced_rag.corpora.musique import MUSIQUE_CORPUS
from aieng.advanced_rag.graphrag.adapters.hipporag2 import HippoRAG2Retriever
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance
from synthesis_lab import graph_release
from synthesis_lab.graph import PendingGraphArtifact, resolve_index
from synthesis_lab.graph_release import hippo_precompute_files, native_metadata, release_entry
from synthesis_lab.tutorial import synthetic_corpus


def native_fixture(tmp_path, backend):
    corpus, _, _ = synthetic_corpus()
    corpus = replace(corpus, name="musique", provenance={"revision": "fixture", "corpus_sha256": MUSIQUE_CORPUS.sha256})
    entry = release_entry(backend)
    key = CacheKey(
        **(
            entry["key"]
            | {
                "config": entry["key"]["config"]
                | {
                    "dataset_revision": "fixture",
                    "passages": len(corpus.passages),
                    "corpus_passages": len(corpus.passages),
                }
            }
        )
    )
    directory = tmp_path / "native"
    directory.mkdir()
    if backend == "hipporag2":
        (directory / "precompute").mkdir()
        for name in hippo_precompute_files():
            (directory / "precompute" / name).write_bytes(b"fixture: never unpickle")
        (directory / "kg_graphml").mkdir()
        graph = nx.Graph()
        for p in corpus.passages:
            graph.add_node("node-" + p.passage_id, type="passage", id=f"{p.title}\n{p.text}")
        nx.write_graphml(graph, directory / "kg_graphml/musique_graph.graphml")
    else:
        docs = {p.passage_id: {"content": f"{p.title}\n{p.text}"} for p in corpus.passages}
        statuses = {p.passage_id: {"status": "processed"} for p in corpus.passages}
        chunks = {
            "chunk-" + p.passage_id: {"full_doc_id": p.passage_id, "file_path": p.passage_id} for p in corpus.passages
        }
        for name, data in (("full_docs", docs), ("doc_status", statuses), ("text_chunks", chunks)):
            (directory / f"kv_store_{name}.json").write_text(json.dumps(data))
        for name in (
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
            "graph_chunk_entity_relation.graphml",
        ):
            (directory / name).write_text("fixture")
    return corpus, key, directory, entry


@pytest.mark.parametrize(
    "backend,digest",
    [
        ("hipporag2", "ccd372ca76bb838d549f8f9292ada8da47221296a1bb50bd09d1ff414e64c6cd"),
        ("lightrag", "93d5ddddf5f3c26978b0315d9a435adb0c249ff2624fc51f5c388621080ab03c"),
    ],
)
def test_release_keys_match_published_builds(backend, digest):
    assert CacheKey(**release_entry(backend)["key"]).digest == digest


@pytest.mark.parametrize("backend", ["hipporag2", "lightrag"])
def test_native_release_restores_without_a_sidecar_and_rejects_corruption(tmp_path, monkeypatch, backend):
    corpus, key, directory, entry = native_fixture(tmp_path, backend)
    cache = TieredResultCache(local_dir=tmp_path / "cache", global_dir=tmp_path / "global")
    artifact = store_directory(cache, key, directory)
    entry = entry | {
        "key": {
            "ri_id": key.ri_id,
            "artifact_type": key.artifact_type,
            "producer_id": key.producer_id,
            "config": key.config,
        }
    }
    entry.update(
        payload_sha256=hashlib.sha256(artifact.path.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(artifact.envelope_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(graph_release, "release_entry", lambda _: entry)
    wrong_corpus = replace(corpus, provenance=corpus.provenance | {"revision": "wrong"})
    with pytest.raises(ValueError, match="complete pinned corpus"):
        resolve_index(cache, key, wrong_corpus, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    restored, metadata = resolve_index(cache, key, corpus, tmp_path / "restored")
    assert not (restored / "ri2_handoff.json").exists()
    assert metadata["backend"] == ("atlas-rag" if backend == "hipporag2" else "lightrag")
    assert set(metadata["retriever_id_map"].values()) == {p.passage_id for p in corpus.passages}
    assert metadata["settings"]["embedding_revision"] is None
    entry["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="release checksum"):
        resolve_index(cache, key, corpus, restored)


@pytest.mark.parametrize("backend", ["hipporag2", "lightrag"])
def test_native_mapping_rejects_unknown_passages(tmp_path, backend):
    corpus, key, directory, entry = native_fixture(tmp_path, backend)
    if backend == "hipporag2":
        path = directory / "kg_graphml/musique_graph.graphml"
        graph = nx.read_graphml(path)
        graph.nodes[next(iter(graph))]["id"] = "unknown source text"
        nx.write_graphml(graph, path)
    else:
        path = directory / "kv_store_text_chunks.json"
        chunks = json.loads(path.read_text())
        chunks[next(iter(chunks))]["full_doc_id"] = "unknown"
        path.write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="unknown|resolve"):
        native_metadata(directory, key, corpus, entry)


def test_missing_hippo_precompute_never_constructs_a_model_client(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resolve_api_key", lambda: pytest.fail("credential read before index validation"))
    with pytest.raises(PendingGraphArtifact, match="rebuilding is disabled"):
        asyncio.run(
            runner.ranker(
                tmp_path,
                {
                    "backend": "atlas-rag",
                    "settings": {"implementation": "atlas-rag", "encoder": "BAAI/bge-small-en-v1.5"},
                },
            )
        )


def test_atlas_backend_uses_shared_adapter_pinned_encoder_and_metered_client(tmp_path, monkeypatch):

    corpus, key, directory, entry = native_fixture(tmp_path, "hipporag2")
    metadata = native_metadata(directory, key, corpus, entry)
    calls = []
    fake = SimpleNamespace(
        retrieve=lambda question, k: [Evidence("source", Provenance(doc_id="node-demo::00"), retriever="fixture")]
    )
    fake.with_client = lambda client: fake
    monkeypatch.setattr(HippoRAG2Retriever, "build", lambda **kwargs: calls.append(kwargs) or fake)
    monkeypatch.setattr(runner, "resolve_api_key", lambda: "fixture-not-a-key")
    monkeypatch.setattr(runner, "DEFAULT_CACHE", tmp_path / "cache")

    async def exercise():
        session = await runner.ranker(directory, metadata)
        assert await session.rank("fixture", 20) == ["demo::00"]
        await session.close()

    asyncio.run(exercise())
    assert calls[0]["encoder_revision"] == graph_release.MODEL_REVISION
    assert calls[0]["keyword"] == "musique"
    assert calls[0]["include_events"] is False and calls[0]["include_concept"] is False


@pytest.mark.parametrize("revision", [None, "pinned-query-revision"])
def test_encoder_revision_does_not_change_the_producer_precompute_key(monkeypatch, revision):
    calls = {}

    def transformer(model, **kwargs):
        calls["encoder"] = (model, kwargs)
        return "local encoder"

    def precompute(**kwargs):
        calls["precompute"] = kwargs
        return {}

    modules = {
        "atlas_rag.llm_generator": {"LLMGenerator": lambda **kw: None},
        "atlas_rag.retriever": {"HippoRAG2Retriever": lambda **kw: SimpleNamespace(text_id_to_node_name={})},
        "atlas_rag.vectorstore": {"create_embeddings_and_index": precompute},
        "atlas_rag.vectorstore.embedding_model": {"SentenceEmbedding": lambda model: model},
        "sentence_transformers": {"SentenceTransformer": transformer},
    }
    for name, contents in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**contents))
    HippoRAG2Retriever.build(
        "fixture",
        None,
        "fixture-model",
        encoder_model="BAAI/bge-small-en-v1.5",
        keyword="musique",
        encoder_revision=revision,
    )
    assert calls["encoder"][1] == {"device": "cpu", "revision": revision}
    assert calls["precompute"]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert calls["precompute"]["keyword"] == "musique"


def test_grid_preserves_the_actual_hipporag_implementation_label():
    from synthesis_lab.decision_grid import decision_grid  # noqa: PLC0415

    report = {
        "completed": True,
        "score_replayed": True,
        "spec": {
            "corpus": "musique",
            "method": "hipporag2",
            "implementation": "atlas-rag",
            "questions": [{"id": "fixture"}],
        },
        "scores": {"2-hop": {"questions": 1, "metrics": {"failure": 0}, "usage": {}}},
        "spec_sha256": "fixture",
        "cost_note": "fixture",
    }
    cell = next(
        row
        for row in decision_grid([report])
        if row["corpus"] == "musique" and row["complexity"] == "2-hop" and row["method"] == "hipporag2"
    )
    assert cell["implementation"] == "atlas-rag" and cell["status"] == "measured"


@pytest.mark.parametrize("provider_fails", [False, True])
def test_nested_atlas_threads_keep_question_usage_and_surface_swallowed_errors(tmp_path, monkeypatch, provider_fails):
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import openai  # noqa: PLC0415
    from aieng.advanced_rag import metering  # noqa: PLC0415
    from aieng.advanced_rag.graphrag import proxy  # noqa: PLC0415

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            if provider_fails:
                raise RuntimeError("fixture provider failed")
            metering.report_call(
                model="fixture", usage=metering.CallUsage(prompt_tokens=7, completion_tokens=3), seconds=0.1
            )
            return "fixture"

        def copy(self):
            return Client()

        def close(self):
            return None

    def with_client(client):
        def retrieve(question, k):
            # atlas-rag catches this; RI2 must still record a failure.
            with ThreadPoolExecutor(max_workers=1) as pool, suppress(RuntimeError):
                pool.submit(client.chat.completions.create).result()
            return [Evidence("source", Provenance(doc_id="node-demo::00"), retriever="fixture")]

        return SimpleNamespace(retrieve=retrieve)

    corpus, key, directory, entry = native_fixture(tmp_path, "hipporag2")
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(proxy, "govern", lambda client, **kw: client)
    monkeypatch.setattr(HippoRAG2Retriever, "build", lambda **kw: SimpleNamespace(with_client=with_client))
    monkeypatch.setattr(runner, "resolve_api_key", lambda: "fixture-not-a-key")
    monkeypatch.setattr(runner, "DEFAULT_CACHE", tmp_path / "runtime")
    meter = metering.UsageMeter()

    async def exercise():
        session = await runner.ranker(directory, native_metadata(directory, key, corpus, entry))

        async def one(qid):
            with meter.question(qid):
                if provider_fails:
                    with pytest.raises(RuntimeError, match="retrieval provider failed"):
                        await session.rank(qid, 20)
                else:
                    assert await session.rank(qid, 20) == ["demo::00"]

        await asyncio.gather(one("first"), one("second"))
        await session.close()

    asyncio.run(exercise())
    if not provider_fails:
        assert meter.per_question()["first"].calls == 1
        assert meter.per_question()["second"].calls == 1
        assert meter.per_question()["first"].prompt_tokens == 7


def test_processed_status_cannot_hide_missing_chunk_vectors_or_passages(tmp_path):
    corpus, key, directory, entry = native_fixture(tmp_path, "lightrag")
    path = directory / "kv_store_text_chunks.json"
    chunks = json.loads(path.read_text())
    del chunks[next(iter(chunks))]
    path.write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="cover the complete corpus"):
        native_metadata(directory, key, corpus, entry)


def test_incomplete_mode_records_missing_passages_and_keeps_identity_checks(tmp_path):
    corpus, key, directory, entry = native_fixture(tmp_path, "lightrag")
    path = directory / "kv_store_text_chunks.json"
    chunks = json.loads(path.read_text())
    removed = chunks.pop(next(iter(chunks)))
    path.write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="complete corpus"):
        native_metadata(directory, key, corpus, entry)
    metadata = native_metadata(directory, key, corpus, entry, allow_incomplete=True)
    coverage = metadata["index_coverage"]
    assert coverage["complete"] is False and coverage["allow_incomplete"] is True
    assert coverage["missing_passage_ids"] == [removed["full_doc_id"]]
    chunks[next(iter(chunks))]["full_doc_id"] = "unknown"
    path.write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="resolve"):
        native_metadata(directory, key, corpus, entry, allow_incomplete=True)


def test_incomplete_mode_cannot_bypass_release_checks_for_other_backends(tmp_path):
    corpus, key, directory, entry = native_fixture(tmp_path, "hipporag2")
    cache = TieredResultCache(local_dir=tmp_path / "local", global_dir=tmp_path / "global")
    with pytest.raises(ValueError, match="pinned LightRAG"):
        resolve_index(cache, key, corpus, tmp_path / "restore", allow_incomplete=True)


def test_evidence_cohorts_partition_questions_without_losing_denominators():
    corpus, _, _ = synthetic_corpus()
    missing = {corpus.questions[0].supporting_ids[0]}
    cohorts = runner.evidence_cohorts(corpus, missing)
    assert cohorts["directly_missing_support"]
    ids = [qid for group in cohorts.values() for qid in group]
    assert sorted(ids) == sorted(q.question_id for q in corpus.questions)
    assert len(ids) == len(set(ids))


def test_grid_labels_incomplete_index_as_exploratory():
    from synthesis_lab.decision_grid import decision_grid  # noqa: PLC0415

    report = {
        "completed": True,
        "score_replayed": True,
        "spec": {
            "corpus": "musique",
            "method": "lightrag",
            "questions": [{"id": "fixture"}],
            "index_coverage": {"complete": False, "missing_passage_ids": ["missing"]},
        },
        "scores": {"2-hop": {"questions": 1, "metrics": {"failure": 0}, "usage": {}}},
        "spec_sha256": "fixture",
        "cost_note": "fixture",
    }
    cell = next(r for r in decision_grid([report]) if r["method"] == "lightrag" and r["complexity"] == "2-hop")
    assert cell["status"] == "exploratory_incomplete_index"
    assert cell["index_coverage"]["complete"] is False
