"""External handoffs cannot silently enable articles or substitute graph indexes."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest
import run_graph_comparison as graph_runner
from aieng.advanced_rag.cache import CacheKey, TieredResultCache, store_directory
from aieng.advanced_rag.corpora import multihop_articles
from aieng.advanced_rag.corpora.multihop_articles import ANSWERABLE_TYPES, load_articles, parse_articles
from run_graph_comparison import mapped_passages, ranker
from synthesis_lab.decision_grid import decision_grid
from synthesis_lab.graph import PendingGraphArtifact, resolve_index
from synthesis_lab.tutorial import graph_example, ircot_example, source_answer, synthetic_corpus


def test_article_ruling_is_checked_before_any_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        multihop_articles, "fetch", lambda *a, **k: pytest.fail("article fetch before the ruling check")
    )
    ruling = tmp_path / "pending.json"
    ruling.write_text(json.dumps({"status": "pending"}))
    with pytest.raises(ValueError, match="lacks status: approved"):
        load_articles(ruling=ruling)


def test_article_types_and_evidence_mapping():
    articles = [{"title": "A", "body": "A report", "author": "Author"}]
    questions = [
        {
            "query": "What happened?",
            "answer": "A",
            "question_type": kind,
            "evidence_list": [{"title": "A", "author": "Author"}],
        }
        for kind in sorted(ANSWERABLE_TYPES)
    ]
    p, q = parse_articles(json.dumps(questions).encode(), json.dumps(articles).encode())
    assert {r.category for r in q} == ANSWERABLE_TYPES
    assert all(r.supporting_ids == (p[0].passage_id,) for r in q)
    articles.append(articles[0])
    with pytest.raises(ValueError, match="ambiguous"):
        parse_articles(json.dumps(questions).encode(), json.dumps(articles).encode())


def test_graph_payload_is_checked_and_wrong_corpus_rejected(tmp_path):
    corpus, _, _ = synthetic_corpus()
    cache = TieredResultCache(local_dir=tmp_path / "cache", global_dir=tmp_path / "empty")
    key = CacheKey("ri4", "index", "fixture", {})
    with pytest.raises(PendingGraphArtifact, match="absent"):
        resolve_index(cache, key, corpus, tmp_path / "dest")
    directory = tmp_path / "index"
    directory.mkdir()
    (directory / "ri2_handoff.json").write_text(json.dumps({"corpus_revision": "wrong"}))
    store_directory(cache, key, directory)
    with pytest.raises(ValueError, match="complete pinned"):
        resolve_index(cache, key, corpus, tmp_path / "dest")
    (tmp_path / "dest/ri2_handoff.json").write_text("tampered")
    with pytest.raises(ValueError, match="differs"):
        resolve_index(cache, key, corpus, tmp_path / "dest")


def test_grid_never_turns_pending_into_a_measured_loss():
    cells = decision_grid([])
    assert cells and all(c["status"] == "pending" and "metrics" not in c for c in cells)
    with pytest.raises(ValueError, match="completed"):
        decision_grid([{"completed": False}])


def test_notebook_methods_execute_real_adapters_offline():
    answer, trace = asyncio.run(ircot_example())
    assert answer.answer.text == "Oslo"
    assert trace["stop_reason"] == "sufficient"
    answer, trace = asyncio.run(ircot_example(rounds=1))
    assert answer.abstained
    with_bridge, _ = asyncio.run(graph_example())
    missing_bridge, _ = asyncio.run(graph_example(include_bridge=False))
    assert not with_bridge.abstained and missing_bridge.abstained


def test_graph_chunks_are_deduplicated_before_the_original_passage_budget():
    assert mapped_passages(["c1", "c2", "c3"], {"c1": "p1", "c2": "p1", "c3": "p2"}, {"p1", "p2"}, 2) == ["p1", "p2"]
    with pytest.raises(ValueError, match="unknown"):
        mapped_passages(["unmapped"], {}, {"p1"}, 1)


def test_incomplete_graph_artifact_cannot_initialize_empty_stores(tmp_path):
    with pytest.raises(PendingGraphArtifact, match="prebuilt"):
        asyncio.run(ranker(tmp_path, {"backend": "lightrag", "passage_ids": ["p"]}))


def test_query_storage_writes_cannot_mutate_the_verified_index(tmp_path, monkeypatch):
    corpus, _, _ = synthetic_corpus()
    corpus = replace(corpus, name="musique", provenance={"revision": "fixture", "corpus_sha256": "a" * 64})
    cache = TieredResultCache(local_dir=tmp_path / "cache", global_dir=tmp_path / "global")
    key = CacheKey("ri4", "index", "fixture", {})
    source = tmp_path / "source"
    source.mkdir()
    metadata = {
        "backend": "lightrag",
        "settings": {},
        "producer_revision": "fixture",
        "corpus_revision": "fixture",
        "corpus_sha256": "a" * 64,
        "passage_ids": [p.passage_id for p in corpus.passages],
    }
    (source / "ri2_handoff.json").write_text(json.dumps(metadata))
    store_directory(cache, key, source)
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"key": asdict(key)}))
    monkeypatch.setattr(graph_runner, "DEFAULT_CACHE", tmp_path / "reference")
    monkeypatch.setattr(graph_runner, "TieredResultCache", lambda: cache)
    monkeypatch.setattr(graph_runner, "load_corpus", lambda *a, **k: corpus)
    monkeypatch.setattr(graph_runner, "CachedProxyCompletion", lambda *a, **k: source_answer)
    opened = []
    closed = []

    async def factory(directory, meta, *, models=None):
        opened.append(directory)
        (directory / "query-cache.json").write_text("temporary query state")

        async def rank(question, k):
            preferred = ["demo::00", "demo::10"]
            return (preferred + [p.passage_id for p in corpus.passages if p.passage_id not in preferred])[:k]

        async def close():
            assert directory.exists()
            closed.append(True)

        return graph_runner.GraphSession(rank, close)

    monkeypatch.setattr(graph_runner, "ranker", factory)
    report = asyncio.run(graph_runner.run(handoff, tmp_path / "results.json", per_hop=0))
    assert report["completed"] and report["scores"]["overall"]["metrics"]["exact_match"] == 1
    assert closed == [True] and not opened[0].exists()
    pristine = tmp_path / "ri2-complete/graph" / key.digest
    assert (pristine / "ri2_handoff.json").exists()
    assert not (pristine / "query-cache.json").exists()
