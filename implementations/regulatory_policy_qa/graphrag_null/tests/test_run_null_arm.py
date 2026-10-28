"""Protect score-only runs against indexing, model calls, and mislabelling."""

from __future__ import annotations

import asyncio
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_null_arm as runner  # noqa: E402
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.pipeline.spine import SpineConfig  # noqa: E402
from aieng.advanced_rag.provenance import Provenance  # noqa: E402
from aieng.advanced_rag.retrieval.corpus import Chunk, Query, Slice  # noqa: E402
from export_store import main as export_main  # noqa: E402
from export_store import write_bundle  # noqa: E402


@pytest.mark.parametrize("status", ["processed", "processing", "failed"])
def test_duplicate_attempt_receipts_do_not_replace_the_original_status(tmp_path, status):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows = {
        "doc-original": {"status": status},
        "dup-receipt": {"status": "failed", "metadata": {"is_duplicate": True, "original_doc_id": "doc-original"}},
    }
    (workspace / "kv_store_doc_status.json").write_text(json.dumps(rows))
    assert runner.document_status(tmp_path, "workspace") == status


def test_orphan_duplicate_receipt_is_not_a_processed_index(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "dup-receipt": {"status": "failed", "metadata": {"is_duplicate": True, "original_doc_id": "missing"}},
            }
        )
    )
    assert runner.document_status(tmp_path, "workspace") == "orphan-duplicate"


@pytest.mark.parametrize("failed_id", ["doc-failed", "dup-unverified"])
def test_genuine_or_unverified_failed_records_remain_failures(tmp_path, failed_id):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-good": {"status": "processed"},
                failed_id: {"status": "failed"},
            }
        )
    )
    assert runner.document_status(tmp_path, "workspace") == "failed/processed"


def test_cuad_default_output_cannot_overwrite_contract_nli_results():
    args = runner.parse_args(["--corpus", "cuad"])
    assert args.out == runner.HERE.parent / "outputs/cuad_null"
    assert runner.parse_args([]).out == runner.HERE.parent / "outputs/graphrag_null/measured_thresholds"
    full = runner.parse_args(["--documents", "123", "--arms", "dense"])
    assert full.out == runner.HERE.parent / "outputs/full_contract_nli/dense"


def test_cache_only_rejects_partial_population_before_client_construction(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "dense.json").write_text('{"q1": []}')
    runner.ResponseCache(tmp_path / "proxy_cache.sqlite")
    corpus = SimpleNamespace(queries=[Query("q1", "a", "x", "question"), Query("q2", "b", "x", "question")], chunks=[])
    monkeypatch.setattr(runner, "corpus_slice", lambda *args: (corpus, {}))
    monkeypatch.setattr(runner, "scoring_completer", lambda *args: pytest.fail("client creation is forbidden"))
    with pytest.raises(ValueError, match="cached evidence IDs"):
        runner.main(["--store", str(tmp_path), "--arms", "dense", "--score-only", "--cache-only"])


@pytest.mark.parametrize("status", ["failed", "processed"])
def test_cached_index_report_cannot_hide_an_unhealthy_or_missing_store(tmp_path, monkeypatch, status):
    paths = runner.Paths(store=tmp_path / "store", out=tmp_path / "out", env=tmp_path / "missing.env")
    paths.out.mkdir()
    paths.index_report.write_text(json.dumps({"documents": 1, "per_document": {"doc": {"status": status}}}))
    monkeypatch.setattr(runner, "document_texts", lambda *args: [("doc", "text")])
    monkeypatch.setattr(runner, "connection", lambda *args, **kwargs: pytest.fail("model client opened"))
    with pytest.raises(RuntimeError, match="index"):
        asyncio.run(
            runner.index_documents(paths, 1, 1, False, runner.ProxyBatch(runner.DEFAULT_MODEL, api_key="unused"))
        )


def test_fresh_graph_retrieval_rejects_missing_index_before_loading_models(tmp_path, monkeypatch):
    paths = runner.Paths(store=tmp_path, out=tmp_path / "out", env=tmp_path / "missing.env")
    corpus = SimpleNamespace(queries=[Query("q", "doc", "x", "question")])
    monkeypatch.setattr(runner, "SentenceTransformerEmbedder", lambda *args: pytest.fail("model loaded"))
    with pytest.raises(RuntimeError, match="index"):
        asyncio.run(
            runner.retrieve_lightrag(paths, corpus, 1, 1, 10, runner.ProxyBatch(runner.DEFAULT_MODEL, api_key="unused"))
        )


def test_cached_graph_evidence_does_not_reindex_when_only_dense_table_is_missing(tmp_path, monkeypatch):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/lightrag-mix.json").write_text('{"q": []}')
    corpus = SimpleNamespace(queries=[Query("q", "doc", "x", "question")], chunks=[], summary=lambda: {"queries": 1})
    monkeypatch.setattr(runner, "corpus_slice", lambda *args: (corpus, {}))
    monkeypatch.setattr(runner, "index_documents", lambda *args: pytest.fail("graph indexing is forbidden"))
    monkeypatch.setattr(runner, "connection", lambda *args, **kwargs: pytest.fail("model client opened"))

    class DenseTableNeededError(Exception):
        pass

    def dense_only(*args):
        raise DenseTableNeededError

    monkeypatch.setattr(runner, "retrieve_dense", dense_only)
    with pytest.raises(DenseTableNeededError):
        runner.main(["--store", str(tmp_path), "--arms", "dense", "lightrag-mix"])


def test_all_fresh_scoring_tables_are_validated_before_a_model_client(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    runner.save_evidence(evidence / "dense.json", {"q": []})
    runner.save_evidence(
        evidence / "lightrag-mix.json",
        {
            "q": [Evidence(text="source", provenance=Provenance(doc_id="foreign"))],
        },
    )
    corpus = SimpleNamespace(
        queries=[Query("q", "expected", "x", "question")], chunks=[], summary=lambda: {"queries": 1}
    )
    monkeypatch.setattr(runner, "corpus_slice", lambda *args: (corpus, {}))
    monkeypatch.setattr(runner, "connection", lambda *args, **kwargs: pytest.fail("model client opened"))
    with pytest.raises(ValueError, match="scope"):
        runner.main(["--store", str(tmp_path), "--arms", "dense", "lightrag-mix"])


def test_score_only_rejects_missing_tables_before_fetching_or_calling_models(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a score-only run must fail before any work when evidence is missing")

    monkeypatch.setattr(runner, "corpus_slice", forbidden)
    monkeypatch.setattr(runner, "index_documents", forbidden)
    monkeypatch.setattr(runner, "connection", forbidden)
    with pytest.raises(FileNotFoundError, match="lightrag-mix"):
        runner.main(["--store", str(tmp_path), "--score-only", "--arms", "dense", "lightrag-mix"])


@pytest.mark.parametrize("force", ["--force-index", "--force-retrieve"])
def test_score_only_rejects_force_flags(tmp_path, force):
    with pytest.raises(ValueError, match="--score-only"):
        runner.main(["--store", str(tmp_path), "--score-only", force])


def test_summary_identifies_the_model_that_scored_the_cached_retrieval(tmp_path):
    paths = runner.Paths(store=tmp_path, out=tmp_path / "out", env=tmp_path / ".env")
    corpus = SimpleNamespace(summary=lambda: {"contracts": 20, "queries": 340})
    runner.write_outputs(paths, corpus, {}, [], {}, SpineConfig(), model="test-scoring-model")
    summary = json.loads((paths.out / "null_arm_summary.json").read_text())
    assert summary["model"] == "test-scoring-model"
    assert summary["pipeline"]["sufficient"] == 0.6
    assert summary["pipeline"]["insufficient"] == 0.0


def test_score_only_rejects_incomplete_tables_before_opening_client(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "dense.json").write_text("{}")
    runner.ResponseCache(tmp_path / "proxy_cache.sqlite")
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    monkeypatch.setattr(runner, "corpus_slice", lambda *_: (SimpleNamespace(queries=[query], chunks=[]), {}))
    monkeypatch.setattr(runner, "connection", lambda *_: pytest.fail("client opened before validation"))
    with pytest.raises(ValueError, match="IDs"):
        runner.main(["--store", str(tmp_path), "--score-only", "--arms", "dense"])


def test_cached_scoped_evidence_cannot_silently_cross_contracts():
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    table = {query.query_id: [Evidence(text="wrong contract", provenance=Provenance(doc_id="2"))]}
    with pytest.raises(ValueError, match="scope"):
        runner.validate_table(table, [query], "dense")


def test_graph_scope_restores_only_stripped_identifier():
    item = Evidence(text="original passage", provenance=Provenance(doc_id="contract"))
    restored = runner.restore_graph_scope([item], "contract ")
    assert restored[0].doc_id == "contract "
    assert restored[0].text == item.text
    assert item.doc_id == "contract"
    with pytest.raises(ValueError, match="scope"):
        runner.restore_graph_scope([item], "different contract")


def test_cached_graph_query_cost_requires_matching_evidence(tmp_path):
    import hashlib

    paths = runner.Paths(store=tmp_path, out=tmp_path / "out", env=tmp_path / ".env")
    paths.evidence.mkdir()
    evidence = paths.evidence / "lightrag-mix.json"
    evidence.write_text("{}")
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    prepared = SimpleNamespace(queries=[query])
    report = {
        "documents": 1,
        "questions": 1,
        "calls": 41,
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    runner.write_checkpoint(paths.out / "lightrag_query_cost.json", report)
    assert runner.cached_graph_costs(paths, prepared)["lightrag_query"]["calls"] == 41
    evidence.write_text('{"changed": []}')
    assert runner.cached_graph_costs(paths, prepared) == {}


def test_span_beyond_generator_passage_limit_is_not_reached():
    gold = "The receiving party must delete all copies."
    evidence = Evidence(text="x" * 1200 + gold, provenance=Provenance(doc_id="1"))
    assert not runner.visible_span_reached([evidence], [gold])
    assert runner.visible_span_reached([Evidence(text=gold, provenance=evidence.provenance)], [gold])


def test_unpacked_score_only_bundle_rejects_missing_tables_before_any_work(tmp_path, monkeypatch):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"bundle": "score-only", "files": []}))

    def forbidden(*args, **kwargs):
        pytest.fail("an incomplete bundle must fail before corpus or model access")

    monkeypatch.setattr(runner, "corpus_slice", forbidden)
    monkeypatch.setattr(runner, "connection", forbidden)
    with pytest.raises(FileNotFoundError, match="cached evidence"):
        runner.main(["--store", str(tmp_path)])


@pytest.mark.parametrize("force", ["--force-index", "--force-retrieve"])
def test_unpacked_score_only_bundle_rejects_force_flags_without_explicit_score_only(tmp_path, force):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"bundle": "score-only", "files": []}))
    with pytest.raises(ValueError, match="--score-only"):
        runner.main(["--store", str(tmp_path), force])


def test_verify_unpack_and_short_runner_command_reuses_all_arms(tmp_path, monkeypatch):
    store = tmp_path / "source"
    evidence = store / "evidence"
    evidence.mkdir(parents=True)
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    for arm in runner.RETRIEVAL_ARMS:
        (evidence / f"{arm}.json").write_text(json.dumps({query.query_id: []}))
    runner.ResponseCache(store / "proxy_cache.sqlite")
    archive = tmp_path / "null-arm-store.tar.gz"
    write_bundle(store, archive)
    assert export_main(["--verify", str(archive)]) == 0
    unpacked = tmp_path / "unpacked"
    assert not unpacked.exists()
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(unpacked, filter="data")

    def forbidden(*args, **kwargs):
        pytest.fail("unpacked evidence must never trigger indexing or retrieval")

    for name in (
        "index_documents",
        "retrieve_dense",
        "retrieve_dense_unscoped",
        "retrieve_shuffled",
        "retrieve_none",
        "retrieve_lightrag",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(runner, "corpus_slice", lambda *_: (Slice([], [query], {query.query_id: set()}), {}))
    monkeypatch.setattr(runner, "connection", forbidden)
    monkeypatch.setattr(runner, "HERE", tmp_path / "graphrag_null")
    assert runner.main(["--store", str(unpacked)]) == 0
    results = tmp_path / "outputs/graphrag_null/measured_thresholds"
    summary = json.loads((results / "null_arm_summary.json").read_text())
    assert {row["arm"] for row in summary["arms"]} == set(runner.RETRIEVAL_ARMS) | {"always-answer", "always-abstain"}
    assert summary["model"] == "Qwen3_6-35B-A3B"
    assert summary["pipeline"]["sufficient"] == 0.6
    assert summary["pipeline"]["insufficient"] == 0.0
    assert summary["cost"]["pipeline"]["calls"] == 0


def scored_store(tmp_path: Path) -> tuple[Path, Query]:
    """Return a score-only store whose dense table holds one real passage."""
    store = tmp_path / "store"
    (store / "evidence").mkdir(parents=True)
    query = Query(query_id="1::q", doc_id="1", category="c", question="Must the party delete copies?")
    item = Evidence(text="The party must delete all copies.", provenance=Provenance(doc_id="1", chunk_id="1::0000"))
    runner.save_evidence(store / "evidence" / "dense.json", {query.query_id: [item]})
    runner.ResponseCache(store / "proxy_cache.sqlite")
    return store, query


def test_score_only_cache_miss_fails_without_a_client_or_key(tmp_path, monkeypatch):
    store, query = scored_store(tmp_path)
    chunk = Chunk(chunk_id="1::0000", doc_id="1", text="The party must delete all copies.", start=0, end=33)
    monkeypatch.setattr(runner, "corpus_slice", lambda *_: (Slice([chunk], [query], {query.query_id: set()}), {}))
    monkeypatch.setattr(runner, "connection", lambda *_: pytest.fail("score-only must not open a client"))
    monkeypatch.setattr(runner, "HERE", tmp_path / "graphrag_null")
    with pytest.raises(FileNotFoundError, match="cache-only"):
        runner.main(["--store", str(store), "--score-only", "--arms", "dense", "--env", str(tmp_path / "none.env")])


def test_score_only_requires_the_response_cache(tmp_path):
    store, _ = scored_store(tmp_path)
    (store / "proxy_cache.sqlite").unlink()
    with pytest.raises(FileNotFoundError, match="response cache"):
        runner.main(["--store", str(store), "--score-only", "--arms", "dense"])
    assert not (store / "proxy_cache.sqlite").exists()


def test_unpacked_full_bundle_refuses_live_retrieval_for_a_missing_table(tmp_path, monkeypatch):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"bundle": "full", "files": []}))
    monkeypatch.setattr(runner, "corpus_slice", lambda *_: pytest.fail("refusal must come first"))
    with pytest.raises(FileNotFoundError, match="--force-retrieve"):
        runner.main(["--store", str(tmp_path), "--arms", "dense"])


def test_unpacked_store_edited_after_verification_is_refused(tmp_path):
    store, _ = scored_store(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(store, archive)
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(unpacked, filter="data")
    (unpacked / "evidence" / "dense.json").write_text("{}")
    with pytest.raises(ValueError, match="MANIFEST"):
        runner.main(["--store", str(unpacked), "--arms", "dense"])


def test_no_retrieval_table_must_be_empty():
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    table = {query.query_id: [Evidence(text="clause", provenance=Provenance(doc_id="1"))]}
    with pytest.raises(ValueError, match="no evidence"):
        runner.validate_table(table, [query], "no-retrieval")


def test_unscoped_evidence_must_come_from_the_slice():
    query = Query(query_id="1::q", doc_id="1", category="c", question="q")
    table = {query.query_id: [Evidence(text="clause", provenance=Provenance(doc_id="9"))]}
    with pytest.raises(ValueError, match="outside the slice"):
        runner.validate_table(table, [query], "dense-unscoped", {"1", "2"})


def test_lightrag_calls_keep_their_cache_key_and_retry_only_transient_errors(tmp_path, monkeypatch):
    import httpx
    import litellm
    from aieng.advanced_rag import metering
    from infermesh import LMClient

    def error(kind, status):
        response = httpx.Response(status, request=httpx.Request("POST", "http://proxy.test/v1"))
        return kind(message=f"HTTP {status}", model=runner.DEFAULT_MODEL, llm_provider="openai", response=response)

    replies = [error(litellm.RateLimitError, 429), "entities", error(litellm.BadRequestError, 400)]
    sent = []

    async def provider(**kwargs):
        sent.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return litellm.ModelResponse(
            id="r1",
            model=runner.DEFAULT_MODEL,
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )

    monkeypatch.setattr(litellm, "acompletion", provider)
    monkeypatch.setattr(metering, "before_call", lambda model: None)
    monkeypatch.setattr(LMClient, "_compute_retry_wait", staticmethod(lambda *_: 0.0))
    cache = runner.ResponseCache(tmp_path / "cache.sqlite")
    meter = runner.Meter()
    batch = runner.ProxyBatch(
        runner.DEFAULT_MODEL,
        cache=cache,
        key=runner.lightrag_key(runner.DEFAULT_MODEL),
        payload="text",
        api_key="test-key",
        base_url="http://proxy.test/v1",
    )
    llm = runner.async_completer(batch, meter)

    assert asyncio.run(llm("extract", system_prompt="system", max_tokens=99)) == "entities"
    assert len(sent) == 2
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "extract"}]
    key = runner.ResponseCache.key(
        {"model": runner.DEFAULT_MODEL, "messages": messages, "max_tokens": 99, "temperature": 0.0, "thinking": False}
    )
    assert cache.get(key) == "entities"
    assert asyncio.run(llm("extract", system_prompt="system", max_tokens=99)) == "entities"
    assert meter.calls == 1 and meter.cache_hits == 1
    with pytest.raises(litellm.BadRequestError):
        asyncio.run(llm("rejected"))
    assert len(sent) == 3
