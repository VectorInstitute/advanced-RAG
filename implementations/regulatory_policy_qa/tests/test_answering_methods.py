"""Guard experiment isolation, source evidence, and offline replay behavior."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RI))

import run_threshold_sweep as threshold_runner  # noqa: E402
from advanced_lab import agent as agent_lab  # noqa: E402
from advanced_lab import common  # noqa: E402
from advanced_lab import scope as retrieval_lab  # noqa: E402
from advanced_lab.agent import audited_tools  # noqa: E402
from advanced_lab.answering import TextExtractor, balanced_sample, source_quote  # noqa: E402
from advanced_lab.null_comparison import (  # noqa: E402
    paired_decisions,
    reader_budgets,
    validate_pair,
    verify_judging_sources,
)
from advanced_lab.scope import retrieval_scores  # noqa: E402
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query  # noqa: E402
from aieng.advanced_rag.retrieval.sparse import BM25Retriever  # noqa: E402


def test_balanced_selection_preserves_document_and_answerability():
    queries = [Query(str(i), "a", "x", "q", (AnswerSpan("yes", 0, 3),) if i % 2 else ()) for i in range(6)]
    assert [q.query_id for q in balanced_sample(queries)] == ["0", "1"]


def test_mrr_cannot_credit_gold_beyond_the_evaluation_cutoff():
    ranks = {"q": [f"irrelevant-{i}" for i in range(10)] + ["gold"]}
    assert retrieval_scores(ranks, {"q": {"gold"}}, k=10) == {"retrieval_failure": 1.0, "mrr": 0.0}


def test_source_quotes_preserve_the_actual_original_span():
    assert source_quote("A party\n  shall retain records.", "party shall retain") == "party\n  shall retain"
    with pytest.raises(ValueError, match="wording"):
        source_quote("A party shall retain records.", "A party may retain records.")


def test_agent_fetch_cannot_escape_a_physically_isolated_contract_index():
    retriever = BM25Retriever()
    retriever.index([Chunk("doc-a::0", "doc-a", "Confidential records must be deleted.", 0, 36)])
    trace = []
    tools = {tool.__name__: tool for tool in audited_tools(retriever, "doc-a", trace)}
    assert tools["fetch_passages"](["doc-b::0"])["passages"] == []
    found = tools["search_corpus"]("confidential records")
    assert found["passages"][0]["chunk_id"] == "doc-a::0"
    assert trace[-1]["returned_chunk_ids"] == ["doc-a::0"]


@pytest.mark.parametrize("custom_corpus", [False, True])
def test_editable_agent_factory_is_used_with_isolated_indexes(monkeypatch, custom_corpus):
    chunks = [Chunk(f"{doc}::0", doc, "confidential records", 0, 20) for doc in ("a", "b")]
    queries = [
        Query(f"{doc}-{null}", doc, "x", "question", () if null else (AnswerSpan("records", 13, 20),))
        for doc in ("a", "b")
        for null in (True, False)
    ]
    corpus = SimpleNamespace(chunks=chunks, queries=queries)
    monkeypatch.setattr(agent_lab.contract_nli, "fetch_contract_nli", lambda: None)
    monkeypatch.setattr(agent_lab.contract_nli, "build_slice", lambda *args, **kwargs: corpus)
    monkeypatch.setattr(agent_lab, "SentenceTransformerEmbedder", lambda *args: None)
    monkeypatch.setattr(agent_lab, "DenseRetriever", lambda _: BM25Retriever())
    monkeypatch.setattr(agent_lab, "write_json", lambda *args: None)
    scopes = []

    class Loop:
        model = common.MODEL
        max_model_calls = 3

        async def arun(self, question):
            return SimpleNamespace(text="offline test", tool_calls=[], model_calls=0, abstained=False)

    def factory(retriever, scope, question_meter, trace):
        assert set(retriever.store.document_ids()) == {scope}
        scopes.append(scope)
        return Loop()

    report = asyncio.run(
        agent_lab.run_agent_examples(factory, corpus=corpus, labels_known=False)
        if custom_corpus
        else agent_lab.run_agent_examples(factory)
    )
    assert scopes == ["a", "a", "b", "b"]
    assert report["max_model_calls"] == 3 and len(report["results"]) == 4
    assert [row["is_null"] for row in report["results"]] == ([None] * 4 if custom_corpus else [True, False] * 2)


@pytest.mark.parametrize("case", ["empty", "too-many", "foreign-document"])
def test_invalid_starter_corpus_fails_before_model_loading(monkeypatch, case):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid corpus must be rejected before embedding or agent calls")

    monkeypatch.setattr(agent_lab, "SentenceTransformerEmbedder", forbidden)
    queries = (
        []
        if case == "empty"
        else [
            Query(str(i), "foreign" if case == "foreign-document" else "a", "x", "q", ())
            for i in range(21 if case == "too-many" else 1)
        ]
    )
    corpus = SimpleNamespace(chunks=[Chunk("a::0", "a", "text", 0, 4)], queries=queries)
    with pytest.raises(ValueError):
        asyncio.run(agent_lab.run_agent_examples(forbidden, corpus=corpus, labels_known=False))


def test_scope_audit_retrieves_globally_instead_of_filtering_to_an_empty_doc_id(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text("{}")
    chunks = [Chunk(f"{doc}::0", doc, "confidential records", 0, 20) for doc in ("a", "b")]
    queries = [Query(doc, doc, "x", "confidential records", (AnswerSpan("records", 13, 20),)) for doc in ("a", "b")]
    corpus = SimpleNamespace(chunks=chunks, queries=queries, qrels={doc: {f"{doc}::0"} for doc in ("a", "b")})
    monkeypatch.setattr(retrieval_lab, "require_compute", lambda: None)
    monkeypatch.setattr(retrieval_lab.contract_nli, "fetch_contract_nli", lambda: path)
    monkeypatch.setattr(retrieval_lab.contract_nli, "build_slice", lambda *args, **kwargs: corpus)
    monkeypatch.setattr(retrieval_lab, "SentenceTransformerEmbedder", lambda *args: None)
    monkeypatch.setattr(retrieval_lab, "DenseRetriever", lambda _: BM25Retriever())
    monkeypatch.setattr(retrieval_lab, "write_json", lambda *args: None)
    report = retrieval_lab.measure_scope()
    assert report["arms"]["scoped"]["cross_contract_hits"] == 0
    assert report["arms"]["unscoped"]["cross_contract_hits"] == 2


def test_government_text_extraction_discards_scripts():
    parser = TextExtractor()
    parser.feed("<p>Public statute.</p><script>not source text</script><style>hidden</style>")
    assert parser.parts == ["Public statute."]


def test_paired_analysis_handles_identical_and_one_sided_decisions():
    assert paired_decisions([True], [True])["mcnemar_exact_two_sided_p"] == 1
    assert paired_decisions([True] * 5, [False] * 5)["mcnemar_exact_two_sided_p"] == 0.0625
    with pytest.raises(ValueError, match="equally sized"):
        paired_decisions([True], [])


def test_paired_analysis_rejects_answerability_mismatch():
    first = {"q": {"doc_id": "a", "is_null": False, "abstained": False}}
    second = {"q": {"doc_id": "a", "is_null": True, "abstained": False}}
    with pytest.raises(ValueError, match="labels differ"):
        validate_pair({"dense": first, "lightrag-mix": second})


def test_reader_budget_counts_only_truncated_passages(tmp_path):
    (tmp_path / "evidence").mkdir()
    for arm in ("dense", "lightrag-mix"):
        (tmp_path / "evidence" / f"{arm}.json").write_text(json.dumps({"q": [{"text": "x" * 2000}] * 6}))
    report = reader_budgets(tmp_path, ["q"], 5)
    assert report["dense"]["mean"] == 6000


def test_paired_analysis_rejects_judgments_from_changed_answers(tmp_path):
    evidence = tmp_path / "store/evidence"
    evidence.mkdir(parents=True)
    inputs = {"null_arm_summary.json": tmp_path / "null_arm_summary.json"}
    judging = {"report_sha256": {}, "input_sha256": {}}
    for arm in ("dense", "lightrag-mix"):
        inputs[f"outcomes_{arm}.json"] = tmp_path / f"outcomes_{arm}.json"
        inputs[f"evidence/{arm}.json"] = evidence / f"{arm}.json"
        path = tmp_path / f"answer_judgements_{arm}.json"
        path.write_text("{}")
        judging["report_sha256"][path.name] = common.digest(path)
    for role, path in inputs.items():
        path.write_text("{}")
        judging["input_sha256"][role] = common.digest(path)
    verify_judging_sources(tmp_path, tmp_path / "store", judging)
    (tmp_path / "outcomes_dense.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="judging input hash mismatch"):
        verify_judging_sources(tmp_path, tmp_path / "store", judging)


def test_threshold_retrieval_honors_shared_interface_depth(monkeypatch):
    calls = []

    class Retriever:
        name = "dense"

        def __init__(self, _):
            pass

        def index(self, chunks):
            assert chunks == []

        def search(self, query, *, k):
            calls.append((query.question, query.doc_id, k))
            return []

    monkeypatch.setattr(threshold_runner, "SentenceTransformerEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr(threshold_runner, "DenseRetriever", Retriever)
    retrieve, _ = threshold_runner.build_retrieve([], device="cpu", top_k=10)
    assert retrieve("question", "doc", 3) == []
    assert calls == [("question", "doc", 3)]


def test_threshold_retrieval_honors_unscoped_shared_interface(monkeypatch):
    monkeypatch.setattr(threshold_runner, "SentenceTransformerEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr(threshold_runner, "DenseRetriever", lambda _: BM25Retriever())
    chunks = [Chunk(f"{doc}::0", doc, "confidential records", 0, 20) for doc in ("a", "b")]
    retrieve, _ = threshold_runner.build_retrieve(chunks, device="cpu", top_k=10)
    assert {item.provenance.doc_id for item in retrieve("records", None, 2)} == {"a", "b"}


def test_cache_only_sweep_rejects_incomplete_ids_before_model_or_retrieval(tmp_path, monkeypatch):
    query = Query("q1", "doc", "x", "question")
    corpus = SimpleNamespace(chunks=[], summary=lambda: {}, queries=[query])
    monkeypatch.setattr(threshold_runner, "load_arm", lambda *args, **kwargs: (corpus, [query], "test"))
    monkeypatch.setattr(
        threshold_runner, "build_retrieve", lambda *args, **kwargs: pytest.fail("retrieval is forbidden")
    )
    monkeypatch.setattr(
        threshold_runner, "ProxyClient", lambda *args, **kwargs: pytest.fail("client creation is forbidden")
    )
    for name in ("heuristic", "llm"):
        (tmp_path / f"traces_contract-nli_{name}.json").write_text("[]")
    with pytest.raises(ValueError, match="cached trace IDs"):
        threshold_runner.main(["--cache-only", "--replay-dir", str(tmp_path)])


def test_cached_sweep_cannot_relabel_recorded_generation_as_a_smaller_reader_budget(tmp_path, monkeypatch):
    query = Query("q", "doc", "x", "question")
    corpus = SimpleNamespace(summary=lambda: {}, queries=[query], chunks=[])
    monkeypatch.setattr(threshold_runner, "load_arm", lambda *args, **kwargs: (corpus, [query], "test"))
    monkeypatch.setattr(threshold_runner, "ProxyClient", lambda *args, **kwargs: pytest.fail("no client allowed"))
    with pytest.raises(ValueError, match="different reader budget"):
        threshold_runner.main(["--cache-only", "--replay-dir", str(tmp_path), "--evidence-budget", "3"])


def test_unknown_grader_is_not_silently_measured_as_heuristic(monkeypatch):
    monkeypatch.setattr(threshold_runner, "load_arm", lambda *args, **kwargs: pytest.fail("invalid arm accepted"))
    with pytest.raises(ValueError, match="graders must"):
        threshold_runner.main(["--graders", "misspelled-grader", "--estimate-only"])


def test_coverage_audit_reports_missing_and_duplicate_ids_without_model_calls(tmp_path, monkeypatch):
    queries = [Query(q, "doc", "x", "question") for q in ("q1", "q2")]
    corpus = SimpleNamespace(summary=lambda: {"queries": 2})
    monkeypatch.setattr(threshold_runner, "load_arm", lambda *args, **kwargs: (corpus, queries, "test"))
    monkeypatch.setattr(threshold_runner, "ProxyClient", lambda *args, **kwargs: pytest.fail("no client allowed"))
    source = tmp_path / "corpus.json"
    source.write_text("{}")
    monkeypatch.setattr(threshold_runner.contract_nli, "fetch_contract_nli", lambda: source)
    for name in ("heuristic", "llm"):
        (tmp_path / f"traces_contract-nli_{name}.json").write_text(json.dumps([{"query_id": "q1"}] * 2))
    out = tmp_path / "out"
    assert threshold_runner.main(["--coverage-only", "--replay-dir", str(tmp_path), "--output-dir", str(out)]) == 0
    report = json.loads((out / "cache_status.json").read_text())
    assert report["new_model_calls"] == 0 and report["full_split_measurement_complete"] is False
    assert report["threshold_trace_coverage"]["llm"]["duplicate_rows"] == 1
    assert report["threshold_trace_coverage"]["llm"]["missing_queries"] == 1
