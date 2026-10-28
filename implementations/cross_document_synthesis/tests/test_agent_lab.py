"""Migrated lab: bounded local tools, independent grounding, and proxy configuration."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aieng.advanced_rag.corpora.musique import Passage
from synthesis_lab.agent_lab import MusiqueTools, parse_lab_answer
from synthesis_lab.runtime import PROXY_URL


def tools():
    passages = [
        Passage("p0", "Person", "Alice was born in Canada."),
        Passage("p1", "Country", "Canada is in North America."),
    ]
    return MusiqueTools(passages, np.eye(2, dtype=np.float32), lambda texts: [[1.0, 0.0] for _ in texts], k=1)


def test_search_requires_plan_and_runs_over_corpus_with_local_encoder():
    lab = tools()
    assert "error" in lab.search_musique("Alice")
    assert lab.searches == 0
    lab.plan_retrieval(["Find birthplace", "Find continent"])
    hits = lab.search_musique("Alice")
    assert hits["hits"][0]["passage_id"] == "p0"
    assert lab.exposed_ids == {"p0"}
    assert not lab.read_ids


def test_budget_cannot_be_reset_and_covers_dense_and_lexical_search():
    lab = tools()
    lab.plan_retrieval(["Find continent"])
    lab.search_musique("Alice")
    lab.grep_musique("NORTH AMERICA")
    lab.search_musique("Canada")
    assert "error" in lab.plan_retrieval(["Restart"])
    assert "error" in lab.grep_musique("Alice")
    assert lab.searches == 3
    assert [t["search_number"] for t in lab.trace if "search_number" in t] == [1, 2, 3]


def test_reads_require_real_search_ids_and_are_recorded_independently():
    lab = tools()
    assert "error" in lab.read_musique(["p0"])
    lab.plan_retrieval(["Find person"])
    lab.search_musique("Alice")
    assert "error" in lab.read_musique(["p1"])
    assert lab.read_musique(["p0"])["passages"][0]["text"].endswith("Canada.")
    assert lab.read_ids == {"p0"}
    assert lab.trace[-1] == {"tool": "read_musique", "passage_ids": ["p0"]}


def test_reflection_requires_read_evidence_and_state_is_per_question():
    first = tools()
    assert "error" in first.reflect_evidence([])
    first.plan_retrieval(["Find continent"])
    first.grep_musique("North America")
    first.read_musique(["p1"])
    assert first.reflect_evidence([])["missing"] == []
    second = tools()
    assert not second.plan and not second.exposed_ids and not second.read_ids
    assert second.searches == 0


def test_answer_contract_requires_citations_to_read_passages():
    lab = tools()
    payload = {"status": "answered", "answer": "Canada", "citations": ["p0"]}
    lab.plan_retrieval(["Find birthplace"])
    lab.search_musique("Alice")
    with pytest.raises(ValueError, match="actually read"):
        parse_lab_answer(json.dumps(payload), lab)
    lab.read_musique(["p0"])
    assert parse_lab_answer(json.dumps(payload), lab) == payload
    payload["citations"] = ["p0", "p0"]
    with pytest.raises(ValueError, match="uniquely"):
        parse_lab_answer(json.dumps(payload), lab)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": []},
        {"status": "answered", "answer": "", "citations": []},
        {"status": "not_stated", "answer": "guess", "citations": []},
        {"status": "answered", "answer": "Canada", "citations": [1]},
    ],
)
def test_malformed_or_ungrounded_outputs_are_rejected(payload):
    with pytest.raises(ValueError):
        parse_lab_answer(json.dumps(payload), tools())


def test_not_stated_is_an_explicit_empty_output():
    payload = {"status": "not_stated", "answer": "", "citations": []}
    assert parse_lab_answer(json.dumps(payload), tools()) == payload


@pytest.mark.parametrize("path", ["synthesis_lab/agent_lab.py", "run_agent_lab.py"])
def test_migrated_lab_has_no_hosted_embeddings_or_legacy_endpoint(path):
    source = (Path(__file__).resolve().parents[1] / path).read_text()
    assert "embeddings.create" not in source
    assert "text-embedding-" not in source
    assert "api.openai.com" not in source
    assert "OPENAI_REASONING_EFFORT" not in source


def test_live_builder_pins_proxy_and_uses_shared_loop_and_meter(monkeypatch):
    from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: PLC0415
    from synthesis_lab.agent_lab import LAB_INSTRUCTION  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.invalid/v1")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    loop = AgentLoop(tools().functions(), base_url=PROXY_URL, api_key="test-only", instruction=LAB_INSTRUCTION)
    agent = loop.build_agent()
    assert loop.base_url == PROXY_URL
    assert agent.model._additional_args["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in agent.model._additional_args


@pytest.mark.asyncio
async def test_model_call_limit_abstains_and_preserves_spent_usage(monkeypatch, tmp_path):
    import run_agent_lab as runner  # noqa: PLC0415
    from aieng.advanced_rag.agents import loop as loop_module  # noqa: PLC0415
    from aieng.advanced_rag.corpora.musique import MusiqueQuestion  # noqa: PLC0415
    from aieng.advanced_rag.metering import CallUsage, report_call  # noqa: PLC0415
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError  # noqa: PLC0415

    class CappedLoop:
        def __init__(self, functions, **kwargs):
            assert kwargs["base_url"] == PROXY_URL
            assert not kwargs["enable_thinking"]

        async def arun(self, question):
            for _ in range(10):
                report_call(CallUsage(1, 1, 2, reported=True), 0.1)
            raise LlmCallsLimitExceededError("test cap")

    passages = [Passage("p0", "Title", "Text")]
    questions = [MusiqueQuestion(f"{h}hop__1_2", f"asked {h}", "gold", (), h, ("p0",), ()) for h in (2, 3, 4)]
    monkeypatch.setattr(loop_module, "AgentLoop", CappedLoop)
    monkeypatch.setattr(runner, "resolve_api_key", lambda: "test-only")
    monkeypatch.setattr(runner, "fetch_musique", lambda **kwargs: None)
    monkeypatch.setattr(runner, "load_musique", lambda paths: (passages, questions))
    monkeypatch.setattr(runner, "local_embedder", lambda path: SimpleNamespace(encode_queries=lambda texts: [[1.0]]))
    monkeypatch.setattr(runner, "embeddings", lambda *args, **kwargs: (np.ones((1, 1)), None, {}))
    output = tmp_path / "capped.json"
    report = await runner.run_live(tmp_path, output=output)
    assert len(report["results"]) == 3
    assert all(row["stop_reason"] == "model_call_limit" for row in report["results"])
    assert all(row["status"] == "not_stated" and not row["answer"] for row in report["results"])
    assert report["usage"]["run"]["calls"] == 30
    assert report["usage"]["run"]["total_tokens"] == 60
    assert json.loads(output.read_text())["usage"] == report["usage"]
