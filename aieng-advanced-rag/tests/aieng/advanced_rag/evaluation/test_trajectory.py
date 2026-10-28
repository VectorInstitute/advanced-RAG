"""Tests for agent trajectory scores and the Langfuse score recorder."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from aieng.advanced_rag.evaluation import trajectory
from aieng.advanced_rag.evaluation.trajectory import (
    score_agent_run,
    tools_in_order,
    trajectory_scores,
)
from aieng.advanced_rag.tracing.scores import record_scores


def make_run(text="Ontario law governs.", tools=("search_corpus", "fetch_passages")):
    return SimpleNamespace(
        text=text,
        tool_calls=tuple(tools),
        model_calls=3,
        elapsed_seconds=1.23456,
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        abstained=not text.strip(),
    )


class FakeLoop:
    def __init__(self, run):
        self.result = run
        self.asked = []

    def run(self, question):
        self.asked.append(question)
        return self.result


JUDGE_REPLY = json.dumps(
    {"correct": True, "grounded": True, "abstained": False, "reason": "matches"}
)


def test_tools_in_order_counts_the_matched_prefix():
    assert tools_in_order(["a", "b"], ["a", "x", "b"]) == 1.0
    assert tools_in_order(["a", "b"], ["b", "a"]) == 0.5
    assert tools_in_order(["a", "b"], []) == 0.0
    assert tools_in_order([], ["a"]) == 1.0


def test_trajectory_scores_describe_the_path():
    scores = trajectory_scores(
        make_run(tools=("search_corpus", "search_corpus", "fetch_passages")),
        expected_tools=["search_corpus", "fetch_passages"],
    )
    assert scores["tool_calls"] == 3.0
    assert scores["distinct_tools"] == 2.0
    assert scores["repeated_tool_calls"] == 1.0
    assert scores["model_calls"] == 3.0
    assert scores["prompt_tokens"] == 100.0
    assert scores["elapsed_seconds"] == 1.235
    assert scores["abstained"] == 0.0
    assert scores["tools_in_order"] == 1.0


def test_an_empty_answer_scores_as_abstained_and_expected_tools_are_optional():
    scores = trajectory_scores(make_run(text=" ", tools=()))
    assert scores["abstained"] == 1.0
    assert "tools_in_order" not in scores


def test_scoring_a_run_judges_it_and_records_the_scores(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        trajectory,
        "record_scores",
        lambda trace_id, scores, comment=None: (
            sent.update(trace_id=trace_id, scores=scores, comment=comment)
            or len(scores)
        ),
    )
    loop = FakeLoop(make_run())
    result = score_agent_run(
        loop,
        "which law governs?",
        name="t",
        reference="Ontario",
        complete=lambda prompt: JUDGE_REPLY,
    )
    assert loop.asked == ["which law governs?"]
    assert result.scores["correct"] == 1.0
    assert result.judgement is not None and result.judgement.correct
    assert result.sent == len(result.scores)
    assert sent["comment"] == "matches"


def test_without_a_reference_there_is_no_judge_call():
    calls = []
    result = score_agent_run(
        FakeLoop(make_run()),
        "q",
        name="t",
        complete=lambda prompt: calls.append(prompt) or JUDGE_REPLY,
    )
    assert calls == []
    assert "correct" not in result.scores
    assert result.trace_id is None and result.sent == 0


def test_record_scores_is_a_no_op_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert record_scores("trace-1", {"x": 1.0}) == 0
    assert record_scores(None, {"x": 1.0}) == 0


def test_record_scores_sends_each_score_as_a_number(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    created = []

    class Client:
        def create_score(self, **kwargs):
            created.append(kwargs)

        def flush(self):
            created.append("flushed")

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=Client))
    sent = record_scores("t1", {"correct": True, "tool_calls": 2}, comment="why")
    assert sent == 2
    assert created[0] == {
        "trace_id": "t1",
        "name": "correct",
        "value": 1.0,
        "data_type": "NUMERIC",
        "comment": "why",
    }
    assert created[-1] == "flushed"


def test_a_missing_sdk_sends_nothing(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert record_scores("t1", {"x": 1.0}) == 0


@pytest.mark.parametrize("trace_id", ["", None])
def test_no_trace_id_means_nothing_is_sent(trace_id):
    assert record_scores(trace_id, {"x": 1.0}) == 0


ABSTAINING_JUDGE = json.dumps(
    {"correct": True, "grounded": True, "abstained": True, "reason": "no info"}
)


def test_an_abstention_is_not_correct_against_a_specific_reference():
    result = score_agent_run(
        FakeLoop(make_run(text="The corpus does not say.")),
        "q",
        name="t",
        reference="SUPPLY CONTRACT",
        complete=lambda prompt: ABSTAINING_JUDGE,
    )
    assert result.judgement is not None and result.judgement.correct
    assert result.scores["correct"] == 0.0


def test_an_abstention_is_correct_when_the_reference_says_the_information_is_absent():
    result = score_agent_run(
        FakeLoop(make_run(text="The corpus does not say.")),
        "q",
        name="t",
        reference="Not stated in the provided sources.",
        complete=lambda prompt: ABSTAINING_JUDGE,
    )
    assert result.scores["correct"] == 1.0
