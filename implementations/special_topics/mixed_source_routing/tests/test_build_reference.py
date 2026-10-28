"""The pipeline pieces that need no model or corpus behave as documented."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
from routing import Reply


# Other topics also ship a build_reference.py, and CI collects every
# implementation's tests in one session, so a bare import can return theirs.
_SPEC = importlib.util.spec_from_file_location(
    "mixed_source_routing_build_reference",
    Path(__file__).resolve().parents[1] / "build_reference.py",
)
assert _SPEC is not None and _SPEC.loader is not None
br = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = br
_SPEC.loader.exec_module(br)


def test_extract_sql_reads_a_fenced_block_or_the_bare_reply() -> None:
    assert br.extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert br.extract_sql("SELECT 2;") == "SELECT 2"


def test_render_rows_caps_its_output() -> None:
    text = br.render_rows(["a", "b"], [("x" * 100, None)] * 200)
    assert text.startswith("a | b\n")
    assert text.endswith("(more rows left out)")
    assert len(text) <= br.SQL_CHARS + 200


@pytest.mark.parametrize(
    ("sql_calls", "text_calls", "route"),
    [(2, 3, "both"), (1, 0, "table"), (0, 4, "prose"), (0, 0, "prose")],
)
def test_agent_route(sql_calls: int, text_calls: int, route: str) -> None:
    assert br.agent_route(sql_calls, text_calls) == route


def test_fold_ignores_case_and_punctuation() -> None:
    assert br.fold("St. Paul, Minnesota") == br.fold("st paul minnesota")


class FakeTables:
    """Answer the second query only, as a model that needed one retry would see."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def schema(self) -> str:
        return "cells (10 rows)"

    def query(self, sql: str, *, max_rows: int) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        self.queries.append(sql)
        if "bad" in sql:
            raise sqlite3.OperationalError("no such column: bad")
        if len(self.queries) == 1:
            return ["text"], [], False
        return ["row_label", "text"], [("Net sales", "$ 32,765")], False


def test_sql_evidence_retries_once_with_feedback() -> None:
    prompts: list[str] = []

    def complete(prompt: str, max_tokens: int) -> Reply:
        prompts.append(prompt)
        return Reply("```sql\nSELECT text FROM cells\n```", 50, 5)

    sources = br.Sources(index=None, tables=FakeTables(), complete=complete, labels={})  # type: ignore[arg-type]
    found = sources.sql_evidence("What were net sales in 2018?")
    assert [a["rows"] for a in found["attempts"]] == [0, 1]
    assert "returned no rows" in prompts[1]
    assert "Net sales | $ 32,765" in found["evidence"]
    assert found["cost"] == {"model_calls": 2, "prompt_tokens": 100, "completion_tokens": 10}


def test_sql_evidence_reports_errors_and_gives_up_after_two_attempts() -> None:
    sources = br.Sources(
        index=None,  # type: ignore[arg-type]
        tables=FakeTables(),
        complete=lambda p, n: Reply("SELECT bad FROM cells"),  # type: ignore[arg-type]
        labels={},
    )
    found = sources.sql_evidence("q")
    assert len(found["attempts"]) == 2
    assert all(a["error"] for a in found["attempts"])
    assert found["evidence"] == ""


def _questions() -> list[br.Question]:
    return [
        br.Question("q1", "table", "F", "What was revenue?", "10"),
        br.Question("q2", "both", "F", "Revenue and auditor?", "(1) 10; (2) X"),
    ]


def _pipelines() -> list[dict[str, Any]]:
    def route(calls: int, tokens: int) -> dict[str, Any]:
        return {"answer": "a", "chunk_ids": [], "model_calls": calls, "prompt_tokens": tokens, "completion_tokens": 1}

    rows = []
    for qid, qtype in (("q1", "table"), ("q2", "both")):
        rows.append(
            {
                "qid": qid,
                "qtype": qtype,
                "model": "m",
                "sql": [],
                "routes": {
                    "prose": route(1, 100),
                    "table": route(2, 200),
                    "both": route(2, 300),
                    "U5": route(1, 110),
                    "U10": route(1, 220),
                },
                "llm_route": {"route": "table", "model_calls": 1, "prompt_tokens": 50, "completion_tokens": 5},
            }
        )
    return rows


def test_arm_records_add_the_router_cost_to_the_chosen_route() -> None:
    verdicts = [
        {"qid": "q1", "route": "table", "correct": True},
        {"qid": "q2", "route": "both", "correct": True},
        {"qid": "q2", "route": "table", "correct": False},
        {"qid": "q1", "route": "A", "correct": True},
    ]
    agent = [
        {
            "qid": "q1",
            "qtype": "table",
            "model": "m",
            "route": "table",
            "model_calls": 6,
            "prompt_tokens": 9000,
            "completion_tokens": 90,
            "exhausted": False,
            "tool_counts": {"sql": 2},
        }
    ]
    embedded = {"q1": {"route": "prose", "model_calls": 0}, "q2": {"route": "both", "model_calls": 0}}
    records = br.arm_records(_questions(), _pipelines(), agent, verdicts, embedded)
    by = {(r["qid"], r["arm"]): r for r in records}
    assert by[("q1", "RL")]["model_calls"] == 3
    assert by[("q1", "RL")]["prompt_tokens"] == 250
    assert by[("q1", "RL")]["correct"] is True
    assert by[("q2", "RL")]["correct"] is False
    assert by[("q2", "O")]["route"] == "both"
    assert by[("q2", "O")]["correct"] is True
    assert by[("q1", "RE")]["route"] == "prose"
    assert by[("q1", "U5")]["route"] == ""
    assert by[("q1", "A")]["model_calls"] == 6

    rows, outcome = br.summaries(records, verdicts)
    arms = {r["arm"]: r for r in rows}
    assert arms["RL"]["routing"]["mismatches"] == 1
    assert arms["RL"]["routing"]["costly"] == 1
    assert arms["A"]["tool_mix"] == {"sql": 2}
    assert outcome["both"]["hindsight"] == {"both": 1}
    assert outcome["table"]["route_accuracy"]["table"] == 1.0


def test_capped_keeps_every_question_on_a_cheap_model() -> None:
    questions = [br.Question(f"q{i}", t, "F", "q", "r") for i in range(10) for t in ("prose", "table", "both")]
    assert len(br.capped(questions, "Qwen3_6-35B-A3B")) == 30
    assert len(br.capped(questions, "claude-opus-5-5")) == 3 * br.LIVE_CAP_PER_TYPE


def test_question_round_trips_through_its_record() -> None:
    question = br.Question("q", "both", "F", "text", "ref", ("a", "b"))
    assert br.Question.from_dict(question.to_dict()) == question


def test_funnel_counts_entries_and_passes() -> None:
    funnel = br.Funnel()
    funnel.check("table", "written", True)
    funnel.check("table", "written", False)
    assert funnel.to_rows() == [{"kind": "table", "stage": "written", "entered": 2, "passed": 1}]


def test_heldout_summary_counts_answers_rows_and_paired_wins() -> None:
    def run(qid: str, plain_rows: int, linked_rows: int) -> dict[str, Any]:
        cost = {"model_calls": 2, "prompt_tokens": 100, "completion_tokens": 10}
        return {
            "qid": qid,
            "sql": {
                "sql_plain": [{"sql": "s", "error": "", "rows": plain_rows}],
                "sql_linked": [{"sql": "s", "error": "", "rows": linked_rows}],
            },
            "routes": {name: {"answer": "a", **cost} for name in br.HELDOUT_ROUTES},
        }

    questions = [br.Question(q, "table", "F", "q", "r") for q in ("q1", "q2")]
    runs = [run("q1", 0, 3), run("q2", 4, 0)]
    right = {("q1", "sql_linked"), ("q2", "sql_plain"), ("q2", "table_chunks")}
    verdicts = [{"qid": q, "route": n, "correct": (q, n) in right} for q in ("q1", "q2") for n in br.HELDOUT_ROUTES]
    summary = br.heldout_summary(questions, runs, verdicts)
    assert summary["sql_plain"]["correct"] == 1
    assert summary["sql_plain"]["sql_no_rows"] == 1
    assert summary["sql_linked"]["sql_no_rows"] == 1
    assert "sql_no_rows" not in summary["table_chunks"]
    assert summary["paired"] == {"linked_only": 1, "plain_only": 1, "any_source": 2}
