"""Checks on the committed reference results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ARMS = ("answer", "inline", "quotes", "fulldoc", "shuffled", "rank1", "posthoc")
RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "results.json"


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_the_results_name_the_models_that_produced_them(results: dict) -> None:
    produced = results["produced_by"]
    assert produced["model"] == "Qwen3_6-35B-A3B"
    assert produced["judging"]["model"] == "Qwen3_6-35B-A3B"
    assert produced["local_checker"].startswith("lytang/MiniCheck")


def test_every_arm_is_scored_by_gold_and_by_both_checkers(results: dict) -> None:
    for arm in ARMS:
        row = results["all"]["arms"][arm]
        assert row["llm"]["gold_precision"] is not None
        assert row["llm"]["judged_recall"] is not None
        assert row["minicheck"]["judged_recall"] is not None
        for checker in ("llm", "minicheck"):
            table = row["agreement"][checker]["citation"]
            assert table["total"] == table["both"] + table["judge_only"] + table["gold_only"] + table["neither"]


def test_the_shuffled_control_loses_gold_precision(results: dict) -> None:
    arms = results["all"]["arms"]
    assert arms["shuffled"]["llm"]["gold_precision"] < arms["inline"]["llm"]["gold_precision"]


def test_the_committed_records_carry_no_contract_text(results: dict) -> None:
    for row in results["questions"]:
        assert set(row) <= {"query_id", "label", "gold_spans", *ARMS}


def test_the_question_set_matches_its_description(results: dict) -> None:
    counts = results["all"]
    assert counts["documents"] == 24
    assert counts["questions"] == counts["labelled"] + counts["nulls"]
    assert counts["nulls"] == 48
