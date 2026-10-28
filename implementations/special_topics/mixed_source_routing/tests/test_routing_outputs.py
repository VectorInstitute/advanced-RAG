"""The committed results are consistent, name their models, and carry no corpus text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
RESULTS = ("question_set.json", "results.json", "followup_sources.json", "heldout_tables.json")
# Keys under which text from the corpus or a model would sit.
TEXT_KEYS = {"question", "reference", "answer", "text", "sql", "evidence", "passage", "note", "reason"}
# Mappings keyed by tool name, where "sql" is the agent's tool, not SQL text.
TOOL_COUNTS = {"tool_mix", "tool_counts"}
SOURCE = re.compile(r"^[A-Za-z0-9_.-]+(::\d{4}|#i\d{4}@r\d+c\d+)$")


def load(name: str) -> Any:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set(value)
        for key, child in value.items():
            if key not in TOOL_COUNTS:
                found |= walk_keys(child)
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found |= walk_keys(child)
        return found
    return set()


@pytest.mark.parametrize("name", RESULTS)
def test_no_result_file_carries_corpus_or_model_text(name: str) -> None:
    assert not walk_keys(load(name)) & TEXT_KEYS


@pytest.mark.parametrize("name", RESULTS)
def test_every_result_names_the_models_that_built_it(name: str) -> None:
    produced = load(name)["produced_by"]
    assert produced["model"] == "Qwen3_6-35B-A3B"
    assert produced["judge"] == "Qwen3_6-35B-A3B"
    assert produced["endpoint"] == "vector-cluster"
    assert produced["embeddings"].startswith("BAAI/bge-small-en-v1.5")


def test_question_set_identifies_sources_by_identifier_only() -> None:
    qset = load("question_set.json")
    assert qset["counts"] == {"prose": 20, "table": 20, "both": 20}
    assert len(qset["questions"]) == 60
    for q in qset["questions"]:
        assert all(SOURCE.match(s) for s in q["sources"]), q["sources"]
        assert len(q["sources"]) == (2 if q["qtype"] == "both" else 1)


def test_results_cover_every_arm_and_question() -> None:
    results = load("results.json")
    arms = [s["arm"] for s in results["summaries"]]
    assert arms == ["U5", "U10", "RE", "RL", "B", "A", "O"]
    qids = {q["qid"] for q in load("question_set.json")["questions"]}
    rows = results["per_question"]
    assert len(rows) == len(arms) * len(qids)
    assert {r["qid"] for r in rows} == qids


def test_summaries_agree_with_the_per_question_rows() -> None:
    results = load("results.json")
    for summary in results["summaries"]:
        rows = [r for r in results["per_question"] if r["arm"] == summary["arm"]]
        assert summary["correct"] == sum(r["correct"] for r in rows)
        assert summary["calls"] == pytest.approx(sum(r["model_calls"] for r in rows) / len(rows), abs=0.01)
        for qtype, cell in summary["by_type"].items():
            typed = [r for r in rows if r["qtype"] == qtype]
            assert cell["accuracy"] == pytest.approx(sum(r["correct"] for r in typed) / len(typed), abs=1e-4)


def test_the_oracle_always_takes_the_gold_route() -> None:
    rows = [r for r in load("results.json")["per_question"] if r["arm"] == "O"]
    assert all(r["route"] == r["qtype"] for r in rows)


def test_followup_scores_every_router_under_every_route_set() -> None:
    followup = load("followup_sources.json")
    assert set(followup["results"]) == {"sql_plain", "table_chunks", "sql_linked"}
    for block in followup["results"].values():
        assert [a["arm"] for a in block["arms"]] == ["RE", "REm", "REl", "RL", "B", "O"]
        assert all(a["questions"] == 60 for a in block["arms"])


def test_the_plain_sql_route_set_matches_the_main_results() -> None:
    main = {s["arm"]: s for s in load("results.json")["summaries"]}
    sql = {a["arm"]: a for a in load("followup_sources.json")["results"]["sql_plain"]["arms"]}
    for arm in ("RE", "RL", "B", "O"):
        assert sql[arm]["correct"] == main[arm]["correct"]


def test_heldout_questions_use_filings_the_reference_set_never_touched() -> None:
    heldout = load("heldout_tables.json")
    reference = {q["filing"] for q in load("question_set.json")["questions"]}
    filings = {q["filing"] for q in heldout["per_question"]}
    assert len(heldout["per_question"]) == 20
    assert not filings & reference
    assert heldout["filings"] == len(filings)
    assert all(len(q["sources"]) == 1 and SOURCE.match(q["sources"][0]) for q in heldout["per_question"])


def test_heldout_summary_agrees_with_the_per_question_rows() -> None:
    heldout = load("heldout_tables.json")
    for name, entry in heldout["results"].items():
        if name == "paired":
            continue
        assert entry["correct"] == sum(q["correct"][name] for q in heldout["per_question"]), name
        if "sql_no_rows" in entry:
            assert entry["sql_no_rows"] == sum(q["sql_rows"][name] == 0 for q in heldout["per_question"]), name
    paired = heldout["results"]["paired"]
    answered = [q["correct"] for q in heldout["per_question"]]
    assert paired["linked_only"] == sum(a["sql_linked"] and not a["sql_plain"] for a in answered)
    assert paired["any_source"] == sum(any(a[n] for n in ("sql_plain", "sql_linked", "table_chunks")) for a in answered)


def test_heldout_reference_in_rows_counts_agree_with_the_rows() -> None:
    heldout = load("heldout_tables.json")
    for name in ("sql_plain", "sql_linked"):
        found = [q["reference_in_rows"][name] for q in heldout["per_question"]]
        assert heldout["results"][name]["sql_reference_in_rows"] == sum(found)
        # A figure cannot be in rows that were never returned.
        assert not any(f and q["sql_rows"][name] == 0 for f, q in zip(found, heldout["per_question"], strict=True))
