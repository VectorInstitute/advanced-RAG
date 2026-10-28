"""The committed results name their models, cover the full sets, and hold no text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
GEMINI_X1 = (
    "x1_musique_gemini_standard.json",
    "x1_musique_gemini_standard_repeat.json",
    "x1_musique_gemini_best_guess.json",
    "x1_musique_gemini_best_guess_top10.json",
)
RESULTS = ("x1_musique.json", "x3_serverfault.json", "x2_readiness.json", *GEMINI_X1)
# Keys under which question, passage, post, or model text would sit.
TEXT_KEYS = {"question", "answer", "text", "sql", "passage", "quote", "prompt", "reply"}


def load(name: str) -> Any:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set(value)
        for child in value.values():
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
def test_every_result_names_its_model(name: str) -> None:
    produced = load(name)["produced_by"]
    assert produced["producer"].startswith("run_graph_or_not.py")
    assert produced["endpoint"] in {"vector-cluster", "vector-proxy-paid"}
    assert produced["label"] == (
        "reference trajectories" if produced["endpoint"] == "vector-cluster" else "paid-model rerun"
    )


def test_x1_covers_ri4_questions_with_ri4_reader_and_dense() -> None:
    x1 = load("x1_musique.json")
    assert x1["test"]["questions"] == 300
    assert x1["test"]["reader"] == "Qwen3_6-35B-A3B"
    assert x1["test"]["dense_rankings_identical_to_ri4"] >= 290
    for arm, rows in x1["rows"].items():
        assert len(rows) == 300, arm
        assert f"{arm}_vs_hipporag2" in x1["comparisons"]


@pytest.mark.parametrize("name", GEMINI_X1)
def test_gemini_x1_compares_with_ri4_rows_for_the_same_reader_and_prompt(name: str) -> None:
    x1 = load(name)
    test = x1["test"]
    assert test["reader"] == x1["produced_by"]["model"] == "gemini-3.5-flash-lite"
    assert x1["produced_by"]["endpoint"] == "vector-proxy-paid"
    baseline = json.loads((OUTPUTS.parents[2] / test["baseline_rows"]).read_text(encoding="utf-8"))
    assert baseline["test"]["model"] == test["reader"]
    assert baseline["test"]["answer_prompt"] == test["answer_prompt"]
    for arm, rows in x1["rows"].items():
        assert len(rows) == 300, arm
        if arm != "hipporag2":
            assert f"{arm}_vs_hipporag2_this_run" in x1["comparisons"]


def test_x3_has_fifteen_questions_per_family_and_every_arm_scored() -> None:
    x3 = load("x3_serverfault.json")
    assert x3["test"]["per_family"] == {"count": 15, "co_mention": 15, "relation": 15, "path": 15}
    for arm, scores in x3["scores"].items():
        assert len(scores) == 60, arm
    assert 0.0 <= x3["graph_coverage"]["exact_name_recall"] <= x3["graph_coverage"]["substring_recall"] <= 1.0


def test_hand_check_summary_matches_labels() -> None:
    x3 = load("x3_serverfault.json")
    labels = load("x3_hand_check.json")
    facts = labels["relation_facts"]
    assert x3["hand_check"]["relation_facts"] == {
        "checked": len(facts),
        "supported": sum(f["supported"] for f in facts),
    }
    asked = {q["qid"] for q in x3["questions"]}
    assert {f["qid"] for f in facts} <= asked
    assert {m["qid"] for m in labels["lexical_only_mentions"]} <= asked


def test_x2_rows_hold_shares() -> None:
    rows = load("x2_readiness.json")["rows"]
    assert any(name.startswith("musique") for name in rows)
    assert any(name.startswith("serverfault") for name in rows)
    for row in rows.values():
        for key, value in row.items():
            if key.endswith("_share") and value is not None:
                assert 0.0 <= value <= 1.0, key
