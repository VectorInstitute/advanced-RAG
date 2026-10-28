"""The committed results are consistent, name their models, and carry no corpus text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest


HERE = Path(__file__).resolve().parents[1]
OUTPUTS = HERE / "outputs"
RESULTS = [
    "question_set.json",
    "results.json",
    "costs.json",
    "description_audit.json",
    "followup_e2_pictures_only.json",
]
ROWS = ("T", "E1", "E2", "L", "VT", "VV", "TV", "H", "O-text", "O-enriched", "O-image", "O-both")
GROUPS = ("figure-only", "duplicated", "table", "control", "null")
# Keys under which text from the corpus or a model would sit.
TEXT_KEYS = {"question", "answer", "statement", "text", "description", "summary", "changed", "chart_label"}
PAGE = re.compile(r"^[A-Za-z0-9_.-]+#p\d{3}$")


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
    payload = load(name)
    assert not walk_keys(payload) & TEXT_KEYS


@pytest.mark.parametrize("name", RESULTS)
def test_every_result_names_the_models_that_built_it(name: str) -> None:
    produced = load(name)["produced_by"]
    assert produced["model"] == "Qwen3_8-27B"
    assert produced["reader"] == "Qwen3_8-27B"
    assert produced["question_checker"] == "Qwen3_8-Flash-Next"
    assert produced["judge"] == "Qwen3_6-35B-A3B"
    assert produced["endpoint"] == "vector-cluster"


def test_question_identifiers_are_unique_and_pages_are_identifiers() -> None:
    summary = load("question_set.json")
    ids = [q["query_id"] for q in summary["questions"]]
    assert len(ids) == len(set(ids))
    for q in summary["questions"]:
        for page in q["gold_pages"]:
            assert PAGE.match(page), page
        assert (q["group"] == "null") == (not q["gold_pages"])
    assert set(summary["subset"]) <= set(ids)
    counts: dict[str, int] = {}
    for q in summary["questions"]:
        group = q["subgroup"] if q["group"] == "figure" else q["group"]
        counts[group] = counts.get(group, 0) + 1
    assert counts == summary["counts"]


def test_the_funnel_never_passes_more_than_it_admits() -> None:
    for row in load("question_set.json")["funnel"]:
        assert 0 <= row["passed"] <= row["entered"]


def test_results_cover_eight_arms_and_four_oracles() -> None:
    results = load("results.json")
    assert tuple(results["answers"]) == ROWS
    assert set(results["retrieval"]) == set(ROWS[:8])
    for row in results["answers"].values():
        for group in GROUPS:
            rate = row["accuracy"][group]
            assert rate is None or 0.0 <= rate <= 1.0
        assert row["model"] == "Qwen3_8-27B"


def test_every_row_answered_the_same_questions() -> None:
    answers = load("results.json")["answers"]
    counts = {json.dumps(row["n"], sort_keys=True) for row in answers.values()}
    assert len(counts) == 1


def test_retrieval_loss_is_the_oracle_minus_the_arm() -> None:
    results = load("results.json")
    accuracy = results["answers"]
    for arm, row in results["retrieval_loss"].items():
        for group, loss in row["loss"].items():
            expected = accuracy[row["oracle"]]["accuracy"][group] - accuracy[arm]["accuracy"][group]
            assert loss == pytest.approx(expected, abs=1e-4)


def test_paired_tests_are_well_formed() -> None:
    for row in load("results.json")["paired"]:
        assert row["left_only"] + row["right_only"] <= row["n"]
        assert 0.0 <= row["p"] <= 1.0


def test_costs_record_the_page_index_and_the_cpu_probe() -> None:
    costs = load("costs.json")
    index = costs["page_index"]
    assert index["pages"] == 1000
    assert index["megabytes_float16"] > 0
    assert index["cpu_seconds_per_page"] > 0
    assert index["cpu_page_vector_cosine"] > 0.99
    for name in ("e1-pictures", "e2-pictures", "e2-tables"):
        assert costs["enrichment"][name]["elements"] > 0
