"""The committed results are consistent, name their models, and carry no corpus text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "graph_build_stability.json"
# Keys under which corpus text, model text, or entity names would sit.
TEXT_KEYS = {"question", "answer", "text", "original_text", "raw", "evidence", "passage", "entity_relation_output"}


@pytest.fixture(scope="module")
def results() -> Any:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def walk(value: Any) -> tuple[set[str], list[Any]]:
    keys: set[str] = set()
    hubs: list[Any] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            keys |= set(item)
            if "top_hubs" in item:
                hubs.append(item["top_hubs"])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return keys, hubs


def test_results_carry_no_corpus_text_and_no_entity_names(results: Any) -> None:
    keys, hubs = walk(results)
    assert not keys & TEXT_KEYS
    assert all(h is None for h in hubs)


def test_results_name_the_reader_and_every_build_model(results: Any) -> None:
    produced = results["produced_by"]
    assert produced["reader"] == "Qwen3_6-35B-A3B"
    assert produced["embeddings"].startswith("BAAI/bge-small-en-v1.5")
    assert "no response cache" in produced["independence"]
    built = {b["model"] for b in results["musique"]["builds"].values() if b.get("model")}
    assert built == set(produced["build_models"])
    assert set(produced["paid_build_models"]) <= built


def test_every_extraction_build_reached_the_model_on_every_call(results: Any) -> None:
    for name, build in results["musique"]["builds"].items():
        cost = build.get("cost")
        if not cost or "calls" not in cost:
            continue
        assert cost.get("cache_hits", 0) == 0, name
        assert cost["calls"] >= build["structure"]["documents"], name


def test_musique_rows_cover_the_same_questions_for_every_graph(results: Any) -> None:
    rows = results["musique"]["rows"]
    ids = {r["question_id"] for r in rows["dense"]}
    assert len(ids) == results["musique"]["questions"] == 100
    for name, members in rows.items():
        assert {r["question_id"] for r in members} == ids, name
        assert all(len(r["ranked"]) <= 10 for r in members)


def test_summaries_agree_with_rows(results: Any) -> None:
    musique = results["musique"]
    for name, build in musique["builds"].items():
        if "retrieval" not in build:
            continue
        rows = musique["rows"][name]
        f1 = sum(r["f1"] for r in rows) / len(rows)
        assert build["retrieval"]["all"]["f1"] == pytest.approx(f1, abs=1e-3), name


def test_serverfault_compares_repeats_of_each_arm(results: Any) -> None:
    for name, entry in results["serverfault"].items():
        assert len(entry["repeats"]) >= 2, name
        assert entry["windows"] == 203
        assert 0.0 <= entry["as_returned"]["agreement"]["graph_nodes"] <= 1.0


def test_temperature_zero_builds_were_sent_temperature_zero(results: Any) -> None:
    group = results["musique"]["repeats"]["atlas_A3B_t0"]
    assert len(group["builds"]) == 3
    for name in group["builds"]:
        assert results["musique"]["builds"][name]["temperature_sent"] == 0.0, name
    assert 0.0 <= group["response_identity"]["identical"] <= 1.0
    assert set(results["greedy_probe"]) == {"t0", "t0_seed7", "t0_topk1", "t0.7_seed7"}
