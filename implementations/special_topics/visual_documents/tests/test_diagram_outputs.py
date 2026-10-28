"""The committed diagram result is consistent, names its models, and holds no text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULT = Path(__file__).resolve().parents[1] / "outputs" / "diagrams.json"
# "text" is also the name of the text-only query group, so it is checked separately.
TEXT_KEYS = {"query", "answer", "description", "fact", "relation", "markdown"}
REPRESENTATIONS = (
    "text-layer",
    "ocr",
    "ocr-boxes",
    "vlm-generic",
    "vlm-relations",
    "vlm-relations-haiku",
    "vlm-text-only",
)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value))
    return set()


def _result() -> Any:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_no_slide_or_model_text() -> None:
    result = _result()
    assert not _keys(result) & TEXT_KEYS
    assert isinstance(result["groups"]["text"], dict)
    assert isinstance(result["retrieval"]["groups"]["text"], int)


def test_the_models_are_named() -> None:
    produced = _result()["produced_by"]
    assert produced["model"] == "gemini-3.5-flash-lite"
    assert produced["reader"] == "gemini-3.5-flash-lite"
    assert produced["judge"] == "gemini-3.8-flash"
    assert produced["relation_lister"] == "claude-sonnet-5-5"
    assert set(produced["describers"]) == set(REPRESENTATIONS[3:])


def test_relation_recall_covers_every_representation_and_is_a_share() -> None:
    for block in _result()["relations"].values():
        assert set(block["recall"]) == set(REPRESENTATIONS)
        for row in block["recall"].values():
            assert 0.0 <= row["recall"] <= 1.0
            assert 0.0 <= row["all_stated"] <= 1.0


def test_groups_partition_the_picked_queries() -> None:
    result = _result()
    groups = result["groups"]
    assert groups["diagram"]["pairs"] == result["selection"]["diagram_relation"]
    assert groups["diagram"]["pairs"] + groups["other-picture"]["pairs"] == result["selection"]["picture_pairs"]
    assert not set(groups["diagram"]["query_ids"]) & set(groups["other-picture"]["query_ids"])
    for block in groups.values():
        assert all(0 <= n <= block["pairs"] for n in block["correct"].values())


def test_retrieval_recall_is_a_share_for_every_ranking() -> None:
    recall = _result()["retrieval"]["recall"]
    assert len(recall) == 4
    assert all(v is None or 0.0 <= v <= 1.0 for row in recall.values() for v in row.values())
