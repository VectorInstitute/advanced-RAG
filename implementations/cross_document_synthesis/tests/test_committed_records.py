"""Committed per-question records carry scores, never the model's answer text."""

import json
from pathlib import Path

import pytest


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
REPORTS = sorted(
    path
    for folder in ("evaluation", "examples/smoke")
    for path in (OUTPUTS / folder).rglob("*.json")
    if "records" in json.loads(path.read_text())
)


def test_the_scored_reports_are_found():
    assert len(REPORTS) >= 16


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: str(p.relative_to(OUTPUTS)))
def test_records_hold_scores_in_place_of_answers(path):
    records = json.loads(path.read_text())["records"]
    assert records
    assert all("answer" not in row and {"exact_match", "token_f1"} <= set(row) for row in records)


#: Keys that carry question, answer, plan, or trace text. Those ship in RI #2's
#: reference release; see ``synthesis_lab/reference.py``.
TEXT_KEYS = {"question", "answer", "raw_answer", "raw_plan", "retrieval_queries", "query", "next_query", "facts"}
ALL_OUTPUTS = sorted(OUTPUTS.rglob("*.json"))


def _text_keys(value, found):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEXT_KEYS and item not in ("", None, [], {}):
                found.add(key)
            _text_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _text_keys(item, found)
    return found


@pytest.mark.parametrize("path", ALL_OUTPUTS, ids=lambda p: str(p.relative_to(OUTPUTS)))
def test_outputs_hold_no_question_or_answer_text(path):
    assert _text_keys(json.loads(path.read_text()), set()) == set()
