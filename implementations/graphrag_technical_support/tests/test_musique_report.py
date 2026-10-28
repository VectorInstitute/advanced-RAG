"""Per-hop summaries and paired comparisons over per-question rows."""

from __future__ import annotations

import pytest
from musique_report import Row, compare, groups, summarise


GOLD = {
    "a": ("p1", "p2"),
    "b": ("p3", "p4"),
    "c": ("p5", "p6", "p7"),
    "d": ("p1", "p9"),
}


def row(qid, hops, ranked, em=0.0, f1=0.0):
    return Row(qid, hops, tuple(ranked), em, f1)


ROWS = [
    row("a", 2, ["p1", "p2"], em=1.0, f1=1.0),
    row("b", 2, ["p3", "x"], em=0.0, f1=0.5),
    row("c", 3, ["p5", "p6", "p7"], em=1.0, f1=1.0),
    row("d", 2, ["x", "y"], em=0.0, f1=0.0),
]


def test_rows_group_by_hop_and_add_all():
    found = groups(ROWS)
    assert list(found) == ["2hop", "3hop", "all"]
    assert [r.question_id for r in found["2hop"]] == ["a", "b", "d"]
    assert len(found["all"]) == 4


def test_the_summary_asks_for_every_supporting_passage_by_hop():
    summary = summarise(ROWS, GOLD)
    two = summary["2hop"]
    assert two["questions"] == 3
    assert two["all_supporting_recall_at_10"] == pytest.approx(1 / 3, abs=1e-4)
    assert two["mean_supporting_recall_at_10"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)
    assert two["exact_match"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["3hop"]["all_supporting_recall_at_5"] == 1.0
    assert summary["all"]["questions"] == 4


def test_a_shallower_depth_cannot_beat_a_deeper_one():
    ranked = ["x", "x2", "x3", "x4", "x5", "p1", "p2"]
    summary = summarise([row("a", 2, ranked)], GOLD)["all"]
    assert summary["all_supporting_recall_at_5"] == 0.0
    assert summary["all_supporting_recall_at_10"] == 1.0


def test_a_comparison_names_the_questions_only_one_arm_gets_right():
    baseline = [
        row("a", 2, ["x"], em=0.0, f1=0.0),
        row("b", 2, ["x"], em=0.0, f1=0.0),
        row("c", 3, ["x"], em=0.0, f1=0.0),
        row("d", 2, ["x"], em=0.0, f1=0.0),
    ]
    result = compare(ROWS, baseline, GOLD, samples=200)
    overall = result["all"]
    assert overall["all_supporting_at_10"]["only_candidate"] == 2
    assert overall["all_supporting_at_10"]["only_baseline"] == 0
    assert overall["exact_match"]["only_candidate"] == 2
    assert overall["f1_difference"] > 0


def test_a_comparison_needs_the_same_questions():
    with pytest.raises(ValueError, match="same questions"):
        compare(ROWS, ROWS[:2], GOLD)


def test_a_comparison_of_an_arm_with_itself_finds_no_difference():
    overall = compare(ROWS, ROWS, GOLD, samples=200)["all"]
    assert overall["f1_difference"] == 0.0
    assert overall["all_supporting_at_10"]["p_value"] == 1.0
    assert not overall["f1_interval_excludes_zero"]


def test_a_row_serialises_without_text():
    record = ROWS[0].to_dict()
    assert record == {
        "question_id": "a",
        "hops": 2,
        "ranked": ["p1", "p2"],
        "em": 1.0,
        "f1": 1.0,
    }
