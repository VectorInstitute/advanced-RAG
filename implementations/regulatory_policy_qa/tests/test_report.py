"""Tests for the sweep's report writer."""

from __future__ import annotations

import json

from aieng.advanced_rag.evaluation.metrics import NullReport
from aieng.advanced_rag.evaluation.sweep import AnswerableReport, SweepRow
from sweep_lab.report import markdown_table, write_json


def row(label: str, recall: float, cost: float, thresholds: bool = True) -> SweepRow:
    """Build a sweep row with the two rates the ordering depends on."""
    return SweepRow(
        label=label,
        grader="test",
        sufficient=0.6 if thresholds else None,
        insufficient=0.3 if thresholds else None,
        max_attempts=1,
        null=NullReport(
            null_queries=10,
            answerable_queries=10,
            abstain_precision=0.5,
            abstain_recall=recall,
            false_answer_rate=1.0 - recall,
        ),
        answerable=AnswerableReport(answerable_queries=10, abstain_rate=cost, grounded_answer_rate=0.5),
        model_calls=100,
        mean_attempts=1.0,
    )


class TestMarkdownTable:
    def test_it_renders_a_header_and_one_line_per_row(self):
        table = markdown_table([row("a", 0.8, 0.1), row("b", 0.6, 0.1)])
        assert len(table.splitlines()) == 4

    def test_it_can_order_candidates_by_balanced_accuracy(self):
        table = markdown_table([row("worse", 0.2, 0.4), row("better", 0.9, 0.1)], by_score=True)
        lines = table.splitlines()
        assert "better" in lines[2]
        assert "worse" in lines[3]

    def test_controls_stay_at_the_bottom_however_they_score(self):
        table = markdown_table(
            [
                row("always abstain", 1.0, 1.0, thresholds=False),
                row("candidate", 0.5, 0.2),
            ],
            by_score=True,
        )
        assert "candidate" in table.splitlines()[2]
        assert "always abstain" in table.splitlines()[3]

    def test_the_row_order_is_kept_when_scoring_is_off(self):
        table = markdown_table([row("first", 0.1, 0.9), row("second", 0.9, 0.1)])
        assert "first" in table.splitlines()[2]


class TestWriteJson:
    def test_it_creates_missing_parent_directories(self, tmp_path):
        path = write_json(tmp_path / "deep" / "nested" / "out.json", {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}

    def test_it_ends_the_file_with_a_newline(self, tmp_path):
        path = write_json(tmp_path / "out.json", [1, 2])
        assert path.read_text().endswith("\n")
