"""Tests for rendering memo tables back out of a written sweep."""

from __future__ import annotations

import json

import pytest
from aieng.advanced_rag.evaluation.metrics import NullReport
from aieng.advanced_rag.evaluation.sweep import AnswerableReport, SweepRow
from summarise_sweep import load_rows, main, select


def row(sufficient: float | None, insufficient: float | None, attempts: int) -> SweepRow:
    """Build a row at one grid point."""
    label = "control" if sufficient is None else f"{sufficient:.2f} / {insufficient:.2f}"
    return SweepRow(
        label=label,
        grader="test",
        sufficient=sufficient,
        insufficient=insufficient,
        max_attempts=attempts,
        null=NullReport(
            null_queries=10,
            answerable_queries=10,
            abstain_precision=0.5,
            abstain_recall=0.5,
            false_answer_rate=0.5,
        ),
        answerable=AnswerableReport(answerable_queries=10, abstain_rate=0.2, grounded_answer_rate=0.5),
        model_calls=10,
        mean_attempts=1.0,
    )


ROWS = [
    row(0.6, 0.3, 1),
    row(0.6, 0.3, 2),
    row(0.8, 0.3, 2),
    row(0.8, 0.5, 2),
]


class TestSelect:
    def test_it_keeps_only_the_requested_attempt_budget(self):
        assert len(select(ROWS, attempts=1)) == 1

    def test_fixing_the_rewrite_threshold_sweeps_the_answer_threshold_alone(self):
        chosen = select(ROWS, attempts=2, insufficient=0.3)
        assert [r.label for r in chosen] == ["0.60 / 0.30", "0.80 / 0.30"]

    def test_fixing_the_answer_threshold_sweeps_the_rewrite_threshold_alone(self):
        chosen = select(ROWS, attempts=2, sufficient=0.8)
        assert [r.label for r in chosen] == ["0.80 / 0.30", "0.80 / 0.50"]

    def test_both_axes_can_be_fixed_at_once(self):
        chosen = select(ROWS, attempts=2, sufficient=0.8, insufficient=0.5)
        assert [r.label for r in chosen] == ["0.80 / 0.50"]

    def test_naming_points_overrides_the_other_filter(self):
        chosen = select(ROWS, attempts=2, insufficient=0.3, points=["0.80/0.50"])
        assert [r.label for r in chosen] == ["0.80 / 0.50"]

    def test_a_point_that_is_not_on_the_grid_is_simply_absent(self):
        assert select(ROWS, attempts=2, points=["0.95/0.90"]) == []


class TestLoadRows:
    def test_it_rebuilds_rows_and_controls_from_a_written_sweep(self, tmp_path):
        payload = {
            "arms": {
                "heuristic": {
                    "null_only": False,
                    "rows": [row(0.6, 0.3, 2).to_dict()],
                    "controls": [row(None, None, 1).to_dict()],
                }
            }
        }
        path = tmp_path / "sweep.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        rows, controls = load_rows(json.loads(path.read_text()), "heuristic")
        assert rows[0].label == "0.60 / 0.30"
        assert controls[0].sufficient is None

    def test_an_all_null_control_arm_has_no_sweep_rows_to_render(self):
        payload = {"arms": {"llm": {"null_only": True, "rows": [], "controls": []}}}
        with pytest.raises(ValueError, match="all-null control"):
            load_rows(payload, "llm")

    def test_an_unknown_grader_arm_raises(self):
        with pytest.raises(KeyError):
            load_rows({"arms": {}}, "heuristic")


class TestMain:
    def test_it_prints_a_table_with_a_header_row(self, tmp_path, capsys):
        payload = {
            "arms": {
                "heuristic": {
                    "null_only": False,
                    "rows": [row(0.6, 0.3, 2).to_dict()],
                    "controls": [],
                }
            }
        }
        path = tmp_path / "sweep.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert main([str(path), "--grader", "heuristic", "--attempts", "2"]) == 0
        printed = capsys.readouterr().out
        assert "Abstain recall" in printed
        assert "0.60 / 0.30" in printed
