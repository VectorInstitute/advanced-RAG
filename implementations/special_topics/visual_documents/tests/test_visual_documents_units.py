"""Unit tests for the visual-documents code that need no model, network, or corpus."""

from __future__ import annotations

import math

import numpy as np
import pytest
from vdocs import arms, questions, scoring
from vdocs.enrich import render_table_summary
from vdocs.pages import Universe, page_id, split_page_id
from vdocs.visual import PageIndex


class TestPageIdentifiers:
    def test_round_trip(self) -> None:
        assert page_id("AES_2022_10K", 7) == "AES_2022_10K#p007"
        assert split_page_id("AES_2022_10K#p007") == ("AES_2022_10K", 7)

    def test_rejects_other_shapes(self) -> None:
        with pytest.raises(ValueError, match="not a page identifier"):
            split_page_id("AES_2022_10K")

    def test_universe_groups_claim_pages_in_order(self) -> None:
        universe = Universe(figure=["a#p001"], table=["a#p002"], control=["a#p001", "a#p003"], distractor=["a#p004"])
        assert universe.pages == ["a#p001", "a#p002", "a#p003", "a#p004"]
        assert universe.group_of("a#p001") == "figure"
        assert Universe.from_dict(universe.to_dict()) == universe


class TestNumbers:
    def test_reads_currency_separators_and_parentheses(self) -> None:
        assert scoring.numbers_in("Revenue was $2,868 million, down (1.5)%") == [2868.0, -1.5]

    @pytest.mark.parametrize(
        ("answer", "value", "expected"),
        [
            ("It was $2,868 million.", 2868, True),
            ("About 2,880.", 2868, True),
            ("About 2,900.", 2868, False),
            ("Margin fell by 5 points.", -5, True),
            ("Zero.", 0, False),
            ("0 units", 0, True),
        ],
    )
    def test_numeric_match_is_relative(self, answer: str, value: float, expected: bool) -> None:
        assert scoring.numeric_match(answer, value) is expected

    def test_abstention(self) -> None:
        assert scoring.abstained("NOT STATED")
        assert scoring.abstained("The evidence does not state the 2019 value.")
        assert scoring.abstained("")
        assert not scoring.abstained("It was 40%.")


class TestScoreAnswer:
    question = {"query_id": "q", "group": "figure", "question": "?", "answer": "40%", "value": 40.0}

    def test_numeric_match_needs_no_judge(self) -> None:
        verdict = scoring.score_answer(self.question, "Mexico had 40% of capacity.", None)
        assert verdict == scoring.Verdict(True, "numeric")

    def test_judge_breaks_the_tie(self) -> None:
        verdict = scoring.score_answer(self.question, "Two fifths.", lambda _p: '{"correct": true}')
        assert verdict == scoring.Verdict(True, "judge")

    def test_abstaining_on_an_answerable_question_is_wrong(self) -> None:
        verdict = scoring.score_answer(self.question, "NOT STATED", lambda _p: '{"correct": true}')
        assert verdict == scoring.Verdict(False, "abstained")

    def test_a_null_is_right_only_when_abstained(self) -> None:
        null = {**self.question, "group": "null", "value": None}
        assert scoring.score_answer(null, "NOT STATED", None).correct
        assert not scoring.score_answer(null, "It was 40%.", None).correct


class TestPageMetrics:
    def test_metrics_skip_questions_without_gold(self) -> None:
        rankings = {"a": ["p1", "p2"], "b": ["p3", "p4"], "n": ["p1"]}
        gold = {"a": ["p2"], "b": ["p9"], "n": []}
        found = scoring.page_metrics(rankings, gold, k=5)
        assert found["n"] == 2
        assert found["recall"] == 0.5
        assert found["failure"] == 0.5

    def test_empty_is_nan(self) -> None:
        assert math.isnan(scoring.page_metrics({}, {}, 5)["recall"])


class TestCostScenario:
    def test_unenriched_arms_pay_nothing_at_indexing(self) -> None:
        scenario = scoring.Scenario(pages=1000, pictures_per_page=0.1, tables_per_page=0.5, questions=100, reindexes=2)
        table = scoring.scenario_cost(
            scenario,
            enrichment_usd_per_element=0.001,
            reader_usd_per_question={"T": 0.0001, "E2": 0.0001, "VV": 0.001},
            machine_seconds_per_page={"VV": 0.75},
        )
        assert table["T"]["index_usd"] == 0.0
        assert table["VV"]["index_usd"] == 0.0
        assert table["E2"]["index_usd"] == pytest.approx(1.2)
        assert table["VV"]["index_hours"] == pytest.approx(0.42, abs=0.01)

    def test_break_even(self) -> None:
        assert scoring.break_even_questions(10.0, 0.001, 0.0, 0.011) == pytest.approx(1000.0)
        assert math.isinf(scoring.break_even_questions(10.0, 0.01, 0.0, 0.001))


class TestRankings:
    def test_collapse_keeps_first_occurrence(self) -> None:
        pages = {"c1": "p1", "c2": "p1", "c3": "p2", "c4": "p3"}
        assert arms.collapse_to_pages(["c2", "c1", "c3", "c4"], pages, k=2) == ["p1", "p2"]

    def test_reciprocal_rank_fusion_rewards_agreement(self) -> None:
        fused = arms.reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
        assert fused[0] == "b"

    def test_every_arm_has_an_oracle_and_a_description(self) -> None:
        assert len(arms.ARMS) == 8
        assert len(arms.ORACLES) == 4
        assert set(arms.DESCRIPTIONS) == set(arms.ROWS)


class TestPageIndex:
    def test_maxsim_matches_brute_force_across_blocks(self) -> None:
        rng = np.random.default_rng(0)
        pages = [rng.normal(size=(n, 8)).astype(np.float32) for n in (5, 9, 3, 7)]
        pages = [p / np.linalg.norm(p, axis=1, keepdims=True) for p in pages]
        index = PageIndex.build(["a", "b", "c", "d"], pages, {})
        query = rng.normal(size=(4, 8)).astype(np.float32)
        expected = sorted(
            (
                (name, float((m.astype(np.float16).astype(np.float32) @ query.T).max(axis=0).sum()))
                for name, m in zip("abcd", pages)
            ),
            key=lambda item: -item[1],
        )
        found = index.search_many([query], k=4, block=4)[0]
        assert [name for name, _ in found] == [name for name, _ in expected]
        assert [score for _, score in found] == pytest.approx([score for _, score in expected], rel=1e-5)

    def test_save_and_load(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        index = PageIndex.build(["a"], [np.ones((2, 4), dtype=np.float32)], {"model": "m"})
        index.save(tmp_path)
        loaded = PageIndex.load(tmp_path)
        assert loaded.page_ids == ["a"]
        assert loaded.meta == {"model": "m"}
        assert loaded.vectors.dtype == np.float16


class TestQuestionChecks:
    def test_region_check_finds_the_label_in_a_description(self) -> None:
        candidate = {"chart_label": "Operating Margin"}
        assert questions.region_check(candidate, "", ["Chart 2: Operating margin (in millions)"])
        assert not questions.region_check(candidate, "Revenue", [])

    def test_funnel_counts_by_kind_and_stage(self) -> None:
        funnel = questions.Funnel()
        funnel.check("figure", "a", True)
        funnel.check("figure", "a", False)
        funnel.check("table", "b", True)
        assert funnel.to_rows() == [
            {"kind": "figure", "stage": "a", "entered": 2, "passed": 1},
            {"kind": "table", "stage": "b", "entered": 1, "passed": 1},
        ]

    def test_stratified_subset_keeps_every_subgroup(self) -> None:
        found = [{"query_id": f"q{i}", "group": "figure", "subgroup": "a" if i < 90 else "b"} for i in range(100)]
        subset = questions.stratified_subset(found, 10)
        assert any(int(q[1:]) >= 90 for q in subset)
        assert 10 <= len(subset) <= 12


def test_table_summary_rendering() -> None:
    text = render_table_summary({"summary": "Revenue by segment.", "key_entities": ["A", "B"], "content_type": "x"})
    assert text == "Revenue by segment. Key entities: A; B. Content type: x."
