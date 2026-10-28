"""Metric correctness on hand-built rankings.

Every case here is one where the right answer can be computed by hand, so a
regression in the metric code shows up as a failing assertion rather than as a
plausible-looking number in a report.
"""

from __future__ import annotations

import math

import pytest

from aieng.advanced_rag.evaluation.metrics import (
    budget_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_failure,
    retrieved_chars,
    score_null_detection,
    score_retrieval,
)


class TestRetrievalFailure:
    def test_relevant_inside_k_is_not_a_failure(self):
        assert retrieval_failure(["a", "b", "c"], {"c"}, k=3) is False

    def test_relevant_beyond_k_is_a_failure(self):
        # "c" is retrieved, but at rank 3, outside the top-2 the reader sees.
        assert retrieval_failure(["a", "b", "c"], {"c"}, k=2) is True

    def test_nothing_relevant_is_a_failure(self):
        assert retrieval_failure(["a", "b"], {"z"}, k=10) is True

    def test_empty_ranking_is_a_failure(self):
        assert retrieval_failure([], {"a"}, k=10) is True


class TestRecall:
    def test_partial_recall(self):
        assert recall_at_k(["a", "b"], {"a", "c"}, k=2) == 0.5

    def test_full_recall(self):
        assert recall_at_k(["a", "c"], {"a", "c"}, k=2) == 1.0

    def test_k_truncates_before_counting(self):
        assert recall_at_k(["a", "c"], {"a", "c"}, k=1) == 0.5

    def test_no_relevant_chunks_scores_zero(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize("rank,expected", [(1, 1.0), (2, 0.5), (4, 0.25)])
    def test_first_relevant_position(self, rank, expected):
        ranking = [f"x{i}" for i in range(10)]
        ranking[rank - 1] = "gold"
        assert reciprocal_rank(ranking, {"gold"}) == expected

    def test_missing_scores_zero(self):
        assert reciprocal_rank(["a", "b"], {"gold"}) == 0.0


class TestNdcg:
    def test_ideal_ranking_scores_one(self):
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_known_value_for_a_single_relevant_at_rank_two(self):
        # gain = 1/log2(3), ideal = 1/log2(2) = 1
        assert ndcg_at_k(["x", "gold"], {"gold"}, k=2) == pytest.approx(
            1 / math.log2(3)
        )

    def test_lower_rank_scores_lower(self):
        high = ndcg_at_k(["gold", "x", "y"], {"gold"}, k=3)
        low = ndcg_at_k(["x", "y", "gold"], {"gold"}, k=3)
        assert high > low

    def test_no_relevant_chunks_scores_zero(self):
        assert ndcg_at_k(["a"], set(), k=1) == 0.0


class TestScoreRetrieval:
    def test_perfect_run(self):
        runs = {"q1": ["a", "b"], "q2": ["c"]}
        qrels = {"q1": {"a"}, "q2": {"c"}}
        report = score_retrieval(runs, qrels, k=2)
        assert report.retrieval_failure_rate == 0.0
        assert report.hit_rate == 1.0
        assert report.mrr == 1.0
        assert report.queries == 2

    def test_total_failure_run(self):
        runs = {"q1": ["x"], "q2": ["y"]}
        qrels = {"q1": {"a"}, "q2": {"c"}}
        report = score_retrieval(runs, qrels, k=2)
        assert report.retrieval_failure_rate == 1.0
        assert report.hit_rate == 0.0
        assert report.mrr == 0.0

    def test_failure_rate_and_hit_rate_are_complements(self):
        runs = {"q1": ["a"], "q2": ["x"], "q3": ["c"], "q4": ["z"]}
        qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}, "q4": {"d"}}
        report = score_retrieval(runs, qrels, k=1)
        assert report.retrieval_failure_rate == 0.5
        assert report.retrieval_failure_rate + report.hit_rate == 1.0

    def test_unjudged_queries_are_excluded_not_counted_as_failures(self):
        # A null query has an empty judgment set. Scoring it would report a
        # failure for a question that has no answer to retrieve.
        runs = {"q1": ["a"], "null": ["x"]}
        qrels = {"q1": {"a"}, "null": set()}
        report = score_retrieval(runs, qrels, k=1)
        assert report.queries == 1
        assert report.retrieval_failure_rate == 0.0

    def test_all_unjudged_raises(self):
        with pytest.raises(ValueError, match="No scored queries"):
            score_retrieval({"null": ["x"]}, {"null": set()}, k=1)

    def test_the_serialised_form_names_the_depth(self):
        report = score_retrieval({"q": ["a"]}, {"q": {"a"}}, k=10)
        payload = report.to_dict()
        assert "recall_at_10" in payload and "ndcg_at_10" in payload
        assert report.table_row("dense").startswith("| dense |")


class TestNullDetection:
    def test_perfect_abstention(self):
        abstained = {"n1": True, "n2": True, "a1": False}
        report = score_null_detection(abstained, truly_null={"n1", "n2"})
        assert report.abstain_precision == 1.0
        assert report.abstain_recall == 1.0
        assert report.false_answer_rate == 0.0
        assert report.null_queries == 2
        assert report.answerable_queries == 1

    def test_always_answering_never_abstains(self):
        abstained = {"n1": False, "n2": False, "a1": False}
        report = score_null_detection(abstained, truly_null={"n1", "n2"})
        assert report.abstain_recall == 0.0
        assert report.false_answer_rate == 1.0

    def test_over_abstention_costs_precision(self):
        abstained = {"n1": True, "a1": True}
        report = score_null_detection(abstained, truly_null={"n1"})
        assert report.abstain_recall == 1.0
        assert report.abstain_precision == 0.5

    def test_nothing_to_score_raises(self):
        with pytest.raises(ValueError, match="No queries"):
            score_null_detection({}, truly_null=set())


class TestReaderBudget:
    def test_retrieved_chars_measures_what_the_reader_receives(self):
        runs = {"q1": ["a", "b", "c"], "q2": ["b", "c", "a"]}
        sizes = {"a": 1000, "b": 1000, "c": 1000}
        assert retrieved_chars(runs, sizes, 2) == 2000

    def test_large_chunks_reach_the_budget_at_a_smaller_k(self):
        # The point of the equal-budget comparison: ten 4,000-character chunks
        # put four times as much text in front of the reader as ten 1,000s.
        runs = {"q1": ["a", "b", "c"], "q2": ["b", "c", "a"]}
        assert budget_k(runs, {"a": 1000, "b": 1000, "c": 1000}, 2000, depth=10) == 2
        assert budget_k(runs, {"a": 4000, "b": 4000, "c": 4000}, 2000, depth=10) == 1

    def test_an_unreachable_budget_falls_back_to_the_full_depth(self):
        runs = {"q1": ["a", "b", "c"]}
        assert budget_k(runs, {"a": 1, "b": 1, "c": 1}, 99999, depth=3) == 3


class TestRepeatedIdentifiers:
    def test_a_repeated_relevant_identifier_counts_once(self):
        assert ndcg_at_k(["a", "a"], {"a"}, 2) == pytest.approx(1.0)

    def test_ndcg_never_exceeds_one(self):
        assert ndcg_at_k(["a", "b", "a", "b"], {"a", "b"}, 4) <= 1.0

    def test_untied_rankings_are_unchanged(self):
        assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(
            (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        )


class TestAgainstIrMeasures:
    """Pin agreement with trec_eval through ``ir_measures`` when it is installed."""

    def test_standard_measures_match_trec_eval(self):
        ir_measures = pytest.importorskip("ir_measures")
        from aieng.advanced_rag.evaluation.metrics import recall_at_k

        ranking = ["d3", "d9", "d1", "d7", "d2", "d5"]
        relevant = {"d1", "d2", "d4"}
        qrels = [ir_measures.Qrel("q", d, 1) for d in relevant]
        run = [
            ir_measures.ScoredDoc("q", d, float(len(ranking) - i))
            for i, d in enumerate(ranking)
        ]
        got = ir_measures.calc_aggregate(
            [ir_measures.nDCG @ 5, ir_measures.R @ 5, ir_measures.RR], qrels, run
        )
        assert ndcg_at_k(ranking, relevant, 5) == pytest.approx(
            got[ir_measures.nDCG @ 5], abs=1e-4
        )
        assert recall_at_k(ranking, relevant, 5) == pytest.approx(
            got[ir_measures.R @ 5], abs=1e-4
        )
        assert reciprocal_rank(ranking, relevant) == pytest.approx(
            got[ir_measures.RR], abs=1e-4
        )
