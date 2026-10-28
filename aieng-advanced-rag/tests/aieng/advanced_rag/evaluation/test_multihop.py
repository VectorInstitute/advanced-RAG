"""Multi-hop scoring: all supporting passages, SQuAD answer scoring, paired tests."""

from __future__ import annotations

import pytest

from aieng.advanced_rag.evaluation.multihop import (
    all_gold_hit,
    all_gold_recall_at_k,
    exact_match,
    mcnemar_exact,
    normalise_answer,
    paired_bootstrap,
    supporting_recall_at_k,
    token_f1,
)


def test_a_question_needs_every_supporting_passage_in_the_top_k():
    assert all_gold_hit(["a", "b", "x"], {"a", "b"}, 3)
    assert not all_gold_hit(["a", "x", "b"], {"a", "b"}, 2)
    assert not all_gold_hit(["a", "b"], set(), 2)


def test_recall_is_the_share_of_questions_with_all_their_evidence():
    runs = {"q1": ["a", "b"], "q2": ["a", "x"], "q3": ["z"]}
    qrels = {"q1": {"a", "b"}, "q2": {"a", "b"}, "q3": {"z"}}
    assert all_gold_recall_at_k(runs, qrels, 2) == pytest.approx(2 / 3)
    assert all_gold_recall_at_k(runs, qrels, 1) == pytest.approx(1 / 3)


def test_recall_with_nothing_to_score_is_an_error():
    with pytest.raises(ValueError):
        all_gold_recall_at_k({"q": ["a"]}, {"q": set()}, 5)


def test_partial_recall_counts_the_supporting_passages_found():
    assert supporting_recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5
    assert supporting_recall_at_k([], set(), 2) == 0.0


def test_answers_are_normalised_like_squad():
    assert normalise_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalise_answer("An apple, a day") == "apple day"


def test_exact_match_takes_the_best_over_aliases():
    assert exact_match("the USA", ["United States", "USA"]) == 1.0
    assert exact_match("France", ["Spain"]) == 0.0


def test_f1_gives_partial_credit_and_takes_the_best_alias():
    assert token_f1("big red dog", ["red dog"]) == pytest.approx(0.8)
    assert token_f1("red dog", ["blue cat", "red dog"]) == 1.0
    assert token_f1("", ["x"]) == 0.0
    assert token_f1("", [""]) == 1.0


def test_the_bootstrap_interval_is_reproducible_and_excludes_zero_for_a_clear_gap():
    a = [1.0] * 40 + [0.0] * 10
    b = [0.0] * 40 + [1.0] * 10
    first = paired_bootstrap(a, b, samples=500, seed=3)
    assert first == paired_bootstrap(a, b, samples=500, seed=3)
    assert first.mean_difference == pytest.approx(0.6)
    assert first.excludes_zero and first.low > 0


def test_the_bootstrap_interval_spans_zero_when_the_arms_match():
    a = [1.0, 0.0] * 20
    result = paired_bootstrap(a, list(reversed(a)), samples=500, seed=1)
    assert not result.excludes_zero


def test_the_bootstrap_refuses_mismatched_arms():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_mcnemar_counts_the_disagreements_and_gives_an_exact_p():
    a = [True] * 9 + [False] * 1 + [True] * 5
    b = [False] * 9 + [True] * 1 + [True] * 5
    result = mcnemar_exact(a, b)
    assert (result.only_a, result.only_b) == (9, 1)
    assert result.p_value == pytest.approx(0.021484375)


def test_mcnemar_with_no_disagreement_is_not_significant():
    assert mcnemar_exact([True, False], [True, False]).p_value == 1.0


def test_an_even_split_of_disagreements_has_p_one():
    result = mcnemar_exact([True, False], [False, True])
    assert result.p_value == 1.0


class TestMcNemarAgainstLibraries:
    """The hand-written test must match the maintained implementations."""

    CASES = [(0, 0), (1, 0), (5, 5), (13, 7), (31, 29), (2, 40), (0, 12)]

    def test_matches_scipy_binomtest(self):
        stats = pytest.importorskip("scipy.stats")
        from aieng.advanced_rag.evaluation.multihop import mcnemar_p

        for only_a, only_b in self.CASES:
            n = only_a + only_b
            expected = (
                1.0 if n == 0 else stats.binomtest(min(only_a, only_b), n, 0.5).pvalue
            )
            assert mcnemar_p(only_a, only_b) == pytest.approx(expected, abs=1e-12)

    def test_matches_statsmodels_exact(self):
        pytest.importorskip("scipy")
        contingency = pytest.importorskip("statsmodels.stats.contingency_tables")
        from aieng.advanced_rag.evaluation.multihop import mcnemar_p

        for only_a, only_b in self.CASES:
            if only_a + only_b == 0:
                continue
            table = [[0, only_a], [only_b, 0]]
            expected = contingency.mcnemar(table, exact=True).pvalue
            assert mcnemar_p(only_a, only_b) == pytest.approx(expected, abs=1e-12)

    def test_the_committed_values_hold(self):
        from aieng.advanced_rag.evaluation.multihop import mcnemar_p

        assert round(mcnemar_p(13, 7), 4) == 0.2632
        assert round(mcnemar_p(31, 29), 4) == 0.8974

    def test_mcnemar_exact_uses_the_shared_helper(self):
        from aieng.advanced_rag.evaluation.multihop import mcnemar_exact, mcnemar_p

        a = [True] * 13 + [False] * 7 + [True] * 5
        b = [False] * 13 + [True] * 7 + [True] * 5
        assert mcnemar_exact(a, b).p_value == mcnemar_p(13, 7)
