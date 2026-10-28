"""The reranker report and the candidate list.

No test loads a model. What is worth pinning is the arithmetic the cost
decision rests on, the ordering contract that makes a rerank reproducible, and
that the paid option stays named rather than quietly dropped.
"""

from __future__ import annotations

from aieng.advanced_rag.retrieval.rerankers import (
    CANDIDATES,
    EXCLUDED,
    CrossEncoderReranker,
    RerankReport,
)


class StubModel:
    """Stands in for a CrossEncoder so ordering can be checked offline."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.batches: list[int] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.batches.append(len(pairs))
        return [self.scores.get(text, 0.0) for _, text in pairs]


def make_reranker(scores: dict[str, float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "stub"
    reranker.batch_size = 32
    reranker.load_seconds = 0.0
    reranker._model = StubModel(scores)
    reranker.report = RerankReport(model="stub")
    return reranker


class TestRerankReport:
    def test_throughput_and_latency(self):
        report = RerankReport(model="m", queries=10, pairs=500, seconds=5.0)
        assert report.pairs_per_second == 100.0
        assert report.ms_per_query == 500.0

    def test_an_unrun_report_does_not_divide_by_zero(self):
        report = RerankReport(model="m")
        assert report.pairs_per_second == 0.0
        assert report.ms_per_query == 0.0

    def test_the_serialised_form_carries_the_decision_numbers(self):
        payload = RerankReport(model="m", queries=2, pairs=20, seconds=1.0).to_dict()
        for key in ("pairs_per_second", "ms_per_query", "load_seconds", "parameters"):
            assert key in payload


class TestReranking:
    def test_it_reorders_candidates_by_score(self):
        reranker = make_reranker({"low": 0.1, "high": 0.9, "middle": 0.5})
        ranked = reranker.rerank(
            "q", [("c::0", "low"), ("c::1", "high"), ("c::2", "middle")], k=3
        )
        assert ranked == ["c::1", "c::2", "c::0"]

    def test_k_truncates_the_result(self):
        reranker = make_reranker({"low": 0.1, "high": 0.9})
        assert reranker.rerank("q", [("c::0", "low"), ("c::1", "high")], k=1) == [
            "c::1"
        ]

    def test_ties_break_on_the_identifier(self):
        reranker = make_reranker({})
        ranked = reranker.rerank(
            "q", [("c::2", "x"), ("c::0", "x"), ("c::1", "x")], k=3
        )
        assert ranked == ["c::0", "c::1", "c::2"]

    def test_no_candidates_costs_nothing(self):
        reranker = make_reranker({})
        assert reranker.rerank("q", [], k=5) == []
        assert reranker.score("q", []) == []
        assert reranker.report.queries == 0

    def test_scoring_accumulates_the_cost_counters(self):
        reranker = make_reranker({"a": 1.0, "b": 2.0})
        reranker.rerank("q1", [("c::0", "a"), ("c::1", "b")], k=2)
        reranker.rerank("q2", [("c::0", "a")], k=1)
        assert reranker.report.queries == 2
        assert reranker.report.pairs == 3
        assert reranker.report.seconds > 0


class TestCandidates:
    def test_every_candidate_says_why_it_is_here(self):
        for candidate in CANDIDATES:
            assert candidate.note and candidate.model

    def test_the_small_baseline_is_present(self):
        # Without it, a large model's win cannot be read as worth its latency.
        assert any("MiniLM" in c.name for c in CANDIDATES)

    def test_remote_code_is_flagged_rather_than_silent(self):
        jina = next(c for c in CANDIDATES if "Jina" in c.name)
        assert jina.trust_remote_code is True
        assert "trust_remote_code" in jina.note

    def test_paid_options_are_named_and_excluded(self):
        assert "Cohere Rerank" in EXCLUDED
        assert "billed" in EXCLUDED["Cohere Rerank"]
