"""The routers pick routes as documented, and the scoring counts errors correctly."""

from __future__ import annotations

import numpy as np
import pytest
import routing
from routing import (
    EmbeddingRouter,
    FixedRouter,
    LLMRouter,
    OracleRouter,
    Reply,
    Route,
    hindsight_route,
    parse_choice,
    routing_report,
    summarise_arm,
)


class KeywordEmbedder:
    """Embed text as counts of three marker words, so similarity is predictable."""

    WORDS = ("revenue", "auditor", "and")

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(w)) + 1e-3 for w in self.WORDS]

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(t) for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)


ROUTES = (
    Route("prose", "text", ("who is the auditor", "name the auditor")),
    Route("table", "figures", ("what was revenue", "revenue in 2020")),
    Route("both", "both", ("revenue and auditor and", "revenue and the auditor and")),
)


def test_embedding_router_picks_the_route_with_the_closest_examples() -> None:
    router = EmbeddingRouter(ROUTES, KeywordEmbedder(), top_k=1)
    assert router.route("What was total revenue?").route == "table"
    assert router.route("Which firm is the auditor?").route == "prose"
    assert router.route("revenue and auditor and more").route == "both"


def test_embedding_router_falls_back_to_both_under_the_threshold() -> None:
    router = EmbeddingRouter(ROUTES, KeywordEmbedder(), threshold=1.01)
    decision = router.route("What was total revenue?")
    assert decision.route == routing.FALLBACK_ROUTE
    assert "threshold" in decision.note
    assert decision.model_calls == 0


def test_embedding_router_rejects_a_route_without_examples() -> None:
    with pytest.raises(ValueError, match="utterance"):
        EmbeddingRouter((Route("prose", "text"),), KeywordEmbedder())


def test_routes_must_have_distinct_names() -> None:
    with pytest.raises(ValueError, match="share a name"):
        LLMRouter((Route("prose", "a"), Route("prose", "b")), lambda p, n: Reply(""))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"choice": 2, "reason": "a figure"}', (2, "a figure")),
        ('<think>hmm</think> Sure: {"choice": "3"}', (3, "")),
        ('{"choice": 7}', (None, "choice 7 is outside 1 to 3")),
        ("table", (None, "no JSON in the reply")),
        ('{"choice": "two"}', (None, "no numeric choice")),
    ],
)
def test_parse_choice(text: str, expected: tuple[int | None, str]) -> None:
    assert parse_choice(text, 3) == expected


def test_llm_router_counts_its_call_and_falls_back_on_a_bad_reply() -> None:
    prompts: list[str] = []

    def complete(prompt: str, max_tokens: int) -> Reply:
        prompts.append(prompt)
        return Reply('{"choice": 2, "reason": "asks for a figure"}', 120, 9)

    decision = LLMRouter(routing.FILING_ROUTES, complete).route("What was revenue in 2019?")
    assert decision.route == "table"
    assert (decision.model_calls, decision.prompt_tokens, decision.completion_tokens) == (1, 120, 9)
    assert "(1) The text of the filings" in prompts[0]
    assert "What was revenue in 2019?" in prompts[0]

    bad = LLMRouter(routing.FILING_ROUTES, lambda p, n: Reply("I think tables")).route("q")
    assert bad.route == "both"
    assert bad.note.startswith("fell back")


def test_oracle_and_fixed_routers() -> None:
    assert OracleRouter({"q": "table"}).route("q").route == "table"
    assert FixedRouter("both").route("anything").route == "both"
    with pytest.raises(KeyError):
        OracleRouter({}).route("q")


def test_filing_routes_cover_every_route_with_examples() -> None:
    assert tuple(r.name for r in routing.FILING_ROUTES) == routing.ROUTE_NAMES
    assert all(len(r.utterances) >= 5 for r in routing.FILING_ROUTES)


def test_hindsight_prefers_the_cheapest_correct_route() -> None:
    assert hindsight_route({"prose": False, "table": True, "both": True}) == "table"
    assert hindsight_route({"prose": False, "table": False, "both": False}) is None
    assert hindsight_route({"table": True, "both": True}, {"table": 5, "both": 3}) == "both"


def test_routing_report_separates_costly_and_harmless_mismatches() -> None:
    gold = {"a": "table", "b": "table", "c": "prose", "d": "both"}
    chosen = {"a": "table", "b": "prose", "c": "table", "d": "table"}
    outcome = {
        "a": {"table": True},
        "b": {"prose": False, "table": True},  # costly: gold route was right
        "c": {"table": True, "prose": False},  # rescued: chosen route was right
        "d": {"table": False, "both": False},  # neither: a mismatch only
    }
    report = routing_report(chosen, gold, outcome)
    assert (report.mismatches, report.costly, report.rescued, report.questions) == (3, 1, 1, 4)
    assert report.confusion["table"] == {"table": 1, "prose": 1}
    assert report.agreement == pytest.approx(0.25)
    assert report.to_dict()["agreement"] == 0.25


def test_summarise_arm_reports_per_type_accuracy_and_cost() -> None:
    rows = [
        {
            "qtype": "table",
            "correct": True,
            "model_calls": 3,
            "prompt_tokens": 900,
            "completion_tokens": 30,
            "model": "m",
            "route": "table",
        },
        {
            "qtype": "table",
            "correct": False,
            "model_calls": 2,
            "prompt_tokens": 700,
            "completion_tokens": 20,
            "model": "m",
            "route": "prose",
        },
        {
            "qtype": "prose",
            "correct": True,
            "model_calls": 1,
            "prompt_tokens": 400,
            "completion_tokens": 10,
            "model": "m",
            "route": "prose",
        },
    ]
    summary = summarise_arm("RL", rows)
    assert (summary.questions, summary.correct, summary.accuracy) == (3, 2, 0.6667)
    assert summary.calls == 2.0
    assert summary.by_type["table"]["accuracy"] == 0.5
    assert list(summary.by_type) == ["prose", "table"]
    assert summary.routes == {"table": 1, "prose": 2}
    with pytest.raises(ValueError, match="no records"):
        summarise_arm("RL", [])
