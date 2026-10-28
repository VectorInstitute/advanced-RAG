"""Tests for the threshold sweep: harvest, replay, controls, and both-sided scoring.

The load-bearing test is ``TestReplayMatchesTheSpine``. A replay that drifts from
``AdvancedRagSpine.answer`` would report an operating point the pipeline does not
have, so the two are asserted to agree over every threshold pair in the grid
rather than assumed to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieng.advanced_rag.evaluation.sweep import (
    TABLE_HEADER,
    AttemptTrace,
    QueryTrace,
    SweepRow,
    always_abstain,
    always_answer,
    control_rows,
    generator_only,
    harvest_traces,
    null_only_sweep,
    replay,
    score_answerable,
    score_arm,
    sweep,
    threshold_grid,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline.answering import (
    CorrectivePolicy,
    FixedPromptSelector,
    LoopDecision,
    LoopState,
    RewriterAnalysis,
)
from aieng.advanced_rag.pipeline.grading import (
    Action,
    ChunkGrade,
    EvidenceGrade,
    GradingThresholds,
    Verdict,
)
from aieng.advanced_rag.pipeline.rewrite import Rewrite, RewritePlan
from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig
from aieng.advanced_rag.provenance import Provenance


def corrective(sufficient: float, insufficient: float) -> CorrectivePolicy:
    """Return the shipped loop at one grid point, which is what ``replay`` takes."""
    return CorrectivePolicy(GradingThresholds(sufficient, insufficient))


def evidence_for(doc_id: str, texts: list[str]) -> list[Evidence]:
    """Build evidence with predictable chunk identifiers."""
    return [
        Evidence(
            text=text,
            provenance=Provenance(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::{index:03d}",
                start=index * 100,
                end=index * 100 + len(text),
            ),
            score=1.0 - index / 100,
        )
        for index, text in enumerate(texts)
    ]


class ScriptedGrader:
    """Returns a fixed pooled score per question, so a test states the loop.

    The thresholds are held here so the advisory verdict on each grade reads
    against the same boundary the loop policy applies. The boundary the
    decision is made at lives on ``SpineConfig``, which is what the policy
    reads.
    """

    name = "scripted"

    def __init__(
        self, scores: dict[str, float], thresholds: GradingThresholds | None = None
    ) -> None:
        self.scores = scores
        self.thresholds = thresholds or GradingThresholds()

    def grade(self, question, evidence, query_id=""):
        score = self.scores.get(question, 0.0)
        grades = tuple(
            ChunkGrade(chunk_id=item.chunk_id or item.doc_id, score=score, reason="s")
            for item in evidence
        )
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=self.thresholds.verdict(score),
            score=score,
            reason=f"scored {score}",
            chunk_grades=grades,
            thresholds=self.thresholds,
        )


class ScriptedRewriter:
    """Appends a marker so the retry retrieves with a different question."""

    name = "scripted-rewriter"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        if not self.enabled:
            return RewritePlan(original=question)
        return RewritePlan(
            original=question,
            rewrites=(Rewrite(question=f"{question} retried", strategy=self.name),),
        )


class ScriptedQuery:
    """The three fields the harvest reads off a corpus query."""

    def __init__(self, query_id: str, question: str, answerable: bool) -> None:
        self.query_id = query_id
        self.question = question
        self.doc_id = "doc1"
        self._answerable = answerable

    @property
    def is_answerable(self) -> bool:
        return self._answerable


class PreparingRewriter:
    """A pre-retrieval stage, which is what a query-analysis hook wraps."""

    name = "scripted-prepare"

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        return RewritePlan(
            original=question,
            rewrites=(Rewrite(question=f"{question} prepared", strategy=self.name),),
        )


class SplittingRewriter:
    """A pre-retrieval stage returning two candidates, so the plan is multi-step.

    This is the shape a query-decomposition stage has: one question in, several
    retrievals out. Driving it through both loops is what shows the harvest can
    carry a plan of more than one step.
    """

    name = "scripted-split"

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        return RewritePlan(
            original=question,
            rewrites=(
                Rewrite(question=f"{question} alpha", strategy=self.name),
                Rewrite(question=f"{question} beta", strategy=self.name),
            ),
        )


# A prose template, which is the second answer format the package already
# ships. It asks for no ``not_stated`` field, so a completer can tell the two
# apart, and it is the template R8 measured the harvest could not represent.
PROSE_TEMPLATE = (
    "Answer the question from the sources below, citing each as [1].\n\n"
    "Question: {query}\n\nSources:\n{sources}\n"
)


def test_the_harvest_retrieves_at_the_configured_depth() -> None:
    """A harvest at a depth above the default asks the retriever for it.

    Truncating a default-depth result cannot deepen it, so a harvest that only
    sliced would record fewer passages than a spine at the same setting, and the
    replay would then disagree with the pipeline it is meant to reproduce.
    """
    asked: list[int] = []

    def retrieve_deep(
        question: str, scope: str | None = None, k: int = 10
    ) -> list[Evidence]:
        asked.append(k)
        return evidence_for("doc1", [f"passage {n}" for n in range(k)])

    traces = harvest_traces(
        [ScriptedQuery("q1", "q", answerable=True)],
        retrieve_deep,
        ScriptedGrader({"q": 0.9}),
        prose_or_json_complete,
        {"q1": {"doc1::raw::0"}},
        top_k=20,
    )

    assert asked == [20]
    assert len(traces[0].attempts[0].evidence_ids) == 20


def retrieve_two(
    question: str, scope: str | None = None, k: int = 10
) -> list[Evidence]:
    """Return two passages whose identifiers differ between attempts."""
    doc = "doc1"
    if question.endswith("retried"):
        return evidence_for(doc, ["retried alpha", "retried beta"])[::-1]
    return evidence_for(doc, ["alpha text", "beta text"])


def retrieve_tagged(
    question: str, scope: str | None = None, k: int = 10
) -> list[Evidence]:
    """Return passages whose identifiers name the question that found them.

    Retrieval is question-sensitive, so a stage that changes the question before
    the first retrieval changes the identifiers. That is what makes a
    query-analysis hook visible in ``final_evidence_ids``.
    """
    if question.endswith("retried"):
        tag = "retried"
    elif question.endswith("prepared"):
        tag = "prepared"
    else:
        tag = "raw"
    return [
        Evidence(
            text=f"{tag} passage {index}",
            provenance=Provenance(
                doc_id="doc1",
                chunk_id=f"doc1::{tag}::{index}",
                start=index * 100,
                end=index * 100 + 20,
            ),
            score=1.0 - index / 100,
        )
        for index in range(2)
    ]


def retrieve_by_last_word(
    question: str, scope: str | None = None, k: int = 10
) -> list[Evidence]:
    """Return passages whose identifiers name the question's last word.

    ``retrieve_tagged`` collapses every sub-question of a multi-step plan onto
    one tag, and ``to_runs`` would then fold their identifiers together. Keying
    on the last word keeps each step's ranking distinct, so a plan of two steps
    is visible in ``final_evidence_ids``.
    """
    tag = question.rsplit(" ", 1)[-1]
    return [
        Evidence(
            text=f"{tag} passage {index}",
            provenance=Provenance(
                doc_id="doc1",
                chunk_id=f"doc1::{tag}::{index}",
                start=index * 100,
                end=index * 100 + 20,
            ),
            score=1.0 - index / 100,
        )
        for index in range(2)
    ]


class InterleavingPolicy:
    """A policy that chooses its next retrieval from a model call.

    It stands for IRCoT, whose retrieval target depends on the reasoning derived
    so far, so no recorded trace can answer it.

    It takes the thresholds a grid point would build it with and ignores them,
    so the class itself is a policy factory a sweep can be handed.
    """

    name = "interleaving"
    replayable = False

    def __init__(self, thresholds: GradingThresholds | None = None) -> None:
        self.thresholds = thresholds

    def next_move(self, state: LoopState) -> LoopDecision:
        raise AssertionError("a guarded policy is never asked for a move")


class StrictCorrectivePolicy:
    """A participant's own loop: answer only on a sufficient grade.

    It reads the thresholds it was built with, so a grid point it was not built
    at is a different measurement. That is what makes it the policy a sweep has
    to rebuild per point rather than reuse.
    """

    name = "strict-corrective"
    replayable = True

    def __init__(self, thresholds: GradingThresholds) -> None:
        self.thresholds = thresholds

    def next_move(self, state: LoopState) -> LoopDecision:
        if not state.had_evidence:
            return LoopDecision(Action.ABSTAIN)
        if self.thresholds.verdict(state.score) is Verdict.SUFFICIENT:
            return LoopDecision(Action.ANSWER)
        return LoopDecision(Action.ABSTAIN)


class AlwaysAnswerPolicy:
    """A replayable policy that is not the corrective loop."""

    name = "always-answer"
    replayable = True

    def next_move(self, state: LoopState) -> LoopDecision:
        return LoopDecision(Action.ANSWER)


def prose_or_json_complete(prompt: str) -> str:
    """Answer in the format the prompt asked for.

    The shipped template names ``not_stated`` and is answered with JSON that
    cites a source. The prose template is answered in prose with no citation
    marker, which the citation gate reads as an abstention. A completer that
    returns the same string whatever it is shown cannot tell the two templates
    apart, which is why the harvest carrying the wrong one used to go unnoticed.
    """
    if "not_stated" in prompt:
        return '{"answer": "yes", "sources": [1], "not_stated": false}'
    return "Ontario law governs."


def answering_complete(prompt: str) -> str:
    """Answer, citing source one, whatever the prompt asked."""
    return '{"answer": "yes", "sources": [1], "not_stated": false}'


def refusing_complete(prompt: str) -> str:
    """Report that the sources do not answer, whatever the prompt asked."""
    return '{"answer": "", "sources": [], "not_stated": true}'


def trace_with(
    scores: tuple[float, ...],
    *,
    query_id: str = "q1",
    is_null: bool = False,
    gold: frozenset[str] = frozenset({"doc1::000"}),
    generated_abstained: tuple[bool, ...] = (False, False),
    rewrite_available: tuple[bool, ...] = (True, False),
) -> QueryTrace:
    """Build a two-attempt trace with one pooled score per attempt."""
    attempts = tuple(
        AttemptTrace(
            question=f"q{index}",
            strategy="original" if index == 0 else "rewritten",
            evidence_ids=("doc1::000", "doc1::001"),
            scores=(score,),
            reason="because",
            generated_abstained=generated_abstained[index],
            rewrite_available=rewrite_available[index],
            grader_calls=5,
            rewrite_calls=1,
            generation_calls=1,
        )
        for index, score in enumerate(scores)
    )
    return QueryTrace(
        query_id=query_id,
        question="q0",
        is_null=is_null,
        gold_chunk_ids=gold,
        attempts=attempts,
    )


class TestAttemptTrace:
    def test_the_pooled_score_is_the_maximum_not_the_mean(self):
        attempt = trace_with((0.0,), rewrite_available=(False,)).attempts[0]
        assert attempt.score == 0.0

        many = AttemptTrace(
            question="q",
            strategy="original",
            evidence_ids=("a", "b", "c"),
            scores=(0.1, 0.9, 0.2),
            reason="r",
            generated_abstained=False,
            rewrite_available=False,
        )
        assert many.score == 0.9

    def test_an_empty_result_set_pools_to_zero(self):
        empty = AttemptTrace(
            question="q",
            strategy="original",
            evidence_ids=(),
            scores=(),
            reason="retrieval returned no passages",
            generated_abstained=True,
            rewrite_available=False,
        )
        assert empty.score == 0.0

    def test_a_trace_round_trips_through_its_dict_form(self):
        original = trace_with((0.8, 0.2))
        assert QueryTrace.from_dict(original.to_dict()) == original


class TestReplay:
    def test_a_sufficient_grade_answers_on_the_first_attempt(self):
        outcome = replay(trace_with((0.9, 0.1)), corrective(0.6, 0.3), 2)
        assert outcome.abstained is False
        assert outcome.attempts_used == 1
        assert outcome.abstained_by == ""

    def test_an_insufficient_grade_with_no_attempts_left_abstains(self):
        outcome = replay(trace_with((0.1, 0.1)), corrective(0.6, 0.3), 1)
        assert outcome.abstained is True
        assert outcome.abstained_by == "grader"
        assert outcome.attempts_used == 1

    def test_an_ambiguous_grade_on_the_last_attempt_answers(self):
        outcome = replay(trace_with((0.4, 0.4)), corrective(0.6, 0.3), 1)
        assert outcome.abstained is False

    def test_a_weak_first_grade_spends_the_second_attempt(self):
        outcome = replay(trace_with((0.1, 0.9)), corrective(0.6, 0.3), 2)
        assert outcome.attempts_used == 2
        assert outcome.abstained is False

    def test_lowering_the_answer_threshold_answers_the_same_trace(self):
        trace = trace_with((0.5, 0.5))
        assert replay(trace, corrective(0.6, 0.3), 1).attempts_used == 1
        strict = replay(trace, corrective(0.6, 0.3), 2)
        loose = replay(trace, corrective(0.4, 0.3), 2)
        assert strict.attempts_used == 2
        assert loose.attempts_used == 1

    def test_the_generator_can_abstain_on_evidence_the_grader_accepted(self):
        trace = trace_with((0.9, 0.9), generated_abstained=(True, True))
        outcome = replay(trace, corrective(0.6, 0.3), 2)
        assert outcome.abstained is True
        assert outcome.abstained_by == "generator"

    def test_an_unavailable_rewrite_settles_on_the_final_attempts_rule(self):
        trace = trace_with((0.1,), rewrite_available=(False,))
        outcome = replay(trace, corrective(0.6, 0.3), 2)
        assert outcome.abstained is True
        assert outcome.attempts_used == 1

    def test_model_calls_count_only_the_attempts_taken(self):
        answered = replay(trace_with((0.9, 0.9)), corrective(0.6, 0.3), 2)
        assert answered.model_calls == 6

        retried = replay(trace_with((0.1, 0.9)), corrective(0.6, 0.3), 2)
        assert retried.model_calls == 12

        abstained = replay(trace_with((0.1, 0.1)), corrective(0.6, 0.3), 2)
        assert abstained.model_calls == 11

    def test_replaying_past_the_end_of_a_trace_raises(self):
        trace = trace_with((0.1,), rewrite_available=(True,))
        with pytest.raises(ValueError, match="holds 1 attempts"):
            replay(trace, corrective(0.6, 0.3), 2)

    def test_a_zero_attempt_budget_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            replay(trace_with((0.9, 0.9)), CorrectivePolicy(GradingThresholds()), 0)


class TestReplayMatchesTheSpine:
    """The replay has to reproduce the spine, or the sweep measures nothing.

    Both hooks the answering interface adds are driven on both sides here. A
    harvest that could not carry them was measured to disagree with the spine:
    a non-default template moved the ``abstained`` flag, and a query-analysis
    stage moved the chunk identifiers.

    The multi-step point drives a plan of two retrievals through both loops. It
    is the path a query-decomposition stage takes, and the two sides can part
    on it in ways one step cannot show: the harvest has to record every step's
    ranking, and the corrective retry has to stay as wide as the first attempt.
    """

    @pytest.mark.parametrize("max_attempts", [1, 2])
    @pytest.mark.parametrize("rewriter_enabled", [True, False])
    @pytest.mark.parametrize("score", [0.0, 0.2, 0.35, 0.6, 0.95])
    @pytest.mark.parametrize(
        "thresholds",
        threshold_grid([0.4, 0.6, 0.8], [0.1, 0.3, 0.5]),
        ids=str,
    )
    @pytest.mark.parametrize(
        "hooks", ["shipped", "analysis-and-prompt", "multi-step-analysis"]
    )
    def test_the_replay_agrees_with_the_spine_on_every_grid_point(
        self, hooks, thresholds, score, rewriter_enabled, max_attempts
    ):
        if hooks == "multi-step-analysis":
            analysis = RewriterAnalysis(SplittingRewriter(), max_steps=2)
            retrieve = retrieve_by_last_word
            asked = "q alpha | q beta"
            gold = "doc1::alpha::0"
        elif hooks == "analysis-and-prompt":
            analysis = RewriterAnalysis(PreparingRewriter())
            retrieve = retrieve_tagged
            asked = "q prepared"
            gold = "doc1::raw::0"
        else:
            analysis = None
            retrieve = retrieve_tagged
            asked = "q"
            gold = "doc1::raw::0"
        prompt = FixedPromptSelector(PROSE_TEMPLATE) if hooks != "shipped" else None
        scores = {asked: score, "q retried": score}
        rewriter = ScriptedRewriter(enabled=rewriter_enabled)

        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve,
            ScriptedGrader(scores),
            prose_or_json_complete,
            {"q1": {gold}},
            analysis=analysis,
            rewriter=rewriter,
            prompt=prompt,
            attempts=max_attempts,
        )
        replayed = replay(traces[0], CorrectivePolicy(thresholds), max_attempts)

        spine = AdvancedRagSpine(
            retrieve=retrieve,
            grader=ScriptedGrader(scores, thresholds),
            complete=prose_or_json_complete,
            rewriter=rewriter,
            config=SpineConfig(max_attempts=max_attempts, thresholds=thresholds),
            analysis=analysis,
            prompt=prompt,
        )
        result = spine.answer("q1", "q", scope="doc1")

        assert replayed.abstained == result.abstained
        assert replayed.attempts_used == len(result.attempts)
        assert list(replayed.final_evidence_ids) == result.retrieved_ids

    def test_a_multi_step_plan_reaches_the_replay_as_the_plan_it_was(self):
        # `replay` built the plan with no steps at all. `CorrectivePolicy` reads
        # neither the steps nor the outcome, so the grid agreed, and a policy
        # that chose its next retrieval from `state.plan.steps` would have seen
        # a decomposition on the spine and an empty plan here.
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_by_last_word,
            ScriptedGrader({"q alpha | q beta": 0.9}),
            answering_complete,
            {"q1": set()},
            analysis=RewriterAnalysis(SplittingRewriter(), max_steps=2),
            attempts=1,
        )
        trace = traces[0]

        assert trace.step_questions == ("q alpha", "q beta")
        assert trace.attempts[0].rewrite_outcome == "rewritten"
        assert list(trace.attempts[0].evidence_ids) == [
            "doc1::alpha::0",
            "doc1::alpha::1",
            "doc1::beta::0",
            "doc1::beta::1",
        ]

        seen: list[tuple[str, ...]] = []

        class PlanReadingPolicy:
            """Records the plan it was handed, which is what it decides on."""

            name = "plan-reading"
            replayable = True

            def next_move(self, state: LoopState) -> LoopDecision:
                seen.append(tuple(step.question for step in state.plan.steps))
                return LoopDecision(Action.ANSWER)

        replay(trace, PlanReadingPolicy(), 1)
        assert seen == [("q alpha", "q beta")]

    def test_a_trace_round_trips_through_its_dict_with_the_new_fields(self):
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_by_last_word,
            ScriptedGrader({"q alpha | q beta": 0.9}),
            answering_complete,
            {"q1": set()},
            analysis=RewriterAnalysis(SplittingRewriter(), max_steps=2),
            attempts=1,
        )
        restored = QueryTrace.from_dict(json.loads(json.dumps(traces[0].to_dict())))

        assert restored.step_questions == ("q alpha", "q beta")
        assert restored.attempts[0].rewrite_outcome == "rewritten"

    def test_the_analysis_stage_reaches_both_loops(self):
        analysis = RewriterAnalysis(PreparingRewriter())
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_tagged,
            ScriptedGrader({"q prepared": 0.9}),
            answering_complete,
            {"q1": set()},
            analysis=analysis,
            attempts=1,
        )
        attempt = traces[0].attempts[0]
        assert attempt.question == "q prepared"
        assert attempt.strategy == "scripted-prepare"
        assert attempt.parent == "q"
        assert attempt.scope == "doc1"
        assert list(attempt.evidence_ids) == ["doc1::prepared::0", "doc1::prepared::1"]

        spine = AdvancedRagSpine(
            retrieve=retrieve_tagged,
            grader=ScriptedGrader({"q prepared": 0.9}, GradingThresholds(0.6, 0.3)),
            complete=answering_complete,
            config=SpineConfig(max_attempts=1, thresholds=GradingThresholds(0.6, 0.3)),
            analysis=analysis,
        )
        result = spine.answer("q1", "q", scope="doc1")
        assert list(replay(traces[0], corrective(0.6, 0.3), 1).final_evidence_ids) == (
            result.retrieved_ids
        )

    def test_it_also_agrees_when_the_generator_refuses_under_another_template(self):
        thresholds = GradingThresholds(0.6, 0.3)
        scores = {"q": 0.9, "q retried": 0.9}
        prompt = FixedPromptSelector(PROSE_TEMPLATE)
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader(scores),
            prose_or_json_complete,
            {"q1": {"doc1::000"}},
            rewriter=ScriptedRewriter(),
            prompt=prompt,
            attempts=2,
        )
        spine = AdvancedRagSpine(
            retrieve=retrieve_two,
            grader=ScriptedGrader(scores, thresholds),
            complete=prose_or_json_complete,
            rewriter=ScriptedRewriter(),
            config=SpineConfig(max_attempts=2, thresholds=thresholds),
            prompt=prompt,
        )
        replayed = replay(traces[0], CorrectivePolicy(thresholds), 2)
        result = spine.answer("q1", "q", scope="doc1")

        # The grade is sufficient, so the loop answers and the generator is the
        # only stage that can decline. It declines because the prose answer
        # carries no citation. Under the shipped template the same completer
        # answers, so the two sides agree here only if both render the template
        # the spine was given.
        assert replayed.abstained is True
        assert replayed.abstained_by == "generator"
        assert replayed.abstained == result.abstained


# Up from this file to the repository root: evaluation, advanced_rag, aieng,
# tests, aieng-advanced-rag.
REPO_ROOT = Path(__file__).resolve().parents[5]
COMMITTED_TRACES = sorted(
    (REPO_ROOT / "implementations/regulatory_policy_qa/outputs").glob("traces_*.json")
)


@pytest.mark.parametrize("path", COMMITTED_TRACES, ids=lambda path: path.stem)
def test_a_committed_trace_loads_without_the_fields_added_after_it(path):
    """The recorded runs the reported figures came from have to keep loading.

    Six harvests are committed and none of them carries ``step_questions`` or
    ``rewrite_outcome``. A required key on the read side would make every one of
    them unreplayable, so the figures could not be re-derived from the recording
    they were measured on.
    """
    payload = json.loads(path.read_text())
    traces = [QueryTrace.from_dict(entry) for entry in payload]

    assert traces
    assert all("step_questions" not in entry for entry in payload)
    assert all(trace.step_questions == () for trace in traces)
    assert all(
        attempt.rewrite_outcome == "" for trace in traces for attempt in trace.attempts
    )
    # A trace recorded before the plan was is replayed as the plan with no
    # steps, which is the fallback `retrieval_steps` already applied.
    replayed = replay(traces[0], corrective(0.6, 0.3), 1)
    assert replayed.query_id == traces[0].query_id


def test_the_committed_traces_are_all_present():
    # The parametrized load test would pass vacuously if the outputs moved.
    assert len(COMMITTED_TRACES) == 6


class TestHarvest:
    def test_it_records_every_attempt_whatever_the_grade_says(self):
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.99, "q retried": 0.99}),
            answering_complete,
            {"q1": {"doc1::000"}},
            rewriter=ScriptedRewriter(),
            attempts=2,
        )
        assert len(traces[0].attempts) == 2
        assert traces[0].attempts[1].question == "q retried"

    def test_it_stops_when_the_rewriter_has_nothing_to_offer(self):
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.1}),
            answering_complete,
            {"q1": set()},
            rewriter=ScriptedRewriter(enabled=False),
            attempts=2,
        )
        assert len(traces[0].attempts) == 1
        assert traces[0].attempts[0].rewrite_available is False

    def test_it_records_why_generation_declined(self):
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.9}),
            refusing_complete,
            {"q1": {"doc1::000"}},
            attempts=1,
        )
        attempt = traces[0].attempts[0]
        assert attempt.generated_abstained is True
        assert "do not answer" in attempt.generated_reason

    def test_an_answered_attempt_records_no_reason(self):
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.9}),
            answering_complete,
            {"q1": {"doc1::000"}},
            attempts=1,
        )
        assert traces[0].attempts[0].generated_reason == ""

    def test_it_carries_the_null_label_and_the_judgment_set(self):
        traces = harvest_traces(
            [
                ScriptedQuery("q1", "q", answerable=True),
                ScriptedQuery("q2", "q", answerable=False),
            ],
            retrieve_two,
            ScriptedGrader({"q": 0.9}),
            answering_complete,
            {"q1": {"doc1::000"}},
            attempts=1,
        )
        assert traces[0].is_null is False
        assert traces[0].gold_chunk_ids == frozenset({"doc1::000"})
        assert traces[1].is_null is True
        assert traces[1].gold_chunk_ids == frozenset()

    def test_a_shared_counter_prices_each_stage_separately(self):
        seen = {"calls": 0}

        def counted(prompt: str) -> str:
            seen["calls"] += 1
            return answering_complete(prompt)

        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.9, "q retried": 0.9}),
            counted,
            {"q1": {"doc1::000"}},
            rewriter=ScriptedRewriter(),
            attempts=2,
            count_calls=lambda: seen["calls"],
        )
        attempt = traces[0].attempts[0]
        assert attempt.grader_calls == 0
        assert attempt.generation_calls == 1
        assert attempt.rewrite_calls == 0

    def test_the_progress_hook_sees_each_finished_trace(self):
        seen: list[str] = []
        harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader({"q": 0.9}),
            answering_complete,
            {"q1": {"doc1::000"}},
            attempts=1,
            on_query=lambda trace: seen.append(trace.query_id),
        )
        assert seen == ["q1"]


class TestControls:
    def test_always_answer_never_abstains(self):
        assert always_answer(trace_with((0.0, 0.0))).abstained is False

    def test_always_abstain_never_answers_and_costs_nothing(self):
        outcome = always_abstain(trace_with((0.9, 0.9)))
        assert outcome.abstained is True
        assert outcome.model_calls == 0

    def test_both_trivial_controls_land_on_a_balanced_accuracy_of_a_half(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        rows = control_rows(traces)
        assert [row.label for row in rows] == [
            "no grader",
            "always answer",
            "always abstain",
        ]
        trivial = [row for row in rows if row.label != "no grader"]
        assert all(row.balanced_accuracy == 0.5 for row in trivial)

    def test_the_generator_only_control_keeps_the_generators_own_decision(self):
        answered = trace_with((0.0, 0.0), generated_abstained=(False, False))
        declined = trace_with((1.0, 1.0), generated_abstained=(True, True))
        assert generator_only(answered).abstained is False
        assert generator_only(declined).abstained is True
        assert generator_only(declined).abstained_by == "generator"

    def test_the_generator_only_control_ignores_the_thresholds(self):
        trace = trace_with((0.0, 0.0), generated_abstained=(False, False))
        assert replay(trace, corrective(0.6, 0.3), 1).abstained is True
        assert generator_only(trace).abstained is False

    def test_the_generator_only_control_spends_one_generation(self):
        assert generator_only(trace_with((0.5, 0.5))).model_calls == 1
        assert generator_only(trace_with((0.5, 0.5))).attempts_used == 1

    def test_always_answer_has_a_false_answer_rate_of_one(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        _, answer, abstain = control_rows(traces)
        assert answer.null.false_answer_rate == 1.0
        assert abstain.null.abstain_recall == 1.0
        assert abstain.answerable.abstain_rate == 1.0


class TestAnswerableSide:
    def test_the_cost_is_the_share_of_answerable_questions_refused(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a"),
            trace_with((0.1, 0.1), query_id="b"),
        ]
        outcomes = [replay(t, corrective(0.6, 0.3), 1) for t in traces]
        report = score_answerable(outcomes, traces)
        assert report.answerable_queries == 2
        assert report.abstain_rate == 0.5

    def test_an_answer_off_the_wrong_passage_is_not_a_grounded_answer(self):
        traces = [trace_with((0.9, 0.9), query_id="a", gold=frozenset({"other"}))]
        outcomes = [replay(traces[0], corrective(0.6, 0.3), 1)]
        report = score_answerable(outcomes, traces)
        assert report.abstain_rate == 0.0
        assert report.grounded_answer_rate == 0.0

    def test_only_the_passages_the_generator_read_can_ground_an_answer(self):
        traces = [trace_with((0.9, 0.9), query_id="a", gold=frozenset({"doc1::001"}))]
        outcomes = [replay(traces[0], corrective(0.6, 0.3), 1)]
        assert score_answerable(outcomes, traces).grounded_answer_rate == 1.0
        narrow = score_answerable(outcomes, traces, evidence_budget=1)
        assert narrow.grounded_answer_rate == 0.0

    def test_null_only_trace_sets_raise_rather_than_report_zero(self):
        traces = [trace_with((0.1, 0.1), query_id="a", is_null=True, gold=frozenset())]
        outcomes = [replay(traces[0], corrective(0.6, 0.3), 1)]
        with pytest.raises(ValueError, match="No answerable queries"):
            score_answerable(outcomes, traces)


class TestSweep:
    def test_the_grid_drops_inverted_threshold_pairs(self):
        grid = threshold_grid([0.4, 0.6], [0.3, 0.5])
        assert (0.4, 0.5) not in [(t.sufficient, t.insufficient) for t in grid]
        assert len(grid) == 3

    def test_a_sweep_scores_every_pairing_at_every_attempt_budget(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        rows = sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1, 2))
        assert len(rows) == 2
        assert [row.max_attempts for row in rows] == [1, 2]

    def test_a_row_records_how_many_abstentions_the_grader_made(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        row = sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))[0]
        assert row.grader_abstentions == 1
        assert row.to_dict()["grader_abstentions"] == 1

    def test_a_row_reports_the_null_side_and_the_answerable_side(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        row = sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))[0]
        assert row.null.abstain_recall == 1.0
        assert row.answerable.abstain_rate == 0.0
        assert row.balanced_accuracy == 1.0

    def test_a_row_renders_under_the_shared_table_header(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        row = sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))[0]
        header = TABLE_HEADER.splitlines()[0]
        assert header.count("|") == row.table_row().count("|")
        assert row.to_dict()["label"] == "0.60 / 0.30"

    def test_a_row_round_trips_through_its_dict_form(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        row = sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))[0]
        assert SweepRow.from_dict(row.to_dict()) == row

    def test_a_control_row_round_trips_with_no_thresholds_on_it(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        control = control_rows(traces)[0]
        assert SweepRow.from_dict(control.to_dict()) == control

    def test_score_arm_labels_a_row_without_thresholds(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=False),
            trace_with((0.1, 0.1), query_id="b", is_null=True, gold=frozenset()),
        ]
        outcomes = [always_answer(trace) for trace in traces]
        row = score_arm(
            traces,
            outcomes,
            label="always answer",
            grader="control",
            sufficient=None,
            insufficient=None,
            max_attempts=1,
        )
        assert row.to_dict()["sufficient"] is None


class TestTheLoopPolicyOwnsTheThresholds:
    """``SpineConfig.thresholds`` is what the decision is made at.

    The loop policy reads that field and computes the verdict from the pooled
    score, exactly as the replay does, so a sweep turns one setting rather than
    two. The grader keeps its own thresholds for grading a single passage and
    for making a report self-describing, and the verdict it reports is
    advisory: it is recorded on the attempt and it does not drive the action.
    """

    def test_spine_config_thresholds_drive_the_decision(self):
        scores = {"q": 0.5}
        spine = AdvancedRagSpine(
            retrieve=retrieve_two,
            grader=ScriptedGrader(scores, GradingThresholds(0.4, 0.1)),
            complete=answering_complete,
            config=SpineConfig(max_attempts=1, thresholds=GradingThresholds(0.9, 0.8)),
        )
        result = spine.answer("q1", "q", scope="doc1")
        # The grader calls 0.5 sufficient against 0.4 and says so on the record.
        assert result.attempts[0].verdict is Verdict.SUFFICIENT
        # The configuration calls it insufficient against 0.8, and with no
        # attempts left that abstains without a generation call.
        assert result.attempts[0].action is Action.ABSTAIN
        assert result.abstained is True

    def test_the_grader_is_followed_when_the_configuration_states_nothing(self):
        scores = {"q": 0.5}
        spine = AdvancedRagSpine(
            retrieve=retrieve_two,
            grader=ScriptedGrader(scores, GradingThresholds(0.9, 0.8)),
            complete=answering_complete,
            config=SpineConfig(max_attempts=1),
            policy=CorrectivePolicy(),
        )
        result = spine.answer("q1", "q", scope="doc1")
        assert result.attempts[0].action is Action.ABSTAIN
        assert result.abstained is True

    def test_the_replay_and_the_spine_read_the_same_field(self):
        scores = {"q": 0.5}
        thresholds = GradingThresholds(0.9, 0.8)
        traces = harvest_traces(
            [ScriptedQuery("q1", "q", answerable=True)],
            retrieve_two,
            ScriptedGrader(scores),
            answering_complete,
            {"q1": {"doc1::000"}},
            attempts=1,
        )
        spine = AdvancedRagSpine(
            retrieve=retrieve_two,
            grader=ScriptedGrader(scores, GradingThresholds(0.4, 0.1)),
            complete=answering_complete,
            config=SpineConfig(max_attempts=1, thresholds=thresholds),
        )
        assert (
            replay(traces[0], CorrectivePolicy(thresholds), 1).abstained
            == spine.answer("q1", "q", scope="doc1").abstained
        )


class TestASweptPolicyTracksTheGridPoint:
    """Every row reports the measurement its own threshold label was made at.

    A custom policy used to be built once and handed to every point, so
    ``thresholds`` was dead and a six-point grid printed one measurement under
    six different labels. The sweep takes a factory and builds the policy at
    each point instead.
    """

    def traces(self) -> list[QueryTrace]:
        """Return one answerable trace and one null trace, both mid-scored."""
        return [
            trace_with((0.5, 0.5), query_id="a", is_null=False),
            trace_with((0.2, 0.2), query_id="b", is_null=True, gold=frozenset()),
        ]

    def test_each_point_builds_the_policy_at_its_own_thresholds(self):
        seen: list[GradingThresholds] = []

        def factory(thresholds: GradingThresholds) -> CorrectivePolicy:
            seen.append(thresholds)
            return CorrectivePolicy(thresholds)

        grid = threshold_grid([0.4, 0.6], [0.1, 0.3])
        sweep(self.traces(), grid, attempts=(1,), policy_factory=factory)
        assert seen == list(grid)

    def test_a_custom_policy_is_measured_at_every_point(self):
        grid = threshold_grid([0.4, 0.6], [0.1, 0.3])
        rows = sweep(
            self.traces(),
            grid,
            attempts=(1,),
            policy_factory=StrictCorrectivePolicy,
        )
        # The strict loop answers the 0.5 trace at a 0.40 boundary and declines
        # it at 0.60, so the answer threshold moves the measurement and the
        # rows cannot all carry the same one.
        assert [row.label for row in rows] == [
            "0.40 / 0.10",
            "0.40 / 0.30",
            "0.60 / 0.10",
            "0.60 / 0.30",
        ]
        assert [row.grader_abstentions for row in rows] == [1, 1, 2, 2]
        assert len({row.balanced_accuracy for row in rows}) > 1


class TestTheReplayableGuard:
    """A policy that cannot be replayed has to be refused, not approximated.

    A policy that calls a model inside its own decision cannot be answered from
    a recording at any thresholds. Reporting a grid row for it would report an
    operating point nobody ran.
    """

    def test_replay_refuses_a_policy_that_cannot_be_replayed(self):
        with pytest.raises(ValueError, match="not replayable"):
            replay(trace_with((0.9, 0.9)), InterleavingPolicy(), 2)

    def test_the_error_says_what_to_do_instead(self):
        with pytest.raises(ValueError, match="has to be run, not replayed"):
            replay(trace_with((0.9, 0.9)), InterleavingPolicy(), 2)

    def test_a_sweep_refuses_it_too(self):
        traces = [trace_with((0.9, 0.9), query_id="a", is_null=False)]
        with pytest.raises(ValueError, match="not replayable"):
            sweep(
                traces,
                threshold_grid([0.6], [0.3]),
                attempts=(1,),
                policy_factory=InterleavingPolicy,
            )

    def test_a_null_only_sweep_refuses_it_too(self):
        traces = [
            trace_with((0.9, 0.9), query_id="a", is_null=True, gold=frozenset()),
        ]
        with pytest.raises(ValueError, match="not replayable"):
            null_only_sweep(
                traces,
                threshold_grid([0.6], [0.3]),
                attempts=(1,),
                policy_factory=InterleavingPolicy,
            )

    def test_a_replayable_policy_is_driven_over_the_recording(self):
        outcome = replay(trace_with((0.0, 0.0)), AlwaysAnswerPolicy(), 2)
        assert outcome.abstained is False
        assert outcome.attempts_used == 1


class TestNullOnlySweep:
    """The out-of-domain control has no answerable side to report."""

    def test_it_reports_the_share_of_far_nulls_declined(self):
        traces = [
            trace_with((0.1, 0.1), query_id="a", is_null=True, gold=frozenset()),
            trace_with((0.9, 0.9), query_id="b", is_null=True, gold=frozenset()),
        ]
        rows = null_only_sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))
        assert rows[0]["abstain_rate"] == 0.5
        assert rows[0]["false_answer_rate"] == 0.5
        assert rows[0]["null_queries"] == 2

    def test_it_splits_the_two_abstention_paths(self):
        traces = [
            trace_with((0.1, 0.1), query_id="a", is_null=True, gold=frozenset()),
            trace_with(
                (0.9, 0.9),
                query_id="b",
                is_null=True,
                gold=frozenset(),
                generated_abstained=(True, True),
            ),
        ]
        row = null_only_sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))[0]
        assert row["grader_abstentions"] == 1
        assert row["generator_abstentions"] == 1

    def test_an_answerable_query_in_the_control_raises(self):
        traces = [trace_with((0.1, 0.1), query_id="a", is_null=False)]
        with pytest.raises(ValueError, match="answerable queries in a null-only"):
            null_only_sweep(traces, threshold_grid([0.6], [0.3]), attempts=(1,))


class TestVerdictBoundaries:
    def test_a_score_exactly_on_the_answer_threshold_answers(self):
        assert GradingThresholds(0.6, 0.3).verdict(0.6) is Verdict.SUFFICIENT

    def test_a_score_exactly_on_the_rewrite_threshold_is_ambiguous(self):
        assert GradingThresholds(0.6, 0.3).verdict(0.3) is Verdict.AMBIGUOUS
