"""Tests for the answering interface: query analysis, a loop policy, a prompt switch.

The load-bearing test is ``TestCorrectivePolicy``. The policy has to reproduce
``decide`` at every cell, because both loops now take their decision from it and
a policy that drifted would move a measured number on the sweep's side alone.
"""

from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import pytest

from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline import spine as spine_module
from aieng.advanced_rag.pipeline.answering import (
    Answerer,
    CorrectivePolicy,
    DecompositionAnalysis,
    FixedPromptSelector,
    LoopDecision,
    LoopState,
    PassthroughAnalysis,
    PromptSelector,
    QueryPlan,
    RetrievalStep,
    RetrieverPromptSelector,
    RewriterAnalysis,
    SingleShotPolicy,
    merge_steps,
    step_bound,
)
from aieng.advanced_rag.pipeline.generate import GENERATION_TEMPLATE
from aieng.advanced_rag.pipeline.grading import (
    Action,
    EvidenceGrade,
    GradingThresholds,
    Verdict,
    decide,
)
from aieng.advanced_rag.pipeline.rewrite import Rewrite, RewritePlan, TemplateRewriter
from aieng.advanced_rag.provenance import Provenance


def state_for(
    score: float,
    *,
    attempts_left: int = 1,
    had_evidence: bool = True,
    rewrite_available: bool = True,
    grade: EvidenceGrade | None = None,
    notes: tuple[str, ...] = (),
) -> LoopState:
    """Build a loop state over one question, with nothing else varying."""
    plan = QueryPlan(question="who signs")
    return LoopState(
        plan=plan,
        step=plan.retrieval_steps()[0],
        score=score,
        had_evidence=had_evidence,
        attempts_left=attempts_left,
        rewrite_available=rewrite_available,
        grade=grade,
        notes=notes,
    )


def grade_at(thresholds: GradingThresholds, score: float) -> EvidenceGrade:
    """Return a grade carrying the thresholds it was made with."""
    return EvidenceGrade(
        query_id="q1",
        question="who signs",
        verdict=thresholds.verdict(score),
        score=score,
        reason="scored",
        thresholds=thresholds,
    )


def passage(retriever: str, index: int = 0) -> Evidence:
    """Return one passage attributed to a named retrieval method."""
    return Evidence(
        text=f"passage {index}",
        provenance=Provenance(doc_id="doc1", chunk_id=f"doc1::{index:03d}"),
        score=1.0,
        retriever=retriever,
    )


class FixedRewriter:
    """Returns one rewrite, or nothing when built with an empty string."""

    name = "fixed-rewriter"

    def __init__(self, rewritten: str = "stripped question") -> None:
        self.rewritten = rewritten

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        if not self.rewritten:
            return RewritePlan(original=question, outcome="no-change")
        return RewritePlan(
            original=question,
            rewrites=(Rewrite(self.rewritten, self.name),),
        )


class ThreeRewriter:
    """Returns three candidates, which is what a bound above one retrieves."""

    name = "three-rewriter"

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        return RewritePlan(
            original=question,
            rewrites=tuple(
                Rewrite(f"{question} v{index}", self.name) for index in (1, 2, 3)
            ),
        )


def test_the_answerer_protocol_names_what_it_returns():
    # It returned `Any`, which checks nothing. The circular import it cited is
    # avoided by `TYPE_CHECKING` plus `from __future__ import annotations`.
    hints = get_type_hints(
        Answerer.answer, localns={"SpineResult": spine_module.SpineResult}
    )
    assert hints["return"] is spine_module.SpineResult


class TestQueryPlan:
    def test_a_plan_with_no_steps_retrieves_the_question_as_asked(self):
        plan = QueryPlan(question="who signs")
        assert plan.retrieval_steps() == (RetrievalStep(question="who signs"),)

    def test_a_decomposition_keeps_its_steps_in_order(self):
        steps = (
            RetrievalStep(question="who signs"),
            RetrievalStep(question="when was it signed"),
        )
        plan = QueryPlan(question="who signs and when", steps=steps)
        assert plan.retrieval_steps() == steps

    def test_a_plan_carries_no_advisory_field_that_nothing_reads(self):
        # `hints` was documented as steering for the retrieval interface with no
        # writer and no reader. `retrieval.interface.Hints` is the live record.
        with pytest.raises(TypeError):
            QueryPlan(question="who signs", hints={"entities": ("Acme",)})

    def test_merging_one_step_returns_it_unchanged(self):
        step = RetrievalStep(question="who signs", scope="doc1", strategy="template")
        assert merge_steps([step]) is step

    def test_merging_several_steps_names_every_question_retrieved(self):
        merged = merge_steps(
            [
                RetrievalStep(question="who signs", scope="doc1", parent="both"),
                RetrievalStep(question="when", scope="doc1", parent="both"),
            ]
        )
        assert merged.question == "who signs | when"
        assert merged.scope == "doc1"
        assert merged.parent == "both"

    def test_merging_nothing_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            merge_steps([])


class TestAnalysis:
    def test_no_analysis_retrieves_the_question_under_the_callers_scope(self):
        plan = PassthroughAnalysis().analyse("who signs", "doc1")
        assert plan.question == "who signs"
        assert plan.retrieval_steps() == (
            RetrievalStep(question="who signs", scope="doc1"),
        )

    def test_a_rewriter_becomes_a_one_step_plan_naming_its_strategy(self):
        plan = RewriterAnalysis(FixedRewriter()).analyse("who signs", "doc1")
        step = plan.retrieval_steps()[0]
        assert plan.question == "who signs"
        assert step.question == "stripped question"
        assert step.strategy == "fixed-rewriter"
        assert step.parent == "who signs"
        assert step.scope == "doc1"

    def test_a_rewriter_that_finds_nothing_keeps_the_question_as_asked(self):
        plan = RewriterAnalysis(FixedRewriter(rewritten="")).analyse("who signs")
        assert plan.retrieval_steps() == (RetrievalStep(question="who signs"),)

    def test_the_shipped_template_rewriter_satisfies_the_protocol(self):
        analysis = RewriterAnalysis(TemplateRewriter())
        question = (
            'Highlight the parts (if any) of this contract related to "Governing '
            'Law" that should be reviewed by a lawyer.'
        )
        step = analysis.analyse(question).retrieval_steps()[0]
        assert step.question == '"Governing Law"'
        assert analysis.name == TemplateRewriter().name

    def test_one_step_is_the_default_whatever_the_rewriter_offers(self):
        # The regression lock. `pooled_score` pools by maximum, so a second
        # candidate can only raise an attempt's score, which moves the
        # abstention rate the sweep reports. Every committed figure was measured
        # at one step, so one step is what the default has to retrieve.
        analysis = RewriterAnalysis(ThreeRewriter())
        plan = analysis.analyse("who signs", "doc1")

        assert analysis.max_steps == 1
        assert plan.steps == (
            RetrievalStep(
                question="who signs v1",
                scope="doc1",
                strategy="three-rewriter",
                parent="who signs",
            ),
        )

    def test_a_wider_bound_retrieves_one_step_per_candidate_in_order(self):
        plan = RewriterAnalysis(ThreeRewriter(), max_steps=3).analyse(
            "who signs", "doc1"
        )

        assert [step.question for step in plan.steps] == [
            "who signs v1",
            "who signs v2",
            "who signs v3",
        ]
        assert {step.strategy for step in plan.steps} == {"three-rewriter"}
        assert {step.parent for step in plan.steps} == {"who signs"}
        assert {step.scope for step in plan.steps} == {"doc1"}

    def test_the_bound_caps_the_candidates_it_does_not_pad_them(self):
        plan = RewriterAnalysis(ThreeRewriter(), max_steps=2).analyse("who signs")
        assert len(plan.steps) == 2

        plan = RewriterAnalysis(FixedRewriter(), max_steps=3).analyse("who signs")
        assert len(plan.steps) == 1

    def test_a_bound_below_one_would_plan_no_retrieval_and_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            RewriterAnalysis(FixedRewriter(), max_steps=0)

    def test_the_plan_carries_what_the_rewriter_did(self):
        rewritten = RewriterAnalysis(FixedRewriter()).analyse("who signs")
        untouched = RewriterAnalysis(FixedRewriter(rewritten="")).analyse("who signs")

        assert rewritten.rewrite_outcome == "rewritten"
        # The step is the question as asked either way, so the outcome is the
        # only record that a rewriter ran at all.
        assert untouched.rewrite_outcome == "no-change"
        assert untouched.retrieval_steps() == (RetrievalStep(question="who signs"),)

    def test_a_passthrough_plan_names_no_rewrite_outcome(self):
        assert PassthroughAnalysis().analyse("who signs").rewrite_outcome == ""


class TestDecompositionAnalysis:
    """The multi-step stage a decomposition composes instead of reimplementing."""

    def test_it_emits_one_step_per_sub_question(self):
        analysis = DecompositionAnalysis(
            lambda question: ["who signs", "when was it signed"]
        )
        plan = analysis.analyse("who signs and when", "doc1")

        assert plan.question == "who signs and when"
        assert plan.steps == (
            RetrievalStep(
                question="who signs",
                scope="doc1",
                strategy="decomposition",
                parent="who signs and when",
            ),
            RetrievalStep(
                question="when was it signed",
                scope="doc1",
                strategy="decomposition",
                parent="who signs and when",
            ),
        )

    def test_the_trace_row_names_every_sub_question_retrieved(self):
        # `merge_steps` is what collapses the plan into the one step an attempt
        # records, so this stage does not need a collapsing rule of its own.
        plan = DecompositionAnalysis(lambda question: ["who signs", "when"]).analyse(
            "both", "doc1"
        )
        merged = merge_steps(plan.retrieval_steps())

        assert merged.question == "who signs | when"
        assert merged.strategy == "decomposition"
        assert merged.parent == "both"

    def test_a_splitter_that_declines_leaves_the_question_as_asked(self):
        plan = DecompositionAnalysis(lambda question: []).analyse("who signs", "doc1")
        assert plan.retrieval_steps() == (
            RetrievalStep(question="who signs", scope="doc1"),
        )

    def test_repeated_and_empty_sub_questions_are_dropped(self):
        plan = DecompositionAnalysis(
            lambda question: ["who signs", "", "who signs", "when"]
        ).analyse("both")
        assert [step.question for step in plan.steps] == ["who signs", "when"]

    def test_the_bound_caps_the_sub_questions_retrieved(self):
        split = DecompositionAnalysis(
            lambda question: ["a question", "b question", "c question"], max_steps=2
        )
        assert len(split.analyse("both").steps) == 2

    def test_a_bound_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            DecompositionAnalysis(lambda question: ["a"], max_steps=0)

    def test_its_name_labels_every_step_and_is_configurable(self):
        split = DecompositionAnalysis(lambda question: ["a question"], name="ircot")
        assert split.name == "ircot"
        assert split.analyse("both").steps[0].strategy == "ircot"


class TestStepBound:
    """The retry reads the first attempt's bound off the analysis stage."""

    def test_an_unbounded_stage_reports_no_bound(self):
        assert (
            step_bound(DecompositionAnalysis(lambda question: ["a question"])) is None
        )

    def test_a_rewriter_stage_reports_its_own_bound(self):
        assert step_bound(RewriterAnalysis(FixedRewriter(), max_steps=3)) == 3

    def test_a_stage_with_no_bound_of_its_own_is_the_shipped_single_step_kind(self):
        assert step_bound(PassthroughAnalysis()) == 1


class TestCorrectivePolicy:
    """The policy has to be ``decide`` over the pooled score, at every cell."""

    @pytest.mark.parametrize("attempts_left", [0, 1, 2])
    @pytest.mark.parametrize("score", [0.0, 0.05, 0.2, 0.35, 0.6, 0.95, 1.0])
    @pytest.mark.parametrize(
        "thresholds",
        [
            GradingThresholds(0.6, 0.0),
            GradingThresholds(0.6, 0.3),
            GradingThresholds(0.4, 0.1),
            GradingThresholds(0.8, 0.5),
        ],
        ids=str,
    )
    def test_it_reproduces_decide_over_the_pooled_score(
        self, thresholds, score, attempts_left
    ):
        policy = CorrectivePolicy(thresholds)
        state = state_for(score, attempts_left=attempts_left)
        expected = decide(thresholds.verdict(score), attempts_left)
        assert policy.next_move(state).action is expected

    @pytest.mark.parametrize("attempts_left", [0, 1])
    def test_an_empty_result_set_is_insufficient_whatever_the_score_says(
        self, attempts_left
    ):
        # At the shipped ``insufficient`` of 0.0 no score is below it, so an
        # empty result set cannot be told from a poor one by score alone.
        policy = CorrectivePolicy(GradingThresholds())
        state = state_for(0.0, attempts_left=attempts_left, had_evidence=False)
        assert policy.verdict_for(state) is Verdict.INSUFFICIENT
        assert policy.next_move(state).action is decide(
            Verdict.INSUFFICIENT, attempts_left
        )

    def test_it_asks_for_a_retry_and_leaves_the_settling_to_the_driver(self):
        # The corrective rewriter runs after the decision and is paid for even
        # when it returns nothing, so the driver settles rather than the policy.
        policy = CorrectivePolicy(GradingThresholds(0.6, 0.3))
        state = state_for(0.1, attempts_left=1, rewrite_available=False)
        assert policy.next_move(state).action is Action.REWRITE
        settled = policy.next_move(replace(state, attempts_left=0))
        assert settled.action is Action.ABSTAIN

    def test_with_no_thresholds_it_follows_the_thresholds_of_the_grade(self):
        policy = CorrectivePolicy()
        strict = grade_at(GradingThresholds(0.9, 0.8), 0.5)
        assert policy.verdict_for(state_for(0.5, grade=strict)) is (
            Verdict.INSUFFICIENT
        )
        loose = grade_at(GradingThresholds(0.4, 0.1), 0.5)
        assert policy.verdict_for(state_for(0.5, grade=loose)) is Verdict.SUFFICIENT

    def test_with_no_thresholds_and_no_grade_it_falls_back_to_the_defaults(self):
        policy = CorrectivePolicy()
        assert policy.verdict_for(state_for(0.95)) is Verdict.SUFFICIENT
        assert policy.verdict_for(state_for(0.1)) is Verdict.AMBIGUOUS

    def test_it_is_replayable(self):
        assert CorrectivePolicy().replayable is True


class TestSingleShotPolicy:
    def test_it_always_answers(self):
        policy = SingleShotPolicy()
        for score in (0.0, 0.5, 1.0):
            for attempts_left in (0, 1):
                state = state_for(score, attempts_left=attempts_left)
                assert policy.next_move(state).action is Action.ANSWER

    def test_it_answers_even_with_nothing_retrieved(self):
        state = state_for(0.0, had_evidence=False)
        assert SingleShotPolicy().next_move(state).action is Action.ANSWER

    def test_it_is_replayable(self):
        assert SingleShotPolicy().replayable is True


class TestLoopState:
    def test_the_question_is_the_one_the_plan_carries(self):
        assert state_for(0.5).question == "who signs"

    def test_the_step_can_differ_from_the_question_as_asked(self):
        plan = QueryPlan(question="who signs")
        state = LoopState(
            plan=plan,
            step=RetrievalStep(
                question="signatories", strategy="template", parent="who signs"
            ),
            score=0.5,
            had_evidence=True,
            attempts_left=1,
            rewrite_available=False,
        )
        assert state.question == "who signs"
        assert state.step.question == "signatories"

    def test_reasoning_accumulates_for_an_interleaved_policy_to_read(self):
        state = state_for(0.5, notes=("first hop named Acme",))
        assert state.notes == ("first hop named Acme",)

    def test_a_move_can_name_the_retrieval_to_run_next(self):
        move = LoopDecision(
            Action.REWRITE, step=RetrievalStep(question="Acme's counterparty")
        )
        assert move.step is not None
        assert move.step.question == "Acme's counterparty"


class TestPromptSwitch:
    def test_a_fixed_prompt_returns_its_template_whatever_was_retrieved(self):
        prompt = FixedPromptSelector("answer {query} from {sources}")
        plan = QueryPlan(question="who signs")
        assert prompt.get_template(plan, []) == "answer {query} from {sources}"
        assert prompt.get_template(plan, [passage("dense")]) == (
            "answer {query} from {sources}"
        )

    def test_a_fixed_prompt_defaults_to_the_shipped_template(self):
        plan = QueryPlan(question="who signs")
        assert FixedPromptSelector().get_template(plan, []) == GENERATION_TEMPLATE

    def test_a_methods_own_template_wins_when_it_owns_the_result_set(self):
        prompt = RetrieverPromptSelector("default", {"lightrag": "graph template"})
        plan = QueryPlan(question="who signs")
        evidence = [passage("lightrag", 0), passage("lightrag", 1)]
        assert prompt.get_template(plan, evidence) == "graph template"

    def test_a_pooled_result_set_falls_back_to_the_project_default(self):
        # A decomposition pools several retrievals, so no single method owns the
        # passages and keying on one of them would answer in another's format.
        prompt = RetrieverPromptSelector("default", {"lightrag": "graph template"})
        plan = QueryPlan(question="who signs")
        evidence = [passage("lightrag"), passage("dense", 1)]
        assert prompt.get_template(plan, evidence) == "default"

    def test_an_unlabelled_or_empty_result_set_falls_back(self):
        prompt = RetrieverPromptSelector("default", {"lightrag": "graph template"})
        plan = QueryPlan(question="who signs")
        assert prompt.get_template(plan, []) == "default"
        assert prompt.get_template(plan, [passage("")]) == "default"

    def test_a_method_with_no_template_registered_falls_back(self):
        prompt = RetrieverPromptSelector("default", {"lightrag": "graph template"})
        plan = QueryPlan(question="who signs")
        assert prompt.get_template(plan, [passage("bm25")]) == "default"


def test_the_retired_answering_names_still_import():
    from aieng.advanced_rag.pipeline import answering

    assert answering.Move is LoopDecision
    assert answering.Step is RetrievalStep
    assert answering.PromptSource is PromptSelector
    assert answering.FixedPrompt is FixedPromptSelector
    assert answering.MethodPrompt is RetrieverPromptSelector
    assert answering.NoAnalysis is PassthroughAnalysis


def test_the_retired_selector_method_still_answers():
    selector = FixedPromptSelector("TEMPLATE {query} {sources}")
    plan = QueryPlan(question="who signs")
    assert selector.template_for(plan, ()) == selector.get_template(plan, ())
