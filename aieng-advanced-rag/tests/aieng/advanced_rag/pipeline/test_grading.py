"""Tests for evidence grading and the action it drives."""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.pipeline.grading import (
    Action,
    ChunkGrade,
    GradingThresholds,
    HeuristicGrader,
    LLMGrader,
    Verdict,
    content_terms,
    decide,
    first_json_object,
    no_evidence_grade,
    pooled_score,
)


class Recorder:
    def __init__(self, *responses: str) -> None:
        self.queue = list(responses)
        self.last = ""
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.queue:
            self.last = self.queue.pop(0)
        return self.last


def grade_json(score: float, reason: str = "states the answer") -> str:
    return json.dumps({"score": score, "reason": reason})


@pytest.mark.parametrize(
    ("verdict", "attempts_left", "expected"),
    [
        (Verdict.SUFFICIENT, 1, Action.ANSWER),
        (Verdict.SUFFICIENT, 0, Action.ANSWER),
        (Verdict.AMBIGUOUS, 1, Action.REWRITE),
        (Verdict.AMBIGUOUS, 0, Action.ANSWER),
        (Verdict.INSUFFICIENT, 1, Action.REWRITE),
        (Verdict.INSUFFICIENT, 0, Action.ABSTAIN),
    ],
)
def test_the_decision_table_is_the_whole_control_flow(verdict, attempts_left, expected):
    assert decide(verdict, attempts_left) is expected


def test_thresholds_must_be_ordered_and_in_range():
    with pytest.raises(ValueError):
        GradingThresholds(sufficient=0.2, insufficient=0.8)
    with pytest.raises(ValueError):
        GradingThresholds(sufficient=1.5)


def test_the_shipped_thresholds_leave_abstention_to_the_generator():
    # Measured, not chosen. The sweep is in
    # implementations/regulatory_policy_qa/THRESHOLDS.md. Pinned here so that
    # moving either number is a deliberate act with a measurement behind it.
    thresholds = GradingThresholds()
    assert (thresholds.sufficient, thresholds.insufficient) == (0.6, 0.0)
    assert thresholds.verdict(0.0) is Verdict.AMBIGUOUS
    assert thresholds.verdict(0.6) is Verdict.SUFFICIENT


def test_an_empty_result_set_still_abstains_at_the_shipped_thresholds():
    # `insufficient` at 0.0 means no score reaches the verdict, so the only
    # route to it is retrieval coming back with nothing. That route has to stay
    # open or the pipeline would answer from an empty source list.
    grade = no_evidence_grade("q1", "who signs", GradingThresholds())
    assert grade.verdict is Verdict.INSUFFICIENT
    assert decide(grade.verdict, 0) is Action.ABSTAIN


def test_a_score_on_the_boundary_is_sufficient():
    thresholds = GradingThresholds(sufficient=0.6, insufficient=0.3)
    assert thresholds.verdict(0.6) is Verdict.SUFFICIENT
    assert thresholds.verdict(0.59) is Verdict.AMBIGUOUS
    assert thresholds.verdict(0.3) is Verdict.AMBIGUOUS
    assert thresholds.verdict(0.29) is Verdict.INSUFFICIENT


def test_the_pool_is_the_maximum_not_the_mean():
    grades = [ChunkGrade("a", 1.0), ChunkGrade("b", 0.0), ChunkGrade("c", 0.0)]
    assert pooled_score(grades) == 1.0
    assert pooled_score([]) == 0.0


def test_no_evidence_grades_insufficient_without_calling_the_model():
    model = Recorder(grade_json(1.0))
    grade = LLMGrader(model).grade("who signs", [], query_id="q1")
    assert grade.verdict is Verdict.INSUFFICIENT
    assert grade.score == 0.0
    assert model.prompts == []
    assert "no passages" in grade.reason


def test_the_heuristic_grader_prefers_the_on_topic_passage(make_evidence):
    evidence = make_evidence(
        [
            "This Agreement is governed by the laws of the Province of Ontario.",
            "The parties agree to meet quarterly to review delivery schedules.",
        ]
    )
    grade = HeuristicGrader().grade("Which law governs the agreement?", evidence)
    assert grade.chunk_grades[0].score > grade.chunk_grades[1].score


def test_crude_stemming_matches_governing_law_to_governed_by_the_laws():
    terms = content_terms("Which law governs the agreement?")
    passage = content_terms("This Agreement is governed by the laws of Ontario.")
    assert terms & passage >= {"law", "govern", "agreement"}


def test_the_heuristic_grader_reports_which_passage_carried_the_score(
    make_evidence,
):
    evidence = make_evidence(["governing law of Ontario", "unrelated text"])
    grade = HeuristicGrader().grade("governing law", evidence, query_id="q1")
    assert grade.query_id == "q1"
    assert evidence[0].chunk_id in grade.reason


def test_the_llm_grader_makes_one_call_per_graded_passage(make_evidence):
    model = Recorder(grade_json(0.9), grade_json(0.1), grade_json(0.1))
    evidence = make_evidence(["a", "b", "c", "d"])
    grade = LLMGrader(model, max_chunks=2).grade("q", evidence)
    assert len(model.prompts) == 2
    assert [g.reason for g in grade.chunk_grades[2:]] == ["not graded", "not graded"]
    assert grade.score == 0.9


def test_an_ungraded_passage_cannot_raise_the_pooled_score(make_evidence):
    # Thresholds are stated rather than defaulted: the shipped `insufficient`
    # is 0.0, under which no score is below it and this verdict would be
    # ambiguous. What is being checked here is the pooling, so the boundary it
    # is read against has to be fixed.
    model = Recorder(grade_json(0.0))
    grade = LLMGrader(
        model, max_chunks=1, thresholds=GradingThresholds(0.6, 0.3)
    ).grade("q", make_evidence(["a", "b"]))
    assert grade.score == 0.0
    assert grade.verdict is Verdict.INSUFFICIENT


def test_an_empty_completion_scores_zero_rather_than_raising(make_evidence):
    # The reasoning trap: content=None becomes "", and a truncated grade is not
    # evidence that the passage supports the question.
    grade = LLMGrader(Recorder("")).grade("q", make_evidence(["a"]))
    assert grade.score == 0.0
    assert grade.chunk_grades[0].reason == "unreadable grade"


def test_scores_are_clamped_and_non_numeric_scores_are_zero(make_evidence):
    model = Recorder(grade_json(7.0), '{"score": "high"}')
    evidence = make_evidence(["a", "b"])
    grade = LLMGrader(model, max_chunks=2).grade("q", evidence)
    assert grade.chunk_grades[0].score == 1.0
    assert grade.chunk_grades[1].score == 0.0


def test_the_grade_records_the_thresholds_it_was_read_against(make_evidence):
    thresholds = GradingThresholds(sufficient=0.8, insufficient=0.4)
    grade = LLMGrader(Recorder(grade_json(0.5)), thresholds=thresholds).grade(
        "q", make_evidence(["a"])
    )
    assert grade.verdict is Verdict.AMBIGUOUS
    assert grade.to_dict()["sufficient"] == 0.8
    assert grade.supporting == ()


def test_the_grade_serialises_for_a_report(make_evidence):
    grade = HeuristicGrader().grade("governing law", make_evidence(["governing law"]))
    assert json.loads(json.dumps(grade.to_dict()))["verdict"] == "sufficient"


def test_at_least_one_passage_must_be_graded():
    with pytest.raises(ValueError):
        LLMGrader(Recorder(), max_chunks=0)


def test_json_is_recovered_from_prose_and_code_fences():
    assert first_json_object('here:\n```json\n{"score": 1}\n```')["score"] == 1
    assert first_json_object("no object here") is None
    assert first_json_object("[1, 2]") is None
    assert first_json_object("") is None
    assert first_json_object("{not json}") is None
