"""Tests for answer judging at a fixed evidence budget."""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.evaluation import (
    COVERAGE_TABLE_HEADER,
    AnswerJudgement,
    AnswerReport,
    CoverageReport,
    answer_with_budget,
    judge_answer,
    judge_coverage,
    parse_coverage_verdicts,
    parse_judgement,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance


def make_evidence(count: int) -> list[Evidence]:
    return [
        Evidence(
            text=f"passage {i}",
            provenance=Provenance(doc_id=f"doc{i}", chunk_id=f"doc{i}::0"),
            score=1.0 - i / 100,
            retriever="test",
        )
        for i in range(count)
    ]


def judgement(**overrides) -> AnswerJudgement:
    base = {
        "query_id": "q",
        "correct": True,
        "grounded": True,
        "abstained": False,
        "reason": "",
    }
    base.update(overrides)
    return AnswerJudgement(**base)


def test_parse_tolerates_prose_and_fences_around_the_json():
    raw = 'Here you go:\n```json\n{"correct": true, "grounded": false, "abstained": false}\n```'
    parsed = parse_judgement("q1", raw)
    assert parsed.correct and not parsed.grounded


def test_missing_fields_read_as_false_rather_than_raising():
    parsed = parse_judgement("q1", json.dumps({"correct": True}))
    assert parsed.correct and not parsed.grounded and not parsed.abstained


def test_a_response_with_no_json_is_an_error():
    with pytest.raises(ValueError):
        parse_judgement("q1", "I cannot grade this.")


def test_reason_is_truncated_so_one_verbose_judge_cannot_bloat_a_report():
    parsed = parse_judgement("q1", json.dumps({"correct": True, "reason": "x" * 1000}))
    assert len(parsed.reason) == 300


def test_abstentions_are_excluded_from_the_hallucination_rate():
    report = AnswerReport(
        method="m",
        evidence_budget=5,
        answered=1,
        judgements=[
            judgement(grounded=False, abstained=True),
            judgement(grounded=True, abstained=False),
        ],
    )
    # One abstention, one grounded attempt: no hallucination.
    assert report.hallucination == 0.0
    assert report.abstention == 0.5


def test_hallucination_counts_ungrounded_attempts():
    report = AnswerReport(
        method="m",
        evidence_budget=5,
        answered=2,
        judgements=[judgement(grounded=False), judgement(grounded=True)],
    )
    assert report.hallucination == 0.5


def test_empty_report_does_not_divide_by_zero():
    report = AnswerReport(method="m", evidence_budget=5, answered=0)
    assert report.correctness == 0.0 and report.hallucination == 0.0


def test_budget_truncates_evidence_and_reports_what_was_used():
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "an answer"

    _, used = answer_with_budget("q?", make_evidence(10), complete, budget=3)
    assert len(used) == 3
    assert "passage 2" in prompts[0] and "passage 3" not in prompts[0]


def test_a_method_returning_less_than_the_budget_is_not_padded():
    _, used = answer_with_budget("q?", make_evidence(2), lambda p: "a", budget=5)
    assert len(used) == 2


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        answer_with_budget("q?", make_evidence(2), lambda p: "a", budget=0)


def test_judge_prompt_carries_question_reference_evidence_and_answer():
    seen = {}

    def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps({"correct": True, "grounded": True, "abstained": False})

    judge_answer(
        query_id="q1",
        question="what colour is the sky",
        answer_text="blue",
        reference="blue",
        evidence=make_evidence(2),
        complete=complete,
    )
    prompt = seen["prompt"]
    assert "what colour is the sky" in prompt
    assert "blue" in prompt
    assert "passage 0" in prompt


def test_judge_reports_empty_evidence_explicitly():
    seen = {}

    def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps({"correct": False, "grounded": False, "abstained": True})

    judge_answer("q1", "q?", "not stated", "something", [], complete)
    assert "(no evidence retrieved)" in seen["prompt"]


def test_table_row_matches_the_header_column_count():
    from aieng.advanced_rag.evaluation import TABLE_HEADER

    report = AnswerReport("m", 5, 1, [judgement()])
    assert report.table_row().count("|") == TABLE_HEADER.splitlines()[0].count("|")


class TestContributionCoverage:
    """The fan-out judge, which asks what each document contributes.

    The judge is a model, so what is tested here is everything around it: that a
    document it forgot to mention is read as not covered, that malformed JSON
    does not lose the run, and that the two rates say different things.
    """

    def test_a_full_verdict_is_read_back(self):
        raw = json.dumps(
            {
                "verdicts": [
                    {"document": "d1", "covered": True},
                    {"document": "d2", "covered": False},
                ]
            }
        )
        assert parse_coverage_verdicts(raw, ["d1", "d2"]) == {"d1": True, "d2": False}

    def test_a_document_the_judge_omitted_is_not_covered(self):
        raw = json.dumps({"verdicts": [{"document": "d1", "covered": True}]})
        assert parse_coverage_verdicts(raw, ["d1", "d2"]) == {"d1": True, "d2": False}

    def test_a_document_the_judge_invented_is_ignored(self):
        raw = json.dumps(
            {"verdicts": [{"document": "not-asked-about", "covered": True}]}
        )
        assert parse_coverage_verdicts(raw, ["d1"]) == {"d1": False}

    @pytest.mark.parametrize("raw", ["", "no json here", "{not json}"])
    def test_an_unparseable_reply_is_read_as_nothing_covered(self, raw):
        assert parse_coverage_verdicts(raw, ["d1"]) == {"d1": False}

    def test_prose_around_the_object_is_tolerated(self):
        raw = (
            'Sure! {"verdicts": [{"document": "d1", "covered": true}]} Hope that helps.'
        )
        assert parse_coverage_verdicts(raw, ["d1"]) == {"d1": True}

    def test_the_prompt_carries_every_contribution(self):
        seen = {}

        def complete(prompt: str) -> str:
            seen["prompt"] = prompt
            return json.dumps({"verdicts": []})

        judge_coverage(
            "how do the two hosts differ",
            {"d1": "states the kernel version", "d2": "states the mount options"},
            "an answer",
            complete,
        )
        assert "states the kernel version" in seen["prompt"]
        assert "states the mount options" in seen["prompt"]
        assert "an answer" in seen["prompt"]

    def test_a_contribution_that_was_not_recorded_is_marked_as_such(self):
        seen = {}

        def complete(prompt: str) -> str:
            seen["prompt"] = prompt
            return json.dumps({"verdicts": []})

        judge_coverage("q", {"d1": ""}, "an answer", complete)
        assert "(not recorded)" in seen["prompt"]

    def test_a_long_answer_is_truncated_before_the_judge_sees_it(self):
        # A global-search report can run to thousands of words, and the judge
        # is billed per token.
        seen = {}

        def complete(prompt: str) -> str:
            seen["prompt"] = prompt
            return json.dumps({"verdicts": []})

        judge_coverage("q", {"d1": "x"}, "y" * 50, complete, max_answer_chars=10)
        assert "y" * 10 in seen["prompt"]
        assert "y" * 11 not in seen["prompt"]

    def test_mean_coverage_is_the_share_of_contributions_stated(self):
        report = CoverageReport(
            "dense control",
            [{"d1": True, "d2": False}, {"d1": True, "d2": True}],
        )
        assert report.questions == 2
        assert report.mean_coverage == pytest.approx(0.75)

    def test_complete_coverage_is_the_harsher_reading(self):
        # Three of four contributions is still an incomplete fan-out answer.
        report = CoverageReport(
            "dense control",
            [{"d1": True, "d2": True, "d3": True, "d4": False}],
        )
        assert report.mean_coverage == pytest.approx(0.75)
        assert report.complete_coverage_rate == 0.0

    def test_an_empty_report_does_not_divide_by_zero(self):
        report = CoverageReport("dense control", [])
        assert report.mean_coverage == 0.0
        assert report.complete_coverage_rate == 0.0

    def test_the_table_row_matches_the_header_column_count(self):
        report = CoverageReport("dense control", [{"d1": True}])
        header = COVERAGE_TABLE_HEADER.splitlines()[0]
        assert report.table_row().count("|") == header.count("|")
        assert report.to_dict()["complete_coverage_rate"] == 1.0
