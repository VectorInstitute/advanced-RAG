"""Tests for query rewriting."""

from __future__ import annotations

import re

import pytest

from aieng.advanced_rag.pipeline.rewrite import (
    LLMRewriter,
    Rewrite,
    RewritePlan,
    TemplateRewriter,
    parse_variants,
)


CUAD_QUESTION = (
    'Highlight the parts (if any) of this contract related to "Governing Law" '
    "that should be reviewed by a lawyer. Details: Which state or country law "
    "governs the interpretation of the contract?"
)


class Recorder:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_the_cuad_template_is_stripped_and_the_category_survives():
    plan = TemplateRewriter().rewrite(CUAD_QUESTION)
    rewritten = plan.questions(include_original=False)[0]
    assert "Highlight the parts" not in rewritten
    assert "reviewed by a lawyer" not in rewritten
    assert "Details:" not in rewritten
    assert "Governing Law" in rewritten
    assert "governs the interpretation" in rewritten


def test_a_question_with_no_boilerplate_produces_nothing_to_try():
    plan = TemplateRewriter().rewrite("Who signed the agreement?")
    assert plan.is_empty
    assert plan.questions(include_original=False) == ()


def test_a_rewrite_that_removes_everything_is_discarded():
    rewriter = TemplateRewriter(patterns=[re.compile(r".*")])
    assert rewriter.rewrite("Who signed?").is_empty


def test_stripping_nothing_and_stripping_everything_are_told_apart():
    # Both return an empty plan, and they are opposite findings: the patterns
    # do not match this corpus, or they match so much that nothing is left.
    # `is_empty` reads the same for both, so the outcome is what distinguishes
    # them in a trace.
    untouched = TemplateRewriter().rewrite("Who signed the agreement?")
    emptied = TemplateRewriter(patterns=[re.compile(r".*")]).rewrite("Who signed?")

    assert untouched.outcome == "no-change"
    assert emptied.outcome == "emptied"
    assert untouched.is_empty is emptied.is_empty is True


def test_a_stripped_question_reports_that_it_was_rewritten():
    assert TemplateRewriter().rewrite(CUAD_QUESTION).outcome == "rewritten"


def test_questions_keeps_the_original_first_and_de_duplicates():
    plan = RewritePlan(
        original="who signs",
        rewrites=(
            Rewrite("signatory", "llm"),
            Rewrite("who signs", "llm"),
            Rewrite("signatory", "llm"),
        ),
    )
    assert plan.questions() == ("who signs", "signatory")
    assert plan.questions(include_original=False) == ("signatory", "who signs")


def test_numbering_bullets_and_quotes_are_stripped_from_variants():
    raw = '1. governing law clause\n- "choice of law"\n* jurisdiction'
    assert parse_variants(raw) == [
        "governing law clause",
        "choice of law",
        "jurisdiction",
    ]


def test_a_preamble_line_and_an_echo_of_the_original_are_dropped():
    raw = "Here are two queries:\nWho signed?\ncontract signatory name"
    assert parse_variants(raw, original="who signed?") == ["contract signatory name"]


def test_an_empty_completion_yields_no_variants():
    # This is the content=None case: ChatCompleter turns it into "".
    assert parse_variants("") == []
    assert LLMRewriter(Recorder("")).rewrite("who signs").is_empty


def test_an_unusable_completion_and_a_rejected_one_are_told_apart():
    # A model that returned only the question back has nothing to add, and the
    # prompt is working. A truncated or chatty completion is a prompt fault.
    # Both leave no variants, so the outcome is what names which happened.
    nothing_parseable = LLMRewriter(Recorder("Here are the queries:\n")).rewrite(
        "who signs"
    )
    every_candidate_dropped = LLMRewriter(Recorder("who signs\nWHO SIGNS")).rewrite(
        "who signs"
    )

    assert nothing_parseable.outcome == "no-change"
    assert every_candidate_dropped.outcome == "all-rejected"
    assert nothing_parseable.is_empty is every_candidate_dropped.is_empty is True


def test_a_parsed_rewrite_reports_that_it_was_rewritten():
    plan = LLMRewriter(Recorder("choice of law clause")).rewrite("who signs")
    assert plan.outcome == "rewritten"


def test_variants_are_capped_at_the_requested_count():
    raw = "one query\ntwo query\nthree query\nfour query"
    assert len(parse_variants(raw, limit=2)) == 2


def test_the_graders_reason_reaches_the_rewrite_prompt():
    model = Recorder("assignment consent requirement")
    plan = LLMRewriter(model).rewrite(
        "who must consent", feedback="passages describe termination, not assignment"
    )
    assert "passages describe termination" in model.prompts[0]
    assert plan.rewrites[0].reason.startswith("rewritten after:")


def test_without_feedback_the_prompt_carries_no_feedback_block():
    model = Recorder("assignment consent requirement")
    LLMRewriter(model).rewrite("who must consent")
    assert "grader said" not in model.prompts[0]


def test_at_least_one_variant_must_be_requested():
    with pytest.raises(ValueError):
        LLMRewriter(Recorder(), variants=0)
