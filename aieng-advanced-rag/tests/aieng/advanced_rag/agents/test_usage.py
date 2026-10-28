"""Reading an agent event's cost, and handing it to the shared meter.

The events here are the shape ADK yields rather than ADK objects, so this runs
with nothing installed. What it pins is the translation: ADK names its counts
``prompt_token_count`` and ``candidates_token_count``, a tool call is a model
call and not an abstention, and a tool result is not a model call at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aieng.advanced_rag.agents.usage import (
    read_event,
    report_event,
    sum_usage,
    usage_of_event,
)
from aieng.advanced_rag.metering import CallUsage, UsageMeter


@dataclass
class FakeUsage:
    prompt_token_count: int | None = 41
    candidates_token_count: int | None = 7
    total_token_count: int | None = 48
    thoughts_token_count: int | None = None


@dataclass
class FakePart:
    text: str | None = None
    function_call: Any = None
    thought: bool | None = None


@dataclass
class FakeCall:
    name: str


@dataclass
class FakeContent:
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeEvent:
    content: FakeContent | None = None
    usage_metadata: FakeUsage | None = None


def answer_event(text: str = "Toronto has 2.8 million people.") -> FakeEvent:
    return FakeEvent(
        content=FakeContent([FakePart(text=text)]), usage_metadata=FakeUsage()
    )


def tool_call_event(name: str = "search_corpus") -> FakeEvent:
    return FakeEvent(
        content=FakeContent([FakePart(function_call=FakeCall(name))]),
        usage_metadata=FakeUsage(),
    )


def test_adk_token_names_are_read_onto_the_meters_own_names():
    usage = usage_of_event(answer_event())
    assert usage == CallUsage(
        prompt_tokens=41, completion_tokens=7, total_tokens=48, reported=True
    )


def test_a_missing_total_is_the_sum_of_the_other_two():
    event = answer_event()
    event.usage_metadata = FakeUsage(total_token_count=None)
    usage = usage_of_event(event)
    assert usage is not None
    assert usage.total_tokens == 48


def test_a_usage_block_with_no_numbers_reports_nothing_rather_than_zero():
    event = answer_event()
    event.usage_metadata = FakeUsage(None, None, None)
    assert usage_of_event(event) == CallUsage()


def test_an_event_without_usage_is_not_a_model_call():
    assert (
        usage_of_event(FakeEvent(content=FakeContent([FakePart(text="12 C")]))) is None
    )
    assert read_event(FakeEvent()) is None


def test_a_tool_call_is_a_model_call_and_not_an_abstention():
    facts = read_event(tool_call_event())
    assert facts is not None
    assert facts.tool_calls == ("search_corpus",)
    assert facts.empty is False
    assert facts.text == ""


def test_a_response_with_neither_text_nor_a_tool_call_is_empty():
    facts = read_event(FakeEvent(content=None, usage_metadata=FakeUsage()))
    assert facts is not None
    assert facts.empty is True


def test_thinking_alone_is_not_an_answer():
    event = FakeEvent(
        content=FakeContent([FakePart(text="let me think", thought=True)]),
        usage_metadata=FakeUsage(),
    )
    facts = read_event(event)
    assert facts is not None
    assert facts.text == ""
    assert facts.empty is True


def test_totals_are_whole_only_when_every_call_reported():
    reported = CallUsage(10, 2, 12, reported=True)
    assert sum_usage([reported, reported]) == CallUsage(20, 4, 24, reported=True)
    assert sum_usage([reported, CallUsage()]).reported is False
    assert sum_usage([]).reported is False


def test_a_total_summed_from_records_built_without_one_is_not_short():
    hand_built = [CallUsage(10, 2, reported=True), CallUsage(7, 3, reported=True)]
    summed = sum_usage(hand_built)
    assert summed.total_tokens == 22
    assert summed.reported is True


def test_a_reported_event_lands_on_the_question_bound_here():
    meter = UsageMeter()
    with meter.question("q1"):
        report_event(tool_call_event(), 0.25)
        report_event(answer_event(), 0.5)
        report_event(FakeEvent(content=FakeContent([FakePart(text="12 C")])), 0.1)
    cost = meter.per_question()["q1"]
    assert cost.calls == 2
    assert cost.tool_calls == 1
    assert cost.total_tokens == 96
    assert cost.service_seconds == 0.75
    assert cost.empty_responses == 0
    assert cost.tokens_complete is True


def test_a_meter_handed_over_is_charged_instead_of_the_context():
    meter = UsageMeter()
    charged = meter.for_question("q2")
    report_event(answer_event(), 0.5, charge_to=charged)
    assert meter.per_question()["q2"].calls == 1
    assert meter.unattributed().calls == 0


def test_a_call_outside_any_question_leaves_a_meter_untouched():
    meter = UsageMeter()
    assert report_event(answer_event(), 0.5) is not None
    assert meter.totals().calls == 0


def test_the_event_facts_name_still_imports():
    from aieng.advanced_rag.agents import usage

    assert usage.EventFacts is usage.ParsedEvent
