"""Tests for the shared cost meter.

Covers what the older meters could not do: attribution to a question rather than
to a phase label, figures that survive eight-way concurrency, the two seconds
kept apart, a tool call counted as a model call, and a streamed response whose
usage never arrives. Nothing here touches a network or a model.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from aieng.advanced_rag.metering import (
    METERED_PROVIDER,
    CallUsage,
    MeteredClient,
    MeteredCompleter,
    MissingUsageWarning,
    UsageMeter,
    _CallRecorder,
    _metered_completion_class,
    current_question,
    default_meter,
    meter_client,
    read_completion,
    register_metered_completion,
    report_cache_hit,
    report_call,
    reset_default_meter,
    reset_unmetered_calls,
    unmetered_calls,
    usage_of,
    using,
)
from aieng.advanced_rag.pipeline.generate import ChatCompleter
from aieng.advanced_rag.reasoning import THINKING_OFF


WORKERS = 8


def chat_response(
    content: str | None = "ok",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    tool_calls: int = 0,
    with_usage: bool = True,
    with_choices: bool = True,
) -> Any:
    """Build one chat completion shaped like the proxy's, tool calls included."""
    calls = [
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(
                name="get_temperature", arguments='{"city": "Toronto"}'
            ),
        )
        for index in range(tool_calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    usage = (
        SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            # The proxy returns both detail blocks as null.
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        if with_usage
        else None
    )
    choices = (
        [
            SimpleNamespace(
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ]
        if with_choices
        else []
    )
    return SimpleNamespace(choices=choices, usage=usage)


def text_chunk(text: str) -> Any:
    """One streamed chunk carrying content."""
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def tool_chunk(index: int = 0, arguments: str = '{"city": ') -> Any:
    """One streamed chunk carrying part of a tool call."""
    call = SimpleNamespace(
        index=index,
        function=SimpleNamespace(name="get_temperature", arguments=arguments),
    )
    delta = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def usage_chunk(prompt_tokens: int = 282, completion_tokens: int = 26) -> Any:
    """Build the final streamed chunk, the only one that carries usage."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeCompletions:
    """Replays a queue of responses, recording every request it was sent."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Return the next queued response, or a stream over a queued list."""
        self.requests.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, list):
            return iter(result)
        return result


class AsyncFakeCompletions(FakeCompletions):
    """The same queue, behind a coroutine, as ``AsyncOpenAI`` presents it."""

    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        """Return the next queued response, streaming a queued list."""
        self.requests.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, list):

            async def stream() -> Any:
                for chunk in result:
                    yield chunk

            return stream()
        return result


def fake_client(
    responses: list[Any], asynchronous: bool = False
) -> tuple[Any, FakeCompletions]:
    """Build a client exposing ``chat.completions.create`` and nothing else."""
    completions: FakeCompletions = (
        AsyncFakeCompletions(responses) if asynchronous else FakeCompletions(responses)
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class StubCompleter:
    """A completer that reports one call the way ``ChatCompleter`` does."""

    def __init__(
        self,
        prompt_tokens: int = 10,
        completion_tokens: int = 2,
        sleep: float = 0.0,
        text: str = "answer",
        tool_calls: int = 0,
        with_usage: bool = True,
        attempts: int = 1,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.sleep = sleep
        self.text = text
        self.tool_calls = tool_calls
        self.with_usage = with_usage
        self.attempts = attempts

    def __call__(self, prompt: str) -> str:
        """Report one call per attempt, then return the text."""
        for _ in range(self.attempts):
            started = time.perf_counter()
            if self.sleep:
                time.sleep(self.sleep)
            usage = (
                CallUsage(
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.prompt_tokens + self.completion_tokens,
                    reported=True,
                )
                if self.with_usage
                else CallUsage()
            )
            report_call(
                usage,
                time.perf_counter() - started,
                tool_calls=self.tool_calls,
                empty=not self.text and not self.tool_calls,
            )
        return self.text


def test_usage_comes_from_the_three_totals_the_proxy_returns():
    usage = usage_of(chat_response(prompt_tokens=282, completion_tokens=26))
    assert usage.prompt_tokens == 282
    assert usage.completion_tokens == 26
    assert usage.total_tokens == 308
    assert usage.reported


def test_a_response_with_no_usage_block_is_not_read_as_zero_tokens():
    usage = usage_of(chat_response(with_usage=False))
    assert usage.total_tokens == 0
    assert not usage.reported


def test_a_dict_shaped_usage_is_read_the_same_way():
    usage = usage_of(
        {"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        5,
        1,
        6,
    )
    assert usage.reported


def test_a_total_the_response_omits_falls_back_to_the_sum():
    usage = usage_of(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6))
    )
    assert usage.total_tokens == 10


def test_a_hand_built_record_falls_back_to_the_sum_as_well():
    assert CallUsage(10, 2).total_tokens == 12
    assert CallUsage(prompt_tokens=10, completion_tokens=2).total_tokens == 12
    assert CallUsage().total_tokens == 0


def test_a_provider_total_that_disagrees_with_the_sum_is_kept():
    assert CallUsage(10, 2, 15).total_tokens == 15


def test_a_meter_fed_hand_built_records_reports_the_whole_total():
    meter = UsageMeter()
    with meter.question("q1") as charged:
        charged.record(CallUsage(10, 2, reported=True), 0.25)
    cost = meter.per_question()["q1"]
    assert cost.total_tokens == 12
    assert cost.tokens_complete


def test_a_tool_call_is_not_an_empty_response():
    facts = read_completion(chat_response(content=None, tool_calls=1))
    assert facts.tool_calls == 1
    assert not facts.empty
    assert facts.text == ""
    assert facts.finish_reason == "tool_calls"


def test_null_content_with_no_tool_call_is_an_empty_response():
    facts = read_completion(chat_response(content=None))
    assert facts.empty
    assert facts.tool_calls == 0


def test_a_response_with_no_choices_is_an_empty_response():
    assert read_completion(chat_response(with_choices=False)).empty


def test_a_question_keeps_its_own_calls_tokens_and_seconds():
    meter = UsageMeter()
    with meter.question("q1") as charged:
        charged.record(CallUsage(100, 20, 120, reported=True), 1.5)
        charged.record(CallUsage(50, 10, 60, reported=True), 0.5)
        charged.record_cache_hit()

    cost = meter.per_question()["q1"]
    assert cost.calls == 2
    assert cost.total_tokens == 180
    assert cost.cache_hits == 1
    assert cost.service_seconds == pytest.approx(2.0)
    assert cost.tokens_complete
    assert meter.totals().calls == 2
    assert meter.totals().questions == 1


def test_calls_made_with_no_question_open_land_in_the_unattributed_bucket():
    meter = UsageMeter()
    meter.record(CallUsage(3, 1, 4, reported=True), 0.1)
    assert meter.per_question() == {}
    assert meter.unattributed().calls == 1
    assert meter.totals().calls == 1


def test_a_report_no_meter_took_is_counted_rather_than_dropped():
    reset_unmetered_calls()
    report_call(CallUsage(10, 2, reported=True), 0.1)
    assert unmetered_calls() == 1

    meter = UsageMeter()
    with meter.question("q1"):
        report_call(CallUsage(10, 2, reported=True), 0.1)
    assert meter.totals().calls == 1
    # A metered call is not a dropped one, so the count has not moved.
    assert unmetered_calls() == 1
    assert reset_unmetered_calls() == 1
    assert unmetered_calls() == 0


def test_a_second_meter_does_not_inherit_another_meters_question():
    first = UsageMeter()
    second = UsageMeter()
    with first.question("q1"):
        second.record(CallUsage(3, 1, 4, reported=True), 0.1)

    assert first.per_question()["q1"].calls == 0
    assert second.per_question() == {}
    assert second.unattributed().calls == 1


def test_eight_questions_answered_at_once_do_not_cross_contaminate():
    meter = UsageMeter()
    barrier = threading.Barrier(WORKERS)

    def answer(index: int) -> None:
        with meter.question(f"q{index}"):
            complete = MeteredCompleter(
                StubCompleter(
                    prompt_tokens=100 * (index + 1), completion_tokens=index + 1
                ),
                meter,
            )
            barrier.wait()
            for _ in range(3):
                complete("prompt")
                time.sleep(0.001)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(answer, range(WORKERS)))

    per_question = meter.per_question()
    assert len(per_question) == WORKERS
    for index in range(WORKERS):
        cost = per_question[f"q{index}"]
        assert cost.calls == 3
        assert cost.prompt_tokens == 300 * (index + 1)
        assert cost.completion_tokens == 3 * (index + 1)
    assert meter.totals().calls == 3 * WORKERS
    assert meter.unattributed().calls == 0


def test_eight_questions_awaited_at_once_do_not_cross_contaminate():
    meter = UsageMeter()

    async def answer(index: int) -> None:
        with meter.question(f"q{index}"):
            complete = MeteredCompleter(
                StubCompleter(
                    prompt_tokens=100 * (index + 1), completion_tokens=index + 1
                ),
                meter,
            )
            for _ in range(3):
                complete("prompt")
                await asyncio.sleep(0)

    async def run() -> None:
        await asyncio.gather(*(answer(index) for index in range(WORKERS)))

    asyncio.run(run())

    per_question = meter.per_question()
    for index in range(WORKERS):
        cost = per_question[f"q{index}"]
        assert cost.calls == 3
        assert cost.prompt_tokens == 300 * (index + 1)


def test_a_worker_thread_sees_no_question_until_one_is_bound():
    # This is why a meter can be passed explicitly: a new thread starts with an
    # empty context, so the parent's open question is invisible inside a worker.
    meter = UsageMeter()
    seen: list[Any] = []
    with meter.question("q1") as charged, ThreadPoolExecutor(1) as pool:
        pool.submit(lambda: seen.append(current_question())).result()
        pool.submit(lambda: seen.append(_bound_question(charged))).result()

    assert seen[0] is None
    assert seen[1] is charged


def _bound_question(charged: Any) -> Any:
    """Bind a question meter handed to this thread and report what is active."""
    with charged.bind():
        return current_question()


def test_a_scope_opened_in_one_thread_can_be_passed_to_another():
    meter = UsageMeter()
    charged = meter.for_question("q1")

    def answer(index: int) -> None:
        MeteredCompleter(StubCompleter(prompt_tokens=10), meter, charge_to=charged)(
            "prompt"
        )

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(answer, range(4)))

    assert meter.per_question()["q1"].calls == 4
    assert meter.per_question()["q1"].prompt_tokens == 40
    assert meter.unattributed().calls == 0


def test_service_seconds_are_the_calls_and_elapsed_is_the_question():
    meter = UsageMeter()
    with meter.question("q1"):
        complete = MeteredCompleter(StubCompleter(sleep=0.02), meter)
        complete("prompt")
        complete("prompt")
        time.sleep(0.02)

    cost = meter.per_question()["q1"]
    assert cost.service_seconds > 0.0
    assert cost.elapsed_seconds > cost.service_seconds


def test_a_question_that_fans_out_spends_more_service_seconds_than_it_takes():
    # The reason both numbers are reported: four concurrent calls inside one
    # question bill four times the wall clock the question took.
    meter = UsageMeter()
    with meter.question("q1") as charged:
        complete = MeteredCompleter(StubCompleter(sleep=0.05), meter, charge_to=charged)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(complete, ["prompt"] * 4))

    cost = meter.per_question()["q1"]
    assert cost.calls == 4
    assert cost.service_seconds > cost.elapsed_seconds * 1.5


def test_measuring_one_question_twice_at_once_is_refused():
    meter = UsageMeter()
    with (
        meter.question("q1"),
        pytest.raises(RuntimeError, match="already being measured"),
        meter.question("q1"),
    ):
        pass


def test_a_question_measured_in_two_spans_adds_them_up():
    meter = UsageMeter()
    for _ in range(2):
        with meter.question("q1"):
            time.sleep(0.01)

    assert meter.per_question()["q1"].elapsed_seconds > 0.015


def test_the_completer_wrapper_counts_the_call_the_inner_completer_reported():
    meter = UsageMeter()
    with meter.question("q1"):
        assert MeteredCompleter(StubCompleter(), meter)("prompt") == "answer"

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.prompt_tokens == 10
    assert cost.completion_tokens == 2
    assert cost.tokens_complete


def test_a_completer_wrapped_twice_is_refused_rather_than_counted_twice():
    meter = UsageMeter()
    nested = MeteredCompleter(MeteredCompleter(StubCompleter(), meter), meter)
    with meter.question("q1"), pytest.raises(RuntimeError, match="not reentrant"):
        nested("prompt")
    cost = meter.per_question()["q1"]
    assert cost.calls == 0
    assert cost.calls_missing_usage == 0
    assert cost.tokens_complete


def test_a_completer_that_reports_nothing_still_has_its_call_counted():
    meter = UsageMeter()
    with meter.question("q1"):
        MeteredCompleter(lambda prompt: "answer", meter)("prompt")

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.calls_missing_usage == 1
    assert not cost.tokens_complete
    assert cost.service_seconds > 0.0


def test_an_unreported_empty_answer_is_still_read_as_an_abstention():
    meter = UsageMeter()
    with meter.question("q1"):
        MeteredCompleter(lambda prompt: "", meter)("prompt")

    assert meter.per_question()["q1"].empty_responses == 1


def test_a_retrying_completer_reports_every_attempt_it_paid_for():
    meter = UsageMeter()
    with meter.question("q1"):
        MeteredCompleter(StubCompleter(attempts=3), meter)("prompt")

    assert meter.per_question()["q1"].calls == 3


def test_a_tool_call_through_the_completer_is_a_call_not_an_abstention():
    meter = UsageMeter()
    with meter.question("q1"):
        MeteredCompleter(StubCompleter(text="", tool_calls=1), meter)("prompt")

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.tool_calls == 1
    assert cost.empty_responses == 0


def test_a_cache_hit_is_counted_as_a_hit_rather_than_a_call():
    meter = UsageMeter()

    def cached(prompt: str) -> str:
        report_cache_hit()
        return "answer"

    with meter.question("q1"):
        MeteredCompleter(cached, meter)("prompt")

    cost = meter.per_question()["q1"]
    assert cost.cache_hits == 1
    assert cost.calls == 0


def test_a_failed_call_is_not_counted_as_a_model_call():
    meter = UsageMeter()

    def broken(prompt: str) -> str:
        raise RuntimeError("proxy down")

    with meter.question("q1"), pytest.raises(RuntimeError, match="proxy down"):
        MeteredCompleter(broken, meter)("prompt")

    assert meter.per_question()["q1"].calls == 0


def test_the_meter_records_what_the_package_completer_spends():
    meter = UsageMeter()
    client, completions = fake_client(
        [chat_response(prompt_tokens=282, completion_tokens=26)]
    )
    complete = MeteredCompleter(ChatCompleter(client, "test-model"), meter)
    with meter.question("q1"):
        assert complete("say ok") == "ok"

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.prompt_tokens == 282
    assert cost.completion_tokens == 26
    assert cost.service_seconds > 0.0
    assert completions.requests[0]["extra_body"] == THINKING_OFF


def test_the_client_wrapper_records_a_completion():
    meter = UsageMeter()
    client, _ = fake_client([chat_response(prompt_tokens=282, completion_tokens=26)])
    with meter.question("q1"):
        MeteredClient(client, meter).chat.completions.create(model="m", messages=[])

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.total_tokens == 308


def test_the_client_wrapper_records_a_tool_call_as_a_call():
    meter = UsageMeter()
    client, _ = fake_client([chat_response(content=None, tool_calls=2)])
    with meter.question("q1"):
        MeteredClient(client, meter).chat.completions.create(model="m", messages=[])

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.tool_calls == 2
    assert cost.empty_responses == 0


def test_the_client_wrapper_records_an_awaited_call():
    meter = UsageMeter()
    client, _ = fake_client([chat_response()], asynchronous=True)

    async def ask() -> None:
        with meter.question("q1"):
            await MeteredClient(client, meter).chat.completions.create(
                model="m", messages=[]
            )

    asyncio.run(ask())
    assert meter.per_question()["q1"].calls == 1


def test_a_streamed_call_asks_for_usage_and_counts_what_comes_back():
    meter = UsageMeter()
    chunks = [text_chunk("ok"), usage_chunk()]
    client, completions = fake_client([chunks])
    with meter.question("q1"):
        stream = MeteredClient(client, meter).chat.completions.create(
            model="m", messages=[], stream=True
        )
        assert len(list(stream)) == 2

    assert completions.requests[0]["stream_options"] == {"include_usage": True}
    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.total_tokens == 308
    assert cost.tokens_complete
    assert cost.empty_responses == 0


def test_a_stream_reporting_no_usage_is_counted_as_unknown_not_zero():
    meter = UsageMeter()
    client, _ = fake_client([[text_chunk("ok")]])
    with meter.question("q1"):
        stream = MeteredClient(client, meter).chat.completions.create(
            model="m", messages=[], stream=True
        )
        with pytest.warns(MissingUsageWarning, match="include_usage"):
            list(stream)

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.calls_missing_usage == 1
    assert not cost.tokens_complete


def test_a_caller_asking_for_no_streamed_usage_keeps_their_own_request():
    meter = UsageMeter()
    client, completions = fake_client([[text_chunk("ok"), usage_chunk()]])
    stream = MeteredClient(client, meter).chat.completions.create(
        model="m",
        messages=[],
        stream=True,
        stream_options={"include_usage": False},
    )
    list(stream)
    assert completions.requests[0]["stream_options"] == {"include_usage": False}


def test_a_streamed_tool_call_is_counted_once_with_its_tokens():
    meter = UsageMeter()
    chunks = [tool_chunk(), tool_chunk(arguments='"Toronto"}'), usage_chunk()]
    client, _ = fake_client([chunks])
    with meter.question("q1"):
        list(
            MeteredClient(client, meter).chat.completions.create(
                model="m", messages=[], stream=True
            )
        )

    cost = meter.per_question()["q1"]
    assert cost.calls == 1
    assert cost.tool_calls == 1
    assert cost.empty_responses == 0
    assert cost.total_tokens == 308


def test_an_awaited_stream_is_metered_as_it_is_consumed():
    meter = UsageMeter()
    client, _ = fake_client([[text_chunk("ok"), usage_chunk()]], asynchronous=True)

    async def ask() -> int:
        with meter.question("q1"):
            stream = await MeteredClient(client, meter).chat.completions.create(
                model="m", messages=[], stream=True
            )
            return len([chunk async for chunk in stream])

    assert asyncio.run(ask()) == 2
    assert meter.per_question()["q1"].total_tokens == 308


def test_metering_a_client_in_place_is_idempotent():
    meter = UsageMeter()
    client, completions = fake_client([chat_response(), chat_response()])
    meter_client(meter_client(client, meter), meter)
    with meter.question("q1"):
        client.chat.completions.create(model="m", messages=[])

    assert meter.per_question()["q1"].calls == 1
    assert len(completions.requests) == 1


def test_an_in_place_metered_client_composes_with_the_thinking_switch():
    from aieng.advanced_rag.reasoning import disable_thinking

    meter = UsageMeter()
    client, completions = fake_client([chat_response()])
    disable_thinking(meter_client(client, meter))
    with meter.question("q1"):
        client.chat.completions.create(model="m", messages=[])

    assert completions.requests[0]["extra_body"] == THINKING_OFF
    assert meter.per_question()["q1"].calls == 1


def test_the_record_carries_the_run_the_questions_and_the_unattributed():
    meter = UsageMeter()
    with meter.question("q1") as charged:
        charged.record(CallUsage(10, 2, 12, reported=True), 0.25)
    meter.record(CallUsage(1, 1, 2, reported=True), 0.5)

    record = meter.to_dict()
    assert record["run"]["calls"] == 2
    assert record["run"]["questions"] == 1
    assert record["unattributed"]["calls"] == 1
    assert record["questions"]["q1"]["total_tokens"] == 12
    assert record["questions"]["q1"]["service_seconds"] == 0.25
    assert record["questions"]["q1"]["elapsed_seconds"] >= 0.0


def test_reset_drops_every_total():
    meter = UsageMeter()
    with meter.question("q1") as charged:
        charged.record(CallUsage(10, 2, 12, reported=True), 0.25)
    meter.reset()

    assert meter.per_question() == {}
    assert meter.totals().calls == 0


def test_the_default_meter_is_shared_and_resettable():
    meter = reset_default_meter()
    assert meter is default_meter()
    with meter.question("q1"):
        MeteredCompleter(StubCompleter())("prompt")

    assert default_meter().per_question()["q1"].calls == 1
    reset_default_meter()
    assert default_meter().per_question() == {}


def test_a_scoped_default_meter_keeps_two_runs_apart():
    shared = reset_default_meter()
    first = UsageMeter()
    with using(first) as installed:
        assert installed is first
        assert default_meter() is first
        with first.question("q1"):
            MeteredCompleter(StubCompleter())("prompt")

    assert default_meter() is shared
    assert first.totals().calls == 1
    assert shared.totals().calls == 0


def test_a_scoped_default_meter_is_restored_when_the_block_raises():
    shared = reset_default_meter()
    with pytest.raises(RuntimeError), using(UsageMeter()):
        raise RuntimeError("the run failed")
    assert default_meter() is shared


class FakeCompletionBase:
    """Stands in for GraphRAG's LiteLLM provider, without its configuration."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def completion(self, /, **kwargs: Any) -> Any:
        """Answer one request, recording what was asked."""
        self.requests.append(kwargs)
        return chat_response(prompt_tokens=100, completion_tokens=10)

    async def completion_async(self, /, **kwargs: Any) -> Any:
        """Answer one request asynchronously."""
        self.requests.append(kwargs)
        return chat_response(prompt_tokens=200, completion_tokens=20)


def test_the_graphrag_provider_meters_the_sync_and_the_async_path():
    meter = UsageMeter()
    provider = _metered_completion_class(
        FakeCompletionBase, _CallRecorder(meter, None)
    )()
    with meter.question("q1"):
        provider.completion(messages=[])
        asyncio.run(provider.completion_async(messages=[]))

    cost = meter.per_question()["q1"]
    assert cost.calls == 2
    assert cost.prompt_tokens == 300
    assert cost.completion_tokens == 30


def test_registering_the_graphrag_provider_is_idempotent():
    pytest.importorskip("graphrag_llm")
    from graphrag_llm.completion.completion_factory import completion_factory

    assert register_metered_completion() == METERED_PROVIDER
    assert METERED_PROVIDER in completion_factory
    assert register_metered_completion() == METERED_PROVIDER


def test_the_cost_names_still_import_as_the_usage_names():
    from aieng.advanced_rag import metering

    assert metering.CostMeter is UsageMeter
    assert metering.QuestionCost is metering.QuestionUsage
    assert metering.RunCost is metering.RunUsage
    assert metering.CompletionFacts is metering.ParsedCompletion


def test_the_scope_names_still_reach_a_question_meter():
    from aieng.advanced_rag import metering

    meter = UsageMeter()
    assert metering.QuestionScope is metering.QuestionMeter
    assert meter.scope("q1") is meter.for_question("q1")


class TestPerModelTotals:
    def test_calls_are_totalled_per_model_requested(self):
        meter = UsageMeter()
        with meter.question("q1") as question:
            question.record(CallUsage(10, 2, reported=True), 0.1, model="a")
            question.record(CallUsage(5, 1, reported=True), 0.1, model="b")
            question.record(CallUsage(1, 1, reported=True), 0.1, model="a")
        by_model = meter.by_model()
        assert by_model["a"].calls == 2
        assert by_model["a"].prompt_tokens == 11
        assert by_model["b"].completion_tokens == 1
        assert meter.totals().calls == 3

    def test_a_call_with_no_model_is_kept_under_unknown(self):
        from aieng.advanced_rag.metering import UNKNOWN_MODEL

        meter = UsageMeter()
        complete = MeteredCompleter(lambda prompt: "ok", meter)
        with meter.question("q1"):
            complete("p")
        assert meter.by_model()[UNKNOWN_MODEL].calls == 1

    def test_report_call_carries_the_model_through_a_wrapper(self):
        from aieng.advanced_rag.metering import report_call

        def complete(prompt):
            report_call(
                CallUsage(7, 3, reported=True),
                0.0,
                model="m1",
                response_model="m1-2026",
            )
            return "ok"

        meter = UsageMeter()
        MeteredCompleter(complete, meter)("p")
        usage = meter.by_model()["m1"]
        assert usage.prompt_tokens == 7
        assert usage.response_models == ("m1-2026",)

    def test_a_cache_hit_keeps_what_it_would_have_cost_apart(self):
        meter = UsageMeter()
        with meter.question("q1"):
            report_cache_hit(model="m1", usage=CallUsage(100, 20, reported=True))
        usage = meter.by_model()["m1"]
        assert usage.calls == 0 and usage.prompt_tokens == 0
        assert usage.cache_hits == 1
        assert usage.cached_prompt_tokens == 100
        assert meter.to_dict()["by_model"]["m1"]["cached_completion_tokens"] == 20

    def test_a_metered_client_reads_the_model_from_the_request(self):
        client, _ = fake_client_for_models()
        meter = UsageMeter()
        meter_client(client, meter)
        client.chat.completions.create(model="gpt-6-luna", messages=[])
        usage = meter.by_model()["gpt-6-luna"]
        assert usage.calls == 1
        assert usage.response_models == ("gpt-6-luna-snapshot",)

    def test_reset_drops_the_per_model_totals(self):
        meter = UsageMeter()
        meter.record(CallUsage(1, 1, reported=True), 0.0, model="a")
        meter.reset()
        assert meter.by_model() == {}

    def test_the_last_reported_snapshot_is_observable(self):
        from aieng.advanced_rag.metering import observed_response_model, report_call

        report_call(CallUsage(), 0.0, model="alias-x", response_model="alias-x-v2")
        assert observed_response_model("alias-x") == "alias-x-v2"

    def test_a_call_hook_sees_every_live_call_and_can_stop_the_run(self):
        from aieng.advanced_rag.metering import (
            add_call_hook,
            remove_call_hook,
            report_call,
        )

        seen = []

        def hook(model, usage):
            seen.append((model, usage.prompt_tokens))
            if len(seen) > 1:
                raise RuntimeError("stop")

        add_call_hook(hook)
        try:
            report_call(CallUsage(3, 1, reported=True), 0.0, model="a")
            with pytest.raises(RuntimeError, match="stop"):
                report_call(CallUsage(4, 1, reported=True), 0.0, model="a")
        finally:
            remove_call_hook(hook)
        report_call(CallUsage(5, 1, reported=True), 0.0, model="a")
        assert seen == [("a", 3), ("a", 4)]


def fake_client_for_models():
    from types import SimpleNamespace

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1, total_tokens=5),
            model=kwargs["model"] + "-snapshot",
        )

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), calls
