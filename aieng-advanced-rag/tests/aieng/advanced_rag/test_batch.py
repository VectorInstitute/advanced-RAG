"""The infermesh batch layer: keys shared with govern, retries, metering, guards."""

from __future__ import annotations

import asyncio
import warnings

import httpx
import litellm
import pytest
from infermesh import LMClient

from aieng.advanced_rag import runs, spend
from aieng.advanced_rag.batch import (
    DEFAULT_RPM,
    BatchCompleter,
    EmptyReplyError,
    ProxyBatch,
    proxy_rpm,
)
from aieng.advanced_rag.graphrag.proxy import (
    CallLedger,
    RateLimiter,
    ResponseCache,
    govern,
    request_key,
)
from aieng.advanced_rag.metering import UsageMeter, using
from aieng.advanced_rag.spend import ModelPrice, SpendLimitReached, SpendSession


ONPREM = "Qwen3_6-35B-A3B"
PROXY = "http://proxy.test/v1"


def model_response(
    text: str | None = "ok", model: str = ONPREM
) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="r1",
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    )


def provider_error(status: int) -> Exception:
    response = httpx.Response(status, request=httpx.Request("POST", PROXY))
    kinds = {
        400: litellm.BadRequestError,
        429: litellm.RateLimitError,
        502: litellm.BadGatewayError,
    }
    return kinds[status](
        message=f"HTTP {status}", model=ONPREM, llm_provider="openai", response=response
    )


class FakeLiteLLM:
    """Stands in for ``litellm.acompletion``: echoes the prompt, or fails on cue."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.failures: dict[str, list[Exception]] = {}

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        queued = self.failures.get(prompt)
        if queued:
            raise queued.pop(0)
        return model_response(f"reply to {prompt}", kwargs["model"].split("/", 1)[1])

    def prompts(self) -> list[str]:
        return [call["messages"][-1]["content"] for call in self.calls]


@pytest.fixture
def fake(monkeypatch):
    stub = FakeLiteLLM()
    monkeypatch.setattr(litellm, "acompletion", stub)
    # Retries wait for real otherwise.
    monkeypatch.setattr(LMClient, "_compute_retry_wait", staticmethod(lambda *_: 0.0))
    return stub


def make(tmp_path, model: str = ONPREM, **kwargs) -> ProxyBatch:
    kwargs.setdefault("cache", ResponseCache(tmp_path / "responses.sqlite"))
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("base_url", PROXY)
    return ProxyBatch(model, **kwargs)


PARAMS = {"max_tokens": 64, "temperature": 0.0}


class TestCacheSharedWithGovern:
    def test_a_cache_govern_wrote_answers_the_batch_with_no_client(self, tmp_path):
        from openai.types.chat import ChatCompletion

        cache = ResponseCache(tmp_path / "responses.sqlite")

        class Client:
            class chat:  # noqa: N801 - the attribute name OpenAI uses
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        data = model_response(
                            f"stored {kwargs['messages'][-1]['content']}"
                        )
                        return ChatCompletion.model_validate(data.model_dump())

        client = govern(Client(), cache=cache, limiter=RateLimiter(per_minute=1000))
        for prompt in ("a", "b"):
            client.chat.completions.create(
                model=ONPREM, messages=[{"role": "user", "content": prompt}], **PARAMS
            )

        batch = make(tmp_path, cache=cache, api_key=None)
        outcomes = batch.complete_batch(["a", "b"], **PARAMS)

        assert [o.text for o in outcomes] == ["stored a", "stored b"]
        assert all(o.cached for o in outcomes)
        assert batch._client is None
        assert batch.ledger.cache_hits == 2 and batch.ledger.calls == 0

    def test_govern_reads_what_the_batch_wrote(self, tmp_path, fake):
        cache = ResponseCache(tmp_path / "responses.sqlite")
        with make(tmp_path, cache=cache) as batch:
            batch.complete_batch(["a"], **PARAMS)

        class NoCalls:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        raise AssertionError("the cache should have answered")

        client = govern(NoCalls(), cache=cache, limiter=RateLimiter())
        response = client.chat.completions.create(
            model=ONPREM, messages=[{"role": "user", "content": "a"}], **PARAMS
        )
        assert response.choices[0].message.content == "reply to a"

    def test_the_key_is_the_one_govern_uses(self, tmp_path, fake):
        cache = ResponseCache(tmp_path / "responses.sqlite")
        with make(tmp_path, cache=cache) as batch:
            batch.complete("a", system="s", **PARAMS)
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "a"},
        ]
        assert cache.entry(request_key(ONPREM, messages, PARAMS)) is not None
        assert cache.entry(request_key(ONPREM, messages, PARAMS, thinking=True)) is None


class TestRequests:
    def test_an_onprem_request_carries_the_thinking_switch(self, tmp_path, fake):
        with make(tmp_path) as batch:
            batch.complete("a", **PARAMS)
        sent = fake.calls[0]
        assert sent["model"] == f"openai/{ONPREM}"
        assert sent["api_base"] == PROXY
        assert sent["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert sent["max_tokens"] == 64 and sent["temperature"] == 0.0

    def test_a_paid_request_is_adapted_for_its_family(self, tmp_path, fake):
        with make(tmp_path, model="claude-haiku-5-5") as batch:
            batch.complete("a", **PARAMS)
        sent = fake.calls[0]
        assert "temperature" not in sent
        assert "chat_template_kwargs" not in sent.get("extra_body", {})

    def test_a_callers_extra_body_is_merged_with_the_switch(self, tmp_path, fake):
        with make(tmp_path) as batch:
            batch.complete("a", extra_body={"top_k": 5}, **PARAMS)
        assert fake.calls[0]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 5,
        }

    def test_thinking_on_sends_no_switch(self, tmp_path, fake):
        with make(tmp_path, thinking=True) as batch:
            batch.complete("a", **PARAMS)
        assert "extra_body" not in fake.calls[0]

    def test_streaming_is_refused(self, tmp_path, fake):
        with make(tmp_path) as batch, pytest.raises(ValueError, match="streaming"):
            batch.complete("a", stream=True)

    def test_the_default_rate_is_under_the_measured_cap(self):
        assert proxy_rpm(ONPREM) < 90
        assert proxy_rpm("some-other-model") == DEFAULT_RPM

    def test_cache_control_warns_once(self, tmp_path, fake):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "doc",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "q"},
        ]
        with make(tmp_path) as batch, warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            batch.complete(messages, **PARAMS)
            batch.complete([*messages[:1], {"role": "user", "content": "r"}], **PARAMS)
        assert sum("cache_control" in str(w.message) for w in seen) == 1


class TestRetries:
    def test_a_400_fails_its_item_at_once_and_the_batch_carries_on(
        self, tmp_path, fake
    ):
        fake.failures["bad"] = [provider_error(400)]
        with make(tmp_path) as batch:
            outcomes = batch.complete_batch(["bad", "good"], **PARAMS)
        assert fake.prompts().count("bad") == 1
        assert isinstance(outcomes[0].error, litellm.BadRequestError)
        assert outcomes[1].text == "reply to good"
        assert batch.ledger.failed_attempts == 1

    def test_a_429_and_a_502_are_retried(self, tmp_path, fake):
        fake.failures["a"] = [provider_error(429), provider_error(502)]
        with make(tmp_path) as batch:
            outcome = batch.create("a", **PARAMS)
        assert outcome.text == "reply to a"
        assert outcome.metrics is not None and outcome.metrics.retries == 2
        assert batch.ledger.failed_attempts == 2 and batch.ledger.calls == 1

    def test_a_single_call_raises_its_error(self, tmp_path, fake):
        fake.failures["bad"] = [provider_error(400)]
        with make(tmp_path) as batch, pytest.raises(litellm.BadRequestError):
            batch.complete("bad", **PARAMS)

    def test_a_rerun_sends_only_what_failed(self, tmp_path, fake):
        fake.failures["b"] = [provider_error(400)]
        with make(tmp_path) as batch:
            first = batch.complete_batch(["a", "b", "c"], **PARAMS)
        assert [o.ok for o in first] == [True, False, True]
        with make(tmp_path) as batch:
            second = batch.complete_batch(["a", "b", "c"], **PARAMS)
        assert [o.cached for o in second] == [True, False, True]
        assert sorted(fake.prompts()) == ["a", "b", "b", "c"]
        assert second[1].text == "reply to b"


class TestMetering:
    def test_calls_and_hits_land_on_the_callers_question(self, tmp_path, fake):
        meter = UsageMeter()
        with make(tmp_path) as batch, using(meter), meter.question("q1"):
            batch.complete_batch(["a", "b"], **PARAMS)
            batch.complete_batch(["a", "c"], **PARAMS)
        totals = meter.per_question()["q1"]
        assert totals.calls == 3 and totals.cache_hits == 1
        assert totals.prompt_tokens == 30

    def test_an_empty_response_is_counted(self, tmp_path, fake, monkeypatch):
        async def empty(**kwargs):
            return model_response(None)

        monkeypatch.setattr(litellm, "acompletion", empty)
        with make(tmp_path) as batch:
            outcome = batch.create("a", **PARAMS)
        assert outcome.text == "" and batch.ledger.empty_responses == 1
        assert isinstance(outcome.error, EmptyReplyError)
        assert outcome.usage.prompt_tokens == 10

    @pytest.mark.parametrize("content", [None, "", "  \n"])
    def test_an_empty_completion_is_not_cached_and_is_asked_again(
        self, tmp_path, monkeypatch, content
    ):
        replies = [content, "answer"]

        async def reply(**kwargs):
            return model_response(replies.pop(0))

        monkeypatch.setattr(litellm, "acompletion", reply)
        cache = ResponseCache(tmp_path / "responses.sqlite")
        with make(tmp_path, cache=cache) as batch:
            first = batch.create("a", **PARAMS)
            assert not first.ok and len(cache) == 0
            second = batch.create("a", **PARAMS)
        assert second.ok and not second.cached and second.text == "answer"
        assert len(cache) == 1

    def test_progress_counts_hits_first(self, tmp_path, fake):
        with make(tmp_path) as batch:
            batch.complete_batch(["a"], **PARAMS)
            seen: list[tuple[int, int]] = []
            batch.complete_batch(
                ["a", "b"], on_progress=lambda d, t: seen.append((d, t)), **PARAMS
            )
        assert seen[0] == (1, 2) and seen[-1] == (2, 2)


PRICED = {
    "claude-haiku-5-5": ModelPrice("claude-haiku-5-5", 1_000_000.0, 1_000_000.0),
}


class TestGuards:
    def test_the_spend_limit_stops_a_batch(self, tmp_path, fake):
        account = spend.ProxyAccount(None, PROXY)
        session = SpendSession(
            UsageMeter(), limit_usd=1.0, source=account, table=PRICED
        )
        try:
            with (
                make(tmp_path, model="claude-haiku-5-5", max_parallel=1) as batch,
                pytest.raises(SpendLimitReached),
            ):
                batch.complete_batch(["a", "b", "c"], confirm=True, **PARAMS)
            assert len(fake.calls) == 1
            # The response that crossed the limit was kept.
            assert len(batch.cache) == 1
            with pytest.raises(SpendLimitReached):
                batch.complete("d", **PARAMS)
            assert len(fake.calls) == 1
        finally:
            session.close()

    def test_a_paid_batch_over_the_threshold_needs_confirm(self, tmp_path, fake):
        budget = spend.Budget(100.0, 0.0)
        with make(tmp_path, model="claude-haiku-5-5", table=PRICED) as batch:
            with pytest.raises(runs.ConfirmationRequiredError):
                batch.complete_batch(["a", "b"], budget=budget, **PARAMS)
            assert fake.calls == []
            outcomes = batch.complete_batch(
                ["a", "b"], budget=budget, confirm=True, **PARAMS
            )
        assert all(o.ok for o in outcomes)

    def test_a_step_load_or_run_checked_is_not_stopped_again(self, tmp_path, fake):
        budget = spend.Budget(100.0, 0.0)
        token = runs._STEP_CHECKED.set(True)
        try:
            with make(tmp_path, model="claude-haiku-5-5", table=PRICED) as batch:
                outcomes = batch.complete_batch(["a"], budget=budget, **PARAMS)
        finally:
            runs._STEP_CHECKED.reset(token)
        assert outcomes[0].ok

    def test_an_onprem_batch_never_asks(self, tmp_path, fake):
        with make(tmp_path) as batch:
            outcomes = batch.complete_batch(
                ["a"], budget=spend.Budget(0.01, 0.0), **PARAMS
            )
        assert outcomes[0].ok


class TestCompleter:
    def test_calling_sends_one_request_and_batch_sends_many(self, tmp_path, fake):
        with make(tmp_path) as batch:
            complete = batch.completer(system="s", **PARAMS)
            assert isinstance(complete, BatchCompleter)
            assert complete("a") == "reply to a"
            fake.failures["c"] = [provider_error(400)]
            replies = complete.batch(["a", "b", "c"])
        assert replies[:2] == ["reply to a", "reply to b"]
        assert isinstance(replies[2], litellm.BadRequestError)
        assert fake.calls[0]["messages"][0] == {"role": "system", "content": "s"}


class TestAsync:
    def test_the_async_batch_meters_on_the_callers_question(self, tmp_path, fake):
        meter = UsageMeter()

        async def run():
            with make(tmp_path) as batch, using(meter), meter.question("q"):
                return await batch.acomplete_batch(["a", "b"], **PARAMS)

        outcomes = asyncio.run(run())
        assert [o.text for o in outcomes] == ["reply to a", "reply to b"]
        assert meter.per_question()["q"].calls == 2


def test_the_ledger_passed_in_is_the_one_counted(tmp_path, fake):
    ledger = CallLedger("x")
    with make(tmp_path, ledger=ledger) as batch:
        batch.complete("a", **PARAMS)
    assert batch.ledger is ledger and ledger.calls == 1


class TestBuilders:
    """Each bulk builder sends one batch when given a ProxyBatch, with one result."""

    def test_graph_extraction_sends_every_window_as_one_batch(
        self, tmp_path, fake, monkeypatch
    ):
        from aieng.advanced_rag.graphrag import extraction

        payload = (
            '{"entities": [{"name": "nginx", "type": "Software", "evidence": "nginx"}],'
            ' "relations": []}'
        )

        async def answer(**kwargs):
            fake.calls.append(kwargs)
            return model_response(payload)

        monkeypatch.setattr(litellm, "acompletion", answer)
        documents = [
            ("q1", "Title: nginx\n\nQuestion:\nnginx fails.\n\nAnswer:\nRestart nginx.")
        ]
        batches: list[int] = []
        with make(tmp_path) as batch:
            complete = batch.completer(**PARAMS)
            original = complete.batch

            def counted(prompts, **kwargs):
                batches.append(len(prompts))
                return original(prompts, **kwargs)

            complete.batch = counted  # type: ignore[method-assign]
            graph = extraction.build_graph(documents, complete)
            threaded = extraction.build_graph(documents, lambda p: payload)
        assert batches == [len(fake.calls)] and batches[0] >= 1
        assert graph.to_dict() == threaded.to_dict()
        assert "nginx" in graph.entities

    def test_judges_run_as_one_batch_and_keep_failures_as_none(
        self, tmp_path, fake, monkeypatch
    ):
        from aieng.advanced_rag.evaluation.answers import AnswerCase, judge_answers

        async def answer(**kwargs):
            fake.calls.append(kwargs)
            prompt = kwargs["messages"][-1]["content"]
            if "broken" in prompt:
                raise provider_error(400)
            if "prose" in prompt:
                return model_response("no verdict")
            return model_response(
                '{"correct": true, "grounded": true, "abstained": false, "reason": "r"}'
            )

        monkeypatch.setattr(litellm, "acompletion", answer)
        cases = [
            AnswerCase("q1", "What?", "It is X.", "X", []),
            AnswerCase("q2", "broken", "It is Y.", "Y", []),
            AnswerCase("q3", "prose", "It is Z.", "Z", []),
        ]
        with make(tmp_path) as batch:
            verdicts = judge_answers(cases, batch.completer(**PARAMS))
        assert verdicts[0] is not None and verdicts[0].correct
        assert verdicts[1] is None and verdicts[2] is None
        assert len(fake.calls) == 3

    def test_judges_without_a_batch_give_the_same_verdicts(self):
        from aieng.advanced_rag.evaluation.answers import (
            AnswerCase,
            judge_answer,
            judge_answers,
        )

        def complete(prompt):
            return '{"correct": false, "grounded": true, "abstained": false}'

        case = AnswerCase("q1", "What?", "It is X.", "X", [])
        assert judge_answers([case], complete) == [judge_answer(*case, complete)]

    def test_contextual_chunking_and_names_replay_from_a_batch(self, tmp_path, fake):
        from aieng.advanced_rag.retrieval.chunkers import fixed_chars
        from aieng.advanced_rag.retrieval.contextual import (
            generate_contexts,
            name_documents,
        )

        document = (
            "MUTUAL NDA between Acme Ltd and Globex.\n\n" + "The parties agree. " * 60
        )
        chunks = fixed_chars("acme", document, chunk_chars=400, overlap_chars=80)
        with make(tmp_path) as batch:
            first = generate_contexts(
                {"acme": document}, chunks, batch, model=ONPREM, progress_every=0
            )
            names, usage = name_documents({"acme": document}, batch, model=ONPREM)
            calls = len(fake.calls)
            again = generate_contexts(
                {"acme": document}, chunks, batch, model=ONPREM, progress_every=0
            )
        assert calls == len(chunks) + 1 and len(fake.calls) == calls
        assert set(first.records) == {c.chunk_id for c in chunks}
        assert first.to_json() == again.to_json()
        assert first.records[chunks[0].chunk_id].prompt_tokens == 10
        assert list(names) == ["acme"] and names["acme"].startswith("reply to")
        assert usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 3}


class TestCallerKeysAndTextPayloads:
    """A cache with its own keys and text payloads, as older modules wrote them."""

    @staticmethod
    def old_key(messages, params):
        return "old:" + messages[-1]["content"] + f":{params['max_tokens']}"

    def test_a_text_cache_under_old_keys_answers_with_no_client(self, tmp_path):
        cache = ResponseCache(tmp_path / "old.sqlite")
        cache.put("old:a:64", "stored text", 5, 2)
        batch = make(
            tmp_path, cache=cache, api_key=None, key=self.old_key, payload="text"
        )
        outcome = batch.create("a", **PARAMS)
        assert (outcome.text, outcome.cached, outcome.response) == (
            "stored text",
            True,
            None,
        )
        assert outcome.usage.prompt_tokens == 5 and batch._client is None

    def test_a_live_reply_is_stored_as_text_under_the_callers_key(self, tmp_path, fake):
        cache = ResponseCache(tmp_path / "old.sqlite")
        with make(tmp_path, cache=cache, key=self.old_key, payload="text") as batch:
            outcomes = batch.complete_batch(["a", "b"], **PARAMS)
        assert [o.text for o in outcomes] == ["reply to a", "reply to b"]
        assert cache.get("old:a:64") == "reply to a"

    def test_an_empty_reply_is_not_cached_as_text(self, tmp_path, monkeypatch):
        async def empty(**kwargs):
            return model_response(None)

        monkeypatch.setattr(litellm, "acompletion", empty)
        cache = ResponseCache(tmp_path / "old.sqlite")
        with make(tmp_path, cache=cache, key=self.old_key, payload="text") as batch:
            assert batch.create("a", **PARAMS).text == ""
        assert len(cache) == 0

    @pytest.mark.parametrize("content", ["", "  \n"])
    def test_a_blank_reply_is_not_cached_as_text_and_is_asked_again(
        self, tmp_path, monkeypatch, content
    ):
        replies = [content, "answer"]

        async def reply(**kwargs):
            return model_response(replies.pop(0))

        monkeypatch.setattr(litellm, "acompletion", reply)
        cache = ResponseCache(tmp_path / "old.sqlite")
        with make(tmp_path, cache=cache, key=self.old_key, payload="text") as batch:
            first = batch.create("a", **PARAMS)
            assert isinstance(first.error, EmptyReplyError)
            assert cache.get("old:a:64") is None
            assert batch.create("a", **PARAMS).text == "answer"
        assert cache.get("old:a:64") == "answer"

    def test_an_unknown_payload_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="payload"):
            make(tmp_path, payload="bytes")

    def test_async_llm_takes_the_callers_key(self, tmp_path):
        from aieng.advanced_rag.graphrag.proxy import async_llm

        cache = ResponseCache(tmp_path / "old.sqlite")
        cache.put("old:hi:4000", "from the old cache", 1, 1)

        class NoCalls:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    async def create(**kwargs):
                        raise AssertionError("the cache should have answered")

        llm = async_llm(
            NoCalls(), ONPREM, cache=cache, limiter=RateLimiter(), key=self.old_key
        )
        assert asyncio.run(llm("hi")) == "from the old cache"


def test_an_older_cache_file_gains_the_token_columns(tmp_path):
    import sqlite3

    path = tmp_path / "sweep.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        db.execute("INSERT INTO responses VALUES ('k', 'old')")
    cache = ResponseCache(path)
    cache.put("k2", "new", 3, 4)
    assert cache.entry("k").usage.reported is False
    assert cache.entry("k2").usage.prompt_tokens == 3


class TestAsyncCreate:
    def test_acreate_answers_from_the_cache_then_live(self, tmp_path, fake):
        meter = UsageMeter()

        async def run():
            with make(tmp_path) as batch, using(meter), meter.question("q"):
                first = await batch.acreate("a", **PARAMS)
                again = await batch.acreate("a", **PARAMS)
                return first, again

        first, again = asyncio.run(run())
        assert first.text == again.text == "reply to a"
        assert (first.cached, again.cached) == (False, True)
        assert meter.per_question()["q"].calls == 1

    def test_acreate_raises_like_create(self, tmp_path, fake):
        fake.failures["bad"] = [provider_error(400)]

        async def run():
            with make(tmp_path) as batch:
                await batch.acreate("bad", **PARAMS)

        with pytest.raises(litellm.BadRequestError):
            asyncio.run(run())
