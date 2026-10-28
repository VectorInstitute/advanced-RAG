"""The shared proxy path: cache, request budget, retries, and the ledger."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from aieng.advanced_rag.graphrag import proxy
from aieng.advanced_rag.graphrag.proxy import (
    CallLedger,
    RateLimiter,
    ResponseCache,
    async_llm,
    backoff_seconds,
    govern,
)


def completion(text="hello", prompt=10, completion_tokens=3):
    return ChatCompletion(
        id="c1",
        created=0,
        model="m",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            total_tokens=prompt + completion_tokens,
        ),
    )


REQUEST = httpx.Request("POST", "http://proxy.test/v1/chat/completions")


def status_error(status):
    """Return the error the OpenAI client raises for an HTTP status."""
    response = httpx.Response(status, request=REQUEST)
    kinds = {
        400: openai.BadRequestError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }
    return kinds[status](f"HTTP {status}", response=response, body=None)


class FakeCreate:
    """A create function that can fail a set number of times first."""

    def __init__(self, text="hello", failures=0, content_none=False, error=None):
        self.calls = []
        self.failures = failures
        self.text = text
        self.content_none = content_none
        self.error = error or openai.APIConnectionError(request=REQUEST)

    def _response(self):
        return completion(None if self.content_none else self.text)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise self.error
        return self._response()


class AsyncFakeCreate(FakeCreate):
    async def __call__(self, **kwargs):  # type: ignore[override]
        return FakeCreate.__call__(self, **kwargs)


def sync_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(proxy, "backoff_seconds", lambda attempt: 0.0)


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache.sqlite")


def test_the_cache_key_is_stable_and_order_independent():
    assert ResponseCache.key({"a": 1, "b": 2}) == ResponseCache.key({"b": 2, "a": 1})
    assert ResponseCache.key({"a": 1}) != ResponseCache.key({"a": 2})


def test_the_cache_stores_and_counts(cache):
    assert cache.get("k") is None
    cache.put("k", "payload", 3, 4)
    assert cache.get("k") == "payload"
    assert len(cache) == 1


def test_the_cache_survives_a_new_connection(tmp_path):
    ResponseCache(tmp_path / "c.sqlite").put("k", "v", 0, 0)
    assert ResponseCache(tmp_path / "c.sqlite").get("k") == "v"


def test_the_limiter_admits_a_budget_then_makes_the_next_wait():
    limiter = RateLimiter(per_minute=2, window=0.3)
    limiter.acquire()
    limiter.acquire()
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started >= 0.25


def test_the_limiter_waits_asynchronously_too():
    async def run():
        limiter = RateLimiter(per_minute=1, window=0.2)
        await limiter.aacquire()
        started = time.monotonic()
        await limiter.aacquire()
        return time.monotonic() - started

    assert asyncio.run(run()) >= 0.15


def test_the_backoff_grows_and_is_capped():
    assert [backoff_seconds(i) for i in range(3)] == [1.0, 2.0, 4.0]
    assert backoff_seconds(20) == 30.0


def test_a_governed_client_answers_the_second_identical_request_from_the_cache(cache):
    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    first = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}]
    )
    second = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}]
    )
    assert (
        first.choices[0].message.content == second.choices[0].message.content == "hello"
    )
    assert len(create.calls) == 1
    assert client.ledger.calls == 1 and client.ledger.cache_hits == 1


def test_the_cached_response_keeps_its_usage(cache):
    client = govern(sync_client(FakeCreate()), cache=cache, limiter=RateLimiter(1000))
    request = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    client.chat.completions.create(**request)
    again = client.chat.completions.create(**request)
    assert again.usage.prompt_tokens == 10


def test_different_parameters_are_different_requests(cache):
    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    messages = [{"role": "user", "content": "q"}]
    client.chat.completions.create(model="m", messages=messages, max_tokens=10)
    client.chat.completions.create(model="m", messages=messages, max_tokens=20)
    assert len(create.calls) == 2


def test_reasoning_is_switched_off_unless_asked_to_stay_on(cache):
    off, on = FakeCreate(), FakeCreate()
    govern(
        sync_client(off), cache=cache, limiter=RateLimiter(1000)
    ).chat.completions.create(model="m", messages=[])
    govern(
        sync_client(on), cache=cache, limiter=RateLimiter(1000), thinking=True
    ).chat.completions.create(model="m", messages=[])
    assert off.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "extra_body" not in on.calls[0]


def test_a_failed_attempt_is_retried_and_counted(cache):
    create = FakeCreate(failures=2)
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    response = client.chat.completions.create(model="m", messages=[])
    assert response.choices[0].message.content == "hello"
    assert len(create.calls) == 3
    assert client.ledger.failed_attempts == 2 and client.ledger.calls == 1


@pytest.mark.parametrize("status", [429, 500])
def test_a_rate_limit_or_server_error_is_retried(cache, status):
    create = FakeCreate(failures=1, error=status_error(status))
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    response = client.chat.completions.create(model="m", messages=[])
    assert response.choices[0].message.content == "hello"
    assert len(create.calls) == 2


def test_a_timeout_is_retried(cache):
    create = FakeCreate(failures=1, error=openai.APITimeoutError(request=REQUEST))
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    client.chat.completions.create(model="m", messages=[])
    assert len(create.calls) == 2


def test_a_bad_request_raises_on_the_first_attempt_without_waiting(cache, monkeypatch):
    waits = []
    monkeypatch.setattr(proxy, "backoff_seconds", waits.append)
    create = FakeCreate(failures=99, error=status_error(400))
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="m", messages=[])
    assert len(create.calls) == 1
    assert waits == []
    assert client.ledger.failed_attempts == 1 and client.ledger.calls == 0


def test_a_call_that_always_fails_raises_instead_of_returning_nothing(cache):
    client = govern(
        sync_client(FakeCreate(failures=99)),
        cache=cache,
        limiter=RateLimiter(1000),
        max_retries=3,
    )
    with pytest.raises(RuntimeError, match="after 3 tries"):
        client.chat.completions.create(model="m", messages=[])


def test_an_empty_response_is_counted_not_hidden(cache):
    client = govern(
        sync_client(FakeCreate(content_none=True)),
        cache=cache,
        limiter=RateLimiter(1000),
    )
    client.chat.completions.create(model="m", messages=[])
    assert client.ledger.empty_responses == 1


def test_streaming_is_refused(cache):
    client = govern(sync_client(FakeCreate()), cache=cache, limiter=RateLimiter(1000))
    with pytest.raises(ValueError, match="streaming"):
        client.chat.completions.create(model="m", messages=[], stream=True)


def test_the_ledger_sums_calls_and_tokens():
    ledger = CallLedger("run")
    from aieng.advanced_rag.metering import CallUsage

    ledger.record(CallUsage(10, 5), 1.5)
    ledger.record(CallUsage(20, 5), 0.5)
    ledger.hit()
    report = ledger.to_dict()
    assert report["calls"] == 2 and report["total_tokens"] == 40
    assert report["cache_hits"] == 1 and report["call_seconds"] == 2.0


def async_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_the_lightrag_function_builds_messages_caches_and_meters(cache):
    create = AsyncFakeCreate(text="entities")
    llm = async_llm(
        async_client(create), "model-x", cache=cache, limiter=RateLimiter(1000)
    )

    async def run():
        first = await llm("extract", system_prompt="sys", history_messages=[])
        second = await llm("extract", system_prompt="sys", history_messages=[])
        return first, second

    first, second = asyncio.run(run())
    assert first == second == "entities"
    assert len(create.calls) == 1
    sent = create.calls[0]
    assert sent["model"] == "model-x"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert llm.ledger.calls == 1 and llm.ledger.cache_hits == 1


def test_lightrag_only_kwargs_never_reach_the_proxy(cache):
    create = AsyncFakeCreate()
    llm = async_llm(async_client(create), "m", cache=cache, limiter=RateLimiter(1000))
    asyncio.run(llm("p", hashing_kv=object(), keyword_extraction=True, max_tokens=64))
    assert create.calls[0]["max_tokens"] == 64
    assert not {"hashing_kv", "keyword_extraction"} & set(create.calls[0])


def test_the_async_function_retries_and_then_raises(cache):
    flaky = AsyncFakeCreate(failures=1)
    llm = async_llm(async_client(flaky), "m", cache=cache, limiter=RateLimiter(1000))
    assert asyncio.run(llm("p")) == "hello"
    dead = AsyncFakeCreate(failures=99)
    llm = async_llm(
        async_client(dead), "m", cache=cache, limiter=RateLimiter(1000), max_retries=2
    )
    with pytest.raises(RuntimeError, match="after 2 tries"):
        asyncio.run(llm("other"))


def test_the_async_function_retries_a_rate_limit(cache):
    create = AsyncFakeCreate(failures=1, error=status_error(429))
    llm = async_llm(async_client(create), "m", cache=cache, limiter=RateLimiter(1000))
    assert asyncio.run(llm("p")) == "hello"
    assert len(create.calls) == 2
    assert llm.ledger.failed_attempts == 1


def test_the_async_function_raises_a_bad_request_without_waiting(cache, monkeypatch):
    waits = []
    monkeypatch.setattr(proxy, "backoff_seconds", waits.append)
    create = AsyncFakeCreate(failures=99, error=status_error(400))
    llm = async_llm(async_client(create), "m", cache=cache, limiter=RateLimiter(1000))
    with pytest.raises(openai.BadRequestError):
        asyncio.run(llm("p"))
    assert len(create.calls) == 1
    assert waits == []
    assert llm.ledger.failed_attempts == 1


def test_the_onprem_lightrag_key_is_the_one_the_reference_caches_hold(cache):
    # The reference indexes were built when the key named only max_tokens and
    # temperature. A new key shape would turn every cached extraction into a miss.
    create = AsyncFakeCreate(text="entities")
    llm = async_llm(
        async_client(create), "Qwen3_6-35B-A3B", cache=cache, limiter=RateLimiter(1000)
    )
    asyncio.run(llm("extract", max_tokens=64))
    messages = [{"role": "user", "content": "extract"}]
    legacy = proxy._request(
        "chat-text", "Qwen3_6-35B-A3B", messages, {"max_tokens": 64, "temperature": 0.0}
    )
    assert cache.get(legacy) == "entities"


def test_a_paid_model_in_lightrag_gets_its_own_parameters_and_key(cache):
    create = AsyncFakeCreate(text="entities")
    llm = async_llm(
        async_client(create), "gpt-6-luna", cache=cache, limiter=RateLimiter(1000)
    )
    asyncio.run(llm("extract", max_tokens=64))
    sent = create.calls[0]
    assert sent["max_completion_tokens"] == 64
    assert "temperature" not in sent
    assert "chat_template_kwargs" not in sent.get("extra_body", {})
    messages = [{"role": "user", "content": "extract"}]
    legacy = proxy._request(
        "chat-text", "gpt-6-luna", messages, {"max_tokens": 64, "temperature": 0.0}
    )
    assert cache.get(legacy) is None


def test_the_cache_stores_the_requested_and_reported_model(cache):
    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    client.chat.completions.create(
        model="Qwen3_8-27B", messages=[{"role": "user", "content": "x"}]
    )
    (key,) = [row[0] for row in cache._db.execute("SELECT key FROM responses")]
    entry = cache.entry(key)
    assert entry is not None
    assert entry.model == "Qwen3_8-27B"
    # The fake response reports "m", standing in for a resolved snapshot.
    assert entry.response_model == "m"
    assert entry.usage.prompt_tokens == 10


def test_a_cache_hit_reports_the_stored_tokens_per_model(cache):
    from aieng.advanced_rag.metering import UsageMeter

    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    request = {"model": "Qwen3_8-27B", "messages": [{"role": "user", "content": "x"}]}
    client.chat.completions.create(**request)
    meter = UsageMeter()
    with meter.question("q"):
        client.chat.completions.create(**request)
    usage = meter.by_model()["Qwen3_8-27B"]
    assert usage.calls == 0
    assert usage.cache_hits == 1
    assert usage.cached_prompt_tokens == 10
    assert usage.cached_completion_tokens == 3


def test_a_cache_file_from_before_the_model_columns_opens_and_reads(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(str(path))
    db.execute(
        "CREATE TABLE responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
        "prompt_tokens INT, completion_tokens INT)"
    )
    db.execute("INSERT INTO responses VALUES ('k', 'text', 5, 1)")
    db.commit()
    db.close()
    cache = ResponseCache(path)
    entry = cache.entry("k")
    assert entry is not None
    assert entry.payload == "text"
    assert entry.model is None
    assert cache.get("k") == "text"


def test_a_governed_client_adapts_a_library_request_for_claude(cache):
    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    request = {
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": "Return JSON."}],
        "max_tokens": 64,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    client.chat.completions.create(**request)
    client.chat.completions.create(**request)
    (sent,) = create.calls
    assert "temperature" not in sent and "response_format" not in sent
    assert sent["max_tokens"] == 64
    assert client.ledger.cache_hits == 1


def test_a_governed_client_keys_the_cache_on_the_request_as_written(cache):
    create = FakeCreate()
    client = govern(sync_client(create), cache=cache, limiter=RateLimiter(1000))
    messages = [{"role": "user", "content": "q"}]
    client.chat.completions.create(
        model="gpt-6-luna", messages=messages, max_tokens=8, temperature=0.0
    )
    written = proxy._request(
        "chat-completion",
        "gpt-6-luna",
        messages,
        {"max_tokens": 8, "temperature": 0.0, "thinking": False},
    )
    assert cache.entry(written) is not None
    assert create.calls[0]["max_completion_tokens"] == 8
