"""The sweep's cached completer: its cache keys, its counters, and its sends."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import litellm
import pytest
from aieng.advanced_rag import metering
from aieng.advanced_rag.graphrag.proxy import ResponseCache


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_lab.proxy import DEFAULT_MODEL, ProxyClient  # noqa: E402


class FakeLiteLLM:
    """Stands in for ``litellm.acompletion`` with a queue of reply texts."""

    def __init__(self, replies: list[str | None]) -> None:
        self.replies = replies
        self.requests: list[dict] = []

    async def __call__(self, **kwargs):
        self.requests.append(kwargs)
        return litellm.ModelResponse(
            id="r1",
            model=kwargs["model"].split("/", 1)[1],
            choices=[
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": self.replies.pop(0)}}
            ],
            usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        )


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(metering, "before_call", lambda model: None)

    def make(replies: list[str | None], cache: Path | None = None) -> tuple[ProxyClient, FakeLiteLLM]:
        fake = FakeLiteLLM(replies)
        monkeypatch.setattr(litellm, "acompletion", fake)
        proxy = ProxyClient(
            model=DEFAULT_MODEL,
            base_url="http://proxy.test/v1",
            api_key="test-key",
            cache_path=cache or tmp_path / "responses.sqlite",
        )
        return proxy, fake

    return make


def test_a_repeat_is_served_from_the_cache_and_counted_apart(client):
    proxy, fake = client(["answer"])
    complete = proxy.completer()
    assert complete("prompt") == "answer"
    assert complete("prompt") == "answer"
    assert len(fake.requests) == 1
    assert fake.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert proxy.usage() == {
        "logical_calls": 2,
        "network_calls": 1,
        "empty_responses": 0,
        "retried_failures": 0,
        "cache_hits": 1,
    }


def test_an_earlier_sweeps_cache_still_hits_and_stays_writable(client, tmp_path):
    path = tmp_path / "old.sqlite"
    key = ResponseCache.key({"model": DEFAULT_MODEL, "prompt": "seen", "max_tokens": 512, "temperature": 0.0})
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("INSERT INTO responses VALUES (?, ?)", (key, "recorded"))
    proxy, fake = client(["fresh"], cache=path)
    complete = proxy.completer()
    assert complete("seen") == "recorded"
    assert complete("unseen") == "fresh"
    assert len(fake.requests) == 1
    unseen = ResponseCache.key({"model": DEFAULT_MODEL, "prompt": "unseen", "max_tokens": 512, "temperature": 0.0})
    assert ResponseCache(path).get(unseen) == "fresh"


def test_an_empty_completion_fails_the_sweep_and_is_not_cached(client):
    proxy, _ = client([None])
    complete = proxy.completer()
    with pytest.raises(RuntimeError, match="empty completion"):
        complete("prompt")
    assert proxy.usage()["empty_responses"] == 1
    assert proxy.cache is not None and len(proxy.cache) == 0
