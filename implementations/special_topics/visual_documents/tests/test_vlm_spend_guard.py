"""The vision client sends paid calls under ADVANCED_RAG_SPEND_LIMIT_USD, offline."""

from __future__ import annotations

import types

import openai
import pytest
from aieng.advanced_rag import spend
from aieng.advanced_rag.spend import ProxyAccount, SpendLimitReached
from openai.types.chat import ChatCompletion
from vdocs import enrich
from vdocs.vlm import ProxyModels


class _Completions:
    def __init__(self) -> None:
        self.sent = 0

    def create(self, **kwargs: object) -> ChatCompletion:
        self.sent += 1
        return ChatCompletion.model_validate(
            {
                "id": f"fake-{self.sent}",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000},
            }
        )


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> _Completions:
    completions = _Completions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = types.SimpleNamespace(completions=completions)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    spend.reset_session()
    spend.use_account(ProxyAccount(None, "https://proxy.example/v1"))
    yield completions
    spend.use_account(None)
    spend.reset_session()


def test_crop_judging_stops_at_the_ceiling_without_an_open_session(
    fake_openai: _Completions, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # gemini-3.8-flash bills $0.75 per 1M input tokens offline (or the proxy's rate),
    # so one call passes a $0.10 ceiling and nothing after it is sent.
    monkeypatch.setenv(spend.SPEND_LIMIT_ENV, "0.10")
    models = ProxyModels("test-key", "https://proxy.example/v1", cache_path=tmp_path / "c.sqlite", rpm=100_000)
    with pytest.raises(SpendLimitReached):
        enrich.parallel_map(lambda i: models.complete("gemini-3.8-flash", f"crop {i}"), list(range(30)), workers=4)
    assert spend._SESSION.session is not None
    # Workers that were already past the pre-call check when the first call
    # returned can still send; nothing queued after that does.
    assert fake_openai.sent <= 4


def test_on_prem_calls_do_not_open_the_session(fake_openai: _Completions, tmp_path) -> None:
    models = ProxyModels("test-key", "https://proxy.example/v1", cache_path=tmp_path / "c.sqlite", rpm=100_000)
    models.complete("Qwen3_8-27B", "describe")
    assert spend._SESSION.session is None
