"""RI1 uses shared credentials, model roles, and bounded, replayable requests."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import litellm
import pytest
from aieng.advanced_rag import metering
from aieng.advanced_rag.models import REFERENCE_MODELS, choose_models, request_params
from aieng.advanced_rag.spend import SpendLimitReached


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_not_stated_retry as retry_judge  # noqa: E402
from advanced_lab import common  # noqa: E402
from graphrag_null import judge_null_arm, null_lab  # noqa: E402


@pytest.fixture
def configuration(monkeypatch, tmp_path):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VECTOR_PROXY_API_KEY",
        "ADVANCED_RAG_MODEL_FAMILY",
        "ADVANCED_RAG_GENERATION_MODEL",
        "ADVANCED_RAG_EXTRACTION_MODEL",
        "ADVANCED_RAG_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "STORE", tmp_path)
    monkeypatch.setattr(common.spend, "session", lambda: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    return tmp_path


class FakeLiteLLM:
    """Stands in for ``litellm.acompletion``: records each request and answers it.

    A prompt that is a number ``n`` costs ``n + 1`` prompt tokens, so concurrent
    calls can be told apart by their usage.
    """

    def __init__(self, barrier=None):
        self.requests = []
        self.barrier = barrier

    async def __call__(self, **kwargs):
        self.requests.append(kwargs)
        if self.barrier is not None:
            import asyncio  # noqa: PLC0415

            await asyncio.get_running_loop().run_in_executor(None, self.barrier.wait, 5)
        prompt = kwargs["messages"][-1]["content"]
        tokens = int(prompt) + 1 if prompt.isdigit() else 10
        return litellm.ModelResponse(
            id="r1",
            model=kwargs["model"].split("/", 1)[1],
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
            usage={"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1},
        )


@pytest.fixture
def fake_proxy(monkeypatch):
    fake = FakeLiteLLM()
    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


def batch(model=REFERENCE_MODELS.generation.model, cache=None):
    return null_lab.proxy_batch(model, cache, api_key="test-key", base_url="http://proxy.test/v1")


@pytest.mark.parametrize("role", ["generation", "judging", "extraction"])
def test_advanced_client_uses_onboarding_and_shared_roles(configuration, monkeypatch, fake_proxy, role):
    monkeypatch.setenv("OPENAI_API_KEY", "onboard-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "older-alias")
    monkeypatch.setenv("ADVANCED_RAG_EXTRACTION_MODEL", "gpt-oss-120b")
    monkeypatch.setattr(metering, "before_call", lambda model: None)
    complete = common.completer(role=role)
    assert complete.model == getattr(choose_models(), role).model
    assert complete("question") == "answer"
    assert fake_proxy.requests[0]["api_key"] == "onboard-key"
    assert fake_proxy.requests[0]["api_base"] == "https://proxy.vectorinstitute.ai/v1"


def test_reference_cache_replay_keeps_the_original_request_identity(configuration, monkeypatch):
    cache = null_lab.ResponseCache(configuration / "responses.sqlite")
    key = cache.key(
        {
            "model": REFERENCE_MODELS.generation.model,
            "prompt": "question",
            "max_tokens": 1024,
            "temperature": 0.0,
            "thinking": False,
        }
    )
    cache.put(key, "recorded answer", 10, 5)
    monkeypatch.setattr(common, "resolve_api_key", lambda **kwargs: pytest.fail("replay read credentials"))
    complete = common.completer(cache_only=True)
    assert complete.model == REFERENCE_MODELS.generation.model
    assert complete("question") == "recorded answer"
    assert complete.meter.calls == 0
    assert complete.meter.cache_hits == 1


def test_paid_family_request_and_offline_cache_replay_match(configuration, monkeypatch, fake_proxy):
    model = choose_models("google").generation.model
    cache = null_lab.ResponseCache(configuration / "paid.sqlite")
    monkeypatch.setattr(metering, "before_call", lambda model: None)
    complete = null_lab.MeteredCompleter(batch(model, cache))
    assert complete("question") == "answer"
    params = request_params(model, max_tokens=512, temperature=0, thinking=False)
    assert all(fake_proxy.requests[0][name] == value for name, value in params.items())
    replay = null_lab.MeteredCompleter(null_lab.proxy_batch(model, cache, cache_only=True))
    assert replay("question") == "answer"
    assert replay.meter.calls == 0 and replay.meter.cache_hits == 1


def test_spend_refusal_does_not_retry_or_call_the_provider(configuration, monkeypatch, fake_proxy):
    checks = []

    def refuse(model):
        checks.append(model)
        raise SpendLimitReached("fixture budget")

    monkeypatch.setattr(metering, "before_call", refuse)
    complete = null_lab.MeteredCompleter(batch())
    with pytest.raises(SpendLimitReached):
        complete("question")
    assert len(checks) == 1 and fake_proxy.requests == []
    assert complete.meter.calls == 0


def test_live_judge_defaults_and_reference_replay_are_distinct(configuration):
    live = judge_null_arm.parse_args(["--store", str(configuration)])
    offline = judge_null_arm.parse_args(["--store", str(configuration), "--cache-only"])
    assert live.model == choose_models().judging.model
    assert offline.model == judge_null_arm.DEFAULT_MODEL
    live_retry = retry_judge.parse_args([])
    assert live_retry.generator_model == choose_models().generation.model
    assert live_retry.judge_model == choose_models().judging.model
    replay_retry = retry_judge.parse_args(["--cache-only"])
    assert replay_retry.generator_model == retry_judge.GENERATOR_MODEL
    assert replay_retry.judge_model == retry_judge.JUDGE_MODEL


def test_concurrent_calls_preserve_each_responses_usage(configuration, monkeypatch):
    # Both responses complete before either worker reads its usage.
    monkeypatch.setattr(litellm, "acompletion", FakeLiteLLM(barrier=Barrier(2)))
    monkeypatch.setattr(metering, "before_call", lambda model: None)
    complete = null_lab.MeteredCompleter(batch())
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(complete, ["0", "1"])) == ["answer", "answer"]
    assert complete.meter.calls == 2
    assert complete.meter.prompt_tokens == 3
    assert complete.meter.completion_tokens == 2
    assert complete.meter.total_tokens == 5
