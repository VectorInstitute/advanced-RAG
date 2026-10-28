"""The proxy model sends each family the completion parameters it accepts."""

from __future__ import annotations

import litellm
import pytest

from aieng.advanced_rag.evaluation.synthetic.llm import proxy_model


def _completion(model: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="x",
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Return the keyword arguments of every request the model sends."""
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return _completion(kwargs["model"].split("/", 1)[1])

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


def _send(name: str, tmp_path) -> None:
    model = proxy_model(
        name,
        cache_path=tmp_path / f"{name}.sqlite",
        per_minute=600,
        api_key="test-key",
        base_url="http://127.0.0.1:9",
    )
    assert model("system", "prompt") == "ok"
    # A second identical call is answered by the cache and still tallied.
    assert model("system", "prompt") == "ok"
    ledger, tally = model.ledger
    assert ledger.calls == 1 and ledger.cache_hits == 1
    assert tally.to_dict() == {
        "cold_requests": 2,
        "cold_prompt_tokens": 6,
        "cold_completion_tokens": 2,
    }


def test_an_openai_model_gets_max_completion_tokens(sent, tmp_path):
    _send("gpt-6-luna", tmp_path)
    params = sent[-1]
    assert "max_tokens" not in params
    assert "max_completion_tokens" in params
    assert "temperature" not in params


def test_an_onprem_model_keeps_the_classic_parameters(sent, tmp_path):
    _send("Qwen3_6-35B-A3B", tmp_path)
    params = sent[-1]
    assert params["max_tokens"] and params["temperature"] == 0.0
    assert "max_completion_tokens" not in params


def test_an_anthropic_model_gets_max_tokens_without_temperature(sent, tmp_path):
    _send("claude-haiku-5-5", tmp_path)
    params = sent[-1]
    assert "max_tokens" in params and "temperature" not in params
