"""Tests for the thinking-off client wrapper."""

from __future__ import annotations

from types import SimpleNamespace

from aieng.advanced_rag.reasoning import THINKING_OFF, ThinkingOffClient


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.model_dump_called = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[])

    def helper(self) -> str:
        return "passthrough"


def build() -> tuple[ThinkingOffClient, RecordingCompletions]:
    completions = RecordingCompletions()
    inner = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        api_key="secret",
        models=SimpleNamespace(list=lambda: ["a"]),
    )
    return ThinkingOffClient(inner), completions


def test_thinking_is_disabled_on_every_call():
    client, completions = build()
    client.chat.completions.create(model="m", messages=[])
    assert completions.calls[0]["extra_body"] == THINKING_OFF


def test_the_switch_is_the_one_the_deployment_honours():
    # reasoning_effort is ignored by this deployment; chat_template_kwargs is not.
    assert THINKING_OFF["chat_template_kwargs"]["enable_thinking"] is False


def test_caller_supplied_extra_body_is_merged_not_dropped():
    client, completions = build()
    client.chat.completions.create(model="m", messages=[], extra_body={"top_k": 5})
    body = completions.calls[0]["extra_body"]
    assert body["top_k"] == 5
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_caller_can_override_the_switch_per_call():
    client, completions = build()
    client.chat.completions.create(
        model="m",
        messages=[],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert completions.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_other_kwargs_pass_through_untouched():
    client, completions = build()
    client.chat.completions.create(
        model="m", messages=[{"role": "user"}], temperature=0.0
    )
    call = completions.calls[0]
    assert call["model"] == "m" and call["temperature"] == 0.0


def test_unrelated_client_attributes_pass_through():
    client, _ = build()
    assert client.api_key == "secret"
    assert client.models.list() == ["a"]


def test_unrelated_completions_attributes_pass_through():
    client, _ = build()
    assert client.chat.completions.helper() == "passthrough"


def test_a_custom_extra_body_replaces_the_default():
    completions = RecordingCompletions()
    inner = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = ThinkingOffClient(inner, extra_body={"custom": 1})
    client.chat.completions.create(model="m", messages=[])
    assert completions.calls[0]["extra_body"] == {"custom": 1}


class TestDisableInPlace:
    """The in-place form exists because atlas-rag type-checks its client."""

    def test_the_client_object_is_unchanged_so_isinstance_still_passes(self):
        from aieng.advanced_rag.reasoning import disable_thinking

        client, _ = build()
        assert disable_thinking(client) is client

    def test_calls_carry_the_switch(self):
        from aieng.advanced_rag.reasoning import disable_thinking

        client, completions = build()
        disable_thinking(client)
        client.chat.completions.create(model="m", messages=[])
        assert completions.calls[0]["extra_body"] == THINKING_OFF

    def test_applying_it_twice_does_not_double_wrap(self):
        from aieng.advanced_rag.reasoning import disable_thinking

        client, completions = build()
        disable_thinking(disable_thinking(client))
        client.chat.completions.create(model="m", messages=[])
        assert completions.calls[0]["extra_body"] == THINKING_OFF

    def test_caller_extra_body_still_wins(self):
        from aieng.advanced_rag.reasoning import disable_thinking

        client, completions = build()
        disable_thinking(client)
        client.chat.completions.create(model="m", messages=[], extra_body={"top_k": 3})
        assert completions.calls[0]["extra_body"]["top_k"] == 3
