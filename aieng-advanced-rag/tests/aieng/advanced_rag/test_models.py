"""Tests for the model set and the per-family request adapter."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from aieng.advanced_rag import models
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import (
    DEFAULT_FAMILY,
    DEFAULT_MODELS,
    FAMILIES,
    FAMILY_ENV,
    ROLE_ENV,
    ROLES,
    SHARED_JUDGE,
    UnlistedModelWarning,
    adapt_request,
    adapt_response_format,
    choose_models,
    endpoint_of,
    family_of,
    request_params,
    thinking_body,
)
from aieng.advanced_rag.pipeline.generate import ChatCompleter
from aieng.advanced_rag.reasoning import THINKING_OFF, disable_thinking


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv(FAMILY_ENV, raising=False)
    for name in ROLE_ENV.values():
        monkeypatch.delenv(name, raising=False)


class TestTheDefaultTable:
    def test_every_family_has_a_model_for_every_role(self):
        assert set(DEFAULT_MODELS) == set(FAMILIES)
        for family, roles in DEFAULT_MODELS.items():
            assert set(roles) == set(ROLES), family

    def test_every_default_belongs_to_its_family_except_the_shared_judge(self):
        for family, roles in DEFAULT_MODELS.items():
            for role, model in roles.items():
                if role == "judging" and family != "onprem":
                    assert model == SHARED_JUDGE
                else:
                    assert family_of(model) == family, model

    def test_generation_and_extraction_share_the_cheap_tier(self):
        for family in ("anthropic", "google", "openai"):
            roles = DEFAULT_MODELS[family]
            assert roles["generation"] == roles["extraction"]
            assert roles["judging"] != roles["generation"]

    def test_no_default_is_a_frontier_model(self):
        frontier = set(models.FRONTIER_MODELS.values())
        defaults = {m for roles in DEFAULT_MODELS.values() for m in roles.values()}
        assert not frontier & defaults

    def test_the_shut_down_flash_lite_preview_is_not_a_default(self):
        defaults = {m for roles in DEFAULT_MODELS.values() for m in roles.values()}
        assert "gemini-3.1-flash-lite-preview" not in defaults

    def test_the_onprem_family_is_the_reference_set(self):
        assert DEFAULT_MODELS["onprem"]["generation"] == "Qwen3_6-35B-A3B"
        assert models.REFERENCE_MODELS.family == "onprem"

    def test_every_default_has_an_offline_price(self):
        from aieng.advanced_rag.spend import OFFLINE_PRICES

        for roles in DEFAULT_MODELS.values():
            for model in roles.values():
                assert model in OFFLINE_PRICES, model
        for model in models.FRONTIER_MODELS.values():
            assert model in OFFLINE_PRICES, model


class TestFamilies:
    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("claude-haiku-5-5", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("gemini-3.8-flash", "google"),
            ("gpt-6-luna", "openai"),
            ("o4-mini", "openai"),
            ("gpt-oss-120b", "onprem"),
            ("Qwen3_8-27B", "onprem"),
            ("Nemotron-3-Nano-Omni-30B-A3B", "onprem"),
            ("my-local-model", None),
        ],
    )
    def test_a_name_maps_to_its_family(self, model, family):
        assert family_of(model) == family

    def test_the_endpoint_class_follows_the_family(self):
        assert endpoint_of("Qwen3_6-35B-A3B") == "vector-cluster"
        assert endpoint_of("gpt-6-luna") == "vector-proxy-paid"
        assert endpoint_of("my-local-model") == "unknown"


class TestChooseModels:
    def test_the_default_family_is_one_constant(self):
        assert choose_models().family == DEFAULT_FAMILY

    def test_an_argument_picks_the_family(self):
        chosen = choose_models("anthropic")
        assert chosen.generation.model == "claude-haiku-5-5"
        assert chosen.judging.model == SHARED_JUDGE
        assert all(choice.source == "default" for choice in chosen)

    def test_every_paid_family_shares_one_judge(self):
        judges = {
            choose_models(family).judging.model
            for family in ("anthropic", "google", "openai")
        }
        assert judges == {SHARED_JUDGE}

    def test_the_environment_picks_the_family(self, monkeypatch):
        monkeypatch.setenv(FAMILY_ENV, "OpenAI")
        assert choose_models().generation.model == "gpt-6-luna"

    def test_an_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(FAMILY_ENV, "openai")
        assert choose_models("onprem").family == "onprem"

    def test_a_role_override_from_an_argument(self):
        chosen = choose_models("anthropic", judging="claude-opus-5-5")
        assert chosen.judging.model == "claude-opus-5-5"
        assert chosen.judging.source == "argument"
        assert chosen.generation.model == "claude-haiku-5-5"

    def test_a_role_override_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(ROLE_ENV["generation"], "gpt-6-luna")
        chosen = choose_models("google")
        assert chosen.generation.model == "gpt-6-luna"
        assert chosen.generation.family == "openai"
        assert chosen.generation.source == "environment"

    def test_an_argument_override_beats_the_role_variable(self, monkeypatch):
        monkeypatch.setenv(ROLE_ENV["extraction"], "gpt-6-luna")
        chosen = choose_models("google", extraction="Qwen3_8-27B")
        assert chosen.extraction.model == "Qwen3_8-27B"

    def test_an_unlisted_model_is_accepted_with_a_warning(self):
        with pytest.warns(UnlistedModelWarning):
            chosen = choose_models("openai", generation="gpt-9-experimental")
        assert chosen.generation.model == "gpt-9-experimental"

    def test_a_listed_model_raises_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            choose_models("anthropic", judging="claude-opus-5-5")

    def test_an_unknown_family_is_refused(self):
        with pytest.raises(ValueError, match="unknown model family"):
            choose_models("mistral")

    def test_the_set_serialises(self):
        record = choose_models("onprem").as_dict()
        assert record["family"] == "onprem"
        assert record["embeddings"] == "BAAI/bge-small-en-v1.5"
        assert record["judging"] == "Qwen3_6-35B-A3B"

    def test_a_choice_names_its_model_and_endpoint_for_a_cache_key(self):
        assert choose_models("google").generation.config() == {
            "model": "gemini-3.5-flash-lite",
            "endpoint": "vector-proxy-paid",
        }

    def test_for_role_refuses_an_unknown_role(self):
        with pytest.raises(ValueError):
            choose_models().for_role("ranking")  # type: ignore[arg-type]


class TestRequestParams:
    def test_onprem_sends_the_vllm_switch_and_the_classic_parameters(self):
        params = request_params("Qwen3_6-35B-A3B", max_tokens=512)
        assert params == {
            "max_tokens": 512,
            "temperature": 0.0,
            "extra_body": THINKING_OFF,
        }

    def test_onprem_with_thinking_on_sends_no_switch(self):
        assert "extra_body" not in request_params(
            "Qwen3_6-35B-A3B", max_tokens=8, thinking=True
        )

    def test_openai_uses_max_completion_tokens_and_no_temperature(self):
        params = request_params("gpt-6-luna", max_tokens=256, temperature=0.0)
        assert params["max_completion_tokens"] == 256
        assert "max_tokens" not in params
        assert "temperature" not in params
        assert params["reasoning_effort"] == "none"

    def test_google_turns_reasoning_off_with_the_effort_the_proxy_accepts(self):
        params = request_params("gemini-3.5-flash-lite", max_tokens=64)
        assert params["extra_body"] == {"reasoning_effort": "minimal"}
        assert params["temperature"] == 0.0
        assert "chat_template_kwargs" not in params["extra_body"]

    def test_claude_from_4_7_on_gets_no_temperature(self):
        assert "temperature" not in request_params("claude-haiku-5-5", max_tokens=8)
        assert "temperature" not in request_params("claude-sonnet-4-7", max_tokens=8)

    def test_older_claude_keeps_its_temperature(self):
        assert request_params("claude-sonnet-4-6", max_tokens=8)["temperature"] == 0.0
        # A date suffix is not a minor version.
        assert (
            request_params("claude-haiku-4-5-20251001", max_tokens=8)["temperature"]
            == 0.0
        )

    def test_no_paid_family_receives_the_onprem_switch(self):
        for family in ("anthropic", "google", "openai"):
            model = DEFAULT_MODELS[family]["generation"]
            params = request_params(model, max_tokens=8)
            assert "chat_template_kwargs" not in params.get("extra_body", {}), model

    def test_an_unknown_name_is_treated_as_self_hosted_vllm(self):
        assert (
            request_params("my-local-model", max_tokens=8)["extra_body"] == THINKING_OFF
        )

    def test_the_result_is_a_fresh_copy(self):
        first = request_params("Qwen3_6-35B-A3B", max_tokens=8)
        first["extra_body"]["chat_template_kwargs"] = "changed"
        assert (
            request_params("Qwen3_6-35B-A3B", max_tokens=8)["extra_body"]
            == THINKING_OFF
        )

    def test_google_response_schemas_lose_additional_properties(self):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "a": {"type": "object", "additionalProperties": False}
                    },
                }
            },
        }
        adapted = adapt_response_format("gemini-3.5-flash-lite", schema)
        assert "additionalProperties" not in str(adapted)
        assert adapt_response_format("gpt-6-luna", schema) is schema


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2, total_tokens=12
            ),
            model=kwargs["model"] + "-20261001",
        )


def recording_client() -> tuple[SimpleNamespace, RecordingCompletions]:
    completions = RecordingCompletions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class TestTheAdapterThroughTheCompleter:
    @pytest.mark.parametrize("family", ["anthropic", "google", "openai", "onprem"])
    def test_the_completer_sends_what_the_adapter_returns(self, family):
        client, completions = recording_client()
        model = DEFAULT_MODELS[family]["generation"]
        ChatCompleter(client, model, max_tokens=99)("prompt")
        sent = dict(completions.calls[0])
        assert sent.pop("model") == model
        sent.pop("messages")
        assert sent == request_params(model, max_tokens=99)

    def test_the_completer_reports_the_model_to_the_meter(self):
        client, _ = recording_client()
        meter = UsageMeter()
        with meter.question("q1"):
            ChatCompleter(client, "gpt-6-luna")("prompt")
        usage = meter.by_model()["gpt-6-luna"]
        assert usage.calls == 1
        assert usage.prompt_tokens == 10
        assert usage.response_models == ("gpt-6-luna-20261001",)

    def test_the_completer_keeps_the_reported_model(self):
        client, _ = recording_client()
        complete = ChatCompleter(client, "claude-haiku-5-5")
        complete("prompt")
        assert complete.last_response_model == "claude-haiku-5-5-20261001"


class TestThinkingSwitchPerModel:
    def test_a_patched_client_sends_the_onprem_switch_only_to_onprem_models(self):
        client, completions = recording_client()
        disable_thinking(client)
        client.chat.completions.create(model="Qwen3_6-35B-A3B", messages=[])
        client.chat.completions.create(model="gpt-6-luna", messages=[])
        client.chat.completions.create(model="gemini-3.5-flash-lite", messages=[])
        assert completions.calls[0]["extra_body"] == THINKING_OFF
        assert completions.calls[1]["extra_body"] == {"reasoning_effort": "none"}
        assert completions.calls[2]["extra_body"] == {"reasoning_effort": "minimal"}

    def test_the_body_per_model(self):
        assert thinking_body("Qwen3_8-27B") == THINKING_OFF
        assert thinking_body("gpt-6-luna") == {"reasoning_effort": "none"}
        assert thinking_body("gpt-6-astra") == {"reasoning_effort": "low"}
        assert thinking_body("claude-sonnet-5-5") == {
            "thinking": {"type": "between_tools"}
        }

    def test_the_body_folds_a_top_level_switch_in_with_the_extra_body(self):
        assert thinking_body("claude-opus-5-5") == {
            "thinking": {"type": "adaptive"},
            "reasoning_effort": "low",
        }


class TestPerModelProfiles:
    """The values each model accepted through the proxy's chat completions."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-haiku-5-5", {"extra_body": {"thinking": {"type": "disabled"}}}),
            (
                "claude-sonnet-5-5",
                {"extra_body": {"thinking": {"type": "between_tools"}}},
            ),
            (
                "claude-opus-5-5",
                {
                    "reasoning_effort": "low",
                    "extra_body": {"thinking": {"type": "adaptive"}},
                },
            ),
            ("gpt-6-luna", {"reasoning_effort": "none"}),
            ("gpt-6.1-sol", {"reasoning_effort": "low"}),
            ("gpt-6-astra", {"reasoning_effort": "low"}),
        ],
    )
    def test_each_model_gets_the_switch_it_accepted(self, model, expected):
        params = request_params(model, max_tokens=16)
        sent = {
            k: v for k, v in params.items() if k in ("reasoning_effort", "extra_body")
        }
        assert sent == expected
        assert models.profile_for(model).confirmed

    def test_openai_models_keep_the_family_parameters(self):
        params = request_params("gpt-6.1-sol", max_tokens=16)
        assert params["max_completion_tokens"] == 16
        assert "temperature" not in params

    def test_a_dated_snapshot_takes_its_undated_entry(self):
        assert (
            models.profile_for("claude-sonnet-5-5-20260928")
            is models.MODEL_PROFILES["claude-sonnet-5-5"]
        )

    def test_an_unlisted_model_takes_its_family_profile_unconfirmed(self):
        profile = models.profile_for("claude-sonnet-4-6")
        assert profile is models.REQUEST_PROFILES["anthropic"]
        assert not profile.confirmed
        assert not models.profile_for("gemini-3.5-flash-lite").confirmed

    def test_every_default_and_frontier_paid_model_outside_google_is_confirmed(self):
        paid = {
            model
            for family in ("anthropic", "openai")
            for model in [
                *DEFAULT_MODELS[family].values(),
                models.FRONTIER_MODELS[family],
            ]
            if not model.startswith("gemini-")
        }
        assert all(models.profile_for(model).confirmed for model in paid)

    def test_only_anthropic_marks_its_cache(self):
        marked = {
            family
            for family, profile in models.REQUEST_PROFILES.items()
            if profile.cache_control
        }
        assert marked == {"anthropic"}
        assert all(
            profile.cache_control
            for name, profile in models.MODEL_PROFILES.items()
            if name.startswith("claude-")
        )


class TestOneReportPerCall:
    def test_a_completer_over_a_governed_client_counts_each_call_once(self, tmp_path):
        from openai.types.chat import ChatCompletion

        from aieng.advanced_rag.graphrag.proxy import RateLimiter, ResponseCache, govern

        def create(**kwargs):
            return ChatCompletion.model_validate(
                {
                    "id": "c",
                    "created": 0,
                    "model": kwargs["model"],
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 1,
                        "total_tokens": 8,
                    },
                }
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        governed = govern(
            client,
            cache=ResponseCache(tmp_path / "c.sqlite"),
            limiter=RateLimiter(1000),
        )
        meter = UsageMeter()
        complete = ChatCompleter(governed, "Qwen3_6-35B-A3B")
        with meter.question("q"):
            complete("p")
            complete("p")
        usage = meter.by_model()["Qwen3_6-35B-A3B"]
        assert (usage.calls, usage.cache_hits, usage.prompt_tokens) == (1, 1, 7)
        assert meter.totals().calls == 1


#: The request atlas-rag's HippoRAG 2 filter sends, whatever the model.
ATLAS_FILTER = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
    "timeout": 120,
}


class TestAdaptRequest:
    @pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5-5"])
    def test_claude_loses_temperature_and_json_object(self, model):
        assert adapt_request(model, ATLAS_FILTER) == {
            "max_tokens": 4096,
            "timeout": 120,
        }

    def test_an_older_claude_keeps_its_temperature(self):
        assert adapt_request("claude-haiku-4-5", ATLAS_FILTER)["temperature"] == 0.0

    def test_claude_keeps_an_explicit_schema(self):
        schema = {"type": "json_schema", "json_schema": {"name": "r", "schema": {}}}
        sent = adapt_request("claude-haiku-5-5", {"response_format": schema})
        assert sent["response_format"] == schema

    def test_openai_gets_its_completion_cap_name(self):
        sent = adapt_request("gpt-6-luna", ATLAS_FILTER)
        assert "max_tokens" not in sent and sent["max_completion_tokens"] == 4096
        assert "temperature" not in sent
        assert sent["response_format"] == {"type": "json_object"}

    def test_an_explicit_completion_cap_wins_over_the_renamed_one(self):
        sent = adapt_request(
            "gpt-6-luna", {"max_tokens": 10, "max_completion_tokens": 20}
        )
        assert sent["max_completion_tokens"] == 20

    def test_google_keeps_temperature_and_loses_additional_properties(self):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "object", "additionalProperties": False}
            },
        }
        sent = adapt_request(
            "gemini-3.5-flash-lite", {**ATLAS_FILTER, "response_format": schema}
        )
        assert sent["temperature"] == 0.0 and sent["max_tokens"] == 4096
        assert sent["response_format"]["json_schema"]["schema"] == {"type": "object"}

    @pytest.mark.parametrize("model", ["Qwen3_6-35B-A3B", "my-local-model"])
    def test_on_prem_and_unknown_models_are_unchanged(self, model):
        body = {**ATLAS_FILTER, "extra_body": {"chat_template_kwargs": {"x": 1}}}
        assert adapt_request(model, body) == body

    def test_paid_models_lose_the_vllm_switch_but_keep_other_body_fields(self):
        body = {"extra_body": {"chat_template_kwargs": {"x": 1}, "user_tag": "t"}}
        assert adapt_request("gpt-6-luna", body) == {"extra_body": {"user_tag": "t"}}

    def test_the_input_is_not_changed(self):
        original = dict(ATLAS_FILTER)
        adapt_request("claude-haiku-5-5", original)
        assert original == ATLAS_FILTER

    def test_a_patched_client_sends_claude_the_adapted_request(self):
        calls = []
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: calls.append(kw))
            )
        )
        disable_thinking(client).chat.completions.create(
            model="claude-haiku-5-5", messages=[], **ATLAS_FILTER
        )
        (sent,) = calls
        assert "temperature" not in sent and "response_format" not in sent
        assert sent["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_a_patched_client_sends_on_prem_the_request_unchanged(self):
        calls = []
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: calls.append(kw))
            )
        )
        disable_thinking(client).chat.completions.create(
            model="Qwen3_6-35B-A3B", messages=[], **ATLAS_FILTER
        )
        (sent,) = calls
        assert {key: sent[key] for key in ATLAS_FILTER} == ATLAS_FILTER
        assert sent["extra_body"] == THINKING_OFF
