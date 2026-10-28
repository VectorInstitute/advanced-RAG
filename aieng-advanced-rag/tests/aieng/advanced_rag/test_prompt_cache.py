"""Tests for prompt-cache tokens: reading them, metering them, and pricing them.

The usage payloads are the ones the proxy's ``/v1/chat/completions`` returned to
the OpenAI SDK, copied field for field.
"""

from __future__ import annotations

import copy

import pytest
from openai.types.chat import ChatCompletion

from aieng.advanced_rag.metering import CallUsage, UsageMeter, usage_of
from aieng.advanced_rag.models import cacheable_system
from aieng.advanced_rag.spend import OFFLINE_PRICES, price, usage_line


# Two Anthropic calls with the same list-form system message carrying
# cache_control. prompt_tokens and total_tokens leave the cache tokens out.
HAIKU_WRITE = {
    "completion_tokens": 5,
    "prompt_tokens": 11,
    "total_tokens": 16,
    "completion_tokens_details": None,
    "prompt_tokens_details": None,
    "cache_creation_input_tokens": 18211,
    "cache_creation_1h_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
HAIKU_READ = {**HAIKU_WRITE, "cache_creation_input_tokens": 0}
HAIKU_READ["cache_read_input_tokens"] = 18211

# The same with ``"ttl": "1h"``: the write lands in its own field.
HAIKU_WRITE_1H = {
    **HAIKU_WRITE,
    "cache_creation_input_tokens": 0,
    "cache_creation_1h_input_tokens": 17511,
}

# The pair from the first measurement: a 10,803-token system prompt, with only
# 19 prompt tokens reported on each call.
ANTHROPIC_WRITE_10803 = {
    "prompt_tokens": 19,
    "completion_tokens": 5,
    "total_tokens": 24,
    "cache_creation_input_tokens": 10803,
    "cache_read_input_tokens": 0,
}
ANTHROPIC_READ_10803 = {
    "prompt_tokens": 19,
    "completion_tokens": 5,
    "total_tokens": 24,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 10803,
}

# Two OpenAI calls with the same 8,118-token prompt. OpenAI caches without a
# marker and counts the cache inside prompt_tokens.
LUNA_WRITE = {
    "completion_tokens": 5,
    "prompt_tokens": 8118,
    "total_tokens": 8123,
    "completion_tokens_details": {
        "accepted_prediction_tokens": 0,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
    },
    "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 8115,
        "cached_tokens": 0,
    },
}
LUNA_READ = copy.deepcopy(LUNA_WRITE)
LUNA_READ.update(completion_tokens=4, total_tokens=8122)
LUNA_READ["prompt_tokens_details"].update(cache_write_tokens=0, cached_tokens=8115)

# The first OpenAI measurement: 6,818 of 6,821 prompt tokens read from cache.
OPENAI_READ_6818 = {
    "prompt_tokens": 6821,
    "completion_tokens": 2,
    "total_tokens": 6823,
    "prompt_tokens_details": {"cached_tokens": 6818},
}


def sdk_response(usage: dict, model: str) -> ChatCompletion:
    """Return the payload as the OpenAI SDK object the proxy's reply parses into."""
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "OK."},
                }
            ],
            "usage": usage,
        }
    )


class TestReadingAnthropicUsage:
    def test_a_cache_write_is_added_back_into_prompt_tokens(self):
        usage = usage_of({"usage": ANTHROPIC_WRITE_10803})
        assert usage.prompt_tokens == 10822
        assert usage.prompt_cache_write_tokens == 10803
        assert usage.prompt_cache_read_tokens == 0
        assert usage.uncached_prompt_tokens == 19
        assert usage.total_tokens == 10827

    def test_a_cache_read_is_added_back_into_prompt_tokens(self):
        usage = usage_of({"usage": ANTHROPIC_READ_10803})
        assert usage.prompt_tokens == 10822
        assert usage.prompt_cache_read_tokens == 10803
        assert usage.prompt_cache_write_tokens == 0

    def test_the_sdk_object_carries_the_extra_fields_through(self):
        usage = usage_of(sdk_response(HAIKU_READ, "claude-haiku-5-5"))
        assert usage.prompt_tokens == 11 + 18211
        assert usage.prompt_cache_read_tokens == 18211
        assert usage.total_tokens == 16 + 18211

    def test_a_one_hour_write_is_counted_and_kept_apart(self):
        usage = usage_of(sdk_response(HAIKU_WRITE_1H, "claude-haiku-5-5"))
        assert usage.prompt_tokens == 11 + 17511
        assert usage.prompt_cache_write_tokens == 17511
        assert usage.prompt_cache_write_1h_tokens == 17511

    def test_a_call_with_no_cache_activity_is_unchanged(self):
        payload = {**HAIKU_WRITE, "cache_creation_input_tokens": 0}
        usage = usage_of({"usage": payload})
        assert (usage.prompt_tokens, usage.total_tokens) == (11, 16)
        assert usage.uncached_prompt_tokens == 11


# Two Gemini calls with the same 8,595-token prompt, as the proxy reports them.
# Gemini caches without a marker and counts the read inside prompt_tokens, but
# names it as Anthropic does.
GEMINI_READ = {
    "completion_tokens": 1,
    "prompt_tokens": 8595,
    "total_tokens": 8596,
    "completion_tokens_details": {"reasoning_tokens": 0},
    "prompt_tokens_details": None,
    "cache_read_input_tokens": 6117,
    "thinking_tokens": 0,
}
GEMINI_COLD = {**GEMINI_READ, "cache_read_input_tokens": 0}


class TestReadingGeminiUsage:
    def test_a_cache_read_is_a_share_of_prompt_tokens(self):
        usage = usage_of(sdk_response(GEMINI_READ, "gemini-3.5-flash-lite"))
        assert usage.prompt_tokens == 8595
        assert usage.prompt_cache_read_tokens == 6117
        assert usage.uncached_prompt_tokens == 8595 - 6117
        assert usage.total_tokens == 8596

    def test_a_cold_call_reports_no_cache(self):
        usage = usage_of({"usage": GEMINI_COLD})
        assert (usage.prompt_tokens, usage.prompt_cache_read_tokens) == (8595, 0)


class TestReadingOpenAIUsage:
    def test_cached_tokens_are_a_share_of_prompt_tokens(self):
        usage = usage_of({"usage": OPENAI_READ_6818})
        assert usage.prompt_tokens == 6821
        assert usage.prompt_cache_read_tokens == 6818
        assert usage.uncached_prompt_tokens == 3

    def test_a_cache_write_is_read_from_the_details(self):
        usage = usage_of(sdk_response(LUNA_WRITE, "gpt-6-luna"))
        assert usage.prompt_tokens == 8118
        assert usage.prompt_cache_write_tokens == 8115
        assert usage.prompt_cache_read_tokens == 0

    def test_the_read_after_it(self):
        usage = usage_of(sdk_response(LUNA_READ, "gpt-6-luna"))
        assert usage.prompt_tokens == 8118
        assert usage.prompt_cache_read_tokens == 8115
        assert usage.total_tokens == 8122

    def test_null_details_report_no_cache(self):
        usage = usage_of(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": None,
                }
            }
        )
        assert usage == CallUsage(10, 2, 12, reported=True)


class TestMetering:
    def test_the_meter_totals_cache_tokens_per_model_and_per_question(self):
        meter = UsageMeter()
        with meter.question("q1"):
            for payload in (ANTHROPIC_WRITE_10803, ANTHROPIC_READ_10803):
                meter.record(
                    usage_of({"usage": payload}), 0.1, model="claude-haiku-5-5"
                )
        row = meter.by_model()["claude-haiku-5-5"]
        assert row.prompt_tokens == 2 * 10822
        assert row.prompt_cache_write_tokens == 10803
        assert row.prompt_cache_read_tokens == 10803
        question = meter.per_question()["q1"].to_dict()
        assert question["prompt_tokens"] == 2 * 10822
        assert question["prompt_cache_read_tokens"] == 10803
        assert meter.totals().prompt_cache_write_tokens == 10803


class TestPricing:
    def test_haiku_prices_a_write_at_1_25x_and_a_read_at_0_1x_input(self):
        rate = OFFLINE_PRICES["claude-haiku-5-5"]
        write = rate.cost_of(usage_of({"usage": ANTHROPIC_WRITE_10803}))
        read = rate.cost_of(usage_of({"usage": ANTHROPIC_READ_10803}))
        assert write == pytest.approx((19 * 0.10 + 10803 * 0.125 + 5 * 0.50) / 1e6)
        assert read == pytest.approx((19 * 0.10 + 10803 * 0.01 + 5 * 0.50) / 1e6)

    def test_reading_prompt_tokens_as_reported_undercounts_by_hundreds(self):
        usage = usage_of({"usage": ANTHROPIC_READ_10803})
        assert usage.prompt_tokens / ANTHROPIC_READ_10803["prompt_tokens"] > 500

    def test_a_one_hour_write_bills_at_2x_input(self):
        rate = OFFLINE_PRICES["claude-sonnet-5-5"]
        cost = rate.cost_of(usage_of({"usage": HAIKU_WRITE_1H}))
        assert cost == pytest.approx((11 * 2.0 + 17511 * 4.0 + 5 * 10.0) / 1e6)

    def test_sonnet_and_opus_read_at_0_05x_input(self):
        for model in ("claude-sonnet-5-5", "claude-opus-5-5"):
            rate = OFFLINE_PRICES[model]
            assert rate.cached_input_per_mtok == pytest.approx(
                rate.input_per_mtok * 0.05
            )
            assert rate.cache_write_per_mtok == pytest.approx(
                rate.input_per_mtok * 1.25
            )

    def test_openai_reads_and_writes_at_their_rates(self):
        rate = OFFLINE_PRICES["gpt-6-luna"]
        write = rate.cost_of(usage_of({"usage": LUNA_WRITE}))
        read = rate.cost_of(usage_of({"usage": LUNA_READ}))
        assert write == pytest.approx((3 * 0.10 + 8115 * 0.125 + 5 * 0.50) / 1e6)
        assert read == pytest.approx((3 * 0.10 + 8115 * 0.01 + 4 * 0.50) / 1e6)

    def test_google_has_no_write_fee(self):
        rate = OFFLINE_PRICES["gemini-3.5-flash-lite"]
        assert rate.cache_write_per_mtok is None
        assert rate.cost(100, 0, cache_write_tokens=100) == rate.cost(100, 0)

    def test_a_model_with_no_cache_rate_bills_reads_at_the_input_rate(self):
        rate = OFFLINE_PRICES["Qwen3_6-35B-A3B"]
        assert rate.cost(1000, 10, cache_read_tokens=900) == 0.0

    def test_the_estimate_and_the_line_use_the_cache_rates(self):
        meter = UsageMeter()
        for payload in (ANTHROPIC_WRITE_10803, ANTHROPIC_READ_10803):
            meter.record(usage_of({"usage": payload}), 0.1, model="claude-haiku-5-5")
        estimate = price(meter, table=OFFLINE_PRICES)
        expected = (2 * 19 * 0.10 + 10803 * (0.125 + 0.01) + 2 * 5 * 0.50) / 1e6
        assert estimate.by_model["claude-haiku-5-5"] == pytest.approx(expected)
        line = usage_line(meter, table=OFFLINE_PRICES)
        assert "21,644 in (10,803 read from and 10,803 written to" in line


class TestCacheableSystem:
    def test_anthropic_gets_list_form_content_with_one_breakpoint(self):
        message = cacheable_system("claude-haiku-5-5", "Rules.", "<document>")
        assert message == {
            "role": "system",
            "content": [
                {"type": "text", "text": "Rules."},
                {
                    "type": "text",
                    "text": "<document>",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }

    @pytest.mark.parametrize(
        "model", ["gpt-6-luna", "gemini-3.5-flash-lite", "Qwen3_6-35B-A3B", "x-1"]
    )
    def test_other_families_get_a_plain_string(self, model):
        assert cacheable_system(model, "Rules.", "", "<document>") == {
            "role": "system",
            "content": "Rules.\n\n<document>",
        }

    def test_each_call_gets_its_own_marker(self):
        first = cacheable_system("claude-opus-5-5", "Rules.")
        first["content"][0]["cache_control"]["ttl"] = "1h"
        second = cacheable_system("claude-opus-5-5", "Rules.")
        assert second["content"][0]["cache_control"] == {"type": "ephemeral"}
