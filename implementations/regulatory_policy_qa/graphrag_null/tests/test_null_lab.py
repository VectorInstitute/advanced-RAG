"""Tests for the GraphRAG null-arm measurement pieces.

Covers the accounting that every cost figure in the memo comes from, the span
containment rule that makes a GraphRAG arm scoreable at all, and the controls
the other rows are read against. Nothing here touches a network or a model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "aieng-advanced-rag"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import litellm  # noqa: E402
import pytest  # noqa: E402
from aieng.advanced_rag.batch import ProxyBatch  # noqa: E402
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.provenance import Provenance  # noqa: E402
from null_lab import (  # noqa: E402
    DEFAULT_MODEL,
    Meter,
    MeteredCompleter,
    QueryOutcome,
    ResponseCache,
    answerable_cost,
    completion_key,
    constant_outcomes,
    evidence_covers_spans,
    evidence_from_dict,
    evidence_to_dict,
    load_api_key,
    load_evidence,
    normalise,
    null_report,
    proxy_batch,
    save_evidence,
    table_retrieve,
)


GOLD = (
    "Recipient shall not disclose the Confidential Information to any third "
    "party without the prior written consent of the Discloser."
)


def passage(text: str, chunk_id: str = "d1::0000", doc_id: str = "d1") -> Evidence:
    """Build one piece of evidence."""
    return Evidence(text=text, provenance=Provenance(doc_id=doc_id, chunk_id=chunk_id), score=1.0)


def outcome(query_id: str, is_null: bool, abstained: bool, **kwargs: Any) -> QueryOutcome:
    """Build one query outcome, defaulting everything the test does not set."""
    defaults: dict[str, Any] = {
        "doc_id": query_id.split("::", maxsplit=1)[0],
        "reason": "",
        "verdict": "sufficient",
        "score": 1.0,
        "retrieved": 5,
        "evidence_chars": 500,
        "span_reached": not is_null,
        "answer": "text",
    }
    defaults.update(kwargs)
    return QueryOutcome(query_id=query_id, is_null=is_null, abstained=abstained, **defaults)


class FakeLiteLLM:
    """Stands in for ``litellm.acompletion``: replays a queue of replies or errors."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return litellm.ModelResponse(
            id="r1",
            model=kwargs["model"].split("/", 1)[1],
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a factory for a proxy batch whose calls reach a fake LiteLLM."""
    from infermesh import LMClient  # noqa: PLC0415

    monkeypatch.setattr(LMClient, "_compute_retry_wait", staticmethod(lambda *_: 0.0))

    def make(replies: list[Any], cache: ResponseCache | None = None) -> tuple[ProxyBatch, FakeLiteLLM]:
        fake = FakeLiteLLM(replies)
        monkeypatch.setattr(litellm, "acompletion", fake)
        batch = proxy_batch(DEFAULT_MODEL, cache, api_key="test-key", base_url="http://proxy.test/v1")
        return batch, fake

    return make


def http_error(kind: type[Exception], status: int) -> Exception:
    """Return a LiteLLM error carrying ``status``."""
    response = httpx.Response(status, request=httpx.Request("POST", "http://proxy.test/v1"))
    return kind(message=f"HTTP {status}", model=DEFAULT_MODEL, llm_provider="openai", response=response)


def test_meter_separates_calls_from_cache_hits() -> None:
    meter = Meter(phase="index")
    meter.record(100, 20, 1.5)
    meter.record(50, 10, 0.5)
    meter.record_cache_hit()
    meter.phase = "query"
    meter.record(10, 2, 0.1)
    meter.record_empty()

    assert meter.calls == 3
    assert meter.total_tokens == 192
    assert meter.cache_hits == 1
    assert meter.phase_totals("index")["calls"] == 2
    assert meter.phase_totals("index")["cache_hits"] == 1
    assert meter.phase_totals("query")["total_tokens"] == 12
    assert meter.phase_totals("index")["call_seconds"] == 2.0
    assert meter.phase_totals("never-ran")["calls"] == 0


def test_response_cache_round_trips_on_the_full_request(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    first = ResponseCache.key({"model": "m", "prompt": "a"})
    second = ResponseCache.key({"model": "m", "prompt": "b"})

    assert first != second
    assert cache.get(first) is None
    cache.put(first, "answer", 10, 2)
    assert cache.get(first) == "answer"
    assert cache.get(second) is None


def test_completer_meters_a_call_and_serves_the_repeat_from_cache(tmp_path: Path, proxy: Any) -> None:
    batch, fake = proxy(["hello"], ResponseCache(tmp_path / "c.sqlite"))
    meter = Meter(phase="pipeline")
    completer = MeteredCompleter(batch, meter=meter)

    assert completer("say hello") == "hello"
    assert completer("say hello") == "hello"
    assert len(fake.requests) == 1
    assert meter.calls == 1
    assert meter.cache_hits == 1
    assert meter.total_tokens == 10


def test_completer_disables_reasoning_on_every_call(proxy: Any) -> None:
    batch, fake = proxy(["ok"])
    MeteredCompleter(batch, meter=Meter())("prompt")

    assert fake.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_cache_key_keeps_the_shape_existing_caches_were_written_with() -> None:
    expected = ResponseCache.key(
        {"model": DEFAULT_MODEL, "prompt": "p", "max_tokens": 512, "temperature": 0.0, "thinking": False}
    )
    key = completion_key(DEFAULT_MODEL)([{"role": "user", "content": "p"}], {"max_tokens": 512, "temperature": 0.0})
    assert key == expected


def test_a_cache_written_before_the_port_answers_the_completer(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    key = ResponseCache.key(
        {"model": DEFAULT_MODEL, "prompt": "p", "max_tokens": 512, "temperature": 0.0, "thinking": False}
    )
    cache.put(key, "recorded", 4, 1)
    replay = MeteredCompleter(proxy_batch(DEFAULT_MODEL, cache, cache_only=True))
    assert replay("p") == "recorded"
    assert replay.meter.cache_hits == 1 and replay.meter.calls == 0


def test_cache_only_needs_a_cache() -> None:
    with pytest.raises(ValueError, match="requires a response cache"):
        proxy_batch(DEFAULT_MODEL, None, cache_only=True)


def test_empty_completions_cannot_be_scored_as_abstentions(tmp_path: Path, proxy: Any) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    batch, _ = proxy([None], cache)
    meter = Meter()

    with pytest.raises(RuntimeError, match="empty completion"):
        MeteredCompleter(batch, meter=meter)("prompt")
    assert meter.empty_responses == 1
    assert len(cache) == 0


def test_cache_only_miss_never_calls_a_model(tmp_path) -> None:
    complete = MeteredCompleter(proxy_batch(DEFAULT_MODEL, ResponseCache(tmp_path / "cache.sqlite"), cache_only=True))
    with pytest.raises(FileNotFoundError, match="cache-only"):
        complete("uncached")


def test_a_rate_limit_is_retried_and_the_answer_cached(tmp_path: Path, proxy: Any) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    batch, fake = proxy([http_error(litellm.RateLimitError, 429), "valid"], cache)
    meter = Meter()
    assert MeteredCompleter(batch, meter=meter)("prompt") == "valid"
    assert MeteredCompleter(proxy_batch(DEFAULT_MODEL, cache, cache_only=True))("prompt") == "valid"
    assert len(fake.requests) == 2
    assert meter.calls == 1


def test_a_rejected_request_fails_the_run_at_once_instead_of_abstaining(proxy: Any) -> None:
    batch, fake = proxy([http_error(litellm.BadRequestError, 400), "unused"])
    meter = Meter()

    with pytest.raises(litellm.BadRequestError):
        MeteredCompleter(batch, meter=meter)("prompt")
    assert len(fake.requests) == 1
    assert meter.calls == 0


def test_a_gold_span_is_found_however_the_passage_is_whitespaced() -> None:
    reflowed = GOLD.replace(" ", "\n  ")
    assert evidence_covers_spans([passage(f"...{reflowed}...")], [GOLD])


def test_a_span_the_retriever_missed_is_not_found() -> None:
    assert not evidence_covers_spans([passage("Unrelated boilerplate.")], [GOLD])
    assert not evidence_covers_spans([], [GOLD])


def test_a_short_span_is_ignored_because_contract_language_repeats_it() -> None:
    assert not evidence_covers_spans([passage("the Receiving Party may")], ["the Party"])


def test_null_report_and_answerable_cost_price_the_same_arm_both_ways() -> None:
    outcomes = [
        outcome("d1::a", is_null=True, abstained=True),
        outcome("d1::b", is_null=True, abstained=False),
        outcome("d1::c", is_null=False, abstained=False),
        outcome("d1::d", is_null=False, abstained=True),
    ]
    report = null_report(outcomes)
    cost = answerable_cost(outcomes)

    assert report.null_queries == 2
    assert report.answerable_queries == 2
    assert report.abstain_recall == 0.5
    assert report.abstain_precision == 0.5
    assert report.false_answer_rate == 0.5
    assert cost.abstention_rate == 0.5
    assert cost.span_reach_rate == 1.0
    assert cost.grounded_answer_rate == 0.5


def test_always_abstain_scores_the_null_rate_and_answers_nothing() -> None:
    outcomes = [
        outcome("d1::a", is_null=True, abstained=False),
        outcome("d1::b", is_null=True, abstained=False),
        outcome("d1::c", is_null=False, abstained=False),
    ]
    always = constant_outcomes(outcomes, abstained=True)
    report = null_report(always)
    cost = answerable_cost(always)

    assert report.abstain_recall == 1.0
    assert report.abstain_precision == pytest.approx(2 / 3, abs=1e-4)
    assert report.false_answer_rate == 0.0
    assert cost.abstention_rate == 1.0
    assert cost.grounded_answer_rate == 0.0


def test_always_answer_scores_zero_on_every_abstention_measure() -> None:
    outcomes = [
        outcome("d1::a", is_null=True, abstained=True),
        outcome("d1::c", is_null=False, abstained=True),
    ]
    report = null_report(constant_outcomes(outcomes, abstained=False))

    assert report.abstain_recall == 0.0
    assert report.abstain_precision == 0.0
    assert report.false_answer_rate == 1.0


def test_evidence_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    table = {"d1::a": [passage("first"), passage("second", chunk_id="d1::0001")]}
    path = tmp_path / "arm.json"
    save_evidence(path, table)
    restored = load_evidence(path)

    assert [item.text for item in restored["d1::a"]] == ["first", "second"]
    assert restored["d1::a"][1].provenance.chunk_id == "d1::0001"
    assert evidence_from_dict(evidence_to_dict(passage("x"))).text == "x"


def test_the_retrieval_table_is_reached_through_scope_and_question() -> None:
    table = {"d1::a": [passage("evidence")]}
    query_ids = {("d1", "Is it stated?"): "d1::a"}
    retrieve = table_retrieve(table, query_ids)

    assert len(retrieve("Is it stated?", "d1")) == 1
    assert retrieve("Is it stated?", "d2") == []
    assert retrieve("A question nobody asked", "d1") == []


@pytest.fixture
def no_exported_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_the_api_key_is_read_past_the_spaces_the_repo_env_writes(tmp_path: Path, no_exported_key: None) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nVECTOR_PROXY_API_KEY = 'sk-example'\n")

    assert load_api_key(env) == "sk-example"


def test_a_generic_process_openai_key_is_not_sent_to_the_vector_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_exported_key: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "belongs-to-another-provider")
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")

    with pytest.raises(RuntimeError, match="VECTOR_PROXY_API_KEY"):
        load_api_key(env)


def test_an_exported_platform_key_wins_over_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_exported_key: None
) -> None:
    env = tmp_path / ".env"
    env.write_text("VECTOR_PROXY_API_KEY=stale\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "exported")

    assert load_api_key(env) == "exported"


def test_a_missing_env_file_is_not_an_error_when_the_key_is_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_exported_key: None
) -> None:
    monkeypatch.setenv("VECTOR_PROXY_API_KEY", "exported")

    assert load_api_key(tmp_path / "absent.env") == "exported"


def test_a_env_file_with_no_key_says_so(tmp_path: Path, no_exported_key: None) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")

    with pytest.raises(RuntimeError, match="VECTOR_PROXY_API_KEY"):
        load_api_key(env)


def test_normalise_collapses_whitespace_and_case() -> None:
    assert normalise("  A  \n B ") == "a b"
    assert normalise("") == ""
