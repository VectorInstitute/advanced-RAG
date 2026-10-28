"""Tests for pricing, the proxy's usage routes, and the budget guards, all offline."""

from __future__ import annotations

import urllib.error
import warnings

import pytest

from aieng.advanced_rag import spend
from aieng.advanced_rag.metering import CallUsage, UsageMeter, report_call
from aieng.advanced_rag.models import choose_models
from aieng.advanced_rag.spend import (
    OFFLINE_PRICES,
    Budget,
    BudgetWarning,
    ProxyAccount,
    SpendLimitReached,
    SpendSession,
    budget_warnings,
    describe,
    price,
    price_of,
    price_table,
)


PROXY = "https://proxy.vectorinstitute.ai/v1"


class FakeProxy:
    """Answers the three self-service routes from canned payloads."""

    def __init__(self, *, budget="300.00", spent="0.00", models=None, fail=False):
        self.budget = budget
        self.spent = spent
        self.models = (
            models
            if models is not None
            else [
                {
                    "name": "gpt-6-luna",
                    "cost_per_input_token": "0.00000012",
                    "cost_per_output_token": "0.0000006",
                },
                {
                    "name": "Qwen3_6-35B-A3B",
                    "cost_per_input_token": "0",
                    "cost_per_output_token": "0",
                },
                {
                    "name": "unpriced",
                    "cost_per_input_token": None,
                    "cost_per_output_token": None,
                },
            ]
        )
        self.fail = fail
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url, headers):
        self.requests.append((url, dict(headers)))
        if self.fail:
            raise urllib.error.URLError("offline")
        if "/admin/me/models" in url:
            return self.models
        if "/admin/me/key-budget" in url:
            return {"budget_usd": self.budget}
        if "/admin/me/key-usage" in url:
            return {"summary": {"total_cost_usd": self.spent}, "by_model": []}
        raise AssertionError(url)


def account_for(fake: FakeProxy | None, key: str | None = "test-key") -> ProxyAccount:
    return ProxyAccount(key, PROXY, fetch=fake)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # No test here reaches the network, whatever the environment holds.
    spend.use_account(ProxyAccount(None, PROXY))
    monkeypatch.delenv(spend.SPEND_LIMIT_ENV, raising=False)
    monkeypatch.delenv(spend.KEY_BUDGET_ENV, raising=False)
    yield
    spend.use_account(None)
    spend.reset_session()


def meter_with(*calls: tuple[str, int, int]) -> UsageMeter:
    meter = UsageMeter()
    for model, prompt, completion in calls:
        meter.record(CallUsage(prompt, completion, reported=True), 0.0, model=model)
    return meter


class TestPrices:
    def test_the_offline_table_has_the_on_prem_models_at_zero(self):
        assert OFFLINE_PRICES["Qwen3_6-35B-A3B"].free
        assert OFFLINE_PRICES["Qwen3_8-27B"].free

    def test_the_offline_table_carries_the_verified_rates(self):
        haiku = OFFLINE_PRICES["claude-haiku-5-5"]
        assert (
            haiku.input_per_mtok,
            haiku.output_per_mtok,
            haiku.cached_input_per_mtok,
        ) == (
            0.10,
            0.50,
            0.01,
        )
        assert OFFLINE_PRICES["gpt-6-astra"].output_per_mtok == 50.0
        assert "gemini-3.1-flash-lite-preview" not in OFFLINE_PRICES

    def test_proxy_rates_replace_the_offline_ones(self):
        table = price_table(account_for(FakeProxy()))
        luna = table["gpt-6-luna"]
        assert luna.source == "proxy"
        assert luna.input_per_mtok == pytest.approx(0.12)
        # The offline note survives, because the proxy does not report it.
        assert "272K" in luna.note
        assert table["claude-haiku-5-5"].source == "offline"

    def test_a_model_the_proxy_lists_without_a_rate_stays_unpriced(self):
        assert "unpriced" not in price_table(account_for(FakeProxy()))

    def test_an_unreachable_proxy_falls_back_to_the_offline_table(self):
        fake = FakeProxy(fail=True)
        source = account_for(fake)
        assert price_table(source) == dict(OFFLINE_PRICES)
        assert source.errors and "URLError" in source.errors[0]

    def test_no_key_means_no_request(self):
        fake = FakeProxy()
        assert account_for(fake, key=None).models() is None
        assert fake.requests == []

    def test_an_endpoint_that_is_not_the_proxy_is_never_asked(self):
        source = ProxyAccount("k", "https://api.example.com/v1")
        assert not source.reachable
        assert source.key_budget() is None

    def test_the_admin_routes_sit_beside_v1(self):
        fake = FakeProxy()
        account_for(fake).key_budget()
        url, headers = fake.requests[0]
        assert url == "https://proxy.vectorinstitute.ai/admin/me/key-budget"
        assert headers["Authorization"] == "Bearer test-key"

    def test_a_dated_snapshot_prices_as_its_alias(self):
        assert (
            price_of("claude-haiku-4-5-20251001", OFFLINE_PRICES)
            is OFFLINE_PRICES["claude-haiku-4-5"]
        )


class TestPrice:
    def test_per_model_totals_are_priced_at_their_rates(self):
        meter = meter_with(
            ("gpt-6-luna", 1_000_000, 100_000), ("Qwen3_6-35B-A3B", 5, 5)
        )
        estimate = price(meter, table=OFFLINE_PRICES)
        assert estimate.by_model["gpt-6-luna"] == pytest.approx(0.10 + 0.05)
        assert estimate.by_model["Qwen3_6-35B-A3B"] == 0.0
        assert estimate.total == pytest.approx(0.15)

    def test_an_unknown_model_is_unpriced_not_free(self):
        estimate = price(meter_with(("mystery", 10, 10)), table=OFFLINE_PRICES)
        assert estimate.by_model["mystery"] is None
        assert estimate.unpriced == ("mystery",)
        assert "no price for mystery" in estimate.line()

    def test_cache_hits_price_what_a_cold_run_would_have_cost(self):
        meter = UsageMeter()
        meter.record_cache_hit(
            model="gpt-6-luna", usage=CallUsage(2_000_000, 0, reported=True)
        )
        assert price(meter, table=OFFLINE_PRICES).total == 0.0
        assert price(
            meter, table=OFFLINE_PRICES, include_cached=True
        ).total == pytest.approx(0.2)

    def test_the_cell_line_names_calls_tokens_models_and_dollars(self):
        meter = meter_with(("gemini-3.5-flash-lite", 434_921, 18_000))
        line = spend.usage_line(meter, table=OFFLINE_PRICES)
        assert line.startswith(
            "1 calls, 434,921 in + 18,000 out tokens on gemini-3.5-flash-lite"
        )
        assert "est. $0.18" in line

    def test_the_cell_line_says_what_the_cache_saved(self):
        meter = UsageMeter()
        meter.record_cache_hit(
            model="gpt-6-luna", usage=CallUsage(1_000_000, 0, reported=True)
        )
        assert "would have cost $0.10" in spend.usage_line(meter, table=OFFLINE_PRICES)


class TestBudget:
    def test_the_budget_line(self):
        assert Budget(300.0, 41.12).line() == (
            "key budget $300.00, spent $41.12, left $258.88 (86%)"
        )
        assert "on-prem models only" in Budget(0.0, 0.0).line()
        assert "unknown" in Budget(None, None).line()

    @pytest.mark.parametrize(
        ("spent", "expected"),
        [(100.0, []), (150.0, ["50%"]), (240.0, ["80%"]), (299.0, ["80%"])],
    )
    def test_warnings_at_half_and_four_fifths(self, spent, expected):
        messages = budget_warnings(Budget(300.0, spent))
        assert [
            m for m in expected if any(m + " mark" in x for x in messages)
        ] == expected
        assert len(messages) == len(expected)

    def test_a_session_warns_once_per_threshold(self):
        seen: set[float] = set()
        assert budget_warnings(Budget(300.0, 160.0), seen)
        assert not budget_warnings(Budget(300.0, 170.0), seen)
        assert budget_warnings(Budget(300.0, 250.0), seen)
        assert not budget_warnings(Budget(300.0, 260.0), seen)

    def test_a_zero_budget_never_warns(self):
        assert budget_warnings(Budget(0.0, 0.0)) == []


class TestAssumedBudget:
    def test_a_reported_budget_is_kept(self):
        budget = Budget(300.0, 41.12)
        assert budget.assuming() is budget
        assert not budget.assuming().assumed

    def test_a_missing_budget_is_assumed_at_three_hundred(self):
        budget = Budget(None, 41.12).assuming()
        assert budget.assumed
        assert budget.budget_usd == spend.ASSUMED_BUDGET_USD == 300.0
        assert budget.left_usd == pytest.approx(258.88)
        assert budget.line() == (
            "key budget not set on the key; assuming $300.00 (set "
            "ADVANCED_RAG_KEY_BUDGET_USD to override), spent $41.12, left $258.88"
        )

    def test_a_silent_proxy_counts_nothing_spent_against_the_assumption(self):
        budget = Budget(None, None).assuming()
        assert budget.left_usd == 300.0
        assert budget.fraction_spent is None
        assert budget.line().startswith("key budget unknown (the proxy did not answer)")
        assert "assuming $300.00" in budget.line()

    def test_the_variable_overrides_the_assumption(self, monkeypatch):
        monkeypatch.setenv(spend.KEY_BUDGET_ENV, "120")
        assert Budget(None, 0.0).assuming().budget_usd == 120.0
        assert Budget(300.0, 0.0).assuming().budget_usd == 300.0

    def test_a_malformed_variable_is_refused(self, monkeypatch):
        monkeypatch.setenv(spend.KEY_BUDGET_ENV, "lots")
        with pytest.raises(ValueError, match=spend.KEY_BUDGET_ENV):
            Budget(None, 0.0).assuming()

    def test_warnings_fire_against_the_assumed_budget_and_say_so(self):
        messages = budget_warnings(Budget(None, 160.0).assuming())
        assert len(messages) == 1
        assert "53% of an assumed $300.00 budget" in messages[0]
        assert spend.KEY_BUDGET_ENV in messages[0]
        assert "50% mark" in messages[0]

    def test_read_budget_assumes_for_a_key_without_one(self):
        source = account_for(FakeProxy(budget=None, spent="5.00"))
        budget = spend.read_budget(source)
        assert budget.assumed and budget.left_usd == pytest.approx(295.0)
        assert not spend.read_budget(account_for(FakeProxy())).assumed

    def test_describe_warns_past_half_an_assumed_budget(self):
        with pytest.warns(BudgetWarning, match="assumed"):
            text = describe(
                choose_models("onprem"),
                source=account_for(FakeProxy(budget=None, spent="200")),
            )
        assert "assuming $300.00" in text

    def test_the_report_warns_past_four_fifths_of_an_assumed_budget(self):
        session = SpendSession(
            UsageMeter(),
            source=account_for(FakeProxy(budget=None, spent="250")),
            table=OFFLINE_PRICES,
        )
        with pytest.warns(BudgetWarning, match="80% mark"):
            text = session.report()
        assert "assuming $300.00" in text


class TestDescribe:
    def test_the_set_is_printed_with_prices_and_the_budget(self):
        text = describe(
            choose_models("openai"),
            source=account_for(FakeProxy(spent="41.12")),
        )
        assert "family openai" in text
        assert "gpt-6-luna" in text and "$0.12 in / $0.60 out" in text
        assert "embeddings  BAAI/bge-small-en-v1.5" in text
        assert "left $258.88" in text

    def test_it_warns_past_half_the_budget(self):
        with pytest.warns(BudgetWarning, match="50%"):
            describe(
                choose_models("onprem"), source=account_for(FakeProxy(spent="200"))
            )

    def test_it_degrades_to_offline_prices_when_the_proxy_is_silent(self):
        text = describe(
            choose_models("google"), source=account_for(FakeProxy(fail=True))
        )
        assert "$0.30 in / $2.50 out" in text
        assert "key budget unknown" in text

    def test_an_unlisted_model_is_shown_as_unknown(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = choose_models("openai", generation="gpt-9-experimental")
        text = describe(models, source=account_for(None, key=None))
        assert "price unknown" in text


class TestSession:
    def test_the_report_puts_the_estimate_beside_the_proxy_figure(self):
        fake = FakeProxy(spent="10.00")
        meter = meter_with(("gpt-6-luna", 1_000_000, 0))
        session = SpendSession(meter, source=account_for(fake), table=OFFLINE_PRICES)
        fake.spent = "10.25"
        text = session.report()
        assert "this run, estimated   est. $0.10 over 1 model" in text
        assert "this run, proxy usage $0.25 (team key" in text
        assert "key budget $300.00, spent $10.25" in text

    def test_the_report_is_the_estimate_alone_without_the_proxy(self):
        session = SpendSession(
            meter_with(("gpt-6-luna", 10, 10)),
            source=account_for(None, key=None),
            table=OFFLINE_PRICES,
        )
        assert session.report().splitlines() == [
            "this run, estimated   est. $0.00 over 1 model"
        ]

    def test_the_report_warns_past_four_fifths(self):
        fake = FakeProxy(spent="250")
        session = SpendSession(
            UsageMeter(), source=account_for(fake), table=OFFLINE_PRICES
        )
        with pytest.warns(BudgetWarning, match="80%"):
            session.report()

    def test_the_ceiling_stops_the_run_at_the_next_call(self):
        meter = UsageMeter()
        with SpendSession(
            meter,
            limit_usd=0.15,
            source=account_for(None, key=None),
            table=OFFLINE_PRICES,
        ):
            report_call(CallUsage(1_000_000, 0, reported=True), 0.0, model="gpt-6-luna")
            with pytest.raises(SpendLimitReached, match=r"\$0.20"):
                report_call(
                    CallUsage(1_000_000, 0, reported=True), 0.0, model="gpt-6-luna"
                )
        # Closed: the ceiling no longer applies.
        report_call(CallUsage(10_000_000, 0, reported=True), 0.0, model="gpt-6-luna")

    def test_the_ceiling_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(spend.SPEND_LIMIT_ENV, "0.05")
        with SpendSession(
            UsageMeter(), source=account_for(None, key=None), table=OFFLINE_PRICES
        ) as session:
            assert session.limit_usd == 0.05
            with pytest.raises(SpendLimitReached):
                report_call(
                    CallUsage(1_000_000, 0, reported=True), 0.0, model="gpt-6-luna"
                )

    def test_free_models_never_reach_the_ceiling(self):
        with SpendSession(
            UsageMeter(),
            limit_usd=0.0001,
            source=account_for(None, key=None),
            table=OFFLINE_PRICES,
        ):
            report_call(
                CallUsage(10_000_000, 10_000_000, reported=True),
                0.0,
                model="Qwen3_8-27B",
            )

    def test_a_malformed_ceiling_is_refused(self, monkeypatch):
        monkeypatch.setenv(spend.SPEND_LIMIT_ENV, "ten dollars")
        with pytest.raises(ValueError, match=spend.SPEND_LIMIT_ENV):
            SpendSession(UsageMeter(), source=account_for(None, key=None))

    def test_the_module_report_uses_one_shared_session(self):
        assert spend.session() is spend.session()
        assert spend.report().startswith("this run, estimated")


def test_the_model_cell_prints_the_set_and_opens_the_session(capsys):
    choose_models("onprem", show=True)
    printed = capsys.readouterr().out
    assert "family onprem" in printed
    assert "$0 (Vector cluster)" in printed
    assert spend._SESSION.session is not None


class _FakeCompletions:
    """Answers every request with a million prompt tokens and counts the requests."""

    def __init__(self) -> None:
        self.sent = 0

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion

        self.sent += 1
        return ChatCompletion.model_validate(
            {
                "id": f"fake-{self.sent}",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                    "total_tokens": 1_000_000,
                },
            }
        )


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class TestCeilingBeforeTheCall:
    """Calls queued after the ceiling is passed are refused, not sent and billed."""

    def _session(self) -> SpendSession:
        return SpendSession(
            UsageMeter(),
            limit_usd=0.15,
            source=account_for(None, key=None),
            table=OFFLINE_PRICES,
        )

    def test_a_governed_client_stops_sending_past_the_ceiling(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        from aieng.advanced_rag.graphrag.proxy import RateLimiter, ResponseCache, govern

        fake = _FakeClient()
        client = govern(
            fake,
            cache=ResponseCache(tmp_path / "cache.sqlite"),
            limiter=RateLimiter(100_000),
        )

        def ask(index: int) -> str:
            response = client.chat.completions.create(
                model="gpt-6-luna",
                messages=[{"role": "user", "content": f"question {index}"}],
            )
            return str(response.choices[0].message.content)

        with self._session(), ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(ask, i) for i in range(20)]
            outcomes = [f.exception() for f in futures]
        # gpt-6-luna is $0.12 per 1M input tokens: the first call stays under
        # $0.15, the second passes it and raises on return, the rest never leave.
        assert fake.completions.sent == 2
        assert outcomes[0] is None
        assert all(isinstance(e, SpendLimitReached) for e in outcomes[1:])
        assert all("was not sent" in str(e) for e in outcomes[2:])

    def test_a_metered_client_stops_sending_past_the_ceiling(self):
        from aieng.advanced_rag.metering import MeteredClient

        fake = _FakeClient()
        client = MeteredClient(fake, UsageMeter())
        with self._session():
            client.chat.completions.create(model="gpt-6-luna", messages=[])
            with pytest.raises(SpendLimitReached):
                client.chat.completions.create(model="gpt-6-luna", messages=[])
            with pytest.raises(SpendLimitReached, match="was not sent"):
                client.chat.completions.create(model="gpt-6-luna", messages=[])
        assert fake.completions.sent == 2

    def test_closing_the_session_removes_the_pre_call_check(self):
        from aieng.advanced_rag.metering import MeteredClient

        fake = _FakeClient()
        client = MeteredClient(fake, UsageMeter())
        with self._session(), pytest.raises(SpendLimitReached):
            for _ in range(3):
                client.chat.completions.create(model="gpt-6-luna", messages=[])
        client.chat.completions.create(model="gpt-6-luna", messages=[])
        assert fake.completions.sent == 3
