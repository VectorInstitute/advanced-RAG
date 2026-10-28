"""Tests for load_or_run: your result, the labelled reference, or a live run."""

from __future__ import annotations

import functools
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.metering import CallUsage, report_call
from aieng.advanced_rag.models import choose_models
from aieng.advanced_rag.pipeline.generate import ChatCompleter
from aieng.advanced_rag.runs import (
    ConfirmationRequiredError,
    Provenance,
    ReferenceMissingError,
    Step,
    code_hash,
    confirmation_needed,
    load_or_run,
    rebuild_estimate,
)
from aieng.advanced_rag.spend import OFFLINE_PRICES, Budget, ProxyAccount


test_runs_module = sys.modules[__name__]


STEP = Step(
    ri_id="ri-test",
    artifact_type="answers",
    producer_id="unit",
    config={"questions": 3, "prompt": "abc123"},
    role="generation",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv(spend.KEY_BUDGET_ENV, raising=False)
    spend.use_account(ProxyAccount(None, "https://proxy.vectorinstitute.ai/v1"))
    yield
    spend.use_account(None)


@pytest.fixture
def cache(tmp_path):
    return TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )


def fake_client(text="answer"):
    def create(**kwargs):
        message = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=1_000_000,
                completion_tokens=100_000,
                total_tokens=1_100_000,
            ),
            model=kwargs["model"] + "-snap1",
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def build_with_completer(choice):
    complete = ChatCompleter(fake_client(), choice.model)
    return {"answers": [complete("q1"), complete("q2")], "model": choice.model}


def build_reference(cache):
    """Store the reference result the way the maintainers build it."""
    return load_or_run(
        STEP,
        build_with_completer,
        models=choose_models("onprem"),
        live=True,
        cache=cache,
        show=False,
        table=OFFLINE_PRICES,
    )


class TestReference:
    def test_with_live_off_a_paid_model_gets_the_labelled_reference(
        self, cache, capsys
    ):
        build_reference(cache)
        result = load_or_run(
            STEP,
            lambda choice: pytest.fail("must not run"),
            models=choose_models("openai"),
            cache=cache,
            table=OFFLINE_PRICES,
        )
        assert result.origin == "reference"
        assert result.value["model"] == "Qwen3_6-35B-A3B"
        printed = capsys.readouterr().out
        assert (
            "answers reference artifact, built with Qwen3_6-35B-A3B on Vector's cluster at $0"
            in printed
        )
        assert "2 calls, 2,000,000 prompt + 200,000 completion tokens" in printed
        # 2M prompt at $0.10 and 0.2M completion at $0.50 per 1M on gpt-6-luna.
        assert "rebuilding with gpt-6-luna: about 2 calls, est. $0.30" in printed
        assert result.estimate_usd == pytest.approx(0.30)

    def test_the_reference_digest_is_the_reference_key(self, cache):
        built = build_reference(cache)
        result = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("google"),
            cache=cache,
            show=False,
        )
        assert result.digest == built.digest

    def test_no_reference_and_live_off_says_what_to_do(self, cache):
        with pytest.raises(ReferenceMissingError, match="RUN_EXPENSIVE"):
            load_or_run(
                STEP, build_with_completer, models=choose_models("openai"), cache=cache
            )

    def test_a_reference_stored_without_provenance_uses_the_steps_record(self, cache):
        step = Step(
            "ri-test",
            "graph",
            "legacy",
            {"documents": 2},
            role="extraction",
            reference_model="Qwen3_8-27B",
            reference=Provenance.for_model(
                "Qwen3_8-27B",
                usage={
                    "Qwen3_8-27B": {
                        "calls": 20077,
                        "prompt_tokens": 10_960_000,
                        "completion_tokens": 15_170_000,
                    }
                },
            ),
        )
        cache.load_or_compute(step.key(step.reference_choice), lambda: {"nodes": 3})
        result = load_or_run(
            step,
            lambda choice: pytest.fail("must not run"),
            models=choose_models("anthropic"),
            cache=cache,
            show=False,
            table=OFFLINE_PRICES,
        )
        assert result.value == {"nodes": 3}
        assert "built with Qwen3_8-27B" in result.label
        assert "20,077 calls" in result.label
        # 10.96M at $0.10 plus 15.17M at $0.50 on claude-haiku-5-5.
        assert result.estimate_usd == pytest.approx(1.096 + 7.585)


class TestLive:
    def test_a_live_run_records_the_model_tokens_and_reported_snapshot(
        self, cache, capsys
    ):
        result = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            cache=cache,
            table=OFFLINE_PRICES,
            budget=Budget(300.0, 0.0),
        )
        assert result.origin == "live"
        assert result.value["model"] == "gpt-6-luna"
        provenance = result.provenance
        assert provenance is not None
        assert provenance.model == "gpt-6-luna"
        assert provenance.response_models == ("gpt-6-luna-snap1",)
        assert provenance.endpoint == "vector-proxy-paid"
        assert provenance.usage["gpt-6-luna"]["calls"] == 2
        printed = capsys.readouterr().out
        assert "answers your run: gpt-6-luna" in printed
        assert "est. $0.30" in printed

    def test_a_second_call_returns_your_cached_result_without_running(self, cache):
        load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            cache=cache,
            show=False,
            budget=Budget(300.0, 0.0),
        )
        again = load_or_run(
            STEP,
            lambda choice: pytest.fail("must not run"),
            models=choose_models("openai"),
            live=False,
            cache=cache,
            show=False,
        )
        assert again.origin == "yours"
        assert again.value["model"] == "gpt-6-luna"
        assert (
            "your cached result, built with gpt-6-luna through the proxy" in again.label
        )

    def test_models_never_share_a_key(self, cache):
        first = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            cache=cache,
            show=False,
            budget=Budget(300.0, 0.0),
        )
        second = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("anthropic"),
            live=True,
            cache=cache,
            show=False,
            budget=Budget(300.0, 0.0),
        )
        assert first.digest != second.digest
        assert second.value["model"] == "claude-haiku-5-5"

    def test_the_key_names_the_model_and_endpoint(self):
        key = STEP.key(choose_models("google").generation)
        assert key.config["model"] == "gemini-3.5-flash-lite"
        assert key.config["endpoint"] == "vector-proxy-paid"

    def test_upstream_digests_are_part_of_the_key(self):
        over_reference = Step(**{**STEP.__dict__, "upstream": {"graph": "a" * 64}})
        over_rebuild = Step(**{**STEP.__dict__, "upstream": {"graph": "b" * 64}})
        choice = choose_models("google").generation
        assert over_reference.key(choice).digest != over_rebuild.key(choice).digest

    def test_an_expensive_step_waits_for_confirmation(self, cache):
        build_reference(cache)
        with pytest.raises(ConfirmationRequiredError, match="confirm=True"):
            load_or_run(
                STEP,
                build_with_completer,
                models=choose_models("openai"),
                live=True,
                cache=cache,
                show=False,
                table=OFFLINE_PRICES,
                budget=Budget(1.0, 0.0),
            )
        result = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            confirm=True,
            cache=cache,
            show=False,
            table=OFFLINE_PRICES,
            budget=Budget(1.0, 0.0),
        )
        assert result.origin == "live"

    def test_a_moved_alias_makes_your_result_stale(self, cache):
        load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            cache=cache,
            show=False,
            budget=Budget(300.0, 0.0),
        )
        # A later live call reports a new snapshot behind the same name.
        report_call(
            CallUsage(), 0.0, model="gpt-6-luna", response_model="gpt-6-luna-snap2"
        )
        quiet = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            cache=cache,
            show=False,
        )
        assert quiet.origin == "yours"
        assert "now reports gpt-6-luna-snap2" in quiet.label
        rebuilt = load_or_run(
            STEP,
            build_with_completer,
            models=choose_models("openai"),
            live=True,
            cache=cache,
            show=False,
            budget=Budget(300.0, 0.0),
        )
        assert rebuilt.origin == "live"
        report_call(
            CallUsage(), 0.0, model="gpt-6-luna", response_model="gpt-6-luna-snap1"
        )


class TestAssumedBudget:
    """The guard against a key that reports no budget, and against one that does."""

    @staticmethod
    def run(cache, family, budget, **kwargs):
        return load_or_run(
            STEP,
            build_with_completer,
            models=choose_models(family),
            live=True,
            cache=cache,
            table=OFFLINE_PRICES,
            budget=budget,
            **kwargs,
        )

    def test_a_reported_budget_keeps_the_ten_percent_gate_alone(self, cache, capsys):
        build_reference(cache)
        # 2M prompt at $0.30 and 0.2M completion at $2.50 per 1M on
        # gemini-3.5-flash-lite: over $1, under 10% of $300.
        result = self.run(cache, "google", Budget(300.0, 0.0))
        assert result.origin == "live"
        printed = capsys.readouterr().out
        assert "est. $1.10" in printed
        assert "assum" not in printed

    def test_an_unknown_budget_is_assumed_and_a_step_over_a_dollar_stops(
        self, cache, capsys
    ):
        build_reference(cache)
        with pytest.raises(ConfirmationRequiredError) as raised:
            self.run(cache, "google", Budget(None, 12.40))
        message = str(raised.value)
        assert "$1.10 is over $1.00" in message
        assert "$300.00 is assumed" in message
        assert spend.KEY_BUDGET_ENV in message
        assert "confirm=True" in message
        printed = capsys.readouterr().out
        assert (
            "key budget not set on the key; assuming $300.00 (set "
            "ADVANCED_RAG_KEY_BUDGET_USD to override), spent $12.40, left $287.60"
        ) in printed
        result = self.run(cache, "google", Budget(None, 12.40), confirm=True)
        assert result.origin == "live"

    def test_an_assumed_budget_lets_a_step_under_a_dollar_run(self, cache):
        build_reference(cache)
        result = self.run(cache, "openai", Budget(None, 0.0), show=False)
        assert result.origin == "live"

    def test_a_silent_proxy_gets_the_assumed_budget_too(self, cache):
        build_reference(cache)
        # The offline account answers nothing, so the guard reads Budget(None, None).
        with pytest.raises(ConfirmationRequiredError, match="assumed"):
            self.run(cache, "google", None, show=False)

    def test_the_ten_percent_gate_runs_against_the_assumed_budget(self, cache):
        build_reference(cache)
        # $0.30 is under $1 but over 10% of the $2.00 left of an assumed $300.
        with pytest.raises(ConfirmationRequiredError) as raised:
            self.run(cache, "openai", Budget(None, 298.0), show=False)
        assert "over 10% of the $2.00 left on the key" in str(raised.value)
        assert "assumed" in str(raised.value)

    def test_the_variable_overrides_the_assumed_budget(self, cache, monkeypatch):
        build_reference(cache)
        monkeypatch.setenv(spend.KEY_BUDGET_ENV, "2")
        with pytest.raises(ConfirmationRequiredError, match=r"10% of the \$2\.00"):
            self.run(cache, "openai", Budget(None, 0.0), show=False)
        # A reported budget ignores the variable.
        assert self.run(cache, "openai", Budget(300.0, 0.0), show=False).origin == (
            "live"
        )

    def test_replay_never_stops_while_the_budget_is_assumed(self, cache):
        build_reference(cache)
        # The reference with live off: the guard never runs.
        shown = load_or_run(
            STEP,
            lambda choice: pytest.fail("must not run"),
            models=choose_models("google"),
            cache=cache,
            show=False,
            table=OFFLINE_PRICES,
        )
        assert shown.origin == "reference"
        assert shown.estimate_usd == pytest.approx(1.10)
        self.run(cache, "google", Budget(None, 0.0), confirm=True, show=False)
        # Your cached result, even with live on and no confirm.
        for live in (False, True):
            again = load_or_run(
                STEP,
                lambda choice: pytest.fail("must not run"),
                models=choose_models("google"),
                live=live,
                cache=cache,
                show=False,
                table=OFFLINE_PRICES,
                budget=Budget(None, 0.0),
            )
            assert again.origin == "yours"

    def test_confirmation_needed_names_each_reason(self):
        assert confirmation_needed(3.0, Budget(300.0, 0.0)) is None
        assert "10% of the $20.00" in str(confirmation_needed(3.0, Budget(20.0, 0.0)))
        assert "over $1.00" in str(confirmation_needed(1.5, Budget(None, 0.0)))
        assert confirmation_needed(1.0, Budget(None, 0.0)) is None


class TestDirectorySteps:
    def test_a_directory_step_stores_and_restores_with_provenance(
        self, cache, tmp_path
    ):
        step = Step(
            "ri-test",
            "index",
            "dir-unit",
            {"n": 1},
            role="extraction",
            kind="directory",
        )

        def build(choice, directory):
            (directory / "graph.json").write_text(
                ChatCompleter(fake_client("g"), choice.model)("x")
            )

        built = load_or_run(
            step,
            build,
            models=choose_models("onprem"),
            live=True,
            cache=cache,
            workdir=tmp_path / "work",
            show=False,
        )
        assert (built.value / "graph.json").read_text() == "g"
        reused = load_or_run(
            step,
            lambda *a: pytest.fail("must not run"),
            models=choose_models("google"),
            cache=cache,
            workdir=tmp_path / "work",
            show=False,
            table=OFFLINE_PRICES,
        )
        assert reused.origin == "reference"
        assert (reused.value / "graph.json").read_text() == "g"
        assert reused.provenance is not None
        assert reused.provenance.model == "Qwen3_6-35B-A3B"


def test_the_estimate_is_none_without_tokens_or_a_price():
    assert rebuild_estimate(None, "gpt-6-luna") == (0, None)
    provenance = Provenance.for_model(
        "Qwen3_6-35B-A3B",
        usage={
            "Qwen3_6-35B-A3B": {"calls": 1, "prompt_tokens": 5, "completion_tokens": 5}
        },
    )
    assert rebuild_estimate(provenance, "mystery-model", table=OFFLINE_PRICES) == (
        1,
        None,
    )


def test_provenance_round_trips():
    record = Provenance.for_model(
        "gpt-6-luna",
        response_models=("x",),
        usage={"gpt-6-luna": {"calls": 1, "prompt_tokens": 2, "completion_tokens": 3}},
    )
    assert Provenance.from_dict(record.to_dict()) == record


def _producer(tmp_path, name, body):
    """Return a module loaded from a file the test can edit afterwards."""
    path = tmp_path / f"{name}.py"
    path.write_text(body)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


class TestCodeHash:
    def test_ignores_order_paths_and_line_endings(self, tmp_path):
        a, _ = _producer(tmp_path, "prod_a", "X = 1\n")
        b, _ = _producer(tmp_path, "prod_b", "Y = 2\n")
        crlf, _ = _producer(tmp_path, "prod_crlf", "X = 1\r\n")
        assert code_hash([a, b]) == code_hash([b, a])
        assert code_hash([a]) == code_hash([crlf])
        assert len(code_hash([a])) == 16

    def test_a_partial_counts_as_the_function_it_wraps(self):
        assert code_hash([functools.partial(build_with_completer)]) == code_hash(
            [build_with_completer]
        )

    def test_is_none_without_a_source_file(self):
        assert code_hash([len]) is None
        assert code_hash([]) is None

    def test_a_step_defaults_to_the_file_that_defines_compute(self):
        assert STEP.code_hash(build_with_completer) == code_hash([test_runs_module])


class TestCodeChanged:
    def step(self, *code):
        return Step(
            ri_id="ri-test",
            artifact_type="answers",
            producer_id="hashed",
            config={"questions": 3},
            code=code,
        )

    def build(self, step, cache):
        return load_or_run(
            step,
            build_with_completer,
            models=choose_models("onprem"),
            live=True,
            cache=cache,
            show=False,
            table=OFFLINE_PRICES,
        )

    def replay(self, step, cache, family):
        return load_or_run(
            step,
            build_with_completer,
            models=choose_models(family),
            cache=cache,
            show=False,
            table=OFFLINE_PRICES,
        )

    def test_a_live_run_records_the_hash(self, cache, tmp_path):
        module, _ = _producer(tmp_path, "prod_record", "PROMPT = 'a'\n")
        built = self.build(self.step(module), cache)
        assert built.provenance.code_hash == code_hash([module])

    @pytest.mark.parametrize("family", ["onprem", "anthropic"])
    def test_replay_after_an_edit_warns_and_still_returns(
        self, cache, tmp_path, family
    ):
        module, path = _producer(tmp_path, "prod_edit", "PROMPT = 'a'\n")
        step = self.step(module)
        built = self.build(step, cache)
        path.write_text("PROMPT = 'b'\n")
        replayed = self.replay(step, cache, family)
        assert replayed.code_changed
        assert replayed.value == built.value
        assert "code changed" in replayed.label

    def test_with_live_on_an_edit_rebuilds_your_result(self, cache, tmp_path):
        module, path = _producer(tmp_path, "prod_live", "PROMPT = 'a'\n")
        step = self.step(module)
        self.build(step, cache)
        path.write_text("PROMPT = 'b'\n")
        rebuilt = self.build(step, cache)
        assert rebuilt.origin == "live"
        assert rebuilt.provenance.code_hash == code_hash([module])
        assert not self.replay(step, cache, "onprem").code_changed

    def test_replay_of_unchanged_code_is_silent(self, cache, tmp_path):
        module, _ = _producer(tmp_path, "prod_same", "PROMPT = 'a'\n")
        step = self.step(module)
        self.build(step, cache)
        replayed = self.replay(step, cache, "anthropic")
        assert not replayed.code_changed
        assert "code changed" not in replayed.label

    def test_an_artifact_without_a_hash_replays_silently(self, cache, tmp_path):
        # A builtin has no source file, so the build records no hash, as an
        # artifact built before hashes existed does.
        self.build(self.step(len), cache)
        module, _ = _producer(tmp_path, "prod_legacy", "PROMPT = 'a'\n")
        replayed = self.replay(self.step(module), cache, "anthropic")
        assert replayed.provenance.code_hash == ""
        assert not replayed.code_changed
        assert "code changed" not in replayed.label


def test_a_produced_by_block_without_a_hash_reads_as_empty():
    record = Provenance.for_model("gpt-6-luna").to_dict()
    del record["code_hash"]
    assert Provenance.from_dict(record).code_hash == ""
