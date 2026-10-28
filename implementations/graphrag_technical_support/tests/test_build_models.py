"""The build model and the answering model are chosen separately.

``LLM_MODEL`` used to set both. A participant who set it to answer on a paid
model got a cache miss on the reference index, because the index key named the
answering model. These tests pin the split.
"""

from __future__ import annotations

import asyncio
import warnings

import build_indexes
import build_serverfault_graph
import pytest
import run_musique_test
from aieng.advanced_rag.models import ROLE_ENV
from musique_arms import ANSWER, ANSWER_PROMPTS, ReplayArm, WithholdArm, split_flag


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in (*ROLE_ENV.values(), "LLM_MODEL", "ADVANCED_RAG_MODEL_FAMILY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("module", [build_indexes, build_serverfault_graph])
def test_the_build_defaults_to_the_reference_model(module):
    assert module.build_model() == module.REFERENCE_EXTRACTION_MODEL


@pytest.mark.parametrize("module", [build_indexes, build_serverfault_graph])
def test_the_flag_and_the_extraction_variable_pick_the_build_model(module, monkeypatch):
    monkeypatch.setenv(ROLE_ENV["extraction"], "gpt-6-luna")
    assert module.build_model() == "gpt-6-luna"
    assert module.build_model("claude-haiku-5-5") == "claude-haiku-5-5"


@pytest.mark.parametrize("module", [build_indexes, build_serverfault_graph])
def test_llm_model_still_builds_with_a_deprecation_warning(module, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "Qwen3_8-27B")
    with pytest.warns(DeprecationWarning, match=ROLE_ENV["extraction"]):
        assert module.build_model() == "Qwen3_8-27B"


def test_the_index_key_ignores_the_answering_model(monkeypatch):
    reference = build_indexes.cache_key("lightrag", 11656, 11656)
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv(ROLE_ENV["generation"], "claude-haiku-5-5")
    assert build_indexes.cache_key("lightrag", 11656, 11656) == reference
    assert reference.config["model"] == "Qwen3_6-35B-A3B"


def test_the_graph_key_defaults_to_the_reference_build(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    key = build_serverfault_graph.cache_key(9999)
    assert key.config["model"] == "Qwen3_8-27B"
    assert build_serverfault_graph.cache_key(9999, "gpt-6-luna") != key


def test_a_rebuilt_index_has_its_own_key():
    reference = build_indexes.cache_key("hipporag2", 10, 10)
    rebuilt = build_indexes.cache_key("hipporag2", 10, 10, "gpt-6-luna")
    assert reference.digest != rebuilt.digest


def test_the_answering_model_comes_from_the_family():
    assert run_musique_test.answer_model(None, "onprem") == "Qwen3_6-35B-A3B"
    assert run_musique_test.answer_model(None, "openai") == "gpt-6-luna"
    assert run_musique_test.answer_model("claude-haiku-5-5", "onprem") == "claude-haiku-5-5"


def test_the_generation_variable_picks_the_answering_model(monkeypatch):
    monkeypatch.setenv(ROLE_ENV["generation"], "gemini-3.5-flash-lite")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run_musique_test.answer_model(None, "onprem") == "gemini-3.5-flash-lite"


def test_llm_model_answers_with_a_warning_but_restores_the_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    with pytest.warns(DeprecationWarning):
        assert run_musique_test.answer_model(None, "onprem") == "claude-haiku-5-5"
    seen = {}

    def fake_fetch(cache, key, destination, replace):
        seen["key"] = key
        return destination

    monkeypatch.setattr(run_musique_test, "fetch_directory", fake_fetch)
    run_musique_test.restore("lightrag", 11656, 11656, None, tmp_path)
    assert seen["key"] == build_indexes.cache_key("lightrag", 11656, 11656)


def test_a_missing_index_names_the_build_model(monkeypatch, tmp_path):
    monkeypatch.setattr(run_musique_test, "fetch_directory", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="built with Qwen3_6-35B-A3B"):
        run_musique_test.restore("hipporag2", 5, 5, None, tmp_path)


def test_the_judge_comes_from_the_flag_then_the_family():
    assert run_musique_test.judging_model("gemini-3.8-flash", "onprem") == "gemini-3.8-flash"
    assert run_musique_test.judging_model(None, "google") == "gemini-3.8-flash"


def test_produced_by_labels_a_paid_answer_as_a_rerun():
    paid = run_musique_test.produced_by("gemini-3.5-flash-lite", "gemini-3.8-flash")
    assert paid == {
        "model": "gemini-3.5-flash-lite",
        "endpoint": "vector-proxy-paid",
        "producer": "run_musique_test.py",
        "label": "paid-model rerun",
        "judge": "gemini-3.8-flash",
    }
    onprem = run_musique_test.produced_by("Qwen3_6-35B-A3B")
    assert onprem["label"] == "reference trajectories"
    assert "judge" not in onprem


def test_only_the_lightrag_arm_reads_the_lightrag_index():
    users = run_musique_test.NEEDS_INDEX
    assert users["lightrag"] == ("lightrag",)
    assert "hybrid" in users["hipporag2"]
    assert set(users["hipporag2"]) | set(users["lightrag"]) <= set(run_musique_test.ARM_NAMES)


def test_judged_shares_split_by_hop_count():
    from musique_report import Row  # noqa: PLC0415

    rows = [Row("a", 2, ("p",), 1.0, 1.0), Row("b", 2, ("p",), 0.0, 0.0), Row("c", 4, ("p",), 0.0, 0.5)]
    shares = run_musique_test.judged_shares(rows, {"a": True, "c": True})
    assert shares == {"all": 0.6667, "2hop": 0.5, "4hop": 1.0}


def test_replay_arm_returns_the_stored_ranking():
    arm = ReplayArm("dense", {"Who?": ["musique::1", "musique::2", "musique::3"]})
    assert asyncio.run(arm.rank("Who?", 2)) == ["musique::1", "musique::2"]
    with pytest.raises(KeyError):
        asyncio.run(arm.rank("Unseen?", 2))


def test_the_best_guess_prompt_drops_only_the_abstention_option():
    assert ANSWER_PROMPTS["standard"] == ANSWER
    assert '"unknown"' in ANSWER
    assert "unknown" not in ANSWER_PROMPTS["best-guess"]


def test_rankings_from_and_answer_prompt_parse(monkeypatch, tmp_path):
    def fake_run(coroutine):
        coroutine.close()
        raise SystemExit(0)

    monkeypatch.setattr(run_musique_test.asyncio, "run", fake_run)
    with pytest.raises(SystemExit):
        run_musique_test.main(["--rankings-from", str(tmp_path / "r.json"), "--answer-prompt", "best-guess"])


def test_abstention_counts_unknown_and_empty_answers():
    assert run_musique_test.abstention({"a": "Unknown.", "b": "", "c": "Paris"}) == 0.6667


def test_the_flagged_prompt_marks_guesses_and_the_mark_is_split_off():
    assert "unknown" not in ANSWER_PROMPTS["flagged"]
    assert "(unsure)" in ANSWER_PROMPTS["flagged"]
    assert split_flag("Paris (unsure)") == ("Paris", True)
    assert split_flag("Paris (Unsure).") == ("Paris", True)
    assert split_flag("Paris") == ("Paris", False)


def test_outcomes_label_declined_flagged_and_answered():
    labels = run_musique_test.outcomes({"a": "unknown", "b": "Paris (unsure)", "c": "Paris", "d": ""})
    assert labels == {"a": "declined", "b": "flagged", "c": "answered", "d": "declined"}


def test_withhold_arm_drops_the_chosen_passages_and_keeps_depth():
    ranking = [f"musique::{i}" for i in range(10)]
    arm = WithholdArm(ReplayArm("dense", {"Who?": ranking}), {"Who?": ("musique::0", "musique::3")})
    assert asyncio.run(arm.rank("Who?", 5)) == ["musique::1", "musique::2", "musique::4", "musique::5", "musique::6"]
    assert asyncio.run(WithholdArm(arm, {}, everything=True).rank("Who?", 5)) == []


def test_withheld_passages_by_mode():
    from aieng.advanced_rag.corpora.musique import MusiqueQuestion  # noqa: PLC0415

    question = MusiqueQuestion("2hop__1_2", "Who?", "X", (), 2, ("musique::1", "musique::2"), ())
    assert run_musique_test.withheld_passages([question], "none") == {}
    assert run_musique_test.withheld_passages([question], "supporting") == {"Who?": ("musique::1", "musique::2")}
    assert run_musique_test.withheld_passages([question], "final") == {"Who?": ("musique::2",)}
