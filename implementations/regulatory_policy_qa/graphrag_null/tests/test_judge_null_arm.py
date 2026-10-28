"""Check replay configuration and saved-report verification."""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_null_arm as runner  # noqa: E402


def test_dense_only_judging_does_not_require_a_graph_arm(tmp_path, monkeypatch):
    options = runner.parse_args(["--store", str(tmp_path), "--arms", "dense", "--cache-only"])
    assert options.arms == ("dense",)
    inputs = runner.ReplayInputs({}, [], {}, {"dense": {}}, {"dense": {}}, {}, 5)
    monkeypatch.setattr(
        runner, "score_answers", lambda *args, **kwargs: {"method": args[0], "correct_count": 0, "questions": 0}
    )
    reports = runner.score_replay(inputs, lambda _: pytest.fail("no model call allowed"), runner.Meter(), workers=1)
    assert [report["method"] for report in reports] == ["dense"]


def test_cached_report_verification_does_not_rewrite_provenance(tmp_path):
    report = {"method": "dense", "judgements": []}
    inputs = {"example": "digest"}
    summary = {"input_sha256": inputs, "cost": {"calls": 12}}
    runner.publish_reports(tmp_path, [report], summary)
    before = (tmp_path / "answer_judging_summary.json").read_bytes()
    runner.verify_reports(tmp_path, [report], inputs)
    assert (tmp_path / "answer_judging_summary.json").read_bytes() == before
    with pytest.raises(ValueError, match="inputs have changed"):
        runner.verify_reports(tmp_path, [report], {"example": "changed"})
    (tmp_path / "answer_judgements_dense.json").write_text("{}")
    with pytest.raises(ValueError, match="does not reproduce"):
        runner.verify_reports(tmp_path, [report], inputs)


def test_invalid_inputs_fail_before_opening_a_model_client(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("model client created before inputs were validated")

    monkeypatch.setattr(runner, "create_completer", forbidden)
    with pytest.raises(FileNotFoundError):
        runner.main(["--store", str(tmp_path), "--results", str(tmp_path)])


def test_cache_only_client_needs_no_credentials_and_fails_on_a_miss(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("credentials read during cached verification")

    monkeypatch.setattr(runner, "resolve_api_key", forbidden)
    options = runner.JudgeOptions(store=tmp_path, results=tmp_path, env=tmp_path / "missing.env", cache_only=True)
    meter = runner.Meter(phase="answer-judge")
    complete = runner.create_completer(options, meter)
    with pytest.raises(FileNotFoundError, match="cache"):
        complete("uncached judge prompt")
    assert meter.to_dict()["calls"] == 0


def test_explicit_cached_publication_keeps_cold_run_cost(tmp_path):
    report = {"method": "dense", "judgements": []}
    previous = {"input_sha256": {"example": "digest"}, "cost": {"calls": 12}}
    runner.publish_reports(tmp_path, [report], previous)
    before = (tmp_path / "answer_judging_summary.json").read_bytes()
    out = tmp_path / "new"
    options = runner.JudgeOptions(
        store=tmp_path, results=tmp_path, env=tmp_path / "missing.env", cache_only=True, out=out
    )
    summary = {"input_sha256": previous["input_sha256"], "cost": {"calls": 0}}
    runner.finish_replay(options, [report], summary)
    published = json.loads((out / "answer_judging_summary.json").read_text())
    assert published["cold_run_cost"] == previous["cost"]
    assert published["cost"]["calls"] == 0
    assert (tmp_path / "answer_judging_summary.json").read_bytes() == before
