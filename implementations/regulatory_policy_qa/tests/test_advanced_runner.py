"""Check experiment selection, input/output routing and model-free defaults."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import run_advanced
from advanced_lab import answering as answering_lab
from advanced_lab import common, ragas_eval


@pytest.mark.parametrize("task", sorted(common.REPORT_NAMES))
def test_default_replay_never_imports_live_experiments(monkeypatch, capsys, task):
    def forbidden(*args):
        pytest.fail("reference replay imported a live experiment")

    monkeypatch.setattr(run_advanced.importlib, "import_module", forbidden)
    run_advanced.main([task])
    assert json.loads(capsys.readouterr().out) == common.saved(task)


@pytest.mark.parametrize(
    "arguments",
    [
        ["policies", "--output-dir", "local"],
        ["ragas", "--input-dir", "local"],
        ["policies", "--cache-only"],
        ["agent", "--run", "--cache-only"],
        ["regulatory", "--run", "--input-dir", "local"],
    ],
)
def test_inapplicable_options_fail_before_any_experiment_import(monkeypatch, arguments):
    monkeypatch.setattr(run_advanced.importlib, "import_module", lambda *args: pytest.fail("imported live experiment"))
    with pytest.raises(SystemExit) as error:
        run_advanced.main(arguments)
    assert error.value.code == 2


def test_explicit_ragas_inputs_and_output_directory_reach_only_selected_experiment(tmp_path, monkeypatch, capsys):
    inputs, store = tmp_path / "inputs", tmp_path / "cache"
    monkeypatch.setattr(common, "STORE", tmp_path)
    imports = []

    def measure(**options):
        assert options == {"input_dir": inputs}
        common.write_json(common.OUTPUT / "ragas.json", {"evaluated": "custom inputs"})
        return {"evaluated": "custom inputs"}

    def load(name):
        imports.append(name)
        return SimpleNamespace(measure_ragas=measure)

    monkeypatch.setattr(run_advanced.importlib, "import_module", load)
    run_advanced.main(["ragas", "--run", "--input-dir", str(inputs), "--output-dir", str(store)])
    assert imports == ["advanced_lab.ragas_eval"]
    path = store / "runs" / common.RUN_ID / "ragas.json"
    assert json.loads(path.read_text()) == {"evaluated": "custom inputs"}
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"evaluated": "custom inputs"}
    assert str(path) in captured.err
    # Already-imported helpers must follow the CLI's new store too.
    assert answering_lab.run_path("policy-progress.json").parent == path.parent


def test_invalid_custom_ragas_inputs_fail_before_a_model_client_is_created(tmp_path, monkeypatch):
    monkeypatch.setattr(ragas_eval, "completer", lambda *args, **kwargs: pytest.fail("created model client"))
    with pytest.raises(FileNotFoundError):
        ragas_eval.measure_ragas(input_dir=tmp_path)


def test_missing_pinned_source_is_not_silently_replaced_by_a_new_download(tmp_path, monkeypatch):
    monkeypatch.setattr(answering_lab, "OUTPUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        answering_lab.load_regulatory()
    assert list(tmp_path.iterdir()) == []
