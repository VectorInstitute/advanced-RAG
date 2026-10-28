"""Guard portable evaluation inputs and preservation of shipped controls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from advanced_lab import common
from advanced_lab.regulatory_inputs import regulatory_inputs


def test_reference_inputs_reconstruct_the_complete_evaluation_without_private_files():
    rows = regulatory_inputs()
    assert len(rows) == 7 and sum(not row["is_null"] for row in rows) == 6
    assert all(row["retrieved_contexts"] for row in rows)
    assert all(0 < len(text) <= 1200 for row in rows for text in row["retrieved_contexts"])
    assert len({row["query_id"] for row in rows}) == len(rows)


@pytest.mark.parametrize("corrupt", ["source", "evidence", "duplicate", "label"])
def test_invalid_public_inputs_fail_before_optional_model_imports(tmp_path, corrupt):
    for name in ("regulatory.json", "regulatory_source.json"):
        shutil.copyfile(common.OUTPUT / name, tmp_path / name)
    path = tmp_path / ("regulatory_source.json" if corrupt == "source" else "regulatory.json")
    value = json.loads(path.read_text())
    if corrupt == "source":
        value["text"] += "changed"
    elif corrupt == "evidence":
        value["outcomes"][0]["retrieved_ids"] = ["unknown-source"]
    elif corrupt == "duplicate":
        value["outcomes"].append(value["outcomes"][0])
    else:
        value["outcomes"][0]["is_null"] = "false"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        regulatory_inputs(tmp_path)


def test_live_variants_cannot_overwrite_the_shipped_report(tmp_path, monkeypatch):
    output = tmp_path / "reference"
    output.mkdir()
    reference = output / "policies.json"
    reference.write_text('{"reference": true}\n')
    before = reference.read_bytes()
    monkeypatch.setattr(common, "OUTPUT", output)
    monkeypatch.setattr(common, "STORE", tmp_path / "local")
    common.write_json(reference, {"variant": "changed policy"})
    assert reference.read_bytes() == before
    variant = common.STORE / "runs" / common.RUN_ID / reference.name
    assert json.loads(variant.read_text()) == {"variant": "changed policy"}
    assert not list((output / "cache").rglob("*.json"))


def test_login_host_guard_does_not_require_slurm_on_a_participant_machine(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(common.socket, "gethostname", lambda: "participant-laptop")
    common.require_compute()
    monkeypatch.setattr(common.socket, "gethostname", lambda: "blogin02")
    with pytest.raises(RuntimeError, match="compute node"):
        common.require_compute()


@pytest.mark.parametrize(
    "name", ["05_answering_and_null_detection.ipynb", "06_regulatory_qa_and_ragas.ipynb", "99_starter_agent.ipynb"]
)
def test_added_notebooks_ship_executed_and_keep_expensive_work_off(name):
    ri = Path(__file__).resolve().parents[1]
    notebook = json.loads((ri / name).read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "RUN_EXPENSIVE = False" in "".join(code[0]["source"])
    assert [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
    assert all(output["output_type"] != "error" for cell in code for output in cell["outputs"])
