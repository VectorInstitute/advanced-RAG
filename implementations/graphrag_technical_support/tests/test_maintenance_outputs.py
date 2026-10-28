"""Check the committed graph-maintenance results for internal consistency."""

import json
from pathlib import Path

import graph_maintenance as gm


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs" / "maintenance"
NOTEBOOK = HERE / "02_graph_maintenance.ipynb"


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def test_every_result_names_its_model() -> None:
    for name in ("retraction.json", "dedup.json", "gate.json", "revision.json", "lightrag.json"):
        assert _load(name)["model"], name
    drift = _load("drift.json")
    assert drift["reference_model"] == gm.REFERENCE_MODEL
    assert drift["drift_model"] == gm.DRIFT_MODEL


def test_retractions_leave_nothing_citing_the_post() -> None:
    retraction = _load("retraction.json")
    assert retraction["retract_equals_rebuild_of_remaining_units"] is True
    assert set(retraction["demo"]) == set(gm.DEMO_POSTS)
    assert all(row["remaining_citations"] == 0 for row in retraction["demo"].values())
    assert len(retraction["rows"]) == retraction["sample"]["posts"] == gm.RETRACTION_SAMPLE


def test_the_gate_passes_the_provenance_retraction_and_fails_the_broken_ones() -> None:
    gate = _load("gate.json")
    assert gate["provenance_retraction"]["passed"]
    assert gate["dedup_merge"]["passed"]
    for broken in ("stale_serving_index", "naive_delete", "mentions_only"):
        assert not gate[broken]["passed"], broken
    assert all(row["rank_of_target"] for row in gate["smoke_before"].values())


def test_every_labelled_dedup_group_has_a_label() -> None:
    sample = [tuple(row["group"]) for row in _load("dedup.json")["label_sample"]]
    labelled = {tuple(row["group"]): row["label"] for row in _load("dedup_labels.json")["groups"]}
    assert set(sample) == set(labelled)
    assert set(labelled.values()) <= {"same", "different", "unclear"}


def test_revision_cost_counts_only_edited_sections() -> None:
    revision = _load("revision.json")
    assert revision["edited_window_calls"] + revision["unchanged_windows_reused"] == (
        revision["cost"]["calls"] + revision["cost"]["cache_hits"]
    )
    assert revision["after_revisions_validation_problems"] == 0


def test_lightrag_delete_and_merge_make_no_model_call() -> None:
    steps = _load("lightrag.json")["steps"]
    assert steps[0]["model_calls"] == 0 and steps[1]["model_calls"] == 0
    assert _load("lightrag.json")["smoke"]["cites_deleted"] is False


def test_notebook_has_outputs_and_no_secrets() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] for c in code)
    text = NOTEBOOK.read_text()
    assert "RUN_EXPENSIVE = False" in text
    for needle in ("VECTOR_PROXY_API_KEY", "LANGFUSE", "/home/"):
        assert needle not in text
