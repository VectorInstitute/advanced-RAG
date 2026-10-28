"""Check the committed ServerFault graph results for internal consistency."""

import json
from pathlib import Path

from aieng.advanced_rag.graphrag.extraction import ENTITY_TYPES, RELATION_TYPES


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
NOTEBOOK = Path(__file__).resolve().parents[1] / "01_construction_provenance_store.ipynb"


def _summary() -> dict:
    return json.loads((OUTPUTS / "serverfault_graph_summary.json").read_text())


def test_type_counts_add_up_and_stay_on_the_schema() -> None:
    graph = _summary()["graph"]
    assert sum(graph["entity_types"].values()) == graph["entities"]
    assert sum(graph["relation_types"].values()) == graph["relations"]
    assert set(graph["entity_types"]) <= set(ENTITY_TYPES)
    assert set(graph["relation_types"]) <= set(RELATION_TYPES)


def test_cost_figures_agree() -> None:
    summary = _summary()
    cost = summary["cost"]
    assert cost["prompt_tokens"] + cost["completion_tokens"] == cost["total_tokens"]
    assert round(cost["calls"] / summary["documents"], 2) == summary["calls_per_document"]
    assert summary["graph"]["located_mentions"] <= summary["graph"]["mentions"]


def test_sample_validates_and_carries_no_post_text() -> None:
    sample = json.loads((OUTPUTS / "serverfault_graph_sample.json").read_text())
    assert sample["validation_problems"] == 0
    assert sample["entities"]
    assert {e["type"] for e in sample["entities"]} <= set(ENTITY_TYPES)
    assert {r["type"] for r in sample["relations"]} <= set(RELATION_TYPES)
    names = {e["name"] for e in sample["entities"]}
    assert all(r["source"] in names and r["target"] in names for r in sample["relations"])
    for fact in sample["entities"] + sample["relations"]:
        assert fact["cites"]
        for cite in fact["cites"]:
            assert set(cite) == {"section", "block", "start", "end", "located"}


def test_notebook_has_outputs_and_no_secrets() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] for c in code)
    text = NOTEBOOK.read_text()
    assert "RUN_EXPENSIVE = False" in text
    for needle in ("VECTOR_PROXY_API_KEY", "LANGFUSE", "/home/"):
        assert needle not in text
