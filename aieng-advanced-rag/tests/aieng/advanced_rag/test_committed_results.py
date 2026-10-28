"""Every committed result records the model that produced it.

A participant who picks a paid model and leaves ``RUN_EXPENSIVE`` off reads the
committed reference figures, which were built on the on-prem Qwen models. A
figure with no model beside it reads as their model's. This test keeps every
result JSON under ``implementations/*/outputs`` and
``implementations/special_topics/*/outputs`` labelled.

A dict-shaped result carries a ``produced_by`` block with a ``model`` key. A
list-shaped result, whose readers expect a bare list, is labelled in the
``produced_by.json`` manifest in its directory. A few results written before
``produced_by`` existed name their model in another field, listed in
:data:`LEGACY_FIELDS`; new results write ``produced_by``. Results that involve
no chat model at all are listed in :data:`MODEL_FREE` with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aieng.advanced_rag.models import family_of


# Up from this file to the repository root: advanced_rag, aieng, tests,
# aieng-advanced-rag.
REPO_ROOT = Path(__file__).resolve().parents[4]
IMPLEMENTATIONS = REPO_ROOT / "implementations"
MANIFEST = "produced_by.json"

#: Output directories, relative to ``implementations``. A special topic sits one
#: level deeper than a use case, so ``*/outputs`` does not reach it.
OUTPUT_DIRS: tuple[str, ...] = ("*/outputs", "special_topics/*/outputs")

#: Dotted paths that name the model in results written before ``produced_by``.
LEGACY_FIELDS: tuple[str, ...] = (
    "model",
    "test.model",
    "generation_model",
    "answering_model",
    "single_pass_samples.model",
)

#: Results that call no chat model, relative to ``implementations``, and why.
MODEL_FREE: dict[str, str] = {
    "table_structured_reasoning/outputs/ingestion_summary.json": (
        "Docling parsing statistics; no chat model is called"
    ),
    "table_structured_reasoning/outputs/table_retrieval_scores.json": (
        "retrieval scores over local bge-small-en-v1.5 embeddings"
    ),
}


def committed_results() -> list[Path]:
    """Return every result JSON under an ``outputs`` directory, manifests excluded.

    ``baseline_cache`` holds cache envelopes, whose producer identifier is the
    model and which :func:`test_a_cache_envelope_names_its_producer` checks.
    """
    return sorted(
        path
        for path in _under_outputs("**/*.json")
        if path.name != MANIFEST and "baseline_cache" not in path.parts
    )


def _under_outputs(pattern: str) -> set[Path]:
    """Return the files matching ``pattern`` inside any output directory."""
    return {
        path
        for outputs in OUTPUT_DIRS
        for path in IMPLEMENTATIONS.glob(f"{outputs}/{pattern}")
    }


def _dig(payload: Any, dotted: str) -> tuple[bool, Any]:
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def recorded_model(path: Path) -> tuple[bool, Any, dict[str, Any] | None]:
    """Return whether a result records its model, the model, and its block."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("produced_by"), dict):
        block = payload["produced_by"]
        return "model" in block, block.get("model"), block
    manifest = path.parent / MANIFEST
    if manifest.is_file():
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        if path.name in files:
            block = files[path.name]
            return "model" in block, block.get("model"), block
    if isinstance(payload, dict):
        for dotted in LEGACY_FIELDS:
            found, value = _dig(payload, dotted)
            if found and isinstance(value, str):
                return True, value, None
    return False, None, None


RESULTS = committed_results()


def test_the_results_are_found():
    # The parametrized test would pass vacuously if the outputs moved.
    assert len(RESULTS) >= 30


@pytest.mark.parametrize(
    "path", RESULTS, ids=lambda path: str(path.relative_to(IMPLEMENTATIONS))
)
def test_every_committed_result_records_its_model(path):
    relative = str(path.relative_to(IMPLEMENTATIONS))
    if relative in MODEL_FREE:
        return
    found, model, block = recorded_model(path)
    assert found, f"{relative} records no model; add a produced_by block"
    if model is None:
        # Only a result that made no model call may record a null model.
        assert block is not None and block.get("model_calls") == 0, relative
        return
    assert family_of(model) is not None, f"{relative}: unrecognised model {model!r}"


@pytest.mark.parametrize("path", sorted(_under_outputs("**/" + MANIFEST)))
def test_every_manifest_entry_names_a_committed_file(path):
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    assert files
    for name in files:
        assert (path.parent / name).is_file(), f"{path}: {name} is missing"


def test_a_cache_envelope_names_its_producer():
    for path in _under_outputs("baseline_cache/**/*.json"):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        assert envelope["producer_id"], path


def test_the_model_free_list_has_no_stale_entries():
    for relative in MODEL_FREE:
        assert (IMPLEMENTATIONS / relative).is_file(), relative
