"""Guards for the notebook conventions in CONTRIBUTING.md.

Every committed notebook in the repository is checked, because the conventions
exist for the machine a notebook runs on that is not its author's. The
credential check is the one that matters most: a notebook's outputs are
committed on purpose, so anything a cell printed is published.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
NAME_PATTERN = re.compile(r"^\d{2}_[a-z0-9]+(_[a-z0-9]+)*\.ipynb$")
BASELINE_NOTEBOOK = REPO / "implementations" / "regulatory_policy_qa" / "01_harness_and_baseline.ipynb"

# Shapes that must never reach a committed output. The Langfuse key prefixes and
# the proxy key's own prefix are literal; the host and project patterns catch a
# trace URL pasted or printed by an SDK.
SECRET_PATTERNS = (
    ("a Langfuse public key", re.compile(r"pk-lf-[0-9a-f-]{8,}")),
    ("a Langfuse secret key", re.compile(r"sk-lf-[0-9a-f-]{8,}")),
    ("an OpenAI-style key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("a Langfuse host", re.compile(r"https?://[\w.-]*langfuse[\w.-]*")),
    ("a Langfuse trace URL", re.compile(r"/project/[A-Za-z0-9]{10,}/traces?/")),
)


#: Directories whose notebooks are not ours to check.
IGNORED = frozenset({".ipynb_checkpoints", ".venv", ".claude", "node_modules", "site-packages"})


def notebooks() -> list[Path]:
    """Return every committed notebook, ignoring checkpoints and caches.

    The ignore test is on the path relative to the repository, because a git
    worktree can live inside the repository directory, and matching on absolute parts
    excluded every notebook in the repository when run from one.
    """
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "-z", "--", "*.ipynb"],
        cwd=REPO,
        text=True,
    ).split("\0")
    return sorted(
        REPO / name for name in set(names) if name and (REPO / name).is_file() and not IGNORED & set(Path(name).parts)
    )


def cells(path: Path, kind: str = "code") -> list[dict]:
    """Return a notebook's cells of one type."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in loaded["cells"] if cell["cell_type"] == kind]


def output_text(cell: dict) -> str:
    """Return everything one cell printed, across every output form."""
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        chunks.append("".join(output.get("text", "")))
        for value in (output.get("data") or {}).values():
            chunks.append("".join(value) if isinstance(value, list) else str(value))
        chunks.append(str(output.get("evalue", "")))
        chunks.extend(output.get("traceback", []) or [])
    return "\n".join(chunks)


def ids(paths: list[Path]) -> list[str]:
    """Return readable test identifiers for a list of notebooks."""
    return [str(path.relative_to(REPO)) for path in paths]


def notebook_source(path: Path) -> str:
    """Return every cell's source as one searchable string."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in loaded["cells"])


def notebook_outputs(path: Path) -> str:
    """Return the text committed by every code cell."""
    return "\n".join(output_text(cell) for cell in cells(path))


# A repository with no notebooks would pass every test below vacuously, so this
# fails instead and says so, rather than reporting a green suite that checked
# nothing.
def test_there_is_at_least_one_notebook_to_check() -> None:
    assert notebooks(), "no committed notebook found, so these guards checked nothing"


@pytest.mark.parametrize("path", notebooks(), ids=ids(notebooks()))
class TestEveryNotebookFollowsTheConventions:
    """One class per rule in CONTRIBUTING.md's notebook section."""

    def test_it_is_named_for_its_place_in_the_order(self, path: Path) -> None:
        assert NAME_PATTERN.match(path.name), f"{path.name} is not NN_lower_snake_case.ipynb"

    def test_expensive_work_sits_behind_a_flag_that_ships_off(self, path: Path) -> None:
        source = "\n".join("".join(cell["source"]) for cell in cells(path))
        assert "RUN_EXPENSIVE" in source, "no RUN_EXPENSIVE flag"
        assert re.search(r"^RUN_EXPENSIVE\s*=\s*False\s*$", source, re.MULTILINE), (
            "RUN_EXPENSIVE is not declared at module level with a False default"
        )

    def test_every_code_cell_ran_in_order(self, path: Path) -> None:
        counts = [cell.get("execution_count") for cell in cells(path)]
        assert None not in counts, "a code cell was never executed"
        assert counts == sorted(counts), "cells were not executed top to bottom"

    def test_the_outputs_are_committed(self, path: Path) -> None:
        printed = [cell for cell in cells(path) if cell.get("outputs")]
        assert printed, "no cell has committed output, so a reader sees nothing"

    def test_no_output_carries_a_credential_or_a_trace_identifier(self, path: Path) -> None:
        for index, cell in enumerate(cells(path)):
            text = output_text(cell)
            for label, pattern in SECRET_PATTERNS:
                found = pattern.search(text)
                assert found is None, f"cell {index} output holds {label}"

    def test_no_cell_failed(self, path: Path) -> None:
        for index, cell in enumerate(cells(path)):
            errors = [output for output in cell.get("outputs", []) if output.get("output_type") == "error"]
            assert not errors, f"cell {index} committed a traceback"


class TestHarnessAndBaselineNotebookCoversItsAcceptanceCriteria:
    """Guard the measured and teaching content required of RI #1's first notebook."""

    def test_it_fetches_the_pinned_corpus_for_a_fresh_environment(self) -> None:
        source = notebook_source(BASELINE_NOTEBOOK)
        assert "cuad.fetch_cuad_test()" in source
        assert "cuad.build_slice(" in source

    def test_it_recomputes_the_promptless_dense_acceptance_baseline(self) -> None:
        source = notebook_source(BASELINE_NOTEBOOK)
        outputs = notebook_outputs(BASELINE_NOTEBOOK)
        assert 'query_prompt=""' in source
        assert "run_retriever(dense_promptless" in source
        assert "DENSE_PROMPTLESS = (0.2400, 0.4348)" in source
        assert "dense, promptless   failure 0.2400  MRR 0.4348" in outputs

    def test_it_keeps_the_prompted_improvement_separate(self) -> None:
        source = notebook_source(BASELINE_NOTEBOOK)
        outputs = notebook_outputs(BASELINE_NOTEBOOK)
        assert "DENSE_WITH_PROMPT = (0.2400, 0.4632)" in source
        assert "dense, BGE prompt   failure 0.2400  MRR 0.4632" in outputs

    def test_it_scores_null_detection_without_using_gold_as_a_detector(self) -> None:
        source = notebook_source(BASELINE_NOTEBOOK)
        outputs = notebook_outputs(BASELINE_NOTEBOOK)
        assert source.count("score_null_detection(") >= 2
        assert "always answer" in outputs
        assert "always abstain" in outputs
        assert "false-answer" in outputs

    def test_it_measures_the_chunk_size_confound_at_equal_characters(self) -> None:
        source = notebook_source(BASELINE_NOTEBOOK)
        outputs = notebook_outputs(BASELINE_NOTEBOOK)
        assert "budget_k(" in source
        assert "retrieved_chars(" in source
        assert "CHAR_BUDGET = 10_000" in source
        assert "fixed k chars" in outputs
        assert "budget k chars" in outputs
        assert "0.1520" in outputs
        assert "0.2840" in outputs


CHUNKING_NOTEBOOK = REPO / "implementations" / "regulatory_policy_qa" / "03_chunking_variants.ipynb"


class TestChunkingVariantsNotebookCoversItsAcceptanceCriteria:
    """Guard what RI #1's chunking notebook has to measure and say."""

    def test_it_scores_every_arm_scoped_and_unscoped(self) -> None:
        outputs = notebook_outputs(CHUNKING_NOTEBOOK)
        assert "scoped: 1,188 answerable questions" in outputs
        assert "unscoped, named: 1,188 answerable questions" in outputs
        for arm in (
            "plain",
            "hierarchical",
            "metadata header",
            "contextual, head + neighbourhood",
            "contextual, full document",
        ):
            assert outputs.count(f"\n{arm} ") >= 2, arm

    def test_it_reports_indexing_cost_and_estimated_prices(self) -> None:
        outputs = notebook_outputs(CHUNKING_NOTEBOOK)
        assert "indexing cost per arm" in outputs
        assert "estimated price at list" in outputs
        assert "claude-haiku-5-5" in outputs

    def test_it_labels_the_context_model_and_the_contrast_source(self) -> None:
        outputs = notebook_outputs(CHUNKING_NOTEBOOK)
        assert "contexts by Qwen3_6-35B-A3B" in outputs
        assert "source: contextual chunking runs on CUAD and ServerFault" in outputs

    def test_it_names_the_winner_and_its_scope(self) -> None:
        loaded = json.loads(CHUNKING_NOTEBOOK.read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in loaded["cells"] if c["cell_type"] == "markdown")
        assert "the measured winner is contextual chunking with the head-and-neighbourhood prompt" in prose
        assert "Under single-contract scope there is no winner" in prose
