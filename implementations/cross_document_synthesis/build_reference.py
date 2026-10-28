#!/usr/bin/env python3
"""Store RI #2's recorded worked examples in the artifact cache for its release.

Each result goes under the cache key of its step in ``synthesis_lab.reference``,
built with ``Qwen3_6-35B-A3B``, so ``load_or_run`` finds it as the reference.
``scripts/publish_artifacts.py 2`` then stages the stored artifacts as RI #2's
release and writes ``reference-artifacts.json``.

``store`` reads results recorded earlier: the decomposition, agent, IRCoT, and
starter-agent worked examples and the dense baseline's predictions, from a
directory laid out as ``outputs/`` was when they were committed. It makes no
model calls. ``graph`` runs the graph worked example live on the reference
model, which is free on Vector's cluster.

Examples
--------
Store the recorded results, where ``<recorded_dir>`` holds ``examples/`` and
``reference/dense_baseline/``::

    $ python build_reference.py store --from <recorded_dir>

Build the graph worked example on the reference model::

    $ python build_reference.py graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.models import REFERENCE_MODELS
from aieng.advanced_rag.runs import Provenance, Step
from synthesis_lab import reference
from synthesis_lab.runtime import LAB_DIR


def store(step: Step, value: Any, provenance: Provenance, cache: TieredResultCache | None = None) -> str:
    """Store a reference result under its step's reference key and return its digest."""
    key = step.key(step.reference_choice)
    wrapped = {"result": value, "produced_by": provenance.to_dict()}
    (cache or TieredResultCache()).load_or_compute(key, lambda: wrapped, force_refresh=True)
    return str(key.digest)


def _code(step: Step) -> str:
    return step.code_hash(lambda: None) or ""


def _read(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def recorded(root: Path) -> list[tuple[Step, Any, Provenance]]:
    """Return every recorded worked example with its step and ``produced_by`` block."""
    rows: list[tuple[Step, Any, Provenance]] = []

    run = reference.scrub(_read(root, "examples/decomposition/run.json"))
    step = reference.decomposition_step()
    if run["selection"] != step.config["selection"]:
        raise ValueError("the recorded decomposition run used another selection")
    rows.append(
        (
            step,
            {"report": run, "example": _read(root, "examples/decomposition/notebook_example.json")},
            reference.reference_provenance(
                "run_decomposition",
                reference.recorded_usage([r["usage"] for r in run["records"]]),
                "30 MuSiQue questions, 10 per hop count, planned and answered by Qwen3_6-35B-A3B.",
                _code(step),
            ),
        )
    )

    lab = reference.scrub(_read(root, "examples/agent_lab.json"))
    step = reference.agent_lab_step()
    rows.append(
        (
            step,
            lab,
            reference.reference_provenance(
                "run_agent_lab",
                reference.recorded_usage([r["usage"] for r in lab["results"]]),
                "One MuSiQue question per hop count through the ADK agent.",
                _code(step),
            ),
        )
    )

    ircot = _read(root, "examples/ircot.json")
    full = _read(LAB_DIR / "outputs", "evaluation/musique_ircot.json")
    usage = {r["question_id"]: r["usage"] for r in full["records"]}
    for example in ircot["examples"]:
        example["usage"] = usage[example["question_id"]]
    step = reference.ircot_step()
    rows.append(
        (
            step,
            ircot,
            reference.reference_provenance(
                "run_benchmark/ircot",
                reference.recorded_usage([e["usage"] for e in ircot["examples"]]),
                "IRCoT traces for the first MuSiQue question per hop category, from the 1,000-question run.",
                _code(step),
            ),
        )
    )

    starter = _read(root, "examples/starter_agent.json")
    step = reference.starter_step()
    if starter["record"]["question_id"] != step.config["question_id"] or starter["config"] != step.config["config"]:
        raise ValueError("the recorded starter-agent run used another question or bounds")
    rows.append(
        (
            step,
            starter,
            reference.reference_provenance(
                "99_starter_agent",
                reference.recorded_usage([starter["record"]["usage"]]),
                "One MuSiQue question through the starter IRCoT agent.",
                _code(step),
            ),
        )
    )

    predictions = _read(root, "reference/dense_baseline/predictions.json")
    step = reference.dense_step()
    rows.append(
        (
            step,
            {"predictions": predictions},
            reference.reference_provenance(
                "run_dense_baseline",
                reference.recorded_usage([p["usage"] for p in predictions]),
                "Cited answers to the 1,000 MuSiQue questions from five dense passages.",
                _code(step),
            ),
        )
    )
    return rows


def main() -> None:
    """Store the recorded results or build the graph worked example."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    stored = commands.add_parser("store", help="store recorded results; no model calls")
    stored.add_argument("--from", dest="source", type=Path, required=True)
    commands.add_parser("graph", help="run the graph worked example on the reference model")
    args = parser.parse_args()
    if args.command == "store":
        for step, value, provenance in recorded(args.source):
            print(f"{step.artifact_type:24} {store(step, value, provenance)}")
        return
    for backend in ("hipporag2",):
        result = reference.graph_examples(backend=backend, models=REFERENCE_MODELS, live=True)
        # The graph adapters call the model on worker threads the step's meter does
        # not see, so the stored provenance takes its usage from the per-question rows.
        step = reference.graph_step(backend)
        provenance = reference.reference_provenance(
            f"run_graph_comparison/{backend}",
            reference.recorded_usage([r["usage"] for r in result.value["records"]]),
            "The first MuSiQue question per hop category over RI #4's released index.",
            _code(step),
        )
        print(f"{backend}: {result.origin} {store(step, result.value, provenance)}")


if __name__ == "__main__":
    main()
