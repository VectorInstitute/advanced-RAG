#!/usr/bin/env python3
"""Combine the abstention-cost runs into one results file per answering model.

Each run answers the 300 MuSiQue test questions from stored rankings with one
answering prompt (``standard``, ``best-guess``, or ``flagged``) and one
``--withhold`` mode. ``none`` leaves the questions answerable; ``supporting``
and ``final`` drop supporting passages from what the reader sees, so the
passages no longer state the answer, and ``everything`` drops every passage.
This script reads the twelve runs of one model and writes, per arm and prompt,
how often the reader declines, marks a guess unsure, or answers outright, and
the F1 it scores.

Example
-------
Combine twelve run files named ``<prefix>_<prompt>_<withhold>.json``::

    $ python musique_abstention.py --runs <run_dir>/onprem --out <out.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROMPTS = ("standard", "best-guess", "flagged")
WITHHOLD = ("none", "supporting", "final", "everything")
#: Token F1 at or above which an answer to a withheld question counts as the
#: reference answer anyway, from the reader's own knowledge or another passage.
GOLD_ANYWAY_F1 = 0.5


def condition(rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]) -> dict[str, Any]:
    """Summarise one arm under one prompt and one withhold mode.

    ``f1_unflagged_only`` scores a flagged answer as a decline, which is what a
    caller gets when it hides every answer marked unsure.
    """
    n = len(rows)
    share = {
        kind: sum(labels[r["question_id"]] == kind for r in rows) / n for kind in ("declined", "flagged", "answered")
    }
    by_kind: dict[str, list[float]] = {"flagged": [], "answered": []}
    for r in rows:
        kind = labels[r["question_id"]]
        if kind in by_kind:
            by_kind[kind].append(r["f1"])
    spoke = [r for r in rows if labels[r["question_id"]] != "declined"]
    return {
        "questions": n,
        "declined": round(share["declined"], 4),
        "flagged": round(share["flagged"], 4),
        "answered_unflagged": round(share["answered"], 4),
        "f1": round(sum(r["f1"] for r in rows) / n, 4),
        "exact_match": round(sum(r["em"] for r in rows) / n, 4),
        "f1_unflagged_only": round(sum(by_kind["answered"]) / n, 4),
        "f1_of_unflagged_answers": round(sum(by_kind["answered"]) / len(by_kind["answered"]), 4)
        if by_kind["answered"]
        else None,
        "f1_of_flagged_answers": round(sum(by_kind["flagged"]) / len(by_kind["flagged"]), 4)
        if by_kind["flagged"]
        else None,
        "gold_anyway": round(sum(r["f1"] >= GOLD_ANYWAY_F1 for r in spoke) / n, 4),
    }


def combine(runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Return the combined results from runs keyed by (prompt, withhold)."""
    first = next(iter(runs.values()))
    arms = list(first["rows"])
    summary: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    per_question: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    for (prompt, withhold), run in runs.items():
        if run["test"]["model"] != first["test"]["model"]:
            raise ValueError("the runs were answered by different models")
        for arm in run["rows"]:
            rows = run["rows"][arm]
            labels = run["outcomes"][arm]
            summary[arm].setdefault(prompt, {})[withhold] = condition(rows, labels)
            per_question[arm][f"{prompt}/{withhold}"] = {
                r["question_id"]: {"outcome": labels[r["question_id"]], "f1": r["f1"], "em": r["em"]} for r in rows
            }
    return {
        "test": {
            "questions": first["test"]["questions"],
            "seed": first["test"]["seed"],
            "dataset_revision": first["test"]["dataset_revision"],
            "model": first["test"]["model"],
            "rankings_from": first["test"]["rankings_from"],
            "answer_passages": first["test"]["answer_passages"],
            "prompts": list(PROMPTS),
            "withhold": {
                "none": "answerable: the stored top five",
                "supporting": "every supporting passage dropped; the reader sees the next-ranked five",
                "final": "only the passage that states the final answer dropped; bridge passages stay",
                "everything": "no passages: a closed-book answer, run on the dense arm only",
            },
            "gold_anyway_f1": GOLD_ANYWAY_F1,
        },
        "produced_by": {**first["produced_by"], "producer": "run_musique_test.py, musique_abstention.py"},
        "summary": summary,
        "per_question": per_question,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Read the twelve runs and write the combined file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, help="path prefix of the run files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {
        (prompt, withhold): json.loads(Path(f"{args.runs}_{prompt}_{withhold}.json").read_text(encoding="utf-8"))
        for withhold in WITHHOLD
        for prompt in PROMPTS
    }
    results = combine(runs)
    args.out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    for arm, prompts in results["summary"].items():
        for prompt, modes in prompts.items():
            line = " | ".join(
                f"{w}: dec {c['declined']:.2f} flag {c['flagged']:.2f} F1 {c['f1']:.3f} gold {c['gold_anyway']:.2f}"
                for w, c in modes.items()
            )
            print(f"{arm:10s} {prompt:10s} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
