"""Render memo tables out of a written sweep, so no number is transcribed by hand.

`run_threshold_sweep.py` writes the whole grid to `outputs/sweep_<corpus>.json`.
This reads that file back and prints the slices through it that `THRESHOLDS.md`
quotes: a sweep of the answer threshold at a fixed rewrite threshold, a set of
named operating points, and the controls.

Examples
--------
Sweep the rewrite threshold at the shipped answer threshold::

    python3 summarise_sweep.py outputs/sweep_contract-nli.json \
        --grader heuristic --attempts 2 --sufficient 0.6

Compare named points against the controls::

    python3 summarise_sweep.py outputs/sweep_contract-nli.json \
        --grader llm --attempts 2 --points 0.60/0.30 1.00/0.50
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.sweep import SweepRow
from sweep_lab import markdown_table


def load_rows(payload: dict[str, Any], grader: str) -> tuple[list[SweepRow], list[SweepRow]]:
    """Return one arm's grid rows and its controls, rebuilt from the file.

    Raises
    ------
    KeyError
        If the payload holds no such grader arm.
    ValueError
        If the arm is an all-null control, which carries no sweep rows.
    """
    arm = payload["arms"][grader]
    if arm.get("null_only"):
        raise ValueError(
            f"the {grader} arm of this file is an all-null control; it has no "
            "answerable side and no sweep rows to render"
        )
    return (
        [SweepRow.from_dict(row) for row in arm["rows"]],
        [SweepRow.from_dict(row) for row in arm["controls"]],
    )


def select(
    rows: Sequence[SweepRow],
    *,
    attempts: int,
    sufficient: float | None = None,
    insufficient: float | None = None,
    points: Sequence[str] = (),
) -> list[SweepRow]:
    """Return the rows a table should show, in grid order.

    Parameters
    ----------
    rows : sequence of SweepRow
        Every row in the arm.
    attempts : int
        Keep rows at this attempt budget.
    sufficient : float or None, optional
        Keep rows at this answer threshold, which is how the rewrite threshold
        is swept on its own. ``None`` keeps every answer threshold.
    insufficient : float or None, optional
        Keep rows at this rewrite threshold, which is how the answer threshold
        is swept on its own. ``None`` keeps every rewrite threshold.
    points : sequence of str, optional
        Labels like ``"0.60/0.30"``. Naming any switches off the other filter
        and keeps exactly those operating points, which is how a memo puts the
        recommendation next to the shipped default.

    Returns
    -------
    list of SweepRow
        The rows to render.
    """
    at_budget = [row for row in rows if row.max_attempts == attempts]
    if points:
        wanted = {point.replace("/", " / ") for point in points}
        return [row for row in at_budget if row.label in wanted]
    return [
        row
        for row in at_budget
        if (sufficient is None or row.sufficient == sufficient)
        and (insufficient is None or row.insufficient == insufficient)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Print one Markdown table for the requested slice through a sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--grader", default="heuristic")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--sufficient", type=float, default=None)
    parser.add_argument("--insufficient", type=float, default=None)
    parser.add_argument("--points", nargs="*", default=[])
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--by-score", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows, controls = load_rows(payload, args.grader)
    chosen = select(
        rows,
        attempts=args.attempts,
        sufficient=args.sufficient,
        insufficient=args.insufficient,
        points=args.points,
    )
    if args.no_controls:
        controls = []
    print(markdown_table(chosen + controls, by_score=args.by_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
