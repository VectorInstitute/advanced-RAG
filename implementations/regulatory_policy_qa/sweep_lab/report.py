"""Writing the sweep out: raw JSON that every number traces to, and the tables.

Every figure quoted in ``THRESHOLDS.md`` comes from a file under ``outputs/``,
so the memo can be checked against the run rather than believed. The rendering
here is deliberately thin: ``SweepRow`` already knows how to render itself, and
this only orders the rows and stacks the header on them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.sweep import TABLE_HEADER, SweepRow


def markdown_table(rows: Sequence[SweepRow], *, by_score: bool = False) -> str:
    """Render sweep rows as one Markdown table.

    Parameters
    ----------
    rows : sequence of SweepRow
        Rows to render, controls included.
    by_score : bool, optional
        Order by balanced accuracy, best first, instead of the order given.
        Controls keep their place at the bottom either way, because a control
        is a floor to read against rather than a candidate.

    Returns
    -------
    str
        The table, header included.
    """
    controls = [row for row in rows if row.sufficient is None]
    candidates = [row for row in rows if row.sufficient is not None]
    if by_score:
        candidates = sorted(candidates, key=lambda row: -row.balanced_accuracy)
    body = "\n".join(row.table_row() for row in candidates + controls)
    return f"{TABLE_HEADER}\n{body}"


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON, creating parent directories.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : Any
        Anything ``json.dumps`` accepts.

    Returns
    -------
    Path
        The path written, so a caller can log it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
