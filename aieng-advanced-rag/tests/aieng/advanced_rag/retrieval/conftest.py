"""Shared fixtures: a fake converted table and a filing-shaped example.

The example's labels and figures are invented. Its shape is the one filings use:
the period header sits two rows above the value, so a chunker that cuts between
them separates ``4,812`` from ``December 31, 2022``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from aieng.advanced_rag.retrieval.tables import TableGrid


@dataclass
class FakeCell:
    """The subset of Docling's TableCell the grid builder reads."""

    text: str
    start_row_offset_idx: int
    start_col_offset_idx: int
    column_header: bool = False
    row_header: bool = False

    @property
    def end_row_offset_idx(self) -> int:
        """Return the exclusive end row. Every fake cell spans one row."""
        return self.start_row_offset_idx + 1

    @property
    def end_col_offset_idx(self) -> int:
        """Return the exclusive end column. Every fake cell spans one column."""
        return self.start_col_offset_idx + 1


def build_grid(rows: list[list[str]], header_rows: set[int] | None = None) -> TableGrid:
    """Build a grid from a list of rows, optionally flagging header rows."""
    cells = []
    for row_index, row in enumerate(rows):
        for column_index, text in enumerate(row):
            cells.append(
                FakeCell(
                    text=text,
                    start_row_offset_idx=row_index,
                    start_col_offset_idx=column_index,
                    column_header=bool(header_rows and row_index in header_rows),
                )
            )
    return TableGrid(cells, len(rows), max(len(r) for r in rows))


FILING_ROWS = [
    ["", "December 31, 2022", "December 31, 2021"],
    ["", "(In millions)", "(In millions)"],
    ["Net revenue:", "", ""],
    ["Instruments", "4,812", "2,957"],
    ["Retail", "5,318", "4,406"],
    ["Mobile", "3,926", "3,140"],
    ["Services", "1,873", "571"],
]


@pytest.fixture
def make_grid() -> Callable[..., TableGrid]:
    """Return the grid factory, so a test can build its own shape."""
    return build_grid


@pytest.fixture
def filing_grid() -> TableGrid:
    """Return a filing-shaped table whose period headers sit two rows above the data."""
    return build_grid(FILING_ROWS)


@pytest.fixture
def filing_grid_flagged() -> TableGrid:
    """Return the same table with the converter's column_header flag set."""
    return build_grid(FILING_ROWS, header_rows={0, 1})
