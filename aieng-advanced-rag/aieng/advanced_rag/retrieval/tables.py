"""Table structure for filings, and a metric for whether chunking preserved it.

The problem a table-aware chunker exists to solve: a financial table exports
with row labels and column headers attached, but the period header can sit two
rows above its values. A chunker that cuts inside the table separates ``4,812``
from ``December 31, 2022``, and the chunk stops being answerable. A fixed-width
window does not know a table is there.

``TableGrid`` rebuilds a Docling table as a grid and says, for each numeric
cell, which column header and row label make it readable. ``attachment_report``
then scores any chunk set on whether it kept those three strings together.

The metric is structural because filings carry no retrieval judgments. For every
numeric cell in every table, find the chunks that contain the cell's text, and
ask whether they also contain the cell's column header and row label. The check
is textual and chunker-agnostic, so the same function scores a naive window and
a table-aware chunk without either being given credit for how it was built. It
is generous to the naive chunker: a header string that happens to appear
elsewhere in the window counts as attached.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .corpus import Chunk


# A cell is numeric when it is a number, possibly with a currency symbol,
# thousands separators, a percentage, or accounting parentheses for negatives.
_NUMERIC = re.compile(r"^[\s$€£(]*-?[\d][\d,\.]*\s*[%)]*[\s)]*$")
_HAS_DIGIT = re.compile(r"\d")


def contains_value(text: str, value: str) -> bool:
    """Return whether ``text`` holds ``value`` as a whole figure.

    Plain substring matching is wrong for numbers: ``1,043`` is a substring of
    ``11,043``, so a chunk holding a different segment's revenue would be
    credited with holding this one. The value has to be bounded by something
    that is not part of a number.

    Parameters
    ----------
    text : str
        The chunk text to search.
    value : str
        The cell value to look for.

    Returns
    -------
    bool
        True when the value appears as a complete number.
    """
    pattern = r"(?<![\d.,])" + re.escape(value.strip()) + r"(?![\d.,]*\d)"
    return re.search(pattern, text) is not None


def is_numeric_cell(text: str) -> bool:
    """Return whether a cell holds a number rather than a label.

    Years on their own are numbers by this test, which is correct: in a filing a
    bare ``2022`` in a data column is usually a value, and when it is a header it
    is excluded by being in a header row rather than by its shape.

    Parameters
    ----------
    text : str
        The cell text.

    Returns
    -------
    bool
        True when the cell reads as a figure.
    """
    stripped = text.strip()
    if not stripped or not _HAS_DIGIT.search(stripped):
        return False
    return bool(_NUMERIC.match(stripped))


@dataclass(frozen=True)
class CellFact:
    """One numeric cell and the strings that make it readable.

    Parameters
    ----------
    value : str
        The cell text.
    header_parts : tuple of str
        One string per header row, because a filing header is often split
        across rows: the period on one, the units on the next. A chunk has to
        carry every part to make the cell readable, and each part is matched on
        its own since the rendered table puts them on separate lines.
    row_label : str
        The label of the cell's row.
    row : int
        Row index in the grid.
    column : int
        Column index in the grid.
    table_span : tuple of int, or None, optional
        Where the cell's table sits in the serialised document.
    """

    value: str
    header_parts: tuple[str, ...]
    row_label: str
    row: int
    column: int
    table_span: tuple[int, int] | None = None

    @property
    def column_header(self) -> str:
        """Return the header parts as one string, for display."""
        return " ".join(self.header_parts)

    @property
    def is_answerable(self) -> bool:
        """Return whether this cell has both of the strings it needs."""
        return bool(self.value and self.header_parts and self.row_label)


class TableGrid:
    """A Docling table as a dense grid, with headers and labels resolved.

    Parameters
    ----------
    cells : sequence
        Docling ``TableCell`` objects, or anything with the same offset and
        header attributes.
    num_rows : int
        Grid height.
    num_cols : int
        Grid width.
    """

    def __init__(self, cells: Sequence[Any], num_rows: int, num_cols: int) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid: list[list[str]] = [
            ["" for _ in range(num_cols)] for _ in range(num_rows)
        ]
        self.flagged_header_rows: set[int] = set()
        self.flagged_label_cols: set[int] = set()
        for cell in cells:
            text = (getattr(cell, "text", "") or "").strip()
            row_range = range(
                cell.start_row_offset_idx,
                min(cell.end_row_offset_idx, num_rows),
            )
            for row in row_range:
                column_range = range(
                    cell.start_col_offset_idx,
                    min(cell.end_col_offset_idx, num_cols),
                )
                for column in column_range:
                    if text:
                        self.grid[row][column] = text
                    if getattr(cell, "column_header", False):
                        self.flagged_header_rows.add(row)
                    if getattr(cell, "row_header", False):
                        self.flagged_label_cols.add(column)

    @classmethod
    def from_table(cls, table: Any) -> TableGrid:
        """Build a grid from a Docling ``TableItem``."""
        data = table.data
        return cls(
            list(getattr(data, "table_cells", None) or []),
            int(getattr(data, "num_rows", 0) or 0),
            int(getattr(data, "num_cols", 0) or 0),
        )

    @property
    def label_column(self) -> int:
        """Return the column holding row labels."""
        if self.flagged_label_cols:
            return min(self.flagged_label_cols)
        return 0

    @property
    def header_rows(self) -> list[int]:
        """Return the rows that hold column headers.

        Docling's ``column_header`` flag when it is set. When it is not, the
        leading rows whose label cell is empty but whose data cells are not: in
        a filing that is exactly the period row and the units row, which is the
        pattern that makes a mid-table split unanswerable.
        """
        if self.flagged_header_rows:
            return sorted(self.flagged_header_rows)
        rows: list[int] = []
        label = self.label_column
        for row in range(self.num_rows):
            values = [self.grid[row][c] for c in range(self.num_cols) if c != label]
            if self.grid[row][label].strip():
                break
            if not any(v.strip() for v in values):
                break
            rows.append(row)
        return rows

    def header_parts(self, column: int) -> tuple[str, ...]:
        """Return every distinct non-empty header string above a column.

        In a two-row filing header the period is on one row and the units on the
        next, and both are needed: ``4,812`` is only readable as a December 2022
        figure in millions.
        """
        seen: list[str] = []
        for row in self.header_rows:
            text = self.grid[row][column].strip()
            if text and text not in seen:
                seen.append(text)
        return tuple(seen)

    def column_header(self, column: int) -> str:
        """Return the header parts as one string, for display."""
        return " ".join(self.header_parts(column))

    def row_label(self, row: int) -> str:
        """Return the row's label, walking up to the nearest labelled row."""
        label = self.label_column
        for candidate in range(row, -1, -1):
            text = self.grid[candidate][label].strip()
            if text and not is_numeric_cell(text):
                return text
        return ""

    def facts(self, table_span: tuple[int, int] | None = None) -> list[CellFact]:
        """Return every numeric data cell with its column header and row label.

        Parameters
        ----------
        table_span : tuple of int, or None, optional
            Where this table sits in the serialised document. The metric uses it
            to scope a cell to the chunks that overlap its own table, so a
            number that also appears in a different table of the same filing is
            not counted against the chunker that split neither of them.

        Returns
        -------
        list of CellFact
            One entry per numeric data cell.
        """
        headers = set(self.header_rows)
        label = self.label_column
        out: list[CellFact] = []
        for row in range(self.num_rows):
            if row in headers:
                continue
            for column in range(self.num_cols):
                if column == label:
                    continue
                value = self.grid[row][column].strip()
                if not is_numeric_cell(value):
                    continue
                out.append(
                    CellFact(
                        value=value,
                        header_parts=self.header_parts(column),
                        row_label=self.row_label(row),
                        row=row,
                        column=column,
                        table_span=table_span,
                    ),
                )
        return out

    def is_section_row(self, row: int) -> bool:
        """Return whether a row labels the rows under it and holds no data.

        ``Net revenue:`` above four segment rows. A split that starts below it
        loses the only statement of what the numbers under it are.
        """
        label = self.label_column
        if not self.grid[row][label].strip():
            return False
        return not any(
            self.grid[row][c].strip() for c in range(self.num_cols) if c != label
        )

    def governing_rows(self, row: int) -> list[int]:
        """Return the rows above ``row`` that a split has to carry with it.

        Two cases, both from real filings. A continuation row has an empty label
        cell and takes its label from the row above. A section row labels a block
        and holds no data. Either one left behind makes the rows below it
        unreadable.
        """
        label = self.label_column
        needed: set[int] = set()
        if not self.grid[row][label].strip():
            for candidate in range(row - 1, -1, -1):
                if self.grid[candidate][label].strip():
                    needed.add(candidate)
                    break
        for candidate in range(row - 1, -1, -1):
            if candidate in self.header_rows:
                break
            if self.is_section_row(candidate):
                needed.add(candidate)
                break
        return sorted(needed)

    def to_markdown(
        self,
        rows: Iterable[int] | None = None,
        repeat_headers: bool = True,
    ) -> str:
        """Render the grid, optionally a row subset with its headers repeated."""
        header_rows = self.header_rows
        selected = list(rows) if rows is not None else list(range(self.num_rows))
        if repeat_headers:
            body = [r for r in selected if r not in set(header_rows)]
            selected = header_rows + body
        lines = []
        for row in selected:
            cells = [
                self.grid[row][c].replace("|", "\\|") for c in range(self.num_cols)
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def attachment_report(
    chunks: Sequence[Chunk],
    facts: Sequence[CellFact],
) -> dict[str, float | int]:
    """Score a chunk set on whether numeric cells kept what makes them readable.

    For each numeric cell, the chunks considered are those that overlap the
    cell's own table and contain the cell's value. A chunk counts as complete
    when it also holds the cell's row label and every one of its column-header
    strings. Both conditions matter and both fail differently: a mid-table split
    that keeps the label and drops the period leaves ``4,812`` attached to
    ``Instruments`` with no year, and a split that drops the label leaves a bare
    number.

    A cell can appear in more than one chunk when windows overlap, and every
    appearance is scored: a reader who retrieves the second copy is no better
    off for the first one being complete.

    ``cells_never_complete`` is the harsher statement of the same thing: cells
    for which no chunk anywhere holds the value, its label and its header
    together, so no retrieval can recover the fact.

    Parameters
    ----------
    chunks : sequence of Chunk
        The chunk set under test.
    facts : sequence of CellFact
        Numeric cells, from ``TableGrid.facts`` or ``chunkers.all_facts``.

    Returns
    -------
    dict
        Pair counts, the attachment rate, and the per-cell recoverable rate.
    """
    pairs = 0
    complete = 0
    header_only = 0
    label_only = 0
    neither = 0
    covered_cells = 0
    complete_cells = 0
    scored_facts = [f for f in facts if f.is_answerable]

    for fact in scored_facts:
        holding = [
            chunk.text
            for chunk in chunks
            if _in_table(chunk, fact) and contains_value(chunk.text, fact.value)
        ]
        if holding:
            covered_cells += 1
        cell_complete = False
        for text in holding:
            pairs += 1
            has_header = all(part in text for part in fact.header_parts)
            has_label = fact.row_label in text
            if has_header and has_label:
                complete += 1
                cell_complete = True
            elif has_header:
                header_only += 1
            elif has_label:
                label_only += 1
            else:
                neither += 1
        if cell_complete:
            complete_cells += 1

    return {
        "numeric_cells": len(facts),
        "scored_cells": len(scored_facts),
        "chunk_cell_pairs": pairs,
        "attached_pairs": complete,
        "attachment_rate": round(complete / pairs, 4) if pairs else 0.0,
        "header_only_pairs": header_only,
        "label_only_pairs": label_only,
        "neither_pairs": neither,
        "cells_covered": covered_cells,
        "cells_ever_complete": complete_cells,
        "cells_never_complete": len(scored_facts) - complete_cells,
        "cell_recoverable_rate": round(complete_cells / len(scored_facts), 4)
        if scored_facts
        else 0.0,
    }


def _in_table(chunk: Chunk, fact: CellFact) -> bool:
    """Whether a chunk overlaps the table the cell belongs to.

    A cell with no recorded table span is scored against every chunk, which is
    what the unit tests use.
    """
    if fact.table_span is None:
        return True
    start, end = fact.table_span
    return chunk.start < end and start < chunk.end
