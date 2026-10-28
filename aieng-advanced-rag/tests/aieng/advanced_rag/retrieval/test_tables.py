"""The table grid, the cell facts, and the attachment metric."""

from __future__ import annotations

import pytest

from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.tables import (
    TableGrid,
    attachment_report,
    contains_value,
    is_numeric_cell,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("4,812", True),
        ("$ 4,812", True),
        ("(1,234)", True),
        ("12.5%", True),
        ("-17", True),
        ("2022", True),
        ("", False),
        ("Instruments", False),
        ("December 31, 2022", False),
        ("(In millions)", False),
        ("Note 5", False),
    ],
)
def test_is_numeric_cell(text: str, expected: bool) -> None:
    assert is_numeric_cell(text) is expected


def test_contains_value_does_not_match_a_longer_number() -> None:
    assert contains_value("| Instruments | 1,043 |", "1,043")
    assert not contains_value("| Instruments | 11,043 |", "1,043")
    assert not contains_value("| Instruments | 1,0435 |", "1,043")
    assert contains_value("$ 1,043.0", "1,043.0")


def test_header_rows_found_without_the_converter_flag(filing_grid: TableGrid) -> None:
    # Both header rows have an empty label cell and populated data cells, which
    # is the pattern that ends at "Net revenue:".
    assert filing_grid.header_rows == [0, 1]


def test_header_rows_prefer_the_converter_flag(filing_grid_flagged: TableGrid) -> None:
    assert filing_grid_flagged.header_rows == [0, 1]


def test_column_header_joins_the_period_and_the_units(filing_grid: TableGrid) -> None:
    assert filing_grid.column_header(1) == "December 31, 2022 (In millions)"


def test_row_label_walks_up_past_an_empty_label(filing_grid: TableGrid) -> None:
    # Row 3 is "Instruments"; a row whose label cell were empty would inherit it.
    assert filing_grid.row_label(3) == "Instruments"


def test_facts_cover_every_numeric_cell(filing_grid: TableGrid) -> None:
    facts = filing_grid.facts()
    assert len(facts) == 8  # four rows of data, two periods each
    fact = next(f for f in facts if f.value == "4,812")
    assert fact.row_label == "Instruments"
    assert fact.column_header == "December 31, 2022 (In millions)"
    assert all(f.is_answerable for f in facts)


def test_section_row_is_not_a_header_row(make_grid) -> None:
    # "Net revenue:" has a label and empty data cells, so the header prefix must
    # stop before it rather than swallowing it.
    grid = make_grid(
        [
            ["", "2022"],
            ["Net revenue:", ""],
            ["Instruments", "4,812"],
        ]
    )
    assert grid.header_rows == [0]
    assert grid.row_label(2) == "Instruments"


def test_attachment_report_penalises_a_split_that_drops_the_header(
    filing_grid: TableGrid,
) -> None:
    facts = filing_grid.facts()
    header_only = Chunk(
        "d::0", "d", filing_grid.to_markdown(rows=[0, 1], repeat_headers=False), 0, 1
    )
    data_only = Chunk(
        "d::1",
        "d",
        filing_grid.to_markdown(rows=[3, 4, 5, 6], repeat_headers=False),
        1,
        2,
    )
    report = attachment_report([header_only, data_only], facts)
    assert report["attachment_rate"] == 0.0
    assert report["cells_never_complete"] == len(facts)
    # The data chunk still carries the row labels, which is what makes the
    # failure specific: the value and its row survive, the period does not.
    assert report["label_only_pairs"] == report["chunk_cell_pairs"]


def test_attachment_report_rewards_repeated_headers(filing_grid: TableGrid) -> None:
    facts = filing_grid.facts()
    parts = [
        Chunk(
            "d::0", "d", filing_grid.to_markdown(rows=[3, 4], repeat_headers=True), 0, 1
        ),
        Chunk(
            "d::1", "d", filing_grid.to_markdown(rows=[5, 6], repeat_headers=True), 0, 1
        ),
    ]
    report = attachment_report(parts, facts)
    assert report["attachment_rate"] == 1.0
    assert report["cell_recoverable_rate"] == 1.0


def test_attachment_report_scores_every_appearance_of_a_cell(
    filing_grid: TableGrid,
) -> None:
    facts = [f for f in filing_grid.facts() if f.value == "4,812"]
    complete = Chunk(
        "d::0", "d", filing_grid.to_markdown(rows=[3], repeat_headers=True), 0, 1
    )
    broken = Chunk("d::1", "d", "Instruments | 4,812", 0, 1)
    report = attachment_report([complete, broken], facts)
    assert report["chunk_cell_pairs"] == 2
    assert report["attachment_rate"] == 0.5
    # One chunk holds the whole fact, so the fact is still recoverable.
    assert report["cell_recoverable_rate"] == 1.0


def test_facts_without_a_label_are_not_scored(make_grid) -> None:
    grid = make_grid([["", "2022"], ["", "4,812"]])
    report = attachment_report([Chunk("d::0", "d", "4,812", 0, 1)], grid.facts())
    assert report["scored_cells"] == 0
    assert report["attachment_rate"] == 0.0
