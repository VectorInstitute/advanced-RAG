"""The four-valued table setting, its rendering, and the ingest-time check.

The grids here are built from rows the same way the retrieval tests build
theirs, because the check under test reads a ``TableGrid`` and nothing about a
real converter changes what it reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aieng.advanced_rag.ingestion.tables import (
    TableStorage,
    TableValueCheck,
    check_values_on_page,
    describe,
    metadata_header,
    render_table_block,
)
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


ROWS = [
    ["", "December 31, 2022"],
    ["", "(In millions)"],
    ["Instruments", "4,812"],
    ["Retail", "5,318"],
]


def build_grid(rows: list[list[str]], header_rows: set[int]) -> TableGrid:
    cells = [
        FakeCell(
            text=text,
            start_row_offset_idx=row_index,
            start_col_offset_idx=column_index,
            column_header=row_index in header_rows,
        )
        for row_index, row in enumerate(rows)
        for column_index, text in enumerate(row)
    ]
    return TableGrid(cells, len(rows), max(len(row) for row in rows))


@pytest.fixture
def grid() -> TableGrid:
    return build_grid(ROWS, header_rows={0, 1})


def test_the_setting_has_exactly_four_values() -> None:
    assert [member.value for member in TableStorage] == [
        "markdown",
        "markdown_with_image",
        "image_only",
        "both",
    ]


@pytest.mark.parametrize(
    ("storage", "keeps_markdown", "keeps_image", "describes"),
    [
        (TableStorage.MARKDOWN, True, False, False),
        (TableStorage.MARKDOWN_WITH_IMAGE, True, True, True),
        (TableStorage.IMAGE_ONLY, False, True, False),
        (TableStorage.BOTH, True, True, False),
    ],
)
def test_each_value_answers_what_it_keeps(
    storage: TableStorage,
    keeps_markdown: bool,
    keeps_image: bool,
    describes: bool,
) -> None:
    assert storage.keeps_markdown is keeps_markdown
    assert storage.keeps_image is keeps_image
    assert storage.describes is describes


def test_the_check_passes_when_every_value_is_in_the_page_text(
    grid: TableGrid,
) -> None:
    page_text = "Instruments revenue was 4,812 million and Retail was 5,318 million."
    check = check_values_on_page("acme", "#/tables/0", grid, page_text, page=1)
    assert check.passed
    assert check.missing_values == ()
    assert check.report["scored_cells"] == 2


def test_the_check_names_the_values_the_page_text_does_not_hold(
    grid: TableGrid,
) -> None:
    check = check_values_on_page(
        "acme",
        "#/tables/0",
        grid,
        "The discussion below covers operating results.",
        page=2,
    )
    assert not check.passed
    assert check.missing_values == ("4,812", "5,318")
    assert check.report["cells_never_complete"] == 2


def test_the_check_does_not_credit_a_value_inside_a_longer_number(
    grid: TableGrid,
) -> None:
    # `contains_value` is reused precisely so 4,812 does not match 14,812.
    check = check_values_on_page(
        "acme", "#/tables/0", grid, "Revenue of 14,812 and 15,318.", page=1
    )
    assert check.missing_values == ("4,812", "5,318")


def test_a_check_survives_the_round_trip_through_the_manifest(
    grid: TableGrid,
) -> None:
    check = check_values_on_page("acme", "#/tables/0", grid, "nothing here", page=3)
    assert TableValueCheck.from_json(check.as_json()) == check


def test_the_metadata_header_names_the_table_the_page_and_the_image(
    grid: TableGrid,
) -> None:
    header = metadata_header("#/tables/0", 4, grid, "../tables/t000.png")
    assert header == (
        "<!-- table ref=#/tables/0 page=4 rows=4 cols=2 image=../tables/t000.png -->"
    )


def test_the_metadata_header_says_so_when_there_is_no_page_or_image(
    grid: TableGrid,
) -> None:
    header = metadata_header("#/tables/0", None, grid, None)
    assert "page=unknown" in header
    assert "image=none" in header


def test_the_description_names_the_row_labels_and_the_column_headers(
    grid: TableGrid,
) -> None:
    assert describe(grid) == (
        "Table of 2 data rows by 2 columns. Rows: Instruments, Retail. "
        "Columns: December 31, 2022 (In millions)."
    )


@pytest.mark.parametrize(
    ("storage", "expected"),
    [
        (TableStorage.MARKDOWN, ["grid"]),
        (TableStorage.MARKDOWN_WITH_IMAGE, ["header", "description", "grid", "image"]),
        (TableStorage.IMAGE_ONLY, ["image"]),
        (TableStorage.BOTH, ["grid", "image"]),
    ],
)
def test_each_value_renders_a_different_block(
    grid: TableGrid, storage: TableStorage, expected: list[str]
) -> None:
    block = render_table_block(
        storage,
        "| Instruments | 4,812 |",
        table_ref="#/tables/0",
        page=1,
        grid=grid,
        image_path="../tables/t000.png",
    )
    present = {
        "header": "<!-- table ref=" in block,
        "description": "Table of 2 data rows" in block,
        "grid": "| Instruments | 4,812 |" in block,
        "image": "![#/tables/0](../tables/t000.png)" in block,
    }
    assert sorted(name for name, found in present.items() if found) == sorted(expected)


def test_a_missing_image_falls_back_to_the_grid_rather_than_dropping_it(
    grid: TableGrid,
) -> None:
    block = render_table_block(
        TableStorage.IMAGE_ONLY,
        "| Instruments | 4,812 |",
        table_ref="#/tables/0",
        page=1,
        grid=grid,
        image_path=None,
    )
    assert block == "| Instruments | 4,812 |"
