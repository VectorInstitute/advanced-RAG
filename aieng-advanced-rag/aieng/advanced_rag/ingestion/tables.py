"""How a converted table is stored on disk, and the check that gates one form.

A converted filing table can be kept as a markdown grid, as a page-cropped
image, or as both, and the choice changes what a retriever can do with it. A
grid is searchable by a sparse index and cheap to put in a prompt. An image
keeps the layout a vision model needs and survives a converter that flattened
the grid. Neither one is right for every reference implementation, so the form
is a setting with four values rather than a decision baked into the harness.

:class:`TableStorage` is that setting. ``MARKDOWN_WITH_IMAGE`` is the only value
that annotates the grid, and the annotation is what makes the ingest-time check
worth running: a metadata header that points at the image, and a description of
the table's shape, are both claims about a table the harness has not verified.
:func:`check_values_on_page` verifies it, by asking whether the table's numeric
cells appear in the page's own text.

The check reuses :mod:`aieng.advanced_rag.retrieval.tables`. ``contains_value``
there already knows that ``1,043`` must not match inside ``11,043``, and
``attachment_report`` already scores a text against a table's cell facts. A
second numeric matcher here would drift from that one.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from aieng.advanced_rag.cache import JsonValue
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.tables import (
    CellFact,
    TableGrid,
    attachment_report,
    contains_value,
)


class TableStorage(enum.Enum):
    """What the harness keeps on disk for every table it converts.

    The four values produce four different page files, so the setting is
    readable off the output rather than only out of the run that made it.
    """

    #: The markdown grid alone. No image is rendered and none is kept.
    MARKDOWN = "markdown"

    #: The grid, a metadata header pointing at the image, a description of the
    #: table's shape, and the image. The only value whose output carries claims
    #: the harness has to check, so the only one that runs the ingest check.
    MARKDOWN_WITH_IMAGE = "markdown_with_image"

    #: The image alone, referenced from the page file. The grid is dropped, so
    #: a reader resolves the table by looking at the image at query time.
    IMAGE_ONLY = "image_only"

    #: The grid and the image, with no header and no description. Both forms
    #: are on the page, so a query picks either one without a second pass.
    BOTH = "both"

    @property
    def keeps_markdown(self) -> bool:
        """Return whether the page file carries the markdown grid."""
        return self is not TableStorage.IMAGE_ONLY

    @property
    def keeps_image(self) -> bool:
        """Return whether an image is rendered and kept beside the page files."""
        return self is not TableStorage.MARKDOWN

    @property
    def describes(self) -> bool:
        """Return whether the page file carries the header and the description.

        This also turns on :func:`check_values_on_page`, because the header and
        the description are the output's only unverified claims about the table.
        """
        return self is TableStorage.MARKDOWN_WITH_IMAGE


@dataclass(frozen=True)
class TableValueCheck:
    """Whether a table's numeric cells appear in the text of its own page.

    Parameters
    ----------
    table_ref : str
        The Docling ``self_ref`` of the table, for example ``#/tables/0``.
    page : int or None
        The page the table sits on, or None when the converter recorded no
        provenance for it.
    missing_values : tuple of str
        Cell values the page text does not hold, in document order. Empty when
        the check passed.
    report : dict
        The ``attachment_report`` scores for the page text, kept so a failure
        says whether the values were absent or only their labels were.
    """

    table_ref: str
    page: int | None
    missing_values: tuple[str, ...]
    report: dict[str, float | int]

    @property
    def passed(self) -> bool:
        """Return whether every numeric cell was found in the page text."""
        return not self.missing_values

    def as_json(self) -> dict[str, JsonValue]:
        """Return the check as a JSON object, for the cached manifest."""
        return {
            "missing_values": list(self.missing_values),
            "page": self.page,
            "passed": self.passed,
            "report": cast(dict[str, JsonValue], dict(self.report)),
            "table_ref": self.table_ref,
        }

    @classmethod
    def from_json(cls, fields: JsonValue) -> TableValueCheck:
        """Rebuild a check read back out of the cached manifest.

        Parameters
        ----------
        fields : JsonValue
            One entry of the manifest's ``table_checks`` list.

        Returns
        -------
        TableValueCheck
            The corresponding check.

        Raises
        ------
        ValueError
            When the entry is not an object.
        """
        if not isinstance(fields, dict):
            raise ValueError("a table check must be a JSON object")
        page = fields.get("page")
        report = fields.get("report")
        missing = fields.get("missing_values")
        return cls(
            table_ref=str(fields.get("table_ref", "")),
            page=int(page) if isinstance(page, int) else None,
            missing_values=tuple(
                str(value) for value in (missing if isinstance(missing, list) else ())
            ),
            report={
                key: value
                for key, value in (report if isinstance(report, dict) else {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        )


def check_values_on_page(
    doc_id: str,
    table_ref: str,
    grid: TableGrid,
    page_text: str,
    page: int | None = None,
) -> TableValueCheck:
    """Check that a table's numeric cells appear in its page's own text.

    The converter reads a table's cells out of the same page it reads the
    surrounding prose from, so a value that is nowhere in the page text is a
    sign the table structure came from somewhere else: a mis-stitched grid, a
    carried-over header, or a cell the layout model invented. The check is
    textual and generous in the same way ``attachment_report`` is, because the
    page text is the only ground truth available at ingest time.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document, used for the synthetic chunk id.
    table_ref : str
        The Docling ``self_ref`` of the table.
    grid : TableGrid
        The table, already rebuilt as a grid.
    page_text : str
        The page's text, with this table's own rendering excluded. Including it
        would make the check pass by construction.
    page : int or None, optional
        The page number, carried into the result for the report.

    Returns
    -------
    TableValueCheck
        The result. A caller records a failure and continues; one bad table
        must not end the ingestion of a corpus.
    """
    facts: Sequence[CellFact] = grid.facts()
    missing = tuple(
        fact.value for fact in facts if not contains_value(page_text, fact.value)
    )
    probe = Chunk(
        chunk_id=f"{table_ref}::page",
        doc_id=doc_id,
        text=page_text,
        start=0,
        end=len(page_text),
    )
    return TableValueCheck(
        table_ref=table_ref,
        page=page,
        missing_values=missing,
        report=attachment_report([probe], facts),
    )


def metadata_header(
    table_ref: str,
    page: int | None,
    grid: TableGrid,
    image_path: str | None,
) -> str:
    """Return the HTML comment that heads a ``MARKDOWN_WITH_IMAGE`` table.

    A comment rather than a visible line, so a page file stays readable as
    markdown and a retriever that indexes the rendered text does not score the
    header as content.

    Parameters
    ----------
    table_ref : str
        The Docling ``self_ref`` of the table.
    page : int or None
        The page the table sits on.
    grid : TableGrid
        The table, for its dimensions.
    image_path : str or None
        The image's path relative to the page file, or None when no image was
        rendered.

    Returns
    -------
    str
        One line, beginning ``<!-- table``.
    """
    fields = [
        f"ref={table_ref}",
        f"page={page if page is not None else 'unknown'}",
        f"rows={grid.num_rows}",
        f"cols={grid.num_cols}",
        f"image={image_path if image_path else 'none'}",
    ]
    return "<!-- table " + " ".join(fields) + " -->"


def describe(grid: TableGrid) -> str:
    """Return a one-line description of what a table holds.

    Derived from the grid rather than written by a model, so it costs nothing
    and says the same thing on every run. It names the row labels and the column
    headers, which is what a reader needs to decide whether to open the image.

    Parameters
    ----------
    grid : TableGrid
        The table.

    Returns
    -------
    str
        A sentence, or a bare row and column count when the grid carries
        neither labels nor headers.
    """
    header_rows = set(grid.header_rows)
    label_column = grid.label_column
    labels: list[str] = []
    for row in range(grid.num_rows):
        if row in header_rows:
            continue
        label = grid.grid[row][label_column].strip()
        if label and label not in labels:
            labels.append(label)
    headers: list[str] = []
    for column in range(grid.num_cols):
        if column == label_column:
            continue
        header = grid.column_header(column).strip()
        if header and header not in headers:
            headers.append(header)

    data_rows = grid.num_rows - len(header_rows)
    parts = [f"Table of {data_rows} data rows by {grid.num_cols} columns."]
    if labels:
        parts.append("Rows: " + ", ".join(labels) + ".")
    if headers:
        parts.append("Columns: " + ", ".join(headers) + ".")
    return " ".join(parts)


def render_table_block(
    storage: TableStorage,
    rendered_markdown: str,
    *,
    table_ref: str,
    page: int | None,
    grid: TableGrid,
    image_path: str | None,
) -> str:
    """Return the block a page file carries for one table.

    Parameters
    ----------
    storage : TableStorage
        The setting in force.
    rendered_markdown : str
        The table as ``table_aware_chunks`` rendered it, which already carries
        the section heading above it and repeats the header rows.
    table_ref : str
        The Docling ``self_ref`` of the table.
    page : int or None
        The page the table sits on.
    grid : TableGrid
        The table, for the header and the description.
    image_path : str or None
        The image's path relative to the page file, or None when no image was
        rendered. A storage value that keeps images falls back to the grid when
        this is None, so a failed render never drops the data.

    Returns
    -------
    str
        The block, with no trailing newline.
    """
    # The setting decides, not the caller: MARKDOWN never references an image
    # even when one happens to have been rendered for a different setting.
    keep_image = storage.keeps_image and bool(image_path)
    reference = f"![{table_ref}]({image_path})" if keep_image else None
    blocks: list[str] = []

    if storage.describes:
        blocks.append(metadata_header(table_ref, page, grid, image_path))
        blocks.append(describe(grid))

    if storage.keeps_markdown or reference is None:
        blocks.append(rendered_markdown)
    if reference is not None:
        blocks.append(reference)

    return "\n\n".join(block for block in blocks if block.strip())


__all__ = [
    "TableStorage",
    "TableValueCheck",
    "check_values_on_page",
    "describe",
    "metadata_header",
    "render_table_block",
]
