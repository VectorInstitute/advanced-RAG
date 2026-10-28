"""Chunk ingested filings and derive page-level retrieval judgments.

A filing ingested by :mod:`aieng.advanced_rag.ingestion` is a folder holding the
converter's ``document.json``. This module turns that into chunks, two ways over
the same serialised text, and says which pages each chunk covers. FinanceBench's
gold evidence is a list of zero-based pages, so a chunk is relevant when it
covers a gold page.

The two chunkings differ in one thing. ``plain`` slices the text into
overlapping character windows, as RI #1 does for contracts. ``table_aware`` keeps
a table with its section heading and repeats the header rows when it has to split
a table, so a cell value stays with its column header and row label.

Docling numbers pages from 1 and FinanceBench from 0. Every page in this module
is zero-based, and the conversion happens once, in :func:`chunk_pages`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.corpora.financebench import FinanceQuery

from .chunkers import Item, serialise, table_aware_chunks
from .corpus import Chunk


Strategy = Literal["plain", "table_aware"]


def load_document(folder: Path) -> Any:
    """Load the Docling document an ingestion folder holds.

    Parameters
    ----------
    folder : Path
        ``<root>/<doc_id>``, with a ``document.json`` the harness wrote.

    Returns
    -------
    DoclingDocument
        The converter's export, validated.
    """
    from docling_core.types.doc.document import DoclingDocument  # noqa: PLC0415

    payload = (Path(folder) / "document.json").read_text(encoding="utf-8")
    return DoclingDocument.model_validate_json(payload)


def item_pages(items: Sequence[Item]) -> list[int]:
    """Return one zero-based page per item, carrying the last known page forward.

    An item with no provenance belongs to the page that reading order put it
    after. Items before the first page-bearing one fall on page 0.
    """
    pages: list[int] = []
    current = 0
    for item in items:
        if item.page is not None:
            current = max(0, int(item.page) - 1)
        pages.append(current)
    return pages


def chunk_pages(
    chunks: Sequence[Chunk], items: Sequence[Item]
) -> dict[str, frozenset[int]]:
    """Return the zero-based pages each chunk covers.

    A chunk covers the page of every item its span overlaps. A split table part
    carries the whole table's span, so it covers the table's page.

    Parameters
    ----------
    chunks : sequence of Chunk
        Chunks of the serialised text, in any order.
    items : sequence of Item
        The items ``serialise`` returned for that text.

    Returns
    -------
    dict
        Chunk identifier to its pages. A chunk that overlaps no item covers the
        page of the nearest item before it, or page 0.
    """
    pages = item_pages(items)
    covered: dict[str, frozenset[int]] = {}
    for chunk in chunks:
        hit = {
            page
            for item, page in zip(items, pages, strict=True)
            if chunk.start < item.end and item.start < chunk.end
        }
        if not hit:
            before = [
                page
                for item, page in zip(items, pages, strict=True)
                if item.end <= chunk.start
            ]
            hit = {before[-1] if before else 0}
        covered[chunk.chunk_id] = frozenset(hit)
    return covered


@dataclass(frozen=True)
class FilingChunks:
    """One filing's chunks and the pages they cover.

    Parameters
    ----------
    doc_id : str
        The filing's name.
    chunks : list of Chunk
        In document order.
    pages : dict
        Chunk identifier to zero-based pages.
    items : list of Item
        The serialised items, for structural measures such as table-cell
        attachment.
    """

    doc_id: str
    chunks: list[Chunk]
    pages: dict[str, frozenset[int]]
    items: list[Item] = field(default_factory=list)


def chunk_filing(
    doc_id: str,
    document: Any,
    *,
    strategy: Strategy = "table_aware",
    max_chars: int = 1200,
    overlap_chars: int | None = None,
) -> FilingChunks:
    """Chunk one converted filing.

    Parameters
    ----------
    doc_id : str
        The filing's name, which becomes the chunk identifiers' prefix.
    document : DoclingDocument
        The converted filing.
    strategy : {"plain", "table_aware"}, optional
        ``plain`` for overlapping character windows over the serialised text,
        ``table_aware`` for tables kept whole or split with repeated headers.
    max_chars : int, optional
        Chunk size in characters, the same for both strategies.
    overlap_chars : int or None, optional
        Window overlap for ``plain``. ``table_aware`` does not overlap. The
        default is a sixth of ``max_chars``, which is 200 at the usual 1200.

    Returns
    -------
    FilingChunks
        The chunks, their pages, and the items.

    Raises
    ------
    ValueError
        If ``strategy`` is neither of the two.
    """
    text, items = serialise(document)
    if strategy == "plain":
        chunks = chunk_document(
            doc_id,
            text,
            chunk_chars=max_chars,
            overlap_chars=max_chars // 6 if overlap_chars is None else overlap_chars,
        )
    elif strategy == "table_aware":
        chunks = table_aware_chunks(doc_id, items, max_chars=max_chars)
    else:
        raise ValueError(f"unknown strategy {strategy!r}: use 'plain' or 'table_aware'")
    return FilingChunks(doc_id, chunks, chunk_pages(chunks, items), items)


def relevant_chunk_ids(
    pages: Mapping[str, frozenset[int]], gold_pages: Iterable[int]
) -> set[str]:
    """Return the chunks that cover at least one gold page."""
    gold = set(gold_pages)
    return {chunk_id for chunk_id, covered in pages.items() if covered & gold}


@dataclass(frozen=True)
class FilingSlice:
    """Chunks, queries, and judgments over a set of ingested filings.

    Parameters
    ----------
    chunks : list of Chunk
        Every chunk of every filing, ready to index.
    queries : list of FinanceQuery
        The questions whose filing was supplied.
    qrels : dict
        Query identifier to its relevant chunk identifiers. A query with no
        relevant chunk is absent from the scored set, and is listed in
        ``uncovered`` instead of vanishing.
    uncovered : list of str
        Identifiers of queries no chunk could answer: their filing was not
        supplied, or every gold page produced no content, as with an image-only
        page.
    filings : dict
        Doc identifier to its :class:`FilingChunks`.
    """

    chunks: list[Chunk]
    queries: list[FinanceQuery]
    qrels: dict[str, set[str]]
    uncovered: list[str]
    filings: dict[str, FilingChunks]


def build_filing_slice(
    queries: Iterable[FinanceQuery],
    folders: Mapping[str, Path],
    *,
    strategy: Strategy = "table_aware",
    max_chars: int = 1200,
) -> FilingSlice:
    """Chunk the supplied filings and derive judgments from gold pages.

    Parameters
    ----------
    queries : iterable of FinanceQuery
        Questions to judge. One whose filing is not in ``folders`` is reported in
        ``uncovered`` rather than dropped silently.
    folders : mapping
        Doc identifier to its ingestion folder.
    strategy : {"plain", "table_aware"}, optional
        How to chunk each filing.
    max_chars : int, optional
        Chunk size in characters.

    Returns
    -------
    FilingSlice
        Chunks, queries, judgments, and the uncovered query identifiers.
    """
    wanted = list(queries)
    filings = {
        doc_id: chunk_filing(
            doc_id, load_document(folder), strategy=strategy, max_chars=max_chars
        )
        for doc_id, folder in folders.items()
        if any(query.doc_id == doc_id for query in wanted)
    }
    kept: list[FinanceQuery] = []
    qrels: dict[str, set[str]] = {}
    uncovered: list[str] = []
    for query in wanted:
        filing = filings.get(query.doc_id)
        relevant = (
            relevant_chunk_ids(filing.pages, query.gold_pages) if filing else set()
        )
        if relevant:
            kept.append(query)
            qrels[query.query_id] = relevant
        else:
            uncovered.append(query.query_id)
    return FilingSlice(
        chunks=[chunk for filing in filings.values() for chunk in filing.chunks],
        queries=kept,
        qrels=qrels,
        uncovered=uncovered,
        filings=filings,
    )


__all__ = [
    "FilingChunks",
    "FilingSlice",
    "Strategy",
    "build_filing_slice",
    "chunk_filing",
    "chunk_pages",
    "item_pages",
    "load_document",
    "relevant_chunk_ids",
]
