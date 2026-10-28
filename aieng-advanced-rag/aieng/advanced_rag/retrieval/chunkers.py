"""The chunking strategies, all offset-preserving.

Four strategies split plain text: fixed character windows, recursive token
splitting, semantic boundaries, and sentence windows. A fifth consumes a
converted document instead of a string: ``table_aware_chunks`` chunks a filing
without ever cutting a table away from its headers.

Two more strategies are not splitters and live elsewhere. Late chunking changes
how a span is embedded (``late``), and contextual retrieval changes what text is
embedded for a span (``contextual``). Both take their spans from a splitter
here, which is the reason they can be compared on the same judgments.

Every chunker returns ``corpus.Chunk``, so a chunk carries the character offsets
of the region it came from and the shared metrics can score it directly.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .corpus import Chunk, chunk_contract
from .tables import CellFact, TableGrid
from .tokens import TokenCounter


# Ordered from the strongest structural boundary to the weakest. The recursive
# splitter takes the first separator that actually divides a region, so a
# paragraph break is preferred over a line break, and a hard character cut is
# the last resort.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")

_SENTENCE_END = re.compile(r"(?<=[.!?;:])[\s]+|\n{2,}")

# The control strategy. ``corpus.chunk_contract`` is the same function under the
# name the CUAD slice builder calls it by. The chunking experiment carried a
# second copy and asserted the two agreed; they are one function here.
fixed_chars = chunk_contract


@dataclass
class ChunkingResult:
    """Chunks plus what they cost to produce.

    Parameters
    ----------
    chunks : list of Chunk
        The result.
    seconds : float
        Wall time of the chunking call.
    strategy : str
        Name of the strategy, for the report.
    params : dict
        The settings the strategy ran with.
    """

    chunks: list[Chunk]
    seconds: float
    strategy: str
    params: dict[str, object]


def _atomic_spans(
    text: str,
    start: int,
    end: int,
    separators: Sequence[str],
    budget: int,
    counter: TokenCounter,
    max_chars_per_token: int = 12,
) -> list[tuple[int, int]]:
    """Split ``[start, end)`` until every piece fits ``budget`` tokens."""
    if end <= start:
        return []
    if counter.count(start, end) <= budget:
        return [(start, end)]

    for position, separator in enumerate(separators):
        if not separator:
            continue
        pieces = _split_keeping_separator(text, start, end, separator)
        if len(pieces) < 2:
            continue
        out: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            out.extend(
                _atomic_spans(
                    text,
                    piece_start,
                    piece_end,
                    separators[position + 1 :],
                    budget,
                    counter,
                ),
            )
        return out

    # No separator divides this region, so cut on characters. The width is an
    # over-estimate of the token budget in characters; the recursion below
    # re-checks and cuts again if a piece is still too long.
    width = max(budget * max_chars_per_token, 1)
    out = []
    for cut in range(start, end, width):
        piece_end = min(cut + width, end)
        if counter.count(cut, piece_end) <= budget or piece_end - cut <= 1:
            out.append((cut, piece_end))
        else:
            out.extend(
                _atomic_spans(
                    text,
                    cut,
                    piece_end,
                    (),
                    budget,
                    counter,
                    max_chars_per_token // 2,
                ),
            )
    return out


def _split_keeping_separator(
    text: str,
    start: int,
    end: int,
    separator: str,
) -> list[tuple[int, int]]:
    """Cut on ``separator``, leaving it attached to the piece before it."""
    spans: list[tuple[int, int]] = []
    cursor = start
    search = start
    while True:
        found = text.find(separator, search, end)
        if found < 0:
            break
        cut = found + len(separator)
        if cut > cursor:
            spans.append((cursor, cut))
            cursor = cut
        search = cut
    if cursor < end:
        spans.append((cursor, end))
    return spans


def recursive_tokens(
    doc_id: str,
    text: str,
    counter: TokenCounter,
    *,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split recursively on a separator hierarchy, to a token budget.

    It divides on the strongest structural boundary that applies, recursing into
    pieces that are still over budget, then merges adjacent pieces greedily up to
    ``chunk_tokens`` and repeats the tail of the previous chunk for
    ``overlap_tokens``.

    Chunks are contiguous spans of the source, so a chunk carries the offsets of
    its first and last piece and nothing is reflowed.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    text : str
        The whole document.
    counter : TokenCounter
        A counter built over this document, from ``tokens``.
    chunk_tokens : int, optional
        Token budget per chunk.
    overlap_tokens : int, optional
        Tokens each chunk repeats from the one before it.
    separators : sequence of str, optional
        Boundary hierarchy, strongest first.

    Returns
    -------
    list of Chunk
        Chunks in document order.

    Raises
    ------
    ValueError
        If ``chunk_tokens`` is not positive, or ``overlap_tokens`` is outside
        ``[0, chunk_tokens)``.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

    pieces = _atomic_spans(text, 0, len(text), tuple(separators), chunk_tokens, counter)
    pieces = [(s, e) for s, e in pieces if text[s:e].strip()]
    if not pieces:
        return []

    chunks: list[Chunk] = []
    index = 0
    position = 0
    while position < len(pieces):
        group_start = pieces[position][0]
        group_end = pieces[position][1]
        cursor = position + 1
        while cursor < len(pieces):
            candidate_end = pieces[cursor][1]
            if counter.count(group_start, candidate_end) > chunk_tokens:
                break
            group_end = candidate_end
            cursor += 1

        chunks.append(
            Chunk(
                f"{doc_id}::{index:04d}",
                doc_id,
                text[group_start:group_end],
                group_start,
                group_end,
            ),
        )
        index += 1
        if cursor >= len(pieces):
            break

        # Step back over the tail pieces that fit in the overlap budget, so the
        # next chunk repeats them. Always advance at least one piece.
        back = cursor
        while (
            back > position + 1
            and counter.count(pieces[back - 1][0], group_end) <= overlap_tokens
        ):
            back -= 1
        position = max(back, position + 1)

    return chunks


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return sentence boundaries as character spans covering the document.

    A regex rather than a model: the corpora here are contracts, Stack Exchange
    prose and filings, and a sentence splitter that needs a model would add a
    dependency to the cheapest strategy in the comparison. Trailing whitespace
    stays attached to the sentence before it, so the spans tile the document
    with no gaps and the offsets stay exact.

    Parameters
    ----------
    text : str
        The document.

    Returns
    -------
    list of tuple of int
        Half-open ``(start, end)`` spans, blank ones dropped.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def semantic(
    doc_id: str,
    text: str,
    encode: Callable[[Sequence[str]], list[list[float]]],
    *,
    buffer_size: int = 1,
    breakpoint_percentile: float = 95.0,
    min_chars: int = 100,
    max_chars: int | None = 4000,
) -> list[Chunk]:
    """Cut where consecutive sentence embeddings stop agreeing.

    Embed every sentence with ``buffer_size`` neighbours on each side for
    context, take the cosine distance between consecutive embeddings, and cut
    where the distance exceeds the ``breakpoint_percentile`` of distances in
    that document.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    text : str
        The whole document.
    encode : callable
        Takes a sequence of strings, returns one vector per string.
    buffer_size : int, optional
        Neighbouring sentences included on each side when embedding.
    breakpoint_percentile : float, optional
        Percentile of consecutive distances above which to cut.
    min_chars : int, optional
        Minimum chunk size, so a breakpoint cannot produce a fragment.
    max_chars : int or None, optional
        Hard cap, which the published implementations also carry. Without it a
        document whose distances are all similar returns as one chunk, which
        then exceeds the encoder window and is truncated at index time, so the
        comparison would measure truncation instead of chunking.

    Returns
    -------
    list of Chunk
        Chunks in document order.
    """
    import numpy as np

    sentences = sentence_spans(text)
    if len(sentences) <= 1:
        if not text.strip():
            return []
        return [Chunk(f"{doc_id}::0000", doc_id, text, 0, len(text))]

    combined = []
    for position in range(len(sentences)):
        low = max(0, position - buffer_size)
        high = min(len(sentences), position + buffer_size + 1)
        combined.append(text[sentences[low][0] : sentences[high - 1][1]])

    vectors = np.asarray(encode(combined), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-9, None)
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)

    threshold = float(np.percentile(distances, breakpoint_percentile))
    breakpoints = {int(i) for i in np.flatnonzero(distances > threshold)}

    chunks: list[Chunk] = []
    index = 0
    group_start = sentences[0][0]
    for position, (_, sentence_end) in enumerate(sentences):
        at_end = position == len(sentences) - 1
        long_enough = sentence_end - group_start >= min_chars
        over_cap = max_chars is not None and sentence_end - group_start >= max_chars
        cut = at_end or over_cap or (position in breakpoints and long_enough)
        if not cut:
            continue
        chunks.append(
            Chunk(
                f"{doc_id}::{index:04d}",
                doc_id,
                text[group_start:sentence_end],
                group_start,
                sentence_end,
            ),
        )
        index += 1
        group_start = sentence_end
    return [c for c in chunks if c.text.strip()]


def timed(
    fn: Callable[[], list[Chunk]],
    strategy: str,
    **params: object,
) -> ChunkingResult:
    """Run a chunker and record its wall time.

    Parameters
    ----------
    fn : callable
        A no-argument call that returns chunks.
    strategy : str
        Name to record.
    **params
        Settings to record alongside the result.

    Returns
    -------
    ChunkingResult
        The chunks, the seconds, and the settings.
    """
    began = time.perf_counter()
    chunks = fn()
    return ChunkingResult(chunks, time.perf_counter() - began, strategy, params)


@dataclass
class Item:
    """One serialised document item, with its span in the document string.

    Parameters
    ----------
    kind : str
        ``"heading"``, ``"text"``, or ``"table"``.
    text : str
        The item rendered as it appears in the document string.
    start : int
        Character offset of the item in the document string.
    end : int
        Character offset one past the item's last character.
    page : int or None, optional
        Source page, when the converter recorded one.
    grid : TableGrid or None, optional
        The table's grid, for a table item.
    heading : str, optional
        The most recent section heading above this item.
    """

    kind: str
    text: str
    start: int
    end: int
    page: int | None = None
    grid: TableGrid | None = None
    heading: str = ""


def _page_of(item: Any) -> int | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    page = getattr(provenance[0], "page_no", None)
    return int(page) if page is not None else None


def _caption(table: Any) -> str:
    getter = getattr(table, "caption_text", None)
    if not callable(getter):
        return ""
    try:
        return str(getter(None) or "")
    except Exception:  # noqa: BLE001 - a caption is decoration, never fatal
        return ""


def serialise(document: Any, separator: str = "\n\n") -> tuple[str, list[Item]]:
    """Flatten a Docling document into one string plus item spans.

    Tables are rendered as pipe grids, which is the form a retriever sees today.
    A naive chunker consumes the same string, so the difference between it and
    ``table_aware_chunks`` is the table handling and nothing else.

    Parameters
    ----------
    document : Any
        A Docling ``DoclingDocument``.
    separator : str, optional
        String joining consecutive items.

    Returns
    -------
    tuple of (str, list of Item)
        The document string and the items, with offsets into that string.
    """
    from docling_core.types.doc.document import (
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    parts: list[str] = []
    items: list[Item] = []
    cursor = 0
    heading = ""

    for element, _level in document.iterate_items():
        if isinstance(element, TableItem):
            grid = TableGrid.from_table(element)
            caption = _caption(element)
            body = grid.to_markdown()
            text = "\n".join(x for x in (caption, body) if x.strip())
            kind = "table"
        elif isinstance(element, (TextItem, TitleItem, SectionHeaderItem)):
            text = (getattr(element, "text", "") or "").strip()
            grid = None
            is_heading = isinstance(element, (TitleItem, SectionHeaderItem))
            kind = "heading" if is_heading else "text"
            if kind == "heading" and text:
                heading = text
        else:
            continue
        if not text.strip():
            continue
        start = cursor
        end = start + len(text)
        parts.append(text)
        cursor = end + len(separator)
        items.append(
            Item(
                kind=kind,
                text=text,
                start=start,
                end=end,
                page=_page_of(element),
                grid=grid,
                heading=heading,
            ),
        )
    return separator.join(parts), items


def table_aware_chunks(
    doc_id: str,
    items: list[Item],
    *,
    max_chars: int = 1200,
    min_rows_per_part: int = 1,
) -> list[Chunk]:
    """Chunk a filing without ever splitting a table away from its headers.

    When a table is small it becomes one chunk with the section heading above it
    and its caption. When a table is too large for one chunk it is split by rows,
    and every part repeats the header rows, so a part holding ``4,812`` also
    holds ``December 31, 2022``.

    A chunk produced by splitting a large table is a reassembly rather than a
    contiguous slice of the document, so its ``start`` and ``end`` are the whole
    table's span. Filings carry no span-derived judgments, so nothing depends on
    those offsets; they are kept so the chunk type stays the same everywhere.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    items : list of Item
        The serialised document, from ``serialise``.
    max_chars : int, optional
        Target chunk size in characters.
    min_rows_per_part : int, optional
        Rows a split table part must hold before the budget can end it, so an
        oversized single row still produces a chunk.

    Returns
    -------
    list of Chunk
        Chunks in document order.
    """
    chunks: list[Chunk] = []
    buffer: list[Item] = []
    index = 0

    def flush() -> None:
        nonlocal index, buffer
        if not buffer:
            return
        text = "\n\n".join(i.text for i in buffer)
        chunks.append(
            Chunk(
                f"{doc_id}::{index:04d}", doc_id, text, buffer[0].start, buffer[-1].end
            ),
        )
        index += 1
        buffer = []

    for item in items:
        if item.kind == "table":
            flush()
            for part in _table_parts(
                item,
                max_chars=max_chars,
                min_rows=min_rows_per_part,
            ):
                chunks.append(
                    Chunk(f"{doc_id}::{index:04d}", doc_id, part, item.start, item.end),
                )
                index += 1
            continue

        pending = sum(len(i.text) + 2 for i in buffer)
        if pending + len(item.text) > max_chars and buffer:
            flush()
        if len(item.text) > max_chars:
            flush()
            for cut in range(0, len(item.text), max_chars):
                piece = item.text[cut : cut + max_chars]
                chunks.append(
                    Chunk(
                        f"{doc_id}::{index:04d}",
                        doc_id,
                        piece,
                        item.start + cut,
                        min(item.start + cut + max_chars, item.end),
                    ),
                )
                index += 1
            continue
        buffer.append(item)
    flush()
    return [c for c in chunks if c.text.strip()]


def _table_parts(item: Item, *, max_chars: int, min_rows: int) -> list[str]:
    """Render a table as one chunk, or as row groups that repeat the headers."""
    grid = item.grid
    if grid is None:
        raise ValueError(f"item {item.start}-{item.end} is a table with no grid")
    header_rows = set(grid.header_rows)
    body_rows = [r for r in range(grid.num_rows) if r not in header_rows]

    whole = _render(grid, body_rows, item.heading)
    if len(whole) <= max_chars or not body_rows:
        return [whole]

    parts: list[str] = []
    group: list[int] = []
    for row in body_rows:
        candidate = group + [row]
        over = len(_render(grid, candidate, item.heading)) > max_chars
        if over and len(group) >= min_rows:
            parts.append(_render(grid, group, item.heading))
            group = [row]
        else:
            group = candidate
    if group:
        parts.append(_render(grid, group, item.heading))
    return parts


def _render(grid: TableGrid, rows: list[int], heading: str) -> str:
    """Render a row group with everything it needs to be read on its own.

    That is the section heading, the table's header rows, and any section or
    continuation row above the group that carries its label.
    """
    if not rows:
        return heading
    carried = set(grid.governing_rows(rows[0]))
    body = grid.to_markdown(rows=sorted(carried | set(rows)), repeat_headers=True)
    return "\n".join(x for x in (heading, body) if x.strip())


def all_facts(items: Sequence[Item]) -> list[CellFact]:
    """Return every numeric cell fact in every table of a document.

    Each fact carries its table's span in the serialised document, so the
    attachment metric can scope a cell to the chunks that overlap its own table.

    Parameters
    ----------
    items : sequence of Item
        The serialised document.

    Returns
    -------
    list of CellFact
        Facts across every table item.
    """
    facts: list[CellFact] = []
    for item in items:
        if item.grid is not None:
            facts.extend(item.grid.facts(table_span=(item.start, item.end)))
    return facts
