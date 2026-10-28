"""The harness approach: describe pictures, summarise tables, index the result as text.

Docling parses each filing into text, tables, and pictures. Text and tables
reach the index as text already; pictures reach it as nothing, because the
text layer of a chart is at most its axis labels. Enrichment sends selected
elements to a vision model and inlines what comes back into the chunk text:

- E1 describes every picture over :data:`~.pages.DESCRIBE_AREA` of its page
  with Docling's default prompt, the generic setting a pipeline gets with
  ``do_picture_description`` and no prompt of its own.
- E2 describes the same pictures with a prompt that names what a filing's
  charts carry (axes, series, every value) and asks for chart values as a
  table, and adds a structured summary of every complex table (wide, or with
  a header spanning columns).

Selection keeps the cost proportional to the visual content. Each description
is one model call per element, paid once at indexing and again whenever the
prompt or the model changes.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, TypeVar

from aieng.advanced_rag.retrieval.chunkers import Item, table_aware_chunks
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.filings import chunk_pages
from aieng.advanced_rag.retrieval.tables import TableGrid

from .pages import DESCRIBE_AREA, Corpus, Element, crop_element, data_uri, load_document, page_id, split_page_id
from .vlm import ProxyModels, parse_json


#: Docling's default picture-description prompt (``PictureDescriptionApiOptions``).
GENERIC_PICTURE_PROMPT = "Describe this image in a few sentences."

DOMAIN_PICTURE_PROMPT = """This image is a picture cropped from a page of {document}.

First say what it is, on one line starting "Type:": a chart (bar, line, pie,
area, or histogram), a diagram (flowchart, organisation or structure chart),
a table shown as an image, a map or infographic, or a photo or logo.

Then describe it for a search index:
- the title, subtitle, and any caption or footnote in the image;
- for a chart: the chart type, the axis titles and units, every series or
  legend entry, and every data value, as a markdown table with one row per
  category or period and one column per series. Give printed values exactly.
  When a value has to be read off the axis instead, prefix it with "~";
- for a diagram: every box or node label and every connection between them;
- for a map or infographic: every label and figure it shows;
- for a photo or logo: one sentence.

Use only what the image shows. Do not add context from outside it."""

TABLE_SUMMARY_PROMPT = """The image is a table cropped from a page of {document}. Its first rows,
as a parser extracted them, are:

{rows}

Return one JSON object and nothing else, with these keys:
  "content_type": one of "financial_statement", "segment_breakdown",
    "schedule_or_maturity", "compensation_or_equity", "operating_metrics", "other";
  "summary": two to four sentences saying what the table reports, for which
    periods and units, and what its rows and columns are;
  "key_entities": three to eight short strings naming the line items, segments,
    or measures a reader would search for;
  "header_rows": how many rows at the top are column headers."""

#: Marker that introduces inlined description text in a chunk.
FIGURE_MARKER = "[Figure description]"
TABLE_MARKER = "[Table summary]"
#: Chunks from every filing share this document identifier, so retrieval runs
#: over the whole page universe instead of inside one filing.
UNSCOPED = "universe"

#: Concurrent requests per step. Each model's request budget still caps the rate,
#: so more workers only help when calls are slow.
WORKERS = 8

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(function: Callable[[T], R], items: Sequence[T], workers: int | None = None) -> list[R]:
    """Apply ``function`` to each item on a thread pool, keeping order.

    Each task runs in a copy of the caller's context, so a meter that
    :func:`aieng.advanced_rag.runs.load_or_run` bound in the calling thread
    records the calls made in the workers.
    """
    with ThreadPoolExecutor(max_workers=workers or WORKERS) as pool:
        futures = [pool.submit(contextvars.copy_context().run, function, item) for item in items]
        return [future.result() for future in futures]


def document_label(doc_id: str) -> str:
    """Return a readable name for a FinanceBench filing, such as ``AES 2022 10K``."""
    return doc_id.replace("_", " ")


def describable_pictures(found: Iterable[Element], pages: Iterable[str]) -> list[Element]:
    """Return the pictures on the given pages that pass the area filter."""
    wanted = set(pages)
    return [e for e in found if e.kind == "picture" and e.page_id in wanted and e.area >= DESCRIBE_AREA]


def summarisable_tables(found: Iterable[Element], pages: Iterable[str]) -> list[Element]:
    """Return the complex tables on the given pages."""
    wanted = set(pages)
    return [e for e in found if e.kind == "table" and e.page_id in wanted and e.complex_table]


def describe_pictures(
    llm: ProxyModels,
    model: str,
    corpus: Corpus,
    pictures: Sequence[Element],
    *,
    prompt: str,
    max_tokens: int = 900,
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Describe each picture crop with one call.

    Parameters
    ----------
    llm : ProxyModels
        The proxy client.
    model : str
        A model that accepts images.
    corpus : Corpus
        Where the PDFs are.
    pictures : sequence of Element
        The pictures to describe.
    prompt : str
        :data:`GENERIC_PICTURE_PROMPT`, or :data:`DOMAIN_PICTURE_PROMPT`, which
        takes a ``{document}`` field.
    max_tokens : int, optional
        Completion cap per description.
    workers : int or None, optional
        Concurrent requests, :data:`WORKERS` by default; the model's rate limit
        still applies.

    Returns
    -------
    dict
        Element identifier to ``{"text", "prompt_tokens", "completion_tokens",
        "seconds"}``.
    """

    def one(element: Element) -> tuple[str, dict[str, Any]]:
        image = data_uri(crop_element(corpus, element))
        reply = llm.complete(
            model,
            prompt.format(document=document_label(element.doc_id)),
            [image],
            max_tokens=max_tokens,
        )
        return element.element_id, {
            "text": reply.text.strip(),
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "seconds": round(reply.seconds, 2),
        }

    return dict(parallel_map(one, list(pictures), workers))


def _first_rows(corpus: Corpus, element: Element, rows: int = 8) -> str:
    document = load_document(corpus, element.doc_id)
    for table in document.tables:
        if table.self_ref == element.self_ref:
            grid = TableGrid.from_table(table)
            lines = grid.to_markdown().splitlines()
            return "\n".join(lines[: rows + 1])
    return ""


def summarise_tables(
    llm: ProxyModels,
    model: str,
    corpus: Corpus,
    tables: Sequence[Element],
    *,
    max_tokens: int = 500,
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarise each complex table from its crop and its first parsed rows.

    Returns
    -------
    dict
        Element identifier to ``{"text", "fields", ...}``, where ``text`` is the
        summary as it is inlined and ``fields`` the parsed JSON, or ``None``
        when the reply held none.
    """

    def one(element: Element) -> tuple[str, dict[str, Any]]:
        image = data_uri(crop_element(corpus, element, scale=2.0))
        prompt = TABLE_SUMMARY_PROMPT.format(
            document=document_label(element.doc_id),
            rows=_first_rows(corpus, element),
        )
        reply = llm.complete(model, prompt, [image], max_tokens=max_tokens)
        try:
            fields = parse_json(reply.text)
        except ValueError:
            fields = None
        return element.element_id, {
            "text": render_table_summary(fields) if isinstance(fields, dict) else "",
            "fields": fields if isinstance(fields, dict) else None,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "seconds": round(reply.seconds, 2),
        }

    return dict(parallel_map(one, list(tables), workers))


def render_table_summary(fields: Mapping[str, Any]) -> str:
    """Return a table summary as the sentence block inlined before the table."""
    summary = str(fields.get("summary", "")).strip()
    entities = [str(e).strip() for e in fields.get("key_entities", []) or [] if str(e).strip()]
    kind = str(fields.get("content_type", "")).strip()
    parts = [summary]
    if entities:
        parts.append("Key entities: " + "; ".join(entities) + ".")
    if kind:
        parts.append(f"Content type: {kind}.")
    return " ".join(p for p in parts if p)


def page_items(
    corpus: Corpus,
    doc_id: str,
    pages: Iterable[int],
    *,
    pictures: Mapping[str, str] | None = None,
    tables: Mapping[str, str] | None = None,
) -> list[Item]:
    """Serialise one filing's chosen pages, with descriptions inlined where they exist.

    The serialisation matches :func:`aieng.advanced_rag.retrieval.chunkers.serialise`:
    headings, text, and tables as pipe grids, in reading order. Two additions
    come from enrichment. A picture with a description becomes a text item
    holding it, at the picture's place. A table with a summary is preceded by a
    text item holding the summary.

    Parameters
    ----------
    corpus : Corpus
        Where the Docling folders are.
    doc_id : str
        The filing.
    pages : iterable of int
        Zero-based pages to keep.
    pictures : mapping or None, optional
        Element identifier to description text.
    tables : mapping or None, optional
        Element identifier to summary text.

    Returns
    -------
    list of Item
        Items with offsets into the joined text, as ``serialise`` returns them.
    """
    from docling_core.types.doc.document import (  # noqa: PLC0415
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    keep = {page + 1 for page in pages}
    described = pictures or {}
    summaries = tables or {}
    document = load_document(corpus, doc_id)
    items: list[Item] = []
    cursor = 0
    heading = ""

    def add(kind: str, text: str, page: int | None, grid: TableGrid | None = None) -> None:
        nonlocal cursor
        items.append(
            Item(kind=kind, text=text, start=cursor, end=cursor + len(text), page=page, grid=grid, heading=heading)
        )
        cursor += len(text) + 2

    for element, _level in document.iterate_items():
        prov = getattr(element, "prov", None)
        page = int(prov[0].page_no) if prov else None
        if page is not None and page not in keep:
            continue
        reference = f"{doc_id}|{element.self_ref}"
        if isinstance(element, PictureItem):
            text = described.get(reference, "").strip()
            if text and page is not None:
                add("text", f"{FIGURE_MARKER} {text}", page)
            continue
        if isinstance(element, TableItem):
            summary = summaries.get(reference, "").strip()
            if summary and page is not None:
                add("text", f"{TABLE_MARKER} {summary}", page)
            grid = TableGrid.from_table(element)
            caption = element.caption_text(document) if hasattr(element, "caption_text") else ""
            body = "\n".join(x for x in (caption or "", grid.to_markdown()) if x.strip())
            if body.strip():
                add("table", body, page, grid)
            continue
        if isinstance(element, (TextItem, TitleItem, SectionHeaderItem)):
            text = (getattr(element, "text", "") or "").strip()
            if not text or page is None:
                continue
            if isinstance(element, (TitleItem, SectionHeaderItem)):
                heading = text
                add("heading", text, page)
            else:
                add("text", text, page)
    return items


def build_chunks(
    corpus: Corpus,
    page_ids: Iterable[str],
    *,
    pictures: Mapping[str, str] | None = None,
    tables: Mapping[str, str] | None = None,
    max_chars: int = 1200,
    per_page: bool = False,
) -> tuple[list[Chunk], dict[str, str]]:
    """Chunk the page universe with RI #3's table-aware chunker.

    Parameters
    ----------
    corpus : Corpus
        Where the Docling folders are.
    page_ids : iterable of str
        The universe.
    pictures, tables : mapping or None, optional
        Enrichment text by element identifier. Both empty gives arm T.
    max_chars : int, optional
        Chunk size.
    per_page : bool, optional
        Never let a chunk cross a page boundary. Off, consecutive short pages
        share a chunk and the chunk counts for its first page, which is how
        the reference rankings on filings were built. On suits short pages,
        such as slides or the demonstration document.

    Returns
    -------
    tuple
        The chunks, all under the one unscoped document identifier, and each
        chunk's first page, which is the page a chunk ranking collapses to.
    """
    by_doc: dict[str, set[int]] = {}
    for identifier in page_ids:
        doc_id, page = split_page_id(identifier)
        by_doc.setdefault(doc_id, set()).add(page)
    chunks: list[Chunk] = []
    first_page: dict[str, str] = {}
    for doc_id in sorted(by_doc):
        items = page_items(corpus, doc_id, by_doc[doc_id], pictures=pictures, tables=tables)
        if not items:
            continue
        if per_page:
            found = []
            for page in sorted(by_doc[doc_id]):
                on_page = [i for i in items if i.page == page + 1]
                found.extend(table_aware_chunks(page_id(doc_id, page), on_page, max_chars=max_chars))
            found = [replace(c, doc_id=doc_id) for c in found]
        else:
            found = table_aware_chunks(doc_id, items, max_chars=max_chars)
        covered = chunk_pages(found, items)
        for chunk in found:
            pages = sorted(covered[chunk.chunk_id])
            chunk_page = page_id(doc_id, pages[0])
            if pages[0] not in by_doc[doc_id]:
                chunk_page = page_id(doc_id, min(by_doc[doc_id]))
            first_page[chunk.chunk_id] = chunk_page
            chunks.append(replace(chunk, doc_id=UNSCOPED))
    return chunks, first_page


def page_text_with_enrichment(
    corpus: Corpus,
    identifier: str,
    *,
    pictures: Mapping[str, str] | None = None,
    tables: Mapping[str, str] | None = None,
) -> str:
    """Return one page's text as the enriched arms index it, for a text reader."""
    doc_id, page = split_page_id(identifier)
    items = page_items(corpus, doc_id, [page], pictures=pictures, tables=tables)
    return "\n\n".join(item.text for item in items)


__all__ = [
    "DOMAIN_PICTURE_PROMPT",
    "FIGURE_MARKER",
    "GENERIC_PICTURE_PROMPT",
    "TABLE_MARKER",
    "TABLE_SUMMARY_PROMPT",
    "UNSCOPED",
    "build_chunks",
    "describable_pictures",
    "describe_pictures",
    "document_label",
    "page_items",
    "page_text_with_enrichment",
    "parallel_map",
    "render_table_summary",
    "summarisable_tables",
    "summarise_tables",
]
