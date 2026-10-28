"""Choose the page universe, render pages and picture crops, and read page text.

Every arm in this topic indexes the same pages from the same pinned PDFs, so
page-level retrieval compares one to one. A page is named ``<doc_id>#p<NNN>``
with a zero-based page number, the base FinanceBench's evidence uses and the
base of the ``pages/pNNN.md`` files RI #3's ingestion writes. Docling numbers
pages from 1, and the conversion happens in :func:`docling_page`.

The universe holds four groups of pages from the filings that carry a large
picture: every figure page, a sample of pages with a complex table, the gold
pages of the FinanceBench questions on those filings, and distractor pages
from the same filings. Distractors come from the same filings so that a
retriever has to rank against real, similar non-gold pages.
"""

from __future__ import annotations

import base64
import io
import random
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache.paths import root as cache_root


#: A picture larger than this share of its page makes the page a figure page.
FIGURE_PAGE_AREA = 0.10
#: Pictures at least this share of the page are sent for description. Docling's
#: own ``picture_area_threshold`` default is the same 0.05.
DESCRIBE_AREA = 0.05
#: A table with this many columns or more, or with a cell spanning columns,
#: counts as complex. Complex tables get a structured summary in arm E2.
COMPLEX_TABLE_COLUMNS = 6
#: Render scale for every page image a model or an embedder sees. 1.5 renders a
#: US letter page at 918 by 1188 pixels.
PAGE_SCALE = 1.5
#: Render scale for picture crops, higher so small charts stay legible.
CROP_SCALE = 2.5

# PDFium is not thread-safe, and the model steps render pages from worker
# threads, so every call into it holds this lock.
_PDFIUM = threading.Lock()


def page_id(doc_id: str, page: int) -> str:
    """Return the identifier of one zero-based page of one document."""
    return f"{doc_id}#p{page:03d}"


def split_page_id(identifier: str) -> tuple[str, int]:
    """Return the document and zero-based page a page identifier names.

    Raises
    ------
    ValueError
        If the identifier does not have the ``<doc_id>#pNNN`` shape.
    """
    doc_id, sep, page = identifier.rpartition("#p")
    if not sep or not page.isdigit():
        raise ValueError(f"not a page identifier: {identifier!r}")
    return doc_id, int(page)


def docling_page(page: int) -> int:
    """Return Docling's one-based page number for a zero-based page."""
    return page + 1


@dataclass(frozen=True)
class Corpus:
    """Where a set of documents' PDFs and Docling folders live.

    Parameters
    ----------
    pdf_dir : Path
        Holds ``<doc_id>.pdf``.
    ingestion_dir : Path
        Holds ``<doc_id>/document.json`` and ``<doc_id>/pages/pNNN.md``, as
        :func:`aieng.advanced_rag.ingestion.docling.ingest_document` writes them.
    """

    pdf_dir: Path
    ingestion_dir: Path

    def pdf(self, doc_id: str) -> Path:
        """Return one document's PDF."""
        return self.pdf_dir / f"{doc_id}.pdf"

    def folder(self, doc_id: str) -> Path:
        """Return one document's Docling folder."""
        return self.ingestion_dir / doc_id

    def page_text(self, identifier: str) -> str:
        """Return Docling's text for one page, empty for a page with none."""
        doc_id, page = split_page_id(identifier)
        path = self.folder(doc_id) / "pages" / f"p{page:03d}.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""


def financebench_corpus() -> Corpus:
    """Return the FinanceBench PDFs and the section folders RI #3's ingestion wrote."""
    base = Path(cache_root())
    return Corpus(base / "corpora" / "financebench-pdfs", base / "ingestion" / "financebench")


@lru_cache(maxsize=8)
def _load_document(folder: str) -> Any:
    from aieng.advanced_rag.retrieval.filings import load_document  # noqa: PLC0415

    return load_document(Path(folder))


def load_document(corpus: Corpus, doc_id: str) -> Any:
    """Return one document's ``DoclingDocument``, keeping the last few in memory."""
    return _load_document(str(corpus.folder(doc_id)))


@dataclass(frozen=True)
class Element:
    """One picture or table on one page, with its place and size.

    Parameters
    ----------
    element_id : str
        ``<doc_id>|<self_ref>``, for example ``AES_2022_10K|#/pictures/12``.
    kind : str
        ``picture`` or ``table``.
    page_id : str
        The page it sits on.
    bbox : tuple of float
        Left, top, right, bottom in PDF points from the page's top-left corner.
    area : float
        Its share of the page area.
    columns : int
        Table columns, zero for a picture.
    spanning : bool
        Whether a table has a cell spanning columns.
    """

    element_id: str
    kind: str
    page_id: str
    bbox: tuple[float, float, float, float]
    area: float
    columns: int = 0
    spanning: bool = False

    @property
    def doc_id(self) -> str:
        """Return the element's document."""
        return self.element_id.split("|", 1)[0]

    @property
    def self_ref(self) -> str:
        """Return Docling's reference to the element inside its document."""
        return self.element_id.split("|", 1)[1]

    @property
    def complex_table(self) -> bool:
        """Return whether a table is wide or has a header spanning columns."""
        return self.kind == "table" and (self.columns >= COMPLEX_TABLE_COLUMNS or self.spanning)


def _top_left(bbox: Any, page_height: float) -> tuple[float, float, float, float]:
    left, right = float(bbox.l), float(bbox.r)
    if str(getattr(bbox.coord_origin, "value", bbox.coord_origin)).upper().startswith("BOTTOM"):
        top, bottom = page_height - float(bbox.t), page_height - float(bbox.b)
    else:
        top, bottom = float(bbox.t), float(bbox.b)
    return left, min(top, bottom), right, max(top, bottom)


def elements(corpus: Corpus, doc_id: str) -> list[Element]:
    """Return every picture and table Docling found in one document, with its page.

    Parameters
    ----------
    corpus : Corpus
        Where the document's Docling folder lives.
    doc_id : str
        The document.

    Returns
    -------
    list of Element
        Pictures first, then tables, each in Docling's order.
    """
    document = load_document(corpus, doc_id)
    found: list[Element] = []
    for kind, items in (("picture", document.pictures), ("table", document.tables)):
        for item in items:
            if not item.prov:
                continue
            prov = item.prov[0]
            size = document.pages[prov.page_no].size
            box = _top_left(prov.bbox, float(size.height))
            area = abs((box[2] - box[0]) * (box[3] - box[1])) / float(size.width * size.height)
            columns = int(item.data.num_cols) if kind == "table" else 0
            spanning = kind == "table" and any((cell.col_span or 1) > 1 for cell in item.data.table_cells)
            found.append(
                Element(
                    element_id=f"{doc_id}|{item.self_ref}",
                    kind=kind,
                    page_id=page_id(doc_id, int(prov.page_no) - 1),
                    bbox=box,
                    area=round(area, 4),
                    columns=columns,
                    spanning=spanning,
                ),
            )
    return found


@dataclass
class Universe:
    """The pages every arm indexes, by group.

    Parameters
    ----------
    figure : list of str
        Pages holding a picture over :data:`FIGURE_PAGE_AREA` of the page.
    table : list of str
        Sampled pages holding a complex table and no large picture.
    control : list of str
        Gold pages of the FinanceBench questions on these filings.
    distractor : list of str
        Other pages with text, from the same filings.
    """

    figure: list[str] = field(default_factory=list)
    table: list[str] = field(default_factory=list)
    control: list[str] = field(default_factory=list)
    distractor: list[str] = field(default_factory=list)

    @property
    def pages(self) -> list[str]:
        """Return every page once, sorted."""
        return sorted({*self.figure, *self.table, *self.control, *self.distractor})

    @property
    def doc_ids(self) -> list[str]:
        """Return the documents the universe draws from."""
        return sorted({split_page_id(p)[0] for p in self.pages})

    def group_of(self, identifier: str) -> str:
        """Return the first of figure, table, control, distractor holding a page."""
        for name in ("figure", "table", "control", "distractor"):
            if identifier in getattr(self, name):
                return name
        raise KeyError(identifier)

    def to_dict(self) -> dict[str, list[str]]:
        """Return the groups as JSON-ready lists."""
        return {
            "figure": list(self.figure),
            "table": list(self.table),
            "control": list(self.control),
            "distractor": list(self.distractor),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[str]]) -> Universe:
        """Read the groups :meth:`to_dict` wrote."""
        return cls(
            figure=list(payload["figure"]),
            table=list(payload["table"]),
            control=list(payload["control"]),
            distractor=list(payload["distractor"]),
        )


def page_count(corpus: Corpus, doc_id: str) -> int:
    """Return how many pages Docling converted in one document."""
    return len(load_document(corpus, doc_id).pages)


def select_universe(
    corpus: Corpus,
    doc_ids: Iterable[str],
    control_pages: Iterable[str],
    *,
    size: int = 1000,
    table_pages: int = 100,
    per_document_tables: int = 4,
    min_chars: int = 200,
    seed: int = 0,
) -> Universe:
    """Choose the page universe from a set of filings.

    Parameters
    ----------
    corpus : Corpus
        The filings' PDFs and Docling folders.
    doc_ids : iterable of str
        Candidate filings. Only those with at least one figure page are used.
    control_pages : iterable of str
        Gold pages of real questions on these filings, kept whatever they hold.
    size : int, optional
        Pages in the universe once distractors are added.
    table_pages : int, optional
        Complex-table pages to sample.
    per_document_tables : int, optional
        Most complex-table pages from one filing, so the sample spreads.
    min_chars : int, optional
        A distractor needs at least this much Docling page text.
    seed : int, optional
        Seed for the table and distractor samples.

    Returns
    -------
    Universe
        The four groups. A page belongs to the first of figure, table, control,
        and distractor that claims it.
    """
    rng = random.Random(seed)
    by_doc = {doc_id: elements(corpus, doc_id) for doc_id in sorted(set(doc_ids))}
    figure = sorted(
        {e.page_id for found in by_doc.values() for e in found if e.kind == "picture" and e.area >= FIGURE_PAGE_AREA},
    )
    docs = sorted({split_page_id(p)[0] for p in figure})
    taken = set(figure)
    control = sorted(p for p in set(control_pages) if split_page_id(p)[0] in docs and p not in taken)
    taken.update(control)

    candidates: list[str] = []
    for doc_id in docs:
        pages = sorted({e.page_id for e in by_doc[doc_id] if e.complex_table and e.page_id not in taken})
        rng.shuffle(pages)
        candidates.extend(pages[:per_document_tables])
    rng.shuffle(candidates)
    table = sorted(candidates[:table_pages])
    taken.update(table)

    pool = [
        page_id(doc_id, page)
        for doc_id in docs
        for page in range(page_count(corpus, doc_id))
        if page_id(doc_id, page) not in taken and len(corpus.page_text(page_id(doc_id, page))) >= min_chars
    ]
    rng.shuffle(pool)
    distractor = sorted(pool[: max(0, size - len(taken))])
    return Universe(figure=figure, table=table, control=control, distractor=distractor)


def render_page(corpus: Corpus, identifier: str, scale: float = PAGE_SCALE) -> Any:
    """Render one page of its PDF as an RGB ``PIL.Image``."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    doc_id, page = split_page_id(identifier)
    with _PDFIUM:
        document = pdfium.PdfDocument(str(corpus.pdf(doc_id)))
        try:
            handle = document[page]
            bitmap = handle.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            # Close children before the document, inside the lock, so no
            # finaliser touches PDFium from another thread later.
            bitmap.close()
            handle.close()
            return image
        finally:
            document.close()


def crop_element(corpus: Corpus, element: Element, scale: float = CROP_SCALE, margin: float = 6.0) -> Any:
    """Render one picture or table, with a small margin, as an RGB image."""
    image = render_page(corpus, element.page_id, scale)
    left, top, right, bottom = element.bbox
    box = (
        max(0, int((left - margin) * scale)),
        max(0, int((top - margin) * scale)),
        min(image.width, int((right + margin) * scale)),
        min(image.height, int((bottom + margin) * scale)),
    )
    return image.crop(box)


def data_uri(image: Any, quality: int = 85) -> str:
    """Return an image as a JPEG ``data:`` URI, the form the proxy accepts."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = [
    "COMPLEX_TABLE_COLUMNS",
    "CROP_SCALE",
    "DESCRIBE_AREA",
    "FIGURE_PAGE_AREA",
    "PAGE_SCALE",
    "Corpus",
    "Element",
    "Universe",
    "crop_element",
    "data_uri",
    "docling_page",
    "elements",
    "financebench_corpus",
    "load_document",
    "page_count",
    "page_id",
    "render_page",
    "select_universe",
    "split_page_id",
]
