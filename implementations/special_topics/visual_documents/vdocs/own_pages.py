"""Run the figure-only test and three arms on a team's own pages.

A team brings up to :data:`MAX_PAGES` pages of PDF and up to
:data:`MAX_QUESTIONS` questions with reference answers, and gets three
answers to the question the comparison notebook asks on filings: how many of
these questions need the picture, and which of text, enriched text, and the
page image answers them.

The notebook ships with a demonstration document that this module draws from
scratch (:func:`write_demo_pdf`): a six-page annual review of a fictional
freight railway, with charts whose values are printed only on the chart, a
chart whose headline value is repeated in the text, a table, and a text page.
Its questions are :data:`DEMO_QUESTIONS`. Because the content is made here, the
committed notebook can show it in full.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import CacheKey, JsonValue, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.models import ModelChoice, ModelSet
from aieng.advanced_rag.runs import Step, StepResult, load_or_run

from . import arms, enrich, visual
from .pages import Corpus, data_uri, elements, page_count, page_id, render_page
from .pipeline import REFERENCE_JUDGE, REFERENCE_VLM, digest_of
from .questions import ANSWER_PROMPT, answer_from
from .scoring import numbers_in, score_answer
from .vlm import ProxyModels


RI_ID = "visual-documents-own"
MAX_PAGES = 20
MAX_QUESTIONS = 10
ROWS: tuple[str, ...] = ("T", "E2", "VV")

#: Reader input tokens per page measured on the reference reader, used to price a
#: run before it starts: text per chunk and per page image.
TOKENS_PER_CHUNK = 330
TOKENS_PER_PAGE_IMAGE = 1100
TOKENS_PER_ANSWER = 40

DEMO_NAME = "larkspur_freight_review"

DEMO_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "How many million tonnes of freight did Larkspur Freight Rail carry in 2022?",
        "answer": "21.2 million tonnes",
        "page": 1,
        "kind": "figure",
    },
    {
        "question": "What share of Larkspur Freight Rail's 2024 revenue came from the Automotive segment?",
        "answer": "15%",
        "page": 2,
        "kind": "figure",
    },
    {
        "question": "What was Larkspur Freight Rail's on-time delivery rate in the third quarter of 2024?",
        "answer": "94.2%",
        "page": 4,
        "kind": "figure",
    },
    {
        "question": "What share of Larkspur Freight Rail's 2024 revenue came from the Intermodal segment?",
        "answer": "46%",
        "page": 2,
        "kind": "figure",
    },
    {
        "question": "What was the operating income of Larkspur Freight Rail's Prairie corridor in 2024?",
        "answer": "$29 million",
        "page": 3,
        "kind": "table",
    },
    {
        "question": "What revenue did Larkspur Freight Rail's Coastal corridor earn in 2023?",
        "answer": "$131 million",
        "page": 3,
        "kind": "table",
    },
    {
        "question": "How many reportable safety incidents did Larkspur Freight Rail record in 2024?",
        "answer": "12",
        "page": 5,
        "kind": "text",
    },
    {
        "question": "What was Larkspur Freight Rail's on-time delivery rate in the first quarter of 2023?",
        "answer": "NOT STATED",
        "page": None,
        "kind": "null",
    },
]


# --------------------------------------------------------------------- demo document


def _chart_font(size: int) -> Any:
    from PIL import ImageFont  # noqa: PLC0415

    return ImageFont.load_default(size=size)


def _bar_chart(title: str, labels: Sequence[str], values: Sequence[float], unit: str) -> Any:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), title, fill="black", font=_chart_font(34))
    base, top, left = 540, 120, 90
    draw.line((left, base, 960, base), fill="#444444", width=3)
    peak = max(values) * 1.15
    width = (960 - left) / len(values)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x0 = left + index * width + width * 0.2
        x1 = x0 + width * 0.6
        y = base - (base - top) * value / peak
        draw.rectangle((x0, y, x1, base), fill="#2f6f9f")
        draw.text((x0, y - 40), f"{value:g}{unit}", fill="black", font=_chart_font(28))
        draw.text((x0 + 10, base + 14), label, fill="black", font=_chart_font(28))
    return image


def _pie_chart(title: str, slices: Sequence[tuple[str, float]]) -> Any:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), title, fill="black", font=_chart_font(34))
    colours = ("#2f6f9f", "#e08a2e", "#5a9e4b", "#9a9a9a")
    start = -90.0
    box = ((80.0, 110.0), (520.0, 550.0))
    for (_label, share), colour in zip(slices, colours, strict=False):
        sweep = 360 * share / 100
        draw.pieslice(box, start, start + sweep, fill=colour, outline="white", width=3)
        start += sweep
    for index, ((label, share), colour) in enumerate(zip(slices, colours, strict=False)):
        y = 180 + index * 80
        draw.rectangle((580, y, 620, y + 40), fill=colour)
        draw.text((640, y + 2), f"{label} {share:g}%", fill="black", font=_chart_font(30))
    return image


def _line_chart(title: str, labels: Sequence[str], values: Sequence[float]) -> Any:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), title, fill="black", font=_chart_font(34))
    base, top, left, right = 540, 140, 110, 940
    low, high = 88.0, 96.0
    draw.line((left, base, right, base), fill="#444444", width=3)
    points = []
    for index, value in enumerate(values):
        x = left + 60 + index * (right - left - 120) / (len(values) - 1)
        y = base - (base - top) * (value - low) / (high - low)
        points.append((x, y))
        draw.text((x - 30, base + 14), labels[index], fill="black", font=_chart_font(28))
        draw.text((x - 34, y - 46), f"{value:g}%", fill="black", font=_chart_font(28))
    draw.line(points, fill="#2f6f9f", width=6)
    for x, y in points:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#2f6f9f")
    return image


_DEMO_TABLE = [
    ("Corridor", "2023 revenue", "2024 revenue", "2024 operating income"),
    ("Northern", "142", "158", "41"),
    ("Coastal", "131", "139", "33"),
    ("Prairie", "103", "115", "29"),
    ("Total", "376", "412", "103"),
]

#: The demonstration review, one entry per page: heading, prose, and what follows it.
_DEMO_PAGES: list[dict[str, Any]] = [
    {
        "title": "Larkspur Freight Rail",
        "subtitle": "Annual review 2024",
        "lines": [
            "Larkspur Freight Rail moves bulk, intermodal, and automotive freight on three",
            "corridors: Northern, Coastal, and Prairie. Revenue rose to $412 million in 2024",
            "from $376 million in 2023, and operating income reached $103 million.",
            "",
            "This review covers freight volume, revenue by segment, results by corridor,",
            "service reliability, and safety. Charts show values for each period.",
        ],
    },
    {
        "title": "Freight volume",
        "lines": [
            "Volume grew in each of the last five years as new intermodal terminals opened",
            "on the Northern corridor. The chart shows tonnes carried each year.",
        ],
        "chart": lambda: _bar_chart(
            "Freight carried (million tonnes)",
            ["2020", "2021", "2022", "2023", "2024"],
            [18.4, 19.9, 21.2, 22.7, 24.1],
            "",
        ),
    },
    {
        "title": "Revenue by segment",
        "lines": [
            "Intermodal remained the largest segment at 46% of revenue. The chart shows",
            "each segment's share of 2024 revenue.",
        ],
        "chart": lambda: _pie_chart(
            "2024 revenue by segment", [("Intermodal", 46), ("Bulk", 31), ("Automotive", 15), ("Other", 8)]
        ),
    },
    {
        "title": "Operating results by corridor",
        "lines": ["In $ millions, for the years ended December 31."],
        "table": _DEMO_TABLE,
        "after": [
            "The Northern corridor led revenue growth, helped by two new intermodal",
            "terminals. Coastal volumes were steady.",
        ],
    },
    {
        "title": "Service reliability",
        "lines": [
            "On-time delivery measures the share of trains arriving within 30 minutes of",
            "schedule. The chart shows the rate for each quarter of 2024.",
        ],
        "chart": lambda: _line_chart("On-time delivery rate, 2024", ["Q1", "Q2", "Q3", "Q4"], [91.5, 93.0, 94.2, 92.8]),
    },
    {
        "title": "Safety and risks",
        "lines": [
            "Reportable safety incidents fell to 12 in 2024 from 17 in 2023, after crossing",
            "upgrades on the Prairie corridor.",
            "",
            "The main risks to the business are fuel prices, weather on the Coastal",
            "corridor, and the renewal of two long-term intermodal contracts in 2026.",
        ],
    },
]


class _PdfWriter:
    """Text and images on PDFium pages, in standard fonts so the text layer is real."""

    def __init__(self) -> None:
        import pypdfium2 as pdfium  # noqa: PLC0415
        from pypdfium2 import raw  # noqa: PLC0415

        self.pdfium = pdfium
        self.raw = raw
        self.pdf = pdfium.PdfDocument.new()
        self.font = raw.FPDFText_LoadStandardFont(self.pdf, b"Helvetica")
        self.bold = raw.FPDFText_LoadStandardFont(self.pdf, b"Helvetica-Bold")
        # PDFium reads the text buffers when the content is generated, so they
        # have to outlive the calls that set them.
        self._buffers: list[Any] = []

    def text(self, page: Any, value: str, x: float, y: float, *, size: float = 11, heavy: bool = False) -> None:
        raw = self.raw
        obj = raw.FPDFPageObj_CreateTextObj(self.pdf, self.bold if heavy else self.font, size)
        buffer = ctypes.create_string_buffer((value + "\x00").encode("utf-16-le"))
        self._buffers.append(buffer)
        raw.FPDFText_SetText(obj, ctypes.cast(buffer, ctypes.POINTER(raw.FPDF_WCHAR)))
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
        raw.FPDFPage_InsertObject(page.raw, obj)

    def paragraph(self, page: Any, lines: Sequence[str], y: float) -> float:
        for line in lines:
            self.text(page, line, 72, y)
            y -= 11 * 1.6
        return y - 8

    def picture(self, page: Any, image: Any, y: float) -> None:
        obj = self.pdfium.PdfImage.new(self.pdf)
        obj.set_bitmap(self.pdfium.PdfBitmap.from_pil(image))
        width = 468.0
        height = width * image.height / image.width
        obj.set_matrix(self.pdfium.PdfMatrix().scale(width, height).translate(72, y - height))
        page.insert_obj(obj)

    def table(self, page: Any, rows: Sequence[Sequence[str]], y: float) -> None:
        columns = (72, 200, 310, 420)
        for index, row in enumerate(rows):
            for x, cell in zip(columns, row, strict=True):
                self.text(page, cell, x, y - index * 24, heavy=index == 0)


def write_demo_pdf(path: Path) -> Path:
    """Write the six-page demonstration review to ``path`` and return it.

    The text sits in the PDF's text layer and the charts are images, as in a
    born-digital report, so a text parser reads the prose and the table and
    sees each chart as a picture.
    """
    writer = _PdfWriter()
    pages = []
    for spec in _DEMO_PAGES:
        page = writer.pdf.new_page(612, 792)
        pages.append(page)
        first = "subtitle" in spec
        writer.text(page, spec["title"], 72, 720, size=20 if first else 16, heavy=True)
        if first:
            writer.text(page, spec["subtitle"], 72, 696, size=14)
        writer.paragraph(page, spec["lines"], 660 if first else (694 if "table" in spec else 690))
        if "chart" in spec:
            writer.picture(page, spec["chart"](), 630)
        if "table" in spec:
            writer.table(page, spec["table"], 650)
            writer.paragraph(page, spec["after"], 500)
    for page in pages:
        page.gen_content()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer.pdf.save(str(path))
    return path


# --------------------------------------------------------- the small comparison


def file_digest(paths: Sequence[Path]) -> str:
    """Return one digest over the PDFs' bytes, which names their cached results."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _value_of(answer: str) -> float | None:
    found = numbers_in(answer)
    return found[0] if len(found) == 1 else None


@dataclass
class OwnPages:
    """The figure-only test and arms T, E2, and VV on a few pages of your own.

    Parameters
    ----------
    pdfs : sequence of Path
        Your PDFs. Every page of each is indexed, :data:`MAX_PAGES` in all.
    questions : sequence of dict
        Each with ``question``, ``answer`` (or ``"NOT STATED"``), and ``page``:
        the zero-based page in the first PDF that holds the answer, or
        ``(pdf_stem, page)``, or None for a question the pages do not answer.
    models : ModelSet
        Whose models run the live steps.
    live : bool, optional
        Run model steps and embeddings that are not cached.
    cache : TieredResultCache or None, optional
        The artifact cache.
    """

    pdfs: Sequence[Path]
    questions: Sequence[Mapping[str, Any]]
    models: ModelSet
    live: bool = False
    cache: TieredResultCache = field(default_factory=TieredResultCache)
    show: bool = True
    env_path: Path | None = None
    _llm: ProxyModels | None = None
    _memo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the question count and name the cache folders after the PDFs."""
        if len(self.questions) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions, got {len(self.questions)}")
        self.digest = file_digest(list(self.pdfs))
        root = Path(cache_root()) / "ingestion" / "visual-documents-own" / self.digest
        self.corpus = Corpus(pdf_dir=Path(self.pdfs[0]).parent, ingestion_dir=root)
        self.workdir = Path(tempfile.gettempdir()) / "aieng-visual-documents-own" / self.digest

    @property
    def llm(self) -> ProxyModels:
        """Return the proxy client, built on first use."""
        if self._llm is None:
            self._llm = ProxyModels.from_environment(env_path=self.env_path)
        return self._llm

    # ----------------------------------------------------------------- parse

    def parse(self) -> list[str]:
        """Convert each PDF with Docling and return the page identifiers."""
        if "pages" not in self._memo:
            from aieng.advanced_rag.ingestion.docling import ingest_document  # noqa: PLC0415

            converter: Any = None

            def convert(path: Path) -> Any:
                nonlocal converter
                if converter is None:
                    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
                    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
                    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

                    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
                    converter = DocumentConverter(
                        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
                    )
                return converter.convert(path).document

            started = time.perf_counter()
            for pdf in self.pdfs:
                ingest_document(
                    Path(pdf), root=self.corpus.ingestion_dir, convert=convert, cache=self.cache, ri_id=RI_ID
                )
            pages = [
                page_id(Path(pdf).stem, n) for pdf in self.pdfs for n in range(page_count(self.corpus, Path(pdf).stem))
            ]
            if len(pages) > MAX_PAGES:
                raise ValueError(f"{len(pages)} pages; this notebook takes at most {MAX_PAGES}")
            self._memo["pages"] = pages
            self._memo["parse_seconds"] = time.perf_counter() - started
        return cast(list[str], self._memo["pages"])

    def question_records(self) -> list[dict[str, Any]]:
        """Return the questions in the comparison's record shape."""
        first = Path(self.pdfs[0]).stem
        records = []
        for index, q in enumerate(self.questions):
            where = q.get("page")
            if isinstance(where, (tuple, list)):
                gold = [page_id(str(where[0]), int(where[1]))]
            elif where is None:
                gold = []
            else:
                gold = [page_id(first, int(where))]
            null = not gold
            records.append(
                {
                    "query_id": f"own-{index:02d}",
                    "group": "null" if null else str(q.get("kind", "answerable")),
                    "question": str(q["question"]),
                    "answer": str(q["answer"]),
                    "value": None if null else _value_of(str(q["answer"])),
                    "gold_pages": gold,
                    "rewritten_from": None,
                },
            )
        return records

    def elements(self) -> list[Any]:
        """Return the pictures and tables Docling found."""
        self.parse()
        return [e for pdf in self.pdfs for e in elements(self.corpus, Path(pdf).stem)]

    # -------------------------------------------------------- costs before running

    def estimate(self, what: str, model: str) -> str:
        """Return a one-line price for a step before it runs."""
        records = self.question_records()
        n = len(records)
        if what == "figure-only test":
            prompt = n * (TOKENS_PER_CHUNK * 3 + TOKENS_PER_PAGE_IMAGE)
            calls = 2 * n
        elif what == "enrichment":
            count = len(self._selected())
            prompt, calls = count * 900, count
        elif what in ("T", "E2"):
            prompt, calls = n * arms.CHUNKS_TO_READER * TOKENS_PER_CHUNK, n
        else:
            prompt, calls = n * arms.PAGES_TO_READER * TOKENS_PER_PAGE_IMAGE, n
        completion = calls * (300 if what == "enrichment" else TOKENS_PER_ANSWER)
        cost = spend.estimate_tokens(model, prompt, completion)
        price = "price unknown" if cost is None else f"est. ${cost:,.4f}"
        return f"{what} on {model}: {calls} calls, about {prompt:,} input tokens, {price}"

    def _selected(self) -> list[Any]:
        pages = self.parse()
        found = self.elements()
        return enrich.describable_pictures(found, pages) + enrich.summarisable_tables(found, pages)

    # ----------------------------------------------------------------- model steps

    def _step(self, step: Step, compute: Any) -> StepResult:
        return load_or_run(step, compute, models=self.models, live=self.live, cache=self.cache, show=self.show)

    def figure_only_test(self) -> StepResult:
        """Answer each question from its page's text and from its image."""
        if "figure_only" not in self._memo:
            pages = self.parse()
            texts = {p: enrich.page_text_with_enrichment(self.corpus, p) for p in pages}
            records = [r for r in self.question_records() if r["gold_pages"]]

            judge_model = self.models.judging.model

            def compute(choice: ModelChoice) -> Any:
                def judge(prompt: str) -> str:
                    return self.llm.complete(judge_model, prompt, max_tokens=300).text

                def one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                    page = record["gold_pages"][0]
                    text_answer = answer_from(
                        self.llm, choice.model, record["question"], text=texts[page] or "(no text)"
                    )
                    image_answer = answer_from(
                        self.llm, choice.model, record["question"], images=[data_uri(render_page(self.corpus, page))]
                    )
                    return record["query_id"], {
                        "text": score_answer(record, text_answer, judge).correct,
                        "image": score_answer(record, image_answer, judge).correct,
                        "text_answer": text_answer,
                        "image_answer": image_answer,
                    }

                return dict(enrich.parallel_map(one, records))

            step = Step(
                RI_ID,
                "figure-only-test",
                "text-and-image",
                {
                    "pdfs": self.digest,
                    "questions": digest_of(self.question_records()),
                    "prompt": digest_of(ANSWER_PROMPT),
                },
                role="generation",
                reference_model=REFERENCE_VLM,
            )
            self._memo["figure_only"] = self._step(step, compute)
        return cast(StepResult, self._memo["figure_only"])

    def enrichment(self) -> StepResult:
        """Describe pictures and summarise complex tables with the extraction model."""
        if "enrichment" not in self._memo:
            pages = self.parse()
            found = self.elements()

            def compute(choice: ModelChoice) -> Any:
                pictures = enrich.describe_pictures(
                    self.llm,
                    choice.model,
                    self.corpus,
                    enrich.describable_pictures(found, pages),
                    prompt=enrich.DOMAIN_PICTURE_PROMPT,
                )
                tables = enrich.summarise_tables(
                    self.llm, choice.model, self.corpus, enrich.summarisable_tables(found, pages)
                )
                return {"pictures": pictures, "tables": tables}

            step = Step(
                RI_ID,
                "enrichment",
                "domain-prompt",
                {"pdfs": self.digest, "prompt": digest_of(enrich.DOMAIN_PICTURE_PROMPT, enrich.TABLE_SUMMARY_PROMPT)},
                role="extraction",
                reference_model=REFERENCE_VLM,
            )
            self._memo["enrichment"] = self._step(step, compute)
        return cast(StepResult, self._memo["enrichment"])

    def visual_rankings(self) -> dict[str, Any]:
        """Return VV's page rankings and what the page embedding cost on the CPU."""
        if "visual" not in self._memo:
            pages = self.parse()
            records = self.question_records()
            key = CacheKey(
                RI_ID,
                "own-visual-rankings",
                "colmodernvbert",
                {"pdfs": self.digest, "questions": digest_of(records), "model": visual.MODEL},
            )
            found = self.cache.get(key)
            if found is None:
                if not self.live:
                    raise LookupError(
                        "no cached page embeddings for these pages; set RUN_EXPENSIVE = True to embed them"
                    )

                def compute() -> JsonValue:
                    model = visual.load_model()
                    matrices, per_page = visual.embed_images(model, [render_page(self.corpus, p) for p in pages])
                    index = visual.PageIndex.build(list(pages), matrices, {"providers": ["CPUExecutionProvider"]})
                    queries, per_query = visual.embed_queries(
                        model, [str(r["question"]) for r in records], batch_size=1
                    )
                    hits = index.search_many(queries, k=10)
                    return {
                        "VV": {str(r["query_id"]): [p for p, _ in h] for r, h in zip(records, hits, strict=True)},
                        "costs": {
                            "seconds_per_page": round(per_page, 2),
                            "seconds_per_query": round(per_query, 2),
                            "index_megabytes": index.megabytes,
                        },
                    }

                found_value = self.cache.load_or_compute(key, compute).value
            else:
                found_value = found.value
            self._memo["visual"] = found_value
        return cast(dict[str, Any], self._memo["visual"])

    def text_rankings(self) -> dict[str, Any]:
        """Return chunk and page rankings for T and E2, which take seconds on a CPU."""
        if "text" not in self._memo:
            from .pipeline import _embedder  # noqa: PLC0415

            pages = self.parse()
            records = self.question_records()
            enriched = self.enrichment().value
            pictures, tables = enrich_texts(enriched["pictures"]), enrich_texts(enriched["tables"])
            texts = [str(r["question"]) for r in records]
            ids = [str(r["query_id"]) for r in records]
            embedder = _embedder()
            out: dict[str, Any] = {"chunk_text": {}, "chunk_page": {}, "chunks": {}, "pages": {}}
            for arm in ("T", "E2"):
                chunks, first_page = enrich.build_chunks(
                    self.corpus,
                    pages,
                    pictures=pictures if arm == "E2" else {},
                    tables=tables if arm == "E2" else {},
                    per_page=True,
                )
                ranked = arms.DenseChunks(chunks, embedder).search(texts, 30)
                out["chunk_text"][arm] = {c.chunk_id: c.text for c in chunks}
                out["chunk_page"][arm] = first_page
                out["chunks"][arm] = {q: r[: arms.CHUNKS_TO_READER] for q, r in zip(ids, ranked, strict=True)}
                out["pages"][arm] = {q: arms.collapse_to_pages(r, first_page) for q, r in zip(ids, ranked, strict=True)}
            out["page_text"] = {
                p: enrich.page_text_with_enrichment(self.corpus, p, pictures=pictures, tables=tables) for p in pages
            }
            self._memo["text"] = out
        return cast(dict[str, Any], self._memo["text"])

    def page_rankings(self) -> dict[str, dict[str, list[str]]]:
        """Return page rankings for T, E2, and VV."""
        text = self.text_rankings()
        return {"T": text["pages"]["T"], "E2": text["pages"]["E2"], "VV": self.visual_rankings()["VV"]}

    def answers(self, row: str) -> StepResult:
        """Return one arm's answers on your questions."""
        memo = f"answers:{row}"
        if memo not in self._memo:
            text = self.text_rankings()
            records = self.question_records()
            inputs = arms.ArmInputs(
                chunk_text=text["chunk_text"],
                chunk_page=text["chunk_page"],
                chunk_rankings=text["chunks"],
                page_rankings=self.page_rankings() if row == "VV" else {"T": text["pages"]["T"]},
                page_text=lambda page, _enriched: str(text["page_text"].get(page, "")),
            )

            def compute(choice: ModelChoice) -> Any:
                return arms.run_reader(
                    self.llm, model=choice.model, corpus=self.corpus, row=row, questions=records, inputs=inputs
                )

            step = Step(
                RI_ID,
                "reader-answers",
                f"row-{row}",
                {
                    "pdfs": self.digest,
                    "questions": digest_of(records),
                    "prompt": digest_of(ANSWER_PROMPT),
                    "chunking": "per-page",
                },
                role="generation",
                reference_model=REFERENCE_VLM,
                upstream={"enrichment": self.enrichment().digest},
            )
            self._memo[memo] = self._step(step, compute)
        return cast(StepResult, self._memo[memo])

    def verdicts(self, row: str) -> StepResult:
        """Score one arm's answers with the numeric match and the judging model."""
        memo = f"verdicts:{row}"
        if memo not in self._memo:
            answered = self.answers(row)
            records = {r["query_id"]: r for r in self.question_records()}

            def compute(choice: ModelChoice) -> Any:
                def judge(prompt: str) -> str:
                    return self.llm.complete(choice.model, prompt, max_tokens=300).text

                return {
                    qid: score_answer(records[qid], str(a["answer"]), judge).correct
                    for qid, a in sorted(answered.value.items())
                }

            step = Step(
                RI_ID,
                "answer-verdicts",
                f"row-{row}",
                {"tolerance": 0.01},
                role="judging",
                reference_model=REFERENCE_JUDGE,
                upstream={"answers": answered.digest},
            )
            self._memo[memo] = self._step(step, compute)
        return cast(StepResult, self._memo[memo])

    def visual_costs(self) -> dict[str, float]:
        """Return the CPU seconds per page and per question the page embedding took."""
        return cast(dict[str, float], self.visual_rankings()["costs"])


def enrich_texts(value: Mapping[str, Any]) -> dict[str, str]:
    """Return element identifier to description or summary text."""
    return {eid: str(r.get("text", "")) for eid, r in value.items() if r.get("text")}


def share(values: Sequence[bool]) -> float:
    """Return the share of true values, or NaN for none."""
    return sum(values) / len(values) if values else math.nan


__all__ = [
    "DEMO_NAME",
    "DEMO_QUESTIONS",
    "MAX_PAGES",
    "MAX_QUESTIONS",
    "ROWS",
    "OwnPages",
    "enrich_texts",
    "file_digest",
    "share",
    "write_demo_pdf",
]
