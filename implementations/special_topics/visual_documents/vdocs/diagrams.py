"""Test the diagram claim on ViDoRe v3 physics slides.

The claim under test: OCR returns almost nothing useful on diagrams, while a
vision-model description recovers sequence and connection information. This
module holds what :mod:`diagram_claim` needs that calls no model: the data
reader, the page representations that come from the PDF or from local OCR, the
prompts, and the scoring arithmetic.

A page is turned into text in several ways, and each is scored on whether it
states the facts a query's reference answer needs:

- ``text-layer``: the PDF's own text, read with pypdfium2.
- ``ocr``: RapidOCR on the page image, lines in reading order.
- ``ocr-boxes``: the same lines, each with its box as a share of the page, so
  a reader can use the layout.
- ``vlm-generic`` and ``vlm-relations``: a vision model's description with
  notebook 01's generic prompt, and with a prompt that asks for every arrow,
  line, and order in the diagrams.
- ``vlm-text-only``: the relations prompt on a copy of the page where
  everything outside the OCR boxes is painted white, so the arrows and lines
  are gone and the labels stay. It separates what the model reads from the
  drawing from what it infers from the labels and its own physics.

The ViDoRe v3 physics slides are French course decks under CC BY 4.0. Nothing
here writes their text to the repository; page images, representations, and
model replies stay in the local cache, and the committed results hold
identifiers and counts.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any


#: Hugging Face dataset and the revision the reference run read.
DATASET = "vidore/vidore_v3_physics"
REVISION = "main"
#: Query ids are the same question in six languages, in blocks of 302.
LANGUAGE_OFFSET = {"french": 0, "english": 302}
#: Long side of the page image sent to a vision model and to OCR, in pixels.
#: The dataset's renders are 2667 x 1500.
IMAGE_LONG_SIDE = 1600
#: Region labels in the qrels that mark a picture rather than text or a table.
VISUAL_TYPES = frozenset({"Infographic", "Image"})

SELECT_PROMPT = """The image is a slide from a French physics course. A reader asked:

Question: {query}
Reference answer: {answer}

Does answering need a relation that this slide shows through the drawing itself rather than in
its sentences: an arrow, a line, a flow, an order of steps, a connection between parts, or a
spatial arrangement? A label that sits next to a picture does not count; a fact stated in a
sentence does not count.

Return one JSON object and nothing else:
{{"diagram_relation": true or false, "kind": "sequence|connection|arrangement|none"}}"""

RELATIONS_PROMPT = """The image is a slide. Transcribe its text, then describe each diagram on it:
name every element, and state every arrow, line, flow, order of steps, and connection between
elements as "A -> B: what the link means". Use only what the slide shows."""

READER_PROMPT = """Answer the question from the page content below, which was extracted from one
slide of a physics course. Answer in English, in at most three sentences. If the content does not
give the answer, reply NOT STATED.

Page content:
<<<
{content}
>>>

Question: {query}"""

IMAGE_READER_PROMPT = """Answer the question from the slide above. Answer in English, in at most
three sentences. If the slide does not give the answer, reply NOT STATED.

Question: {query}"""

FACTS_PROMPT = """Split this reference answer to a question into its atomic facts, at most eight.
Mark a fact "relational" when it states an order, a sequence, a cause leading to an effect, a flow,
or a connection between two named things; otherwise "plain".

Question: {query}
Reference answer: {answer}

Return one JSON object and nothing else:
{{"facts": [{{"fact": "<short statement>", "kind": "relational|plain"}}]}}"""

COVERAGE_PROMPT = """Below are numbered facts and a text extracted from a slide (it may be French,
OCR output with coordinates, or a description of the slide). For each fact, decide whether the text
states it or something that directly implies it. A fact the text only hints at, or that needs
outside knowledge, is not stated.

Facts:
{facts}

Text:
<<<
{content}
>>>

Return one JSON object and nothing else:
{{"stated": [<numbers of the facts the text states>]}}"""

GOLD_RELATIONS_PROMPT = """The image is a slide from a French physics course. List every relation that the
slide's drawing shows between two labelled elements: an arrow, a connecting line, a flow, an order
of steps, or one element placed inside or along another. Name each element by its label on the
slide. Do not list a relation that only a sentence on the slide states, and do not add physics the
slide does not show. List at most twelve.

Return one JSON object and nothing else:
{"relations": [{"relation": "<element A> -> <element B>: <what the link means>"}]}"""

ANSWER_JUDGE_PROMPT = """Compare a candidate answer with a reference answer to the same question.
Mark it correct when it gives the reference's main point and contradicts none of it; extra true
detail is fine. NOT STATED is incorrect.

Question: {query}
Reference answer: {answer}
Candidate answer: {candidate}

Return one JSON object and nothing else: {{"correct": true or false}}"""


@dataclass(frozen=True)
class Page:
    """One corpus page: where it sits in the parquet files and in its PDF."""

    corpus_id: int
    doc_id: str
    page: int
    shard: str
    row: int


class Physics:
    """Read ``vidore_v3_physics`` from a local snapshot of the dataset repository.

    Parameters
    ----------
    root : Path
        The snapshot folder, holding ``corpus/``, ``queries/``, ``qrels/``, and
        ``pdfs/``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @cached_property
    def queries(self) -> Any:
        """Return the queries table, every language."""
        import pandas as pd  # noqa: PLC0415

        return pd.read_parquet(self.root / "queries" / "test-00000-of-00001.parquet")

    @cached_property
    def qrels(self) -> Any:
        """Return the relevance judgements, with each region's content type."""
        import pandas as pd  # noqa: PLC0415

        return pd.read_parquet(self.root / "qrels" / "test-00000-of-00001.parquet")

    @cached_property
    def pages(self) -> dict[int, Page]:
        """Return every corpus page by id, read without the image column."""
        import pyarrow.parquet as pq  # noqa: PLC0415

        out: dict[int, Page] = {}
        for shard in sorted((self.root / "corpus").glob("*.parquet")):
            table = pq.read_table(shard, columns=["corpus_id", "doc_id", "page_number_in_doc"])
            for row, (cid, doc, page) in enumerate(
                zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True)
            ):
                out[int(cid)] = Page(int(cid), str(doc), int(page), shard.name, row)
        return out

    def query(self, query_id: int, language: str = "english") -> Mapping[str, Any]:
        """Return one query's row in a language, by its English id."""
        wanted = int(query_id) - LANGUAGE_OFFSET["english"] + LANGUAGE_OFFSET[language]
        frame = self.queries
        row: Mapping[str, Any] = frame[frame.query_id == wanted].iloc[0].to_dict()
        return row

    def english(self) -> list[Mapping[str, Any]]:
        """Return the English queries."""
        frame = self.queries
        return list(frame[frame.language == "english"].to_dict("records"))

    def images(self, corpus_ids: Iterable[int], dest: Path) -> dict[int, Path]:
        """Write each page's image as a JPEG under ``dest`` and return the paths.

        Pages already written are skipped, so a rerun reads no parquet.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        dest.mkdir(parents=True, exist_ok=True)
        wanted = {int(c) for c in corpus_ids}
        paths = {c: dest / f"{c}.jpg" for c in wanted}
        missing = {c for c, p in paths.items() if not p.is_file()}
        by_shard: dict[str, set[int]] = {}
        for cid in missing:
            by_shard.setdefault(self.pages[cid].shard, set()).add(cid)
        for shard, ids in by_shard.items():
            parquet = pq.ParquetFile(self.root / "corpus" / shard)
            for group in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(group, columns=["corpus_id", "image"])
                for cid, image in zip(
                    table.column("corpus_id").to_pylist(), table.column("image").to_pylist(), strict=True
                ):
                    if int(cid) in ids:
                        picture = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                        picture.thumbnail((IMAGE_LONG_SIDE, IMAGE_LONG_SIDE))
                        picture.save(paths[int(cid)], "JPEG", quality=90)
        return paths

    def text_layer(self, corpus_id: int) -> str:
        """Return the PDF text layer of one page."""
        import pypdfium2 as pdfium  # noqa: PLC0415

        page = self.pages[int(corpus_id)]
        document = pdfium.PdfDocument(str(self.root / "pdfs" / f"{page.doc_id}.pdf"))
        try:
            text: str = document[page.page].get_textpage().get_text_range()
        finally:
            document.close()
        return text


# ------------------------------------------------------------------ selection


def visual_pairs(qrels: Any, english_ids: Iterable[int]) -> list[tuple[int, int, int]]:
    """Return (query, corpus id, score) for English pairs whose region is a picture."""
    wanted = {int(q) for q in english_ids}
    out = []
    for row in qrels.itertuples(index=False):
        if int(row.query_id) in wanted and VISUAL_TYPES & {str(t) for t in row.content_type}:
            out.append((int(row.query_id), int(row.corpus_id), int(row.score)))
    return sorted(out)


def text_pairs(qrels: Any, english_ids: Iterable[int]) -> list[tuple[int, int, int]]:
    """Return (query, corpus id, score) for English pairs whose regions are all text."""
    wanted = {int(q) for q in english_ids}
    out = []
    for row in qrels.itertuples(index=False):
        if int(row.query_id) in wanted and {str(t) for t in row.content_type} == {"Text"}:
            out.append((int(row.query_id), int(row.corpus_id), int(row.score)))
    return sorted(out)


def best_page_per_query(pairs: Iterable[tuple[int, int, int]]) -> dict[int, int]:
    """Keep one page per query: the highest score, then the lowest corpus id."""
    best: dict[int, tuple[int, int]] = {}
    for query_id, corpus_id, score in pairs:
        current = best.get(query_id)
        if current is None or (score, -corpus_id) > (current[1], -current[0]):
            best[query_id] = (corpus_id, score)
    return {q: c for q, (c, _) in sorted(best.items())}


# ------------------------------------------------------------- representations


def ocr_lines(image: Any, engine: Any) -> list[dict[str, Any]]:
    """Run RapidOCR on a PIL image; return lines with boxes as shares of the page.

    Lines come in reading order: top to bottom by the box's top edge in bands
    of 2% of the page height, then left to right.
    """
    import numpy as np  # noqa: PLC0415

    width, height = image.size
    result = engine(np.array(image))
    lines = []
    for box, text in zip(result.boxes if result.boxes is not None else [], result.txts or (), strict=False):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(
            {
                "text": str(text),
                "box": [
                    round(min(xs) / width, 4),
                    round(min(ys) / height, 4),
                    round(max(xs) / width, 4),
                    round(max(ys) / height, 4),
                ],
            }
        )
    lines.sort(key=lambda line: (round(line["box"][1] / 0.02), line["box"][0]))
    return lines


def ocr_text(lines: Sequence[Mapping[str, Any]]) -> str:
    """Return OCR lines as plain text, one per line."""
    return "\n".join(str(line["text"]) for line in lines)


def ocr_with_boxes(lines: Sequence[Mapping[str, Any]]) -> str:
    """Return OCR lines prefixed by their box, in hundredths of the page size."""
    out = ["Each line: [left top right bottom] in hundredths of the slide's width and height, then the text."]
    for line in lines:
        left, top, right, bottom = (round(v * 100) for v in line["box"])
        out.append(f"[{left} {top} {right} {bottom}] {line['text']}")
    return "\n".join(out)


def text_only_image(image: Any, lines: Sequence[Mapping[str, Any]], pad: float = 0.004) -> Any:
    """Return a copy of the page with everything outside the OCR boxes painted white."""
    from PIL import Image  # noqa: PLC0415

    width, height = image.size
    out = Image.new("RGB", image.size, "white")
    for line in lines:
        left, top, right, bottom = line["box"]
        box = (
            max(0, int((left - pad) * width)),
            max(0, int((top - pad) * height)),
            min(width, int((right + pad) * width)),
            min(height, int((bottom + pad) * height)),
        )
        out.paste(image.crop(box), box[:2])
    return out


# --------------------------------------------------------------------- scoring


def parse_facts(found: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the facts a judge listed, each of kind ``relational`` or ``plain``."""
    facts = found.get("facts")
    if not isinstance(facts, list):
        return []
    out = []
    for item in facts:
        if isinstance(item, Mapping) and str(item.get("fact", "")).strip():
            kind = "relational" if str(item.get("kind", "")).strip().lower() == "relational" else "plain"
            out.append({"fact": str(item["fact"]).strip(), "kind": kind})
    return out


def parse_relations(found: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return a gold-relation reply as facts of kind ``relational``."""
    items = found.get("relations")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        text = str(item.get("relation", "")).strip() if isinstance(item, Mapping) else ""
        if text:
            out.append({"fact": text, "kind": "relational"})
    return out


def numbered(facts: Sequence[Mapping[str, str]]) -> str:
    """Return facts as a numbered list starting at 1, for the coverage prompt."""
    return "\n".join(f"{i}. {f['fact']}" for i, f in enumerate(facts, start=1))


def stated_indices(found: Mapping[str, Any], count: int) -> set[int]:
    """Return the zero-based indices of facts a coverage reply says are stated."""
    raw = found.get("stated")
    if not isinstance(raw, list):
        return set()
    out = set()
    for value in raw:
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            out.add(index)
    return out


def coverage(
    facts: Mapping[str, Sequence[Mapping[str, str]]], stated: Mapping[str, set[int]], kind: str | None = None
) -> dict[str, Any]:
    """Return fact recall over pairs, and the share of pairs with every fact stated.

    Parameters
    ----------
    facts : mapping
        Pair key to its facts.
    stated : mapping
        Pair key to the indices of the facts a representation states.
    kind : str or None, optional
        Count only facts of this kind; pairs with none are left out.
    """
    total = hit = pairs = complete = 0
    for key, items in facts.items():
        chosen = [i for i, f in enumerate(items) if kind is None or f["kind"] == kind]
        if not chosen:
            continue
        got = sum(1 for i in chosen if i in stated.get(key, set()))
        total += len(chosen)
        hit += got
        pairs += 1
        complete += got == len(chosen)
    return {
        "pairs": pairs,
        "facts": total,
        "recall": round(hit / total, 4) if total else None,
        "all_stated": round(complete / pairs, 4) if pairs else None,
    }


def parse_flag(reply: str, field: str) -> bool | None:
    """Return a boolean field from a JSON reply, or None when it is missing."""
    from .vlm import parse_json  # noqa: PLC0415

    try:
        found = parse_json(reply)
    except (ValueError, json.JSONDecodeError):
        return None
    value = found.get(field)
    return value if isinstance(value, bool) else None


__all__ = [
    "ANSWER_JUDGE_PROMPT",
    "COVERAGE_PROMPT",
    "DATASET",
    "FACTS_PROMPT",
    "GOLD_RELATIONS_PROMPT",
    "IMAGE_LONG_SIDE",
    "IMAGE_READER_PROMPT",
    "LANGUAGE_OFFSET",
    "READER_PROMPT",
    "RELATIONS_PROMPT",
    "SELECT_PROMPT",
    "VISUAL_TYPES",
    "Page",
    "Physics",
    "best_page_per_query",
    "coverage",
    "numbered",
    "ocr_lines",
    "ocr_text",
    "ocr_with_boxes",
    "parse_facts",
    "parse_flag",
    "parse_relations",
    "stated_indices",
    "text_only_image",
    "text_pairs",
    "visual_pairs",
]
