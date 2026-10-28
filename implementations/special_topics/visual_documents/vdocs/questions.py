"""Build the figure-only, table, control, and null questions the arms are compared on.

FinanceBench's 150 questions put one gold page on a large picture, so a test of
figures needs its own questions. The build follows the package's synthetic
question design (statement first, a round trip by a second model, in-domain
near-miss nulls verified against retrieval), adapted where the evidence is a
picture region and not a text span:

- Figure questions. A vision model sees a figure page and crops of its large
  pictures, and writes up to two statements, each a value or comparison read
  off a chart, with a question per statement. The statement has to name the
  chart element it reads from (an axis label, series, or legend entry), and
  that name has to appear in the page text or in the picture's description.
- Modality filter. A checker from another model answers each figure question
  twice from its gold page: from the page text alone and from the page image.
  A question the checker cannot answer from the image is dropped. One it
  answers from the image and not from the text is figure-only; one it answers
  from both is duplicated, because the chart's values are also printed on the
  page.
- Table questions. Code picks a numeric cell of a complex table with its row
  label and column header, and a model writes a question whose answer is that
  cell. The checker answers from the page text, and a question it misses is
  dropped. Each is tagged ``split`` when a plain 1,200-character window
  separates the cell from its header, and ``whole`` otherwise.
- Controls. FinanceBench questions on the same filings, with their human
  answers.
- Nulls. Near-miss rewrites of accepted figure and table questions, kept only
  when the checker finds no answer on the gold page image or in the top pages
  BM25 and dense retrieval return.
"""

from __future__ import annotations

import random
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.retrieval.tables import TableGrid, contains_value

from .enrich import document_label, page_items, page_text_with_enrichment, parallel_map
from .pages import FIGURE_PAGE_AREA, Corpus, Element, crop_element, data_uri, load_document, render_page, split_page_id
from .scoring import NOT_STATED, numeric_match, score_answer
from .vlm import ProxyModels, parse_json


FIGURE_PROMPT = """The first image is page {page} of {document}. The images after it are the
charts or pictures on that page, cropped.

Write up to two questions that can only be answered by reading a value off one
of the charts: a plotted value, a bar or slice label, or a difference between
two plotted values. For each, first write the statement you read from the
chart, then the question.

Rules:
- Each question must name the company and the filing year, so it makes sense
  with no page in front of the reader.
- Ask about the fact itself, as an analyst would. Do not mention the chart,
  graph, figure, image, or page in the question.
- The answer must be one number with its unit, read from the chart.
- Name the chart element the value is read from: the axis label, series name,
  legend entry, or category label exactly as the chart prints it.
- Prefer values that the chart shows and the page's running text does not.
- If no picture is a chart or diagram with readable values (a photo, a logo,
  or decoration), return an empty list.

Return one JSON object and nothing else:
{{"questions": [{{"statement": "...", "question": "...", "answer": "...",
"value": <number>, "unit": "...", "chart_label": "..."}}]}}"""

TABLE_PROMPT = """This table is from {document}, page {page}:

{table}

One cell holds {value}. Its row label is "{row}" and its column header is
"{column}".

Write one question whose answer is exactly that cell's value. The question
must name the company and the filing year, name the line item and the period
or column in the words a financial analyst would use, and make sense with no
table in front of the reader. Do not quote the value.

Return one JSON object and nothing else:
{{"question": "...", "answer": "<the value with its unit>"}}"""

NEAR_MISS_PROMPT = """This question about {document} is answered by the statement below.

Question: {question}
Statement: {statement}

Write a new question a reader of this filing might ask by changing ONE detail
that makes the statement answer it: the period, the segment or series, the
line item, or the measure. Choose a change the filing is unlikely to answer:
a year outside the range shown, or a series or segment that is not plotted
or tabulated. The new question must stay natural, name the company and the
filing year, and not be a negation or a trick.

Return one JSON object and nothing else:
{{"question": "...", "changed": "what you changed, in a few words"}}"""

ANSWER_PROMPT = (
    """Answer the question using only the evidence {where}.

Question: {question}

Reply in one short sentence that states the answer with its number and unit.
If the evidence does not contain the answer, reply exactly: """
    + NOT_STATED
)

_FOLD = re.compile(r"[^a-z0-9%$.]+")


def _fold(text: str) -> str:
    return _FOLD.sub(" ", text.lower()).strip()


class Funnel:
    """Count candidates through each stage, by question kind.

    A stage's pass rate is the share of candidates of that kind that entered it
    and passed. Stages are listed in the order they were first entered.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entered: Counter[tuple[str, str]] = Counter()
        self.passed: Counter[tuple[str, str]] = Counter()
        self.order: list[tuple[str, str]] = []

    def check(self, kind: str, stage: str, ok: bool) -> bool:
        """Count one candidate through a stage and return ``ok``."""
        with self._lock:
            key = (kind, stage)
            if key not in self.entered:
                self.order.append(key)
            self.entered[key] += 1
            if ok:
                self.passed[key] += 1
        return ok

    def to_rows(self) -> list[dict[str, Any]]:
        """Return one row per kind and stage, in order."""
        return [
            {"kind": kind, "stage": stage, "entered": self.entered[(kind, stage)], "passed": self.passed[(kind, stage)]}
            for kind, stage in self.order
        ]


def answer_from(
    llm: ProxyModels,
    model: str,
    question: str,
    *,
    text: str | None = None,
    images: Sequence[str] = (),
    max_tokens: int = 200,
) -> str:
    """Answer from page text, page images, or both, with the shared reader prompt."""
    where = "in the images" if images and not text else "below" if not images else "in the images and below"
    prompt = ANSWER_PROMPT.format(where=where, question=question)
    if text:
        prompt = f"Evidence:\n{text}\n\n{prompt}"
    return llm.complete(model, prompt, images, max_tokens=max_tokens).text.strip()


def _judge(llm: ProxyModels, model: str) -> Callable[[str], str]:
    def complete(prompt: str) -> str:
        return llm.complete(model, prompt, max_tokens=300).text

    return complete


def figure_candidates(
    llm: ProxyModels,
    *,
    model: str,
    corpus: Corpus,
    figure_pages: Sequence[str],
    pictures: Sequence[Element],
    funnel: Funnel,
) -> list[dict[str, Any]]:
    """Ask the generator for up to two chart questions per figure page."""
    by_page: dict[str, list[Element]] = {}
    for element in pictures:
        if element.kind == "picture" and element.area >= FIGURE_PAGE_AREA:
            by_page.setdefault(element.page_id, []).append(element)

    def one(identifier: str) -> list[dict[str, Any]]:
        doc_id, page = split_page_id(identifier)
        crops = sorted(by_page.get(identifier, []), key=lambda e: -e.area)[:2]
        images = [data_uri(render_page(corpus, identifier))] + [data_uri(crop_element(corpus, e)) for e in crops]
        reply = llm.complete(
            model,
            FIGURE_PROMPT.format(page=page + 1, document=document_label(doc_id)),
            images,
            max_tokens=900,
        )
        try:
            items = parse_json(reply.text).get("questions", [])
        except (ValueError, AttributeError):
            items = []
        funnel.check("figure", "page yields a chart question", bool(items))
        out = []
        for index, item in enumerate(items[:2]):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "candidate_id": f"{identifier}/q{index}",
                    "doc_id": doc_id,
                    "gold_pages": [identifier],
                    "element_ids": [e.element_id for e in crops],
                    **{k: item.get(k) for k in ("statement", "question", "answer", "value", "unit", "chart_label")},
                },
            )
        return out

    return [c for found in parallel_map(one, list(figure_pages)) for c in found]


def region_check(candidate: Mapping[str, Any], page_text: str, descriptions: Sequence[str]) -> bool:
    """Return whether the named chart element is in the page text or a description."""
    label = _fold(str(candidate.get("chart_label") or ""))
    if len(label) < 2:
        return False
    haystack = _fold(page_text + " " + " ".join(descriptions))
    return label in haystack


def filter_figure_questions(
    llm: ProxyModels,
    *,
    checker: str,
    judge: str,
    corpus: Corpus,
    candidates: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str],
    funnel: Funnel,
) -> list[dict[str, Any]]:
    """Apply the region check, the image round trip, and the modality split."""
    judged = _judge(llm, judge)

    def one(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
        record = dict(candidate)
        value = record.get("value")
        well_formed = isinstance(value, (int, float)) and not isinstance(value, bool) and bool(record.get("question"))
        if not funnel.check("figure", "one numeric answer", well_formed):
            return None
        identifier = record["gold_pages"][0]
        text = page_text_with_enrichment(corpus, identifier)
        notes = [descriptions.get(e, "") for e in record["element_ids"]]
        if not funnel.check("figure", "chart element named on the page", region_check(record, text, notes)):
            return None
        probe = {"query_id": record["candidate_id"], "group": "figure", **record}
        image_answer = answer_from(llm, checker, record["question"], images=[data_uri(render_page(corpus, identifier))])
        image_ok = score_answer(probe, image_answer, judged).correct
        if not funnel.check("figure", "checker answers from the page image", image_ok):
            return None
        text_answer = answer_from(llm, checker, record["question"], text=text or "(this page has no text)")
        text_ok = score_answer(probe, text_answer, judged).correct
        record["subgroup"] = "duplicated" if text_ok else "figure-only"
        funnel.check("figure", "figure-only (text alone fails)", not text_ok)
        return record

    return [r for r in parallel_map(one, list(candidates)) if r is not None]


def table_facts(corpus: Corpus, table: Element, rng: random.Random) -> dict[str, Any] | None:
    """Pick one numeric cell of a table, with its row label and column header."""
    document = load_document(corpus, table.doc_id)
    for item in document.tables:
        if item.self_ref != table.self_ref:
            continue
        grid = TableGrid.from_table(item)
        facts = [
            f
            for f in grid.facts()
            if f.is_answerable and len(re.sub(r"\D", "", f.value)) >= 2 and len(f.row_label) <= 120
        ]
        if not facts:
            return None
        fact = rng.choice(facts)
        return {
            "markdown": grid.to_markdown(),
            "value": fact.value,
            "row": fact.row_label,
            "column": fact.column_header,
            "header_parts": list(fact.header_parts),
        }
    return None


def split_by_plain_chunking(corpus: Corpus, identifier: str, fact: Mapping[str, Any]) -> bool:
    """Return whether plain 1,200-character windows separate a cell from its header."""
    doc_id, page = split_page_id(identifier)
    text = "\n\n".join(item.text for item in page_items(corpus, doc_id, [page]))
    windows = [c.text for c in chunk_document(doc_id, text, chunk_chars=1200, overlap_chars=200)]
    holding = [w for w in windows if contains_value(w, str(fact["value"]))]
    if not holding:
        return True
    return not any(all(part in w for part in fact["header_parts"]) and fact["row"] in w for w in holding)


def table_questions(
    llm: ProxyModels,
    *,
    generator: str,
    checker: str,
    judge: str,
    corpus: Corpus,
    tables: Sequence[Element],
    funnel: Funnel,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Write and check one question per complex-table page."""
    rng = random.Random(seed)
    picked: list[tuple[Element, dict[str, Any]]] = []
    seen_pages: set[str] = set()
    for table in sorted(tables, key=lambda e: e.element_id):
        if table.page_id in seen_pages:
            continue
        fact = table_facts(corpus, table, rng)
        if funnel.check("table", "numeric cell with header and row label", fact is not None) and fact:
            picked.append((table, fact))
            seen_pages.add(table.page_id)
    judged = _judge(llm, judge)

    def one(pair: tuple[Element, dict[str, Any]]) -> dict[str, Any] | None:
        table, fact = pair
        doc_id, page = split_page_id(table.page_id)
        reply = llm.complete(
            generator,
            TABLE_PROMPT.format(
                document=document_label(doc_id),
                page=page + 1,
                table=fact["markdown"][:6000],
                value=fact["value"],
                row=fact["row"],
                column=fact["column"],
            ),
            max_tokens=300,
        )
        try:
            written = parse_json(reply.text)
        except ValueError:
            written = {}
        question = str(written.get("question", "")).strip()
        if not funnel.check("table", "question written", bool(question)):
            return None
        numbers = re.sub(r"[^\d.]", "", fact["value"])
        value = float(numbers) if numbers.replace(".", "", 1).isdigit() else None
        record = {
            "candidate_id": f"{table.page_id}/t0",
            "doc_id": doc_id,
            "gold_pages": [table.page_id],
            "element_ids": [table.element_id],
            "statement": f"{fact['row']} | {fact['column']} | {fact['value']}",
            "question": question,
            "answer": str(written.get("answer") or fact["value"]),
            "value": value,
            "unit": "",
        }
        text = page_text_with_enrichment(corpus, table.page_id)
        probe = {"query_id": record["candidate_id"], "group": "table", **record}
        reply_text = answer_from(llm, checker, question, text=text)
        if not funnel.check(
            "table", "checker answers from the page text", score_answer(probe, reply_text, judged).correct
        ):
            return None
        record["subgroup"] = "split" if split_by_plain_chunking(corpus, table.page_id, fact) else "whole"
        return record

    return [r for r in parallel_map(one, picked) if r is not None]


def near_miss_nulls(
    llm: ProxyModels,
    *,
    generator: str,
    checker: str,
    corpus: Corpus,
    sources: Sequence[Mapping[str, Any]],
    retrieve_pages: Callable[[str], list[str]],
    page_text: Callable[[str], str],
    funnel: Funnel,
) -> list[dict[str, Any]]:
    """Rewrite accepted questions into near misses that nothing in the corpus answers.

    Parameters
    ----------
    llm : ProxyModels
        The proxy client.
    generator, checker : str
        The model that rewrites and the model that verifies.
    corpus : Corpus
        Where the PDFs are.
    sources : sequence of mapping
        Accepted figure and table questions to rewrite.
    retrieve_pages : callable
        Question to the union of the top pages BM25 and dense retrieval return.
    page_text : callable
        Page identifier to the text the checker reads.
    funnel : Funnel
        Stage counts.

    Returns
    -------
    list of dict
        Null questions, each naming the page it was rewritten from.
    """

    def one(source: Mapping[str, Any]) -> dict[str, Any] | None:
        reply = llm.complete(
            generator,
            NEAR_MISS_PROMPT.format(
                document=document_label(str(source["doc_id"])),
                question=source["question"],
                statement=source.get("statement") or source.get("answer"),
            ),
            max_tokens=300,
        )
        try:
            written = parse_json(reply.text)
        except ValueError:
            written = {}
        question = str(written.get("question", "")).strip()
        if not funnel.check("null", "near-miss written", bool(question)):
            return None
        page = source["gold_pages"][0]
        on_image = answer_from(llm, checker, question, images=[data_uri(render_page(corpus, page))])
        if not funnel.check("null", "gold page image does not answer", NOT_STATED in on_image.upper()):
            return None
        pages = retrieve_pages(question)
        evidence = "\n\n".join(f"[{p}]\n{page_text(p)[:4000]}" for p in pages)
        in_corpus = answer_from(llm, checker, question, text=evidence, max_tokens=200)
        if not funnel.check("null", "top retrieved pages do not answer", NOT_STATED in in_corpus.upper()):
            return None
        return {
            "candidate_id": f"{source['candidate_id']}/null",
            "doc_id": source["doc_id"],
            "gold_pages": [],
            "rewritten_from": page,
            "element_ids": [],
            "statement": "",
            "question": question,
            "answer": NOT_STATED,
            "value": None,
            "unit": "",
            "changed": str(written.get("changed", "")),
            "subgroup": f"near-miss {source['group']}",
        }

    return [r for r in parallel_map(one, list(sources)) if r is not None]


def assign_ids(records: Sequence[Mapping[str, Any]], prefix: str, group: str) -> list[dict[str, Any]]:
    """Give each accepted record a stable query identifier and its group."""
    return [{"query_id": f"{prefix}-{index:04d}", "group": group, **record} for index, record in enumerate(records)]


def stratified_subset(
    questions: Sequence[Mapping[str, Any]], size: int, seed: int = 0, key: str = "subgroup"
) -> list[str]:
    """Return a subset's question identifiers, keeping each subgroup's share."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for q in questions:
        groups.setdefault(str(q.get(key) or q["group"]), []).append(str(q["query_id"]))
    total = len(questions)
    chosen: list[str] = []
    for name in sorted(groups):
        members = sorted(groups[name])
        rng.shuffle(members)
        share = max(1, round(size * len(members) / total)) if total else 0
        chosen.extend(members[:share])
    return sorted(chosen[: size + len(groups)])


__all__ = [
    "ANSWER_PROMPT",
    "FIGURE_PROMPT",
    "NEAR_MISS_PROMPT",
    "TABLE_PROMPT",
    "Funnel",
    "answer_from",
    "assign_ids",
    "figure_candidates",
    "filter_figure_questions",
    "near_miss_nulls",
    "numeric_match",
    "region_check",
    "split_by_plain_chunking",
    "stratified_subset",
    "table_facts",
    "table_questions",
]
