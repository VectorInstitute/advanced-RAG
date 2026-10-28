"""The chunk record, the query record, and CUAD as a judged retrieval corpus.

One property makes every comparison in this harness possible: a chunk keeps the
character offsets of the region it came from. Relevance is then derived from
overlap with a gold answer span, so a chunker that reports honest offsets can be
scored without any fuzzy text matching, and moving a boundary moves the
judgments with it.

CUAD is the corpus the harness was built on. It ships as SQuAD-style extractive
QA over contracts: each contract is one long ``context`` string, and each answer
carries a character offset into it. The judgments come from the dataset, so
retrieval-failure rate is measured rather than estimated by a model.

Two properties of CUAD shape everything downstream:

1. Questions are templated per category. All 41 questions repeat verbatim for
   every contract, differing only in the category and its "Details" gloss. A
   query therefore carries no signal about which contract it belongs to, so
   retrieval is scoped to one contract and the task is locating the clause
   inside it.
2. Most questions are unanswerable. Around 70% of the test split is
   ``is_impossible``: the category simply does not appear in that contract.
   Those are kept as the null cases that "not stated" detection needs, and they
   are excluded from the retrieval metrics, which are only defined where a gold
   span exists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, with offsets back into its source document.

    Parameters
    ----------
    chunk_id : str
        Unique within a run. The convention is ``"{doc_id}::{index:04d}"``.
    doc_id : str
        The document this came from. CUAD retrieval is scoped to it.
    text : str
        The document text between ``start`` and ``end``.
    start : int
        Character offset of the first character, into the source document.
    end : int
        Character offset one past the last character. Half-open with ``start``.
    embed_text : str or None, optional
        What to index, when that differs from ``text``. ``None`` indexes
        ``text``. Contextual retrieval sets it to the span with an LLM-written
        situating sentence prepended, and the span stays the unit of judgment.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    embed_text: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open character span as a pair."""
        return (self.start, self.end)

    @property
    def contract_id(self) -> str:
        """Return ``doc_id`` under the name the CUAD code uses for it."""
        return self.doc_id

    @property
    def indexed_text(self) -> str:
        """Return the string a retriever indexes for this chunk."""
        return self.embed_text if self.embed_text is not None else self.text

    def with_embed_text(self, value: str) -> Chunk:
        """Return a copy that indexes ``value`` while keeping the same span."""
        return Chunk(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            text=self.text,
            start=self.start,
            end=self.end,
            embed_text=value,
        )


@dataclass(frozen=True)
class AnswerSpan:
    """A gold answer with its character offsets into the source document.

    Parameters
    ----------
    text : str
        The answer text as the dataset records it.
    start : int
        Character offset of the first character.
    end : int
        Character offset one past the last character.
    """

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Query:
    """One question against one document.

    Parameters
    ----------
    query_id : str
        Unique within a run.
    doc_id : str
        The document the question is scoped to.
    category : str
        The clause category, parsed out of a templated CUAD question.
    question : str
        The question text, encoded as-is by the dense arm.
    answers : tuple of AnswerSpan, optional
        Gold spans. Empty for an unanswerable question.
    """

    query_id: str
    doc_id: str
    category: str
    question: str
    answers: tuple[AnswerSpan, ...] = field(default=())

    @property
    def is_answerable(self) -> bool:
        """Return whether the dataset records a gold span for this question."""
        return bool(self.answers)

    @property
    def contract_id(self) -> str:
        """Return ``doc_id`` under the name the CUAD code uses for it."""
        return self.doc_id


# CUAD question text is 'Highlight the parts (if any) of this contract related
# to "<Category>" that should be reviewed by a lawyer. Details: <gloss>'.
_CATEGORY_RE = re.compile(r'related to "([^"]+)"')

Span = AnswerSpan | tuple[int, int]


def parse_category(question: str) -> str:
    """Pull the clause category out of a templated CUAD question.

    Parameters
    ----------
    question : str
        The question text.

    Returns
    -------
    str
        The quoted category, or ``"unknown"`` when the template does not match.
    """
    match = _CATEGORY_RE.search(question)
    return match.group(1) if match else "unknown"


def chunk_contract(
    doc_id: str,
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split a document into overlapping fixed-width character windows.

    Fixed-width windows on characters, rather than tokens or sentences, keep the
    offset arithmetic exact. A gold span is located by character offset, so any
    chunker that preserves offsets lets relevance be computed without fuzzy text
    matching. This is the deliberately plain baseline the other strategies in
    ``chunkers`` are measured against, and every number published for this
    project was measured at 1200 and 200.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    text : str
        The whole document.
    chunk_chars : int, optional
        Window width in characters.
    overlap_chars : int, optional
        Characters each window repeats from the one before it.

    Returns
    -------
    list of Chunk
        Windows in document order, blank ones dropped.

    Raises
    ------
    ValueError
        If ``chunk_chars`` is not positive, or ``overlap_chars`` is outside
        ``[0, chunk_chars)``.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must be in [0, chunk_chars)")

    stride = chunk_chars - overlap_chars
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, max(len(text), 1), stride)):
        end = min(start + chunk_chars, len(text))
        window = text[start:end]
        if window.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{index:04d}",
                    doc_id=doc_id,
                    text=window,
                    start=start,
                    end=end,
                ),
            )
        if end >= len(text):
            break
    return chunks


def relevant_chunk_ids(
    chunks: Sequence[Chunk],
    spans: Iterable[Span],
    doc_id: str | None = None,
) -> set[str]:
    """Return the chunks whose window overlaps any of ``spans``.

    Overlap rather than containment: with a 1200-character window and answers
    that can run longer, requiring full containment would mark a chunk holding
    most of the evidence as irrelevant.

    Parameters
    ----------
    chunks : sequence of Chunk
        Candidate chunks.
    spans : iterable
        Gold regions, each an ``AnswerSpan`` or a ``(start, end)`` pair.
    doc_id : str or None, optional
        Restrict the comparison to chunks of one document. ``None`` compares
        against every chunk given, which is what the per-contract CUAD path
        does because it passes one contract's chunks in.

    Returns
    -------
    set of str
        Chunk identifiers.
    """
    relevant: set[str] = set()
    for span in spans:
        start, end = (span.start, span.end) if isinstance(span, AnswerSpan) else span
        for chunk in chunks:
            if doc_id is not None and chunk.doc_id != doc_id:
                continue
            if chunk.start < end and start < chunk.end:
                relevant.add(chunk.chunk_id)
    return relevant


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float | int]:
    """Return the size distribution of a chunk set, in characters.

    Parameters
    ----------
    chunks : sequence of Chunk
        The chunk set to describe.

    Returns
    -------
    dict
        Count, mean, median, 10th and 90th percentile, minimum, and maximum
        characters. An empty set returns only ``{"chunks": 0}``.
    """
    if not chunks:
        return {"chunks": 0}
    lengths = sorted(len(c.text) for c in chunks)
    count = len(lengths)
    return {
        "chunks": count,
        "mean_chars": round(sum(lengths) / count, 1),
        "median_chars": lengths[count // 2],
        "p10_chars": lengths[int(0.10 * count)],
        "p90_chars": lengths[int(0.90 * count)],
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
    }


def load_cuad(path: Path) -> Iterator[tuple[str, str, list[Query]]]:
    """Yield ``(doc_id, document_text, queries)`` from a CUAD SQuAD file.

    Parameters
    ----------
    path : Path
        The ``test.json`` from the CUAD release.

    Yields
    ------
    tuple of (str, str, list of Query)
        One contract at a time, in file order.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for document in payload["data"]:
        doc_id = document["title"]
        for paragraph in document["paragraphs"]:
            text = paragraph["context"]
            queries = []
            for qa in paragraph["qas"]:
                answers = tuple(
                    AnswerSpan(
                        text=answer["text"],
                        start=int(answer["answer_start"]),
                        end=int(answer["answer_start"]) + len(answer["text"]),
                    )
                    for answer in qa["answers"]
                )
                queries.append(
                    Query(
                        query_id=qa["id"],
                        doc_id=doc_id,
                        category=parse_category(qa["question"]),
                        question=qa["question"],
                        answers=answers,
                    ),
                )
            yield doc_id, text, queries


@dataclass
class Slice:
    """A prepared evaluation slice: chunks, queries, and derived judgments.

    Parameters
    ----------
    chunks : list of Chunk
        Every chunk in the slice, across every document.
    queries : list of Query
        Every query, answerable and null.
    qrels : dict
        Query identifier to the set of relevant chunk identifiers. A null query
        maps to the empty set.
    """

    chunks: list[Chunk]
    queries: list[Query]
    qrels: dict[str, set[str]]

    @property
    def answerable(self) -> list[Query]:
        """Return the queries that have at least one gold span."""
        return [q for q in self.queries if q.is_answerable]

    @property
    def null_queries(self) -> list[Query]:
        """Return the queries the dataset marks unanswerable."""
        return [q for q in self.queries if not q.is_answerable]

    def chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        """Return the chunks belonging to one document."""
        return [c for c in self.chunks if c.doc_id == doc_id]

    # The CUAD-facing name for the same lookup.
    chunks_for_contract = chunks_for_doc

    def summary(self) -> dict[str, int | float]:
        """Return the slice's shape: documents, chunks, queries, and null rate."""
        answerable = self.answerable
        judged = sum(len(self.qrels[q.query_id]) for q in answerable)
        return {
            "contracts": len({c.doc_id for c in self.chunks}),
            "chunks": len(self.chunks),
            "queries": len(self.queries),
            "answerable_queries": len(answerable),
            "null_queries": len(self.null_queries),
            "null_rate": round(len(self.null_queries) / max(len(self.queries), 1), 4),
            "mean_gold_chunks": round(judged / max(len(answerable), 1), 2),
        }


def build_slice(
    path: Path,
    *,
    max_contracts: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> Slice:
    """Load CUAD, chunk it, and derive relevance judgments from answer spans.

    Parameters
    ----------
    path : Path
        The CUAD ``test.json``.
    max_contracts : int or None, optional
        Take the first N contracts in file order, which is stable across runs,
        so a smaller slice stays a subset of a larger one. ``None`` takes all.
    chunk_chars : int, optional
        Window width passed to ``chunk_contract``.
    overlap_chars : int, optional
        Window overlap passed to ``chunk_contract``.

    Returns
    -------
    Slice
        Chunks, queries, and judgments.

    Raises
    ------
    ValueError
        If an answerable query has no covering chunk, which means chunking
        dropped part of a contract.
    """
    chunks: list[Chunk] = []
    queries: list[Query] = []
    qrels: dict[str, set[str]] = {}

    for count, (doc_id, text, doc_queries) in enumerate(load_cuad(path)):
        if max_contracts is not None and count >= max_contracts:
            break
        doc_chunks = chunk_contract(
            doc_id,
            text,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
        chunks.extend(doc_chunks)
        for query in doc_queries:
            queries.append(query)
            qrels[query.query_id] = relevant_chunk_ids(doc_chunks, query.answers)

    # A gold span that no chunk covers would silently become an unreachable
    # query and depress every retriever equally. Overlapping windows cover the
    # whole contract, so this should be empty; assert it rather than assume it.
    unreachable = [
        q.query_id for q in queries if q.is_answerable and not qrels[q.query_id]
    ]
    if unreachable:
        raise ValueError(
            f"{len(unreachable)} answerable queries have no covering chunk, "
            f"e.g. {unreachable[:3]}. Chunking dropped part of the contract.",
        )

    return Slice(chunks=chunks, queries=queries, qrels=qrels)
