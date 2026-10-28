"""Turn every arm into a page ranking and a reader input, and run the reader.

Eight arms and four oracle rows, all over one page universe:

====== ============================================= ==============================
Arm    Retrieval                                     Reader input
====== ============================================= ==============================
T      Dense ``bge-small-en-v1.5`` over Docling text  Top 5 chunks
E1     Dense over text plus generic descriptions     Top 5 chunks
E2     Dense over text plus domain descriptions      Top 5 chunks
       and table summaries
L      ColBERT over E2's text                         Top 5 chunks
VT     ColModernVBERT page images                    Top 3 pages' E2 text
VV     ColModernVBERT page images                    Top 3 page images
TV     E2's ranking, collapsed to pages              Top 3 page images
H      Reciprocal rank fusion of E2 and VV pages     Top 3 pages, image and E2 text
====== ============================================= ==============================

The oracle rows give the reader the gold pages (or, for a null, the page it
was rewritten from) as text, as enriched text, as images, and as both. The gap
between an arm and its oracle is retrieval loss; the gap between oracles is
reading loss.

One reader model answers every row with the same instruction, so only the
evidence changes between rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from aieng.advanced_rag.retrieval.corpus import Chunk, Query

from .enrich import UNSCOPED, parallel_map
from .pages import Corpus, data_uri, render_page
from .questions import answer_from
from .vlm import ProxyModels


ARMS: tuple[str, ...] = ("T", "E1", "E2", "L", "VT", "VV", "TV", "H")
ORACLES: tuple[str, ...] = ("O-text", "O-enriched", "O-image", "O-both")
ROWS: tuple[str, ...] = ARMS + ORACLES

#: What each row's reader sees, for tables and labels.
DESCRIPTIONS: Mapping[str, str] = {
    "T": "text chunks, Docling text layer",
    "E1": "text chunks, generic picture descriptions",
    "E2": "text chunks, domain descriptions and table summaries",
    "L": "text chunks, ColBERT over E2 text",
    "VT": "page retrieval, E2 page text",
    "VV": "page retrieval, page images",
    "TV": "E2 retrieval, page images",
    "H": "fused retrieval, page images and E2 text",
    "O-text": "gold page, Docling text",
    "O-enriched": "gold page, E2 text",
    "O-image": "gold page, image",
    "O-both": "gold page, image and E2 text",
}

CHUNKS_TO_READER = 5
PAGES_TO_READER = 3
RRF_K = 60


def collapse_to_pages(chunk_ids: Sequence[str], first_page: Mapping[str, str], k: int = 10) -> list[str]:
    """Return the pages of a chunk ranking, in order of first occurrence."""
    pages: list[str] = []
    for chunk_id in chunk_ids:
        page = first_page[chunk_id]
        if page not in pages:
            pages.append(page)
        if len(pages) == k:
            break
    return pages


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K, depth: int = 10) -> list[str]:
    """Fuse page rankings by reciprocal rank, ties broken on page identifier."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, page in enumerate(ranking, start=1):
            scores[page] = scores.get(page, 0.0) + 1.0 / (k + rank)
    return [p for p, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))][:depth]


class DenseChunks:
    """``bge-small-en-v1.5`` over one chunk set, searched across the whole universe."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Any) -> None:
        from aieng.advanced_rag.retrieval.index import DenseIndex  # noqa: PLC0415

        self.embedder = embedder
        vectors = np.asarray(embedder.encode([c.indexed_text for c in chunks]), dtype=np.float32)
        self.index = DenseIndex([c.chunk_id for c in chunks], vectors)

    def search(self, questions: Sequence[str], k: int = 50) -> list[list[str]]:
        """Return the top ``k`` chunk identifiers for each question."""
        encode = getattr(self.embedder, "encode_queries", self.embedder.encode)
        vectors = np.asarray(encode(list(questions)), dtype=np.float32)
        found: list[list[str]] = self.index.search(vectors, None, k)
        return found


class ColbertChunks:
    """ColBERT late interaction over one chunk set, across the whole universe."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from aieng.advanced_rag.retrieval.representations import ColbertRetriever  # noqa: PLC0415

        self.retriever = ColbertRetriever()
        self.retriever.index(list(chunks))

    def search(self, questions: Sequence[str], k: int = 50) -> list[list[str]]:
        """Return the top ``k`` chunk identifiers for each question."""
        return [
            [hit.chunk_id for hit in self.retriever.search(Query(f"q{i}", UNSCOPED, "", q), k)]
            for i, q in enumerate(questions)
        ]


class BM25Chunks:
    """BM25 over one chunk set, across the whole universe, for null verification."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from aieng.advanced_rag.retrieval.sparse import BM25Retriever  # noqa: PLC0415

        self.retriever = BM25Retriever()
        self.retriever.index(list(chunks))

    def search(self, questions: Sequence[str], k: int = 50) -> list[list[str]]:
        """Return the top ``k`` chunk identifiers for each question."""
        return [
            [hit.chunk_id for hit in self.retriever.search(Query(f"q{i}", UNSCOPED, "", q), k)]
            for i, q in enumerate(questions)
        ]


@dataclass(frozen=True)
class Evidence:
    """What one row's reader sees for one question.

    Parameters
    ----------
    text : str or None
        Text evidence, or None.
    pages : list of str
        Pages whose images are sent, in order.
    used_pages : list of str
        Every page the evidence came from.
    """

    text: str | None
    pages: list[str]
    used_pages: list[str]


@dataclass
class ArmInputs:
    """Everything the readers need, by arm.

    Parameters
    ----------
    chunk_text : dict
        Chunk set name (``T``, ``E1``, ``E2``) to chunk identifier to text.
    chunk_page : dict
        Chunk set name to chunk identifier to first page.
    chunk_rankings : dict
        Arm (``T``, ``E1``, ``E2``, ``L``) to question identifier to ranked
        chunk identifiers.
    page_rankings : dict
        Every arm to question identifier to ranked pages.
    page_text : callable
        ``(page, enriched)`` to the page's text, Docling's alone or with E2.
    """

    chunk_text: dict[str, dict[str, str]]
    chunk_page: dict[str, dict[str, str]]
    chunk_rankings: dict[str, dict[str, list[str]]]
    page_rankings: dict[str, dict[str, list[str]]]
    page_text: Callable[[str, bool], str]


def _chunk_set(arm: str) -> str:
    return "E2" if arm == "L" else arm


def evidence_for(row: str, question: Mapping[str, Any], inputs: ArmInputs) -> Evidence:
    """Return what one row's reader sees for one question."""
    qid = str(question["query_id"])
    if row in ("T", "E1", "E2", "L"):
        source = _chunk_set(row)
        top = inputs.chunk_rankings[row][qid][:CHUNKS_TO_READER]
        pages = [inputs.chunk_page[source][c] for c in top]
        blocks = [f"[{i}] ({pages[i - 1]})\n{inputs.chunk_text[source][c]}" for i, c in enumerate(top, start=1)]
        return Evidence("\n\n".join(blocks), [], sorted(set(pages)))
    if row in ("VT", "VV", "TV", "H"):
        top = inputs.page_rankings[row][qid][:PAGES_TO_READER]
        if row == "VT":
            return Evidence(_page_blocks(top, inputs, enriched=True), [], top)
        if row == "H":
            return Evidence(_page_blocks(top, inputs, enriched=True), top, top)
        return Evidence(None, top, top)
    if row not in ORACLES:
        raise KeyError(row)
    gold = list(question.get("gold_pages") or []) or [question.get("rewritten_from")]
    gold = [g for g in gold if g][:PAGES_TO_READER]
    text = None if row == "O-image" else _page_blocks(gold, inputs, enriched=row != "O-text")
    images = gold if row in ("O-image", "O-both") else []
    return Evidence(text, images, gold)


def _page_blocks(pages: Sequence[str], inputs: ArmInputs, *, enriched: bool) -> str:
    blocks = []
    for index, page in enumerate(pages, start=1):
        text = inputs.page_text(page, enriched) or "(no text on this page)"
        blocks.append(f"[{index}] ({page})\n{text}")
    return "\n\n".join(blocks)


def run_reader(
    llm: ProxyModels,
    *,
    model: str,
    corpus: Corpus,
    row: str,
    questions: Sequence[Mapping[str, Any]],
    inputs: ArmInputs,
    workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Answer every question for one row and record what each answer cost.

    Returns
    -------
    dict
        Question identifier to ``{"answer", "pages", "images", "prompt_tokens",
        "completion_tokens", "seconds"}``.
    """

    @lru_cache(maxsize=256)
    def image(page: str) -> str:
        return data_uri(render_page(corpus, page))

    def one(question: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        evidence = evidence_for(row, question, inputs)
        images = [image(p) for p in evidence.pages]
        text = evidence.text
        if images:
            listing = ", ".join(f"image {i} is {p}" for i, p in enumerate(evidence.pages, start=1))
            text = f"({listing})\n\n{text}" if text else f"({listing})"
        prompt = _reader_prompt(str(question["question"]), text, bool(images), evidence.text is not None)
        reply = llm.complete(model, prompt, images, max_tokens=200)
        return str(question["query_id"]), {
            "answer": reply.text.strip(),
            "pages": evidence.used_pages,
            "images": len(images),
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "seconds": round(reply.seconds, 2),
        }

    return dict(parallel_map(one, list(questions), workers))


def _reader_prompt(question: str, text: str | None, with_images: bool, with_text: bool) -> str:
    # The same template the question checker uses, so a reader and the checker
    # differ only in their evidence.
    from .questions import ANSWER_PROMPT  # noqa: PLC0415

    where = ("in the images and the text below" if with_text else "in the images") if with_images else "below"
    prompt = ANSWER_PROMPT.format(where=where, question=question)
    return f"Evidence:\n{text}\n\n{prompt}" if text else prompt


__all__ = [
    "ARMS",
    "CHUNKS_TO_READER",
    "DESCRIPTIONS",
    "ORACLES",
    "PAGES_TO_READER",
    "ROWS",
    "ArmInputs",
    "BM25Chunks",
    "ColbertChunks",
    "DenseChunks",
    "Evidence",
    "answer_from",
    "collapse_to_pages",
    "evidence_for",
    "reciprocal_rank_fusion",
    "run_reader",
]
