"""Offset-preserving chunking and span-to-judgment derivation.

Both corpora that survive licence review carry gold answers as character
offsets into a long document. Chunk on characters and a chunk is relevant
exactly when it overlaps a gold span, so retrieval judgments come from the
dataset instead of from a model.

It generalises ``chunk_contract`` and ``relevant_chunk_ids`` in
``aieng.advanced_rag.retrieval.corpus`` off CUAD's contract-only naming.
"""

from __future__ import annotations

from collections.abc import Iterable

from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query


def chunk_document(
    doc_id: str,
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split a document into overlapping fixed-width character windows.

    Fixed-width windows on characters, rather than tokens or sentences, keep the
    offset arithmetic exact, which is what lets relevance be computed without
    fuzzy text matching. Chunking strategy is itself a variable later in the
    bootcamp; this is the plain baseline those variants move away from.

    Parameters
    ----------
    doc_id : str
        Stable document identifier. Chunk ids are derived from it.
    text : str
        Full document text.
    chunk_chars : int, optional
        Window width in characters.
    overlap_chars : int, optional
        Overlap between consecutive windows, in characters.

    Returns
    -------
    list of Chunk
        Windows in document order, skipping any that are entirely whitespace.

    Raises
    ------
    ValueError
        If ``chunk_chars`` is not positive, or ``overlap_chars`` falls outside
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
                )
            )
        if end >= len(text):
            break
    return chunks


def relevant_chunk_ids(
    chunks: Iterable[Chunk],
    answers: Iterable[AnswerSpan],
) -> set[str]:
    """Return ids of chunks whose window overlaps any gold answer span.

    Overlap rather than containment: gold spans can run longer than a window,
    and requiring full containment would mark a chunk holding most of the
    evidence as irrelevant.
    """
    windows = list(chunks)
    relevant: set[str] = set()
    for answer in answers:
        for chunk in windows:
            if chunk.start < answer.end and answer.start < chunk.end:
                relevant.add(chunk.chunk_id)
    return relevant


def check_reachable(queries: Iterable[Query], qrels: dict[str, set[str]]) -> None:
    """Raise if an answerable query has no covering chunk.

    A gold span that no chunk covers becomes an unreachable query that depresses
    every retriever equally and hides a chunking bug. Overlapping windows cover
    the whole document, so this should never fire; assert it rather than assume
    it.

    Raises
    ------
    ValueError
        If any answerable query has an empty judgment set.
    """
    unreachable = [
        query.query_id
        for query in queries
        if query.is_answerable and not qrels.get(query.query_id)
    ]
    if unreachable:
        raise ValueError(
            f"{len(unreachable)} answerable queries have no covering chunk, "
            f"e.g. {unreachable[:3]}. Chunking dropped part of the document."
        )
