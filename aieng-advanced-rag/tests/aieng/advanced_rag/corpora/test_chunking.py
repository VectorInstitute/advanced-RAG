"""Chunk-offset arithmetic and the span-to-judgment rule."""

from __future__ import annotations

import pytest


pytest.importorskip(
    "aieng.advanced_rag.retrieval.corpus",
    reason="corpus types have not landed in aieng.advanced_rag.retrieval yet",
)

from aieng.advanced_rag.corpora.chunking import (  # noqa: E402
    check_reachable,
    chunk_document,
    relevant_chunk_ids,
)
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Query  # noqa: E402


def test_windows_cover_the_whole_document() -> None:
    text = "x" * 3000
    chunks = chunk_document("doc", text, chunk_chars=1000, overlap_chars=200)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start < earlier.end, "a gap would make a span unreachable"


def test_chunk_text_matches_its_offsets() -> None:
    text = "".join(f"{index:04d}" for index in range(500))
    for chunk in chunk_document("doc", text, chunk_chars=300, overlap_chars=50):
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.span == (chunk.start, chunk.end)


def test_chunk_ids_carry_the_document_id() -> None:
    chunks = chunk_document("some-doc", "y" * 500, chunk_chars=200, overlap_chars=0)
    assert [c.contract_id for c in chunks] == ["some-doc"] * len(chunks)
    assert chunks[0].chunk_id == "some-doc::0000"


def test_whitespace_only_windows_are_dropped() -> None:
    text = "start" + " " * 400 + "end"
    chunks = chunk_document("doc", text, chunk_chars=100, overlap_chars=0)
    assert all(chunk.text.strip() for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_chars", "overlap_chars"),
    [(0, 0), (-1, 0), (100, 100), (100, 200), (100, -1)],
)
def test_invalid_windows_are_rejected(chunk_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(
            "doc", "text", chunk_chars=chunk_chars, overlap_chars=overlap_chars
        )


def test_relevance_is_overlap_not_containment() -> None:
    chunks = chunk_document("doc", "z" * 3000, chunk_chars=1000, overlap_chars=0)
    # A span longer than one window would match nothing under containment.
    span = AnswerSpan(text="z" * 1500, start=500, end=2000)
    assert relevant_chunk_ids(chunks, [span]) == {"doc::0000", "doc::0001"}


def test_touching_boundaries_do_not_count_as_overlap() -> None:
    chunks = chunk_document("doc", "z" * 2000, chunk_chars=1000, overlap_chars=0)
    span = AnswerSpan(text="z" * 10, start=1000, end=1010)
    assert relevant_chunk_ids(chunks, [span]) == {"doc::0001"}


def test_no_answers_means_no_judgments() -> None:
    chunks = chunk_document("doc", "z" * 500, chunk_chars=100, overlap_chars=0)
    assert relevant_chunk_ids(chunks, []) == set()


def test_check_reachable_flags_an_uncovered_gold_span() -> None:
    query = Query(
        query_id="q1",
        doc_id="doc",
        category="c",
        question="q",
        answers=(AnswerSpan(text="a", start=0, end=1),),
    )
    with pytest.raises(ValueError, match="no covering chunk"):
        check_reachable([query], {"q1": set()})


def test_check_reachable_ignores_null_queries() -> None:
    query = Query(query_id="q1", doc_id="doc", category="c", question="q")
    check_reachable([query], {"q1": set()})
