"""Filing chunks and page-level judgments, on the committed two-page fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aieng.advanced_rag.corpora.financebench import FinanceQuery
from aieng.advanced_rag.retrieval.chunkers import Item
from aieng.advanced_rag.retrieval.filings import (
    build_filing_slice,
    chunk_filing,
    chunk_pages,
    item_pages,
    load_document,
    relevant_chunk_ids,
)


FIXTURE = Path(__file__).parents[1] / "ingestion" / "fixtures" / "acme-10k.docling.json"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    directory = tmp_path / "ACME"
    directory.mkdir()
    shutil.copy(FIXTURE, directory / "document.json")
    return directory


def query(query_id: str, pages: tuple[int, ...], doc: str = "ACME") -> FinanceQuery:
    return FinanceQuery(
        query_id=query_id,
        question="What was Instruments revenue?",
        doc_id=doc,
        category="metrics-generated",
        reasoning="",
        answer="4,812",
        gold_pages=pages,
        evidence_texts=("evidence",),
    )


def item(start: int, end: int, page: int | None) -> Item:
    return Item(kind="text", text="x" * (end - start), start=start, end=end, page=page)


def test_docling_page_one_is_gold_page_zero():
    assert item_pages([item(0, 5, 1), item(5, 9, 2)]) == [0, 1]


def test_an_item_without_a_page_follows_the_one_before_it():
    assert item_pages([item(0, 5, 3), item(5, 9, None)]) == [2, 2]
    assert item_pages([item(0, 5, None)]) == [0]


def test_a_chunk_covers_every_page_it_overlaps():
    from aieng.advanced_rag.retrieval.corpus import Chunk

    items = [item(0, 10, 1), item(12, 20, 2)]
    chunks = [
        Chunk("d::0", "d", "a", 0, 10),
        Chunk("d::1", "d", "b", 5, 15),
        Chunk("d::2", "d", "c", 20, 22),
    ]
    covered = chunk_pages(chunks, items)
    assert covered["d::0"] == frozenset({0})
    assert covered["d::1"] == frozenset({0, 1})
    assert covered["d::2"] == frozenset({1})


def test_table_aware_keeps_a_table_with_its_header_on_one_page(folder):
    filing = chunk_filing("ACME", load_document(folder), strategy="table_aware")
    [revenue] = [c for c in filing.chunks if "4,812" in c.text and "|" in c.text]
    assert "December 31, 2022" in revenue.text
    assert filing.pages[revenue.chunk_id] == frozenset({0})


def test_a_plain_window_that_spans_two_pages_covers_both(folder):
    whole = chunk_filing("ACME", load_document(folder), strategy="plain")
    assert len(whole.chunks) == 1
    assert whole.pages[whole.chunks[0].chunk_id] == frozenset({0, 1})


def test_an_unknown_strategy_is_rejected(folder):
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_filing("ACME", load_document(folder), strategy="semantic")  # type: ignore[arg-type]


def test_relevant_chunks_are_those_on_a_gold_page(folder):
    filing = chunk_filing("ACME", load_document(folder), strategy="table_aware")
    on_second = relevant_chunk_ids(filing.pages, [1])
    assert on_second == {
        c.chunk_id for c in filing.chunks if filing.pages[c.chunk_id] == {1}
    }
    assert relevant_chunk_ids(filing.pages, [0, 1]) == {
        c.chunk_id for c in filing.chunks
    }
    assert relevant_chunk_ids(filing.pages, [7]) == set()


def test_the_slice_judges_by_gold_page_and_lists_what_it_cannot(folder):
    queries = [
        query("q-page-0", (0,)),
        query("q-both", (0, 1)),
        query("q-blank-page", (9,)),
        query("q-no-filing", (0,), doc="MISSING"),
    ]
    sliced = build_filing_slice(queries, {"ACME": folder, "OTHER": folder})
    assert [q.query_id for q in sliced.queries] == ["q-page-0", "q-both"]
    assert sliced.uncovered == ["q-blank-page", "q-no-filing"]
    assert set(sliced.filings) == {"ACME"}, "a filing nobody asks about is not chunked"
    assert sliced.qrels["q-page-0"] < sliced.qrels["q-both"]
    assert {c.chunk_id for c in sliced.chunks} == {
        c.chunk_id for c in sliced.filings["ACME"].chunks
    }


def test_both_strategies_judge_the_same_pages(folder):
    for strategy in ("plain", "table_aware"):
        sliced = build_filing_slice(
            [query("q", (1,))], {"ACME": folder}, strategy=strategy, max_chars=300
        )
        for chunk_id in sliced.qrels["q"]:
            assert 1 in sliced.filings["ACME"].pages[chunk_id]


def test_as_query_carries_the_scope_and_question_only():
    asked = query("q", (0,)).as_query()
    assert (asked.query_id, asked.doc_id, asked.question) == (
        "q",
        "ACME",
        "What was Instruments revenue?",
    )
    assert not asked.is_answerable


def test_table_aware_chunks_keep_every_numeric_cell_attached_where_windows_do_not(
    folder,
):
    from aieng.advanced_rag.retrieval.chunkers import all_facts
    from aieng.advanced_rag.retrieval.tables import attachment_report

    document = load_document(folder)
    rates = {}
    for strategy in ("plain", "table_aware"):
        filing = chunk_filing("ACME", document, strategy=strategy, max_chars=120)
        rates[strategy] = attachment_report(filing.chunks, all_facts(filing.items))
    assert rates["table_aware"]["attachment_rate"] == 1.0
    assert rates["table_aware"]["cell_recoverable_rate"] == 1.0
    assert rates["plain"]["attachment_rate"] < rates["table_aware"]["attachment_rate"]
