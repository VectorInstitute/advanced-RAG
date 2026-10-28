"""The splitters: offsets, budgets, coverage, and the table-aware chunker."""

from __future__ import annotations

import pytest

from aieng.advanced_rag.retrieval.chunkers import (
    DEFAULT_SEPARATORS,
    Item,
    all_facts,
    fixed_chars,
    recursive_tokens,
    semantic,
    sentence_spans,
    table_aware_chunks,
    timed,
)
from aieng.advanced_rag.retrieval.corpus import Chunk, chunk_contract
from aieng.advanced_rag.retrieval.tables import attachment_report
from aieng.advanced_rag.retrieval.tokens import WhitespaceTokenIndex


TEXT = (
    "Section 1. This agreement is between Acme and Beta. It takes effect today.\n\n"
    "Section 2. Either party may terminate on thirty days notice. "
    "Notice must be written.\n\n"
    "Section 3. " + "Boilerplate follows. " * 40
)


def test_fixed_chars_is_the_slice_builder_s_chunker() -> None:
    # The control has to be the same control. Every published number for this
    # project came from chunk_contract at 1200/200, and the chunking experiment
    # kept a second copy that had to agree with it.
    assert fixed_chars is chunk_contract
    chunks = fixed_chars("doc", TEXT, chunk_chars=200, overlap_chars=40)
    assert [(c.start, c.end) for c in chunks] == [
        (c.start, c.end)
        for c in chunk_contract("doc", TEXT, chunk_chars=200, overlap_chars=40)
    ]


@pytest.mark.parametrize("overlap", [0, 200])
def test_fixed_chars_rejects_impossible_settings(overlap: int) -> None:
    with pytest.raises(ValueError):
        fixed_chars("doc", TEXT, chunk_chars=200, overlap_chars=overlap + 200)


def test_chunk_text_always_matches_its_offsets() -> None:
    counter = WhitespaceTokenIndex(TEXT)
    for chunk in recursive_tokens(
        "doc", TEXT, counter, chunk_tokens=25, overlap_tokens=5
    ):
        assert chunk.text == TEXT[chunk.start : chunk.end]


def test_recursive_respects_the_token_budget() -> None:
    counter = WhitespaceTokenIndex(TEXT)
    chunks = recursive_tokens("doc", TEXT, counter, chunk_tokens=25, overlap_tokens=5)
    assert chunks
    assert max(counter.count(c.start, c.end) for c in chunks) <= 25


def test_recursive_rejects_impossible_settings() -> None:
    counter = WhitespaceTokenIndex(TEXT)
    with pytest.raises(ValueError):
        recursive_tokens("doc", TEXT, counter, chunk_tokens=0)
    with pytest.raises(ValueError):
        recursive_tokens("doc", TEXT, counter, chunk_tokens=10, overlap_tokens=10)


def test_recursive_covers_every_word() -> None:
    # A gold span that no chunk covers becomes an unreachable query, so coverage
    # is a correctness property rather than a nicety.
    counter = WhitespaceTokenIndex(TEXT)
    chunks = recursive_tokens("doc", TEXT, counter, chunk_tokens=25, overlap_tokens=5)
    covered = bytearray(len(TEXT))
    for chunk in chunks:
        for position in range(chunk.start, chunk.end):
            covered[position] = 1
    missed = [i for i, ch in enumerate(TEXT) if not ch.isspace() and not covered[i]]
    assert missed == []


def test_recursive_overlap_repeats_the_previous_tail() -> None:
    # Uniform sentences, so every chunk holds several pieces and every boundary
    # can carry overlap. Overlap is only possible between chunks made of more
    # than one piece; a paragraph that fills a chunk on its own has no tail to
    # repeat, which is why this uses its own text rather than TEXT.
    text = "The quick brown fox jumped. " * 40
    counter = WhitespaceTokenIndex(text)
    with_overlap = recursive_tokens(
        "doc", text, counter, chunk_tokens=25, overlap_tokens=10
    )
    without = recursive_tokens("doc", text, counter, chunk_tokens=25, overlap_tokens=0)
    assert len(with_overlap) > len(without)
    assert all(
        a.start < b.start < a.end for a, b in zip(with_overlap, with_overlap[1:])
    )
    assert all(b.end > a.end for a, b in zip(with_overlap, with_overlap[1:]))


def test_recursive_prefers_paragraph_boundaries() -> None:
    counter = WhitespaceTokenIndex(TEXT)
    chunks = recursive_tokens(
        "doc",
        TEXT,
        counter,
        chunk_tokens=20,
        overlap_tokens=0,
        separators=DEFAULT_SEPARATORS,
    )
    # The first paragraph fits inside the budget, so it must not be cut in half.
    assert chunks[0].text.startswith("Section 1.")
    assert "Section 2." not in chunks[0].text


def test_recursive_handles_a_word_longer_than_the_budget() -> None:
    text = "x" * 500
    counter = WhitespaceTokenIndex(text)
    chunks = recursive_tokens("doc", text, counter, chunk_tokens=1, overlap_tokens=0)
    assert "".join(c.text for c in chunks) == text


def test_sentence_spans_tile_the_document() -> None:
    spans = sentence_spans(TEXT)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(TEXT)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_semantic_cuts_where_the_topic_changes() -> None:
    # A fake encoder: sentences about one topic share a vector, so the only
    # large distance is at the topic boundary.
    first = "Alpha runs. " * 6
    second = "Beta stops. " * 6
    text = first + second

    def encode(texts):
        return [[1.0, 0.0] if "Alpha" in t else [0.0, 1.0] for t in texts]

    chunks = semantic(
        "doc", text, encode, buffer_size=0, breakpoint_percentile=90, min_chars=10
    )
    assert len(chunks) == 2
    assert chunks[0].text.strip().endswith("Alpha runs.")
    assert chunks[1].text.strip().startswith("Beta stops.")
    assert "".join(c.text for c in chunks) == text


def test_semantic_caps_chunk_size() -> None:
    text = "Same thing. " * 200

    def encode(texts):
        return [[1.0, 0.0] for _ in texts]

    chunks = semantic("doc", text, encode, breakpoint_percentile=95, max_chars=300)
    assert chunks
    assert max(len(c.text) for c in chunks) < 320


def test_timed_records_the_strategy_and_its_settings() -> None:
    result = timed(
        lambda: fixed_chars("doc", TEXT, chunk_chars=200, overlap_chars=40),
        "fixed_chars",
        chunk_chars=200,
    )
    assert result.strategy == "fixed_chars"
    assert result.params == {"chunk_chars": 200}
    assert result.chunks
    assert result.seconds >= 0.0


BIG_ROWS = [["", "December 31, 2022"], ["", "(In millions)"]] + [
    [f"Segment {i}", f"{i},043"] for i in range(1, 41)
]


def _items(make_grid) -> list[Item]:
    grid = make_grid(BIG_ROWS)
    prose = "Revenue by segment is set out below. " * 5
    table_text = grid.to_markdown()
    return [
        Item("heading", "Note 3. Segment results", 0, 23, page=7),
        Item("text", prose, 25, 25 + len(prose), page=7),
        Item(
            "table",
            table_text,
            30 + len(prose),
            30 + len(prose) + len(table_text),
            page=7,
            grid=grid,
            heading="Note 3. Segment results",
        ),
    ]


def test_a_small_table_stays_in_one_chunk(make_grid) -> None:
    grid = make_grid([["", "2022"], ["Instruments", "4,812"]])
    items = [Item("table", grid.to_markdown(), 0, 40, grid=grid, heading="Note 3")]
    chunks = table_aware_chunks("f", items, max_chars=1200)
    assert len(chunks) == 1
    assert "4,812" in chunks[0].text and "2022" in chunks[0].text
    # The section heading rides along, so the chunk says which note it is.
    assert chunks[0].text.startswith("Note 3")


def test_a_large_table_is_split_by_rows_with_its_headers_repeated(make_grid) -> None:
    chunks = table_aware_chunks("f", _items(make_grid), max_chars=600)
    table_chunks = [c for c in chunks if "|" in c.text]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        assert "December 31, 2022" in chunk.text
        assert "(In millions)" in chunk.text


def test_prose_and_tables_do_not_share_a_chunk(make_grid) -> None:
    chunks = table_aware_chunks("f", _items(make_grid), max_chars=600)
    mixed = [c for c in chunks if "|" in c.text and "Revenue by segment" in c.text]
    assert mixed == []


def test_table_aware_beats_fixed_windows_on_attachment(make_grid) -> None:
    items = _items(make_grid)
    facts = items[2].grid.facts()
    aware = table_aware_chunks("f", items, max_chars=600)

    document = "\n\n".join(i.text for i in items)
    naive = fixed_chars("f", document, chunk_chars=600, overlap_chars=100)

    aware_report = attachment_report(aware, facts)
    naive_report = attachment_report(naive, facts)
    assert aware_report["attachment_rate"] == 1.0
    assert naive_report["attachment_rate"] < 0.5
    assert naive_report["cells_never_complete"] > 0


def test_every_chunk_is_within_budget_or_is_one_indivisible_row(make_grid) -> None:
    chunks = table_aware_chunks("f", _items(make_grid), max_chars=600)
    oversized = [c for c in chunks if len(c.text) > 600]
    # Only a single table row plus its repeated headers may exceed the budget.
    for chunk in oversized:
        assert chunk.text.count("\n") <= len(make_grid(BIG_ROWS).header_rows) + 1


def test_facts_are_scoped_to_their_own_table(make_grid) -> None:
    # The same number and row label can appear in another table of the same
    # filing. A chunker is judged on the table it split, not on that one.
    grid = make_grid([["", "2022"], ["Instruments", "4,812"]])
    items = [
        Item("table", grid.to_markdown(), 0, 40, grid=grid, heading="Note 3"),
        Item("text", "| Instruments | 4,812 |", 45, 70),
    ]
    facts = all_facts(items)
    assert all(f.table_span == (0, 40) for f in facts)

    elsewhere = Chunk("f::9", "f", "| Instruments | 4,812 |", 45, 70)
    report = attachment_report([elsewhere], facts)
    assert report["chunk_cell_pairs"] == 0
