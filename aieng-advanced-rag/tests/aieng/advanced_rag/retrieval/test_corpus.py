"""Chunking, the chunk record, and the span-to-judgment derivation.

The judgments are the part of this harness that could be quietly wrong and
still produce believable numbers, so the offset arithmetic is tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieng.advanced_rag.retrieval.corpus import (
    AnswerSpan,
    Chunk,
    build_slice,
    chunk_contract,
    chunk_stats,
    parse_category,
    relevant_chunk_ids,
)


class TestChunking:
    def test_windows_cover_the_whole_text(self):
        text = "x" * 3000
        chunks = chunk_contract("c", text, chunk_chars=1000, overlap_chars=200)
        assert chunks[0].start == 0
        assert chunks[-1].end == len(text)
        # No gaps: each window starts at or before the previous one's end.
        for previous, current in zip(chunks, chunks[1:]):
            assert current.start <= previous.end

    def test_overlap_is_applied(self):
        chunks = chunk_contract("c", "y" * 3000, chunk_chars=1000, overlap_chars=200)
        assert chunks[1].start == 800

    def test_offsets_slice_back_to_the_chunk_text(self):
        text = "".join(f"clause {i} text. " for i in range(400))
        for chunk in chunk_contract("c", text, chunk_chars=500, overlap_chars=100):
            assert text[chunk.start : chunk.end] == chunk.text

    def test_shorter_than_one_window(self):
        chunks = chunk_contract(
            "c", "short contract", chunk_chars=1000, overlap_chars=0
        )
        assert len(chunks) == 1
        assert chunks[0].text == "short contract"

    def test_blank_text_yields_no_chunks(self):
        assert chunk_contract("c", "   ") == []

    def test_ids_are_unique_and_ordered(self):
        chunks = chunk_contract("c", "z" * 5000, chunk_chars=1000, overlap_chars=200)
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids)

    @pytest.mark.parametrize("overlap", [-1, 1000, 2000])
    def test_invalid_overlap_rejected(self, overlap):
        with pytest.raises(ValueError):
            chunk_contract("c", "text", chunk_chars=1000, overlap_chars=overlap)


class TestChunkRecord:
    def test_indexed_text_defaults_to_the_span(self):
        chunk = Chunk("d::0", "d", "the clause", 0, 10)
        assert chunk.indexed_text == "the clause"

    def test_contextualising_keeps_the_span_intact(self):
        chunk = Chunk("d::0", "d", "the clause", 0, 10)
        contextual = chunk.with_embed_text("From the Acme agreement.\n\nthe clause")
        assert contextual.indexed_text.startswith("From the Acme agreement.")
        # The span is the unit of judgment and must survive contextualisation.
        assert (contextual.start, contextual.end, contextual.text) == (
            0,
            10,
            "the clause",
        )

    def test_contract_id_is_the_document_id(self):
        # ``contract_id`` is an alias of ``doc_id``; both names read it.
        chunk = Chunk("d::0", "d", "the clause", 0, 10)
        assert chunk.contract_id == chunk.doc_id == "d"
        assert chunk.span == (0, 10)


class TestRelevanceDerivation:
    def setup_method(self):
        self.chunks = chunk_contract(
            "c", "w" * 3000, chunk_chars=1000, overlap_chars=200
        )

    def test_span_inside_one_window_marks_it(self):
        # Characters 100-150 fall only in the first window (0-1000).
        relevant = relevant_chunk_ids(self.chunks, (AnswerSpan("w" * 50, 100, 150),))
        assert relevant == {"c::0000"}

    def test_span_in_the_overlap_marks_both_windows(self):
        # Window 0 is [0,1000), window 1 is [800,1800). 850-900 is in both.
        relevant = relevant_chunk_ids(self.chunks, (AnswerSpan("w" * 50, 850, 900),))
        assert relevant == {"c::0000", "c::0001"}

    def test_span_longer_than_a_window_marks_every_window_it_touches(self):
        relevant = relevant_chunk_ids(self.chunks, (AnswerSpan("w" * 2000, 100, 2100),))
        assert len(relevant) >= 3

    def test_multiple_answers_union(self):
        answers = (AnswerSpan("a", 10, 20), AnswerSpan("b", 2500, 2510))
        relevant = relevant_chunk_ids(self.chunks, answers)
        assert "c::0000" in relevant
        assert len(relevant) >= 2

    def test_no_answers_means_no_relevant_chunks(self):
        assert relevant_chunk_ids(self.chunks, ()) == set()

    def test_boundary_span_touching_only_the_second_window(self):
        # A span starting exactly at 1000 is past window 0's exclusive end.
        relevant = relevant_chunk_ids(self.chunks, (AnswerSpan("w" * 10, 1000, 1010),))
        assert "c::0000" not in relevant

    def test_plain_offset_pairs_work_the_same_way(self):
        # The chunking experiment passed (start, end) pairs; the merged function
        # takes either, and the two forms have to agree.
        spans = relevant_chunk_ids(self.chunks, [(850, 900)])
        objects = relevant_chunk_ids(self.chunks, (AnswerSpan("w" * 50, 850, 900),))
        assert spans == objects

    def test_relevance_is_overlap_not_containment(self):
        chunks = [Chunk("d::0", "d", "a", 0, 100), Chunk("d::1", "d", "b", 100, 200)]
        # A span that starts inside the first chunk and runs past its end is
        # evidence in both.
        assert relevant_chunk_ids(chunks, [(90, 150)]) == {"d::0", "d::1"}
        assert relevant_chunk_ids(chunks, [(0, 1)]) == {"d::0"}
        # Half-open: a span touching the boundary belongs to the chunk after it.
        assert relevant_chunk_ids(chunks, [(100, 101)]) == {"d::1"}

    def test_relevance_can_be_scoped_to_one_document(self):
        chunks = [Chunk("a::0", "a", "x", 0, 10), Chunk("b::0", "b", "y", 0, 10)]
        assert relevant_chunk_ids(chunks, [(0, 5)], doc_id="a") == {"a::0"}


class TestChunkStats:
    def test_describes_the_size_distribution(self):
        chunks = [Chunk(f"d::{i}", "d", "x" * (i + 1), 0, i + 1) for i in range(10)]
        stats = chunk_stats(chunks)
        assert stats["chunks"] == 10
        assert stats["min_chars"] == 1
        assert stats["max_chars"] == 10
        assert stats["mean_chars"] == 5.5

    def test_an_empty_set_reports_only_the_count(self):
        assert chunk_stats([]) == {"chunks": 0}


class TestCategoryParsing:
    def test_extracts_the_quoted_category(self):
        question = (
            'Highlight the parts (if any) of this contract related to "Governing Law" '
            "that should be reviewed by a lawyer. Details: Which state's law governs"
        )
        assert parse_category(question) == "Governing Law"

    def test_unparseable_question_falls_back(self):
        assert parse_category("What is the term?") == "unknown"


def _write_cuad(path: Path) -> Path:
    payload = {
        "version": "test",
        "data": [
            {
                "title": "ContractA",
                "paragraphs": [
                    {
                        "context": "A" * 40 + "GOVERNED BY DELAWARE LAW" + "B" * 2000,
                        "qas": [
                            {
                                "id": "ContractA__Governing Law",
                                "question": (
                                    "Highlight the parts (if any) of this contract "
                                    'related to "Governing Law" that should be '
                                    "reviewed by a lawyer."
                                ),
                                "is_impossible": False,
                                "answers": [
                                    {
                                        "text": "GOVERNED BY DELAWARE LAW",
                                        "answer_start": 40,
                                    }
                                ],
                            },
                            {
                                "id": "ContractA__Audit Rights",
                                "question": (
                                    "Highlight the parts (if any) of this contract "
                                    'related to "Audit Rights" that should be '
                                    "reviewed by a lawyer."
                                ),
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload))
    return path


class TestBuildSlice:
    def test_slice_separates_answerable_from_null(self, tmp_path):
        source = _write_cuad(tmp_path / "cuad.json")
        sliced = build_slice(source, chunk_chars=500, overlap_chars=100)

        assert len(sliced.answerable) == 1
        assert len(sliced.null_queries) == 1
        assert sliced.answerable[0].category == "Governing Law"

    def test_null_queries_get_empty_judgments(self, tmp_path):
        sliced = build_slice(_write_cuad(tmp_path / "cuad.json"))
        assert sliced.qrels["ContractA__Audit Rights"] == set()

    def test_gold_chunk_actually_contains_the_answer_text(self, tmp_path):
        sliced = build_slice(
            _write_cuad(tmp_path / "cuad.json"), chunk_chars=500, overlap_chars=100
        )
        gold_ids = sliced.qrels["ContractA__Governing Law"]
        gold_chunks = [c for c in sliced.chunks if c.chunk_id in gold_ids]
        assert gold_chunks
        assert any("GOVERNED BY DELAWARE LAW" in c.text for c in gold_chunks)

    def test_summary_reports_the_null_rate(self, tmp_path):
        sliced = build_slice(_write_cuad(tmp_path / "cuad.json"))
        summary = sliced.summary()
        assert summary["queries"] == 2
        assert summary["null_rate"] == 0.5
        assert summary["contracts"] == 1

    def test_chunks_are_reachable_under_both_lookup_names(self, tmp_path):
        sliced = build_slice(_write_cuad(tmp_path / "cuad.json"))
        assert sliced.chunks_for_doc("ContractA") == sliced.chunks
        assert sliced.chunks_for_contract("ContractA") == sliced.chunks
