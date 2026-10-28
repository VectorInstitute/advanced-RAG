"""Token counting over character spans."""

from __future__ import annotations

from aieng.advanced_rag.retrieval.tokens import OffsetTokenIndex, WhitespaceTokenIndex


class TestOffsetTokenIndex:
    def test_special_tokens_are_dropped(self):
        # A fast tokenizer maps special tokens to (0, 0).
        index = OffsetTokenIndex([(0, 0), (0, 3), (4, 7), (0, 0)])
        assert len(index) == 2

    def test_count_is_the_overlapping_run(self):
        index = OffsetTokenIndex([(0, 3), (4, 7), (8, 12)])
        assert index.count(0, 12) == 3
        assert index.count(0, 3) == 1
        assert index.count(4, 12) == 2

    def test_an_empty_span_counts_nothing(self):
        index = OffsetTokenIndex([(0, 3)])
        assert index.count(5, 5) == 0
        assert index.count(9, 2) == 0

    def test_a_span_inside_one_token_still_counts_it(self):
        index = OffsetTokenIndex([(0, 10)])
        assert index.count(3, 4) == 1

    def test_token_span_bounds_the_run(self):
        index = OffsetTokenIndex([(0, 3), (4, 7), (8, 12)])
        assert index.token_span(4, 12) == (1, 3)


class TestWhitespaceTokenIndex:
    def test_counts_words(self):
        text = "one two three four"
        index = WhitespaceTokenIndex(text)
        assert len(index) == 4
        assert index.count(0, len(text)) == 4

    def test_offsets_line_up_with_the_source(self):
        text = "alpha beta gamma"
        index = WhitespaceTokenIndex(text)
        # "beta" starts at 6 and ends at 10.
        assert index.count(6, 10) == 1
