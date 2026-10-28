"""Late chunking's window arithmetic.

The encoder itself needs torch and a long-context checkpoint, so it is not
exercised here. What is tested is the part that decides which macro window a
chunk is pooled from, because a chunk pooled from a window that truncates it
would silently lose the surrounding document that late chunking exists to add.
"""

from __future__ import annotations

import numpy as np

from aieng.advanced_rag.retrieval.late import _best_window, _normalize_rows


class TestBestWindow:
    def test_a_chunk_inside_one_window_uses_that_window(self):
        assert _best_window(10, 20, [0, 100, 200], size=100) == 0

    def test_a_containing_window_beats_a_closer_truncating_one(self):
        # Window 100 would cut the chunk at its edge even though its centre is
        # nearer, so the window that holds the chunk whole wins.
        assert _best_window(80, 150, [0, 100, 200], size=200) == 0

    def test_among_containing_windows_the_most_centred_one_wins(self):
        # The chunk sits at 100-140. Window 50 centres at 100, window 0 at 50.
        assert _best_window(100, 140, [0, 50], size=100) == 50

    def test_a_chunk_no_window_contains_falls_back_to_the_nearest(self):
        # Longer than the window, so nothing holds it whole.
        assert _best_window(0, 500, [0, 100, 200], size=50) == 200

    def test_a_single_window_document_always_picks_it(self):
        assert _best_window(0, 10, [0], size=8192) == 0


class TestNormalizeRows:
    def test_rows_become_unit_length(self):
        matrix = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = _normalize_rows(matrix)
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
        assert np.allclose(normalized[0], [0.6, 0.8])

    def test_a_zero_row_does_not_divide_by_zero(self):
        # A chunk whose window produced no states pools to zeros.
        normalized = _normalize_rows(np.zeros((1, 4), dtype=np.float32))
        assert np.all(np.isfinite(normalized))
        assert np.allclose(normalized, 0.0)
