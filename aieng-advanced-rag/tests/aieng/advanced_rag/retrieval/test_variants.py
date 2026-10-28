"""Scoring chunking variants on the unit the reader receives, scoped and unscoped."""

from __future__ import annotations

import numpy as np
import pytest

from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query
from aieng.advanced_rag.retrieval.index import DenseIndex
from aieng.advanced_rag.retrieval.variants import (
    Arm,
    document_hit_rate,
    judgments,
    nearest_k,
    rank,
    score_arm,
)


def unit(chunk_id: str, doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(chunk_id, doc_id, "x" * (end - start), start, end)


# Two documents, two units each. Vectors are chosen so the ranking is known.
UNITS = [
    unit("a::0", "a", 0, 100),
    unit("a::1", "a", 100, 200),
    unit("b::0", "b", 0, 100),
    unit("b::1", "b", 100, 200),
]
VECTORS = np.asarray([[1, 0], [0.6, 0.8], [0.9, 0.436], [0, 1]], dtype=np.float32)
INDEX = DenseIndex([u.chunk_id for u in UNITS], VECTORS, [u.doc_id for u in UNITS])
QUERY = Query("a::q", "a", "c", "question", (AnswerSpan("x", 120, 150),))


class TestRank:
    def test_scoped_ranking_stays_in_the_query_document(self):
        runs = rank(INDEX, np.asarray([[0.0, 1.0]]), [QUERY], scoped=True, depth=2)
        assert runs == {"a::q": ["a::1", "a::0"]}

    def test_unscoped_ranking_can_prefer_another_document(self):
        runs = rank(INDEX, np.asarray([[0.0, 1.0]]), [QUERY], scoped=False, depth=2)
        assert runs == {"a::q": ["b::1", "a::1"]}

    def test_children_collapse_to_their_parents_in_first_hit_order(self):
        read_of = {"a::0": "a::p0", "a::1": "a::p0", "b::0": "b::p0", "b::1": "b::p1"}
        runs = rank(
            INDEX,
            np.asarray([[1.0, 0.0]]),
            [QUERY],
            scoped=False,
            depth=2,
            read_of=read_of,
        )
        assert runs == {"a::q": ["a::p0", "b::p0"]}


class TestJudgmentsAndScores:
    def test_judgments_come_from_overlap_inside_the_query_document(self):
        assert judgments(UNITS, [QUERY]) == {"a::q": {"a::1"}}

    def test_the_document_hit_rate_counts_any_unit_of_the_right_document(self):
        doc_of = {u.chunk_id: u.doc_id for u in UNITS}
        assert document_hit_rate({"a::q": ["b::1", "a::0"]}, doc_of, [QUERY], 1) == 0.0
        assert document_hit_rate({"a::q": ["b::1", "a::0"]}, doc_of, [QUERY], 2) == 1.0

    def test_score_arm_reports_fixed_and_budget_depths(self):
        qrels = judgments(UNITS, [QUERY])
        runs = {"a::q": ["b::1", "a::1", "a::0"]}
        scores = score_arm(runs, qrels, UNITS, [QUERY], k=1, budgets={"wide": 200})
        assert scores["fixed"]["failure"] == 1.0
        assert scores["fixed"]["chars"] == 100
        assert scores["budgets"]["wide"]["k"] == 2
        assert scores["budgets"]["wide"]["failure"] == 0.0
        assert scores["budgets"]["wide"]["mrr"] == pytest.approx(0.5)
        assert scores["gold_units_per_query"] == 1.0


def test_an_arm_with_read_units_needs_the_mapping():
    with pytest.raises(ValueError, match="together"):
        Arm("bad", UNITS, read=UNITS)
    assert Arm("flat", UNITS).read_units is UNITS


def test_the_nearest_depth_does_not_overshoot_by_a_whole_unit():
    chars = {"x": 1000, "y": 1000, "z": 1000, "w": 10}
    runs = {"q": ["x", "y", "w", "z"]}
    # The top three hold 2,010 characters, short of 2,100, so the smallest
    # depth that reaches it is four. Three is nearer.
    assert nearest_k(runs, chars, 2100, depth=4) == 3
