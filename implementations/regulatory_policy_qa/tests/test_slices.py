"""Tests for the corpus slicing, the null-control scoping, and the cost estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_lab.slices import estimate_calls, scoped_null_queries


def write_null_file(tmp_path: Path, count: int = 5) -> Path:
    """Write a MultiHop-RAG query file holding only null queries."""
    payload = [
        {
            "query": f"What happened in story {index}?",
            "question_type": "null_query",
            "evidence_list": [],
            "answer": "Insufficient information.",
        }
        for index in range(count)
    ]
    path = tmp_path / "MultiHopRAG.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestScopedNullQueries:
    def test_every_query_gets_a_document_to_retrieve_against(self, tmp_path):
        queries = scoped_null_queries(write_null_file(tmp_path), ["a", "b"])
        assert all(query.doc_id for query in queries)

    def test_the_documents_are_assigned_round_robin(self, tmp_path):
        queries = scoped_null_queries(write_null_file(tmp_path, 4), ["a", "b"])
        assert [query.doc_id for query in queries] == ["a", "b", "a", "b"]

    def test_the_assignment_is_the_same_on_a_second_run(self, tmp_path):
        path = write_null_file(tmp_path)
        first = scoped_null_queries(path, ["a", "b", "c"])
        second = scoped_null_queries(path, ["a", "b", "c"])
        assert [q.doc_id for q in first] == [q.doc_id for q in second]

    def test_every_scoped_query_stays_a_null_case(self, tmp_path):
        queries = scoped_null_queries(write_null_file(tmp_path), ["a"])
        assert not any(query.is_answerable for query in queries)

    def test_the_limit_takes_the_first_queries_in_file_order(self, tmp_path):
        queries = scoped_null_queries(write_null_file(tmp_path, 5), ["a"], limit=2)
        assert len(queries) == 2
        assert queries[0].question == "What happened in story 0?"

    def test_scoping_onto_no_documents_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one document"):
            scoped_null_queries(write_null_file(tmp_path), [])

    def test_a_null_query_carrying_evidence_still_raises_upstream(self, tmp_path):
        path = tmp_path / "MultiHopRAG.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "query": "q",
                        "question_type": "null_query",
                        "evidence_list": [{"fact": "an article sentence"}],
                    }
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="carries evidence"):
            scoped_null_queries(path, ["a"])


class TestEstimateCalls:
    def test_the_llm_arm_is_priced_at_one_call_per_graded_passage(self):
        estimate = estimate_calls(10, attempts=2, max_chunks=5)
        assert estimate["grading"] == 100

    def test_the_model_free_grader_costs_nothing_to_grade_with(self):
        estimate = estimate_calls(10, attempts=2, llm_grader=False)
        assert estimate["grading"] == 0
        assert estimate["total"] == estimate["rewriting"] + estimate["generation"]

    def test_the_rewriter_is_not_called_on_the_final_attempt(self):
        assert estimate_calls(10, attempts=1)["rewriting"] == 0
        assert estimate_calls(10, attempts=2)["rewriting"] == 10
        assert estimate_calls(10, attempts=3)["rewriting"] == 20

    def test_generation_is_priced_on_every_attempt_the_harvest_records(self):
        assert estimate_calls(10, attempts=2)["generation"] == 20

    def test_the_total_is_the_sum_of_the_stages(self):
        estimate = estimate_calls(7, attempts=2, max_chunks=3)
        assert estimate["total"] == (estimate["grading"] + estimate["rewriting"] + estimate["generation"])

    def test_turning_the_rewriter_off_removes_its_calls(self):
        assert estimate_calls(10, attempts=2, rewriter=False)["rewriting"] == 0
