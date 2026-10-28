"""Tests for projecting Langfuse traces onto the scoring harness's inputs."""

from __future__ import annotations

from aieng.advanced_rag.evaluation.metrics import score_null_detection, score_retrieval
from aieng.advanced_rag.tracing import DROPPED, project_traces


def retriever(ids, start="2026-10-01T00:00:01Z", **extra):
    return {"type": "RETRIEVER", "startTime": start, "output": ids, **extra}


def trace(trace_id, query_id, observations, output):
    return {
        "id": trace_id,
        "metadata": {"query_id": query_id},
        "observations": observations,
        "output": output,
    }


def test_the_last_retriever_is_the_final_attempt():
    projection = project_traces(
        [
            trace(
                "t1",
                "q1",
                [
                    retriever(["b", "a"], start="2026-10-01T00:00:09Z"),
                    retriever(["a", "c"], start="2026-10-01T00:00:01Z"),
                ],
                {"abstained": False},
            )
        ]
    )
    assert projection.first_runs == {"q1": ["a", "c"]}
    assert projection.runs == {"q1": ["b", "a"]}
    assert projection.abstained == {"q1": False}


def test_passages_may_be_strings_or_mappings_with_any_identifier_key():
    projection = project_traces(
        [
            trace(
                "t1",
                "q1",
                [
                    retriever(
                        [
                            "plain",
                            {"id": "by-id", "score": 0.9},
                            {"chunk_id": "by-chunk"},
                            {"provenance": {"chunk_id": "by-provenance"}},
                        ]
                    )
                ],
                {"abstained": True},
            )
        ]
    )
    assert projection.runs["q1"] == ["plain", "by-id", "by-chunk", "by-provenance"]


def test_the_projection_feeds_both_scorers():
    projection = project_traces(
        [
            trace("t1", "q1", [retriever(["a"])], {"abstained": False}),
            trace("t2", "q2", [retriever(["z"])], {"abstained": True}),
        ]
    )
    assert score_retrieval(projection.runs, {"q1": {"a"}}, k=1).hit_rate == 1.0
    assert score_null_detection(projection.abstained, ["q2"]).abstain_recall == 1.0


def test_an_abstention_is_read_from_answer_text_when_there_is_no_flag():
    projection = project_traces(
        [
            trace(
                "t1", "q1", [retriever(["a"])], "Not stated in the provided sources."
            ),
            trace("t2", "q2", [retriever(["a"])], {"answer": "Ontario law governs."}),
        ]
    )
    assert projection.abstained == {"q1": True, "q2": False}


def test_a_trace_that_does_not_project_is_skipped_with_a_reason():
    projection = project_traces(
        [
            {"id": "no-query", "observations": [retriever(["a"])], "output": {}},
            trace("no-retriever", "q1", [{"type": "GENERATION"}], {"abstained": True}),
            trace("no-flag", "q2", [retriever(["a"])], None),
            trace("no-id", "q3", [retriever([{"score": 1.0}])], {"abstained": False}),
            trace("fine", "q4", [retriever(["a"])], {"abstained": False}),
        ]
    )
    assert set(projection.skipped) == {"no-query", "no-retriever", "no-flag", "no-id"}
    assert projection.skipped["no-query"] == "no query_id in metadata or input"
    assert list(projection.runs) == ["q4"]


def test_a_repeated_query_keeps_the_later_trace_and_names_the_replaced_one():
    projection = project_traces(
        [
            trace("old", "q1", [retriever(["a"])], {"abstained": True}),
            trace("new", "q1", [retriever(["b"])], {"abstained": False}),
        ]
    )
    assert projection.runs == {"q1": ["b"]}
    assert "new" in projection.skipped["old"]


def test_the_query_id_may_come_from_the_trace_input():
    projection = project_traces(
        [
            {
                "id": "t1",
                "input": {"query_id": "q9"},
                "observations": [retriever(["a"])],
                "output": {"abstained": False},
            }
        ]
    )
    assert list(projection.runs) == ["q9"]


def test_the_losses_are_stated():
    assert len(DROPPED) == 4
