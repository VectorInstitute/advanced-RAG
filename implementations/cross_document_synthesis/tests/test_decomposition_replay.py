"""Validate recorded integration results without corpus downloads or proxy calls."""

import pytest
from ri2_reference import reference_value
from synthesis_lab.decomposition import pool_rankings
from synthesis_lab.reference import decomposition_example


def test_recorded_decomposition_has_grounded_citations_and_complete_hop_scores():
    report = reference_value(decomposition_example)["report"]
    assert report["completed"]
    assert report["corpus_passages"] == 11656
    assert len(report["records"]) == report["selected_questions"] == 30
    assert len({row["question_id"] for row in report["records"]}) == 30
    assert "execution" not in report
    assert report["base_url"] == "https://proxy.vectorinstitute.ai/v1"
    assert report["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert not report["thinking"]
    for hop in ("2", "3", "4"):
        assert report["metrics"][hop]["questions"] == report["matched_dense_baseline"][hop]["questions"] == 10
        assert set(report["metrics"][hop]["retrieval"]) == {"5", "10", "20"}
        assert set(report["metrics"][hop]["answers"]) == {"exact_match", "token_f1"}
    for row in report["records"]:
        result = row["result"]
        assert row["plan"]["question"] == result["question"]
        assert 1 <= len(row["retrieval_queries"]) <= 5
        assert result["question"] in row["retrieval_queries"]
        assert row["pooled_ids"] == pool_rankings(row["step_rankings"], 20)
        assert row["generation_ids"] == row["pooled_ids"][:5]
        assert len(set(row["pooled_ids"])) == 20
        assert set(result["chunk_ids"]) <= set(row["generation_ids"])
        assert len(result["attempts"]) == 1
        assert row["usage"]["calls_missing_usage"] == 0
        assert row["usage"]["calls"] + row["usage"]["cache_hits"] >= 2
    usage = report["usage"]["run"]
    assert usage["total_tokens"] == sum(row["usage"]["total_tokens"] for row in report["records"])
    assert usage["calls"] == sum(row["usage"]["calls"] for row in report["records"])
    assert report["usage"]["unattributed"]["calls"] == 0


@pytest.mark.integration_test
def test_scores_replay_against_digest_checked_pinned_corpus():
    from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique  # noqa: PLC0415
    from run_decomposition import select_questions  # noqa: PLC0415
    from synthesis_lab.dense_baseline import score_by_hops  # noqa: PLC0415
    from synthesis_lab.runtime import DEFAULT_CACHE  # noqa: PLC0415

    report = reference_value(decomposition_example)["report"]
    passages, questions = load_musique(fetch_musique(dest=DEFAULT_CACHE / "corpus"))
    selected = select_questions(questions, 10)
    records = {row["question_id"]: row for row in report["records"]}
    assert [q.question_id for q in selected] == list(records)
    assert all(
        row["hops"] == q.hops and row["plan"]["question"] == q.question
        for q, row in zip(selected, records.values(), strict=True)
    )
    assert {pid for row in records.values() for pid in row["pooled_ids"]} <= {p.passage_id for p in passages}
    runs = {qid: row["pooled_ids"] for qid, row in records.items()}
    predictions = {qid: "" if row["result"]["abstained"] else row["result"]["answer"] for qid, row in records.items()}
    assert score_by_hops(selected, runs, [5, 10, 20], predictions) == report["metrics"]
