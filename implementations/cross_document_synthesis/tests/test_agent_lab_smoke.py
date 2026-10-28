"""Replay recorded ADK migration traces and usage without model calls."""

from ri2_reference import reference_value
from synthesis_lab.reference import agent_lab
from synthesis_lab.runtime import MODEL, MODEL_REVISION, PROXY_URL


def test_recorded_smoke_has_bounded_provenance_and_complete_usage():
    report = reference_value(agent_lab)
    assert report["adk_version"] == "2.10.0"
    assert report["base_url"] == PROXY_URL
    assert report["embedding_model"] == MODEL
    assert report["embedding_revision"] == MODEL_REVISION
    assert report["chat_template_kwargs"] == {"enable_thinking": False}
    assert report["corpus_passages"] == 11656
    assert [row["hops"] for row in report["results"]] == [2, 3, 4]
    for row in report["results"]:
        exposed, read = set(), set()
        searches = 0
        assert row["trace"][0]["tool"] == "plan_retrieval"
        for step in row["trace"]:
            if step["tool"] in ("search_musique", "grep_musique"):
                searches += 1
                exposed.update(step["passage_ids"])
            elif step["tool"] == "read_musique":
                assert set(step["passage_ids"]) <= exposed
                read.update(step["passage_ids"])
        assert searches == row["searches"] <= 3
        assert set(row["exposed_ids"]) == exposed
        assert set(row["read_ids"]) == read
        assert set(row["citations"]) <= read
        assert row["model_calls"] == row["usage"]["calls"] <= 10
        assert row["usage"]["calls_missing_usage"] == 0
        if row["status"] == "not_stated":
            assert not row["answer"] and not row["citations"]
    assert sum(row["model_calls"] for row in report["results"]) == report["usage"]["run"]["calls"]
    assert sum(row["usage"]["total_tokens"] for row in report["results"]) == report["usage"]["run"]["total_tokens"]
