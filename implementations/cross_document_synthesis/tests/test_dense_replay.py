"""Replay every stored baseline score against the checksum-verified gold corpus."""

import json
from pathlib import Path

import pytest
from aieng.advanced_rag.corpora.musique import load_musique
from ri2_reference import reference_value
from synthesis_lab.dense_baseline import score_by_hops
from synthesis_lab.reference import dense_predictions
from synthesis_lab.runtime import MODEL_REVISION


@pytest.mark.integration_test
def test_full_baseline_outputs_replay_against_pinned_gold():
    folder = Path(__file__).resolve().parents[1] / "outputs/reference/dense_baseline"
    retrieval = json.loads((folder / "retrieval.json").read_text())
    answers = json.loads((folder / "answers.json").read_text())
    runs = json.loads((folder / "rankings.json").read_text())
    predictions = reference_value(dense_predictions)["predictions"]
    passages, questions = load_musique()
    assert len(runs) == len(predictions) == len(questions) == 1000
    assert len({row["question_id"] for row in predictions}) == 1000
    assert retrieval["embeddings"]["identity"]["encoder"]["revision"] == MODEL_REVISION
    corpus_ids = {p.passage_id for p in passages}
    for ranking in runs.values():
        assert len(ranking) == len(set(ranking)) == 20
        assert set(ranking) <= corpus_ids
    assert score_by_hops(questions, runs, [5, 10, 20]) == retrieval["metrics"]
    scored = {row["question_id"]: "" if row["abstained"] else row["answer"] for row in predictions}
    assert score_by_hops(questions, runs, [5], scored) == answers["metrics"]
    for row in predictions:
        assert set(row["citation_ids"]) <= set(runs[row["question_id"]][:5])
    assert (
        answers["harness"]["retrieval"]["recall_at_k"]
        == retrieval["metrics"]["overall"]["retrieval"]["5"]["recall_at_5"]
    )
    assert sum(row["usage"]["calls"] for row in predictions) == answers["harness"]["usage"]["calls"]
    assert sum(row["usage"]["total_tokens"] for row in predictions) == answers["harness"]["usage"]["total_tokens"]
