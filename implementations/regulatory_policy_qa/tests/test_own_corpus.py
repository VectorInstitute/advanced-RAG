"""The levers-and-answers notebook's shipped reference results and its comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aieng.advanced_rag.models import family_of


RI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RI))

import byoc  # noqa: E402
import own_corpus  # noqa: E402


def envelopes(artifact_type: str) -> list[dict]:
    paths = sorted(byoc.BASELINE.glob(f"{own_corpus.RI_ID}/{artifact_type}/**/*.json"))
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return [e for e in loaded if e["producer_id"] == own_corpus.PRODUCER]


def test_the_stand_in_lever_scores_cover_every_lever_against_bm25():
    [found] = envelopes("own-corpus-levers")
    value = found["value"]
    assert value["arms"] == list(own_corpus.LEVERS)
    assert value["baseline"] == own_corpus.BASELINE_ARM
    for metric in own_corpus.METRICS:
        assert set(value["mean"][metric]) == set(own_corpus.LEVERS)
        assert set(value["versus_baseline"][metric]) == set(own_corpus.LEVERS) - {own_corpus.BASELINE_ARM}
        for gap in value["versus_baseline"][metric].values():
            assert gap["low"] <= gap["mean_difference"] <= gap["high"]


def test_the_stand_in_answers_cover_every_question_on_free_models():
    [found] = envelopes("own-corpus-answers")
    value = found["value"]
    assert {family_of(m) for m in value["models"].values()} == {"onprem"}
    assert value["models"]["generation"] == own_corpus.REFERENCE_MODELS.generation.model
    assert len(value["rows"]) == value["questions"] == value["answerable"] + value["null"]
    assert value["null"] > 0 and value["judged"] == value["answerable"]
    assert value["calls"]["judging"]["calls"] == value["answerable"]
    assert 0.0 <= value["balanced_decision_accuracy"] <= 1.0


def test_both_results_belong_to_the_same_stand_in_set():
    [levers] = envelopes("own-corpus-levers")
    [answers] = envelopes("own-corpus-answers")
    assert levers["config"]["set"] == answers["config"]["set"]


def test_the_paired_difference_is_against_the_baseline():
    scores = {
        metric: {
            own_corpus.BASELINE_ARM: {"q1": 0.0, "q2": 0.0, "q3": 0.0},
            "splade@1200": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
        }
        for metric in own_corpus.METRICS
    }
    versus = own_corpus._versus(scores, own_corpus.BASELINE_ARM)
    for metric in own_corpus.METRICS:
        assert versus[metric]["splade@1200"] == {"mean_difference": 1.0, "low": 1.0, "high": 1.0}
