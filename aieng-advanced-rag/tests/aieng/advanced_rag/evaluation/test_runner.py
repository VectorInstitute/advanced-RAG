"""Tests for the configuration runner: one knob turned, the rest held fixed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aieng.advanced_rag.evaluation.runner import Experiment, scope_of
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.pipeline.grading import EvidenceGrade, GradingThresholds
from aieng.advanced_rag.pipeline.spine import SpineConfig
from aieng.advanced_rag.provenance import Provenance


BOUNDARY = GradingThresholds(sufficient=0.6, insufficient=0.3)
ANSWER = json.dumps({"answer": "Ontario law governs.", "sources": [1]})


class ScoreGrader:
    """Grades every query at a fixed score."""

    name = "score"

    def __init__(self, score: float) -> None:
        self.score = score
        self.thresholds = BOUNDARY

    def grade(self, question, evidence, query_id=""):
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=BOUNDARY.verdict(self.score),
            score=self.score,
            reason="",
            thresholds=BOUNDARY,
        )


def query(query_id, answerable=True, doc_id="c1"):
    return SimpleNamespace(
        query_id=query_id,
        question=f"question {query_id}",
        doc_id=doc_id,
        is_answerable=answerable,
    )


@pytest.fixture
def make_evidence():
    def build(texts, doc_id="doc1"):
        return [
            Evidence(
                text=text,
                provenance=Provenance(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{index:03d}",
                    start=index * 100,
                    end=index * 100 + len(text),
                ),
                score=1.0 - index / 100,
                retriever="fake",
            )
            for index, text in enumerate(texts)
        ]

    return build


@pytest.fixture
def experiment(make_evidence):
    asked: list[str | None] = []

    def retrieve(question, scope=None, k=10):
        asked.append(scope)
        return make_evidence(["first passage", "second passage"], doc_id=scope or "d")

    exp = Experiment(
        queries=[query("q1"), query("q2", answerable=False, doc_id="c2")],
        qrels={"q1": {"c1::000"}},
        retrieve=retrieve,
        grader=ScoreGrader(0.9),
        complete=lambda prompt: ANSWER,
        spine={"config": SpineConfig(thresholds=BOUNDARY)},
    )
    return exp, asked


def test_a_run_scores_null_detection_and_retrieval(experiment):
    experiment, _ = experiment
    result = experiment.run("base", meter=UsageMeter())
    assert result.name == "base"
    assert result.null.null_queries == 1
    assert result.null.false_answer_rate == 1.0
    assert result.retrieval is not None
    assert result.retrieval.queries == 1
    assert result.retrieval.hit_rate == 1.0


def test_the_scope_is_read_from_doc_id_then_contract_id(experiment):
    experiment, asked = experiment
    experiment.run("base", meter=UsageMeter())
    assert asked == ["c1", "c2"]
    assert scope_of(SimpleNamespace(contract_id="k")) == "k"
    assert scope_of(SimpleNamespace()) is None


def test_with_replaces_one_part_and_leaves_the_original(experiment):
    experiment, _ = experiment
    abstaining = experiment.with_(grader=ScoreGrader(0.0))
    assert abstaining.grader is not experiment.grader
    assert abstaining.retrieve is experiment.retrieve
    result = abstaining.run("abstain", meter=UsageMeter())
    assert result.null.false_answer_rate == 0.0
    assert experiment.run("base", meter=UsageMeter()).null.false_answer_rate == 1.0


def test_a_spine_knob_merges_and_none_removes_it(experiment):
    experiment, _ = experiment
    tighter = experiment.with_(max_chars=50)
    assert tighter.spine["max_chars"] == 50
    assert "config" in tighter.spine
    assert "config" not in tighter.with_(config=None).spine


def test_an_unknown_knob_is_rejected(experiment):
    experiment, _ = experiment
    with pytest.raises(ValueError, match="unknown knob 'retreive'"):
        experiment.with_(retreive=None)


def test_an_unknown_spine_option_is_rejected_at_construction(experiment):
    experiment, _ = experiment
    with pytest.raises(ValueError, match="unknown spine option"):
        Experiment(
            queries=experiment.queries,
            qrels={},
            retrieve=experiment.retrieve,
            grader=experiment.grader,
            complete=experiment.complete,
            spine={"thresholds": BOUNDARY},
        )


def test_compare_runs_each_arm_in_order(experiment):
    experiment, _ = experiment
    arms = experiment.compare(
        {"answers": {}, "abstains": {"grader": ScoreGrader(0.0)}}, meter=UsageMeter()
    )
    assert list(arms) == ["answers", "abstains"]
    assert arms["answers"].null.false_answer_rate == 1.0
    assert arms["abstains"].null.false_answer_rate == 0.0


def test_a_null_only_slice_has_no_retrieval_report(experiment):
    experiment, _ = experiment
    only_null = experiment.with_(queries=[query("q2", answerable=False)], qrels={})
    result = only_null.run("null", meter=UsageMeter())
    assert result.retrieval is None
    assert result.to_dict()["retrieval"] is None


def test_usage_is_the_change_across_the_run(experiment):
    experiment, _ = experiment
    meter = UsageMeter()
    with meter.question("earlier"):
        pass
    result = experiment.run("base", meter=meter)
    assert result.usage.questions == 2


def test_no_queries_is_rejected(experiment):
    experiment, _ = experiment
    with pytest.raises(ValueError, match="at least one query"):
        experiment.with_(queries=[])
