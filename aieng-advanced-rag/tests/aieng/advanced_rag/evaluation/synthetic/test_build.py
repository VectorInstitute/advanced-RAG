"""End-to-end build with stub models: the funnel, the nulls, and the harness fit."""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.evaluation.metrics import score_null_detection
from aieng.advanced_rag.evaluation.runner import Experiment, scope_of
from aieng.advanced_rag.evaluation.synthetic import (
    GenerationConfig,
    SyntheticSet,
    build_single_hop_set,
    filters,
)
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.pipeline.grading import EvidenceGrade, GradingThresholds
from aieng.advanced_rag.pipeline.spine import SpineConfig
from aieng.advanced_rag.retrieval.retrievers import run_retriever
from aieng.advanced_rag.retrieval.sparse import BM25Retriever

from .conftest import model


def config(**overrides) -> GenerationConfig:
    values = {
        "domain": "non-disclosure agreements",
        "passages_per_document": 2,
        "statements_per_passage": 2,
        "passage_chars": 400,
        "scope": "document",
        "dedup_cosine": 0.9999,
        "boilerplate_cosine": 1.01,
        "workers": 4,
        "null_share": 0.25,
    }
    values.update(overrides)
    return GenerationConfig(**values)


def build(corpus, script, embedder, **overrides) -> SyntheticSet:
    return build_single_hop_set(
        corpus,
        config(**overrides),
        generator=model("gen-a", script),
        checker=model("check-b", script),
        judge=model("judge-c", script),
        embedder=embedder,
    )


@pytest.fixture
def generated(corpus, script, embedder) -> SyntheticSet:
    return build(corpus, script, embedder)


def test_the_build_accepts_answerable_questions_and_nulls(generated):
    summary = generated.summary()
    assert summary["answerable"] >= 6
    assert summary["null"] >= 1
    assert summary["by_variant"]["paraphrase"] >= 1
    assert all(q.labels["question_chars"] > 0 for q in generated)


def test_every_candidate_ends_in_exactly_one_outcome(generated):
    funnel = generated.funnel
    assert funnel is not None
    outcomes = funnel.outcomes
    assert len({o.candidate_id for o in outcomes}) == len(outcomes)
    accepted = {o.candidate_id for o in outcomes if o.outcome == "accepted"}
    plain = {q.query_id for q in generated if q.variant == "plain"}
    assert accepted == plain
    # The stub's second quote is in no passage, so the quote check drops it.
    reasons = funnel.to_dict()["drop_reasons"]
    assert any("quote not found" in reason for reason in reasons)


def test_held_out_documents_never_reach_the_index_the_set_targets(generated):
    held = set(generated.meta["documents"]["held_out"])
    indexed = set(generated.meta["documents"]["indexed"])
    assert held and not held & indexed
    for question in generated.answerable:
        assert not set(question.documents) & held
    held_out_nulls = [q for q in generated.nulls if q.kind == "held_out_null"]
    assert held_out_nulls
    for question in held_out_nulls:
        assert question.provenance["held_out_document"] in held
        assert question.scope in indexed


def test_a_null_the_corpus_answers_is_dropped(corpus, script, embedder):
    script.handlers[filters.NULL_SYSTEM] = lambda prompt: {"answered": "partly"}
    generated = build(corpus, script, embedder)
    assert generated.nulls == []
    reasons = generated.funnel.to_dict()["drop_reasons"]
    assert any(r.startswith("null_check") for r in reasons)


def test_a_round_trip_mismatch_is_filtered(corpus, script, embedder):
    script.handlers[filters.ANSWER_SYSTEM] = lambda prompt: {
        "answerable": True,
        "answer": "something else entirely",
    }
    generated = build(corpus, script, embedder, null_share=0.0)
    assert generated.answerable == []
    stages = {(s["kind"], s["stage"]): s for s in generated.funnel.to_dict()["stages"]}
    assert stages[("single_hop", "round_trip")]["passed"] == 0


def test_the_models_are_recorded_with_their_families(generated):
    assert generated.meta["models"] == {
        "generator": "gen-a",
        "checker": "check-b",
        "judge": "judge-c",
    }
    question = generated.answerable[0]
    assert question.provenance["generator"] == "gen-a"
    assert question.provenance["checker"] == "check-b"


def test_qrels_are_reachable_under_two_chunkers(generated, corpus):
    for width, overlap in ((300, 50), (1200, 200)):
        chunks = [
            chunk
            for document in corpus
            for chunk in chunk_document(
                document.doc_id, document.text, chunk_chars=width, overlap_chars=overlap
            )
        ]
        qrels = generated.qrels(chunks)
        assert all(qrels[q.query_id] for q in generated.answerable)
        assert all(not qrels[q.query_id] for q in generated.nulls)


def test_an_unreachable_span_raises(generated, corpus):
    only_first = [
        c
        for c in chunk_document(corpus[0].doc_id, corpus[0].text)
        if c.doc_id == "nda-0"
    ]
    with pytest.raises(ValueError, match="no covering chunk"):
        generated.qrels(only_first)


def test_document_scoped_questions_convert_to_corpus_queries(generated, corpus):
    chunks = [c for d in corpus for c in chunk_document(d.doc_id, d.text)]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    queries = generated.variant("plain").queries()
    runs = run_retriever(bm25, [q for q in queries if q.is_answerable], k=5)
    assert runs and all(runs.values())
    assert {q.doc_id for q in queries} <= {d.doc_id for d in corpus}


def test_a_corpus_scoped_question_is_not_scoped(corpus, script, embedder):
    generated = build(corpus, script, embedder, scope="corpus")
    assert all(scope_of(q) is None for q in generated)


def test_the_set_runs_through_experiment_and_null_detection():
    from aieng.advanced_rag.evaluation.synthetic.testset import (
        EvidenceSpan,
        SyntheticQuestion,
    )

    boundary = GradingThresholds(sufficient=0.6, insufficient=0.3)

    class Grader:
        name = "fixed"
        thresholds = boundary

        def grade(self, question, evidence, query_id=""):
            score = 0.0 if "vault" in question else 0.9
            return EvidenceGrade(
                query_id=query_id,
                question=question,
                verdict=boundary.verdict(score),
                score=score,
                reason="",
                thresholds=boundary,
            )

    from aieng.advanced_rag.graphrag.evidence import Evidence
    from aieng.advanced_rag.provenance import Provenance

    asked: list[str | None] = []

    def retrieve(question, scope=None, k=10):
        asked.append(scope)
        return [
            Evidence(
                text="Clause text",
                provenance=Provenance(
                    doc_id="d1", chunk_id="d1::0000", start=0, end=11
                ),
                score=1.0,
                retriever="fake",
            )
        ]

    questions = [
        SyntheticQuestion(
            "a",
            "what is kept?",
            "pricing",
            (EvidenceSpan("d1", 0, 5, "Claus"),),
            scope="d1",
        ),
        SyntheticQuestion(
            "n", "who audits the vault?", "Not stated.", kind="near_miss_null"
        ),
    ]
    experiment = Experiment(
        queries=questions,
        qrels={"a": {"d1::0000"}, "n": set()},
        retrieve=retrieve,
        grader=Grader(),
        complete=lambda prompt: json.dumps({"answer": "pricing", "sources": [1]}),
        spine={"config": SpineConfig(thresholds=boundary)},
    )
    result = experiment.run("synthetic", meter=UsageMeter())
    assert asked == ["d1", None]
    assert result.null.null_queries == 1
    assert result.null.abstain_recall == 1.0
    assert score_null_detection({"n": True, "a": False}, ["n"]).abstain_precision == 1.0


def test_jsonl_round_trip(generated, tmp_path):
    path = tmp_path / "set.jsonl"
    generated.to_jsonl(path)
    loaded = SyntheticSet.from_jsonl(path)
    assert [q.to_dict() for q in loaded] == [q.to_dict() for q in generated]


def test_a_rerun_with_the_same_seed_is_identical(corpus, script, embedder):
    first = build(corpus, script, embedder)
    second = build(corpus, script, embedder)
    assert [q.to_dict() for q in first] == [q.to_dict() for q in second]


def test_calls_on_worker_threads_reach_the_bound_meter():
    from aieng.advanced_rag import metering
    from aieng.advanced_rag.evaluation.synthetic.build import map_concurrent

    meter = UsageMeter()

    def call(_):
        metering.report_call(metering.CallUsage(10, 2, 12), 0.01, model="gen-a")
        return True

    with metering.using(meter), meter.question("q"):
        map_concurrent(call, range(8), workers=4)
    assert meter.by_model()["gen-a"].calls == 8
