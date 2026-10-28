"""Validate gold-free planning, pooled generation, fallback, and usage accounting."""

import json

import numpy as np
import pytest
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.metering import CallUsage, UsageMeter, report_call
from aieng.advanced_rag.pipeline.answering import Answerer
from aieng.advanced_rag.retrieval.interface import ScopeUnsupported
from synthesis_lab.decomposition import DecompositionAnswerer, ModelDecomposer, pool_rankings


@pytest.mark.parametrize(
    ("raw", "outcome"),
    [
        ("not JSON", "invalid-plan"),
        ('{"subquestions": []}', "declined"),
        ('{"subquestions": "bad"}', "invalid-plan"),
        ('{"subquestions": ["one"]}', "invalid-plan"),
        ('{"subquestions": ["one", 2]}', "invalid-plan"),
        ('{"subquestions": ["one", " "]}', "invalid-plan"),
        ('{"subquestions": ["one", "one"]}', "invalid-plan"),
        ('{"subquestions": ["one", "What is #1?"]}', "invalid-plan"),
        (json.dumps({"subquestions": list("abcde")}), "invalid-plan"),
    ],
)
def test_malformed_or_declined_plan_falls_back(raw, outcome):
    decompose = ModelDecomposer(lambda _: raw)
    assert decompose("original") == []
    assert decompose.outcome == outcome


def test_pooling_deduplicates_and_gives_each_query_a_turn():
    assert pool_rankings([["a", "b", "c"], ["a", "d", "e"], ["f", "g", "h"]], 5) == ["a", "f", "b", "d", "g"]
    assert pool_rankings([], 5) == []
    with pytest.raises(ValueError):
        pool_rankings([["a"]], 0)


def build_answerer(plan='{"subquestions": ["first need", "second need"]}'):
    passages = [Passage(f"p{i}", f"Title {i}", f"Evidence {i}") for i in range(6)]
    vectors = np.eye(6, dtype=np.float32)
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        report_call(CallUsage(100, 10, 110, reported=True), 0.01)
        return plan if "Split the question" in prompt else '{"answer":"supported","sources":[2],"not_stated":false}'

    def encode(queries):
        return [vectors[{"first need": 0, "second need": 4, "ORIGINAL": 5}.get(q, 5)].tolist() for q in queries]

    answerer = DecompositionAnswerer(
        passages, vectors, encode, complete, complete, retrieval_depth=6, evidence_budget=3
    )
    return answerer, prompts


def test_answerer_pools_all_steps_preserves_original_and_meter_scope():
    implementation, prompts = build_answerer()
    answerer: Answerer = implementation
    meter = UsageMeter()
    with meter.question("q1"):
        result = answerer.answer("q1", "ORIGINAL")
    trace = implementation.traces["q1"]
    assert trace["plan_outcome"] == "decomposed"
    assert all(step["parent"] == "ORIGINAL" for step in trace["plan"]["steps"])
    assert trace["generation_ids"] == ["p0", "p4", "p5"]
    assert trace["retrieval_queries"] == ["first need", "second need", "ORIGINAL"]
    assert result.question == "ORIGINAL" and result.answer.chunk_ids == ("p4",)
    assert len(prompts) == 2
    assert "Question: ORIGINAL" in prompts[-1]
    assert all(f"Evidence {i}" in prompts[-1] for i in (0, 4, 5))
    assert "Evidence 1" not in prompts[-1]
    assert meter.per_question()["q1"].calls == 2
    assert meter.per_question()["q1"].total_tokens == 220


def test_invalid_plan_uses_original_dense_ranking_and_still_answers():
    answerer, prompts = build_answerer("bad plan")
    answerer.answer("fallback", "ORIGINAL")
    trace = answerer.traces["fallback"]
    assert trace["plan_outcome"] == "invalid-plan"
    assert trace["retrieval_queries"] == ["ORIGINAL"]
    assert trace["pooled_ids"] == trace["step_rankings"][0]
    assert len(prompts) == 2


def test_scope_and_provider_failures_propagate_without_fabricated_abstentions():
    answerer, prompts = build_answerer()
    with pytest.raises(ScopeUnsupported):
        answerer.answer("scoped", "ORIGINAL", "document")
    assert not prompts

    def fail(_):
        raise RuntimeError("proxy unavailable")

    answerer.decomposer.complete = fail
    with pytest.raises(RuntimeError, match="proxy unavailable"):
        answerer.answer("outage", "ORIGINAL")
    assert "outage" not in answerer.traces


def test_question_state_is_fresh_and_invalid_query_vectors_fail():
    answerer, _ = build_answerer()
    answerer.answer("one", "ORIGINAL")
    answerer.decomposer.complete = lambda _: '{"subquestions": []}'
    answerer.answer("two", "OTHER")
    assert answerer.traces["two"]["retrieval_queries"] == ["OTHER"]
    assert answerer.traces["one"]["plan_outcome"] == "decomposed"
    answerer.encode_queries = lambda queries: [[float("nan")] * 6 for _ in queries]
    with pytest.raises(ValueError, match="query vectors"):
        answerer.answer("invalid", "OTHER")
