"""Strict secondary judging and request provenance used by the full evaluation."""

import json

import pytest
from aieng.advanced_rag.corpora.musique import MusiqueQuestion, Passage, SubQuestion
from aieng.advanced_rag.graphrag.proxy import ResponseCache
from aieng.advanced_rag.graphrag.synthesis import build_prompt
from aieng.advanced_rag.pipeline.generate import parse_answer
from benchmarks.full_evaluation import (
    aggregate,
    holm_adjust,
    paired_interval,
    question_metrics,
    validate_judge_cache,
    validate_prompt_cache,
    validate_records,
)
from benchmarks.judge_full_answers import JUDGE_MODEL, JUDGE_PROMPT, parse_judge
from synthesis_lab.dense_baseline import ANSWER_TEMPLATE, DensePassageRetriever
from synthesis_lab.runtime import ANSWER_MODEL, PROXY_URL


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"correct": "false", "context_supported": True, "citations_supported": True, "reason": "bad"},
        {"correct": True, "context_supported": True, "citations_supported": None, "reason": "bad"},
        {"correct": True, "context_supported": False, "citations_supported": False},
    ],
)
def test_judge_parser_rejects_partial_labels_and_non_boolean_values(payload):
    with pytest.raises(ValueError):
        parse_judge(json.dumps(payload))


def test_judge_fields_remain_independent():
    payload = {
        "correct": True,
        "context_supported": False,
        "citations_supported": False,
        "reason": "Right answer unsupported by evidence",
    }
    assert parse_judge(json.dumps(payload)) == payload


def test_paired_interval_uses_questionwise_differences_and_is_reproducible():
    interval = paired_interval([1.0, 0.5, 1.0], [0.5, 0.0, 0.5], samples=1000)
    assert interval == {"difference": 0.5, "low": 0.5, "high": 0.5, "confidence": 0.95}
    assert paired_interval([1.0, 0.0], [0.0, 1.0], samples=1000) == paired_interval(
        [1.0, 0.0], [0.0, 1.0], samples=1000
    )
    with pytest.raises(ValueError):
        paired_interval([1.0], [])


def test_holm_adjustment_controls_the_entire_em_test_family():
    assert holm_adjust({"overall": 0.01, "2": 0.03, "3": 0.04, "4": 0.9}) == {
        "overall": 0.04,
        "2": 0.09,
        "3": 0.09,
        "4": 0.9,
    }


def test_correct_answer_with_incomplete_evidence_is_not_counted_as_joint_success():
    q = MusiqueQuestion(
        "q",
        "question",
        "gold",
        (),
        2,
        ("p0", "p9"),
        (
            SubQuestion("first", "x", "p0", step_id=1),
            SubQuestion("second", "gold", "p9", step_id=2, depends_on=(1,)),
        ),
    )
    row = {"result": {"answer": "gold", "abstained": False, "chunk_ids": ["p0"], "reason": "", "dropped_citations": []}}
    metrics = question_metrics(q, row, [f"p{i}" for i in range(20)])
    assert metrics["em"] == 1
    assert metrics["joint_em_complete_context"] == 0
    assert metrics["recall_at_5"] == 0.5
    assert metrics["all_supports_at_10"] == 1
    assert metrics["context_dependency_edge_coverage"] == 0
    assert metrics["error_category"] == "correct_incomplete_context"
    summary = aggregate([metrics])
    assert summary["em"]["denominator"] == 1
    assert summary["citation_gold_precision"]["mean"] == 1


def test_abstentions_are_zero_correctness_and_have_no_citation_precision_denominator():
    q = MusiqueQuestion("q", "question", "a", (), 2, ("p0", "p1"), ())
    row = {"result": {"answer": "", "abstained": True, "chunk_ids": [], "reason": "missing", "dropped_citations": []}}
    metrics = question_metrics(q, row, [f"p{i}" for i in range(20)])
    assert metrics["em"] == metrics["f1"] == 0
    summary = aggregate([metrics])
    assert summary["questions"] == 1 and summary["attempted"] == 0
    assert summary["citation_gold_precision"]["denominator"] == 0


def test_prompt_audit_rejects_changed_context_and_changed_raw_response(tmp_path):
    q = MusiqueQuestion("q", "question", "answer", (), 2, ("p0", "p1"), ())
    passages = [Passage(f"p{i}", "Title", f"text {i}") for i in range(5)]
    ids = [p.passage_id for p in passages]
    evidence = DensePassageRetriever(passages, {q.question: ids})(q.question, k=5)
    cache = ResponseCache(tmp_path / "responses.sqlite")
    key = cache.key(
        {
            "prompt": build_prompt(q.question, evidence, ANSWER_TEMPLATE, None),
            "model": ANSWER_MODEL,
            "base_url": PROXY_URL,
            "temperature": 0,
            "max_tokens": 256,
            "thinking": False,
        }
    )
    cache.put(key, "original response", 10, 5)
    row = {"generation_ids": ids, "raw_answer": "original response", "completion_requests": [{"cache_key": key}]}
    validate_prompt_cache({"q": row}, [q], passages, tmp_path / "responses.sqlite")
    row["raw_answer"] = "different response"
    with pytest.raises(ValueError, match="raw response differs"):
        validate_prompt_cache({"q": row}, [q], passages, tmp_path / "responses.sqlite")
    row["generation_ids"] = list(reversed(ids))
    with pytest.raises(ValueError, match="prompt provenance differs"):
        validate_prompt_cache({"q": row}, [q], passages, tmp_path / "responses.sqlite")


def test_dense_trace_has_five_passages_but_retrieval_metrics_use_twenty():
    q = MusiqueQuestion("q", "question", "answer", (), 2, ("p0", "p1"), ())
    passages = [Passage(f"p{i}", "Title", f"text {i}") for i in range(20)]
    ids = [p.passage_id for p in passages]
    raw = '{"answer":"answer","sources":[1],"not_stated":false}'
    evidence = DensePassageRetriever(passages, {q.question: ids})(q.question, k=5)
    result = {**parse_answer("q", raw, evidence).to_dict(), "question": q.question, "retrieved_ids": ids[:5]}
    row = {"result": result, "generation_ids": ids[:5], "raw_answer": raw}
    validate_records({"q": row}, {"q": ids}, [q], passages)
    row["generation_ids"] = ids[1:6]
    with pytest.raises(ValueError, match="generation context differs"):
        validate_records({"q": row}, {"q": ids}, [q], passages)


@pytest.mark.parametrize("role_selected", [False, True])
def test_judge_audit_rejects_tampered_verdict(tmp_path, role_selected):
    q = MusiqueQuestion("q", "question", "answer", (), 2, ("p0", "p1"), ())
    passages = [Passage("p0", "Title", "text")]
    source = {"q": {"result": {"abstained": False, "answer": "answer", "chunk_ids": ["p0"]}, "generation_ids": ["p0"]}}
    prompt = JUDGE_PROMPT.format(
        question="question", references='["answer"]', answer="answer", passages="ID=p0; cited=true\nTitle\ntext"
    )
    cache = ResponseCache(tmp_path / "judge.sqlite")
    configuration = None
    request = {
        "prompt": prompt,
        "model": JUDGE_MODEL,
        "base_url": PROXY_URL,
        "temperature": 0,
        "max_tokens": 2048,
        "thinking": False,
        "reasoning_effort": "low",
    }
    if role_selected:
        from aieng.advanced_rag.models import choose_models, request_params  # noqa: PLC0415

        model = choose_models().judging.model
        params = request_params(model, max_tokens=2048, thinking=False)
        configuration = {"judge_model": model, "base_url": PROXY_URL, "request_params": params}
        request["model"] = model
        request["request_params"] = params
        request.pop("reasoning_effort")
    key = cache.key(request)
    verdict = {"correct": True, "context_supported": True, "citations_supported": True, "reason": "Supported"}
    raw = json.dumps(verdict)
    cache.put(key, raw, 10, 5)
    row = {**verdict, "raw_judgement": raw, "completion_requests": [{"cache_key": key}]}
    validate_judge_cache({"q": row}, source, [q], passages, tmp_path / "judge.sqlite", configuration=configuration)
    row["correct"] = False
    with pytest.raises(ValueError, match="raw judgement replay differs"):
        validate_judge_cache({"q": row}, source, [q], passages, tmp_path / "judge.sqlite", configuration=configuration)
