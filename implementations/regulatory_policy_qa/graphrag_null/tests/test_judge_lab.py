"""Check gold polarity, abstention denominators, and strict judge verdicts."""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "aieng-advanced-rag"))

import judge_lab as judging  # noqa: E402
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.provenance import Provenance  # noqa: E402
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Query  # noqa: E402


def test_references_restore_contradiction_and_entailment(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "labels": {"yes": {"hypothesis": "Must delete."}, "no": {"hypothesis": "May retain."}},
                "documents": [
                    {
                        "id": 1,
                        "text": "Delete everything.",
                        "spans": [[0, 18]],
                        "annotation_sets": [
                            {
                                "annotations": {
                                    "yes": {"choice": "Entailment", "spans": [0]},
                                    "no": {"choice": "Contradiction", "spans": [0]},
                                }
                            }
                        ],
                    }
                ],
            }
        )
    )
    gold = judging.references(corpus, 1)
    assert "supported: Must delete." in gold["1::yes"]
    assert "contradicted: May retain." in gold["1::no"]
    assert "Delete everything." in gold["1::no"]
    assert "Gold label" not in gold["1::no"]
    assert "Contradiction" not in gold["1::no"]


@pytest.mark.parametrize("raw", ["", "{}", '{"correct":"false","grounded":true,"abstained":false,"reason":"x"}'])
def test_invalid_judge_responses_are_failures_not_wrong_answers(raw):
    with pytest.raises(ValueError):
        judging.checked_complete(lambda _: raw)("prompt")


def test_all_answerables_including_abstentions_are_judged_at_fixed_budget():
    queries = [
        Query(
            query_id=qid,
            doc_id="1",
            category="c",
            question="Must delete?",
            answers=(AnswerSpan(text="delete", start=0, end=6),),
        )
        for qid in ("a", "b")
    ]
    outcomes = {
        "a": {"answer": "Yes.", "abstained": False},
        "b": {"answer": judging.ABSTENTION_TEXT, "abstained": True},
    }
    evidence = [
        Evidence(text="delete " + "x" * 1400 + "TRUNCATED", provenance=Provenance(doc_id="1", chunk_id=str(i)))
        for i in range(6)
    ]
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        abstained = f"Candidate answer:\n{judging.ABSTENTION_TEXT}" in prompt
        grounding = "Evidence the system was given:" in prompt
        return json.dumps(
            {
                "correct": not abstained and not grounding,
                "grounded": grounding,
                "abstained": abstained,
                "reason": "test",
            }
        )

    report = judging.score_answers(
        "dense",
        queries,
        outcomes,
        {q.query_id: evidence for q in queries},
        {q.query_id: "Supported." for q in queries},
        complete,
        budget=5,
        workers=1,
    )
    assert len(prompts) == 4
    assert all("TRUNCATED" not in p for p in prompts)
    assert all("[6]" not in p for p in prompts)
    assert all("delete " not in p for p in prompts if "Reference answer:" in p)
    assert all("Reference answer:" not in p for p in prompts if "Evidence the system was given:" in p)
    assert report["questions"] == 2
    assert report["answered"] == 1
    assert report["correctness"] == 0.5
    assert report["pipeline_abstentions"] == 1
    assert report["judge_pipeline_abstention_disagreements"] == 0


def test_correctness_prompt_is_identical_when_only_retrieved_evidence_changes():
    query = Query(
        query_id="1::nda-15",
        doc_id="1",
        category="c",
        question="No rights granted?",
        answers=(AnswerSpan(text="No rights granted.", start=0, end=18),),
    )
    rows = {query.query_id: {"answer": "No rights granted.", "abstained": False}}
    gold = {query.query_id: "The hypothesis is supported: No rights granted."}
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        is_grounding = "Evidence the system was given:" in prompt
        return json.dumps(
            {
                "correct": not is_grounding,
                "grounded": is_grounding and "SOURCE_A" in prompt,
                "abstained": False,
                "reason": "semantic answer",
            }
        )

    reports = []
    for arm, text in (("dense", "SOURCE_A"), ("lightrag-mix", "SOURCE_B")):
        reports.append(
            judging.score_answers(
                arm,
                [query],
                rows,
                {query.query_id: [Evidence(text=text, provenance=Provenance(doc_id="1", chunk_id="0"))]},
                gold,
                complete,
                budget=5,
                workers=1,
            )
        )
    assert prompts[0] == prompts[2]
    assert "SOURCE_A" not in prompts[0] and "SOURCE_B" not in prompts[2]
    assert reports[0]["correctness"] == reports[1]["correctness"] == 1.0
    assert reports[0]["grounding"] == 1.0 and reports[1]["grounding"] == 0.0


def test_duplicate_outcomes_fail_before_judging():
    with pytest.raises(ValueError, match="IDs"):
        judging.validate_outcomes([{"query_id": "a"}, {"query_id": "a"}], [], {})


def test_invalid_response_retries_without_coercing_string_booleans():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        correct = "false" if len(prompts) == 1 else False
        return json.dumps({"correct": correct, "grounded": True, "abstained": False, "reason": "test"})

    assert json.loads(judging.checked_complete(complete)("original inputs"))["correct"] is False
    assert len(prompts) == 2 and prompts[1].startswith("original inputs")


def test_identical_answers_cannot_publish_conflicting_correctness():
    reports = [
        {"judgements": [{"query_id": "a", "correct": True}]},
        {"judgements": [{"query_id": "a", "correct": False}]},
    ]
    outcomes = {arm: {"a": {"answer": "No rights granted."}} for arm in ("dense", "lightrag-mix")}
    with pytest.raises(ValueError, match="inconsistent correctness"):
        judging.correctness_consistency(reports, outcomes)


def test_canonical_abstention_is_not_reclassified_by_an_unreliable_judge():
    query = Query(
        query_id="q",
        doc_id="d",
        category="c",
        question="Must delete?",
        answers=(AnswerSpan(text="delete", start=0, end=6),),
    )
    outcomes = {"q": {"answer": judging.ABSTENTION_TEXT, "abstained": True}}

    def complete(_):
        return json.dumps({"correct": True, "grounded": False, "abstained": False, "reason": "bad judge"})

    report = judging.score_answers(
        "dense", [query], outcomes, {"q": []}, {"q": "Must delete."}, complete, budget=5, workers=1
    )
    assert report["correctness"] == 0.0 and report["abstention"] == 1.0
    assert report["grounding"] == 1.0 and report["hallucination"] == 0.0
    assert report["judge_pipeline_abstention_disagreements"] == 1
    assert report["judgements"][0]["raw_correct"] is True
    assert report["judgements"][0]["raw_grounded"] is False


def test_judge_abstention_on_an_unflagged_refusal_is_not_scored_correct():
    query = Query(
        query_id="q",
        doc_id="d",
        category="c",
        question="Must delete?",
        answers=(AnswerSpan(text="delete", start=0, end=6),),
    )
    outcomes = {"q": {"answer": "The contract does not say.", "abstained": False}}
    evidence = {"q": [Evidence(text="Delete all copies.", provenance=Provenance(doc_id="d"))]}

    def complete(_):
        return json.dumps({"correct": True, "grounded": True, "abstained": True, "reason": "declines"})

    report = judging.score_answers(
        "dense", [query], outcomes, evidence, {"q": "Must delete."}, complete, budget=5, workers=1
    )
    assert report["correctness"] == 0.0
    assert report["judgements"][0]["raw_correct"] is True
