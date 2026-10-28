"""Paired records, failure accounting, corpus integrity, and rewrite contracts."""

import hashlib
import json
from dataclasses import replace

import pytest
from aieng.advanced_rag.corpora.natural_questions import NQPaths, load_nq, parse_nq
from aieng.advanced_rag.corpora.sources import ChecksumMismatchError
from export_results import compact_report, recorded_subset
from run_benchmark import experiment_spec, open_report
from run_evaluation import compare, replay
from synthesis_lab.corpora import Corpus, Question, load_rewrites
from synthesis_lab.decision_grid import decision_grid
from synthesis_lab.evaluation import paired_comparison, summarize
from synthesis_lab.tutorial import synthetic_corpus


def test_nq_parser_preserves_aliases_and_maps_support_text():
    corpus = json.dumps([{"title": "A", "text": "passage"}]).encode()
    raw = [
        {
            "question": "Who?",
            "reference": ["Ada", "A. Ada"],
            "contexts": [{"title": "A", "text": "passage", "is_supporting": True}],
        }
    ]
    p, q = parse_nq(json.dumps(raw).encode(), corpus)
    assert q[0].answers == ("Ada", "A. Ada")
    assert q[0].supporting_ids == (p[0].passage_id,)
    raw[0]["contexts"][0]["text"] = "missing"
    with pytest.raises(ValueError, match="missing"):
        parse_nq(json.dumps(raw).encode(), corpus)


def test_nq_loader_checks_explicit_paths(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]")
    with pytest.raises(ChecksumMismatchError):
        load_nq(NQPaths(path, path))


def test_failed_questions_count_and_incomplete_reports_cannot_score():
    questions = [
        Question("q", "Who?", ("Ada",), ("p",), "single-hop"),
        Question("q2", "Who?", ("Ada",), ("p",), "single-hop"),
    ]
    rows = [
        {
            "question_id": "q",
            "answer": "Ada",
            "abstained": False,
            "retrieved_ids": ["p"],
            "generation_ids": ["p"],
            "citation_ids": ["p"],
        },
        {"question_id": "q2", "failure": {"type": "outage"}},
    ]
    scores = summarize(questions, rows)["overall"]
    assert scores["questions"] == 2
    assert scores["metrics"]["exact_match"] == 0.5
    assert scores["metrics"]["recall_at_20"] == 0.5
    with pytest.raises(ValueError, match="exactly"):
        summarize(questions, rows[:1])
    comparison = paired_comparison(questions, rows, rows, samples=100)
    assert comparison["differences"]["token_f1"] == {"mean": 0, "low": 0, "high": 0}


def test_resume_rejects_changed_spec_and_duplicate_rows(tmp_path):
    path = tmp_path / "report.json"
    spec = {"questions": [{"id": "q", "text": "Who?"}]}
    report = open_report(path, spec)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs"):
        open_report(path, spec | {"prompt": "changed"})
    report["records"] = [{"question_id": "q"}, {"question_id": "q"}]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="coverage"):
        open_report(path, spec)


def test_rewrites_are_paired_and_keep_gold_separate(tmp_path):
    corpus = Corpus("musique", (), (Question("q", "Original", ("Ada",), ("p",), "2-hop"),), {})
    path = tmp_path / "rewrites.json"
    payload = {"version": "v1", "pairs": [{"question_id": "q", "plain": "Plain?", "obfuscated": "Hidden?"}]}
    path.write_text(json.dumps(payload))
    result = load_rewrites(path, corpus)
    assert [q.wording for q in result.questions] == ["plain", "obfuscated"]
    assert all(q.answers == ("Ada",) and q.original_id == "q" for q in result.questions)
    payload["pairs"].append(payload["pairs"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_rewrites(path, corpus)
    with pytest.raises(ValueError, match="exactly"):
        paired_comparison(result.questions, [], [])
    assert replace(result.questions[0], answers=("other",)).question == "Plain?"


def test_compact_export_replays_without_corpus_text():
    corpus, _, _ = synthetic_corpus()
    q = corpus.questions[0]
    spec = {
        "corpus": corpus.name,
        "provenance": corpus.provenance,
        "questions": [{"id": q.question_id, "text": q.question}],
    }
    rows = [
        {
            "question_id": q.question_id,
            "answer": "Oslo",
            "abstained": False,
            "retrieved_ids": list(q.supporting_ids),
            "generation_ids": list(q.supporting_ids),
            "citation_ids": list(q.supporting_ids),
            "trace": {"private": "source text"},
        }
    ]
    report = {
        "completed": True,
        "coverage_complete": True,
        "spec": spec,
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "records": rows,
        "scores": summarize([q], rows),
        "cost_note": "fixture",
    }
    report["workers"] = 3
    report["execution_wall_seconds"] = 12.5
    compact = compact_report(report, corpus)
    assert compact["workers"] == 3 and compact["execution_wall_seconds"] == 12.5
    assert "text" not in compact["spec"]["questions"][0]
    assert "trace" not in compact["records"][0]
    assert "answer" not in compact["records"][0]
    assert compact["records"][0]["token_f1"] == 1.0
    verified, _ = replay(compact, corpus)
    assert verified["score_replayed"]
    compact["records"][0]["token_f1"] = 0.0
    with pytest.raises(ValueError, match="metrics"):
        replay(compact, corpus)
    report["spec"]["unrecorded_prompt_change"] = True
    with pytest.raises(ValueError, match="specification digest"):
        compact_report(report, corpus)


def test_complete_coverage_keeps_failed_sessions_in_the_denominator():
    corpus, _, _ = synthetic_corpus()
    corpus = replace(corpus, name="musique")
    q = corpus.questions[0]
    spec = {
        "corpus": corpus.name,
        "method": "ircot",
        "provenance": corpus.provenance,
        "questions": [{"id": q.question_id, "text": q.question}],
    }
    rows = [{"question_id": q.question_id, "failure": {"type": "TimeoutError"}, "abstained": True}]
    report = {
        "completed": False,
        "coverage_complete": True,
        "spec": spec,
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "records": rows,
        "scores": summarize([q], rows),
        "cost_note": "fixture",
    }
    exported = compact_report(report, corpus)
    assert exported["completed"] and not exported["execution_success"]
    verified, _ = replay(exported, corpus)
    scores = verified["scores"]["overall"]
    assert scores["questions"] == 1
    assert scores["metrics"]["failure"] == 1 and scores["metrics"]["exact_match"] == 0
    assert (
        next(c for c in decision_grid([verified]) if c["method"] == "ircot" and c["complexity"] == "2-hop")["status"]
        == "measured_with_failures"
    )
    exported["records"] = []
    with pytest.raises(ValueError, match="exactly"):
        replay(exported, corpus)


def test_failed_partial_result_receives_no_generation_or_citation_credit():
    question = Question("q", "Who?", ("Ada",), ("p",), "2-hop")
    row = {
        "question_id": "q",
        "answer": "Ada",
        "abstained": False,
        "retrieved_ids": ["p"],
        "generation_ids": ["p"],
        "citation_ids": ["p"],
        "failure": {"type": "provider_error"},
    }
    metrics = summarize([question], [row])["overall"]["metrics"]
    assert metrics["failure"] == 1
    for key in (
        "exact_match",
        "token_f1",
        "recall_at_20",
        "generation_recall",
        "generation_all_supports",
        "answered_with_valid_citations",
    ):
        assert metrics[key] == 0


def test_response_cache_namespace_is_part_of_resume_identity(tmp_path):

    corpus, _, _ = synthetic_corpus()
    first = experiment_spec(corpus, "dense", response_cache=tmp_path / "first")
    second = experiment_spec(corpus, "dense", response_cache=tmp_path / "second")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(open_report(path, first)))
    with pytest.raises(ValueError, match="configuration differs"):
        open_report(path, second)
    assert "response_cache_namespace" not in experiment_spec(corpus, "dense")
    assert first["response_cache_namespace"] == "first"


def test_corrective_cohort_requires_all_three_controls_before_reporting():
    corpus, _, _ = synthetic_corpus()
    corpus = replace(corpus, name="musique")
    with pytest.raises(ValueError, match="requires one dense"):
        compare([], {"musique": corpus}, corrective_reports=[])


def test_corrective_supplement_keeps_its_own_controls_and_updates_grid(tmp_path):
    corpus, _, _ = synthetic_corpus()
    corpus = replace(corpus, name="musique")
    question = corpus.questions[0]
    row = {
        "question_id": question.question_id,
        "answer": "Oslo",
        "abstained": False,
        "retrieved_ids": list(question.supporting_ids),
        "generation_ids": list(question.supporting_ids),
        "citation_ids": list(question.supporting_ids),
    }

    def report(method, cache):
        spec = experiment_spec(corpus, method, response_cache=cache)
        value = open_report(tmp_path / (method + ".json"), spec)
        return value | {
            "completed": True,
            "coverage_complete": True,
            "records": [dict(row)],
            "scores": summarize(corpus.questions, [row]),
        }

    core = report("dense", None)
    cohort = [report(method, tmp_path / "cohort") for method in ("dense", "graded-single-pass", "corrective")]
    result = compare([core], {"musique": corpus}, corrective_reports=cohort)
    assert len(result["reports"]) == 1
    assert len(result["corrective_cohort"]["reports"]) == 3
    assert {c["control"] for c in result["corrective_cohort"]["comparisons"] if c["method"] == "corrective"} == {
        "dense",
        "graded-single-pass",
    }
    cell = next(c for c in result["grid"] if c["method"] == "corrective" and c["complexity"] == "2-hop")
    assert cell["status"] == "measured" and cell["comparison_set"] == "corrective_cohort"
    cohort[1]["spec"]["bounds"]["generation_passages"] = 4
    cohort[1]["spec_sha256"] = hashlib.sha256(json.dumps(cohort[1]["spec"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="unmatched inputs or answer settings"):
        compare([core], {"musique": corpus}, corrective_reports=cohort)
    cohort[1]["spec"]["bounds"]["generation_passages"] = 5
    cohort[1]["spec_sha256"] = hashlib.sha256(json.dumps(cohort[1]["spec"], sort_keys=True).encode()).hexdigest()
    cohort[-1]["spec"]["response_cache_namespace"] = "different"
    with pytest.raises(ValueError, match="namespace"):
        compare([core], {"musique": corpus}, corrective_reports=cohort)


def test_replay_rejects_duplicate_or_oversized_reader_evidence():
    corpus, _, _ = synthetic_corpus()
    question = corpus.questions[0]
    ids = [p.passage_id for p in corpus.passages[:6]]
    spec = {
        "corpus": corpus.name,
        "provenance": corpus.provenance,
        "questions": [{"id": question.question_id, "text": question.question}],
        "bounds": {"generation_passages": 5},
    }
    for generation in (ids, [ids[0], ids[0]]):
        row = {
            "question_id": question.question_id,
            "answer": "Oslo",
            "abstained": False,
            "retrieved_ids": ids,
            "generation_ids": generation,
            "citation_ids": [],
        }
        report = {
            "spec": spec,
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "coverage_complete": True,
            "records": [row],
            "scores": summarize(corpus.questions, [row]),
        }
        with pytest.raises(ValueError, match="generation"):
            replay(report, corpus)


def test_recorded_subset_preserves_parent_identity_and_does_not_fill_full_grid(tmp_path):
    corpus, _, _ = synthetic_corpus()
    first = corpus.questions[0]
    corpus = replace(corpus, name="musique", questions=(first, replace(first, question_id="extra")))
    row = {
        "question_id": first.question_id,
        "answer": "Oslo",
        "abstained": False,
        "retrieved_ids": list(first.supporting_ids),
        "generation_ids": list(first.supporting_ids),
        "citation_ids": list(first.supporting_ids),
    }
    reports = []
    for method in ("dense", "graded-single-pass", "corrective"):
        parent = open_report(
            tmp_path / (method + ".json"), experiment_spec(corpus, method, response_cache=tmp_path / "cache")
        )
        parent["records"] = [row]
        subset = compact_report(recorded_subset(parent, corpus, 1), corpus)
        assert subset["execution_spec_sha256"] == parent["spec_sha256"]
        assert subset["source_report_completed"] is False
        assert subset["selection"]["parent_selected_questions"] == 2
        assert subset["execution_wall_seconds"] is None
        reports.append(subset)
        with pytest.raises(ValueError, match="unfinished"):
            recorded_subset(parent, corpus, 2)
        assert parent["completed"] is False and len(parent["spec"]["questions"]) == 2
    result = compare([], {"musique": corpus}, corrective_reports=reports)
    assert result["corrective_cohort"]["full_corpus_coverage"] is False
    assert result["corrective_cohort"]["reports"][0]["scores"]["overall"]["questions"] == 1
    assert all(c["status"] == "pending" for c in result["grid"] if c["method"] == "corrective")
