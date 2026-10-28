"""Export replayable predictions without corpus text or full agent traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from synthesis_lab.corpora import Corpus, Question, load_corpus
from synthesis_lab.evaluation import question_scores, summarize
from synthesis_lab.runtime import DEFAULT_CACHE, write_json


def recorded_subset(report: dict[str, Any], corpus: Corpus, limit: int) -> dict[str, Any]:
    """Project completed outcomes onto a pinned source-order slice without new calls.

    The parent checkpoint can be incomplete. Every selected question must already
    have exactly one outcome; the parent execution identity and coverage survive.
    Batch wall time cannot be attributed to a subset and is left unavailable.
    """
    if limit < 1 or limit > len(corpus.questions):
        raise ValueError("subset size must be within the pinned corpus")
    digest = hashlib.sha256(json.dumps(report["spec"], sort_keys=True).encode()).hexdigest()
    if digest != report["spec_sha256"]:
        raise ValueError("report specification digest differs")
    spec = report["spec"]
    if spec["corpus"] != corpus.name or spec["provenance"] != corpus.provenance:
        raise ValueError("subset corpus provenance differs")
    questions = corpus.questions[:limit]
    selected = [{"id": q.question_id, "text": q.question} for q in questions]
    parent_questions = {q["id"]: q for q in spec["questions"]}
    if len(parent_questions) != len(spec["questions"]) or any(parent_questions.get(q["id"]) != q for q in selected):
        raise ValueError("subset questions differ from the pinned input")
    records = {row["question_id"]: row for row in report["records"]}
    if len(records) != len(report["records"]) or not set(records) <= set(parent_questions):
        raise ValueError("invalid parent record coverage")
    if not {q.question_id for q in questions} <= set(records):
        raise ValueError("subset contains unfinished questions")
    rows = [records[q.question_id] for q in questions]
    selection = {
        "rule": "first questions in pinned corpus order",
        "questions": limit,
        "parent_selected_questions": len(spec["questions"]),
        "parent_recorded_questions": len(records),
        "parent_completed": bool(report.get("completed")),
        "parent_spec_sha256": digest,
        "parent_checkpoint_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "note": "Subset chosen after execution began; extra outcomes remain local. Not a random full-split estimate.",
    }
    subset_spec = spec | {"questions": selected, "recorded_selection": selection}
    return report | {
        "spec": subset_spec,
        "spec_sha256": hashlib.sha256(json.dumps(subset_spec, sort_keys=True).encode()).hexdigest(),
        "execution_spec_sha256": report.get("execution_spec_sha256", digest),
        "selection": selection,
        "source_report_completed": report.get("completed"),
        "completed": True,
        "coverage_complete": True,
        "execution_wall_seconds": None,
        "records": rows,
        "scores": summarize(questions, rows),
    }


def compact_report(report: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    """Retain scoring evidence and actual usage while hashing source question text.

    Each record's answer text is replaced by its exact match and token F1
    against ``corpus``, so the export carries no model answers.
    """
    if not report.get("coverage_complete") or "scores" not in report:
        raise ValueError("only finalized, fully covered runs can be exported")
    digest = hashlib.sha256(json.dumps(report["spec"], sort_keys=True).encode()).hexdigest()
    if digest != report["spec_sha256"]:
        raise ValueError("report specification digest differs")
    expected = {q["id"] for q in report["spec"]["questions"]}
    actual = [r["question_id"] for r in report["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("export requires exactly one outcome per selected question")
    spec = dict(report["spec"])
    spec["questions"] = [
        {"id": q["id"], "text_sha256": hashlib.sha256(q["text"].encode()).hexdigest()} for q in spec["questions"]
    ]
    spec["question_text_encoding"] = "sha256"
    if report["spec"]["corpus"] != corpus.name:
        raise ValueError("export corpus differs from the report")
    annotations = {q.question_id: q for q in corpus.questions}
    fields = {
        "question_id",
        "abstained",
        "retrieved_ids",
        "generation_ids",
        "citation_ids",
        "dropped_citations",
        "stop_reason",
        "failure",
        "usage",
        "elapsed_seconds",
    }
    return {
        "spec": spec,
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "execution_spec_sha256": report.get("execution_spec_sha256", report["spec_sha256"]),
        "selection": report.get("selection"),
        "completed": True,
        "coverage_complete": True,
        "execution_success": not any(r.get("failure") for r in report["records"]),
        "source_report_completed": report.get("source_report_completed", report.get("completed")),
        "workers": report.get("workers"),
        "execution_wall_seconds": report.get("execution_wall_seconds"),
        "scores": report["scores"],
        "records": [
            {k: v for k, v in row.items() if k in fields} | _answer_scores(annotations[row["question_id"]], row)
            for row in report["records"]
        ],
        "cost_note": report["cost_note"],
        "export_note": (
            "Question text is hashed and answers are replaced by their scores; original passages "
            "and agent traces remain in the local cache."
        ),
        "index_build_cost": report.get("index_build_cost"),
    }


def _answer_scores(question: Question, row: dict[str, Any]) -> dict[str, float]:
    scores = question_scores(question, row)
    return {"exact_match": scores["exact_match"], "token_f1": scores["token_f1"]}


def main() -> None:
    """Write a compact artifact suitable for independent scorer replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-questions", type=int, help="Export only this many completed source-order questions")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_CACHE, help="Pinned corpus cache for subset validation"
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    name = report["spec"]["corpus"]
    corpus = load_corpus(name, args.data_dir if name == "musique" else args.data_dir.parent / "ri2-complete" / name)
    if args.first_questions is not None:
        report = recorded_subset(report, corpus, args.first_questions)
    write_json(args.output, compact_report(report, corpus))


if __name__ == "__main__":
    main()
