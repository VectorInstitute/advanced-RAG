"""Validate and compare two saved null arms without retrieving or calling a model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.multihop import mcnemar_exact

from .common import digest, require_compute, write_json


ARMS = ("dense", "lightrag-mix")


def paired_decisions(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    """Count matched successes and compute the descriptive exact McNemar test."""
    if len(first) != len(second) or not first:
        raise ValueError("paired outcomes must be nonempty and equally sized")
    result = mcnemar_exact(first, second)
    return {
        "questions": len(first),
        "dense_only_correct": result.only_a,
        "lightrag_only_correct": result.only_b,
        "mcnemar_exact_two_sided_p": result.p_value,
        "inference_limit": "Descriptive paired test only: questions within a contract are correlated; this is not contract-level population inference.",
    }


def keyed_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reject duplicate identifiers before aligning two result files."""
    result = {row["query_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate query IDs in saved results")
    return result


def validate_pair(outcomes: Mapping[str, Mapping[str, dict[str, Any]]]) -> list[str]:
    """Require identical populations, document labels, and typed decisions."""
    dense, graph = (outcomes[arm] for arm in ARMS)
    if not dense or dense.keys() != graph.keys():
        raise ValueError("arms must cover the same nonempty query population")
    for query_id in dense:
        a, b = dense[query_id], graph[query_id]
        if (a["doc_id"], a["is_null"]) != (b["doc_id"], b["is_null"]):
            raise ValueError("paired document or answerability labels differ")
        if any(type(row[field]) is not bool for row in (a, b) for field in ("is_null", "abstained")):
            raise ValueError("answerability and abstention must be booleans")
    return sorted(dense)


def reader_budgets(store: Path, ids: Sequence[str], budget: int) -> dict[str, Any]:
    """Measure the source characters actually visible after generator truncation."""
    report = {}
    for arm in ARMS:
        path = store / "evidence" / f"{arm}.json"
        table = json.loads(path.read_text())
        if set(table) != set(ids):
            raise ValueError("evidence IDs differ from the measured population")
        sizes = [sum(min(1200, len(item["text"])) for item in table[q][:budget]) for q in ids]
        report[arm] = {
            "mean": sum(sizes) / len(sizes),
            "maximum": max(sizes),
            "passage_cap": budget,
            "characters_per_passage_cap": 1200,
            "evidence_sha256": digest(path),
        }
    return report


def verify_judging_sources(results: Path, store: Path, judging: Mapping[str, Any]) -> None:
    """Reject verdicts that were produced for different answers or evidence."""
    paths = {"null_arm_summary.json": results / "null_arm_summary.json"}
    for arm in ARMS:
        paths[f"outcomes_{arm}.json"] = results / f"outcomes_{arm}.json"
        paths[f"evidence/{arm}.json"] = store / "evidence" / f"{arm}.json"
        report = results / f"answer_judgements_{arm}.json"
        if digest(report) != judging["report_sha256"].get(report.name):
            raise ValueError(f"judging report hash mismatch: {report.name}")
    for role, path in paths.items():
        if digest(path) != judging["input_sha256"].get(role):
            raise ValueError(f"judging input hash mismatch: {role}")


def arm_measurements(outcomes: Mapping[str, dict[str, Any]], verdicts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Keep decision quality and answer correctness on their own denominators."""
    nulls = [row for row in outcomes.values() if row["is_null"]]
    answerables = [row for row in outcomes.values() if not row["is_null"]]
    if not nulls or not answerables:
        raise ValueError("balanced comparison requires both answerable and null questions")
    correct = sum(row["correct"] for row in verdicts.values())
    refused = sum(row["abstained"] for row in answerables)
    null_recall = sum(row["abstained"] for row in nulls) / len(nulls)
    return {
        "questions": len(outcomes),
        "answerable_queries": len(answerables),
        "null_queries": len(nulls),
        "decision_accuracy": sum(row["abstained"] == row["is_null"] for row in outcomes.values()) / len(outcomes),
        "balanced_accuracy": (null_recall + 1 - refused / len(answerables)) / 2,
        "null_false_answer_rate": 1 - null_recall,
        "answerable_abstentions": refused,
        "correct_count": correct,
        "answerable_correctness": correct / len(answerables),
        "answerable_grounding_including_abstentions": sum(row["grounded"] for row in verdicts.values())
        / len(answerables),
    }


def publish_memo(results: Path, report: Mapping[str, Any]) -> None:
    """Render only validated measurements; no model-generated interpretation."""
    corpus, pipeline = report["corpus"], report["pipeline"]
    lines = [
        f"# {corpus['name']}: matched null-arm comparison",
        "",
        f"The fixed leading {corpus['contracts']} contracts contain {corpus['queries']} questions: "
        f"{corpus['answerable_queries']} answerables and {corpus['null_queries']} nulls. "
        "This is a bounded, nonrandom pilot.",
        "",
        f"Generation uses `{report['model']}` and judging uses `{report['judge']['model']}`. "
        f"Both retrieval arms use contract scope, top-k {pipeline['top_k']}, "
        f"a {pipeline['evidence_budget']}-passage reader cap, and thresholds "
        f"{pipeline['sufficient']}/{pipeline['insufficient']}. "
        "Reader text is capped at 1,200 characters per passage. No answer rewriter is configured.",
        "",
        "| Arm | Balanced decision accuracy | False answers among nulls | Correct answerables | Answerable abstentions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = report["arms"][arm]
        lines.append(
            f"| {arm} | {row['balanced_accuracy']:.4f} | {row['null_false_answer_rate']:.4f} | "
            f"{row['correct_count']}/{row['answerable_queries']} ({row['answerable_correctness']:.4f}) | "
            f"{row['answerable_abstentions']}/{row['answerable_queries']} |"
        )
    lines += [
        "",
        "Both trivial decision controls score 0.5 balanced accuracy. Correctness "
        "includes every answerable question, including pipeline abstentions.",
        "",
    ]
    comparison = report["decision_comparison"]
    lines += [
        f"Decision discordances: dense alone is correct on {comparison['dense_only_correct']} "
        f"questions; LightRAG alone on {comparison['lightrag_only_correct']}. "
        f"The exact question-level McNemar p-value is {comparison['mcnemar_exact_two_sided_p']:.4f}. "
        "Questions within a contract are correlated, so this is descriptive rather than "
        "contract-level population inference.",
        "",
        "## Reader budgets and cost",
        "",
    ]
    for arm in ARMS:
        budget = report["reader_evidence_chars"][arm]
        lines.append(f"- {arm}: mean {budget['mean']:.1f} visible source characters; maximum {budget['maximum']}.")
    indexing = report.get("observed_index_cost_across_runs")
    if indexing:
        lines += [
            "",
            f"Observed indexing lower bound: {indexing['calls']:,} completed responses and "
            f"{indexing['total_tokens']:,} tokens. Interrupted in-flight request billing is unknown. "
            "The cumulative resume ledger is included once.",
        ]
    else:
        lines += [
            "",
            "This scoring artifact does not include a cumulative index ledger; missing "
            "indexing cost must not be read as zero.",
        ]
    query_cost = report.get("graph_query_cost", {})
    if query_cost.get("recovery"):
        lines += [
            "",
            f"Graph retrieval cost was recovered from {query_cost['calls']} unique completed "
            f"keyword responses ({query_cost['total_tokens']:,} tokens). Cache-hit counts, "
            "timing, and possible retry billing were not persisted; this is a lower bound.",
        ]
    if report.get("scope_restoration"):
        lines += [
            "",
            "LightRAG stripped whitespace from one source ID. Only the exact stripped "
            "identifier was restored within its per-contract workspace, preserving text "
            "and ranking. The original table hash and repair audit are retained.",
        ]
    lines += ["", "## Interpretation limits", ""]
    lines += [f"- {limit}" for limit in report["limitations"]]
    diagnostics = report.get("index_diagnostics")
    if diagnostics:
        lines += [
            f"- Index logs contain {diagnostics['format_warning_events']} format-warning events "
            f"and {diagnostics['zero_relation_extraction_events']} zero-relation extraction events. "
            "Events can repeat during recovery. These diagnostics limit conclusions about graph quality."
        ]
    lines += [
        "- Canonical abstentions are incorrect on answerables and vacuously grounded. "
        "Grounding is measured against the returned evidence, separately from reference correctness.",
        "",
        "These results describe the measured configuration and cohort. They do not establish "
        "a universal GraphRAG null-answer claim or a population ranking.",
        "",
        "Input hashes, exact counts, reader budgets, and index diagnostics are in "
        "[`paired_comparison.json`](paired_comparison.json). Separate correctness and grounding "
        "reports are in [`answer_judging_summary.json`](answer_judging_summary.json). "
        "Corpus credit is recorded in [`../ATTRIBUTION.md`](../ATTRIBUTION.md).",
        "",
    ]
    temporary = results / ".COMPARISON.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(results / "COMPARISON.md")


def analyse(
    results: Path, store: Path, *, resume_cost: Path | None = None, index_logs: Sequence[Path] = ()
) -> dict[str, Any]:
    """Publish a comparison only after both generation and judging are complete."""
    require_compute()
    source_paths = [results / "null_arm_summary.json", results / "answer_judging_summary.json"]
    summary = json.loads(source_paths[0].read_text())
    judging = json.loads(source_paths[1].read_text())
    verify_judging_sources(results, store, judging)
    if judging["generation_model"] != summary["model"]:
        raise ValueError("judging and generation model provenance disagree")
    outcomes, verdicts = {}, {}
    for arm in ARMS:
        outcome_path = results / f"outcomes_{arm}.json"
        judge_path = results / f"answer_judgements_{arm}.json"
        outcomes[arm] = keyed_rows(json.loads(outcome_path.read_text()))
        verdicts[arm] = keyed_rows(json.loads(judge_path.read_text())["judgements"])
        source_paths.extend([outcome_path, judge_path])
    ids = validate_pair(outcomes)
    answerables = [q for q in ids if not outcomes["dense"][q]["is_null"]]
    if len(ids) != summary["corpus"]["queries"] or len(answerables) != summary["corpus"]["answerable_queries"]:
        raise ValueError("saved populations differ from the corpus summary")
    for arm in ARMS:
        if set(verdicts[arm]) != set(answerables):
            raise ValueError("judging must include every answerable, including abstentions")
        if any(
            type(row[key]) is not bool for row in verdicts[arm].values() for key in ("correct", "grounded", "abstained")
        ):
            raise ValueError("judging verdicts must be booleans")
        if any(verdicts[arm][q]["abstained"] != outcomes[arm][q]["abstained"] for q in answerables):
            raise ValueError("judging changed the saved canonical abstention decision")
    report = {
        "corpus": summary["corpus"],
        "model": summary["model"],
        "judge": judging["judge"],
        "embedding_controls": {
            "model": summary["embedding_model"],
            "dense_queries": "encode_queries with BGE's published query prompt",
            "lightrag_callback": "encode for all embedding requests; the shared encode_queries method is not used",
        },
        "pipeline": summary["pipeline"],
        "arms": {arm: arm_measurements(outcomes[arm], verdicts[arm]) for arm in ARMS},
        "decision_comparison": paired_decisions(
            *[[outcomes[arm][q]["abstained"] == outcomes[arm][q]["is_null"] for q in ids] for arm in ARMS]
        ),
        "answer_correctness_comparison": paired_decisions(
            *[[verdicts[arm][q]["correct"] for q in answerables] for arm in ARMS]
        ),
        "retrieved_evidence_chars_before_truncation": {
            arm: {
                "mean": sum(row["evidence_chars"] for row in outcomes[arm].values()) / len(ids),
                "maximum": max(row["evidence_chars"] for row in outcomes[arm].values()),
            }
            for arm in ARMS
        },
        "reader_evidence_chars": reader_budgets(store, ids, summary["pipeline"]["evidence_budget"]),
        "input_sha256": {path.name: digest(path) for path in source_paths},
        "limitations": [
            "Fixed leading contracts, not a random or full-corpus sample.",
            "Same-model generation and judging can share errors; annotated null labels may omit valid answers.",
            "Equal passage caps do not guarantee equal realized text lengths; both arm budgets are reported.",
            "Embedding query conventions differ between adapters; this compares implemented configurations and does not isolate graph structure alone.",
        ],
    }
    report["graph_query_cost"] = summary["cost"].get("lightrag_query", {})
    restoration = results / "scope_restoration.json"
    if restoration.exists():
        audit = json.loads(restoration.read_text())
        if audit["evidence_sha256"] != digest(store / "evidence/lightrag-mix.json"):
            raise ValueError("scope restoration audit does not match the graph table")
        report["scope_restoration"] = audit
        report["input_sha256"][restoration.name] = digest(restoration)
    if resume_cost is not None:
        report["prior_index_run"] = json.loads(resume_cost.read_text())
        report["input_sha256"][resume_cost.name] = digest(resume_cost)
        totals = summary["cost"]["lightrag_index"]["totals"]
        report["observed_index_cost_across_runs"] = {
            "calls": totals["calls"] + report["prior_index_run"]["observed_completed_calls"],
            "total_tokens": totals["total_tokens"] + report["prior_index_run"]["total_tokens"],
            "scope": "lower bound: prior unique cached completions plus new metered responses; cache rows do not record repeated calls or interrupted request billing",
        }
    if index_logs:
        lines = [line for path in index_logs for line in path.read_text().splitlines()]
        report["index_diagnostics"] = {
            "format_warning_events": sum("LLM output format error" in line for line in lines),
            "zero_relation_extraction_events": sum(" Rel " in line and " + 0 Rel " in line for line in lines),
            "log_sha256": {path.name: digest(path) for path in index_logs},
            "interpretation": "Warning events can repeat during retries. Malformed graph records limit conclusions about the method; a processed document status does not establish extraction quality.",
        }
    write_json(results / "paired_comparison.json", report)
    publish_memo(results, report)
    return report


def main() -> None:
    """Compare saved arms with no new inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--resume-cost", type=Path)
    parser.add_argument("--index-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(analyse(args.results, args.store, resume_cost=args.resume_cost, index_logs=args.index_log), indent=2)
    )


if __name__ == "__main__":
    main()
