"""Audit complete MuSiQue arms, replay raw answers, and compare paired outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_CORPUS,
    MUSIQUE_QUESTIONS,
    MUSIQUE_REVISION,
    MusiqueQuestion,
    Passage,
    fetch_musique,
    load_musique,
)
from aieng.advanced_rag.evaluation.multihop import exact_match, mcnemar_exact, normalise_answer, token_f1
from aieng.advanced_rag.graphrag.proxy import ResponseCache
from aieng.advanced_rag.graphrag.synthesis import build_prompt
from aieng.advanced_rag.pipeline.generate import parse_answer
from synthesis_lab.decomposition import DECOMPOSITION_TEMPLATE, pool_rankings
from synthesis_lab.dense_baseline import ANSWER_TEMPLATE, DensePassageRetriever, passage_text, score_by_hops
from synthesis_lab.reference import reference_dense_predictions
from synthesis_lab.runtime import ANSWER_MODEL, DEFAULT_CACHE, PROXY_URL, write_json

from benchmarks.judge_full_answers import JUDGE_MODEL, JUDGE_PROMPT, load_arm, parse_judge


LAB = Path(__file__).resolve().parents[1]
OUTPUT = LAB / "outputs/full_evaluation"
SEED = 20261009
BOOTSTRAPS = 20_000


def paired_interval(a: Sequence[float], b: Sequence[float], samples: int = BOOTSTRAPS) -> dict[str, float]:
    """Return a deterministic paired percentile interval for a minus b."""
    if not a or len(a) != len(b) or samples < 1:
        raise ValueError("paired scores must be nonempty, equal length, with positive resamples")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not np.isfinite(diff).all():
        raise ValueError("scores must be finite")
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, samples, 250):
        indices = rng.integers(0, len(diff), size=(min(250, samples - start), len(diff)))
        draws.append(diff[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975])
    return {"difference": float(diff.mean()), "low": float(low), "high": float(high), "confidence": 0.95}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Control the family-wise error rate across the overall and hop-count EM tests."""
    adjusted = {}
    floor = 0.0
    for rank, (label, p_value) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        floor = max(floor, min(1.0, p_value * (len(p_values) - rank)))
        adjusted[label] = floor
    return adjusted


def validate_records(
    records: Mapping[str, dict[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    questions: Sequence[MusiqueQuestion],
    passages: Sequence[Passage],
) -> None:
    """Reject missing IDs, wrong context, unshown citations, or bad replay."""
    expected = {q.question_id for q in questions}
    if set(records) != expected or set(rankings) != expected:
        raise ValueError("arm must cover exactly the full question-ID set")
    valid_ids = {p.passage_id for p in passages}
    for q in questions:
        row = records[q.question_id]
        result = row["result"]
        ranked = list(rankings[q.question_id])
        if len(ranked) != 20 or len(set(ranked)) != 20 or not set(ranked) <= valid_ids:
            raise ValueError("ranking must contain 20 unique corpus IDs")
        if result["query_id"] != q.question_id or result["question"] != q.question:
            raise ValueError("prediction identity differs from pinned question")
        spine_ranking = ranked if "pooled_ids" in row else ranked[:5]
        if row["generation_ids"] != ranked[:5] or result["retrieved_ids"] != spine_ranking:
            raise ValueError("recorded generation context differs from ranked top five")
        if not set(result["chunk_ids"]) <= set(ranked[:5]):
            raise ValueError("answer cites a passage outside its generation context")
        if "pooled_ids" in row:
            if row["pooled_ids"] != pool_rankings(row["step_rankings"], 20):
                raise ValueError("pooled ranking differs from step rankings")
            if row["plan"]["question"] != q.question or q.question not in row["retrieval_queries"]:
                raise ValueError("original question was lost")
        evidence = DensePassageRetriever(passages, {q.question: ranked})(q.question, k=5)
        replay = parse_answer(q.question_id, row["raw_answer"], evidence)
        for field, value in replay.to_dict().items():
            if result.get(field) != value:
                raise ValueError(f"raw answer replay differs at {q.question_id}: {field}")


def question_metrics(q: MusiqueQuestion, row: dict[str, Any], ranking: Sequence[str]) -> dict[str, Any]:
    """Compute unrounded answer, context, citation, and dependency coverage metrics."""
    result = row["result"]
    prediction = "" if result["abstained"] else result["answer"]
    gold = set(q.supporting_ids)
    context = set(ranking[:5])
    cited = set(result["chunk_ids"])
    em = 0.0 if result["abstained"] else exact_match(prediction, (q.answer, *q.answer_aliases))
    f1 = 0.0 if result["abstained"] else token_f1(prediction, (q.answer, *q.answer_aliases))
    complete = gold <= context
    nodes = {node.step_id: node.support_passage_id for node in q.hop_graph.nodes}
    edges = q.hop_graph.edges
    metrics: dict[str, Any] = {
        "em": em,
        "f1": f1,
        "abstained": float(result["abstained"]),
        "joint_em_complete_context": float(bool(em) and complete),
        "citation_gold_recall": len(cited & gold) / len(gold),
        "citation_gold_precision": len(cited & gold) / len(cited) if cited else None,
        "citation_all_gold": float(gold <= cited),
        "context_dependency_edge_coverage": mean(float(nodes[a] in context and nodes[b] in context) for a, b in edges)
        if edges
        else None,
        "error_category": ("abstained" if result["abstained"] else "correct" if em else "incorrect")
        + ("_complete_context" if complete else "_incomplete_context"),
        "abstention_reason": result["reason"] if result["abstained"] else "",
        "dropped_citations": len(result["dropped_citations"]),
    }
    for k in (5, 10, 20):
        retrieved = set(ranking[:k])
        metrics[f"recall_at_{k}"] = len(gold & retrieved) / len(gold)
        metrics[f"all_supports_at_{k}"] = float(gold <= retrieved)
    return metrics


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate with explicit denominators, preserving unanswered questions."""
    numeric = [key for key, value in rows[0].items() if isinstance(value, (float, int)) or value is None]
    report: dict[str, Any] = {"questions": len(rows)}
    for key in numeric:
        values = [row[key] for row in rows if row[key] is not None]
        report[key] = {"mean": mean(values) if values else None, "denominator": len(values)}
    attempted = [row for row in rows if not row["abstained"]]
    report["attempted"] = len(attempted)
    report["em_among_attempted"] = mean(row["em"] for row in attempted) if attempted else None
    report["error_categories"] = dict(Counter(row["error_category"] for row in rows))
    report["abstention_reasons"] = dict(Counter(row["abstention_reason"] for row in rows if row["abstained"]))
    return report


def validate_prompt_cache(
    records: Mapping[str, dict[str, Any]],
    questions: Sequence[MusiqueQuestion],
    passages: Sequence[Passage],
    database: Path,
) -> None:
    """Rebuild exact prompts from corpus text and verify cached raw responses."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        for q in questions:
            row = records[q.question_id]
            evidence = DensePassageRetriever(passages, {q.question: row["generation_ids"]})(q.question, k=5)
            prompts = [(build_prompt(q.question, evidence, ANSWER_TEMPLATE, None), row["raw_answer"])]
            if "raw_plan" in row:
                prompt = DECOMPOSITION_TEMPLATE.format(question=q.question)
                prompts.insert(0, (prompt, row["raw_plan"]))
            requests = row["completion_requests"]
            if len(requests) != len(prompts):
                raise ValueError("unexpected logical completion count")
            for (prompt, raw), request in zip(prompts, requests, strict=True):
                expected = ResponseCache.key(
                    {
                        "prompt": prompt,
                        "model": ANSWER_MODEL,
                        "base_url": PROXY_URL,
                        "temperature": 0,
                        "max_tokens": 256,
                        "thinking": False,
                    }
                )
                if expected != request["cache_key"]:
                    raise ValueError(f"prompt provenance differs: {q.question_id}")
                stored = connection.execute("SELECT payload FROM responses WHERE key=?", (expected,)).fetchone()
                if stored is None or stored[0] != raw:
                    raise ValueError(f"raw response differs from cache: {q.question_id}")


def request_cost(records: Mapping[str, dict[str, Any]], database: Path) -> dict[str, Any]:
    """Recover successful response tokens, including cache hits."""
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    logical = [req for row in records.values() for req in row["completion_requests"]]
    token_rows = []
    for request in logical:
        stored = connection.execute(
            "SELECT prompt_tokens,completion_tokens FROM responses WHERE key=?", (request["cache_key"],)
        ).fetchone()
        if stored is None or any(value is None for value in stored):
            raise ValueError("response token provenance missing from cache")
        token_rows.append(stored)
    connection.close()
    return {
        "logical_completions": len(logical),
        "distinct_cached_requests": len({r["cache_key"] for r in logical}),
        "replay_cache_hits": sum(r["cache_hit"] for r in logical),
        "successful_response_prompt_tokens": sum(r[0] for r in token_rows),
        "successful_response_completion_tokens": sum(r[1] for r in token_rows),
        "successful_response_total_tokens": sum(sum(r) for r in token_rows),
        "note": "Logical-request work including cache hits; not a bill or a claim about uncached latency. Retry costs are only in observed usage.",
    }


def validate_judge_cache(
    judged: Mapping[str, dict[str, Any]],
    source: Mapping[str, dict[str, Any]],
    questions: Sequence[MusiqueQuestion],
    passages: Sequence[Passage],
    database: Path,
    *,
    configuration: dict[str, Any] | None = None,
) -> None:
    """Check that every secondary verdict concerns the actual answer and context."""
    by_id = {p.passage_id: p for p in passages}
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        for q in questions:
            verdict, row = judged[q.question_id], source[q.question_id]
            if row["result"]["abstained"]:
                if verdict["correct"] or verdict["completion_requests"] or verdict["raw_judgement"] is not None:
                    raise ValueError("abstention judge contract differs")
                continue
            parsed = parse_judge(verdict["raw_judgement"])
            if any(parsed[field] != verdict[field] for field in parsed):
                raise ValueError("raw judgement replay differs")
            cited = set(row["result"]["chunk_ids"])
            evidence = "\n\n".join(
                f"ID={pid}; cited={str(pid in cited).lower()}\n{passage_text(by_id[pid])}"
                for pid in row["generation_ids"]
            )
            prompt = JUDGE_PROMPT.format(
                question=q.question,
                references=json.dumps([q.answer, *q.answer_aliases]),
                answer=row["result"]["answer"],
                passages=evidence,
            )
            request = {
                "prompt": prompt,
                "model": JUDGE_MODEL,
                "base_url": PROXY_URL,
                "temperature": 0,
                "max_tokens": 2048,
                "thinking": False,
                "reasoning_effort": "low",
            }
            if configuration is not None:
                request["model"] = configuration["judge_model"]
                request["base_url"] = configuration["base_url"]
                request["request_params"] = configuration["request_params"]
                request.pop("reasoning_effort")
            key = ResponseCache.key(request)
            requests = verdict["completion_requests"]
            if len(requests) != 1 or requests[0]["cache_key"] != key:
                raise ValueError("judge prompt provenance differs")
            stored = connection.execute("SELECT payload FROM responses WHERE key=?", (key,)).fetchone()
            if stored is None or stored[0] != verdict["raw_judgement"]:
                raise ValueError("judge raw response differs from cache")


def run_evaluation(cache_dir: Path) -> dict[str, Any]:  # noqa: PLR0912, PLR0915 - explicit audit and report sequence
    """Audit full-split predictions and save the paired comparison."""
    passages, questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    if len(questions) != 1000 or len(passages) != 11656:
        raise ValueError("unexpected corpus size")
    official = json.loads((OUTPUT / "official_scorer_audit.json").read_text())
    retrieval_audit = json.loads((OUTPUT / "retrieval_audit.json").read_text())
    if (
        official["checked_predictions"] != {"dense": 1000, "decomposition": 1000}
        or official["em_mismatches"]
        or official["f1_mismatches"]
    ):
        raise ValueError("official score audit incomplete")
    if (
        retrieval_audit["dense_questions"] != 1000
        or retrieval_audit["decomposition_questions"] != 1000
        or retrieval_audit["ranking_mismatches"]
    ):
        raise ValueError("local retrieval audit incomplete")
    dense = load_arm("dense")
    multi = load_arm("decomposition")
    dense_runs = json.loads((OUTPUT / "dense_replay/rankings.json").read_text())
    multi_runs = {qid: row["pooled_ids"] for qid, row in multi.items()}
    validate_records(dense, dense_runs, questions, passages)
    validate_records(multi, multi_runs, questions, passages)
    for records in (dense, multi):
        validate_prompt_cache(records, questions, passages, cache_dir / "answers.sqlite")
    original = {row["question_id"]: row for row in reference_dense_predictions()}
    for qid, row in dense.items():
        result = row["result"]
        if (result["answer"], result["abstained"], result["chunk_ids"]) != (
            original[qid]["answer"],
            original[qid]["abstained"],
            original[qid]["citation_ids"],
        ):
            raise ValueError("dense replay differs from original control")
    multi_report = json.loads((LAB / "outputs/decomposition/full.json").read_text())
    dense_report = json.loads((OUTPUT / "dense_replay/answers.json").read_text())
    dense_retrieval = json.loads((OUTPUT / "dense_replay/retrieval.json").read_text())
    if multi_report["embeddings"]["identity"] != dense_retrieval["embeddings"]["identity"]:
        raise ValueError("retrieval substrate differs between arms")
    if multi_report["embeddings"]["sha256"] != dense_retrieval["embeddings"]["sha256"]:
        raise ValueError("embedding vectors differ between arms")
    for field in ("answer_model", "base_url", "evidence_budget", "temperature", "thinking", "max_attempts"):
        if field == "max_attempts":
            continue  # Dense metadata predates this field; every trace is checked below.
        if dense_report[field] != multi_report[field]:
            raise ValueError(f"generation configuration differs: {field}")
    pairs: list[dict[str, Any]] = []
    for q in questions:
        pair: dict[str, Any] = {"question_id": q.question_id, "hops": q.hops}
        for arm, rows, runs in (("dense", dense, dense_runs), ("decomposition", multi, multi_runs)):
            pair[arm] = question_metrics(q, rows[q.question_id], runs[q.question_id])
            if len(rows[q.question_id]["result"]["attempts"]) != 1:
                raise ValueError("unexpected corrective retry")
        pairs.append(pair)
    groups = {"overall": pairs, **{str(h): [row for row in pairs if row["hops"] == h] for h in (2, 3, 4)}}
    comparisons: dict[str, Any] = {}
    for label, selected in groups.items():
        arm_rows = {arm: [row[arm] for row in selected] for arm in ("dense", "decomposition")}
        comparison = {arm: aggregate(rows) for arm, rows in arm_rows.items()}
        comparison["paired_differences"] = {
            metric: paired_interval(
                [r[metric] for r in arm_rows["decomposition"]], [r[metric] for r in arm_rows["dense"]]
            )
            for metric in ("em", "f1", "recall_at_5", "all_supports_at_5", "abstained", "joint_em_complete_context")
        }
        comparison["em_mcnemar"] = asdict(
            mcnemar_exact(
                [bool(r["em"]) for r in arm_rows["decomposition"]], [bool(r["em"]) for r in arm_rows["dense"]]
            )
        )
        comparisons[label] = comparison
    adjusted = holm_adjust({label: group["em_mcnemar"]["p_value"] for label, group in comparisons.items()})
    for label, p_value in adjusted.items():
        comparisons[label]["em_mcnemar"]["holm_adjusted_p_value"] = p_value
    for rows, runs, stored in (
        (multi, multi_runs, multi_report["metrics"]),
        (dense, dense_runs, multi_report["matched_dense_baseline"]),
    ):
        predictions = {qid: "" if row["result"]["abstained"] else row["result"]["answer"] for qid, row in rows.items()}
        if score_by_hops(questions, runs, [5, 10, 20], predictions) != stored:
            raise ValueError("stored scores do not replay")
    report: dict[str, Any] = {
        "completed": True,
        "corpus_revision": MUSIQUE_REVISION,
        "questions_sha256": MUSIQUE_QUESTIONS.sha256,
        "corpus_sha256": MUSIQUE_CORPUS.sha256,
        "questions": len(questions),
        "passages": len(passages),
        "hop_counts": dict(Counter(q.hops for q in questions)),
        "duplicate_question_texts": len(questions) - len({q.question for q in questions}),
        "empty_normalised_reference_ids": [
            q.question_id for q in questions if any(not normalise_answer(g) for g in (q.answer, *q.answer_aliases))
        ],
        "primary": "Paired exact match; F1 and retrieval metrics are secondary",
        "statistics": {
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
            "unit": "question ID",
            "intervals": "95% paired percentile; descriptive secondary intervals, not multiplicity adjusted",
            "em_tests": "Exact two-sided McNemar; Holm correction across overall and three hop groups",
        },
        "audit": {
            "exact_id_coverage": True,
            "original_dense_predictions_reproduced": True,
            "same_embedding_cache": True,
            "same_generation_settings": True,
            "raw_answer_and_citation_replay": True,
            "exact_generation_context": True,
            "exact_prompt_and_raw_cache_provenance": True,
            "official_scorer": official,
            "local_retrieval_replay": retrieval_audit,
        },
        "groups": comparisons,
        "plan_outcomes": dict(Counter(row["plan_outcome"] for row in multi.values())),
        "request_work": {
            arm: request_cost(rows, cache_dir / "answers.sqlite")
            for arm, rows in (("dense", dense), ("decomposition", multi))
        },
        "observed_usage": {"dense_replay": dense_report["usage"]["run"], "decomposition": multi_report["usage"]["run"]},
        "attribution": multi_report["attribution"],
        "limits": [
            "Support-set coverage is not proof of entailment; non-gold passages can contain valid alternative evidence.",
            "Decomposition and pooling change together; this comparison cannot isolate their effects.",
            "Warm cache and rate limits confound observed latency; no uncached latency claim.",
            "This is HippoRAG 2 packaging's 1000-question subset, not all original MuSiQue splits.",
            "Question bootstrap intervals do not measure decoding variability; the proxy model alias is not an immutable weight revision.",
            "The independent model judge is an automatic secondary audit, not human adjudication.",
        ],
    }
    report["secondary_judge"] = {}
    for arm in ("dense", "decomposition"):
        judgement = json.loads((OUTPUT / f"judgements_{arm}.json").read_text())
        judged = {row["question_id"]: row for row in judgement["records"]}
        if (
            not judgement["completed"]
            or len(judged) != len(judgement["records"])
            or set(judged) != {q.question_id for q in questions}
        ):
            raise ValueError("secondary judge coverage incomplete")
        for q in questions:
            row = judged[q.question_id]
            source = (dense if arm == "dense" else multi)[q.question_id]
            if row["abstained"] != source["result"]["abstained"] or row["hops"] != q.hops:
                raise ValueError("judge prediction alignment differs")
        summaries = {}
        if "model_roles" not in judgement and (
            judgement["judge_model"],
            judgement["base_url"],
            judgement["max_tokens"],
            judgement["reasoning_effort"],
        ) != (JUDGE_MODEL, PROXY_URL, 2048, "low"):
            raise ValueError("judge settings differ")
        validate_judge_cache(
            judged,
            dense if arm == "dense" else multi,
            questions,
            passages,
            cache_dir / "answers.sqlite",
            configuration=judgement if "model_roles" in judgement else None,
        )
        for label, selected in groups.items():
            judged_rows = [judged[pair["question_id"]] for pair in selected]
            attempted = [row for row in judged_rows if not row["abstained"]]
            summaries[label] = {
                "questions": len(judged_rows),
                "attempted": len(attempted),
                "judge_correct_label_rate": mean(float(row["correct"]) for row in judged_rows),
                "context_supported_label_rate_among_attempted": mean(
                    float(row["context_supported"]) for row in attempted
                )
                if attempted
                else None,
                "citations_supported_label_rate_among_attempted": mean(
                    float(row["citations_supported"]) for row in attempted
                )
                if attempted
                else None,
                "joint_correct_and_citation_supported_label_rate": mean(
                    float(row["correct"] and row["citations_supported"]) for row in judged_rows
                ),
                "inconsistent_support_labels": sum(
                    row["citations_supported"] is True and row["context_supported"] is False for row in judged_rows
                ),
                "exact_match_false_judge_correct": sum(
                    not pair[arm]["em"] and judged[pair["question_id"]]["correct"] for pair in selected
                ),
                "exact_match_true_judge_incorrect": sum(
                    bool(pair[arm]["em"]) and not judged[pair["question_id"]]["correct"] for pair in selected
                ),
            }
            for pair in selected:
                pair[arm]["judge_correct"] = float(judged[pair["question_id"]]["correct"])
        report["secondary_judge"][arm] = {
            "model": judgement["judge_model"],
            "groups": summaries,
            "usage": judgement["usage"]["run"],
            "request_work": request_cost(judged, cache_dir / "answers.sqlite"),
            "exact_prompt_cache_and_raw_verdict_replay": True,
        }
    report["secondary_judge"]["paired_judge_correct_labels"] = paired_interval(
        [row["decomposition"]["judge_correct"] for row in pairs], [row["dense"]["judge_correct"] for row in pairs]
    )
    spot_check = json.loads((OUTPUT / "judge_spot_check.json").read_text())
    dense_judged = {r["question_id"]: r for r in json.loads((OUTPUT / "judgements_dense.json").read_text())["records"]}
    references = {q.question_id: [q.answer, *q.answer_aliases] for q in questions}
    for example in spot_check["records"]:
        qid = example["question_id"]
        expected_references = example.get("references", [example.get("reference")])
        if (
            dense[qid]["result"]["answer"] != example["prediction"]
            or dense_judged[qid]["correct"] != example["judge_correct"]
            or references[qid] != expected_references
        ):
            raise ValueError("judge spot check is stale; review the new predictions")
    report["secondary_judge"]["interpretation"] = {
        "status": "exploratory_only_failed_spot_check",
        "reason": "The judge accepted mid-3rd century BC against reference 1930 in dense question 4hop2__9988_261673_70784_79935. Raw annotations are retained; model-judged rates are not validated semantic accuracy.",
        "primary_scores_affected": False,
        "review_file": "outputs/full_evaluation/judge_spot_check.json",
    }
    write_json(OUTPUT / "paired_questions.json", pairs)
    write_json(OUTPUT / "comparison.json", report)
    # Hash all inputs and outputs that determine this comparison.
    files = [
        LAB / "outputs/decomposition/full.json",
        OUTPUT / "dense_replay/traces.json",
        OUTPUT / "dense_replay/rankings.json",
        OUTPUT / "judgements_dense.json",
        OUTPUT / "judgements_decomposition.json",
        OUTPUT / "paired_questions.json",
        OUTPUT / "comparison.json",
        OUTPUT / "official_scorer_audit.json",
        OUTPUT / "retrieval_audit.json",
        OUTPUT / "dense_replay/answers.json",
        OUTPUT / "dense_replay/retrieval.json",
        LAB / "benchmarks/full_evaluation.py",
        LAB / "benchmarks/judge_full_answers.py",
        LAB / "benchmarks/audit_official_scorer.py",
        LAB / "benchmarks/audit_full_retrieval.py",
        LAB / "synthesis_lab/dense_baseline.py",
        LAB / "synthesis_lab/decomposition.py",
        LAB / "run_dense_baseline.py",
        LAB / "run_decomposition.py",
        OUTPUT / "judge_spot_check.json",
    ]
    write_json(
        OUTPUT / "manifest.json", {str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    )
    return report


def main() -> None:
    """Run the deterministic comparison audit on compute; no model calls occur here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    report = run_evaluation(args.cache_dir)
    print(json.dumps({label: row["paired_differences"]["em"] for label, row in report["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
