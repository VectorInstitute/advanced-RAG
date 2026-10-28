"""Paired evaluation of answer quality, retrieved evidence, failures, and cost."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np
from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1

from .corpora import Question


def question_scores(question: Question, row: Mapping[str, Any]) -> dict[str, float]:
    """Failures remain in the denominator with zero answer/evidence credit.

    A committed record carries ``exact_match`` and ``token_f1`` in place of the
    answer text; those stored scores are used when the record has no answer.
    """
    if "answer" in row or "token_f1" not in row:
        prediction = "" if row.get("abstained", True) or row.get("failure") else row["answer"]
        answer_em, answer_f1 = float(exact_match(prediction, question.answers)), token_f1(prediction, question.answers)
    else:
        answer_em, answer_f1 = float(row["exact_match"]), float(row["token_f1"])
    gold = set(question.supporting_ids)
    ranked = row.get("retrieved_ids", []) if not row.get("failure") else []
    generation = set(row.get("generation_ids", [])) if not row.get("failure") else set()
    citations = row.get("citation_ids", []) if not row.get("failure") else []
    result = {
        "exact_match": answer_em,
        "token_f1": answer_f1,
        "generation_recall": len(gold & generation) / len(gold),
        "generation_all_supports": float(gold <= generation),
        "abstained": float(row.get("abstained", True)),
        "failure": float(bool(row.get("failure"))),
        "answered_with_valid_citations": float(
            bool(citations)
            and not row.get("abstained", True)
            and len(citations) == len(set(citations))
            and set(citations) <= generation
            and not row.get("dropped_citations")
        ),
    }
    for k in (5, 10, 20):
        found = set(ranked[:k])
        result[f"recall_at_{k}"] = len(gold & found) / len(gold)
        result[f"all_supports_at_{k}"] = float(gold <= found)
    return result


def summarize(questions: Sequence[Question], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Require exact coverage; report all categories and wording variants."""
    by_id = {r["question_id"]: r for r in records}
    if not questions or len(by_id) != len(records) or set(by_id) != {q.question_id for q in questions}:
        raise ValueError("scores require exactly one record per selected question")
    groups = {"overall": list(questions)}
    groups.update({c: [q for q in questions if q.category == c] for c in sorted({q.category for q in questions})})
    if {q.wording for q in questions} != {"original"}:
        groups.update({w: [q for q in questions if q.wording == w] for w in sorted({q.wording for q in questions})})
    report = {}
    for label, selected in groups.items():
        values = [question_scores(q, by_id[q.question_id]) for q in selected]
        usage = [by_id[q.question_id].get("usage", {}) for q in selected]
        report[label] = {
            "questions": len(selected),
            "metrics": {key: mean(v[key] for v in values) for key in values[0]},
            "usage": {
                key: sum(u.get(key, 0) for u in usage)
                for key in (
                    "calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "tool_calls",
                    "cache_hits",
                    "service_seconds",
                    "calls_missing_usage",
                )
            },
            "stop_reasons": dict(Counter(by_id[q.question_id].get("stop_reason", "unknown") for q in selected)),
            "summed_question_wall_seconds": sum(by_id[q.question_id].get("elapsed_seconds", 0) for q in selected),
        }
    return report


def paired_comparison(
    questions: Sequence[Question],
    arm: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    *,
    samples: int = 5000,
) -> dict[str, Any]:
    """Bootstrap original question IDs, retaining paired wording variants together."""
    summarize(questions, arm)
    summarize(questions, control)
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    a, b = ({r["question_id"]: r for r in rows} for rows in (arm, control))
    clusters: dict[str, list[Question]] = {}
    for q in questions:
        clusters.setdefault(q.original_id or q.question_id, []).append(q)
    if len({len(group) for group in clusters.values()}) != 1:
        raise ValueError("paired clusters must have the same number of wording variants")
    rng = np.random.default_rng(20261020)
    result: dict[str, Any] = {
        "questions": len(questions),
        "original_ids": len(clusters),
        "samples": samples,
        "interval": "paired cluster percentile 95%",
        "differences": {},
    }
    for key in ("exact_match", "token_f1", "recall_at_5", "generation_all_supports"):
        differences = np.asarray(
            [
                mean(
                    question_scores(q, a[q.question_id])[key] - question_scores(q, b[q.question_id])[key] for q in group
                )
                for group in clusters.values()
            ]
        )
        draws = np.concatenate(
            [
                differences[rng.integers(0, len(differences), size=(min(250, samples - i), len(differences)))].mean(
                    axis=1
                )
                for i in range(0, samples, 250)
            ]
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        result["differences"][key] = {"mean": float(differences.mean()), "low": float(low), "high": float(high)}
    result["quality_improvement_supported"] = result["differences"]["token_f1"]["low"] > 0
    result["added_calls"] = sum(r.get("usage", {}).get("calls", 0) for r in arm) - sum(
        r.get("usage", {}).get("calls", 0) for r in control
    )
    result["added_calls_plus_cache_hits"] = sum(
        r.get("usage", {}).get("calls", 0) + r.get("usage", {}).get("cache_hits", 0) for r in arm
    ) - sum(r.get("usage", {}).get("calls", 0) + r.get("usage", {}).get("cache_hits", 0) for r in control)
    result["cost_note"] = (
        "Added calls are observed live responses. Compare cache hits separately; differing cache warmth is not a cold-cost saving."
    )
    return result
