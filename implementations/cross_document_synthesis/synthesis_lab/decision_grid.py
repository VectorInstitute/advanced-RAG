"""Evidence-backed grid cells; missing experiments remain explicitly pending."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


CORPUS_STRUCTURES = {
    "musique": "Wikipedia passages",
    "nq": "Wikipedia passages",
    "multihop-rag": "Full publisher articles",
}


def decision_grid(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build comparable corpus/category/method cells from replayed full reports."""
    available: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report in reports:
        if not report.get("completed") or not report.get("score_replayed"):
            raise ValueError("decision grid requires completed reports replayed against pinned annotations")
        spec = report["spec"]
        if any(q["id"].endswith(("::plain", "::obfuscated")) for q in spec["questions"]):
            continue
        for category, scores in report["scores"].items():
            if category == "overall":
                continue
            key = (spec["corpus"], category, spec["method"])
            if key in available:
                raise ValueError("duplicate decision-grid cell; choose one explicit configuration")
            available[key] = {
                "corpus": key[0],
                "corpus_structure": CORPUS_STRUCTURES.get(key[0], "unspecified"),
                "complexity": key[1],
                "method": key[2],
                "status": "measured_with_failures" if scores["metrics"]["failure"] else "measured",
                "questions": scores["questions"],
                "metrics": scores["metrics"],
                "usage": scores["usage"],
                "spec_sha256": report["spec_sha256"],
                "cost_note": report["cost_note"],
            }
            if not spec.get("index_coverage", {}).get("complete", True):
                available[key]["status"] = "exploratory_incomplete_index"
                available[key]["index_coverage"] = spec["index_coverage"]
            if "implementation" in spec:
                available[key]["implementation"] = spec["implementation"]
    rows = []
    structures = {
        "musique": ("2-hop", "3-hop", "4-hop"),
        "nq": ("single-hop",),
        "multihop-rag": ("inference_query", "comparison_query", "temporal_query"),
    }
    for corpus, categories in structures.items():
        methods = (
            ("dense", "decomposition", "ircot", "hipporag2", "lightrag", "corrective")
            if corpus == "musique"
            else ("dense", "decomposition", "ircot")
        )
        for category in categories:
            for method in methods:
                key = (corpus, category, method)
                rows.append(
                    available.get(
                        key,
                        {
                            "corpus": corpus,
                            "corpus_structure": CORPUS_STRUCTURES[corpus],
                            "complexity": category,
                            "method": method,
                            "status": "pending",
                            "reason": "article ruling and validated loader execution required"
                            if corpus == "multihop-rag"
                            else "verified RI #4 index handoff required"
                            if method in {"hipporag2", "lightrag"}
                            else "matched evaluation not yet available",
                        },
                    )
                )
    return rows
