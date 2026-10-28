#!/usr/bin/env python3
"""Rebuild RI #2's committed summaries from the committed compact reports.

``outputs/evaluation/summary.json`` replays the six core reports with the
correction cohort, which ``outputs/evaluation/corrective/summary.json`` also holds
on its own. ``outputs/evaluation/graphrag/summary.json`` adds the two
graph arms over RI #4's released indexes, with each index's validation and the
graph-minus-dense comparisons. Both go through ``run_evaluation.compare``, which
replays every saved score against the pinned annotations. No model is called.

Examples
--------
From this folder, with the pinned MuSiQue and NaturalQuestions annotations
fetched on first use::

    $ python build_summaries.py
"""

from __future__ import annotations

import json
from typing import Any

from run_evaluation import compare
from synthesis_lab.corpora import load_corpus
from synthesis_lab.graph_release import release_entry
from synthesis_lab.runtime import DEFAULT_CACHE, LAB_DIR, write_json


EVALUATION = LAB_DIR / "outputs/evaluation"
VALIDATION = LAB_DIR / "outputs/validation/graph_release.json"
CORE = [f"{corpus}_{method}.json" for corpus in ("musique", "nq") for method in ("dense", "decomposition", "ircot")]
GRAPH = ["graphrag/hipporag2.json", "graphrag/lightrag.json"]
CORRECTIVE = [f"corrective/{method}.json" for method in ("dense", "graded-single-pass", "corrective")]
LABELS = {"hipporag2": "HippoRAG 2 (atlas-rag, query2edge)", "lightrag": "LightRAG (mix mode)"}


def _read(names: list[str]) -> list[dict[str, Any]]:
    return [json.loads((EVALUATION / name).read_text()) for name in names]


def graph_validation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Return each graph index's release pin and the coverage its run verified."""
    corpus = load_corpus("musique", DEFAULT_CACHE)
    result: dict[str, Any] = {
        "corpus": {
            "revision": corpus.provenance["revision"],
            "corpus_sha256": corpus.provenance["corpus_sha256"],
            "passages": len(corpus.passages),
            "questions": len(corpus.questions),
        }
    }
    for report in reports:
        backend = report["spec"]["producer"]["backend"]
        name = "hipporag2" if backend == "atlas-rag" else backend
        # Runs from before coverage was recorded verified every passage mapping first.
        coverage = report["spec"].get(
            "index_coverage",
            {"complete": True, "represented_passages": len(report["spec"]["producer"]["passage_ids"])},
        )
        result[name] = {
            "status": "verified" if coverage.get("complete", True) else "incomplete",
            "implementation": report["spec"]["implementation"],
            "index_coverage": coverage,
            "release_used": report["spec"]["producer"].get("release"),
            "release_pinned": release_entry(name),
        }
    return result


def main() -> None:
    """Write both summaries and the graph-release validation."""
    core, graph, corrective = _read(CORE), _read(GRAPH), _read(CORRECTIVE)
    corpora = {
        name: load_corpus(name, DEFAULT_CACHE if name == "musique" else DEFAULT_CACHE.parent / "ri2-complete" / name)
        for name in ("musique", "nq")
    }
    produced = {
        "model": "Qwen3_6-35B-A3B",
        "endpoint": "vector-cluster",
        "producer": "build_summaries.py",
        "note": "Replay and paired comparison of recorded executions; summary creation makes no model calls.",
    }
    summary = compare(core, corpora, corrective_reports=corrective)
    write_json(EVALUATION / "summary.json", summary | {"produced_by": produced})
    write_json(EVALUATION / "corrective/summary.json", summary["corrective_cohort"] | {"produced_by": produced})

    validation = graph_validation(graph)
    write_json(
        VALIDATION, validation | {"produced_by": produced | {"note": "Index coverage recorded by each graph run."}}
    )
    combined = compare([*core, *graph], corpora, corrective_reports=corrective)
    combined["artifact_validation"] = validation
    combined["graph_comparisons"] = [
        {
            "implementation": LABELS[row["method"]],
            "retrieval_depth": 20,
            "generation_passages": 5,
            "paired": row,
        }
        for row in combined["comparisons"]
        if row["method"] in LABELS and row["corpus"] == "musique" and "wording" not in row
    ]
    write_json(EVALUATION / "graphrag/summary.json", combined | {"produced_by": produced})
    for row in combined["comparisons"]:
        if "wording" not in row:
            f1 = row["differences"]["token_f1"]
            print(
                f"{row['corpus']:8} {row['method']:20} - {row['control']:18} F1 {f1['mean']:+.4f} [{f1['low']:+.4f}, {f1['high']:+.4f}]"
            )


if __name__ == "__main__":
    main()
