"""Score RI #2's recorded MuSiQue predictions on the bootcamp's shared 300 questions.

RI #4, RI #5, and the special topics report MuSiQue on
``stratified_test_set``: 150 two-hop, 75 three-hop, and 75 four-hop questions
drawn with a fixed seed. RI #2's core and graph arms answered all 1,000
questions, so this script selects the shared 300 from the committed
per-question records and scores them with RI #2's own ``summarize`` and
``paired_comparison``. It makes no model calls. The correction cohort ran on
the shared 300 directly, with its own dense and graded single-pass controls in
one response-cache namespace, so its rows pair with those controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aieng.advanced_rag.corpora.musique import TEST_PER_HOP, TEST_SEED, load_musique, stratified_test_set
from synthesis_lab.corpora import load_corpus
from synthesis_lab.evaluation import paired_comparison, summarize
from synthesis_lab.runtime import DEFAULT_CACHE, LAB_DIR, write_json


EVALUATION = LAB_DIR / "outputs/evaluation"
DEFAULT_OUTPUT = EVALUATION / "shared_300.json"
RI4_RESULTS = LAB_DIR.parent / "graphrag_technical_support/outputs/musique_graph_test.json"

#: Arm name to the committed reports that together hold its 1,000 records.
ARMS: dict[str, tuple[str, ...]] = {
    "dense": ("musique_dense.json",),
    "decomposition": ("musique_decomposition.json",),
    "ircot": ("musique_ircot.json",),
    "hipporag2": ("graphrag/hipporag2.json",),
    "lightrag": ("graphrag/lightrag.json",),
}

#: The correction cohort: arm name to its report. Its dense control is a separate
#: execution from the core dense arm, so correction pairs only with these two.
CORRECTION: dict[str, str] = {
    "correction_dense": "corrective/dense.json",
    "graded_single_pass": "corrective/graded-single-pass.json",
    "corrective": "corrective/corrective.json",
}

NOTES = {
    "hipporag2": "atlas-rag variant over RI #4's released index, entity-relation stage only.",
    "lightrag": "Mix mode over RI #4's release reference-ri4-bd05f7bb16c6, which indexes all 11,656 passages.",
    "correction_dense": "Dense control of the correction cohort, run in that cohort's response-cache namespace.",
    "graded_single_pass": "Grades the first five passages and answers once; isolates grading from the retry.",
    "corrective": "Grades five passages, rewrites once on a borderline grade, and allows two retrieval attempts.",
}


def _records(names: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the arm's records and its first spec, checking each digest."""
    records: list[dict[str, Any]] = []
    specs = []
    for name in names:
        report = json.loads((EVALUATION / name).read_text())
        digest = hashlib.sha256(json.dumps(report["spec"], sort_keys=True).encode()).hexdigest()
        if digest != report["spec_sha256"]:
            raise ValueError(f"{name}: report specification digest differs")
        records.extend(report["records"])
        specs.append(report["spec"])
    return records, specs[0]


def _row(scores: dict[str, Any], spec: dict[str, Any], note: str | None) -> dict[str, Any]:
    overall = scores["overall"]
    n = overall["questions"]
    model = spec["model"]
    roles = spec.get("model_roles") or {"generation": model, "extraction": model, "judging": model}
    usage = overall["usage"]
    row: dict[str, Any] = {
        "reader_model": spec["model"],
        "planner_model": roles["extraction"] if spec["method"] in {"decomposition", "ircot", "corrective"} else None,
        "grader_model": roles["judging"] if spec["method"] in {"corrective", "graded-single-pass"} else None,
        "judge_model": None,
        "generation_passages": spec["bounds"]["generation_passages"],
        "exact_match": round(overall["metrics"]["exact_match"], 4),
        "f1": round(overall["metrics"]["token_f1"], 4),
        "f1_by_hops": {label: round(scores[label]["metrics"]["token_f1"], 4) for label in ("2-hop", "3-hop", "4-hop")},
        "supporting_recall_at_5": round(overall["metrics"]["recall_at_5"], 4),
        "generation_all_supports": round(overall["metrics"]["generation_all_supports"], 4),
        # A replayed run answers some requests from the response cache, so the
        # logical request count is live calls plus cache hits.
        "calls_per_question": round((usage["calls"] + usage["cache_hits"]) / n, 2),
        "live_calls_per_question": round(usage["calls"] / n, 2),
    }
    if note:
        row["note"] = note
    return row


def _paired(questions: list[Any], arm: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the paired EM and F1 differences, arm minus control, rounded."""
    paired = paired_comparison(questions, arm, control)
    return {
        key: {k: round(v, 4) for k, v in paired["differences"][key].items()} for key in ("exact_match", "token_f1")
    } | {"samples": paired["samples"], "interval": paired["interval"]}


def _ri4_reference() -> dict[str, Any]:
    """Return RI #4's F1 on the same draw and the settings that differ."""
    ri4 = json.loads(RI4_RESULTS.read_text())
    test = ri4["test"]
    return {
        "source": "implementations/graphrag_technical_support/outputs/musique_graph_test.json",
        "reader_model": ri4["produced_by"]["model"],
        "judge_model": ri4["produced_by"]["judge"],
        "answer_passages": test["answer_passages"],
        "rank_depth": test["rank_depth"],
        "f1": {arm: values["summary"]["all"]["f1"] for arm, values in ri4["arms"].items()},
        "note": (
            "Same questions and reader model; RI #4 uses its own answer prompt and a rank depth "
            "of 10, so its rows are not paired with RI #2's."
        ),
    }


def shared_300(cache: Path) -> dict[str, Any]:
    """Score every arm on the shared draw and pair each against dense."""
    corpus = load_corpus("musique", cache)
    _, raw_questions = load_musique()
    shared_ids = [q.question_id for q in stratified_test_set(raw_questions)]
    by_id = {q.question_id: q for q in corpus.questions}
    questions = [by_id[qid] for qid in shared_ids]
    wanted = set(shared_ids)
    arms: dict[str, Any] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    for arm, names in ARMS.items():
        records, spec = _records(names)
        selected[arm] = [r for r in records if r["question_id"] in wanted]
        arms[arm] = _row(summarize(questions, selected[arm]), spec, NOTES.get(arm))
    comparisons = {}
    for arm in ARMS:
        if arm != "dense":
            comparisons[f"{arm}_minus_dense"] = _paired(questions, selected[arm], selected["dense"])
    for arm, name in CORRECTION.items():
        records, spec = _records((name,))
        if sorted(r["question_id"] for r in records) != sorted(shared_ids):
            raise ValueError(f"{name}: the correction cohort must cover exactly the shared 300")
        selected[arm] = records
        arms[arm] = _row(summarize(questions, records), spec, NOTES.get(arm))
    correction = {
        "corrective_minus_correction_dense": _paired(questions, selected["corrective"], selected["correction_dense"]),
        "corrective_minus_graded_single_pass": _paired(
            questions, selected["corrective"], selected["graded_single_pass"]
        ),
        "graded_single_pass_minus_correction_dense": _paired(
            questions, selected["graded_single_pass"], selected["correction_dense"]
        ),
        "correction_dense_minus_dense": _paired(questions, selected["correction_dense"], selected["dense"]),
    }
    return {
        "produced_by": {
            "model": "Qwen3_6-35B-A3B",
            "endpoint": "vector-cluster",
            "producer": "run_shared_300.py",
            "label": "recorded predictions, rescored",
            "judge": None,
            "note": (
                "Core and graph arms are selected from the committed 1,000-question records; the "
                "correction cohort ran on the shared 300. Rescoring makes no model calls."
            ),
        },
        "test": {
            "questions": len(questions),
            "per_hop": {str(k): v for k, v in TEST_PER_HOP.items()},
            "seed": TEST_SEED,
            "selection": "aieng.advanced_rag.corpora.musique.stratified_test_set",
            "scoring": "Lexical exact match and token F1 against every accepted alias; failures score zero.",
        },
        "arms": arms,
        "comparisons_with_dense": comparisons,
        "correction_comparisons": correction,
        "ri4_reference": _ri4_reference(),
    }


def main() -> None:
    """Write the shared-300 summary from committed records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = shared_300(args.cache)
    write_json(args.output, result)
    for arm, row in result["arms"].items():
        print(f"{arm:14} EM {row['exact_match']:.3f}  F1 {row['f1']:.3f}  calls/q {row['calls_per_question']:.2f}")


if __name__ == "__main__":
    main()
