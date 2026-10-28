"""Replay benchmark scores against pinned annotations and produce paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from synthesis_lab.corpora import Corpus, load_corpus, load_rewrites
from synthesis_lab.decision_grid import decision_grid
from synthesis_lab.evaluation import paired_comparison, summarize
from synthesis_lab.runtime import DEFAULT_CACHE, write_json


def replay(report: dict[str, Any], corpus: Corpus) -> tuple[dict[str, Any], Corpus]:
    """Reconstruct exact selected annotations and refuse changed texts or provenance."""
    if not report.get("coverage_complete") or "scores" not in report:
        raise ValueError("incomplete coverage cannot enter a completed comparison")
    spec = report["spec"]
    if hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != report["spec_sha256"]:
        raise ValueError("report specification digest differs")
    if spec["corpus"] != corpus.name or spec["provenance"] != corpus.provenance:
        raise ValueError("report corpus provenance differs")
    known = {q.question_id: q for q in corpus.questions}
    selected = []
    for item in spec["questions"]:
        if item["id"] not in known:
            raise ValueError("report question ID differs from the pinned input")
        text = known[item["id"]].question
        matches = (
            text == item["text"]
            if "text" in item
            else hashlib.sha256(text.encode()).hexdigest() == item.get("text_sha256")
        )
        if not matches:
            raise ValueError("report question ID/text differs from the pinned input")
        selected.append(known[item["id"]])
    if len({q.question_id for q in selected}) != len(selected):
        raise ValueError("duplicate selected question")
    known_passages = {p.passage_id for p in corpus.passages}
    budget = spec.get("bounds", {}).get("generation_passages", 5)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("invalid generation passage budget")
    for row in report["records"]:
        ranked, generation = row.get("retrieved_ids", []), row.get("generation_ids", [])
        if (
            len(set(ranked)) != len(ranked)
            or not set(ranked) <= known_passages
            or not set(generation) <= set(ranked)
            or len(set(generation)) != len(generation)
            or len(generation) > budget
        ):
            raise ValueError("invalid retrieved or generation passage IDs")
    scores = summarize(selected, report["records"])
    if scores != report["scores"]:
        raise ValueError("saved metrics do not replay")
    return report | {
        "completed": True,
        "execution_success": not any(r.get("failure") for r in report["records"]),
        "score_replayed": True,
    }, replace(corpus, questions=tuple(selected))


def _attach_corrective_cohort(
    result: dict[str, Any],
    checked: list[dict[str, Any]],
    corrective_reports: list[dict[str, Any]],
    corpora: dict[str, Corpus],
) -> None:
    """Attach a separately matched cohort without replacing core controls."""
    methods = [r["spec"]["method"] for r in corrective_reports]
    if len(methods) != 3 or set(methods) != {"dense", "graded-single-pass", "corrective"}:
        raise ValueError("corrective cohort requires one dense, graded-single-pass, and corrective report")
    if any(r["spec"]["corpus"] != "musique" for r in corrective_reports):
        raise ValueError("corrective cohort must use MuSiQue")
    namespace = corrective_reports[0]["spec"].get("response_cache_namespace")
    if not namespace or any(r["spec"].get("response_cache_namespace") != namespace for r in corrective_reports):
        raise ValueError("corrective cohort requires one explicit response cache namespace")
    expected = {q.question_id for q in corpora["musique"].questions}
    selected = [q["id"] for q in corrective_reports[0]["spec"]["questions"]]
    if not selected or any([q["id"] for q in r["spec"]["questions"]] != selected for r in corrective_reports):
        raise ValueError("corrective cohort requires identical nonempty question coverage")
    result["corrective_cohort"] = compare(corrective_reports, corpora)
    result["corrective_cohort"]["selection"] = corrective_reports[0].get("selection")
    result["corrective_cohort"]["full_corpus_coverage"] = set(selected) == expected
    if set(selected) != expected:
        return  # Subset results have their own grid; never populate full-split cells.
    correction = next(r for r in corrective_reports if r["spec"]["method"] == "corrective")
    verified, _ = replay(correction, corpora["musique"])
    result["grid"] = decision_grid([*checked, verified])
    for cell in result["grid"]:
        if cell["method"] == "corrective" and cell["status"] != "pending":
            cell["comparison_set"] = "corrective_cohort"


def _roles(spec: dict[str, Any]) -> dict[str, str]:
    """Return the chat roles; an older report's answer model stands for all three."""
    roles = spec.get("model_roles") or {}
    return {role: roles.get(role, spec["model"]) for role in ("generation", "extraction", "judging")}


def compare(
    reports: list[dict[str, Any]], corpora: dict[str, Corpus], *, corrective_reports: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Pair only matched question sets and answer settings; show each wording arm."""
    checked = []
    selected = {}
    for report in reports:
        verified, corpus = replay(report, corpora[report["spec"]["corpus"]])
        checked.append(verified)
        selected[report["spec_sha256"]] = corpus
    comparisons = []
    for arm in checked:
        if arm["spec"]["method"] == "dense":
            continue
        for control in checked:
            a, b = arm["spec"], control["spec"]
            controls = {"dense", "graded-single-pass"} if a["method"] == "corrective" else {"dense"}
            if b["method"] not in controls or a["corpus"] != b["corpus"]:
                continue
            if (
                a["questions"] != b["questions"]
                or _roles(a) != _roles(b)
                or any(
                    a[k] != b[k]
                    for k in (
                        "provenance",
                        "model",
                        "base_url",
                        "embedding",
                        "answer_prompt_sha256",
                        "temperature",
                        "thinking",
                    )
                )
                or any(
                    a.get("bounds", {}).get(key, default) != b.get("bounds", {}).get(key, default)
                    for key, default in (("retrieval_depth", 20), ("generation_passages", 5))
                )
            ):
                raise ValueError("comparison arms have unmatched inputs or answer settings")
            questions = selected[arm["spec_sha256"]].questions
            paired = paired_comparison(questions, arm["records"], control["records"])
            comparisons.append({"corpus": a["corpus"], "method": a["method"], "control": b["method"], **paired})
            for wording in ("plain", "obfuscated"):
                subset = [q for q in questions if q.wording == wording]
                if subset:
                    ids = {q.question_id for q in subset}
                    comparisons.append(
                        {
                            "corpus": a["corpus"],
                            "method": a["method"],
                            "control": b["method"],
                            "wording": wording,
                            **paired_comparison(
                                subset,
                                [r for r in arm["records"] if r["question_id"] in ids],
                                [r for r in control["records"] if r["question_id"] in ids],
                            ),
                        }
                    )
    wording_comparisons = []
    for report in checked:
        questions = selected[report["spec_sha256"]].questions
        plain = [q for q in questions if q.wording == "plain"]
        if not plain:
            continue
        pairs = {q.original_id: q for q in questions if q.wording == "obfuscated"}
        records = {r["question_id"]: r for r in report["records"]}
        if {q.original_id for q in plain} != set(pairs):
            raise ValueError("wording comparison requires complete pairs")
        canonical = [replace(q, question_id=q.original_id, wording="original") for q in plain]
        a = [records[q.question_id] | {"question_id": q.original_id} for q in plain]
        b = [records[pairs[q.original_id].question_id] | {"question_id": q.original_id} for q in plain]
        wording_comparisons.append(
            {
                "method": report["spec"]["method"],
                "direction": "plain minus obfuscated",
                **paired_comparison(canonical, a, b),
            }
        )
    result: dict[str, Any] = {
        "grid": decision_grid(checked),
        "comparisons": comparisons,
        "wording_comparisons": wording_comparisons,
        "reports": [
            {
                "spec_sha256": r["spec_sha256"],
                "method": r["spec"]["method"],
                "corpus": r["spec"]["corpus"],
                "provenance": r["spec"]["provenance"],
                "scores": r["scores"],
                "cost_note": r["cost_note"],
                "execution_success": r["execution_success"],
                "workers": r.get("workers"),
                "execution_wall_seconds": r.get("execution_wall_seconds"),
            }
            for r in checked
        ],
    }

    if corrective_reports is not None:
        _attach_corrective_cohort(result, checked, corrective_reports, corpora)
    return result


def main() -> None:
    """Load annotations on compute; scoring makes no generation requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corrective-reports", nargs=3, type=Path, help="Separate matched MuSiQue dense/graded/corrective cohort"
    )
    parser.add_argument("--rewrites", type=Path)
    parser.add_argument("--article-ruling", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    corrective = [json.loads(path.read_text()) for path in args.corrective_reports] if args.corrective_reports else None
    corpora = {}
    for name in {r["spec"]["corpus"] for r in [*reports, *(corrective or [])]}:
        corpus = load_corpus(
            name,
            DEFAULT_CACHE if name == "musique" else DEFAULT_CACHE.parent / "ri2-complete" / name,
            ruling=args.article_ruling,
        )
        corpora[name] = load_rewrites(args.rewrites, corpus) if args.rewrites and name == "musique" else corpus
    result = compare(reports, corpora, corrective_reports=corrective)
    write_json(args.output, result)
    print(f"Replayed {len(reports)} reports; wrote {len(result['comparisons'])} paired comparisons.")


if __name__ == "__main__":
    main()
