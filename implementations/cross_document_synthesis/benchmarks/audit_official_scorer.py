"""Check every full-split EM/F1 against MuSiQue's pinned official implementation."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import re
import string
import urllib.request
from pathlib import Path
from typing import Any

from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique
from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1
from synthesis_lab.runtime import DEFAULT_CACHE, write_json

from benchmarks.judge_full_answers import load_arm


REFERENCE_REVISION = "8b8d8ce102f97f968b259ea587a4482163eed2ff"
REFERENCE_URL = f"https://raw.githubusercontent.com/StonyBrookNLP/musique/{REFERENCE_REVISION}/metrics/answer.py"
REFERENCE_SHA256 = "10368f619b4d5ef5d83748c05a96c0afd332a14ab5c010740c98d58dfaefe974"


def audit(cache_dir: Path, *, fetch_reference: bool) -> dict[str, Any]:
    """Evaluate only the pinned scorer's pure answer functions, never its imports."""
    path = cache_dir / "official_answer_metric.py"
    if fetch_reference:
        path.write_bytes(urllib.request.urlopen(REFERENCE_URL, timeout=60).read())  # noqa: S310
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != REFERENCE_SHA256:
        raise ValueError("official scorer source digest differs from pinned reference")
    tree = ast.parse(source)
    names = {"normalize_answer", "get_tokens", "compute_exact", "compute_f1", "metric_max_over_ground_truths"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("official scorer function contract changed")
    namespace: dict[str, Any] = {"re": re, "string": string, "collections": collections}
    module_body: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=module_body, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    _, questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    counts = {}
    for arm in ("dense", "decomposition"):
        records = load_arm(arm)
        for q in questions:
            result = records[q.question_id]["result"]
            prediction = "" if result["abstained"] else result["answer"]
            golds = [q.answer, *q.answer_aliases]
            official_em = namespace["metric_max_over_ground_truths"](namespace["compute_exact"], prediction, golds)
            official_f1 = namespace["metric_max_over_ground_truths"](namespace["compute_f1"], prediction, golds)
            if official_em != exact_match(prediction, golds) or abs(official_f1 - token_f1(prediction, golds)) > 1e-12:
                raise ValueError(f"official scorer mismatch: {arm} {q.question_id}")
        counts[arm] = len(questions)
    report = {
        "reference_revision": REFERENCE_REVISION,
        "reference_url": REFERENCE_URL,
        "reference_sha256": hashlib.sha256(source).hexdigest(),
        "checked_predictions": counts,
        "em_mismatches": 0,
        "f1_mismatches": 0,
        "tolerance": 1e-12,
    }
    write_json(Path(__file__).resolve().parents[1] / "outputs/full_evaluation/official_scorer_audit.json", report)
    return report


def main() -> None:
    """Run on compute; explicitly opt into downloading the pinned reference code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-reference", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(audit(args.cache_dir, fetch_reference=args.fetch_reference))


if __name__ == "__main__":
    main()
