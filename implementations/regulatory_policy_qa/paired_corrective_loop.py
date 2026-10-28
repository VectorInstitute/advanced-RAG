"""Compare the generator-only control with the best corrective loop per question.

Both arms are replayed from the same recorded ContractNLI traces, so they score
the same 680 questions and the comparison is paired. The script reports an exact
McNemar test on the per-question decisions (abstain on a null, answer an
answerable question), split by question type, and a paired bootstrap interval
for the balanced-accuracy gain. It reads only committed files and makes no model
call.

Examples
--------
Write ``outputs/paired_corrective_loop.json``::

    python3 paired_corrective_loop.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.multihop import mcnemar_exact, paired_bootstrap
from aieng.advanced_rag.evaluation.sweep import Outcome, QueryTrace, generator_only, replay
from aieng.advanced_rag.pipeline.answering import CorrectivePolicy
from aieng.advanced_rag.pipeline.grading import GradingThresholds


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def right(trace: QueryTrace, outcome: Outcome) -> bool:
    """Return whether the pipeline abstained on a null or answered an answerable one."""
    return outcome.abstained if trace.is_null else not outcome.abstained


def weighted(traces: Sequence[QueryTrace], outcomes: Sequence[Outcome]) -> list[float]:
    """Return per-question scores whose mean is the balanced accuracy.

    Each question type carries half the weight, so a right decision on a null
    question scores ``n / (2 * nulls)`` and one on an answerable question
    ``n / (2 * answerable)``.
    """
    nulls = sum(trace.is_null for trace in traces)
    answerable = len(traces) - nulls
    return [
        float(right(trace, outcome)) * len(traces) / (2 * (nulls if trace.is_null else answerable))
        for trace, outcome in zip(traces, outcomes, strict=True)
    ]


def compare(traces: Sequence[QueryTrace], best: dict[str, Any]) -> dict[str, Any]:
    """Return the paired comparison of the control and the loop at ``best``."""
    policy = CorrectivePolicy(GradingThresholds(sufficient=best["sufficient"], insufficient=best["insufficient"]))
    loop = [replay(trace, policy, best["max_attempts"]) for trace in traces]
    control = [generator_only(trace) for trace in traces]
    tests = {}
    for name, keep in (
        ("null", lambda t: t.is_null),
        ("answerable", lambda t: not t.is_null),
        ("all", lambda t: True),
    ):
        pairs = [(t, a, b) for t, a, b in zip(traces, loop, control, strict=True) if keep(t)]
        result = mcnemar_exact([right(t, a) for t, a, _ in pairs], [right(t, b) for t, _, b in pairs])
        tests[name] = {
            "questions": len(pairs),
            "loop_only_right": result.only_a,
            "control_only_right": result.only_b,
            "mcnemar_exact_two_sided_p": round(result.p_value, 4),
        }
    gain = paired_bootstrap(weighted(traces, loop), weighted(traces, control))
    return {
        "corpus": "ContractNLI test, LLMGrader traces",
        "control": "no grader: one retrieval, one generation",
        "loop": {k: best[k] for k in ("label", "sufficient", "insufficient", "max_attempts", "model_calls")},
        "balanced_accuracy": {
            "loop": round(sum(weighted(traces, loop)) / len(traces), 4),
            "control": round(sum(weighted(traces, control)) / len(traces), 4),
        },
        "mcnemar": tests,
        "paired_bootstrap": {
            "statistic": "balanced-accuracy gain, loop minus control",
            "mean_difference": round(gain.mean_difference, 4),
            "low": round(gain.low, 4),
            "high": round(gain.high, 4),
            "confidence": gain.confidence,
            "samples": 10_000,
            "seed": 0,
        },
    }


def main() -> int:
    """Replay both arms and write the comparison."""
    traces = [
        QueryTrace.from_dict(item)
        for item in json.loads((OUTPUTS / "traces_contract-nli_llm.json").read_text(encoding="utf-8"))
    ]
    rows = json.loads((OUTPUTS / "sweep_contract-nli.json").read_text(encoding="utf-8"))["arms"]["llm"]["rows"]
    best = max(rows, key=lambda row: row["balanced_accuracy"])
    payload = compare(traces, best)
    (OUTPUTS / "paired_corrective_loop.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["mcnemar"]["all"]), json.dumps(payload["paired_bootstrap"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
