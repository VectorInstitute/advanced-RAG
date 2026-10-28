"""Run the participant path on paid models and write ``outputs/live_check.json``.

The notebook with ``RUN_EXPENSIVE = True`` answers a 40-question stratified
subset with your generation model and judges it with your judging model. This
script does the same for one pair of models, prices it, and records the metrics
beside the on-prem figures for the same subset. It does not touch
``outputs/results.json``.

Examples
--------
Print the estimate and stop::

    python live_check.py --estimate-only

Run it, with a hard ceiling in dollars::

    python live_check.py --generation gpt-6-luna --judging claude-haiku-5-5 --cap 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aieng.advanced_rag import spend  # noqa: E402
from aieng.advanced_rag.cache.paths import root as cache_root  # noqa: E402
from aieng.advanced_rag.models import choose_models  # noqa: E402
from cites import pipeline, report  # noqa: E402
from cites.pipeline import GENERATED, Pipeline  # noqa: E402


REFERENCE = HERE / "outputs" / "results.json"
OUTPUT = HERE / "outputs" / "live_check.json"
#: Questions in the live subset and in the reference set, for scaling usage.
SUBSET_SHARE = pipeline.LIVE_SUBSET / 257


def estimate(generation: str, judging: str) -> dict[str, Any]:
    """Price the reference token counts, scaled to the subset, at these rates."""
    usage = json.loads(REFERENCE.read_text(encoding="utf-8"))["usage"]
    totals = {"answers": [0, 0, 0], "judge": [0, 0, 0]}
    for step, models in usage.items():
        for counts in models.values():
            row = totals[step.split("/")[0]]
            row[0] += counts["calls"]
            row[1] += counts["prompt_tokens"]
            row[2] += counts["completion_tokens"]
    table = spend.price_table()
    out: dict[str, Any] = {}
    total = 0.0
    for step, model in (("answers", generation), ("judge", judging)):
        calls, prompt, completion = (round(v * SUBSET_SHARE) for v in totals[step])
        dollars = spend.estimate_tokens(model, prompt, completion, table=table)
        out[step] = {
            "model": model,
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": dollars,
        }
        total += dollars or 0.0
    out["total_usd"] = round(total, 4)
    return out


def priced(usage: dict[str, Any]) -> dict[str, Any]:
    """Return calls, tokens, and dollars per model from a ``report.usage`` record.

    The pipeline's metering session reads its own meter, which a ``load_or_run``
    step does not write to, so the cost comes from the usage each step stored.
    """
    table = spend.price_table()
    totals: dict[str, dict[str, float]] = {}
    for models in usage.values():
        for model, counts in models.items():
            row = totals.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            for key in row:
                row[key] += counts[key]
    for model, row in totals.items():
        row["usd"] = round(
            spend.estimate_tokens(model, int(row["prompt_tokens"]), int(row["completion_tokens"]), table=table) or 0.0,
            4,
        )
    return {"by_model": totals, "total_usd": round(sum(r["usd"] for r in totals.values()), 4)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Answer and judge the subset, then return the record to write."""
    pipeline.RPM = args.rpm
    pipeline.WORKERS = args.workers
    # Paid responses get their own cache file, so the on-prem build can run beside this.
    pipeline.response_cache_path = lambda: Path(cache_root()) / "responses" / "citations-live.sqlite"
    models = choose_models(generation=args.generation, judging=args.judging)
    session = spend.SpendSession(limit_usd=args.cap)
    account = spend.account()
    before = spend.spent_usd(account.key_usage())
    pipe = Pipeline(models, live=True, confirm=True, env_path=args.env_file)
    arms = [arm for arm in pipeline.ARMS if arm != "posthoc"]
    for arm in arms:
        if arm in GENERATED:
            answers = pipe.answers(arm, "subset")
            print(arm, len(answers), "answered,", sum(a.abstained for a in answers), "abstained", flush=True)
    for arm in arms:
        print(arm, len(pipe.judged(arm, "subset")), "judged", flush=True)
    summary = report.summary(pipe, "subset")
    live = {
        arm: {
            "llm": row["llm"],
            "correctness": row["correctness"],
            "agreement": row["agreement"]["llm"],
            "median_cited_chars": row["median_cited_chars"],
        }
        for arm, row in summary["arms"].items()
    }
    generation = pipe.answer_result("inline", "subset").provenance
    judging = pipe.judge_result("inline", "subset").provenance
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["subset"]["arms"]
    session.close()
    after = spend.spent_usd(account.key_usage())
    usage = report.usage(pipe, "subset")
    cost = {
        **priced(usage),
        "proxy_delta_usd": None if before is None or after is None else round(after - before, 4),
        "note": "the proxy figure is for the team key and includes other callers in the same window",
    }
    print(json.dumps(cost, indent=1))
    return {
        "produced_by": {
            "model": models.generation.model,
            "judging_model": models.judging.model,
            "generation": generation.to_dict() if generation else None,
            "judging": judging.to_dict() if judging else None,
            "note": "one run, explore mode, 40-question stratified subset of the ContractNLI test split",
        },
        "usage": usage,
        "cost": cost,
        "questions": summary["questions"],
        "labelled": summary["labelled"],
        "nulls": summary["nulls"],
        "arms": live,
        "on_prem_same_subset": {
            arm: {
                "llm": reference[arm]["llm"],
                "correctness": reference[arm]["correctness"],
                "agreement": reference[arm]["agreement"]["llm"],
            }
            for arm in live
            if arm in reference
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Estimate, or run and write the record."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--generation", default="gpt-6-luna")
    parser.add_argument("--judging", default="claude-haiku-5-5")
    parser.add_argument("--cap", type=float, default=3.0, help="hard ceiling in dollars on this run's estimate")
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.env_file is not None:
        from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415

        spend.use_account(spend.ProxyAccount(resolve_api_key(args.env_file), resolve_base_url(args.env_file)))
    plan = estimate(args.generation, args.judging)
    print(json.dumps(plan, indent=1))
    if args.estimate_only:
        return 0
    if plan["total_usd"] > args.cap:
        print("estimate is over the cap; not running")
        return 1
    record = run(args)
    record["estimate"] = plan
    OUTPUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
