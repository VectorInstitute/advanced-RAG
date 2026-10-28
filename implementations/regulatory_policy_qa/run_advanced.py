#!/usr/bin/env python3
"""Replay RI #1's answering measurements, or explicitly run a local variant."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from advanced_lab import common
from aieng.advanced_rag import spend
from aieng.advanced_rag.models import choose_models


EXPERIMENTS = {
    "agent": ("agent", "measure_agent"),
    "policies": ("answering", "measure_policies"),
    "ragas": ("ragas_eval", "measure_ragas"),
    "regulatory": ("answering", "measure_regulatory"),
    "scope": ("scope", "measure_scope"),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Keep reference replay separate from opt-in model work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(common.REPORT_NAMES))
    parser.add_argument("--run", action="store_true", help="run the experiment; may make model calls")
    parser.add_argument(
        "--output-dir", type=Path, help="private cache/run directory; shipped references are never overwritten"
    )
    parser.add_argument(
        "--input-dir", type=Path, help="RAGAS only: directory containing regulatory.json and regulatory_source.json"
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="policy run only: require saved model responses; local retrieval still runs",
    )
    args = parser.parse_args(argv)
    if not args.run and (args.output_dir is not None or args.input_dir is not None or args.cache_only):
        parser.error("--output-dir, --input-dir and --cache-only require --run; default mode replays shipped results")
    if args.input_dir is not None and args.task != "ragas":
        parser.error("--input-dir applies only to ragas")
    if args.cache_only and args.task != "policies":
        parser.error("--cache-only applies only to policies; use the default mode for reference replay")
    if not args.run:
        result = common.saved(args.task)
    else:
        if args.output_dir:
            common.STORE = args.output_dir.expanduser().resolve()
        common.require_compute()
        # Import only the selected experiment; replay never imports live clients.
        module, function = EXPERIMENTS[args.task]
        measure = getattr(importlib.import_module(f"advanced_lab.{module}"), function)
        options: dict[str, Any] = {}
        if args.input_dir is not None:
            options["input_dir"] = args.input_dir.expanduser().resolve()
        if args.cache_only:
            options["cache_only"] = True
        live_chat = not args.cache_only and args.task != "scope"
        if live_chat:
            with redirect_stdout(sys.stderr):
                choose_models(show=True)
        try:
            result = measure(**options)
        finally:
            if live_chat:
                print(spend.report(), file=sys.stderr)
        print(f"Saved report: {common.run_path(args.task + '.json')}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
