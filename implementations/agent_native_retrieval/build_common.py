"""What the three question-set builders share: the key file, the models, the summary.

Every model call goes through
:func:`aieng.advanced_rag.evaluation.synthetic.proxy_model`, which caches each
response in one SQLite file under the package cache root, so a re-run on the
same inputs costs nothing. The model names are parameters; the defaults are the
on-prem models the committed sets were built with.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.evaluation.synthetic import Model, proxy_model, usage_of


HERE = Path(__file__).resolve().parent

#: Writes the rewrites and the near-miss edits.
DEFAULT_GENERATOR = "Qwen3_8-27B"

#: Answers from gold passages, judges answers, and checks nulls and keys.
DEFAULT_CHECKER = "Qwen3_8-Flash-Next"


def env_file() -> Path | None:
    """Return the ``.env`` holding the key, looking past a worktree to the main repo.

    ``None`` lets the resolver read the environment and the nearest ``.env``.
    """
    if (HERE.parents[1] / ".env").exists():
        return None
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        cwd=HERE,
    ).stdout.strip()
    candidate = Path(common).parent / ".env" if common else None
    return candidate if candidate is not None and candidate.exists() else None


def response_cache() -> Path:
    """Return the shared response cache the synthetic builders use."""
    path = Path(cache_root()) / "synthetic" / "responses.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model(name: str, per_minute: int, *, max_tokens: int = 600) -> Model:
    """Return a cached, throttled proxy model with thinking off."""
    return proxy_model(
        name,
        cache_path=response_cache(),
        per_minute=per_minute,
        max_tokens=max_tokens,
        env_path=env_file(),
    )


def write_summary(path: Path, summary: Mapping[str, Any], *models: Model) -> None:
    """Write a set's funnel and the calls and tokens each model spent.

    ``cold_*`` counts every request the build made, cached or not, which is
    what a run on an empty cache pays. The ledger's own counts cover only the
    requests this run sent.
    """
    record = dict(summary)
    record["models"] = [usage_of(m) for m in models]
    record["produced_by"] = produced_by(record, path)
    record["price_usd"] = 0.0
    record["price_note"] = "On-prem models through the proxy carry no charge."
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def produced_by(record: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Return the ``produced_by`` block every committed result carries.

    The model is the set's generator, or its checker when no model wrote the
    questions (the ServerFault strings are extracted, then checked).
    """
    usage = {m["model"]: {"calls": m["calls"], "cold_requests": m["cold_requests"]} for m in record["models"]}
    return {
        "model": record.get("generator") or record.get("checker"),
        "family": "onprem",
        "endpoint": "vector-cluster",
        "producer": f"agent_native_retrieval/{PRODUCERS.get(path.name, 'build_common.py')}",
        "usage": usage,
    }


#: The script that writes each summary, for its ``produced_by`` block.
PRODUCERS: dict[str, str] = {
    "musique_near_miss_nulls.summary.json": "build_near_miss_nulls.py",
    "musique_paraphrase_control.summary.json": "build_paraphrase_control.py",
    "serverfault_error_strings.summary.json": "build_serverfault_errors.py",
}


class Clock:
    """Wall-clock seconds since construction, for the summary."""

    def __init__(self) -> None:
        self.start = time.monotonic()

    def seconds(self) -> float:
        """Return elapsed seconds, rounded."""
        return round(time.monotonic() - self.start, 1)
