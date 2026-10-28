"""Shared provenance, private caches, and explicit experiment execution."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.models import REFERENCE_MODELS, ModelSet, choose_models
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


RI = Path(__file__).resolve().parents[1]
ROOT = RI.parents[1]
STORE: Path = cache_root() / "ri1-answering"
OUTPUT = RI / "outputs/advanced"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
REPORT_NAMES = frozenset({"policies", "scope", "regulatory", "ragas", "agent"})
MODEL = REFERENCE_MODELS.generation.model  # Model used by the saved reference reports.
EMBEDDING = "BAAI/bge-small-en-v1.5"
sys.path.insert(0, str(RI / "graphrag_null"))


def digest(path: Path) -> str:
    """Return a file's exact content identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_compute() -> None:
    """Reject cluster login hosts without requiring Slurm on other platforms."""
    if socket.gethostname().startswith(("blogin", "login")):
        raise RuntimeError("Run experiments on a compute node, not a cluster login host")


def run_path(name: str) -> Path:
    """Locate a run artifact under the currently configured private store."""
    return STORE / "runs" / RUN_ID / name


def write_json(path: Path, payload: Any) -> None:
    """Save live variants locally, preserving shipped reference measurements."""
    report_name = path.stem if path.parent == OUTPUT and path.stem in REPORT_NAMES else None
    if report_name:
        path = run_path(path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    if report_name:
        publish_cache(path)


def publish_cache(path: Path) -> None:
    """Cache a live measurement in the local tier, leaving references intact."""
    payload = json.loads(path.read_text())
    key = CacheKey("ri1", "advanced-report", "ri1-lab-v1", {"name": path.stem, "report_sha256": digest(path)})
    cache = TieredResultCache(global_dir=OUTPUT / "cache", local_dir=STORE / "artifact-cache")
    cache.load_or_compute(key, lambda: payload, force_refresh=True, write_tier="local")


def provenance(*, models: ModelSet | None = None) -> dict[str, Any]:
    """Describe the actual execution, without credentials or private prompts."""
    models = models or choose_models()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "compute_node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": models.generation.model,
        "model_roles": models.as_dict(),
        "embedding_model": EMBEDDING,
        "reasoning": False,
        "temperature": 0,
    }


def completer(
    *,
    max_tokens: int = 1024,
    rpm: int = 20,
    cache_only: bool = False,
    role: Literal["generation", "judging", "extraction"] = "generation",
    models: ModelSet | None = None,
) -> Any:
    """Use the requested Vector key and a private resumable response cache."""
    from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: PLC0415
    from null_lab import MeteredCompleter, proxy_batch  # noqa: PLC0415

    models = models or (REFERENCE_MODELS if cache_only else choose_models())
    model = getattr(models, role).model
    connection: dict[str, Any] = {}
    if not cache_only:
        spend.session()
        connection = {
            "rpm": rpm,
            "timeout": 180,
            "api_key": resolve_api_key(start=ROOT),
            "base_url": resolve_base_url(start=ROOT),
        }
    batch = proxy_batch(model, ResponseCache(STORE / "responses.sqlite"), cache_only=cache_only, **connection)
    return MeteredCompleter(batch, max_tokens=max_tokens)


def saved(name: str) -> dict[str, Any]:
    """Read a measured report; absent measurements are an explicit error."""
    path = OUTPUT / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference report {path}; restore the shipped outputs from the repository")
    key = CacheKey("ri1", "advanced-report", "ri1-lab-v1", {"name": name, "report_sha256": digest(path)})
    cached = TieredResultCache(global_dir=OUTPUT / "cache", local_dir=STORE / "artifact-cache").get(key)
    if cached is None or not isinstance(cached.value, dict):
        raise FileNotFoundError(f"Missing shared-cache envelope for {path}")
    return cached.value
