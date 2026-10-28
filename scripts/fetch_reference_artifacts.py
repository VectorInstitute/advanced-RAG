#!/usr/bin/env python3
"""Install every published reference-artifact manifest, skipping what is installed.

``scripts/setup.sh`` runs this on every workspace start. It finds each
``implementations/**/reference-artifacts.json``, and for each artifact in it:

- skips the artifact when the fetched-global tier already holds bytes with the
  pinned SHA-256, so a warm start downloads nothing;
- otherwise installs it through :meth:`TieredResultCache.fetch_manifest`, which
  checks the envelope digest, the payload digest, and the key before anything
  lands in the cache;
- retries a failed download on pooch's schedule (wait 1 s, then 2 s, and so on
  up to 10 s), except a 404, which means the release is not published, and
  stops that manifest.

One failure never stops the next artifact or manifest. The exit status is 1 when
anything failed, so ``setup.sh`` can warn and carry on.

Examples
--------
Install every manifest in the repository::

    $ python scripts/fetch_reference_artifacts.py

Install one, where ``<manifest>`` is its path::

    $ python scripts/fetch_reference_artifacts.py <manifest>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import urllib.error
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import CacheIntegrityError, CacheKey, TieredResultCache


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "reference-artifacts.json"
#: pooch's ``retry_if_failed`` schedule: one second more per retry, at most ten.
MAX_WAIT_SECONDS = 10


def find_manifests(root: Path = REPO_ROOT) -> list[Path]:
    """Return every reference-artifact manifest under ``root/implementations``."""
    return sorted((root / "implementations").rglob(MANIFEST_NAME))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_installed(cache: TieredResultCache, entry: dict[str, Any]) -> bool:
    """Return whether the fetched-global tier holds this entry's pinned bytes."""
    key = CacheKey(**entry["key"])
    envelope = cache.fetched_global_dir / key.relative_path
    if _sha256(envelope) != entry["sha256"]:
        return False
    if entry.get("payload_sha256") is None:
        return True
    return bool(_sha256(envelope.with_suffix(".artifact")) == entry["payload_sha256"])


def _is_not_found(exc: BaseException) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 404


def install_entry(
    cache: TieredResultCache,
    entry: dict[str, Any],
    *,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Install one manifest entry, retrying a failed download.

    Raises
    ------
    OSError or CacheIntegrityError
        The last error, once every attempt has failed, or a 404 at once.
    """
    with tempfile.TemporaryDirectory() as scratch:
        single = Path(scratch) / MANIFEST_NAME
        single.write_text(json.dumps({"artifacts": [entry]}), encoding="utf-8")
        for attempt in range(retries + 1):
            try:
                cache.fetch_manifest(single)
                return
            except (OSError, CacheIntegrityError) as exc:
                if _is_not_found(exc) or attempt == retries:
                    raise
                sleep(min(attempt + 1, MAX_WAIT_SECONDS))


def install_manifest(
    cache: TieredResultCache,
    manifest: Path,
    *,
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, int, list[str]]:
    """Install one manifest.

    Returns
    -------
    tuple
        Artifacts installed now, artifacts already present, and one message
        per failure.
    """
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))["artifacts"]
    except (OSError, ValueError, KeyError) as exc:
        return 0, 0, [f"cannot read the manifest: {exc}"]
    installed = present = 0
    failures: list[str] = []
    for entry in entries:
        if is_installed(cache, entry):
            present += 1
            continue
        name = entry["url"].rsplit("/", 1)[-1]
        try:
            install_entry(cache, entry, retries=retries, sleep=sleep)
            installed += 1
        except (OSError, CacheIntegrityError) as exc:
            failures.append(f"{name}: {exc}")
            if _is_not_found(exc):
                failures.append("the release is not published yet; skipping the rest")
                break
    return installed, present, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Install the named manifests, or every manifest in the repository."""
    parser = argparse.ArgumentParser(
        prog="python scripts/fetch_reference_artifacts.py",
        description="Install reference-artifact manifests, skipping installed artifacts.",
    )
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per artifact after the first attempt. Default: 3.",
    )
    arguments = parser.parse_args(argv)
    manifests = arguments.manifests or find_manifests()
    if not manifests:
        print("No reference-artifact manifests are published yet.")
        return 0
    cache = TieredResultCache()
    failed = False
    for manifest in manifests:
        installed, present, failures = install_manifest(cache, manifest, retries=arguments.retries)
        try:
            shown = manifest.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = manifest
        print(f"{shown}: {installed} installed, {present} already present, {len(failures)} failed")
        for line in failures:
            print(f"  {line}")
        failed = failed or bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
