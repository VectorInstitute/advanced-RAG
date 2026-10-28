#!/usr/bin/env python3
"""Package the GraphRAG null-arm store so another machine scores without re-indexing.

The store is not committed, because it holds contract text and whole model
prompts. This builds a bundle to hand over out of band, with a manifest that
carries a digest per file and the corpus attribution the licence requires.

Two bundles are available. The default one holds the retrieval tables and the
response cache, which is everything scoring needs, because ``run_null_arm.py``
skips retrieval when the tables are present. ``--full`` adds the LightRAG
indexes, which are only needed to retrieve again, and they are over half the
bytes.

Examples
--------
Build the score-only bundle and check it::

    $ python export_store.py --out /tmp/null-arm-store.tar.gz
    $ python export_store.py --verify /tmp/null-arm-store.tar.gz

Verification does not extract the archive. Unpack it into ``./unpacked`` before
running ``run_null_arm.py --store ./unpacked``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "aieng-advanced-rag"))

from aieng.advanced_rag.corpora.contract_nli import (  # noqa: E402
    CONTRACT_NLI_LICENCE,
)


MANIFEST_NAME = "MANIFEST.json"
SCHEMA_VERSION = 1

# What scoring needs: the retrieval tables, and the response cache so a re-score
# pays for no model call it has already paid for.
SCORE_ONLY = ("evidence", "proxy_cache.sqlite")
# The indexes buy the ability to retrieve again, which the tables above make
# unnecessary, and they are the part most likely to break across LightRAG
# versions.
INDEXES = ("lightrag",)
# Deliberately absent: `corpus/` re-downloads itself through the digest-pinned
# loader, so shipping it would add bytes and a second copy to keep honest.
EXCLUDED = ("corpus",)


def default_store() -> Path:
    """Return the store location ``run_null_arm.py`` writes to."""
    return Path.home() / ".cache" / "advanced-rag" / "graphrag-null"


CHUNK = 1024 * 1024


def digest_stream(stream: IO[bytes]) -> str:
    """Return the SHA-256 of an open binary stream, read in chunks.

    Takes a stream rather than a path so the same digest runs over a file on
    disk and over a member inside an archive, which is what makes building a
    manifest and verifying one the same comparison.
    """
    sha = hashlib.sha256()
    while block := stream.read(CHUNK):
        sha.update(block)
    return sha.hexdigest()


def digest(path: Path) -> str:
    """Return the SHA-256 of one file."""
    with path.open("rb") as handle:
        return digest_stream(handle)


def bundle_members(store: Path, *, full: bool) -> list[Path]:
    """Return the files to pack, as paths relative to ``store``.

    Parameters
    ----------
    store : Path
        The store directory to read.
    full : bool
        Include the LightRAG indexes as well as the scoring inputs.

    Returns
    -------
    list of Path
        Relative paths, sorted, so a bundle built twice lists files in one
        order and its manifest compares cleanly.

    Raises
    ------
    FileNotFoundError
        If none of the selected top-level entries exist, which means the path
        given is not a null-arm store.
    """
    wanted = SCORE_ONLY + (INDEXES if full else ())
    found: list[Path] = []
    for name in wanted:
        entry = store / name
        if not entry.exists():
            continue
        if entry.is_file():
            found.append(Path(name))
            continue
        found.extend(item.relative_to(store) for item in sorted(entry.rglob("*")) if item.is_file())
    if not found:
        raise FileNotFoundError(f"{store} holds none of {wanted}, so it is not a null-arm store")
    return sorted(found)


def build_manifest(
    store: Path,
    members: Sequence[Path],
    *,
    full: bool,
    sources: Mapping[Path, Path] | None = None,
) -> dict[str, Any]:
    """Return the manifest describing a bundle.

    The manifest states what the bundle omits as well as what it holds, because
    a reader who does not know that ``corpus/`` re-downloads itself would read
    its absence as a broken bundle. ``sources`` maps a member to the file that
    is packed in its place, such as a snapshot of the response cache, so the
    digest describes the bytes in the archive.
    """
    source = {member: (sources or {}).get(member, store / member) for member in members}
    sizes = {member: source[member].stat().st_size for member in members}
    files: list[dict[str, Any]] = [
        {
            "path": str(member),
            "bytes": sizes[member],
            "sha256": digest(source[member]),
        }
        for member in members
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": "full" if full else "score-only",
        "total_bytes": sum(sizes.values()),
        "holds": list(SCORE_ONLY + (INDEXES if full else ())),
        "omits": list(EXCLUDED if full else EXCLUDED + INDEXES),
        "omitted_because": {
            "corpus": (
                "re-downloads through the digest-pinned ContractNLI loader, so "
                "a second copy would only be another thing to keep honest"
            ),
            "lightrag": ("only needed to retrieve again, which the committed evidence tables make unnecessary"),
        },
        "scores_without_reindexing": True,
        "corpus": {
            "name": "ContractNLI test split, 20 documents",
            "licence": CONTRACT_NLI_LICENCE.name,
            "licence_url": CONTRACT_NLI_LICENCE.url,
            "attribution": CONTRACT_NLI_LICENCE.attribution,
            "note": (
                "The retrieval tables and the response cache quote contract "
                "text, so this attribution travels with the bundle."
            ),
        },
        "files": files,
    }


def write_bundle(store: Path, destination: Path, *, full: bool = False) -> dict[str, Any]:
    """Pack a store into ``destination`` and return the manifest written into it."""
    members = bundle_members(store, full=full)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with tempfile.TemporaryDirectory() as scratch:
        sources = {member: snapshot(store / member, Path(scratch)) for member in members}
        manifest = build_manifest(store, members, full=full, sources=sources)
        manifest_path = Path(scratch) / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with tarfile.open(partial, "w:gz") as archive:
            archive.add(manifest_path, arcname=MANIFEST_NAME)
            for member in members:
                archive.add(sources[member], arcname=str(member))
    # An interrupted export leaves only the partial file, never a truncated
    # archive under the name a reader would verify.
    os.replace(partial, destination)
    return manifest


def snapshot(path: Path, scratch: Path) -> Path:
    """Return a consistent copy of a SQLite file, or ``path`` for any other file.

    The response cache can be open in a running job, and a byte copy of a live
    SQLite file can be torn. The online backup API copies a consistent state.
    """
    if path.suffix != ".sqlite":
        return path
    handle, name = tempfile.mkstemp(dir=scratch, suffix=path.name)
    os.close(handle)
    copy = Path(name)
    with closing(sqlite3.connect(str(path))) as source, closing(sqlite3.connect(str(copy))) as target:
        source.backup(target)
    return copy


def read_manifest(archive: Path) -> dict[str, Any]:
    """Return the manifest stored inside a bundle.

    Raises
    ------
    FileNotFoundError
        If the bundle carries no manifest, so nothing can be verified.
    """
    with tarfile.open(archive, "r:gz") as handle:
        try:
            extracted = handle.extractfile(MANIFEST_NAME)
        except KeyError as error:  # pragma: no cover - tarfile raises per member
            raise FileNotFoundError(f"{archive} holds no {MANIFEST_NAME}") from error
        if extracted is None:
            raise FileNotFoundError(f"{archive} holds no {MANIFEST_NAME}")
        loaded: dict[str, Any] = json.loads(extracted.read().decode("utf-8"))
    return loaded


def verify_bundle(archive: Path) -> list[str]:
    """Return the paths in a bundle whose contents do not match the manifest.

    Returns
    -------
    list of str
        Empty when every file matches. A path appears once whether it is
        missing or altered.
    """
    manifest = read_manifest(archive)
    expected = {str(entry["path"]): str(entry["sha256"]) for entry in manifest["files"]}
    seen: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or not safe_member_name(member.name):
                # Links, devices, and names that escape the unpack directory are
                # reported as mismatches, so extraction is never the first check.
                seen[member.name] = ""
                continue
            if member.name == MANIFEST_NAME:
                continue
            stream = handle.extractfile(member)
            if stream is None:  # pragma: no cover - directories are filtered above
                continue
            seen[member.name] = digest_stream(stream)
    return sorted(path for path in expected if seen.get(path) != expected[path]) + sorted(
        path for path in seen if path not in expected
    )


def safe_member_name(name: str) -> bool:
    """Return whether an archive member name stays inside the unpack directory."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def unpacked_mismatches(store: Path) -> list[str]:
    """Return the files in an unpacked bundle that no longer match its manifest.

    Verifying the archive says nothing about the directory after unpacking, so a
    file edited, truncated, or removed afterwards is caught here.

    Parameters
    ----------
    store : Path
        Directory a bundle was unpacked into, holding its ``MANIFEST.json``.

    Returns
    -------
    list of str
        Manifest paths that are missing or altered. Empty when all match.
    """
    manifest = json.loads((store / MANIFEST_NAME).read_text(encoding="utf-8"))
    bad = []
    for entry in manifest["files"]:
        path = store / str(entry["path"])
        if not path.is_file() or digest(path) != str(entry["sha256"]):
            bad.append(str(entry["path"]))
    return sorted(bad)


def _humanise(total: int) -> str:
    """Return a byte count in the largest unit that leaves it above one."""
    size = float(total)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - the loop returns first


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=default_store())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="include the LightRAG indexes, which scoring does not need",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="check an existing bundle against its manifest and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify a bundle. Return a process exit status."""
    args = parse_args(argv)

    if args.verify is not None:
        bad = verify_bundle(args.verify)
        if bad:
            print(f"verify: {len(bad)} file(s) do not match the manifest", file=sys.stderr)
            for path in bad:
                print(f"  {path}", file=sys.stderr)
            return 1
        manifest = read_manifest(args.verify)
        print(
            f"verify: {len(manifest['files'])} files match, "
            f"{_humanise(int(manifest['total_bytes']))}, {manifest['bundle']} bundle"
        )
        print("Unpack and score from the repository root:")
        print("  mkdir -p unpacked")
        print(f"  tar -xzf {shlex.quote(str(args.verify))} -C unpacked")
        print("  python implementations/regulatory_policy_qa/graphrag_null/run_null_arm.py --store ./unpacked")
        return 0

    if args.out is None:
        print("--out is required when building a bundle", file=sys.stderr)
        return 2

    manifest = write_bundle(args.store, args.out, full=args.full)
    packed = _humanise(int(manifest["total_bytes"]))
    print(f"wrote {args.out} ({manifest['bundle']}, {len(manifest['files'])} files, {packed})")
    print(f"omits: {', '.join(manifest['omits'])}")
    print(
        f"verify it with: python implementations/regulatory_policy_qa/graphrag_null/export_store.py --verify {shlex.quote(str(args.out))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
