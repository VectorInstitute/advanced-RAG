"""Store a directory, such as a built index, as one artifact in the cache.

A graph index is a directory of files: a vector store, a graph, and the tables
that tie them together. :meth:`TieredResultCache.load_or_compute_artifact` keeps
one file, so a directory goes in as a tar archive. The archive is deterministic,
which means the same files always produce the same bytes and so the same digest,
and the cache's own payload check then covers the whole directory.

:func:`fetch_directory` extracts into a temporary sibling and renames it into
place, so a crash leaves nothing at the destination, and it refuses an archive
that would write outside it. That makes an archive from someone else's cache safe
to extract.

Because the result is an ordinary artifact, ``export_local``, ``push_local``, and
``fetch_manifest`` move it between caches like any other.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import ArtifactResult, CacheKey, CacheTier, JsonValue, TieredResultCache


MEDIA_TYPE = "application/x-tar+gzip"

# A fixed timestamp and fixed ownership, so an archive's bytes depend on the
# files and nothing else.
_EPOCH = 0
_FILE_MODE = 0o644
_DIR_MODE = 0o755
_GZIP_LEVEL = 3


def _entries(directory: Path) -> list[Path]:
    """Return every file and directory under ``directory``, in sorted order."""
    found = sorted(
        directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()
    )
    for path in found:
        if path.is_symlink():
            raise ValueError(
                f"{path} is a symbolic link, which an archive does not keep"
            )
    return found


def tree_digest(directory: Path) -> str:
    """Return a digest of a directory's names and file contents.

    The digest depends on relative paths and bytes and not on timestamps,
    permissions, or the order the files were written, so a directory and its
    extracted copy have the same one.
    """
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in _entries(directory):
        relative = path.relative_to(directory).as_posix()
        if path.is_dir():
            digest.update(b"d\0" + relative.encode() + b"\0")
            continue
        digest.update(b"f\0" + relative.encode() + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def pack_directory(directory: Path, archive: Path) -> dict[str, JsonValue]:
    """Write a deterministic tar.gz of ``directory`` to ``archive``.

    Parameters
    ----------
    directory : Path
        The directory to pack. Symbolic links are refused.
    archive : Path
        Where to write the archive.

    Returns
    -------
    dict
        ``files``, ``directories``, ``bytes`` (of the unpacked files), and the
        ``tree_digest`` that :func:`fetch_directory` verifies after extracting.

    Raises
    ------
    NotADirectoryError
        If ``directory`` is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    files = directories = size = 0
    with (
        archive.open("wb") as raw,
        # An empty filename keeps the archive's own path out of the gzip header.
        gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            mtime=_EPOCH,
            compresslevel=_GZIP_LEVEL,
        ) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for path in _entries(directory):
            info = tar.gettarinfo(
                str(path), arcname=path.relative_to(directory).as_posix()
            )
            info.mtime, info.uid, info.gid, info.uname, info.gname = (
                _EPOCH,
                0,
                0,
                "",
                "",
            )
            if path.is_dir():
                info.mode = _DIR_MODE
                directories += 1
                tar.addfile(info)
            else:
                info.mode = _FILE_MODE
                files += 1
                size += info.size
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
    return {
        "files": files,
        "directories": directories,
        "bytes": size,
        "tree_digest": tree_digest(directory),
    }


def store_directory(
    cache: TieredResultCache,
    key: CacheKey,
    directory: Path,
    *,
    force_refresh: bool = False,
    write_tier: CacheTier = "local",
) -> ArtifactResult:
    """Store ``directory`` as an artifact, or return the one already stored.

    Parameters
    ----------
    cache : TieredResultCache
        The cache to store into.
    key : CacheKey
        The artifact's identity. Put whatever makes two builds different into
        its ``config``, such as the model, the embedder, and the corpus pin.
    directory : Path
        The directory to pack. It is read only when the key is not already
        stored, or when ``force_refresh`` is true.
    force_refresh : bool, optional
        Pack and replace even when a result exists.
    write_tier : {"local", "global"}, optional
        Which tier to write.

    Returns
    -------
    ArtifactResult
        ``metadata`` holds the counts and the tree digest.
    """

    def compute(path: Path) -> JsonValue:
        return pack_directory(directory, path)

    return cache.load_or_compute_artifact(
        key,
        compute,
        media_type=MEDIA_TYPE,
        force_refresh=force_refresh,
        write_tier=write_tier,
    )


def _checked_members(tar: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    """Return the archive's members after refusing anything unsafe to extract."""
    root = destination.resolve()
    members = tar.getmembers()
    for member in members:
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(
                f"archive member {member.name!r} would write outside the destination"
            )
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"archive member {member.name!r} is not a file or a directory"
            )
    return members


def fetch_directory(
    cache: TieredResultCache,
    key: CacheKey,
    destination: Path,
    *,
    replace: bool = False,
) -> Path | None:
    """Extract a stored directory to ``destination``, atomically.

    Parameters
    ----------
    cache : TieredResultCache
        The cache holding the artifact.
    key : CacheKey
        The artifact's identity.
    destination : Path
        Where the directory should appear. Its parent is created.
    replace : bool, optional
        Replace ``destination`` when it exists. Without it an existing
        destination is an error, so a stale index is never mixed with a new one.

    Returns
    -------
    Path or None
        ``destination``, or ``None`` when the cache holds no such artifact.

    Raises
    ------
    FileExistsError
        If ``destination`` exists and ``replace`` is false.
    ValueError
        If the archive holds a link, a device, or a path outside the
        destination, or the extracted tree does not match the stored digest.
    """
    result = cache.get_artifact(key)
    if result is None:
        return None
    destination = Path(destination)
    if destination.exists() and not replace:
        raise FileExistsError(
            f"{destination} already exists; pass replace=True to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        with tarfile.open(result.path, mode="r:gz") as tar:
            # Python before 3.11.4 has no extraction filter, and the member checks
            # above cover what the filter would refuse.
            options: dict[str, Any] = (
                {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            )
            for member in _checked_members(tar, staging):
                tar.extract(member, path=staging, set_attrs=False, **options)
        expected = _expected_digest(result.metadata)
        if expected is not None and tree_digest(staging) != expected:
            raise ValueError(
                "the extracted directory does not match the stored tree digest"
            )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        staging = None  # type: ignore[assignment]
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return destination


def _expected_digest(metadata: JsonValue) -> str | None:
    if isinstance(metadata, dict):
        value: Any = metadata.get("tree_digest")
        return str(value) if value else None
    return None


__all__ = [
    "MEDIA_TYPE",
    "fetch_directory",
    "pack_directory",
    "store_directory",
    "tree_digest",
]
