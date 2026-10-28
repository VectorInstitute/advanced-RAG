"""Pinned, checksum-verified download records for the RI #1 corpora.

Every corpus this package loads is addressed by an immutable URL and a SHA-256
digest that :func:`fetch` verifies on each run. A dataset that changes upstream
therefore fails loudly instead of silently altering a measurement. The
ServerFault dump download follows the same convention.

Licence facts live next to the download record rather than in a document
alone, so anything that redistributes corpus text can read the terms it is
bound by. `implementations/regulatory_policy_qa/CORPUS.md` carries the sourcing
and the reasoning behind each verdict.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from aieng.advanced_rag.cache.artifacts import digest_mismatch
from aieng.advanced_rag.cache.paths import corpora_dir


_CACHE_ENV_VAR = "ADVANCED_RAG_CORPUS_CACHE"
_DOWNLOAD_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Licence:
    """The terms a corpus is redistributed under.

    Parameters
    ----------
    name : str
        Licence name, including version.
    url : str
        Canonical URL for the licence text. Cite this, not a summary.
    permits_commercial_use : bool
        What the licence text says about use in a commercial context. This
        records what the licence text says and makes no judgement about any
        particular use.
    attribution : str
        The credit line that has to travel with any redistributed text.
    notes : str
        Anything the licence name alone does not convey, such as rights in the
        underlying content that the licence does not reach.
    """

    name: str
    url: str
    permits_commercial_use: bool
    attribution: str
    notes: str = ""


@dataclass(frozen=True)
class PinnedSource:
    """One immutable file, addressed by URL and digest.

    Parameters
    ----------
    name : str
        Short identifier used for the on-disk cache filename.
    url : str
        Immutable URL. A branch name or a ``main`` reference is not immutable
        and does not belong here; use a commit SHA, a revision SHA, or a DOI.
    sha256 : str
        Digest of the bytes at ``url``.
    licence : Licence
        Terms the downloaded bytes carry.
    member : str or None, optional
        Path inside the archive to extract, when ``url`` points at a zip.
    member_sha256 : str or None, optional
        Digest of the extracted member.
    """

    name: str
    url: str
    sha256: str
    licence: Licence
    member: str | None = None
    member_sha256: str | None = None


class ChecksumMismatchError(RuntimeError):
    """Downloaded bytes do not match the pinned digest."""


def cache_dir(override: Path | None = None) -> Path:
    """Return the directory corpora are cached in, creating it if needed.

    Resolution order: the ``override`` argument, then the
    ``ADVANCED_RAG_CORPUS_CACHE`` environment variable, then
    :func:`aieng.advanced_rag.cache.paths.corpora_dir`, which is
    ``corpora`` under the one cache root.

    ``ADVANCED_RAG_CORPUS_CACHE`` keeps precedence over the shared root so that
    a corpus already downloaded to wherever that variable points stays where it
    is instead of being re-fetched under the new default.
    """
    if override is not None:
        resolved = Path(override)
    else:
        env_value = os.environ.get(_CACHE_ENV_VAR)
        resolved = Path(env_value) if env_value else corpora_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def verify(payload: bytes, expected_sha256: str, *, what: str) -> None:
    """Raise :class:`ChecksumMismatchError` unless ``payload`` hashes as expected."""
    actual = digest_mismatch(payload, expected_sha256)
    if actual is not None:
        raise ChecksumMismatchError(
            f"{what}: expected sha256 {expected_sha256}, got {actual}. "
            "The upstream file changed. Do not use it until CORPUS.md is "
            "updated with the new digest and the licence rechecked."
        )


def cached_path(source: PinnedSource, *, dest: Path | None = None) -> Path:
    """Return where :func:`fetch` keeps ``source``, without downloading it.

    Parameters
    ----------
    source : PinnedSource
        The record.
    dest : Path or None, optional
        Cache directory. Defaults to :func:`cache_dir`.

    Returns
    -------
    Path
        The extracted member's path when ``source.member`` is set, otherwise
        the downloaded file's. The file may not exist yet.
    """
    suffix = ".json" if source.member and source.member.endswith(".json") else ""
    return cache_dir(dest) / f"{source.name}{suffix or Path(source.url).suffix}"


def is_cached(source: PinnedSource, *, dest: Path | None = None) -> bool:
    """Return whether ``source`` is in the cache and matches its pinned digest.

    Nothing is downloaded, which makes this the check to run before deciding
    whether a fetch is needed.
    """
    target = cached_path(source, dest=dest)
    if not target.is_file():
        return False
    expected = source.member_sha256 if source.member else source.sha256
    if expected is None:
        return True
    return digest_mismatch(target.read_bytes(), expected) is None


def fetch(
    source: PinnedSource,
    *,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """Download ``source``, verify its digest, and return the local path.

    A cached file is verified on every call rather than trusted for its name,
    so a partial or tampered download is caught before it reaches a run.

    Parameters
    ----------
    source : PinnedSource
        The record to fetch.
    dest : Path or None, optional
        Cache directory. Defaults to :func:`cache_dir`.
    force : bool, optional
        Re-download even when a valid cached copy exists.

    Returns
    -------
    Path
        Path to the verified file. When ``source.member`` is set this is the
        extracted member, not the archive.

    Raises
    ------
    ChecksumMismatchError
        If either the archive or the extracted member fails verification.
    """
    target = cached_path(source, dest=dest)

    if target.exists() and not force:
        expected = source.member_sha256 if source.member else source.sha256
        if expected is not None:
            try:
                verify(target.read_bytes(), expected, what=str(target))
            except ChecksumMismatchError:
                target.unlink()
            else:
                return target
        else:
            return target

    payload = _download(source.url)
    verify(payload, source.sha256, what=source.url)

    if source.member is None:
        target.write_bytes(payload)
        return target

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member_bytes = archive.read(source.member)
    if source.member_sha256 is not None:
        verify(member_bytes, source.member_sha256, what=f"{source.url}#{source.member}")
    target.write_bytes(member_bytes)
    return target


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload: bytes = response.read()
    return payload
