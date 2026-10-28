"""Checksum verification and the pinning discipline the records have to keep."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip(
    "aieng.advanced_rag.retrieval.corpus",
    reason="corpus types have not landed in aieng.advanced_rag.retrieval yet",
)

from aieng.advanced_rag import corpora  # noqa: E402
from aieng.advanced_rag.corpora import sources  # noqa: E402


PAYLOAD = b'{"hello": "world"}'
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()

LICENCE = sources.Licence(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    permits_commercial_use=True,
    attribution="Example dataset, licensed CC BY 4.0.",
)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace the network with a fixed byte string."""

    def _serve(payload: bytes) -> list[str]:
        calls: list[str] = []

        def fake_urlopen(url: str, timeout: float = 0) -> _FakeResponse:
            calls.append(url)
            return _FakeResponse(payload)

        monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
        return calls

    return _serve


def _zip_of(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def test_fetch_writes_verified_bytes(tmp_path: Path, serve: Any) -> None:
    serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )
    path = sources.fetch(source, dest=tmp_path)
    assert path.read_bytes() == PAYLOAD


def test_fetch_rejects_a_changed_upstream(tmp_path: Path, serve: Any) -> None:
    serve(b"different bytes")
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )
    with pytest.raises(sources.ChecksumMismatchError, match="expected sha256"):
        sources.fetch(source, dest=tmp_path)


def test_a_cached_file_is_reused_without_a_download(tmp_path: Path, serve: Any) -> None:
    calls = serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )
    sources.fetch(source, dest=tmp_path)
    sources.fetch(source, dest=tmp_path)
    assert len(calls) == 1


def test_a_corrupt_cached_file_is_replaced(tmp_path: Path, serve: Any) -> None:
    calls = serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )
    path = sources.fetch(source, dest=tmp_path)
    path.write_bytes(b"truncated")
    assert sources.fetch(source, dest=tmp_path).read_bytes() == PAYLOAD
    assert len(calls) == 2


def test_force_refetches(tmp_path: Path, serve: Any) -> None:
    calls = serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )
    sources.fetch(source, dest=tmp_path)
    sources.fetch(source, dest=tmp_path, force=True)
    assert len(calls) == 2


def test_archive_member_is_extracted_and_verified(tmp_path: Path, serve: Any) -> None:
    archive = _zip_of("inner/test.json", PAYLOAD)
    serve(archive)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.zip",
        sha256=hashlib.sha256(archive).hexdigest(),
        licence=LICENCE,
        member="inner/test.json",
        member_sha256=PAYLOAD_SHA,
    )
    assert sources.fetch(source, dest=tmp_path).read_bytes() == PAYLOAD


def test_a_changed_member_is_rejected(tmp_path: Path, serve: Any) -> None:
    archive = _zip_of("inner/test.json", b"something else")
    serve(archive)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.zip",
        sha256=hashlib.sha256(archive).hexdigest(),
        licence=LICENCE,
        member="inner/test.json",
        member_sha256=PAYLOAD_SHA,
    )
    with pytest.raises(sources.ChecksumMismatchError, match="#inner/test.json"):
        sources.fetch(source, dest=tmp_path)


def test_cache_dir_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADVANCED_RAG_CORPUS_CACHE", str(tmp_path / "elsewhere"))
    assert sources.cache_dir() == tmp_path / "elsewhere"
    assert sources.cache_dir().is_dir()


def test_an_explicit_directory_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADVANCED_RAG_CORPUS_CACHE", str(tmp_path / "ignored"))
    assert sources.cache_dir(tmp_path / "chosen") == tmp_path / "chosen"


@pytest.mark.parametrize("name", sorted(corpora.SOURCES))
def test_every_source_is_pinned_and_licensed(name: str) -> None:
    source = corpora.SOURCES[name]
    assert len(source.sha256) == 64, "an unpinned source is not reproducible"
    assert source.licence.permits_commercial_use, (
        "the registry holds only corpora that permit commercial use. See CORPUS.md."
    )
    assert source.licence.attribution
    assert source.licence.url.startswith("https://")
    for moving in ("/main/", "/master/", "/resolve/main/", "/raw/main/"):
        assert moving not in source.url, f"{name} points at a moving reference"


def test_replacing_a_digest_does_not_mutate_the_record() -> None:
    original = corpora.SOURCES["cuad-test"]
    altered = dataclasses.replace(original, sha256="0" * 64)
    assert original.sha256 != altered.sha256


def test_cached_path_is_where_fetch_writes(tmp_path: Path, serve: Any) -> None:
    serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )

    assert sources.cached_path(source, dest=tmp_path) == sources.fetch(
        source, dest=tmp_path
    )


def test_is_cached_downloads_nothing_and_checks_the_digest(
    tmp_path: Path, serve: Any
) -> None:
    calls = serve(PAYLOAD)
    source = sources.PinnedSource(
        name="example",
        url="https://example.invalid/a.json",
        sha256=PAYLOAD_SHA,
        licence=LICENCE,
    )

    assert sources.is_cached(source, dest=tmp_path) is False
    assert calls == []
    sources.fetch(source, dest=tmp_path)
    assert sources.is_cached(source, dest=tmp_path) is True
    sources.cached_path(source, dest=tmp_path).write_bytes(b"tampered")
    assert sources.is_cached(source, dest=tmp_path) is False
