"""Directories as cache artifacts: round trip, determinism, and safe extraction."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from aieng.advanced_rag.cache import (
    CacheKey,
    TieredResultCache,
    fetch_directory,
    pack_directory,
    store_directory,
    tree_digest,
)


def key(config: str = "a") -> CacheKey:
    return CacheKey(
        ri_id="ri4",
        artifact_type="graph-index",
        producer_id="test",
        config={"build": config},
    )


@pytest.fixture
def cache(tmp_path: Path) -> TieredResultCache:
    return TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )


@pytest.fixture
def index(tmp_path: Path) -> Path:
    directory = tmp_path / "index"
    (directory / "vdb").mkdir(parents=True)
    (directory / "empty").mkdir()
    (directory / "graph.graphml").write_text("<graphml/>")
    (directory / "vdb" / "vectors.bin").write_bytes(bytes(range(256)) * 40)
    return directory


def snapshot(directory: Path) -> dict[str, bytes | None]:
    return {
        p.relative_to(directory).as_posix(): None if p.is_dir() else p.read_bytes()
        for p in sorted(directory.rglob("*"))
    }


def test_a_directory_round_trips_through_the_cache(cache, index, tmp_path):
    result = store_directory(cache, key(), index)
    assert result.cache_hit is False
    assert result.metadata["files"] == 2 and result.metadata["directories"] == 2
    out = fetch_directory(cache, key(), tmp_path / "restored")
    assert out == tmp_path / "restored"
    assert snapshot(out) == snapshot(index)


def test_an_unknown_key_fetches_nothing(cache, tmp_path):
    assert fetch_directory(cache, key("missing"), tmp_path / "x") is None
    assert not (tmp_path / "x").exists()


def test_the_second_store_is_a_hit_that_does_not_read_the_directory(
    cache, index, tmp_path
):
    store_directory(cache, key(), index)
    missing = tmp_path / "gone"
    again = store_directory(cache, key(), missing)
    assert again.cache_hit is True


def test_force_refresh_packs_again(cache, index):
    store_directory(cache, key(), index)
    (index / "graph.graphml").write_text("<graphml>changed</graphml>")
    result = store_directory(cache, key(), index, force_refresh=True)
    assert result.cache_hit is False


def test_the_archive_bytes_depend_on_the_files_and_nothing_else(index, tmp_path):
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    pack_directory(index, first)
    for path in index.rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
        path.chmod(0o755 if path.is_dir() else 0o600)
    pack_directory(index, second)
    assert first.read_bytes() == second.read_bytes()


def test_the_tree_digest_ignores_time_and_changes_with_content(index, tmp_path):
    before = tree_digest(index)
    for path in index.rglob("*"):
        os.utime(path, (5, 5))
    assert tree_digest(index) == before
    (index / "graph.graphml").write_text("<graphml>other</graphml>")
    assert tree_digest(index) != before


def test_a_symbolic_link_is_refused(index, tmp_path):
    (index / "link").symlink_to(index / "graph.graphml")
    with pytest.raises(ValueError, match="symbolic link"):
        pack_directory(index, tmp_path / "x.tar.gz")


def test_an_existing_destination_is_an_error_unless_replaced(cache, index, tmp_path):
    store_directory(cache, key(), index)
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "stale.txt").write_text("old")
    with pytest.raises(FileExistsError):
        fetch_directory(cache, key(), destination)
    fetch_directory(cache, key(), destination, replace=True)
    assert not (destination / "stale.txt").exists()
    assert (destination / "graph.graphml").exists()


def malicious_archive(
    tmp_path: Path, member: tarfile.TarInfo, data: bytes = b""
) -> Path:
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(member, io.BytesIO(data))
    return path


def store_archive(cache: TieredResultCache, archive: Path) -> CacheKey:
    evil = key("evil")
    cache.load_or_compute_artifact(
        evil, lambda path: (path.write_bytes(archive.read_bytes()), {})[1]
    )
    return evil


def test_a_path_that_escapes_the_destination_is_refused(cache, tmp_path):
    info = tarfile.TarInfo("../outside.txt")
    info.size = 3
    evil = store_archive(cache, malicious_archive(tmp_path, info, b"bad"))
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="outside the destination"):
        fetch_directory(cache, evil, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_a_link_or_device_in_an_archive_is_refused(cache, tmp_path):
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    evil = store_archive(cache, malicious_archive(tmp_path, info))
    with pytest.raises(ValueError, match="not a file or a directory"):
        fetch_directory(cache, evil, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_a_failed_extraction_leaves_no_partial_destination_or_staging(cache, tmp_path):
    info = tarfile.TarInfo("../outside.txt")
    info.size = 1
    evil = store_archive(cache, malicious_archive(tmp_path, info, b"x"))
    with pytest.raises(ValueError):
        fetch_directory(cache, evil, tmp_path / "out" / "restored")
    assert [p.name for p in (tmp_path / "out").iterdir()] == []


def test_a_directory_artifact_moves_between_caches_like_any_other(
    cache, index, tmp_path
):
    store_directory(cache, key(), index)
    fork = tmp_path / "fork"
    cache.export_local(fork)
    other = TieredResultCache(global_dir=tmp_path / "g2", local_dir=fork)
    out = fetch_directory(other, key(), tmp_path / "from-fork")
    assert snapshot(out) == snapshot(index)
