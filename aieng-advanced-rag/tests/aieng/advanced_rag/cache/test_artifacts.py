"""Behavioral tests for the reusable global-plus-local RAG artifact cache."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pytest

from aieng.advanced_rag.cache import (
    CacheIntegrityError,
    CacheKey,
    JsonValue,
    TieredResultCache,
)
from aieng.advanced_rag.metering import UsageMeter


def _key(config_value: str = "standard") -> CacheKey:
    return CacheKey(
        ri_id="test-ri",
        artifact_type="index",
        producer_id="test-predictor",
        config={"config": config_value},
    )


def test_global_hit_skips_compute(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    cache.load_or_compute(_key(), lambda: {"result": "shipped"}, write_tier="global")
    calls = 0

    def compute() -> JsonValue:
        nonlocal calls
        calls += 1
        return {"result": "unexpected"}

    result = cache.load_or_compute(_key(), compute)
    assert result.value == {"result": "shipped"}
    assert result.tier == "global"
    assert result.cache_hit is True
    assert calls == 0


def test_changed_config_computes_locally_under_same_layout(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    cache.load_or_compute(_key(), lambda: {"result": "shipped"}, write_tier="global")
    changed = cache.load_or_compute(_key("changed"), lambda: {"result": "local"})
    assert changed.value == {"result": "local"}
    assert changed.tier == "local"
    assert changed.cache_hit is False
    assert changed.path.relative_to(cache.local_dir) == _key("changed").relative_path


def test_cache_key_rejects_accidental_credentials() -> None:
    try:
        CacheKey(
            ri_id="test-ri",
            artifact_type="eval",
            producer_id="runner",
            config={"provider": {"api_key": "do-not-store"}},
        )
    except ValueError as exc:
        assert "must not contain secret field" in str(exc)
    else:
        raise AssertionError("secret-bearing cache configuration was accepted")


def test_force_refresh_overrides_global_with_local_result(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    cache.load_or_compute(_key(), lambda: {"version": 1}, write_tier="global")
    refreshed = cache.load_or_compute(
        _key(), lambda: {"version": 2}, force_refresh=True
    )
    loaded = cache.load_or_compute(_key(), lambda: {"version": 3})
    assert refreshed.cache_hit is False
    assert refreshed.tier == "local"
    assert loaded.value == {"version": 2}
    assert loaded.tier == "local"


def test_completed_items_survive_a_later_compute_crash(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    keys = [_key(str(index)) for index in range(3)]
    cache.load_or_compute(keys[0], lambda: {"item": 0})

    def crash() -> JsonValue:
        raise RuntimeError("simulated interruption")

    try:
        cache.load_or_compute(keys[1], crash)
    except RuntimeError as exc:
        assert str(exc) == "simulated interruption"
    else:
        raise AssertionError("the simulated interruption did not occur")

    resumed = cache.load_or_compute(keys[0], lambda: {"item": "unexpected"})
    assert resumed.value == {"item": 0}
    assert resumed.cache_hit is True
    assert len(list(cache.local_dir.rglob("*.json"))) == 1


def test_export_local_copies_user_results_to_a_fork_store(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    result = cache.load_or_compute(_key("personal"), lambda: {"score": 0.9})
    destination = tmp_path / "fork-cache"
    written = cache.export_local(destination)
    assert written == [destination / result.path.relative_to(cache.local_dir)]
    assert written[0].read_bytes() == result.path.read_bytes()


def test_binary_artifact_is_atomic_validated_and_exported(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )

    def build(path: Path) -> JsonValue:
        path.write_bytes(b"portable qdrant archive")
        return {"collection": "financebench", "points": 3}

    result = cache.load_or_compute_artifact(
        _key("binary"), build, media_type="application/gzip"
    )
    loaded = cache.get_artifact(_key("binary"))
    assert loaded is not None
    assert loaded.cache_hit is True
    assert loaded.metadata == {"collection": "financebench", "points": 3}
    assert loaded.path.read_bytes() == b"portable qdrant archive"

    destination = tmp_path / "fork-cache"
    written = cache.export_local(destination)
    assert destination / result.path.relative_to(cache.local_dir) in written
    assert destination / result.envelope_path.relative_to(cache.local_dir) in written


def test_binary_artifact_does_not_publish_envelope_after_crash(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )

    def crash(path: Path) -> JsonValue:
        path.write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    try:
        cache.load_or_compute_artifact(_key("crash"), crash)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the simulated interruption did not occur")
    assert cache.get_artifact(_key("crash")) is None


def test_fetch_global_verifies_checksum_and_key(tmp_path: Path) -> None:
    source_cache = TieredResultCache(
        global_dir=tmp_path / "source", local_dir=tmp_path / "unused"
    )
    source = source_cache.load_or_compute(
        _key(), lambda: {"large": "artifact"}, write_tier="global"
    ).path
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    target_cache = TieredResultCache(
        global_dir=tmp_path / "empty-global",
        local_dir=tmp_path / "target-local",
    )
    installed = target_cache.fetch_global(_key(), source.as_uri(), sha256=checksum)
    loaded = target_cache.get(_key())
    assert installed.is_file()
    assert loaded is not None
    assert loaded.value == {"large": "artifact"}
    assert loaded.tier == "global"


def test_fetch_global_artifact_verifies_envelope_and_payload(tmp_path: Path) -> None:
    source_cache = TieredResultCache(
        global_dir=tmp_path / "source", local_dir=tmp_path / "unused"
    )

    def build(path: Path) -> JsonValue:
        path.write_bytes(b"large qdrant archive")
        return {"collection": "financebench"}

    source = source_cache.load_or_compute_artifact(
        _key("remote-binary"), build, write_tier="global"
    )
    target_cache = TieredResultCache(
        global_dir=tmp_path / "empty", local_dir=tmp_path / "target"
    )
    installed = target_cache.fetch_global_artifact(
        _key("remote-binary"),
        source.envelope_path.as_uri(),
        source.path.as_uri(),
        envelope_sha256=hashlib.sha256(source.envelope_path.read_bytes()).hexdigest(),
        payload_sha256=hashlib.sha256(source.path.read_bytes()).hexdigest(),
    )
    loaded = target_cache.get_artifact(_key("remote-binary"))
    assert installed.is_file()
    assert loaded is not None
    assert loaded.metadata == {"collection": "financebench"}
    assert loaded.path.read_bytes() == b"large qdrant archive"


def test_fetch_manifest_installs_multiple_artifacts(tmp_path: Path) -> None:
    source_cache = TieredResultCache(
        global_dir=tmp_path / "source", local_dir=tmp_path / "unused"
    )
    source = source_cache.load_or_compute(
        _key(), lambda: {"large": "artifact"}, write_tier="global"
    ).path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "key": {
                            "ri_id": _key().ri_id,
                            "artifact_type": _key().artifact_type,
                            "producer_id": _key().producer_id,
                            "config": _key().config,
                        },
                        "url": source.as_uri(),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_cache = TieredResultCache(
        global_dir=tmp_path / "empty", local_dir=tmp_path / "target"
    )
    installed = target_cache.fetch_manifest(manifest)
    assert installed == [target_cache.fetched_global_dir / _key().relative_path]
    assert target_cache.get(_key()) is not None


def _written_envelope(tmp_path: Path) -> Path:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    return cache.load_or_compute(_key(), lambda: {"result": "shipped"}).path


def _reload(tmp_path: Path, path: Path, fields: dict[str, object]) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    body.update(fields)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    cache.get(_key())


def test_an_envelope_field_this_reader_does_not_know_is_rejected(
    tmp_path: Path,
) -> None:
    path = _written_envelope(tmp_path)
    with pytest.raises(CacheIntegrityError, match="unknown fields: tier"):
        _reload(tmp_path, path, {"tier": "global"})


def test_an_envelope_from_another_schema_version_is_rejected(tmp_path: Path) -> None:
    path = _written_envelope(tmp_path)
    with pytest.raises(CacheIntegrityError, match="schema_version 2"):
        _reload(tmp_path, path, {"schema_version": 2})


def test_a_non_string_cache_key_is_rejected(tmp_path: Path) -> None:
    path = _written_envelope(tmp_path)
    with pytest.raises(CacheIntegrityError, match="cache_key must be a string"):
        _reload(tmp_path, path, {"cache_key": 17})


def test_a_warm_run_counts_one_cache_hit_and_no_call(tmp_path: Path) -> None:
    cache = TieredResultCache(
        global_dir=tmp_path / "global", local_dir=tmp_path / "local"
    )
    meter = UsageMeter()
    with meter.question("q1"):
        cache.load_or_compute(_key(), lambda: {"result": "cold"})
    with meter.question("q2"):
        warm = cache.load_or_compute(_key(), lambda: {"result": "unexpected"})

    assert warm.cache_hit is True
    assert warm.value == {"result": "cold"}
    assert meter.per_question()["q1"].cache_hits == 0
    assert meter.per_question()["q2"].cache_hits == 1
    assert meter.totals().cache_hits == 1
    assert meter.totals().calls == 0


def test_fetch_global_artifact_streams_the_payload_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aieng.advanced_rag.cache import artifacts

    source_cache = TieredResultCache(
        global_dir=tmp_path / "source", local_dir=tmp_path / "unused"
    )
    body = bytes(range(256)) * 40

    def build(path: Path) -> JsonValue:
        path.write_bytes(body)
        return {"collection": "streamed"}

    source = source_cache.load_or_compute_artifact(
        _key("streamed"), build, write_tier="global"
    )
    monkeypatch.setattr(artifacts, "_HASH_BLOCK_BYTES", 1000)
    reads: list[int] = []
    real_open = urllib.request.urlopen

    def spy(url: str, timeout: float = 0):
        response = real_open(url, timeout=timeout)
        original = response.read

        def read(size: int = -1) -> bytes:
            data = original(size)
            reads.append(size)
            return data

        response.read = read  # type: ignore[method-assign]
        return response

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", spy)
    target = TieredResultCache(
        global_dir=tmp_path / "empty", local_dir=tmp_path / "target"
    )
    target.fetch_global_artifact(
        _key("streamed"),
        source.envelope_path.as_uri(),
        source.path.as_uri(),
        envelope_sha256=hashlib.sha256(source.envelope_path.read_bytes()).hexdigest(),
        payload_sha256=hashlib.sha256(body).hexdigest(),
    )
    loaded = target.get_artifact(_key("streamed"))
    assert loaded is not None
    assert loaded.path.read_bytes() == body
    # The payload came in blocks of the configured size, never as one read().
    assert 1000 in reads
    assert reads.count(1000) >= len(body) // 1000


def test_a_corrupt_streamed_payload_leaves_nothing_installed(tmp_path: Path) -> None:
    source_cache = TieredResultCache(
        global_dir=tmp_path / "source", local_dir=tmp_path / "unused"
    )

    def build(path: Path) -> JsonValue:
        path.write_bytes(b"payload")
        return {}

    source = source_cache.load_or_compute_artifact(
        _key("corrupt"), build, write_tier="global"
    )
    target = TieredResultCache(
        global_dir=tmp_path / "empty", local_dir=tmp_path / "target"
    )
    with pytest.raises(CacheIntegrityError, match="payload checksum mismatch"):
        target.fetch_global_artifact(
            _key("corrupt"),
            source.envelope_path.as_uri(),
            source.path.as_uri(),
            envelope_sha256=hashlib.sha256(
                source.envelope_path.read_bytes()
            ).hexdigest(),
            payload_sha256="0" * 64,
        )
    assert target.get_artifact(_key("corrupt")) is None
    leftovers = [p for p in (tmp_path / "target").rglob("*") if p.is_file()]
    assert leftovers == []


def test_artifact_hashing_does_not_read_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aieng.advanced_rag.cache import artifacts

    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 5000)
    monkeypatch.setattr(artifacts, "_HASH_BLOCK_BYTES", 512)
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("read_bytes must not be used")
    )
    assert artifacts._sha256_of_file(path) == hashlib.sha256(b"x" * 5000).hexdigest()
