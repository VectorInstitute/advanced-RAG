"""The one cache root, and what the shipped Tier-1 directory is allowed to hold.

Three conventions competed before `cache.paths` existed, so these tests pin the
resolved behaviour: one root, one variable that moves it, and the corpus
loader's own variable still winning where it is set. The last two tests pin what
Tier 1 ships, which is nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aieng.advanced_rag.cache import artifacts, paths
from aieng.advanced_rag.corpora import sources


PACKAGE_ROOT = Path(artifacts.__file__).resolve().parents[1]


def test_the_root_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.ENV_VAR, "/tmp/elsewhere")
    assert paths.root() == Path("/tmp/elsewhere")
    assert paths.artifacts_dir() == Path("/tmp/elsewhere/artifacts")
    assert paths.corpora_dir() == Path("/tmp/elsewhere/corpora")


def test_the_root_defaults_under_the_home_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.root() == Path.home() / ".cache" / "aieng-advanced-rag"


def test_the_local_tier_is_the_shared_root_not_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "shared"))
    monkeypatch.chdir(tmp_path)
    cache = artifacts.TieredResultCache()
    assert cache.local_dir == tmp_path / "shared" / "artifacts"
    assert (
        cache.fetched_global_dir == tmp_path / "shared" / "artifacts" / "fetched-global"
    )


def test_the_corpus_cache_follows_the_shared_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ADVANCED_RAG_CORPUS_CACHE", raising=False)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "shared"))
    assert sources.cache_dir() == tmp_path / "shared" / "corpora"


def test_the_corpus_variable_still_wins_over_the_shared_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "shared"))
    monkeypatch.setenv("ADVANCED_RAG_CORPUS_CACHE", str(tmp_path / "already-there"))
    assert sources.cache_dir() == tmp_path / "already-there"


def test_a_corpus_digest_mismatch_still_raises_the_corpus_error() -> None:
    # `digest_mismatch` is shared with the artifact fetch paths, which raise a
    # `ValueError`. The corpus loader has to keep raising its `RuntimeError`.
    with pytest.raises(sources.ChecksumMismatchError):
        sources.verify(b"payload", "0" * 64, what="test")


def test_tier_one_ships_no_envelope() -> None:
    # Tier 1 is deliberately empty. The experiment's `precomputed/` store held
    # `text-embedding-3-small` envelopes, and this project embeds locally only.
    shipped = artifacts.TieredResultCache().global_dir
    assert list(shipped.glob("**/*.json")) == []
    assert list(shipped.glob("**/*.artifact")) == []


def test_no_shipped_file_names_a_paid_embedding_model() -> None:
    # The shipped package embeds with local `bge-small-en-v1.5` only, and a
    # cached envelope keyed on a hosted model would read as a standard artifact
    # a participant is meant to reuse. `implementations/regulatory_policy_qa/
    # tests/test_no_paid_embeddings.py` makes the same check per experiment.
    named = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and re.search(r"text-embedding-\w+", path.read_text(encoding="utf-8"))
    ]
    assert named == [], f"paid embedding model named in: {named}"
