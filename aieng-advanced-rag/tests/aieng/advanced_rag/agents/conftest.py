"""Keep every agent test's response cache inside the test's own directory."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_cache_root(tmp_path, monkeypatch):
    """Point the package cache root at a fresh directory for each test.

    ``AgentLoop`` caches responses under the cache root by default, so without
    this a test would read responses another test stored and send nothing.
    """
    monkeypatch.setenv("ADVANCED_RAG_CACHE_DIR", str(tmp_path / "cache-root"))
