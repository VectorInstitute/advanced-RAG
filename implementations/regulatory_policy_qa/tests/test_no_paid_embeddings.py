"""Embeddings stay local. The proxy is for chat completions only.

The repo carries this test in every experiment after a graph library defaulted
to a paid embedding model through the proxy and billed 39 calls before anyone
noticed. See ``docs/models-and-spend.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "sweep_lab"
SOURCES = sorted(PACKAGE.glob("*.py")) + sorted(ROOT.glob("run_*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_embedding_call_through_an_api_client(path: Path) -> None:
    """No source in this experiment asks a hosted API for an embedding."""
    source = path.read_text(encoding="utf-8")
    assert "embeddings.create" not in source
    assert not re.search(r"text-embedding-\w+", source)


def test_every_embedding_model_named_is_a_local_one() -> None:
    """The only embedding model named is the local sentence-transformers one."""
    named: set[str] = set()
    for path in SOURCES:
        named |= set(re.findall(r'"([\w./-]+/[\w.-]+)"', path.read_text()))
    models = {n for n in named if "embed" in n.lower() or n.startswith("BAAI/")}
    assert models <= {"BAAI/bge-small-en-v1.5"}, f"unexpected: {models}"


def test_the_proxy_client_disables_thinking() -> None:
    """Reasoning is off on every call this experiment makes."""
    # The default proxy model bills reasoning as completion tokens, so a short
    # structured call with thinking on costs about 80x what it should. The
    # switch lives in ProxyBatch, which this client sends through, and is on
    # unless a caller passes thinking=True.
    source = (PACKAGE / "proxy.py").read_text(encoding="utf-8")
    assert "ProxyBatch(" in source
    assert "thinking=True" not in source
    assert "chat.completions.create" not in source
