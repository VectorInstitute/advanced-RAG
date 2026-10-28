"""Guard: the harness's own embedding path stays local.

Every embedder the harness ships runs on the machine it is invoked on, so a
participant can score a retriever with no API key and no billing account. That
is a property of this package rather than a rule about what a pipeline built on
it may use, and it is easy to lose by accident: a hosted embedding call once
appeared inside a third-party library's embedding hook and went unnoticed.

The test reads the shipped sources rather than running them, so it costs
nothing and catches the reference before it can be called.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aieng.advanced_rag import retrieval


ROOT = Path(retrieval.__file__).resolve().parent

# Provider-hosted embedding model families, and the OpenAI SDK embeddings call.
HOSTED = (
    re.compile(r"text-embedding-\d"),
    re.compile(r"text-embedding-ada"),
    re.compile(r"\bembeddings\s*\.\s*create\b"),
    re.compile(r"voyage-|cohere\.embed|embed-english-v"),
)


def python_sources() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )


def test_there_are_sources_to_check():
    # A path bug that found nothing would make the assertion below vacuous.
    assert len(python_sources()) >= 8


@pytest.mark.parametrize("pattern", HOSTED, ids=lambda p: p.pattern)
def test_no_hosted_embedding_reference(pattern):
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in python_sources()
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        f"hosted embedding reference ({pattern.pattern}) in: {offenders}. "
        "The harness's embedders run locally so it needs no API key."
    )


def test_every_shipped_embedder_runs_locally():
    # HashEmbedder needs nothing at all; SentenceTransformerEmbedder loads a
    # checkpoint and runs it in-process. Neither takes a base URL or a key.
    from aieng.advanced_rag.retrieval.embedders import (
        HashEmbedder,
        SentenceTransformerEmbedder,
    )

    for embedder in (HashEmbedder, SentenceTransformerEmbedder):
        parameters = set(embedder.__init__.__code__.co_varnames)
        assert not parameters & {"api_key", "base_url", "endpoint", "organization"}
