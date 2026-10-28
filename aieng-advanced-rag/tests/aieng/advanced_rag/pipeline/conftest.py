"""Shared fakes for the pipeline tests.

Every stage takes its model access and its retrieval as callables, so nothing
here needs a client, an index, or a network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance


@pytest.fixture
def make_evidence() -> Callable[..., list[Evidence]]:
    """Return a factory building evidence with predictable chunk identifiers."""

    def build(texts: Sequence[str], doc_id: str = "doc1") -> list[Evidence]:
        return [
            Evidence(
                text=text,
                provenance=Provenance(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{index:03d}",
                    start=index * 100,
                    end=index * 100 + len(text),
                ),
                score=1.0 - index / 100,
                retriever="fake",
            )
            for index, text in enumerate(texts)
        ]

    return build
