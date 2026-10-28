"""A model that reports no token window has nothing to chunk to.

sentence-transformers types ``max_seq_length`` as ``int | None``. The embedder
turns that window into the token budget chunking uses, so an unset one has to
stop construction with a message saying how to proceed. These tests substitute
the model, so they run without the ``local`` extra and load nothing.
"""

from __future__ import annotations

import sys
import types

import pytest

from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder


class WindowlessModel:
    """A stand-in for a sentence-transformers model whose window is unset."""

    max_seq_length: int | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def get_embedding_dimension(self) -> int:
        return 8

    def modules(self) -> list[object]:
        return []


@pytest.fixture(autouse=True)
def substitute_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = WindowlessModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_a_model_with_no_window_is_refused_with_the_way_out() -> None:
    with pytest.raises(ValueError, match="pass max_seq_length explicitly"):
        SentenceTransformerEmbedder("some/model")


def test_the_refusal_names_the_model() -> None:
    with pytest.raises(ValueError, match="some/model reports no max_seq_length"):
        SentenceTransformerEmbedder("some/model")


def test_naming_the_window_is_the_way_out() -> None:
    embedder = SentenceTransformerEmbedder("some/model", max_seq_length=256)

    assert embedder.max_seq_length == 256
