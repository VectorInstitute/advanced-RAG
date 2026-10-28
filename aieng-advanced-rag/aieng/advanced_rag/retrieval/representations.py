"""Learned sparse and late-interaction retrieval behind the retriever interface.

``SpladeRetriever`` ranks by the dot product of learned sparse vectors, which
expand a chunk with related terms its text never contains. ``ColbertRetriever``
ranks by MaxSim over per-token vectors: for each query token, the best dot
product against any chunk token, summed.

Both rank within the query's own document, like the other arms, so a scoped
corpus is a few dozen chunks and MaxSim runs brute force with no vector store.

Both take an encoder rather than a model name, so a test passes a fake and
needs no download. The default encoders wrap fastembed and import it on first
use, which makes the module importable without the ``representations`` extra.

MUVERA has no retriever here. It speeds up MaxSim over a large collection and
lost on retrieval quality at that scale: failure went from 0.0280 to 0.8800
because its first stage drops candidates that rescoring cannot recover.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .corpus import Chunk, Query
from .retrievers import ChunkStoreRetriever, Retrieved
from .sparse import sparse_dot


_FloatArray = NDArray[np.floating[Any]]


#: The only learned-sparse model in fastembed's list that is SPLADE.
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"

#: 96 dimensions per token, about 0.13 GB.
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"

# Batch sizes the published figures were measured with. They change memory use
# and not the vectors.
_SPLADE_BATCH = 16
_COLBERT_BATCH = 8


class SparseEncoding(Protocol):
    """Encodes text into ``{token_id: weight}`` vectors."""

    def encode_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Return one sparse vector per document text."""
        ...

    def encode_query(self, text: str) -> dict[int, float]:
        """Return the sparse vector for one query."""
        ...


class MultiVectorEncoding(Protocol):
    """Encodes text into one ``(tokens, width)`` matrix per text."""

    def encode_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        """Return one token matrix per document text."""
        ...

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Return the token matrix for one query."""
        ...


class FastembedSparseEncoder:
    """SPLADE encoding through fastembed.

    Parameters
    ----------
    model : str, optional
        A fastembed sparse model name.
    """

    def __init__(self, model: str = SPLADE_MODEL) -> None:
        from fastembed import SparseTextEmbedding  # noqa: PLC0415

        self._model = SparseTextEmbedding(
            model, cache_dir=_hub_cache(), local_files_only=_offline()
        )

    def encode_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Return one sparse vector per document text."""
        return [
            _as_dict(vector)
            for vector in self._model.embed(list(texts), batch_size=_SPLADE_BATCH)
        ]

    def encode_query(self, text: str) -> dict[int, float]:
        """Return the sparse vector for one query."""
        return _as_dict(next(iter(self._model.query_embed([text]))))


class FastembedColbertEncoder:
    """ColBERT token vectors through fastembed.

    Parameters
    ----------
    model : str, optional
        A fastembed late-interaction model name.
    """

    def __init__(self, model: str = COLBERT_MODEL) -> None:
        from fastembed import LateInteractionTextEmbedding  # noqa: PLC0415

        self._model = LateInteractionTextEmbedding(
            model, cache_dir=_hub_cache(), local_files_only=_offline()
        )

    def encode_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        """Return one token matrix per document text."""
        return [
            np.asarray(vector, dtype=np.float32)
            for vector in self._model.embed(list(texts), batch_size=_COLBERT_BATCH)
        ]

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Return the token matrix for one query.

        Queries go through ``query_embed``, which applies ColBERT's ``[MASK]``
        augmentation. Encoding them with the document path produces a ranking
        that answers every query and ranks badly.
        """
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


def _hub_cache() -> str:
    """Return the Hugging Face hub cache, where the other local models live.

    fastembed defaults to a temporary directory, which would download both models
    again after every reboot and hide them from the guards' cache check.
    """
    from huggingface_hub import constants  # noqa: PLC0415

    return str(constants.HF_HUB_CACHE)


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1"


def _as_dict(vector: Any) -> dict[int, float]:
    return {
        int(index): float(value)
        for index, value in zip(vector.indices, vector.values, strict=True)
    }


def max_sim(query_tokens: _FloatArray, document_tokens: _FloatArray) -> float:
    """Score one chunk against one query by late interaction.

    Parameters
    ----------
    query_tokens : numpy.ndarray
        Query token vectors, shape ``(tokens, width)``.
    document_tokens : numpy.ndarray
        Chunk token vectors, shape ``(tokens, width)``.

    Returns
    -------
    float
        For each query token, the highest dot product against any chunk token,
        summed. The score grows with query length, which does not matter when a
        ranking compares chunks under one query.

    Raises
    ------
    ValueError
        If either argument is not two-dimensional or the widths differ.
    """
    query = np.asarray(query_tokens, dtype=np.float32)
    document = np.asarray(document_tokens, dtype=np.float32)
    if query.ndim != 2 or document.ndim != 2:
        raise ValueError("both arguments must be 2-D (tokens, width)")
    if query.shape[1] != document.shape[1]:
        raise ValueError(
            f"token width mismatch: query {query.shape[1]}, "
            f"document {document.shape[1]}"
        )
    if query.shape[0] == 0 or document.shape[0] == 0:
        return 0.0
    return float((query @ document.T).max(axis=1).sum())


class SpladeRetriever(ChunkStoreRetriever):
    """Per-document retrieval over learned sparse vectors.

    Parameters
    ----------
    encoder : SparseEncoding or None, optional
        Defaults to :class:`FastembedSparseEncoder`, which needs the
        ``representations`` extra and downloads about 0.5 GB on first use.
    """

    name = "splade"

    def __init__(self, encoder: SparseEncoding | None = None) -> None:
        super().__init__()
        self._encoder = encoder
        self._vectors: dict[str, dict[int, float]] = {}
        self._by_doc: dict[str, list[str]] = {}

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Encode every chunk."""
        super().index(chunks, batch_size)
        if self._encoder is None:
            self._encoder = FastembedSparseEncoder()
        vectors = self._encoder.encode_documents([c.indexed_text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._vectors[chunk.chunk_id] = vector
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` chunks of the query's own document.

        Raises
        ------
        RuntimeError
            If called before ``index``.
        """
        if self._encoder is None or not self._by_doc:
            raise RuntimeError("index() must be called before search()")
        candidates = self._by_doc.get(query.doc_id, [])
        if not candidates:
            return []
        vector = self._encoder.encode_query(query.question)
        scored = [
            Retrieved(cid, sparse_dot(vector, self._vectors[cid])) for cid in candidates
        ]
        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored[:k]


class ColbertRetriever(ChunkStoreRetriever):
    """Per-document retrieval by ColBERT late interaction.

    Parameters
    ----------
    encoder : MultiVectorEncoding or None, optional
        Defaults to :class:`FastembedColbertEncoder`, which needs the
        ``representations`` extra and downloads about 0.13 GB on first use.
    """

    name = "colbert"

    def __init__(self, encoder: MultiVectorEncoding | None = None) -> None:
        super().__init__()
        self._encoder = encoder
        self._tokens: dict[str, NDArray[np.float32]] = {}
        self._by_doc: dict[str, list[str]] = {}

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Encode every chunk into its token matrix."""
        super().index(chunks, batch_size)
        if self._encoder is None:
            self._encoder = FastembedColbertEncoder()
        matrices = self._encoder.encode_documents([c.indexed_text for c in chunks])
        for chunk, matrix in zip(chunks, matrices, strict=True):
            self._tokens[chunk.chunk_id] = matrix
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` chunks of the query's own document.

        Raises
        ------
        RuntimeError
            If called before ``index``.
        """
        if self._encoder is None or not self._by_doc:
            raise RuntimeError("index() must be called before search()")
        candidates = self._by_doc.get(query.doc_id, [])
        if not candidates:
            return []
        tokens = self._encoder.encode_query(query.question)
        scored = [
            Retrieved(cid, max_sim(tokens, self._tokens[cid])) for cid in candidates
        ]
        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored[:k]


__all__ = [
    "COLBERT_MODEL",
    "SPLADE_MODEL",
    "ColbertRetriever",
    "FastembedColbertEncoder",
    "FastembedSparseEncoder",
    "MultiVectorEncoding",
    "SparseEncoding",
    "SpladeRetriever",
    "max_sim",
]
