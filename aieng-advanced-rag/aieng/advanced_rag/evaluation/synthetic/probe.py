"""The generator's own small index, for the checks that need to search.

Three steps search the indexed documents: the uniqueness filter looks for a
second passage that answers the question, the null check looks for any passage
that answers a null question, and the recorded labels note where BM25 and the
dense encoder rank the gold passage. None of them decides a question's fate on
whether a retriever found its gold passage. The uniqueness filter and the null
check ask whether other text answers the question, which is a property of the
corpus, and the ranks are recorded only.

The probe chunks with fixed-width character windows and ranks with the
package's BM25 encoder and a local embedder. It is unrelated to the arms being
compared, which chunk and rank however they like.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.sparse import SparseEncoder, sparse_dot

from .sampling import Document, normalize


class TextEncoder(Protocol):
    """A local encoder: texts in, one vector per text out."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""
        ...


@dataclass(frozen=True)
class Hit:
    """One ranked probe chunk."""

    chunk: Chunk
    score: float
    rank: int


class ProbeIndex:
    """BM25 and dense rankings over fixed-width chunks of the indexed documents.

    Parameters
    ----------
    documents : sequence of Document
        The documents the generated set targets. Held-out documents stay out.
    embedder : TextEncoder
        Anything with ``encode(texts)`` and optionally ``encode_queries(texts)``,
        such as
        :class:`~aieng.advanced_rag.retrieval.embedders.SentenceTransformerEmbedder`.
        Local only; embeddings never go through the proxy.
    chunk_chars, overlap_chars : int, optional
        The probe's chunking.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedder: TextEncoder,
        *,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> None:
        self.embedder = embedder
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document.doc_id,
                document.text,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        ]
        self._encoder = SparseEncoder.fit([c.text for c in self.chunks])
        self._sparse = [self._encoder.encode_document(c.text) for c in self.chunks]
        self._dense = self._encode_documents([c.text for c in self.chunks])
        self._by_doc: dict[str, list[int]] = {}
        for index, chunk in enumerate(self.chunks):
            self._by_doc.setdefault(chunk.doc_id, []).append(index)
        self._query_vectors: dict[str, list[float]] = {}

    def _encode_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            vectors.extend(
                normalize(v) for v in self.embedder.encode(texts[start : start + 64])
            )
        return vectors

    def encode_query(self, text: str) -> list[float]:
        """Return the dense query vector, cached by text."""
        with self._lock:
            if text not in self._query_vectors:
                encode = getattr(self.embedder, "encode_queries", self.embedder.encode)
                self._query_vectors[text] = normalize(encode([text])[0])
            return self._query_vectors[text]

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return document-side vectors for arbitrary texts."""
        with self._lock:
            return [normalize(v) for v in self.embedder.encode(list(texts))]

    def documents(self) -> list[str]:
        """Return the indexed document identifiers."""
        return list(self._by_doc)

    def search(
        self,
        question: str,
        *,
        method: str = "dense",
        k: int = 10,
        scope: str | None = None,
        exclude: Iterable[tuple[str, int, int]] = (),
    ) -> list[Hit]:
        """Rank chunks for a question.

        Parameters
        ----------
        question : str
            The question.
        method : str, optional
            ``"dense"`` or ``"bm25"``.
        k : int, optional
            Hits to return.
        scope : str or None, optional
            One document, or ``None`` for all.
        exclude : iterable of (doc_id, start, end), optional
            Spans whose overlapping chunks are skipped, such as the gold spans
            when looking for a second answering passage.

        Returns
        -------
        list of Hit
            In rank order, ties broken on chunk identifier.

        Raises
        ------
        ValueError
            On an unknown ``method``.
        """
        indices = (
            self._by_doc.get(scope, [])
            if scope is not None
            else range(len(self.chunks))
        )
        blocked = list(exclude)
        if method == "bm25":
            query = self._encoder.encode_query(question)
            scores = {i: sparse_dot(query, self._sparse[i]) for i in indices}
        elif method == "dense":
            vector = self.encode_query(question)
            scores = {
                i: sum(a * b for a, b in zip(vector, self._dense[i])) for i in indices
            }
        else:
            raise ValueError(f"method must be 'dense' or 'bm25', got {method!r}")
        ranked = sorted(scores, key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        hits: list[Hit] = []
        for index in ranked:
            chunk = self.chunks[index]
            if any(
                chunk.doc_id == doc and chunk.start < end and start < chunk.end
                for doc, start, end in blocked
            ):
                continue
            hits.append(Hit(chunk, scores[index], len(hits) + 1))
            if len(hits) >= k:
                break
        return hits

    def gold_rank(
        self,
        question: str,
        spans: Iterable[tuple[str, int, int]],
        *,
        method: str,
        scope: str | None = None,
        depth: int = 100,
    ) -> int | None:
        """Return the best rank of any chunk overlapping a gold span.

        Returns
        -------
        int or None
            One-based rank, or ``None`` when no gold chunk is in the top
            ``depth``.
        """
        gold = list(spans)
        for hit in self.search(question, method=method, k=depth, scope=scope):
            chunk = hit.chunk
            if any(
                chunk.doc_id == doc and chunk.start < end and start < chunk.end
                for doc, start, end in gold
            ):
                return hit.rank
        return None


__all__ = ["Hit", "ProbeIndex", "TextEncoder"]
