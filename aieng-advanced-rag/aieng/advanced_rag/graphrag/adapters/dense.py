"""A plain dense retriever, as the control every comparison needs.

No graph. It is here so a GraphRAG result is never reported without the number
it has to beat: on the ServerFault slice a plain chunk index already answered
99.2% of questions, which is the kind of thing only a control reveals.

The vectors live in an in-process Qdrant collection, the same store as the
harness dense arm, so it needs the ``qdrant`` extra and a local encoder and none
of the graph frameworks.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Sequence

from ...provenance import Provenance
from ...retrieval.interface import Hints, RetrieverBase, ScopeUnsupported
from ..evidence import Evidence


class DenseRetriever(RetrieverBase):
    """Cosine-similarity retrieval over normalized vectors in a Qdrant store.

    Parameters
    ----------
    encode : callable
        Maps a sequence of strings to a list of vectors. Must be a local model.
    name : str, optional
        Label used in reports.
    store : QdrantStore, QdrantClient, str, or None, optional
        Where the vectors go: a store, a client, or a location such as
        ``":memory:"`` (the default) or a server URL.
    """

    def __init__(
        self,
        encode: Callable[[Sequence[str]], list[list[float]]],
        name: str = "dense",
        store: Any = ":memory:",
    ) -> None:
        from ...retrieval.stores import QdrantStore

        self._encode = encode
        self.name = name
        self.vectors = (
            store
            if isinstance(store, QdrantStore)
            else QdrantStore(store, collection=f"graph-dense-{uuid.uuid4().hex[:12]}")
        )
        self._texts: dict[str, str] = {}
        self._provenance: dict[str, Provenance] = {}

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else [0.0] * len(vector)

    def index(self, chunks: Sequence[Any], batch_size: int = 64) -> None:
        """Index chunk objects carrying ``chunk_id``, ``text``, and a document id.

        Parameters
        ----------
        chunks : sequence
            Objects accepted by :meth:`Provenance.from_chunk`.
        batch_size : int, optional
            Encoder batch size.
        """
        keys: list[str] = []
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            for chunk, vector in zip(batch, self._encode([c.text for c in batch])):
                provenance = Provenance.from_chunk(chunk)
                key = provenance.chunk_id or provenance.doc_id
                self._texts[key] = chunk.text
                self._provenance[key] = provenance
                keys.append(key)
                vectors.append(self._normalize(vector))
        self.vectors.add(keys, vectors, [self._provenance[key].doc_id for key in keys])

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return the ``k`` nearest chunks by cosine similarity.

        ``hints`` are ignored: this control has no linking step to seed and no
        keyword field to prioritise, so honouring one would mean changing the
        query text and with it the ranking.

        Parameters
        ----------
        question : str
            The question, encoded as-is.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            Only ``None`` is accepted.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Evidence
            Ranked passages carrying text and provenance.

        Raises
        ------
        ScopeUnsupported
            If ``scope`` is not ``None``. This control ranks over every indexed
            chunk, and filtering a finished ranking would return fewer than
            ``k``, so the scope is refused rather than dropped.
        """
        if scope is not None:
            raise ScopeUnsupported(
                f"{self.name} ranks the whole index and cannot filter to {scope!r}",
            )
        if not len(self.vectors):
            return []
        query_vector = self._normalize(self._encode([question])[0])
        return [
            Evidence(
                text=self._texts[key],
                provenance=self._provenance[key],
                score=float(score),
                retriever=self.name,
            )
            for key, score in self.vectors.search(query_vector, k)
        ]
