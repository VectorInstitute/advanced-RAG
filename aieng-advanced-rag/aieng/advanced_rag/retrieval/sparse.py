"""BM25 sparse retrieval and weighted rank fusion.

The BM25 constants, IDF smoothing, and 31-bit token hashing are the ones the
vector-store spike measured, so a difference between a lexical result here and
one there cannot be an implementation difference. Tokenisation is shared with
``embedders``.

Why a lexical arm exists at all: dense-only was concluded on BEIR FiQA, a corpus
of natural-language finance questions where lexical matching has little to add.
CUAD is the opposite regime. Its queries are templated clause categories and the
answers are often party names, title lines, and dates, which an exact string
matches more naturally than an embedding does. On the full CUAD test split BM25
fails on 0.2548 of answerable questions against 0.3063 for bge-small.

The fusion is weighted. Fusing a strong ranker with a weaker one at equal weight
pulls the strong one down, which is arithmetic rather than a finding about
hybrid retrieval, so the weight is a parameter to sweep.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..graphrag.evidence import Evidence
from .corpus import Chunk, Query
from .embedders import tokenize
from .interface import Hints, RetrieverBase, declares
from .retrievers import ChunkStoreRetriever, Ranker, Retrieved


def sparse_index(token: str) -> int:
    """Map a token to a stable unsigned 31-bit sparse-vector index.

    Parameters
    ----------
    token : str
        A token from ``embedders.tokenize``.

    Returns
    -------
    int
        The dimension this token occupies.
    """
    digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


class SparseEncoder:
    """Corpus-fitted BM25 encoder with stable hashed dimensions.

    Parameters
    ----------
    document_frequency : dict
        Sparse index to the number of documents holding that token.
    document_count : int
        Documents the encoder was fitted on.
    average_length : float
        Mean document length in tokens, for length normalisation.
    k1 : float, optional
        BM25 term-saturation constant.
    b : float, optional
        BM25 length-normalisation constant.
    """

    def __init__(
        self,
        document_frequency: dict[int, int],
        document_count: int,
        average_length: float,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.document_frequency = document_frequency
        self.document_count = document_count
        self.average_length = average_length
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, texts: Sequence[str]) -> SparseEncoder:
        """Fit document frequencies and mean length over a corpus.

        Parameters
        ----------
        texts : sequence of str
            Every document in the corpus.

        Returns
        -------
        SparseEncoder
            A fitted encoder.

        Raises
        ------
        ValueError
            If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("cannot fit a sparse encoder on an empty corpus")
        token_lists = [tokenize(text) for text in texts]
        document_frequency: Counter[int] = Counter()
        for tokens in token_lists:
            document_frequency.update({sparse_index(token) for token in tokens})
        average_length = sum(len(tokens) for tokens in token_lists) / len(token_lists)
        return cls(dict(document_frequency), len(texts), average_length or 1.0)

    def encode_document(self, text: str) -> dict[int, float]:
        """Return the BM25 document vector, saturated and length-normalised."""
        tokens = tokenize(text)
        counts = Counter(sparse_index(token) for token in tokens)
        length_norm = 1 - self.b + self.b * len(tokens) / self.average_length
        vector: dict[int, float] = {}
        for index, frequency in counts.items():
            denominator = frequency + self.k1 * length_norm
            vector[index] = self._idf(index) * frequency * (self.k1 + 1) / denominator
        return vector

    def encode_query(self, text: str) -> dict[int, float]:
        """Return the IDF-weighted query vector."""
        counts = Counter(sparse_index(token) for token in tokenize(text))
        return {
            index: self._idf(index) * frequency for index, frequency in counts.items()
        }

    def _idf(self, index: int) -> float:
        frequency = self.document_frequency.get(index, 0)
        return math.log(1 + (self.document_count - frequency + 0.5) / (frequency + 0.5))


def sparse_dot(left: dict[int, float], right: dict[int, float]) -> float:
    """Return the dot product of two sparse vectors.

    Parameters
    ----------
    left : dict
        Sparse index to weight.
    right : dict
        Sparse index to weight.

    Returns
    -------
    float
        The dot product. Iteration runs over the smaller vector.
    """
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class BM25Retriever(ChunkStoreRetriever):
    """Per-document BM25 retrieval.

    Document frequencies are fitted over the whole slice rather than per
    document, because a term's rarity is a property of the corpus, and fitting
    on the few dozen chunks of one contract would make every term look common.
    Retrieval is still scoped to the query's document, matching the dense arm.

    Parameters
    ----------
    k1 : float, optional
        BM25 term-saturation constant.
    b : float, optional
        BM25 length-normalisation constant.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b
        self._encoder: SparseEncoder | None = None
        self._vectors: dict[str, dict[int, float]] = {}
        self._by_doc: dict[str, list[str]] = {}

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Fit the encoder over every chunk, then encode each one."""
        super().index(chunks, batch_size)
        encoder = SparseEncoder.fit([c.indexed_text for c in chunks])
        encoder.k1, encoder.b = self.k1, self.b
        self._encoder = encoder
        for chunk in chunks:
            self._vectors[chunk.chunk_id] = encoder.encode_document(chunk.indexed_text)
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` chunks of the query's own document.

        Raises
        ------
        RuntimeError
            If called before ``index``.
        """
        if self._encoder is None:
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


def weighted_rrf(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float],
    limit: int,
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked chunk IDs, weighting each ranking's contribution.

    Equal weights reduce to standard reciprocal rank fusion. The weight is what
    a sweep varies, because equal weighting is only correct when the arms are
    equally good.

    Parameters
    ----------
    rankings : sequence of sequence of str
        One ranked list of chunk identifiers per arm.
    weights : sequence of float
        One weight per ranking. A zero-weight arm contributes nothing.
    limit : int
        How many fused results to return.
    rank_constant : int, optional
        The RRF constant. 60 is the value the published recipe uses.

    Returns
    -------
    list of tuple
        ``(chunk_id, score)`` in descending score, ties broken on identifier.

    Raises
    ------
    ValueError
        If the number of weights does not match the number of rankings.
    """
    if len(rankings) != len(weights):
        raise ValueError("one weight per ranking is required")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        if weight == 0.0:
            continue
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += weight / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class HybridRetriever(RetrieverBase):
    """Weighted rank fusion over an already-indexed dense arm and lexical arm.

    Each arm is searched to ``fusion_depth`` before fusion, because fusing two
    top-10 lists throws away the ranks that let a document recovered by only one
    arm climb. The arms are passed in already indexed, so a weight sweep does not
    re-embed the corpus for every weight.

    Parameters
    ----------
    dense : Ranker
        The indexed dense arm.
    lexical : Ranker
        The indexed lexical arm.
    dense_weight : float, optional
        The dense arm's share. The lexical arm gets ``1 - dense_weight``. The
        default, 0.20, is the lowest retrieval failure measured on the full
        CUAD test split (1,244 answerable questions, scoped per contract) for
        ``bge-small-en-v1.5`` with or without its query prompt: 0.2436 with the
        prompt, against 0.2629 at equal weight and 0.2548 for BM25 alone. On
        the 20-contract CUAD and ContractNLI slices weights from 0.4 to 0.5 do
        slightly better, so sweep it on your own corpus.
        ``implementations/regulatory_policy_qa/02_retrieval_levers.ipynb``
        runs the sweep.
    fusion_depth : int, optional
        Depth each arm is searched to before fusion.
    rank_constant : int, optional
        Passed to ``weighted_rrf``.
    """

    def __init__(
        self,
        dense: Ranker,
        lexical: Ranker,
        dense_weight: float = 0.2,
        fusion_depth: int = 50,
        rank_constant: int = 60,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.dense_weight = dense_weight
        self.fusion_depth = fusion_depth
        self.rank_constant = rank_constant
        self.name = f"hybrid rrf (dense {dense_weight:.2f})"

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Do nothing: both arms are indexed by the caller and shared."""

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` fused passages, in fused rank order.

        This arm holds no chunks of its own, because its ``index`` is a no-op so
        that a weight sweep does not re-embed the corpus. Text therefore comes
        from whichever arm retains its own, and the fused score and this arm's
        own name are put back on each passage.

        ``hints`` are ignored, as on both arms.

        Parameters
        ----------
        question : str
            The question, passed to both arms unchanged.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            One document identifier, or ``None`` for every indexed document.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Evidence
            Fused passages carrying text, provenance, and the fused score.
        """
        source = self._text_source()
        targets = [scope] if scope is not None else source.documents()
        hits: list[Retrieved] = []
        for doc_id in targets:
            hits.extend(
                self.search(
                    Query(query_id="", doc_id=doc_id, category="", question=question),
                    k=k,
                ),
            )
        if len(targets) > 1:
            hits.sort(key=lambda hit: -hit.score)
        hits = hits[:k]
        held = {
            item.chunk_id: item for item in source.fetch([h.chunk_id for h in hits])
        }
        return [
            replace(held[hit.chunk_id], score=hit.score, retriever=self.name)
            for hit in hits
            if hit.chunk_id in held
        ]

    def _text_source(self) -> Any:
        """Return the arm that retains its own chunks.

        Raises
        ------
        TypeError
            If neither arm retains its chunks, which leaves this arm holding
            identifiers with no text to attach to them.
        """
        for arm in (self.dense, self.lexical):
            if declares(arm, "fetch") and hasattr(arm, "documents"):
                return arm
        raise TypeError(
            "a hybrid arm returns evidence through its arms, so one of them has "
            "to retain its chunks: it needs fetch and documents, which "
            "ChunkStoreRetriever supplies",
        )

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` chunks after fusing both arms."""
        dense_hits = [h.chunk_id for h in self.dense.search(query, k=self.fusion_depth)]
        lexical_hits = [
            h.chunk_id for h in self.lexical.search(query, k=self.fusion_depth)
        ]
        fused = weighted_rrf(
            [dense_hits, lexical_hits],
            [self.dense_weight, 1.0 - self.dense_weight],
            limit=k,
            rank_constant=self.rank_constant,
        )
        return [Retrieved(chunk_id, score) for chunk_id, score in fused]
