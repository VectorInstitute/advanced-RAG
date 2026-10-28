"""The baseline retrievers: a dense arm and the negative control.

The dense retriever is deliberately the plain one: normalised vectors in a
Qdrant collection, dot product, top-k, scoped to a single document. It is the
control every other method is measured against, so it is boring and legible on
purpose. Qdrant runs in process by default, which is an exact scan and needs no
server; ``retrieval.stores`` says when a server is worth starting.

Retrieval is scoped per document because CUAD's questions are templated and
carry no signal about which contract they belong to. That scoping is a payload
filter on ``doc_id``, the path a vector store takes in production. For a
batched sweep, ``index.DenseIndex`` ranks the same way over the same store.

``index`` and ``search`` stay the internal ranking core of every arm here.
``retrieve`` on :class:`ChunkStoreRetriever` is the surface a caller programs
against, because it returns text-carrying
:class:`~aieng.advanced_rag.graphrag.evidence.Evidence` rather than identifiers
a caller has to join back to the corpus itself.
"""

from __future__ import annotations

import json
import random
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..graphrag.evidence import Evidence
from .corpus import Chunk, Query
from .embedders import Embedder
from .interface import Candidate, ChunkStore, Hints, RetrieverBase


@dataclass
class Retrieved:
    """One ranked hit.

    Parameters
    ----------
    chunk_id : str
        Identifier of the chunk.
    score : float
        Retriever score. Comparable within one ranking, not across retrievers.
    """

    chunk_id: str
    score: float


class Ranker(Protocol):
    """The internal ranking core: index chunks, then rank identifiers.

    Not the surface a caller programs against. ``retrieve`` on
    :class:`~aieng.advanced_rag.retrieval.interface.Retriever` is that, and it
    returns text-carrying evidence rather than identifiers a caller joins back to
    the corpus. This types the two methods each arm implements underneath, and
    :func:`run_retriever` is what consumes it.
    """

    name: str

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Build whatever this ranker searches over."""
        ...

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` hits for one query."""
        ...


class ChunkStoreRetriever(RetrieverBase):
    """A harness retriever that keeps the chunks it was indexed with.

    Each arm is handed whole chunks at ``index`` time and used to keep only their
    vectors, which is why turning a hit into evidence needed a chunk mapping
    passed back in at call time. Retaining the chunks in a :class:`ChunkStore`
    instead puts text on the retrieval boundary, at 1.14 MB per store on the
    committed 20-contract CUAD slice.

    Subclasses define ``search``, which stays the ranking core. This class turns
    that ranking into evidence and delegates fetch, expand, read, and grep to
    the store, so every arm gains the same capabilities from one place.
    """

    name: str

    def __init__(self) -> None:
        if not getattr(self, "name", ""):
            raise AttributeError(
                f"Retriever [{type(self).__name__}] is missing the required "
                '"name" attribute, which every passage cites as the method that '
                "found it"
            )
        self.store = ChunkStore(retriever=self.name)

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Retain the chunks, so a hit can carry its own text and provenance.

        Parameters
        ----------
        chunks : sequence of Chunk
            The chunks to index.
        batch_size : int, optional
            Chunks per batch, for an arm that encodes in batches.
        """
        self.store.add(chunks)

    def documents(self) -> list[str]:
        """Return the documents this arm has indexed, in first-seen order.

        Returns
        -------
        list of str
            One identifier per document, which is the set ``scope`` selects
            from.
        """
        return self.store.document_ids()

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Rank the query's own document. Subclasses implement this.

        Parameters
        ----------
        query : Query
            The query, whose ``doc_id`` scopes the ranking.
        k : int, optional
            Depth to rank to.

        Returns
        -------
        list of Retrieved
            Ranked hits.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError(
            f'Retriever [{type(self).__name__}] is missing the required "search" method'
        )

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` passages, in the order ``search`` ranked them.

        ``hints`` are ignored. Neither the dense nor the lexical arm can honour a
        keyword or an entity hint without changing the query text, and changing
        the query text changes the ranking, which is the one thing a hint is not
        allowed to do.

        Parameters
        ----------
        question : str
            The question, encoded or tokenised as-is.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            One document identifier, or ``None`` for every indexed document.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Evidence
            Ranked passages carrying text and provenance.
        """
        targets = [scope] if scope is not None else self.store.document_ids()
        hits: list[Retrieved] = []
        for doc_id in targets:
            hits.extend(
                self.search(
                    Query(query_id="", doc_id=doc_id, category="", question=question),
                    k=k,
                ),
            )
        if len(targets) > 1:
            # Stable, and on the score alone: each document's own rank order has
            # to survive a tie. Sorting all-zero scores on chunk identifier
            # would rewrite the negative control into document order.
            hits.sort(key=lambda hit: -hit.score)
        return self.store.rank([(hit.chunk_id, hit.score) for hit in hits[:k]])

    def find(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        preview: int = 200,
        hints: Hints | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` candidates, so a caller can pick what to read.

        The preview is applied here and nowhere else. Truncating on the scored
        path would change the text a generator reads.

        Parameters
        ----------
        question : str
            The question, encoded or tokenised as-is.
        k : int, optional
            Maximum candidates to return.
        scope : str or None, optional
            One document identifier, or ``None`` for every indexed document.
        preview : int, optional
            Leading characters of each passage to include.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Candidate
            The same ranking as :meth:`retrieve`, listed cheaply.
        """
        return [
            Candidate(
                chunk_id=item.chunk_id or item.doc_id,
                doc_id=item.doc_id,
                score=item.score,
                preview=item.text[:preview],
                retriever=self.name,
            )
            for item in self.retrieve(question, k, scope=scope, hints=hints)
        ]

    def fetch(self, chunk_ids: Sequence[str]) -> list[Evidence]:
        """Return the named passages in full, from the store.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers, normally from :meth:`find`.

        Returns
        -------
        list of Evidence
            One record per identifier this arm indexed, in the order asked.
        """
        return self.store.fetch(chunk_ids)

    def expand(
        self, chunk_ids: Sequence[str], *, limit: int | None = None
    ) -> list[Evidence]:
        """Return each hit with the windows either side of it.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers to widen.
        limit : int or None, optional
            Maximum passages to return.

        Returns
        -------
        list of Evidence
            The hits and their neighbours, de-duplicated.
        """
        return self.store.expand(chunk_ids, limit=limit)

    def read(self, doc_id: str, *, start: int = 0, end: int | None = None) -> Evidence:
        """Return a character range of one indexed document.

        Parameters
        ----------
        doc_id : str
            The document to read.
        start : int, optional
            First character offset.
        end : int or None, optional
            One past the last character. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the span read.
        """
        return self.store.read(doc_id, start=start, end=end)

    def read_lines(
        self, doc_id: str, *, first: int = 1, last: int | None = None
    ) -> Evidence:
        """Return a line range of one indexed document, counting from 1.

        Parameters
        ----------
        doc_id : str
            The document to read.
        first : int, optional
            First line.
        last : int or None, optional
            Last line, inclusive. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the character span read.
        """
        return self.store.read_lines(doc_id, first=first, last=last)

    def grep(
        self,
        pattern: str,
        k: int = 10,
        *,
        scope: str | None = None,
        preview: int = 200,
    ) -> list[Candidate]:
        """Return up to ``k`` pattern matches over the retained text.

        Parameters
        ----------
        pattern : str
            A Python regular expression.
        k : int, optional
            Maximum matches to return.
        scope : str or None, optional
            One document identifier, or ``None`` for every indexed document.
        preview : int, optional
            Characters of matched text to include.

        Returns
        -------
        list of Candidate
            Matches in document order, which is not a ranking.
        """
        return self.store.grep(pattern, k, scope=scope, preview=preview)


class QdrantRetriever(ChunkStoreRetriever):
    """Per-document dense retrieval over a Qdrant collection.

    The dense arm every RI measures. Vectors live in a
    :class:`~aieng.advanced_rag.retrieval.stores.QdrantStore`, in process by
    default, where search is an exact scan. A document scope is a
    payload filter on ``doc_id``. Ties break on chunk identifier ascending, so
    a run reproduces exactly.

    Moving to a server changes one argument: pass ``store="http://..."``, or a
    ``QdrantStore`` built with quantization or HNSW settings.

    Parameters
    ----------
    embedder : Embedder
        Supplies both sides. Queries go through ``encode_queries`` when the
        embedder defines it, so an asymmetric model applies its query prompt.
        Its vectors must be L2-normalised, because the store scores by dot
        product.
    store : QdrantStore, QdrantClient, str, or None, optional
        Where the vectors go. A string is a location for
        :func:`~aieng.advanced_rag.retrieval.stores.connect`: ``":memory:"``
        (the default), a directory, or a server URL.
    settings : QdrantSettings or None, optional
        Quantization and HNSW settings, when ``store`` is not already a
        ``QdrantStore``. They need a server.

    Raises
    ------
    ImportError
        If the ``qdrant`` extra is not installed.
    """

    name = "dense"

    def __init__(
        self,
        embedder: Embedder,
        store: Any = ":memory:",
        *,
        settings: Any = None,
    ) -> None:
        from .stores import QdrantStore

        super().__init__()
        self.embedder = embedder
        if isinstance(store, QdrantStore):
            if settings is not None:
                raise ValueError("pass settings to the QdrantStore, not to both")
            self.vectors: QdrantStore = store
        else:
            self.vectors = QdrantStore(
                store, collection=f"dense-{uuid.uuid4().hex[:12]}", settings=settings
            )

    def _encode_query(self, question: str) -> list[float]:
        encode = getattr(self.embedder, "encode_queries", self.embedder.encode)
        return encode([question])[0]

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Embed chunks in batches, and hold their vectors and their text.

        Raises
        ------
        ValueError
            If a chunk identifier is already indexed.
        """
        super().index(chunks, batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(self.embedder.encode([c.indexed_text for c in batch]))
        # One add, because a server store waits for its optimizer after each.
        self.vectors.add(
            [c.chunk_id for c in chunks], vectors, [c.doc_id for c in chunks]
        )

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the top ``k`` chunks of the query's own document."""
        if not query.doc_id or not len(self.vectors):
            return []
        hits = self.vectors.search(
            self._encode_query(query.question), k, scope=query.doc_id
        )
        return [Retrieved(chunk_id, score) for chunk_id, score in hits]

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` passages, ranked by the store.

        ``hints`` are ignored, for the reason
        :meth:`ChunkStoreRetriever.retrieve` gives. Without a scope this is one
        search over every document rather than one per document, and ties
        across documents break on chunk identifier.

        Parameters
        ----------
        question : str
            The question, encoded as-is.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            One document identifier, or ``None`` for every indexed document.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Evidence
            Ranked passages carrying text and provenance.
        """
        if not len(self.vectors):
            return []
        hits = self.vectors.search(self._encode_query(question), k, scope=scope)
        return self.store.rank(hits)

    def describe(self) -> dict[str, Any]:
        """Return what identifies these vectors, for an export or a cache key."""
        return {
            "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
            "query_prompt": getattr(self.embedder, "query_prompt", None),
            "document_prompt": getattr(self.embedder, "document_prompt", None),
        }

    def save(self, directory: Path) -> Path:
        """Export the vectors and chunk offsets as portable files.

        Parameters
        ----------
        directory : Path
            Where to write. It must be empty or absent.

        Returns
        -------
        Path
            ``directory``.
        """
        from .stores import export_collection

        return export_collection(
            self.vectors,
            directory,
            metadata=self.describe(),
            payloads=self._payloads(),
        )

    def load(self, directory: Path, chunks: Sequence[Chunk]) -> None:
        """Rebuild an exported collection instead of embedding the chunks.

        The chunks carry the text, so the export holds only vectors and offsets.
        They must be the chunks the export was made from: the same identifiers
        and the same spans.

        Parameters
        ----------
        directory : Path
            A directory :meth:`save` wrote.
        chunks : sequence of Chunk
            The chunks the vectors belong to.

        Raises
        ------
        ValueError
            If the export was made with another embedder or from other chunks.
        """
        from .stores import load_collection

        directory = Path(directory)
        recorded = json.loads((directory / "collection.json").read_text("utf-8"))
        self._check_metadata(dict(recorded.get("metadata", {})), directory)
        with (directory / "points.jsonl").open(encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle if line.strip()]
        self._check_chunks(payloads, chunks, directory)
        load_collection(directory, self.vectors)
        ChunkStoreRetriever.index(self, chunks)

    def load_or_index(
        self,
        cache: Any,
        key: Any,
        chunks: Sequence[Chunk],
        *,
        batch_size: int = 64,
    ) -> bool:
        """Load the collection from the artifact cache, or embed and cache it.

        Parameters
        ----------
        cache : TieredResultCache
            The artifact cache.
        key : CacheKey
            Names what changes the vectors: corpus digest, chunker, embedding
            model, and prompt. Not the store mode, so one artifact loads into
            memory, a path, or a server.
        chunks : sequence of Chunk
            The chunks to index, or that the cached vectors belong to.
        batch_size : int, optional
            Encoder batch size on a miss.

        Returns
        -------
        bool
            ``True`` on a cache hit.
        """
        from ..cache.directories import fetch_directory, store_directory

        with tempfile.TemporaryDirectory() as scratch:
            found = fetch_directory(cache, key, Path(scratch) / "collection")
            if found is not None:
                self.load(found, chunks)
                return True
            self.index(chunks, batch_size)
            store_directory(cache, key, self.save(Path(scratch) / "export"))
            return False

    def _payloads(self) -> list[dict[str, Any]]:
        keys = self.vectors.keys
        spans = {
            item.provenance.chunk_id: (item.provenance.start, item.provenance.end)
            for item in self.store.fetch(keys)
        }
        return [
            dict(zip(("start", "end"), spans.get(key, (None, None)))) for key in keys
        ]

    def _check_metadata(self, metadata: dict[str, Any], directory: Path) -> None:
        mine = self.describe()
        for field in ("embedder", "query_prompt", "document_prompt"):
            if field in metadata and metadata[field] != mine[field]:
                raise ValueError(
                    f"{directory} was embedded with {field}={metadata[field]!r}, "
                    f"and this retriever has {mine[field]!r}"
                )

    @staticmethod
    def _check_chunks(
        payloads: Sequence[dict[str, Any]], chunks: Sequence[Chunk], directory: Path
    ) -> None:
        exported = {p["key"]: (p.get("start"), p.get("end")) for p in payloads}
        given = {c.chunk_id: (c.start, c.end) for c in chunks}
        if exported != given:
            missing = sorted(set(exported) ^ set(given))[:3]
            moved = [k for k in exported if k in given and exported[k] != given[k]][:3]
            raise ValueError(
                f"{directory} holds vectors for other chunks: "
                f"identifiers differing {missing}, spans differing {moved}"
            )


#: The dense arm under the name every RI, guard, and notebook imports. It is
#: :class:`QdrantRetriever`; the pure-Python list index it used to be is kept
#: only as the parity reference in the package tests.
DenseRetriever = QdrantRetriever


class ShuffledRetriever(ChunkStoreRetriever):
    """The negative control.

    Returns chunks from the right document in a fixed random order, so it has
    the same candidate pool as the real retriever and none of the ranking
    signal. Its retrieval-failure rate is what "no retrieval at all" scores on a
    slice, which is the floor a baseline has to beat for the harness to be
    measuring anything. Without it a plausible-looking failure rate could just
    be the pool being small.

    Parameters
    ----------
    seed : int, optional
        Seeds the per-document shuffle, so two instances agree.
    """

    name = "shuffled-control"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._by_doc: dict[str, list[str]] = {}
        self._seed = seed

    def index(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        """Group chunk identifiers by document and shuffle each group."""
        super().index(chunks, batch_size)
        for chunk in chunks:
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
        for doc_id, ids in self._by_doc.items():
            rng = random.Random(f"{self._seed}:{doc_id}")
            rng.shuffle(ids)

    def search(self, query: Query, k: int = 10) -> list[Retrieved]:
        """Return the first ``k`` chunks of the shuffled pool."""
        ids = self._by_doc.get(query.doc_id, [])
        return [Retrieved(cid, 0.0) for cid in ids[:k]]


def run_retriever(
    retriever: Ranker,
    queries: Sequence[Query],
    k: int = 10,
) -> dict[str, list[str]]:
    """Run a retriever over queries and return ranked chunk IDs per query.

    Parameters
    ----------
    retriever : Ranker
        An indexed retriever.
    queries : sequence of Query
        Usually ``Slice.answerable``.
    k : int, optional
        Depth to retrieve to.

    Returns
    -------
    dict
        Query identifier to ranked chunk identifiers, ready for
        ``evaluation.metrics.score_retrieval``.
    """
    return {
        q.query_id: [hit.chunk_id for hit in retriever.search(q, k=k)] for q in queries
    }
