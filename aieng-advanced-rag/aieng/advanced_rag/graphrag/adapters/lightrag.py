"""LightRAG behind the shared retriever interface.

The important detail is that this uses ``aquery_data``, which returns structured
entities, relationships, and chunks, and stops before generation. The older path
called ``aquery(..., only_need_context=True)`` and then regex-scraped the
markdown it produced to recover chunk text. That parser broke on formatting
changes and discarded the chunk identifiers needed for citation.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ...provenance import Provenance
from ...retrieval.interface import Hints, RetrieverBase, ScopeUnsupported
from ..evidence import Evidence
from ..protocol import MissingExtraError


# LightRAG chunk ids look like "doc-<hash>-chunk-0002". When a document was
# inserted without a file path, the document hash in the chunk id is the only
# document-level identifier available, so it is recovered rather than reporting
# every passage as coming from "unknown_source".
_CHUNK_ID = re.compile(r"^(doc-[0-9a-f]+)-chunk-\d+$")


class LightRAGRetriever(RetrieverBase):
    """Adapter over an initialized :class:`lightrag.LightRAG` instance.

    Parameters
    ----------
    rag : Any
        A ``LightRAG`` instance whose storages are already initialized.
    mode : str, optional
        LightRAG query mode: ``local``, ``global``, ``hybrid``, ``mix``, or
        ``naive``. ``mix`` combines graph and vector retrieval.
    name : str, optional
        Label used in reports, defaulting to ``lightrag:<mode>``.

    Notes
    -----
    The identifiers this adapter reports are the corpus's own whenever the
    documents were inserted with ``file_paths=[<passage_id>]``: LightRAG hands
    that value back as each chunk's ``file_path`` and it becomes ``doc_id``,
    with LightRAG's internal chunk id kept in ``provenance.extra`` under
    ``lightrag_chunk_id``. An index built without ``file_paths=`` carries no
    corpus identifier, and then the internal ids are reported instead.
    """

    def __init__(self, rag: Any, mode: str = "mix", name: str | None = None) -> None:
        self._rag = rag
        self.mode = mode
        self.name = name or f"lightrag:{mode}"

    @classmethod
    async def build(
        cls,
        working_dir: str,
        llm_model_func: Any,
        embedding_func: Any,
        mode: str = "mix",
        **kwargs: Any,
    ) -> "LightRAGRetriever":
        """Construct a LightRAG instance, initialize its storages, and wrap it.

        Await this. A synchronous builder cannot hand back a working instance:
        LightRAG binds the shared ``asyncio.Lock`` objects its storages use to
        whichever loop ``initialize_storages`` ran on, so initializing on a
        private loop would return an object whose locks belong to a loop that is
        already closed. Awaiting binds them to the caller's loop, which is the
        loop :meth:`aretrieve` then runs on. Without the initialization the first
        query raises ``TypeError: 'NoneType' object does not support the
        asynchronous context manager protocol``.

        ``embedding_func`` must wrap a local encoder. The Vector proxy hosts no
        embedding model, so routing embeddings through it bills an external
        provider.

        Parameters
        ----------
        working_dir : str
            Directory LightRAG keeps its storages in.
        llm_model_func : Any
            The completion function LightRAG calls, for extraction at index time
            and for keyword extraction at query time.
        embedding_func : Any
            LightRAG's ``EmbeddingFunc`` over a local encoder. Required: the
            default vector storage refuses to construct without it.
        mode : str, optional
            LightRAG query mode the wrapped adapter uses.
        **kwargs : Any
            Passed to ``LightRAG`` unchanged.

        Returns
        -------
        LightRAGRetriever
            An adapter whose first query works.

        Raises
        ------
        MissingExtraError
            If ``lightrag-hku`` is not installed.
        """
        try:
            from lightrag import LightRAG
        except ImportError as error:  # pragma: no cover - import guard
            raise MissingExtraError("lightrag", "lightrag-hku") from error

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            **kwargs,
        )
        # 1.5.6 leaves ``auto_manage_storages_states`` off, so nothing else
        # initializes the storages and the status stays at CREATED.
        await rag.initialize_storages()
        return cls(rag, mode=mode)

    async def aretrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Retrieve asynchronously, which is LightRAG's native mode.

        This is the real method of this adapter, and the synchronous
        :meth:`retrieve` is the wrapper. ``hints.keywords`` is honoured by
        passing it as ``hl_keywords``, which LightRAG documents as the
        high-level keywords to prioritise; every other field is ignored.

        Parameters
        ----------
        question : str
            The question, passed to LightRAG unchanged.
        k : int, optional
            Passed as both ``top_k`` and ``chunk_top_k``, and used to bound the
            passages returned. LightRAG counts ``top_k`` in entities or
            relationships and ``chunk_top_k`` in text chunks, so both are set:
            otherwise the chunk budget stays at LightRAG's own default of 20 and
            ``k`` does not mean passages as it does on a dense arm.
        scope : str or None, optional
            Only ``None`` is accepted.
        hints : Hints or None, optional
            ``hints.keywords`` becomes ``hl_keywords``.

        Returns
        -------
        list of Evidence
            Up to ``k`` passages, with no per-passage score. Fewer than ``k``
            when the graph reached fewer chunks than that.

        Raises
        ------
        MissingExtraError
            If ``lightrag-hku`` is not installed.
        ScopeUnsupported
            If ``scope`` is not ``None``. LightRAG takes no filter, and the null
            arm scopes by building one store and one workspace per document
            instead. Every adapter under ``graphrag/adapters/`` refuses a scope.

        Notes
        -----
        Retrieval costs one model call. In ``local``, ``global``, ``hybrid``, and
        ``mix`` modes LightRAG extracts keywords with a completion call before it
        builds any context, so a sweep pays one call per question here on top of
        whatever answering costs.
        """
        if scope is not None:
            raise ScopeUnsupported(
                f"{self.name} takes no filter and cannot scope to {scope!r}; "
                "build one LightRAG workspace per document instead",
            )
        try:
            from lightrag import QueryParam
        except ImportError as error:  # pragma: no cover - import guard
            raise MissingExtraError("lightrag", "lightrag-hku") from error

        param = QueryParam(mode=self.mode, top_k=k, chunk_top_k=k)
        if hints and hints.keywords:
            param.hl_keywords = list(hints.keywords)
        payload = await self._rag.aquery_data(question, param=param)
        return self._to_evidence(payload, k)

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Retrieve synchronously, by driving the asynchronous method.

        Parameters
        ----------
        question : str
            The question, passed to LightRAG unchanged.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            Only ``None`` is accepted.
        hints : Hints or None, optional
            ``hints.keywords`` becomes ``hl_keywords``.

        Returns
        -------
        list of Evidence
            Exactly what :meth:`aretrieve` returns.

        Raises
        ------
        RuntimeError
            If called from inside a running event loop, where the asynchronous
            :meth:`aretrieve` must be awaited instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(question, k, scope=scope, hints=hints))
        raise RuntimeError(
            "retrieve() cannot run inside an active event loop; await aretrieve() instead"
        )

    def _to_evidence(self, payload: dict[str, Any], k: int) -> list[Evidence]:
        """Map ``aquery_data`` output onto the shared boundary type."""
        if payload.get("status") != "success":
            return []

        data = payload.get("data") or {}
        # Entities and relationships explain *why* a chunk was reached. They are
        # kept as passthrough rather than flattened into text, so a practitioner
        # can show a reasoning path without this adapter deciding the format.
        graph_context = {
            "entities": data.get("entities") or [],
            "relationships": data.get("relationships") or [],
            "mode": self.mode,
        }

        evidence: list[Evidence] = []
        for chunk in (data.get("chunks") or [])[:k]:
            content = chunk.get("content")
            if not content:
                continue
            doc_id, chunk_id = self._identifiers(chunk)
            evidence.append(
                Evidence(
                    text=content,
                    provenance=Provenance(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        extra={
                            "reference_id": chunk.get("reference_id"),
                            "lightrag_chunk_id": chunk.get("chunk_id"),
                        },
                    ),
                    score=None,  # LightRAG does not expose per-chunk scores here
                    retriever=self.name,
                    extra=graph_context if not evidence else {},
                )
            )
        return evidence

    @classmethod
    def _identifiers(cls, chunk: dict[str, Any]) -> tuple[str, str | None]:
        """Resolve the identifiers to report for one LightRAG chunk.

        ``to_runs`` scores ``chunk_id or doc_id``, so reporting LightRAG's
        internal chunk id whenever one exists shadowed the corpus identifier
        arriving as ``file_path``: every recall then read 0.0 for an arm that had
        retrieved the gold passages. The internal id is reported as ``chunk_id``
        only when it is the only identifier there is, which is when the documents
        went in without ``file_paths=``.

        Returns
        -------
        tuple of (str, str or None)
            The document identifier, and the chunk identifier when one is worth
            reporting.
        """
        corpus_id = cls._corpus_id(chunk)
        if corpus_id is not None:
            return corpus_id, None
        return cls._document_id(chunk), str(chunk.get("chunk_id") or "") or None

    @staticmethod
    def _corpus_id(chunk: dict[str, Any]) -> str | None:
        """Return the corpus identifier a chunk carries, when it carries one.

        LightRAG reports the value a document was inserted with as ``file_path``,
        and writes ``"unknown_source"`` when ``insert`` was called without
        ``file_paths=``.
        """
        file_path = chunk.get("file_path")
        if file_path and file_path != "unknown_source":
            return str(file_path)
        return None

    @classmethod
    def _document_id(cls, chunk: dict[str, Any]) -> str:
        """Resolve a document identifier for a LightRAG chunk.

        Prefers the file path, which is set when documents are inserted with
        ``file_paths=``. Falls back to the document hash embedded in the chunk
        id, so citations still group by document when they are not.
        """
        corpus_id = cls._corpus_id(chunk)
        if corpus_id is not None:
            return corpus_id
        match = _CHUNK_ID.match(str(chunk.get("chunk_id") or ""))
        if match:
            return match.group(1)
        return str(chunk.get("reference_id") or "unknown")
