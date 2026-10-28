"""HippoRAG 2 behind the shared retriever interface, without vLLM.

The obvious package, ``hipporag``, pins ``vllm``, ``torch==2.5.1``, and fourteen
CUDA packages, which rules it out of a CPU container. ``atlas-rag``
(AutoSchemaKG) ships its own implementation of HippoRAG 2 in
``atlas_rag.retriever.hipporag2``, with its own extraction prompts and defaults,
so results describe that variant rather than the paper's reference code. It has
around twenty dependencies, no vLLM and no CUDA pins, and it takes a local
sentence-transformers encoder plus any OpenAI-compatible client.

Three shapes are normalized here:

* ``atlas_rag``'s retriever takes ``topN`` while its own base class declares
  ``topk``.
* It returns ``(passages, ids)`` as parallel lists rather than records.
* What it calls an id is the passage node's ``id`` attribute, which the graph
  build sets to the passage text. The identifier is the node key, recovered by
  inverting the ``text_id_to_node_name`` map upstream keeps.
"""

from __future__ import annotations

from copy import copy
from typing import Any

from ...provenance import Provenance
from ...retrieval.interface import Hints, RetrieverBase, ScopeUnsupported
from ..evidence import Evidence
from ..protocol import MissingExtraError


class HippoRAG2Retriever(RetrieverBase):
    """Adapter over ``atlas_rag``'s ``HippoRAG2Retriever``.

    Parameters
    ----------
    retriever : Any
        A constructed ``atlas_rag.retriever.HippoRAG2Retriever``.
    name : str, optional
        Label used in reports.

    Notes
    -----
    The identifier this adapter reports is the graph's passage node key, which
    is the ``text_id`` column the knowledge graph was built from. Build the graph
    with the corpus's own passage identifiers in that column and the numbers the
    harness scores line up with the corpus without any further mapping.
    """

    def __init__(self, retriever: Any, name: str = "hipporag2") -> None:
        self._retriever = retriever
        self.name = name
        # Upstream reports ``text_id_to_node_name[passage_id]`` for each hit, and
        # for a passage node that value is the passage text rather than an
        # identifier. Inverting the map once recovers the node key, which is the
        # id the corpus recognises. Passages with byte-identical text are
        # indistinguishable in what upstream returns, so one key wins for both.
        self._passage_ids = {
            str(node_name): str(passage_id)
            for passage_id, node_name in (
                getattr(retriever, "text_id_to_node_name", None) or {}
            ).items()
        }

    @classmethod
    def build(  # noqa: PLR0917 - preserve the existing positional API
        cls,
        kg_dir: str,
        llm_client: Any,
        llm_model: str,
        encoder_model: str = "BAAI/bge-small-en-v1.5",
        keyword: str = "",
        include_concept: bool = False,
        include_events: bool = False,
        *,
        encoder_revision: str | None = None,
    ) -> "HippoRAG2Retriever":
        """Construct the underlying retriever from a built knowledge graph.

        Four of these arguments are the cache key for everything under
        ``<kg_dir>/precompute/``: ``keyword``, ``encoder_model``,
        ``include_concept``, and ``include_events``. Against a prebuilt index,
        repeat the values it was built with. A wrong ``keyword`` fails loudly,
        because it also names the graph file. A wrong encoder model or either
        flag misses the cache quietly and re-embeds every passage, node, and edge
        on CPU. The defaults here match no real artifact.

        Parameters
        ----------
        kg_dir : str
            Directory holding the graph AutoSchemaKG built.
        llm_client : Any
            An OpenAI-compatible client, normally pointed at the Vector proxy.
        llm_model : str
            Bare model name, e.g. ``Qwen3_6-35B-A3B``.
        encoder_model : str, optional
            Local sentence-transformers model. Embeddings stay local by policy.
            Part of the precompute cache key.
        keyword : str, optional
            Dataset keyword ``atlas_rag`` uses for index naming, and part of the
            cache key. ``atlas_rag`` special-cases ``"musique"``.
        include_concept, include_events : bool, optional
            Whether to index concept and event nodes. Both are part of the cache
            key.

        encoder_revision : str or None, optional
            Hugging Face revision for query encoding. The producer encoder revision
            remains a separate provenance field; this does not rebuild its vectors.

        Returns
        -------
        HippoRAG2Retriever
            An adapter over the constructed upstream retriever.

        Raises
        ------
        MissingExtraError
            If ``atlas-rag`` or ``sentence-transformers`` is not installed.
        FileNotFoundError
            If no graph exists at ``<kg_dir>/kg_graphml/<keyword>_graph.graphml``.
        """
        try:
            from atlas_rag.llm_generator import LLMGenerator
            from atlas_rag.retriever import HippoRAG2Retriever as _Upstream
            from atlas_rag.vectorstore import create_embeddings_and_index
            from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - import guard
            raise MissingExtraError("hipporag", "atlas-rag") from error

        encoder = SentenceEmbedding(
            SentenceTransformer(encoder_model, device="cpu", revision=encoder_revision)
        )
        data = create_embeddings_and_index(
            sentence_encoder=encoder,
            model_name=encoder_model,
            working_directory=kg_dir,
            keyword=keyword,
            include_concept=include_concept,
            include_events=include_events,
            normalize_embeddings=True,
        )
        return cls(
            _Upstream(
                llm_generator=LLMGenerator(client=llm_client, model_name=llm_model),
                sentence_encoder=encoder,
                data=data,
            )
        )

    def with_client(self, client: Any) -> "HippoRAG2Retriever":
        """Give one query its own client while sharing immutable graph and vectors.

        Atlas-rag starts internal threads for completion calls. A caller can bind
        a question's meter in its client without changing another query's client.
        Its selected retrieval function is rebound to the copied query state.
        """
        query = copy(self._retriever)
        query.llm_generator = copy(self._retriever.llm_generator)
        query.llm_generator.client = client
        query.retrieve_node_fn = getattr(
            query, self._retriever.retrieve_node_fn.__name__
        )
        return type(self)(query, name=self.name)

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Retrieve passages and attach provenance.

        ``hints`` are ignored. The upstream retriever takes a query and a depth
        and discards any other keyword argument, so there is nothing to map a
        hint onto.

        Parameters
        ----------
        question : str
            The question, passed upstream unchanged.
        k : int, optional
            Maximum passages to return, passed as ``topN``. Upstream slices its
            ranked passages at ``topN``, so this is a passage count and compares
            directly with a dense arm's ``k``.
        scope : str or None, optional
            Only ``None`` is accepted.
        hints : Hints or None, optional
            Accepted and ignored.

        Returns
        -------
        list of Evidence
            Ranked passages carrying provenance, with no per-passage score. Each
            one is identified by the graph's passage node key.

        Raises
        ------
        ScopeUnsupported
            If ``scope`` is not ``None``. PageRank runs over the whole graph and
            the upstream signature discards extra keyword arguments, so there is
            nowhere to push a filter, and filtering the finished ranking would
            return fewer than ``k``. Every adapter under ``graphrag/adapters/``
            refuses a scope.

        Notes
        -----
        Retrieval costs one model call. Under ``atlas_rag``'s defaults
        (``hipporag_mode="query2edge"``, ``is_filter_edges=True``) the candidate
        triples are filtered by a completion call inside this method, and the
        ``ner2node`` mode calls named-entity recognition instead, so a sweep pays
        one call per question here on top of whatever answering costs.
        """
        if scope is not None:
            raise ScopeUnsupported(
                f"{self.name} runs over the whole graph and cannot filter to "
                f"{scope!r}; build one graph per document instead",
            )
        passages, node_names = self._retriever.retrieve(question, topN=k)

        evidence: list[Evidence] = []
        for rank, passage in enumerate(passages):
            node_name = node_names[rank] if rank < len(node_names) else None
            identifier = self._passage_id(node_name)
            evidence.append(
                Evidence(
                    text=passage,
                    provenance=Provenance(
                        doc_id=identifier or "unknown",
                        chunk_id=identifier,
                    ),
                    # atlas_rag ranks without scoring, so the rank goes in extra
                    # and no pseudo-score is invented.
                    score=None,
                    retriever=self.name,
                    extra={"rank": rank},
                )
            )
        return evidence

    def _passage_id(self, node_name: Any) -> str | None:
        """Recover the graph's passage node key from one upstream identifier.

        Falls back to what upstream returned when the map holds no entry for it,
        which covers a stand-in retriever that already reports identifiers.
        """
        if node_name is None:
            return None
        return self._passage_ids.get(str(node_name), str(node_name))
