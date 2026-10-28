"""The retrieval half of the shared evaluation harness.

Score a retriever in four steps: build a judged slice, index it, run the
queries, and score the ranking.

>>> from aieng.advanced_rag.retrieval import (
...     DenseRetriever,
...     HashEmbedder,
...     build_slice,
...     run_retriever,
... )
>>> from aieng.advanced_rag.evaluation import score_retrieval
>>> sliced = build_slice(cuad_test_json, max_contracts=20)  # doctest: +SKIP
>>> retriever = DenseRetriever(HashEmbedder())  # doctest: +SKIP
>>> retriever.index(sliced.chunks)  # doctest: +SKIP
>>> runs = run_retriever(retriever, sliced.answerable, k=10)  # doctest: +SKIP
>>> score_retrieval(runs, sliced.qrels, k=10)  # doctest: +SKIP

``retrieval.interface`` is the surface a caller programs against:
``retrieve(question, k, *, scope, hints)`` returns text-carrying
:class:`~aieng.advanced_rag.graphrag.evidence.Evidence`, and ``index`` and
``search`` stay each arm's internal ranking core. ``retrieval.conformance``
checks an implementation against that interface and is reached by path, so this
package stays importable without it.

The modules that need a model, numpy, or Qdrant are not imported here, so
importing this package pulls in nothing beyond the standard library. Reach them
by path: ``retrieval.stores`` for the Qdrant vector store, ``retrieval.index``
for batched dense search, ``retrieval.late`` for late chunking,
``retrieval.rerankers`` for cross-encoder reranking. ``DenseRetriever`` imports
the store when it is built, so it needs the ``qdrant`` extra.
"""

from .chunkers import (
    DEFAULT_SEPARATORS,
    ChunkingResult,
    Item,
    all_facts,
    fixed_chars,
    recursive_tokens,
    semantic,
    sentence_spans,
    serialise,
    table_aware_chunks,
    timed,
)
from .contextual import ContextualCost, contextualise
from .corpus import (
    AnswerSpan,
    Chunk,
    Query,
    Slice,
    build_slice,
    chunk_contract,
    chunk_stats,
    load_cuad,
    parse_category,
    relevant_chunk_ids,
)
from .embedders import (
    MODELS,
    CachedQueryEmbedder,
    Embedder,
    HashEmbedder,
    ModelSpec,
    SentenceTransformerEmbedder,
    apply_prompt,
    spec_for,
    tokenize,
)
from .interface import (
    CAPABILITIES,
    CAPABILITY_METHODS,
    Candidate,
    ChunkStore,
    Expandable,
    Fetchable,
    Findable,
    Greppable,
    Hints,
    Indexable,
    Readable,
    Retriever,
    RetrieverBase,
    ScopeUnsupported,
    capabilities_of,
    declares,
    is_retriever,
)
from .retrievers import (
    ChunkStoreRetriever,
    DenseRetriever,
    QdrantRetriever,
    Ranker,
    Retrieved,
    ShuffledRetriever,
    run_retriever,
)
from .sparse import (
    BM25Retriever,
    HybridRetriever,
    SparseEncoder,
    sparse_dot,
    sparse_index,
    weighted_rrf,
)
from .tables import CellFact, TableGrid, attachment_report, is_numeric_cell
from .tokens import (
    OffsetTokenIndex,
    TokenCounter,
    WhitespaceTokenIndex,
    build_token_index,
    load_tokenizer,
)


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_METHODS",
    "DEFAULT_SEPARATORS",
    "MODELS",
    "AnswerSpan",
    "BM25Retriever",
    "CachedQueryEmbedder",
    "Candidate",
    "CellFact",
    "Chunk",
    "ChunkStore",
    "ChunkStoreRetriever",
    "ChunkingResult",
    "ContextualCost",
    "DenseRetriever",
    "Embedder",
    "HashEmbedder",
    "Hints",
    "HybridRetriever",
    "Item",
    "ModelSpec",
    "OffsetTokenIndex",
    "QdrantRetriever",
    "Query",
    "Ranker",
    "RetrieverBase",
    "Retrieved",
    "Retriever",
    "ScopeUnsupported",
    "SentenceTransformerEmbedder",
    "ShuffledRetriever",
    "Slice",
    "SparseEncoder",
    "Expandable",
    "Fetchable",
    "Findable",
    "Greppable",
    "Indexable",
    "Readable",
    "TableGrid",
    "TokenCounter",
    "WhitespaceTokenIndex",
    "all_facts",
    "apply_prompt",
    "attachment_report",
    "build_slice",
    "build_token_index",
    "capabilities_of",
    "chunk_contract",
    "chunk_stats",
    "contextualise",
    "declares",
    "fixed_chars",
    "is_numeric_cell",
    "is_retriever",
    "load_cuad",
    "load_tokenizer",
    "parse_category",
    "recursive_tokens",
    "relevant_chunk_ids",
    "run_retriever",
    "semantic",
    "sentence_spans",
    "serialise",
    "sparse_dot",
    "sparse_index",
    "spec_for",
    "table_aware_chunks",
    "timed",
    "tokenize",
    "weighted_rrf",
]
