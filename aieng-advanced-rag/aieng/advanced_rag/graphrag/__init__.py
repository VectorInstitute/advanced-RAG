"""Method-agnostic GraphRAG retrieval.

Retrieval methods are swappable behind one interface; provenance is uniform;
answer formatting is the practitioner's business. That interface is
:class:`~aieng.advanced_rag.retrieval.interface.Retriever`, shared with every
other retrieval method in the package, because a graph method is a retrieval
method.

    from aieng.advanced_rag.graphrag import DenseRetriever, Evidence

    retriever = DenseRetriever(encode=my_local_encoder)
    retriever.index(chunks)
    evidence = retriever.retrieve("why does nginx return 502?", k=10)

Swap ``DenseRetriever`` for ``LightRAGRetriever`` or ``HippoRAG2Retriever``
without changing anything downstream. Install those with the ``lightrag`` and
``hipporag`` extras.
"""

from .adapters import DenseRetriever, HippoRAG2Retriever, LightRAGRetriever
from .evidence import Evidence, to_runs, unique_documents
from .protocol import GraphRetriever, MissingExtraError
from .synthesis import answer, build_prompt, format_sources


# ``GraphRetriever`` is imported for callers that still name it and is left
# out of ``__all__``, because it is a deprecated alias rather than a surface.
__all__ = [
    "DenseRetriever",
    "Evidence",
    "HippoRAG2Retriever",
    "LightRAGRetriever",
    "MissingExtraError",
    "answer",
    "build_prompt",
    "format_sources",
    "to_runs",
    "unique_documents",
]
