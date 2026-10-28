"""Retrieval adapters. Optional dependencies are imported lazily inside each.

Importing this package never pulls LightRAG or atlas-rag; the adapter classes
raise :class:`~aieng.advanced_rag.graphrag.protocol.MissingExtraError` naming the
extra to install, only when actually constructed.
"""

from .dense import DenseRetriever
from .hipporag2 import HippoRAG2Retriever
from .lightrag import LightRAGRetriever


__all__ = ["DenseRetriever", "HippoRAG2Retriever", "LightRAGRetriever"]
