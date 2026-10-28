"""The boundary type every retrieval method returns.

This exists to break a coupling. LightRAG, HippoRAG 2, and Microsoft GraphRAG
each fuse retrieval, provenance, and answer formatting into one return value, so
using any of them meant taking their output shape too. Recovering the underlying
chunks from LightRAG previously meant regex-scraping a markdown string, which
breaks whenever the library changes its formatting.

Every adapter returns ``list[Evidence]`` instead. Retrieval is what varies;
provenance is uniform; the answer is the practitioner's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..provenance import Provenance


@dataclass(frozen=True)
class Evidence:
    """One retrieved passage and where it came from.

    Parameters
    ----------
    text : str
        The passage content, as it would be given to a generator.
    provenance : Provenance
        Where the passage came from, for citation.
    score : float or None, optional
        Retriever-assigned relevance, when the method exposes one. Scores are
        not comparable across methods and exist for ranking within one result.
    retriever : str, optional
        Name of the method that produced this, so mixed result sets stay
        attributable.
    extra : dict, optional
        Framework-specific passthrough, such as the graph entities a method used
        to reach this passage. Nothing in the shared code reads it; it is there
        so an adapter never has to drop information it cannot express.
    """

    text: str
    provenance: Provenance
    score: float | None = None
    retriever: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Return the source document identifier."""
        return self.provenance.doc_id

    @property
    def chunk_id(self) -> str | None:
        """Return the source chunk identifier, when the method reports one."""
        return self.provenance.chunk_id

    def cite(self) -> str:
        """Render a short citation for this passage."""
        return self.provenance.citation()


def to_runs(evidence: list[Evidence]) -> list[str]:
    """Reduce evidence to ranked chunk IDs for the evaluation harness.

    The harness scores ranked identifiers, so this is the bridge between a
    retriever and ``retrieval-failure rate``. Falls back to the document ID when
    a method does not report chunk-level provenance, which keeps
    document-granular methods scoreable rather than silently empty.

    What comes back has to be in the corpus's own identifier space, because the
    judgments are. An adapter whose library numbers chunks in a space of its own
    reports no chunk ID for that reason, and lands on the document ID here.

    Parameters
    ----------
    evidence : list of Evidence
        Retrieved passages, in rank order.

    Returns
    -------
    list of str
        Identifiers in rank order, de-duplicated, preserving first appearance.
    """
    seen: dict[str, None] = {}
    for item in evidence:
        seen.setdefault(item.chunk_id or item.doc_id, None)
    return list(seen)


def unique_documents(evidence: list[Evidence]) -> list[str]:
    """Return the distinct source documents, in rank order."""
    seen: dict[str, None] = {}
    for item in evidence:
        seen.setdefault(item.doc_id, None)
    return list(seen)
