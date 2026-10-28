"""Difficulty, recorded as numbers on each question and never used to filter.

Difficulty is measured, not rated by an LLM. GRADE (EMNLP 2025 Findings) places
a question by hop count and by how far the question sits from its evidence for
the retriever, and error rates follow that grid.
These labels record the same two axes per question, so a result can be sliced
by them:

* ``hops`` and ``spans``: evidence documents and gold spans.
* ``bm25_rank`` and ``dense_rank``: where the probe index ranks the best gold
  chunk, or ``None`` beyond depth 100.
* ``gold_cosine``: the dense similarity between the question and its closest
  gold quote.
* ``leaked_terms``: content words the question shares with its gold quote.

Filtering on any of these would build a set selected to defeat, or to suit, one
retriever.
"""

from __future__ import annotations

from typing import Any

from .probe import ProbeIndex
from .rewrite import leaked_terms
from .sampling import cosine
from .testset import SyntheticQuestion


def difficulty(
    question: SyntheticQuestion,
    probe: ProbeIndex,
    document_frequency: dict[str, int],
    total_documents: int,
    *,
    leak_share: float = 0.5,
) -> dict[str, Any]:
    """Return the recorded difficulty measures for one question.

    Parameters
    ----------
    question : SyntheticQuestion
        An answerable or null question.
    probe : ProbeIndex
        The generator's index over the indexed documents.
    document_frequency : dict
        Term document frequencies over the indexed documents.
    total_documents : int
        Number of indexed documents.
    leak_share : float, optional
        Terms in more than this share of documents are not counted as leaked.

    Returns
    -------
    dict
        The labels. A null question gets only its length.
    """
    labels: dict[str, Any] = {"question_chars": len(question.question)}
    if not question.is_answerable:
        return labels
    spans = [(s.doc_id, s.start, s.end) for s in question.evidence]
    labels["hops"] = len(question.documents)
    labels["spans"] = len(question.evidence)
    for method in ("bm25", "dense"):
        labels[f"{method}_rank"] = probe.gold_rank(
            question.question, spans, method=method, scope=question.scope
        )
    vector = probe.encode_query(question.question)
    gold_vectors = probe.encode_texts(
        [s.quote for s in question.evidence if s.role == "gold"]
        or [question.evidence[0].quote]
    )
    labels["gold_cosine"] = round(max(cosine(vector, v) for v in gold_vectors), 4)
    quote = " ".join(s.quote for s in question.evidence if s.role == "gold")
    leaked = leaked_terms(
        question.question,
        "",
        quote,
        document_frequency,
        total_documents,
        max_document_share=leak_share,
    )
    labels["leaked_terms"] = list(leaked)
    labels["leak_count"] = len(leaked)
    return labels


__all__ = ["difficulty"]
