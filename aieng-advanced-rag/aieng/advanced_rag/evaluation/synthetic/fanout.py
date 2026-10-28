"""Fan-out questions: the shape a document pair cannot express.

A fan-out question's gold set is every document satisfying a condition, so a
top-k retriever has to find all of them, not the best few. The seed is a
corpus tag (:func:`~aieng.advanced_rag.evaluation.synthetic.sampling.tag_conditions`),
which no method authored, which defines the gold set exactly, and whose size is
known before a model is called. An entity from an extracted graph is not used,
because it would seed the set from one method's graph.

In the experiment on 258 ServerFault documents, fan-out questions had an
all-gold failure rate of 0.667 at k of 10 where pair questions had none, so
question shape kept retrieval headroom that corpus size did not.

Three checks before a question is kept, cheapest first: structural coverage
from the model's own per-document contributions (:func:`coverage`), the
single-document defeat test, and the all-but-one test in
:func:`~aieng.advanced_rag.evaluation.synthetic.filters.all_but_one_suffices`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm import Model, parse_json
from .sampling import Condition, Document


FANOUT_SYSTEM = (
    "You write questions that can only be answered by consulting every document "
    "in a set. Reply with one JSON object and nothing else."
)

FANOUT_PROMPT = """Below are all {count} documents in this corpus matching one
condition: {condition}.

Write one question about them with all of these properties:

1. Answering it completely requires information from EVERY document below.
   Missing any one of them must make the answer wrong or incomplete.
2. It does not state how many documents, items, or cases there are. The reader
   must find that out.
3. It does not mention documents, sources, tags, or identifiers. It reads as a
   question someone would actually ask about this subject area.
4. It has a definite answer, not an opinion.

Documents:
{documents}

Return JSON with these keys:
  "question": the question
  "answer": the full answer, which must draw on every document
  "per_document": an object mapping each document id to the exact sentence,
     copied from that document, that it contributes and no other document does
  "reject": true if these documents have no genuine common subject and no such
     question can be written honestly
  "reason": one sentence, why you wrote this question or why you rejected
"""


@dataclass(frozen=True)
class FanoutProposal:
    """One generated fan-out question.

    Parameters
    ----------
    question, answer : str
        The generated pair.
    condition : Condition
        What defined the gold set.
    per_document : dict
        Document id to the sentence the model claims it contributes. The
        coverage check reads it, and the quote check locates it.
    reason : str
        The model's justification, kept for auditing.
    """

    question: str
    answer: str
    condition: Condition
    per_document: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    @property
    def needs_documents(self) -> list[str]:
        """Return the gold document set: every document satisfying the condition."""
        return list(self.condition.documents)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "question": self.question,
            "answer": self.answer,
            "needs_documents": self.needs_documents,
            "question_type": "fan_out",
            "condition": self.condition.to_dict(),
            "per_document": self.per_document,
            "reason": self.reason,
        }


def render_documents(documents: Sequence[Document], max_chars: int = 1500) -> str:
    """Render the condition's documents for the prompt."""
    return "\n\n".join(
        f"[{d.doc_id}] {d.title}\n{d.text[:max_chars]}" for d in documents
    )


def propose(
    model: Model,
    condition: Condition,
    documents: Sequence[Document],
    max_chars: int = 1500,
) -> FanoutProposal | None:
    """Ask for one fan-out question over a condition's documents.

    Returns
    -------
    FanoutProposal or None
        ``None`` when the model rejects the condition. A tag that groups
        unrelated documents cannot support an honest fan-out question.
    """
    payload = parse_json(
        model(
            FANOUT_SYSTEM,
            FANOUT_PROMPT.format(
                count=condition.size,
                condition=condition.describe(),
                documents=render_documents(documents, max_chars),
            ),
        )
    )
    if payload.get("reject") or not payload.get("question"):
        return None
    per_document = payload.get("per_document")
    return FanoutProposal(
        question=str(payload["question"]).strip(),
        answer=str(payload.get("answer", "")).strip(),
        condition=condition,
        per_document=(
            {str(k): str(v) for k, v in per_document.items()}
            if isinstance(per_document, dict)
            else {}
        ),
        reason=str(payload.get("reason", ""))[:300],
    )


def coverage(proposal: FanoutProposal) -> float:
    """Return the share of gold documents the model claims a distinct contribution from.

    A model asked for a fan-out question sometimes writes one answerable from
    two of eight documents and pads the rest. This is the cheap structural
    check on that, before any model-judged one.
    """
    if not proposal.condition.documents:
        return 0.0
    claimed = {
        doc_id
        for doc_id, contribution in proposal.per_document.items()
        if doc_id in proposal.condition.documents and contribution.strip()
    }
    return len(claimed) / proposal.condition.size


__all__ = ["FanoutProposal", "coverage", "propose", "render_documents"]
