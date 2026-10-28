"""In-domain questions the indexed corpus does not answer.

The harness's null detection needs unanswerable questions that look like the
answerable ones, and a team corpus has none. Out-of-domain nulls, such as
MultiHop-RAG's 301, are too easy to abstain on. Three kinds are built here,
all verified the same way:

* Held-out nulls: single-hop questions written from documents kept out of the
  index. They use the corpus's own vocabulary and their answer is outside the
  index by construction.
* Near-miss nulls: an accepted question with the one detail that selects its
  evidence changed. ContractNLI's ``NotMentioned`` labels are the real-world
  version: in-domain, and the document does not address it.
* Partial nulls: a multi-hop question whose second hop is in a held-out
  document, so a system finds half its evidence and must still abstain.

Verification follows CRUMQs (Liu et al., ECIR 2026): retrieve the top passages
from the indexed corpus with both BM25 and the dense encoder, and keep the
question only if the checker judges that none of them answers it, fully or in
part.
"""

from __future__ import annotations

from typing import Any

from .filters import corpus_answers
from .llm import Model, parse_json
from .probe import Hit, ProbeIndex


NEAR_MISS_SYSTEM = (
    "You write realistic questions that a given document does not answer. Reply "
    "with one JSON object and nothing else."
)

NEAR_MISS_PROMPT = """The documents are {domain}.

This question is answered by the statement below:

Question: {question}
Statement: {statement}

Write a new question that a user of these documents might ask, by changing the
ONE detail that makes the statement answer it: a party, a type of
information, an action, a period, a place, or a condition. The new question
must:
- stay natural and in the same domain, close to the original in form;
- NOT be answered by the statement, and not be answered by a definite yes or
  no that the statement implies;
- not be a trick, a negation of the original, or a question about the
  document's wording.

Return JSON: {{"question": "...", "changed": "what you changed, in a few words"}}"""


def near_miss_question(
    model: Model, question: str, statement: str, domain: str
) -> dict[str, str]:
    """Ask the generator to change the one detail that makes a question answerable.

    Returns
    -------
    dict
        ``question`` (empty when the reply had none) and ``changed``.
    """
    payload = parse_json(
        model(
            NEAR_MISS_SYSTEM,
            NEAR_MISS_PROMPT.format(
                domain=domain, question=question, statement=statement
            ),
        )
    )
    return {
        "question": str(payload.get("question") or "").strip(),
        "changed": str(payload.get("changed") or "")[:200],
    }


def null_evidence(
    probe: ProbeIndex, question: str, *, scope: str | None, k: int = 10
) -> list[Hit]:
    """Return the union of the BM25 and dense top ``k``, BM25 first."""
    seen: set[str] = set()
    hits: list[Hit] = []
    for method in ("bm25", "dense"):
        for hit in probe.search(question, method=method, k=k, scope=scope):
            if hit.chunk.chunk_id not in seen:
                seen.add(hit.chunk.chunk_id)
                hits.append(hit)
    return hits


def check_null(
    checker: Model,
    probe: ProbeIndex,
    question: str,
    *,
    scope: str | None,
    k: int = 10,
) -> dict[str, Any]:
    """Verify that the indexed corpus does not answer a would-be null question.

    Parameters
    ----------
    checker : Model
        Reads the retrieved passages.
    probe : ProbeIndex
        Built over the indexed documents only.
    question : str
        The candidate null question.
    scope : str or None
        The document the question is scoped to, or ``None`` for the corpus.
    k : int, optional
        Passages per retriever.

    Returns
    -------
    dict
        ``passed`` (true when nothing answers it), the checker's ``answered``
        verdict and ``reason``, and ``passages`` shown.
    """
    hits = null_evidence(probe, question, scope=scope, k=k)
    if not hits:
        return {
            "passed": True,
            "answered": "no",
            "reason": "no passages in scope",
            "passages": 0,
        }
    verdict = corpus_answers(checker, question, hits)
    return {
        "passed": verdict["answered"] == "no",
        "answered": verdict["answered"],
        "reason": verdict["reason"],
        "passages": len(hits),
    }


__all__ = [
    "NEAR_MISS_PROMPT",
    "check_null",
    "near_miss_question",
    "null_evidence",
]
