"""The bounded tool loop that turns a document pair into a question, or rejects it.

The similarity band proposes pairs cheaply but cannot tell whether a real
connection exists; two documents can be similar because they share a topic and
share no fact. Shown two documents and asked for a multi-hop question, a model
invents a bridge, and the set then grades methods on retrieving something that
was never in the corpus. So the agent gets tools and has to find the bridge
before it may write. It is expected to reject, and the rejection rate is
reported: a generator that never rejects is not checking. In the experiment,
6 of 30 pairs were rejected after reading.

Two changes from the experiment. The proposal must quote its evidence from
every document it needs, so a multi-hop question carries character spans like
a single-hop one. And its type is assigned from how many documents it needs,
not from the type requested, because the experiment memo found proposals that
one document answers.

The loop is about thirty lines and depends on no agent framework, which keeps
the mechanism readable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import Model, parse_json
from .sampling import CandidatePair
from .statements import locate_quote
from .taxonomy import QuestionType, type_for_documents
from .testset import EvidenceSpan


SYSTEM = (
    "You write evaluation questions for a retrieval system. You always reply "
    "with a single JSON object and nothing else. No prose, no markdown fences."
)

INSTRUCTIONS = """You are checking whether two documents can support a genuine multi-hop question.

Document A: {doc_a}
Document B: {doc_b}
Similarity: {similarity}
Terms both use that are rare in this corpus: {shared_terms}

Question type to write: {type_name}
{type_guidance}

You have tools. Use them before deciding; you have not read the documents yet.

Available actions, one JSON object per reply:
  {{"action": "read", "doc_id": "<id>"}}
      Read a document in full.
  {{"action": "search", "query": "<text>"}}
      Search the whole corpus for passages.
  {{"action": "propose", "question": "...", "answer": "...",
     "needs_documents": ["<id>", "<id>"], "bridge": "<the shared fact>",
     "evidence": [{{"doc_id": "<id>", "quote": "<exact sentence copied from it>"}}],
     "rationale": "why a single document is not enough"}}
      Only when you have found a real, specific connection. Quote the sentence
      each needed document contributes, copied exactly.
  {{"action": "reject", "reason": "..."}}
      When the documents share only a topic and no specific fact, or when any
      question you could write would be answerable from one document alone.

Rules for a proposed question:
- It must be answerable from the documents, and NOT from any one of them alone.
- Do not mention document ids, "document A", or "the documents" in the question.
  It must read like something a person would actually ask.
- Do not give away the bridging fact in the question itself.
- Be specific. Name real terms, values, or parties from the text.

Rejecting is a good outcome when the pair is weak. Most pairs are weak.

You have {remaining} actions left."""


@dataclass
class Proposal:
    """A question the agent believes needs more than one document."""

    question: str
    answer: str
    needs_documents: list[str]
    bridge: str
    rationale: str
    question_type: str
    pair: tuple[str, str]
    steps: int
    quotes: list[tuple[str, str]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "question": self.question,
            "answer": self.answer,
            "needs_documents": self.needs_documents,
            "bridge": self.bridge,
            "rationale": self.rationale,
            "question_type": self.question_type,
            "pair": list(self.pair),
            "agent_steps": self.steps,
            "quotes": [list(q) for q in self.quotes],
            "tool_calls": self.tool_calls,
        }


@dataclass
class Rejection:
    """The agent looked and found no real bridge."""

    pair: tuple[str, str]
    reason: str
    steps: int


def run_agent(
    model: Model,
    pair: CandidatePair,
    question_type: QuestionType,
    *,
    read_document: Callable[[str], str],
    search_corpus: Callable[[str, int], list[tuple[str, str]]],
    max_steps: int = 6,
    max_document_chars: int = 6000,
) -> Proposal | Rejection:
    """Run the bounded tool loop for one candidate pair.

    Parameters
    ----------
    model : Model
        The generator.
    pair : CandidatePair
        The proposed pair.
    question_type : QuestionType
        What shape of question to write.
    read_document : callable
        Document id to full text, or an empty string.
    search_corpus : callable
        ``(query, k)`` to ``(doc_id, snippet)`` pairs.
    max_steps : int, optional
        Tool-call budget before the agent must decide.
    max_document_chars : int, optional
        Truncation for a read, to bound prompt growth.

    Returns
    -------
    Proposal or Rejection
        A question, or the reason there is not one.
    """
    transcript: list[str] = []
    tool_calls: list[str] = []
    for step in range(max_steps):
        prompt = INSTRUCTIONS.format(
            doc_a=pair.doc_a,
            doc_b=pair.doc_b,
            similarity=pair.similarity,
            shared_terms=", ".join(pair.shared_terms) or "(none found)",
            type_name=question_type.name,
            type_guidance=question_type.prompt_guidance,
            remaining=max_steps - step,
        )
        if transcript:
            prompt += "\n\nWhat you have seen so far:\n" + "\n\n".join(transcript)
        action = parse_json(model(SYSTEM, prompt))
        kind = action.get("action")
        if kind == "read":
            doc_id = str(action.get("doc_id", ""))
            tool_calls.append(f"read:{doc_id}")
            body = read_document(doc_id)
            transcript.append(
                f"[read {doc_id}]\n{body[:max_document_chars] if body else '(no such document)'}"
            )
        elif kind == "search":
            query = str(action.get("query", ""))
            tool_calls.append(f"search:{query[:40]}")
            hits = search_corpus(query, 4)
            rendered = "\n".join(f"  ({doc}) {snippet[:300]}" for doc, snippet in hits)
            transcript.append(f"[search {query!r}]\n{rendered or '  (no hits)'}")
        elif kind == "propose":
            question = str(action.get("question", "")).strip()
            if not question:
                transcript.append("[error] propose had no question; try again")
                continue
            needs = [str(d) for d in (action.get("needs_documents") or [])]
            quotes = [
                (str(item.get("doc_id", "")), str(item.get("quote", "")))
                for item in action.get("evidence") or []
                if isinstance(item, dict)
            ]
            return Proposal(
                question=question,
                answer=str(action.get("answer", "")).strip(),
                needs_documents=needs or [pair.doc_a, pair.doc_b],
                bridge=str(action.get("bridge", "")).strip(),
                rationale=str(action.get("rationale", "")).strip(),
                question_type=question_type.name,
                pair=(pair.doc_a, pair.doc_b),
                steps=step + 1,
                quotes=quotes,
                tool_calls=tool_calls,
            )
        elif kind == "reject":
            return Rejection(
                pair=(pair.doc_a, pair.doc_b),
                reason=str(action.get("reason", "")).strip() or "unstated",
                steps=step + 1,
            )
        else:
            transcript.append(
                "[error] unrecognized reply; respond with one JSON object using "
                "one of: read, search, propose, reject"
            )
    return Rejection(
        pair=(pair.doc_a, pair.doc_b),
        reason=f"no decision within {max_steps} actions",
        steps=max_steps,
    )


def locate_evidence(
    proposal: Proposal, texts: dict[str, str]
) -> tuple[list[EvidenceSpan], list[str]]:
    """Locate each quoted sentence in its document.

    Returns
    -------
    tuple of (list of EvidenceSpan, list of str)
        The located spans, and the needed documents with no located quote. A
        proposal with any missing document fails the quote check.
    """
    spans: list[EvidenceSpan] = []
    for doc_id, quote in proposal.quotes:
        text = texts.get(doc_id)
        if text is None:
            continue
        located = locate_quote(text, quote)
        if located is not None:
            spans.append(
                EvidenceSpan(
                    doc_id, located[0], located[1], text[located[0] : located[1]]
                )
            )
    covered = {span.doc_id for span in spans}
    missing = [d for d in proposal.needs_documents if d not in covered]
    return spans, missing


def assigned_type(proposal: Proposal, requested: QuestionType) -> QuestionType:
    """Return the type the proposal is, from how many documents it needs."""
    return type_for_documents(requested, len(set(proposal.needs_documents)))


__all__ = [
    "INSTRUCTIONS",
    "Proposal",
    "Rejection",
    "assigned_type",
    "locate_evidence",
    "run_agent",
]
