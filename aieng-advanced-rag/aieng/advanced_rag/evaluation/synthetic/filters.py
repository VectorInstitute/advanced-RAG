"""The checks a candidate question passes before it joins a set.

In order of cost:

1. Quote located: done in :mod:`.statements`, free.
2. Gold-only round trip (:func:`round_trip`): the checker, which should be a
   different model from the generator, answers from the gold quotes alone, and
   the judge compares that answer with the reference. Alberti et al. (2019)
   keep a generated question only when a QA model reproduces its answer; here
   the evidence shown is only the quote, so the quote must carry the answer.
3. Single-document defeat and all-but-one (:func:`single_document_suffices`,
   :func:`all_but_one_suffices`), for multi-hop and fan-out questions. Ported
   from the experiment. They test the property that defines multi-hop
   directly, without a graph and without favouring a retrieval method.
4. Uniqueness (:func:`uniqueness`): the checker reads the top non-gold
   passages for the question. A passage that gives the same answer joins the
   gold set as an equivalent span; one that gives a different answer makes the
   question ambiguous, and it is dropped. Contracts, policies, and filings
   repeat themselves, so the first case is common.
5. Near-duplicate removal (:func:`deduplicate`) across the accepted set.

What is never a filter: whether a retriever found the gold passage. ARES keeps
only questions a retriever returns at rank 1, which selects for that
retriever. The ranks are recorded in :mod:`.labels` instead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm import Model, parse_json
from .probe import Hit
from .sampling import cosine
from .statements import locate_quote
from .testset import EvidenceSpan


ANSWER_SYSTEM = (
    "You answer questions strictly from the excerpts you are given. Reply with "
    "one JSON object and nothing else."
)

ANSWER_PROMPT = """Answer the question using ONLY the excerpts below. Do not use
anything you know from elsewhere.

Question: {question}

Excerpts:
{excerpts}

If the excerpts do not contain the answer, set "answerable" to false.

Return JSON: {{"answerable": true/false, "answer": "the answer, or empty"}}"""

JUDGE_SYSTEM = "You grade answers against a reference. Reply with one JSON object only."

JUDGE_PROMPT = """Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

Does the candidate give the same answer as the reference? Ignore wording,
length, and extra detail that does not contradict the reference.

- "match": the same essential answer.
- "partial": part of the reference is missing, or the candidate hedges.
- "mismatch": a different or contradictory answer, or no answer.

Return JSON: {{"verdict": "match" | "partial" | "mismatch", "reason": "one sentence"}}"""


def render_excerpts(texts: Sequence[str], max_chars: int = 1600) -> str:
    """Return excerpts numbered for a prompt, each truncated."""
    return "\n\n".join(
        f"[{index}] {text[:max_chars]}" for index, text in enumerate(texts, start=1)
    )


def answer_from(model: Model, question: str, texts: Sequence[str]) -> dict[str, Any]:
    """Ask a model to answer from the given excerpts only.

    Returns
    -------
    dict
        ``answerable`` and ``answer``.
    """
    payload = parse_json(
        model(
            ANSWER_SYSTEM,
            ANSWER_PROMPT.format(question=question, excerpts=render_excerpts(texts)),
        )
    )
    answer = str(payload.get("answer") or "").strip()
    return {
        "answerable": bool(payload.get("answerable")) and bool(answer),
        "answer": answer,
    }


def judge_match(
    model: Model, question: str, reference: str, candidate: str
) -> dict[str, str]:
    """Ask the judge whether a candidate answer matches the reference.

    Returns
    -------
    dict
        ``verdict`` (``match``, ``partial``, or ``mismatch``) and ``reason``.
    """
    payload = parse_json(
        model(
            JUDGE_SYSTEM,
            JUDGE_PROMPT.format(
                question=question, reference=reference, candidate=candidate
            ),
        )
    )
    verdict = str(payload.get("verdict") or "mismatch").strip().lower()
    if verdict not in ("match", "partial", "mismatch"):
        verdict = "mismatch"
    return {"verdict": verdict, "reason": str(payload.get("reason") or "")[:300]}


def round_trip(
    checker: Model,
    judge: Model,
    question: str,
    reference: str,
    quotes: Sequence[str],
) -> dict[str, Any]:
    """Answer from the gold quotes alone and compare with the reference.

    Parameters
    ----------
    checker : Model
        Answers from the quotes. Use a different model from the generator.
    judge : Model
        Compares the checker's answer with the reference.
    question, reference : str
        The candidate.
    quotes : sequence of str
        The gold quotes.

    Returns
    -------
    dict
        ``passed``, the checker's ``answer``, the judge's ``verdict``, and
        ``reason``. The judge is not called when the checker finds no answer.
    """
    answered = answer_from(checker, question, quotes)
    if not answered["answerable"]:
        return {
            "passed": False,
            "answer": "",
            "verdict": "unanswerable",
            "reason": "checker found no answer in the gold quotes",
        }
    verdict = judge_match(judge, question, reference, answered["answer"])
    return {
        "passed": verdict["verdict"] == "match",
        "answer": answered["answer"],
        "verdict": verdict["verdict"],
        "reason": verdict["reason"],
    }


UNIQUE_SYSTEM = (
    "You check whether a question has more than one source. Reply with one JSON "
    "object and nothing else."
)

UNIQUE_PROMPT = """Question: {question}
Reference answer: {reference}

Below are passages from the same collection. For each passage that ON ITS OWN
fully answers the question, copy the sentence that answers it exactly, and say
whether its answer agrees with the reference answer. Skip passages that are
only related, or that answer only part of the question.

{passages}

Return JSON: {{"answering": [{{"passage": <number>, "quote": "<exact sentence>", "agrees": true/false}}]}}
Return {{"answering": []}} when no passage fully answers it."""


@dataclass(frozen=True)
class Uniqueness:
    """What the uniqueness check found.

    Parameters
    ----------
    status : str
        ``unique``, ``equivalent`` (another passage gives the same answer), or
        ``ambiguous`` (another passage gives a different answer).
    equivalent : tuple of EvidenceSpan
        Spans to add to the gold set when ``status`` is ``equivalent``.
    checked : int
        Passages shown to the checker.
    """

    status: str
    equivalent: tuple[EvidenceSpan, ...] = ()
    checked: int = 0


def uniqueness(
    checker: Model,
    question: str,
    reference: str,
    hits: Sequence[Hit],
    texts: dict[str, str],
) -> Uniqueness:
    """Look for a second passage that answers the question.

    Parameters
    ----------
    checker : Model
        Reads the passages.
    question, reference : str
        The candidate.
    hits : sequence of Hit
        Non-gold probe chunks, already excluding the gold spans.
    texts : dict
        Document id to full text, for locating the checker's quotes.

    Returns
    -------
    Uniqueness
        The status and any equivalent spans. An equivalent quote that cannot be
        located falls back to the whole chunk's span.
    """
    if not hits:
        return Uniqueness("unique")
    payload = parse_json(
        checker(
            UNIQUE_SYSTEM,
            UNIQUE_PROMPT.format(
                question=question,
                reference=reference,
                passages=render_excerpts([hit.chunk.text for hit in hits]),
            ),
        )
    )
    answering = payload.get("answering") or []
    spans: list[EvidenceSpan] = []
    for item in answering if isinstance(answering, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("passage", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(hits):
            continue
        if not item.get("agrees", False):
            return Uniqueness("ambiguous", (), len(hits))
        chunk = hits[index].chunk
        text = texts[chunk.doc_id]
        located = locate_quote(
            text, str(item.get("quote") or ""), start=chunk.start, end=chunk.end
        )
        start, end = located if located else (chunk.start, chunk.end)
        spans.append(
            EvidenceSpan(chunk.doc_id, start, end, text[start:end], "equivalent")
        )
    if spans:
        return Uniqueness("equivalent", tuple(spans), len(hits))
    return Uniqueness("unique", (), len(hits))


@dataclass(frozen=True)
class Duplicate:
    """A question dropped as a near-duplicate of a kept one."""

    dropped: str
    kept: str
    similarity: float


def deduplicate(
    items: Sequence[tuple[str, str, str]],
    vectors: dict[str, list[float]],
    threshold: float = 0.92,
) -> tuple[list[str], list[Duplicate]]:
    """Keep one question of each near-duplicate group.

    When two questions are closer than ``threshold``, the one from the document
    with fewer kept questions so far stays, so coverage spreads across the
    corpus; ties keep the earlier one.

    Parameters
    ----------
    items : sequence of (query_id, document, question)
        Candidates in their accepted order.
    vectors : dict
        Query id to unit question vector.
    threshold : float, optional
        Cosine above which two questions are duplicates.

    Returns
    -------
    tuple of (list of str, list of Duplicate)
        Kept identifiers in order, and what was dropped for which.
    """
    kept: list[tuple[str, str]] = []
    per_document: Counter[str] = Counter()
    dropped: list[Duplicate] = []
    for query_id, document, _question in items:
        clash = None
        for index, (other, other_document) in enumerate(kept):
            similarity = cosine(vectors[query_id], vectors[other])
            if similarity > threshold:
                clash = (index, other, other_document, similarity)
                break
        if clash is None:
            kept.append((query_id, document))
            per_document[document] += 1
            continue
        index, other, other_document, similarity = clash
        if per_document[document] + 1 < per_document[other_document]:
            kept[index] = (query_id, document)
            per_document[other_document] -= 1
            per_document[document] += 1
            dropped.append(Duplicate(other, query_id, round(similarity, 4)))
        else:
            dropped.append(Duplicate(query_id, other, round(similarity, 4)))
    return [query_id for query_id, _ in kept], dropped


NULL_SYSTEM = (
    "You check whether a collection answers a question. Reply with one JSON "
    "object and nothing else."
)

NULL_PROMPT = """Question: {question}

Below are the passages a search returned for it. Do they answer the question?
A passage that implies a definite yes or no answers it. A passage that states
the answer for a different case, party, or period does not.

- "yes": some passage answers it.
- "partly": some passage answers part of it.
- "no": nothing here answers it.

{passages}

Return JSON: {{"answered": "yes" | "partly" | "no", "passage": <number or null>, "reason": "one sentence"}}"""


def corpus_answers(
    checker: Model, question: str, hits: Sequence[Hit]
) -> dict[str, Any]:
    """Ask whether any retrieved passage answers a would-be null question.

    The CRUMQs check (Liu et al., ECIR 2026): a null question is kept only if
    the top passages from the corpus cannot answer it.

    Returns
    -------
    dict
        ``answered`` (``yes``, ``partly``, or ``no``), the answering
        ``passage`` or ``None``, and ``reason``.
    """
    payload = parse_json(
        checker(
            NULL_SYSTEM,
            NULL_PROMPT.format(
                question=question,
                passages=render_excerpts(
                    [hit.chunk.text for hit in hits], max_chars=1300
                ),
            ),
        )
    )
    answered = str(payload.get("answered") or "yes").strip().lower()
    if answered not in ("yes", "partly", "no"):
        answered = "yes"
    return {
        "answered": answered,
        "passage": payload.get("passage"),
        "reason": str(payload.get("reason") or "")[:300],
    }


# ---------------------------------------------------------------------------
# Multi-hop checks, ported from the experiment.
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = "You judge evaluation questions. Reply with one JSON object only."

VERIFY_PROMPT = """Judge a question written to require more than one source.

QUESTION (judge naturalness on this text and nothing else):
{question}

Stated answer: {answer}

Internal note from the writer, describing what links the sources. This is
metadata, not part of the question. It is allowed to name source ids and to
state the linking fact outright; do not penalise it for either:
{bridge}

Judge the three fields independently. A question can be natural but not
multi-hop, or multi-hop but wrong.

Reply with exactly:
{{"is_truly_multi_hop": true/false,
  "is_natural": true/false,
  "is_answerable": true/false,
  "reason": "one sentence naming which field failed and why"}}

is_truly_multi_hop: false only if the QUESTION could be fully answered from one
source, or if the link is generic topic overlap rather than a specific shared
fact. Judge this on the question and the link, not on the writing.
is_natural: false only if THE QUESTION TEXT itself reads like a test artefact,
names a source id, or says "document A", "the documents", or "the corpus".
Ignore the internal note entirely when judging this.
is_answerable: false if the stated answer is wrong or unsupported."""

SUFFICIENCY_SYSTEM = (
    "You judge whether a source answers a question. Reply with one JSON object only."
)

SUFFICIENCY_PROMPT = """Can this question be answered COMPLETELY from the single source below?

Question: {question}

Source ({doc_id}):
{document}

Answer "yes" only if every part of the question is answerable from this source
alone. Partial information is "no".

Reply with exactly:
{{"sufficient": true/false, "reason": "one sentence"}}"""

ALL_BUT_ONE_SYSTEM = (
    "You judge whether a question can still be answered completely from a set of "
    "documents. Reply with one JSON object and nothing else."
)

ALL_BUT_ONE_PROMPT = """Question: {question}

Below are documents. One document that the question's full answer depends on has
been REMOVED from this set.

{documents}

Can the question still be answered completely and correctly from what remains?

Return JSON:
  "sufficient": true if the remaining documents fully answer the question
  "reason": one sentence
"""


def verify_multi_hop(
    model: Model, question: str, answer: str, bridge: str
) -> dict[str, Any]:
    """Ask a model whether the question needs more than one source.

    Cheap, and it catches obvious cases, but it shares the generator's blind
    spots, so :func:`single_document_suffices` decides.
    """
    payload = parse_json(
        model(
            VERIFY_SYSTEM,
            VERIFY_PROMPT.format(question=question, answer=answer, bridge=bridge),
        )
    )
    return {
        "is_truly_multi_hop": bool(payload.get("is_truly_multi_hop", False)),
        "is_natural": bool(payload.get("is_natural", False)),
        "is_answerable": bool(payload.get("is_answerable", False)),
        "reason": str(payload.get("reason", "")),
    }


def single_document_suffices(
    model: Model, question: str, doc_id: str, document: str, max_chars: int = 6000
) -> bool:
    """Return whether one document alone answers the question completely.

    This is the definition of not-multi-hop, tested directly.
    """
    payload = parse_json(
        model(
            SUFFICIENCY_SYSTEM,
            SUFFICIENCY_PROMPT.format(
                question=question, doc_id=doc_id, document=document[:max_chars]
            ),
        )
    )
    return bool(payload.get("sufficient", False))


def all_but_one_suffices(
    model: Model,
    question: str,
    remaining: Sequence[tuple[str, str]],
    max_chars: int = 1200,
) -> bool:
    """Return whether the question stays answerable with one gold document removed.

    The defining test for fan-out: if it stays answerable, the gold set is
    larger than the question needs.

    Parameters
    ----------
    model : Model
        The checker.
    question : str
        The question.
    remaining : sequence of (doc_id, text)
        Every gold document except the one removed.
    max_chars : int, optional
        Truncation per document.
    """
    rendered = "\n\n".join(
        f"[{doc_id}]\n{text[:max_chars]}" for doc_id, text in remaining
    )
    payload = parse_json(
        model(
            ALL_BUT_ONE_SYSTEM,
            ALL_BUT_ONE_PROMPT.format(question=question, documents=rendered),
        )
    )
    return bool(payload.get("sufficient", False))


@dataclass
class MultiHopVerdict:
    """Why a multi-hop proposal was kept or dropped."""

    accepted: bool
    verifier: dict[str, Any]
    sufficient_documents: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "accepted": self.accepted,
            "verifier": self.verifier,
            "single_documents_that_suffice": self.sufficient_documents,
            "reason": self.reason,
        }


def multi_hop_filters(
    model: Model,
    question: str,
    *,
    answer: str,
    bridge: str,
    gold: Sequence[str],
    documents: dict[str, str],
    require_natural: bool = True,
    min_documents: int = 2,
) -> MultiHopVerdict:
    """Run the verifier and then the single-document defeat test.

    Parameters
    ----------
    model : Model
        The checker.
    question, answer, bridge : str
        The proposal.
    gold : sequence of str
        The documents the proposal says it needs.
    documents : dict
        Document id to text.
    require_natural : bool, optional
        Drop questions the verifier flags as test artefacts.
    min_documents : int, optional
        Documents the type needs.

    Returns
    -------
    MultiHopVerdict
        The verdict and the signals behind it.
    """
    known = [d for d in gold if d in documents]
    verifier = verify_multi_hop(model, question, answer, bridge)
    if len(known) < min_documents:
        return MultiHopVerdict(
            False,
            verifier,
            [],
            f"needs {len(known)} known documents, requires {min_documents}",
        )
    if not verifier["is_truly_multi_hop"]:
        return MultiHopVerdict(False, verifier, [], "verifier: not multi-hop")
    if require_natural and not verifier["is_natural"]:
        return MultiHopVerdict(
            False, verifier, [], "verifier: reads as a test artefact"
        )
    sufficient = [
        doc_id
        for doc_id in known
        if single_document_suffices(model, question, doc_id, documents[doc_id])
    ]
    if sufficient:
        return MultiHopVerdict(
            False, verifier, sufficient, f"answerable from {sufficient[0]} alone"
        )
    return MultiHopVerdict(True, verifier)


__all__ = [
    "Duplicate",
    "MultiHopVerdict",
    "Uniqueness",
    "all_but_one_suffices",
    "answer_from",
    "corpus_answers",
    "deduplicate",
    "judge_match",
    "multi_hop_filters",
    "render_excerpts",
    "round_trip",
    "single_document_suffices",
    "uniqueness",
    "verify_multi_hop",
]
