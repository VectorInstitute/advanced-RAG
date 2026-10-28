"""Rewording a question so it does not carry its evidence's vocabulary.

A question written while looking at its evidence uses the evidence's terms, and
a lexical retriever handed those terms does not have to understand the
question. MuSiQue obfuscates for this reason. The paraphrase variant measures
whether rewording changes retrieval for single-hop questions on another corpus.

Every rewrite is verified. A rewrite that drops the leaked terms by making the
question vague is worse than none, so it must remove the leak and still pass
the round trip (single-hop), or still need both documents (multi-hop).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import Model, parse_json


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")

#: Function words of four letters or more, which match everywhere and leak nothing.
_STOP = frozenset(
    [
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "what",
        "when",
        "which",
        "shall",
        "such",
        "each",
        "into",
        "than",
        "they",
        "their",
        "there",
        "then",
        "them",
        "been",
        "were",
        "does",
        "would",
        "could",
        "should",
        "about",
        "other",
        "under",
        "upon",
        "only",
        "also",
        "must",
        "more",
        "most",
        "made",
        "make",
        "many",
        "much",
        "some",
        "does",
        "over",
        "any",
        "your",
        "those",
        "these",
        "where",
        "while",
        "being",
    ]
)


def terms(text: str, min_length: int = 4) -> set[str]:
    """Return lowercased terms of ``min_length`` characters or more, minus stopwords."""
    return {
        m.group(0).lower().rstrip(".,;:")
        for m in _WORD_RE.finditer(text)
        if len(m.group(0)) >= min_length and m.group(0).lower() not in _STOP
    }


def term_frequencies(texts: Sequence[str]) -> dict[str, int]:
    """Count how many texts each term appears in."""
    counts: dict[str, int] = {}
    for text in texts:
        for term in terms(text):
            counts[term] = counts.get(term, 0) + 1
    return counts


def leaked_terms(
    question: str,
    first_document: str,
    second_document: str,
    document_frequency: dict[str, int],
    total_documents: int,
    *,
    max_document_share: float = 0.05,
    limit: int = 12,
) -> tuple[str, ...]:
    """Return rare terms the question shares with the second document only.

    For a multi-hop question, the second document is the one the reader is
    supposed to have to find, and terms also in the first document are allowed.
    For a single-hop question, pass an empty first document and the gold quote
    as the second, with a looser ``max_document_share``.

    Parameters
    ----------
    question : str
        The question text.
    first_document : str
        Text whose vocabulary the question may use.
    second_document : str
        The evidence the question should not echo.
    document_frequency : dict
        Term to number of documents containing it.
    total_documents : int
        Corpus size, for ``max_document_share``.
    max_document_share : float, optional
        Terms in more than this share of documents count as common.
    limit : int, optional
        Most-distinctive terms to return.

    Returns
    -------
    tuple of str
        Leaked terms, rarest first.
    """
    ceiling = max(total_documents * max_document_share, 1)
    from_second = terms(second_document) - terms(first_document)
    leaked = [
        term
        for term in terms(question) & from_second
        if document_frequency.get(term, 0) <= ceiling
    ]
    leaked.sort(key=lambda term: (document_frequency.get(term, 0), term))
    return tuple(leaked[:limit])


PARAPHRASE_SYSTEM = (
    "You reword search questions without changing what they ask. Reply with one "
    "JSON object and nothing else."
)

PARAPHRASE_PROMPT = """Reword the question so that it does not reuse these words
from the text that answers it: {leaked}

Question: {question}
Known answer: {answer}

Rules:
1. Replace each listed word with the everyday words a person who has not read
   the text would use. Do not use a close variant of the same word.
2. Keep exactly the same answer. Do not widen or narrow what is asked.
3. Keep it one natural question. Do not mention documents or sources.
4. Do not make it vague. It must still have one specific answer.

Return JSON: {{"question": "the reworded question"}}"""

OBFUSCATE_SYSTEM = (
    "You rewrite multi-hop questions so that they cannot be answered by "
    "searching for the wording of the second document. Reply with one JSON "
    "object and nothing else."
)

OBFUSCATE_PROMPT = """Rewrite the question below so that a keyword or embedding
search cannot jump straight to the second document.

Question: {question}
Known answer: {answer}
How the two documents connect: {bridge}

The rewrite must obey all of these:

1. Remove these terms, which appear in both the question and the second
   document: {leaked}
   Replace each one with a description built from the FIRST document, so the
   reader has to read the first document to know what is being referred to.
2. Keep the answer exactly the same. Do not widen or narrow what is asked.
3. Keep the question a single natural question. Do not mention documents,
   sources, or identifiers.
4. Do not make the question vague. It must still have one specific answer.

First document (wording you may use):
{first_document}

Return JSON with these keys:
  "question": the rewritten question
  "removed": list of the terms you removed
  "substituted": how you referred to each removed term instead
"""


@dataclass(frozen=True)
class Obfuscation:
    """One rewrite and what it changed.

    Parameters
    ----------
    original : str
        The question before rewriting.
    rewritten : str
        The question after rewriting, or the original if the rewrite failed.
    level : str
        The obfuscation level requested; see ``taxonomy.OBFUSCATION_LEVELS``.
    leaked_before : tuple of str
        Leaked terms before the rewrite.
    leaked_after : tuple of str
        Leaked terms after. Empty is the goal.
    accepted : bool
        Whether the rewrite passed every check.
    checks : dict
        Individual check outcomes.
    """

    original: str
    rewritten: str
    level: str
    leaked_before: tuple[str, ...] = ()
    leaked_after: tuple[str, ...] = ()
    accepted: bool = False
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def leak_reduction(self) -> float:
        """Return the share of leaked terms removed. 1.0 means none remain."""
        if not self.leaked_before:
            return 1.0
        return 1 - len(self.leaked_after) / len(self.leaked_before)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return asdict(self)


def paraphrase_question(
    model: Model, question: str, answer: str, leaked: Sequence[str]
) -> str:
    """Ask the generator to reword a single-hop question without its leaked terms.

    Returns
    -------
    str
        The reworded question, or an empty string when the reply had none.
    """
    payload = parse_json(
        model(
            PARAPHRASE_SYSTEM,
            PARAPHRASE_PROMPT.format(
                question=question, answer=answer, leaked=", ".join(leaked)
            ),
        )
    )
    return str(payload.get("question") or "").strip()


def obfuscate_question(
    model: Model,
    question: str,
    *,
    answer: str,
    bridge: str,
    first_document: str,
    leaked: Sequence[str],
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Ask for a multi-hop rewrite that drops the second document's terms."""
    return parse_json(
        model(
            OBFUSCATE_SYSTEM,
            OBFUSCATE_PROMPT.format(
                question=question,
                answer=answer,
                bridge=bridge,
                leaked=", ".join(leaked) or "(none found; tighten the wording anyway)",
                first_document=first_document[:max_chars],
            ),
        )
    )


def check_rewrite(
    rewritten: str,
    answer: str,
    first_document: str,
    second_document: str,
    *,
    document_frequency: dict[str, int],
    total_documents: int,
    verify: Callable[[str, str], dict[str, Any]] | None = None,
    still_needs_both: Callable[[str], bool] | None = None,
    max_document_share: float = 0.05,
) -> dict[str, Any]:
    """Run the checks a multi-hop rewrite has to pass.

    ``verify`` and ``still_needs_both`` are injected so this runs without a
    model and reuses the generation filters instead of a second judge.
    """
    remaining = leaked_terms(
        rewritten,
        first_document,
        second_document,
        document_frequency,
        total_documents,
        max_document_share=max_document_share,
    )
    checks: dict[str, Any] = {
        "leaked_after": list(remaining),
        "leaks_removed": not remaining,
    }
    if verify is not None:
        verdict = verify(rewritten, answer)
        checks["still_answerable"] = bool(verdict.get("is_answerable", False))
        checks["still_multi_hop"] = bool(verdict.get("is_truly_multi_hop", False))
        checks["is_natural"] = bool(verdict.get("is_natural", False))
    if still_needs_both is not None:
        checks["single_document_suffices"] = bool(still_needs_both(rewritten))
    return checks


def accept(checks: dict[str, Any]) -> bool:
    """Decide whether a multi-hop rewrite replaces the original.

    Leaks removed is necessary but not sufficient: a rewrite that becomes
    unanswerable or collapses to one document is a regression. Naturalness is
    recorded and does not veto.
    """
    if not checks.get("leaks_removed", False):
        return False
    if checks.get("single_document_suffices", False):
        return False
    return all(checks.get(key, True) for key in ("still_answerable", "still_multi_hop"))


__all__ = [
    "Obfuscation",
    "accept",
    "check_rewrite",
    "leaked_terms",
    "obfuscate_question",
    "paraphrase_question",
    "term_frequencies",
    "terms",
]
