"""Variants that test whether quoting over a whole agreement can keep abstention.

The reference run found that the quote prompt over the whole agreement almost
never declines a question the agreement is silent on. Each function here is one
design for fixing that, run on the same sources as the ``fulldoc`` arm:

``fulldoc-sentinel``  one call; the prompt names a quote that means "no clause
                      supports this", and a statement resting on it is dropped.
``fulldoc-extract``   two calls; the first lists the clauses that address the
                      question, and an empty list abstains.
``fulldoc-check``     one extra call per answer; a second read asks whether the
                      quotes address the question's proposition or only a
                      related topic, and a "related only" answer abstains.
``answer-standalone`` one call per statement; rewrites the shipped generator's
                      bare "Yes" or "No" statements into sentences that stand
                      alone, for a checker that reads the claim without the
                      question.

Every function takes a ``complete`` callable and returns a
:class:`~aieng.advanced_rag.evaluation.citations.CitedAnswer`, so the model is
a parameter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace

from aieng.advanced_rag.evaluation.citations import (
    QUOTE_TEMPLATE,
    CitedAnswer,
    Statement,
    _first_json,
    parse_structured,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.graphrag.synthesis import build_prompt
from aieng.advanced_rag.provenance import Provenance


#: The quote a statement carries when no clause of the agreement supports it.
SENTINEL = "NO SUPPORTING CLAUSE"

SENTINEL_TEMPLATE = QUOTE_TEMPLATE.replace(
    '- If the sources do not answer the question, set "not_stated" to true and\n'
    '  leave "answer" and "statements" empty.\n',
    "- A clause on a related topic does not support a statement. Quote a clause\n"
    "  only if it states what the question asks about, or states the opposite.\n"
    "- If no clause does, the agreement is silent on the question. Write one\n"
    "  statement that says the agreement does not address it, and give as its\n"
    f'  only quote exactly {SENTINEL}, with source 1. Leave "not_stated" false.\n',
)

EXTRACT_TEMPLATE = """Find the clauses of the agreement below that state what the question asks about, or state the opposite.

Question: {query}

Sources:
{sources}

Rules:
- A clause on a related topic does not count. A clause counts only if it
  settles the question's proposition one way or the other.
- Copy each clause character for character, usually one sentence.
- If the agreement has no such clause, return an empty list.

Return one JSON object and nothing else, in this form:
{{"clauses": ["<exact words>"]}}
"""

CHECK_TEMPLATE = """A reader asked a question about an agreement and got the answer and quotes below.
Decide whether the quotes address the exact proposition in the question, or only a related topic.

Question: {question}

Answer: {answer}

Quotes:
{quotes}

"addressed" is true only if a quote states the proposition or states its
opposite. If the agreement merely has no clause on the proposition, the quotes
cannot address it, even when they are about the same subject.

Return one JSON object and nothing else, in this form:
{{"addressed": true, "reason": "<one sentence, at most 25 words>"}}
"""

STANDALONE_TEMPLATE = """Rewrite the answer as one sentence that makes sense without the question.

Question: {question}

Answer: {statement}

Rules:
- Name what the sentence is about, using the question's words.
- Keep the answer's meaning, including Yes or No. Add nothing.
- Reply with the sentence and nothing else.
"""

#: Each variant arm's prompt text, hashed into its cache key.
TEMPLATES: dict[str, str] = {
    "fulldoc-sentinel": SENTINEL_TEMPLATE,
    "fulldoc-extract": EXTRACT_TEMPLATE + QUOTE_TEMPLATE,
    "fulldoc-check": CHECK_TEMPLATE,
    "answer-standalone": STANDALONE_TEMPLATE,
}

#: Arms the explore pipeline builds, beyond the reference run's seven.
VARIANTS: tuple[str, ...] = tuple(TEMPLATES)
#: Variants that rewrite or re-read the answers of another arm.
BASES: dict[str, str] = {"fulldoc-check": "fulldoc", "answer-standalone": "answer"}


def _abstention(query_id: str, raw: str) -> CitedAnswer:
    return CitedAnswer(query_id, (), "quotes", abstained=True, raw=raw)


def sentinel(
    query_id: str,
    question: str,
    document: Evidence,
    complete: Callable[[str], str],
) -> CitedAnswer:
    """Ask for quotes with a named way to say no clause supports the answer.

    A statement whose only quotes are :data:`SENTINEL` is dropped, and an
    answer left with no statement is an abstention.
    """
    raw = complete(build_prompt(question, [document], SENTINEL_TEMPLATE, None))
    payload = _first_json(raw)
    if payload is None:
        return parse_structured(query_id, raw, [document])
    kept = []
    for item in payload.get("statements") or []:
        quotes = (item.get("quotes") or []) if isinstance(item, dict) else []
        real = [q for q in quotes if not (isinstance(q, dict) and SENTINEL.lower() in str(q.get("quote", "")).lower())]
        if quotes and not real:
            continue
        kept.append({**item, "quotes": real})
    if not kept:
        return _abstention(query_id, raw)
    answer = parse_structured(query_id, json.dumps({**payload, "statements": kept}), [document])
    return replace(answer, raw=raw)


def extract_then_answer(
    query_id: str,
    question: str,
    document: Evidence,
    complete: Callable[[str], str],
) -> CitedAnswer:
    """List the clauses that address the question, then answer from them alone.

    An empty list abstains without a second call. The second call sees only the
    clauses, and its quotes are located in the whole agreement.
    """
    first = complete(build_prompt(question, [document], EXTRACT_TEMPLATE, None))
    payload = _first_json(first) or {}
    clauses = [str(c).strip() for c in payload.get("clauses") or [] if str(c).strip()]
    if not clauses:
        return _abstention(query_id, first)
    provenance = document.provenance
    shown = Evidence(
        text="\n\n".join(clauses),
        provenance=Provenance(doc_id=provenance.doc_id, chunk_id=f"{provenance.doc_id}::clauses"),
        retriever="none",
    )
    second = complete(build_prompt(question, [shown], QUOTE_TEMPLATE, None))
    return replace(parse_structured(query_id, second, [document]), raw=f"{first}\n---\n{second}")


def self_check(
    base: CitedAnswer,
    question: str,
    complete: Callable[[str], str],
) -> CitedAnswer:
    """Re-read an answer's quotes and abstain when they only touch the topic."""
    if base.abstained or not base.statements:
        return base
    quotes = "\n".join(f"- {c.text.strip()}" for c in base.citations) or "- (none)"
    raw = complete(CHECK_TEMPLATE.format(question=question, answer=base.direct, quotes=quotes))
    verdict = (_first_json(raw) or {}).get("addressed", True)
    if verdict is False:
        return CitedAnswer(base.query_id, (), "quotes", abstained=True, raw=f"{base.raw}\n---\n{raw}")
    return base


def standalone(
    base: CitedAnswer,
    question: str,
    complete: Callable[[str], str],
) -> CitedAnswer:
    """Rewrite each statement so it names its subject, keeping its citations."""
    if base.abstained:
        return base
    rewritten: list[Statement] = []
    for statement in base.statements:
        text = complete(STANDALONE_TEMPLATE.format(question=question, statement=statement.text)).strip()
        rewritten.append(Statement(text or statement.text, statement.citations))
    return replace(base, statements=tuple(rewritten))


_SILENCE = re.compile(
    r"\b(does not|doesn't|do not|no (clause|provision|mention|statement)|nothing|silent|neither"
    r"|not (contain|address|mention|include|state|specify|provide|explicitly|define))\b",
    re.I,
)


def lexical_negation(texts: Sequence[str]) -> int:
    """Count texts that say the agreement lacks a clause, by wording alone.

    This is a rough regular-expression count, used only to show how many
    answers marked "No" describe silence. It is not a scorer.
    """
    return sum(bool(_SILENCE.search(t)) for t in texts)


__all__ = [
    "BASES",
    "SENTINEL",
    "TEMPLATES",
    "VARIANTS",
    "extract_then_answer",
    "lexical_negation",
    "self_check",
    "sentinel",
    "standalone",
]
