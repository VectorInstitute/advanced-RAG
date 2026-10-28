"""Turning evidence into an answer prompt. Optional, and meant to be replaced.

Kept separate from retrieval on purpose. Every GraphRAG library ships its own
answer format with sources and citations arranged its way, and adopting a
library previously meant adopting that too. Here the retriever returns evidence
and stops; what an answer looks like is a use-case decision.

Use this for a quick default, or ignore it and render ``list[Evidence]``
however the use case needs.
"""

from __future__ import annotations

from typing import Callable

from .evidence import Evidence


DEFAULT_TEMPLATE = """Answer the question using only the sources below.
Cite sources by their number, like [1]. If the sources do not contain the
answer, say so plainly rather than guessing.

Question: {query}

Sources:
{sources}

Answer:"""


def format_sources(evidence: list[Evidence], max_chars: int | None = 1200) -> str:
    """Render evidence as a numbered source list.

    Parameters
    ----------
    evidence : list of Evidence
        Retrieved passages, in rank order.
    max_chars : int or None, optional
        Truncate each passage to this many characters. ``None`` keeps them
        whole.

    Returns
    -------
    str
        One block per source, numbered from 1 and labelled with its citation.
    """
    blocks = []
    for number, item in enumerate(evidence, start=1):
        text = item.text if max_chars is None else item.text[:max_chars]
        blocks.append(f"[{number}] {item.cite()}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    evidence: list[Evidence],
    template: str = DEFAULT_TEMPLATE,
    max_chars: int | None = 1200,
) -> str:
    """Render a grounded answer prompt from evidence."""
    return template.format(query=query, sources=format_sources(evidence, max_chars))


def answer(
    query: str,
    evidence: list[Evidence],
    complete: Callable[[str], str],
    template: str = DEFAULT_TEMPLATE,
    max_chars: int | None = 1200,
) -> str:
    """Generate an answer from evidence using a caller-supplied completion function.

    Parameters
    ----------
    query : str
        The question.
    evidence : list of Evidence
        Retrieved passages.
    complete : callable
        Takes a prompt and returns text. Keeping this a parameter is what stops
        this module from depending on any particular client.
    template : str, optional
        Prompt template with ``{query}`` and ``{sources}`` placeholders.
    max_chars : int or None, optional
        Per-passage truncation.

    Returns
    -------
    str
        The generated answer.
    """
    return complete(build_prompt(query, evidence, template, max_chars))
