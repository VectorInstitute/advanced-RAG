"""Query rewriting: changing the question before it reaches the retriever.

Retrieval is scored on the question it is handed, and on this corpus that
question is a poor query. CUAD's 41 questions are templated: each one opens with
"Highlight the parts (if any) of this contract related to" and closes with "that
should be reviewed by a lawyer". Only the clause category varies, so the shared
boilerplate is most of the term mass a lexical retriever sees and most of the
sentence a dense encoder embeds. Removing it costs nothing at run time and moves
a number the harness already reports.

Two rewriters live here because the pipeline rewrites at two different moments:

* :class:`TemplateRewriter` runs before the first retrieval. It is
  deterministic, needs no model, and is the one to reach for when a corpus has a
  known query template.
* :class:`LLMRewriter` runs on the corrective retry, after the grader has said
  why the previous evidence was weak. That reason goes into the prompt, so the
  retry aims at the gap the grader named.

Neither rewriter decides anything. Both return candidate questions and leave the
retrieving to the caller, which keeps this stage cheap to test and cheap to
switch off.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Rewrite:
    """One rewritten form of a question.

    Parameters
    ----------
    question : str
        The rewritten query text.
    strategy : str
        Name of the rewriter that produced it, so a recovered query can be
        attributed to the stage that recovered it.
    reason : str, optional
        Why this rewrite exists, kept for auditing.
    """

    question: str
    strategy: str
    reason: str = ""


@dataclass(frozen=True)
class RewritePlan:
    """The candidate questions produced for one original question.

    Parameters
    ----------
    original : str
        The question as asked.
    rewrites : tuple of Rewrite, optional
        Candidates in the order the rewriter produced them.
    outcome : str, optional
        Why the plan looks the way it does. ``"rewritten"`` when there are
        candidates, ``"no-change"`` when the rewriter found nothing to change,
        ``"emptied"`` when stripping removed the whole question, and
        ``"all-rejected"`` when the model returned candidates and every one was
        dropped. An empty plan has four causes and they are not the same
        finding: a rewriter that never fires on a corpus is a configuration
        answer, a rewriter whose output is always discarded is a prompt answer.
        ``is_empty`` cannot tell them apart, so it stays the question of whether
        there is anything to retry with and this says why.
    """

    original: str
    rewrites: tuple[Rewrite, ...] = ()
    outcome: str = "rewritten"

    @property
    def is_empty(self) -> bool:
        """Whether the rewriter produced nothing to try."""
        return not self.rewrites

    def questions(self, include_original: bool = True) -> tuple[str, ...]:
        """Return the queries to retrieve with, de-duplicated in order.

        Parameters
        ----------
        include_original : bool, optional
            Keep the original question at the head of the list. Leave this on
            unless the rewriter is known to dominate it: a rewrite that drops a
            rare term can retrieve worse than the question as asked.

        Returns
        -------
        tuple of str
            Questions in retrieval order, first appearance winning.
        """
        seen: dict[str, None] = {}
        if include_original:
            seen[self.original] = None
        for rewrite in self.rewrites:
            seen.setdefault(rewrite.question, None)
        return tuple(seen)


class Rewriter(Protocol):
    """Anything that proposes alternative phrasings of a question."""

    name: str

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        """Propose rewrites, optionally in response to a grader's reason."""
        ...


# The CUAD question template, split into the parts that carry no query signal.
# "Details:" is a label rather than content, so it goes too.
CUAD_BOILERPLATE: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"highlight the parts \(if any\) of this (?:contract|agreement) related to",
        re.IGNORECASE,
    ),
    re.compile(r"that should be reviewed by a lawyer\.?", re.IGNORECASE),
    re.compile(r"\bdetails:\s*", re.IGNORECASE),
)

_WHITESPACE_RE = re.compile(r"\s+")


def collapse(text: str) -> str:
    """Normalise whitespace and trim dangling punctuation from a rewrite."""
    return _WHITESPACE_RE.sub(" ", text).strip(" \t\n.,;:")


class TemplateRewriter:
    """Removes known boilerplate from a templated question.

    Deterministic and model-free, so it runs on every query for no marginal
    cost. It applies where a corpus asks the same question shape over and over;
    on a corpus of free-form questions it finds nothing to strip and returns an
    empty plan, which the pipeline reads as "keep the original".

    Parameters
    ----------
    patterns : sequence of re.Pattern, optional
        Expressions to delete. Defaults to the CUAD template.
    name : str, optional
        Strategy label recorded on each rewrite.
    """

    def __init__(
        self,
        patterns: Sequence[re.Pattern[str]] = CUAD_BOILERPLATE,
        name: str = "template",
    ) -> None:
        self.patterns = tuple(patterns)
        self.name = name

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        """Strip the configured patterns out of ``question``.

        Parameters
        ----------
        question : str
            The question as asked.
        feedback : str, optional
            Ignored. A fixed pattern list has no way to act on it, and the
            parameter is here so this satisfies the same protocol as
            :class:`LLMRewriter`.

        Returns
        -------
        RewritePlan
            One rewrite, or an empty plan whose ``outcome`` is ``"emptied"``
            when stripping removed the whole question and ``"no-change"`` when
            it changed nothing.
        """
        stripped = question
        for pattern in self.patterns:
            stripped = pattern.sub(" ", stripped)
        stripped = collapse(stripped)
        if not stripped:
            return RewritePlan(original=question, outcome="emptied")
        if stripped == collapse(question):
            return RewritePlan(original=question, outcome="no-change")
        return RewritePlan(
            original=question,
            rewrites=(
                Rewrite(
                    question=stripped,
                    strategy=self.name,
                    reason="removed corpus-wide boilerplate",
                ),
            ),
        )


REWRITE_PROMPT = """Rewrite a search query for a document retrieval system.

Original question:
{question}
{feedback}
Write {variants} alternative queries that would find the passage answering the
original question. Keep every proper noun, number, and legal term from the
original. Vary the wording, not the meaning.

Return the queries one per line. No numbering, no quotes, no other text.
"""

FEEDBACK_TEMPLATE = """
A previous search with this question returned weak evidence. The grader said:
{feedback}
Look for the missing information a different way.
"""

# Model output arrives as a list far more often than as the bare lines the
# prompt asks for, so numbering and bullets are stripped rather than rejected.
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _parsed_lines(raw: str) -> list[str]:
    """Return the lines of ``raw`` that read as a query, before any are dropped.

    Split out so a rewriter can tell a completion that held no queries from one
    whose queries were all discarded. Both leave no variants, and they point at
    different fixes.

    Parameters
    ----------
    raw : str
        The model's raw response.

    Returns
    -------
    list of str
        Cleaned candidate queries in the order they appeared, including any
        repeat of the original and any duplicate.
    """
    lines: list[str] = []
    for line in (raw or "").splitlines():
        candidate = _BULLET_RE.sub("", line).strip().strip('"').strip()
        # A trailing colon marks a preamble such as "Here are two queries:".
        if not candidate or candidate.endswith(":") or len(candidate) < 3:
            continue
        lines.append(candidate)
    return lines


def parse_variants(raw: str, original: str = "", limit: int = 3) -> list[str]:
    """Pull rewritten queries out of a line-per-query response.

    Parameters
    ----------
    raw : str
        The model's raw response. An empty string yields no variants, which is
        what a truncated or content-free completion produces.
    original : str, optional
        The question asked, dropped from the results when the model repeats it.
    limit : int, optional
        Maximum variants to return.

    Returns
    -------
    list of str
        Cleaned queries, de-duplicated, in the order they appeared.
    """
    baseline = original.strip().casefold()
    variants: dict[str, None] = {}
    for candidate in _parsed_lines(raw):
        if candidate.casefold() == baseline:
            continue
        variants.setdefault(candidate, None)
        if len(variants) >= limit:
            break
    return list(variants)


class LLMRewriter:
    """Asks a model for alternative phrasings, optionally given a grader's reason.

    The feedback path is what makes this a corrective stage. Rewriting blind
    produces paraphrases of a query that already failed; rewriting against
    "the passages describe termination rights, the question asks about
    assignment" produces a query aimed at the gap.

    One call per attempt. That is the cost, and ``variants`` does not change it.

    Parameters
    ----------
    complete : callable
        Takes a prompt, returns text. Supply
        :class:`~aieng.advanced_rag.pipeline.generate.ChatCompleter` or any
        cached, metered equivalent.
    variants : int, optional
        How many rewrites to request and keep.
    prompt : str, optional
        Template with ``{question}``, ``{feedback}``, and ``{variants}``.
    name : str, optional
        Strategy label recorded on each rewrite.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        variants: int = 2,
        prompt: str = REWRITE_PROMPT,
        name: str = "llm",
    ) -> None:
        if variants < 1:
            raise ValueError("variants must be at least 1")
        self.complete = complete
        self.variants = variants
        self.prompt = prompt
        self.name = name

    def rewrite(self, question: str, feedback: str = "") -> RewritePlan:
        """Ask the model for rewrites of ``question``.

        Parameters
        ----------
        question : str
            The question as asked.
        feedback : str, optional
            The grader's reason for rejecting the previous attempt's evidence.

        Returns
        -------
        RewritePlan
            The parsed rewrites, or an empty plan whose ``outcome`` is
            ``"all-rejected"`` when the model returned queries and every one
            was dropped, and ``"no-change"`` when it returned nothing that read
            as a query at all.
        """
        block = FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else ""
        raw = self.complete(
            self.prompt.format(
                question=question, feedback=block, variants=self.variants
            )
        )
        variants = parse_variants(raw, question, self.variants)
        if not variants:
            # A model that echoed the question back is saying it has nothing to
            # add, and a truncated or chatty completion is a prompt fault. The
            # two want different responses, so they are reported apart.
            outcome = "all-rejected" if _parsed_lines(raw) else "no-change"
            return RewritePlan(original=question, outcome=outcome)
        reason = f"rewritten after: {feedback}" if feedback else "paraphrase"
        return RewritePlan(
            original=question,
            rewrites=tuple(
                Rewrite(question=variant, strategy=self.name, reason=reason)
                for variant in variants
            ),
        )
