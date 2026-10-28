"""Grounded generation, with citations and abstention as an ordinary outcome.

Most of RI #1's difficulty is here. CUAD's test split is roughly 70%
unanswerable, so a pipeline that always produces an answer is wrong most of the
time while looking fluent, and the number that catches it is not correctness but
``evaluation.metrics.score_null_detection``. This module is built so that "not
stated" is a first-class result: a :class:`GroundedAnswer` carries ``abstained``
and the reason it was set, and :func:`abstention_flags` hands that straight to
the scorer.

Four paths lead to an abstention, and each one names itself in ``reason``:

1. Nothing was retrieved, so no generation call is made at all.
2. The model set ``not_stated`` in its response.
3. The model ignored the response format and wrote a refusal in prose.
4. The answer cited no usable source, with ``require_citations`` on.

The fourth is the citation gate and the one to turn first. An uncited answer is
not grounded in anything the pipeline can point at, so treating it as an
abstention trades correctness on answerable questions for a lower false-answer
rate on the null ones. ``require_citations`` exposes that trade, and the null
report prices it.

Citations are :class:`~aieng.advanced_rag.provenance.Provenance` records, the
same type every retriever already returns, resolved from the source numbers the
model cites. The model never writes a chunk identifier: asked to reproduce a
long opaque string, it invents plausible ones, and a fabricated citation that
parses is worse than no citation at all.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..graphrag.evidence import Evidence
from ..graphrag.synthesis import build_prompt
from ..metering import CallUsage, read_completion, report_call, report_scope
from ..models import request_params
from ..provenance import Provenance
from .grading import first_json_object


ABSTENTION_TEXT = "Not stated in the provided sources."

GENERATION_TEMPLATE = """Answer the question using only the sources below.

Question: {query}

Sources:
{sources}

Rules:
- Use only what the sources state. Add nothing from outside them.
- If the sources do not answer the question, that is a correct outcome: set
  "not_stated" to true and leave "answer" empty. Do not guess.
- Cite the number of every source you used.

Return one JSON object and nothing else, with exactly these keys:
  "answer": the answer, or "" when not_stated is true.
  "sources": the source numbers used, like [1, 3]. Empty when not_stated is true.
  "not_stated": true when the sources do not answer the question.
"""

# Phrases a model reaches for when it declines. Only consulted when the model
# ignored the response format, so a correct answer that happens to contain
# "not specified" in a subordinate clause is not misread as a refusal.
ABSTENTION_MARKERS: tuple[str, ...] = (
    "not stated",
    "not specified",
    "not mentioned",
    "not provided",
    "not included",
    "does not state",
    "do not state",
    "does not specify",
    "does not mention",
    "does not contain",
    "do not contain",
    "no information",
    "cannot be determined",
    "cannot answer",
    "unable to answer",
    "insufficient information",
)

# Answers written to the default synthesis template carry [1]-style markers, so
# they remain readable as citations even when the JSON contract is ignored.
_MARKER_RE = re.compile(r"\[(\d+)\]")


def looks_like_abstention(text: str) -> bool:
    """Whether an answer reads as a refusal to answer."""
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


@dataclass(frozen=True)
class GroundedAnswer:
    """One answer, the sources it stands on, and whether it abstained.

    Parameters
    ----------
    query_id : str
        Identifier of the question answered.
    text : str
        What a reader is shown. On an abstention this is
        :data:`ABSTENTION_TEXT` rather than the rejected draft, so that a
        rejected answer cannot reach a reader or a judge as if it had been
        accepted. The draft stays in ``raw``.
    abstained : bool, optional
        Whether the pipeline declined to answer.
    citations : tuple of Provenance, optional
        Where the answer is grounded, in the order cited.
    reason : str, optional
        Why it abstained, empty on an answered question.
    dropped_citations : tuple of int, optional
        Source numbers the model cited that no retrieved passage carried. A
        non-empty tuple here is the model citing something it was not given.
    raw : str, optional
        The model's unedited response, kept for auditing.
    """

    query_id: str
    text: str
    abstained: bool = False
    citations: tuple[Provenance, ...] = ()
    reason: str = ""
    dropped_citations: tuple[int, ...] = ()
    raw: str = ""

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the chunk identifiers the answer is grounded in."""
        return tuple(
            citation.chunk_id or citation.doc_id for citation in self.citations
        )

    @classmethod
    def abstain(cls, query_id: str, reason: str, raw: str = "") -> "GroundedAnswer":
        """Build an abstention, naming why."""
        return cls(
            query_id=query_id,
            text=ABSTENTION_TEXT,
            abstained=True,
            reason=reason,
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the answer."""
        return {
            "query_id": self.query_id,
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "citations": [citation.encode() for citation in self.citations],
            "chunk_ids": list(self.chunk_ids),
            "dropped_citations": list(self.dropped_citations),
        }


def _cited_numbers(payload: dict[str, Any] | None, answer_text: str) -> list[int]:
    """Return the source numbers the model cited, from JSON or inline markers."""
    listed = payload.get("sources") if payload else None
    numbers: list[int] = []
    if isinstance(listed, list):
        for value in listed:
            try:
                numbers.append(int(value))
            except (TypeError, ValueError):
                continue
    if numbers:
        return numbers
    return [int(match.group(1)) for match in _MARKER_RE.finditer(answer_text)]


def _resolve_citations(
    numbers: Sequence[int], evidence: Sequence[Evidence]
) -> tuple[list[Provenance], list[int]]:
    """Map cited source numbers onto the passages they were rendered from.

    Sources are numbered from 1 by ``graphrag.synthesis.format_sources``, so the
    mapping is positional. A number outside the range is dropped and returned
    separately rather than silently ignored.
    """
    citations: list[Provenance] = []
    dropped: list[int] = []
    seen: set[int] = set()
    for number in numbers:
        if number in seen:
            continue
        seen.add(number)
        if 1 <= number <= len(evidence):
            citations.append(evidence[number - 1].provenance)
        else:
            dropped.append(number)
    return citations, dropped


def parse_answer(
    query_id: str,
    raw: str,
    evidence: Sequence[Evidence],
    *,
    require_citations: bool = True,
    phrase_fallback: bool = True,
) -> GroundedAnswer:
    """Turn a raw generation response into an answer or an abstention.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    raw : str
        The model's response. An empty string, which is what a truncated
        completion becomes, abstains.
    evidence : sequence of Evidence
        The passages the model was shown, in the order they were numbered.
    require_citations : bool, optional
        Treat an answer with no usable citation as an abstention.
    phrase_fallback : bool, optional
        When the model ignored the response format, read a refusal out of the
        prose. Turn this off to measure the format contract on its own.

    Returns
    -------
    GroundedAnswer
        The parsed answer, with ``abstained`` and ``reason`` set on every path
        that declined.
    """
    text = (raw or "").strip()
    if not text:
        return GroundedAnswer.abstain(
            query_id, reason="the model returned no content", raw=raw or ""
        )

    payload = first_json_object(text)
    answer_text = str(payload.get("answer", "")).strip() if payload else text
    citations, dropped = _resolve_citations(
        _cited_numbers(payload, answer_text), evidence
    )

    if payload is not None and "not_stated" in payload:
        if bool(payload["not_stated"]):
            return GroundedAnswer.abstain(
                query_id,
                reason="the model reported the sources do not answer the question",
                raw=text,
            )
    elif phrase_fallback and looks_like_abstention(answer_text):
        return GroundedAnswer.abstain(
            query_id, reason="the answer reads as a refusal", raw=text
        )

    if not answer_text:
        return GroundedAnswer.abstain(
            query_id, reason="the model returned an empty answer", raw=text
        )
    if require_citations and not citations:
        return GroundedAnswer.abstain(
            query_id, reason="the answer cited no usable source", raw=text
        )
    return GroundedAnswer(
        query_id=query_id,
        text=answer_text,
        citations=tuple(citations),
        dropped_citations=tuple(dropped),
        raw=text,
    )


def generate(
    query_id: str,
    question: str,
    evidence: Sequence[Evidence],
    complete: Callable[[str], str],
    *,
    template: str = GENERATION_TEMPLATE,
    budget: int | None = None,
    max_chars: int | None = 1200,
    require_citations: bool = True,
    phrase_fallback: bool = True,
) -> GroundedAnswer:
    """Answer a question from evidence, or abstain.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    question : str
        The question as asked. On a corrective retry this is the original
        question, not the rewrite that found the evidence: the rewrite is a
        retrieval device and answering it instead answers something the reader
        did not ask.
    evidence : sequence of Evidence
        Retrieved passages in rank order. An empty sequence abstains without a
        model call.
    complete : callable
        Takes a prompt, returns text. See :class:`ChatCompleter`.
    template : str, optional
        Prompt template with ``{query}`` and ``{sources}``.
    budget : int or None, optional
        Passages to place in the prompt. Held fixed across methods when answers
        are being compared, matching ``evaluation.answers.answer_with_budget``.
    max_chars : int or None, optional
        Per-passage truncation.
    require_citations : bool, optional
        Treat an uncited answer as an abstention.
    phrase_fallback : bool, optional
        Read refusals out of prose when the response format is ignored.

    Returns
    -------
    GroundedAnswer
        The answer or the abstention.
    """
    used = list(evidence)[:budget] if budget is not None else list(evidence)
    if not used:
        return GroundedAnswer.abstain(query_id, reason="no evidence to answer from")
    raw = complete(build_prompt(question, used, template, max_chars))
    return parse_answer(
        query_id,
        raw,
        used,
        require_citations=require_citations,
        phrase_fallback=phrase_fallback,
    )


def abstention_flags(answers: Iterable[GroundedAnswer]) -> dict[str, bool]:
    """Return the mapping ``score_null_detection`` scores.

    Parameters
    ----------
    answers : iterable of GroundedAnswer
        Answers from one run.

    Returns
    -------
    dict
        Query identifier to whether the pipeline declined to answer.
    """
    return {answer.query_id: answer.abstained for answer in answers}


class ChatCompleter:
    """A prompt-to-text callable over an OpenAI-compatible client.

    Every generating stage in this package takes a ``complete`` callable rather
    than a client, so this is the one place that knows about the proxy. Two of
    its behaviours are not optional there.

    The request's parameters come from
    :func:`~aieng.advanced_rag.models.request_params`, the per-family adapter,
    so the same completer sends ``max_completion_tokens`` to an OpenAI model,
    leaves out a temperature a Claude model would reject, and turns reasoning
    off with the switch each family honours. The on-prem bulk model is a
    reasoning model billed on completion tokens, and on short structured calls
    the reasoning is almost the whole bill: 1,312 completion tokens against 16
    for the same answer. ``request`` holds the parameters as sent.

    ``content`` is treated as optional. When ``max_tokens`` runs out during
    reasoning the API returns ``content=None`` with a normal finish reason, and
    code that assumes a string raises ``AttributeError`` somewhere unrelated.
    Here it becomes an empty string and increments ``empty_responses``, which
    every parser in this package reads as an abstention.

    What a call cost is kept as well as returned. ``calls``, ``prompt_tokens``,
    and ``completion_tokens`` are running totals, and ``last_usage`` holds the
    most recent response's counts, so a harness that differences ``calls`` around
    a stage can difference tokens the same way. Give each concurrent worker its
    own completer before reading those, because they are plain fields. Every call
    is also reported to :func:`~aieng.advanced_rag.metering.report_call`, once
    even over a client such as a governed one that reports its own calls, which
    is what lets :class:`~aieng.advanced_rag.metering.UsageMeter` attribute it to
    the question being answered, whichever worker answers it, and total it per
    model. ``last_response_model`` is the model name the last response reported.

    Parameters
    ----------
    client : Any
        An OpenAI-compatible client, normally pointed at the proxy.
    model : str
        Model name to send.
    max_tokens : int, optional
        Completion cap.
    temperature : float, optional
        Sampling temperature, left out of the request for families that reject
        it. Zero by default, which removes sampling variance
        and nothing more. It does not make a graded run reproducible: variation
        at zero temperature comes from the batch a request lands in, because
        kernels that are not batch-invariant give different results at
        different batch sizes (`Thinking Machines, 2025-09-10
        <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>`_).
        Compare evidence sets, which do reproduce, rather than answers.
        Anthropic's `Messages API
        <https://platform.claude.com/docs/en/api/messages>`_ reference says
        models released after Claude Opus 4.6 accept only the default, which is
        why the adapter drops it for them.
    disable_reasoning : bool, optional
        Leave this on unless the deployment is not a reasoning model.

    Examples
    --------
    >>> from openai import OpenAI  # doctest: +SKIP
    >>> from aieng.advanced_rag.models import choose_models  # doctest: +SKIP
    >>> MODELS = choose_models()  # doctest: +SKIP
    >>> complete = ChatCompleter(OpenAI(), MODELS.generation.model)  # doctest: +SKIP
    >>> complete("Say hello.")  # doctest: +SKIP
    'Hello.'
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        disable_reasoning: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request = request_params(
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=not disable_reasoning,
        )
        self.last_response_model: str | None = None
        self.calls = 0
        self.empty_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_usage: CallUsage | None = None

    def __call__(self, prompt: str) -> str:
        """Send one prompt and return its text, or ``""`` if there is none.

        Parameters
        ----------
        prompt : str
            The prompt to send.

        Returns
        -------
        str
            The response text, or ``""`` when the response carried none.
        """
        started = time.perf_counter()
        with report_scope() as scope:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.request,
            )
        seconds = time.perf_counter() - started
        self.calls += 1
        facts = read_completion(response)
        self.last_usage = facts.usage
        self.last_response_model = facts.model
        self.prompt_tokens += facts.usage.prompt_tokens
        self.completion_tokens += facts.usage.completion_tokens
        # A governed client has already reported this call, or its cache hit.
        if not scope.reports:
            report_call(
                facts.usage,
                seconds,
                tool_calls=facts.tool_calls,
                empty=facts.empty,
                model=self.model,
                response_model=facts.model,
            )
        if facts.empty:
            self.empty_responses += 1
            return ""
        return facts.text
