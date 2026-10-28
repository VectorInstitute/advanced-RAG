"""One usage meter, counting model calls, tokens, and seconds for each question.

Every reference implementation reports what a question cost, and until this
module each one counted something different: the threshold sweep counts model
calls and no tokens or time, the GraphRAG arms count run totals split by a phase
label, and the contextual retriever folds per-thread client counters into a third
record. This is the one meter they share.

The word is usage rather than cost, because nothing here holds a price, a
currency, or a model rate. That matches ``openai.types.CompletionUsage``,
``anthropic.types.Usage``, and ``langchain_core.messages.UsageMetadata``, and it
leaves "cost" free for the money LiteLLM's ``Usage.cost`` carries.

Four things it does that the meters it draws on do not.

Usage lands on a question. A phase label is one mutable field, so the finest
figure the older meters produce is a run total divided by the question count.
Here a question opens a :class:`QuestionMeter`, and every call made inside it is
charged to that question.

The figures hold under concurrency. Every answering run in this project is
eight-way concurrent. The open question's meter travels in a
:class:`~contextvars.ContextVar`, which asyncio copies into each task and a
thread pool leaves empty in a new worker, so a worker either opens its own or is
handed one to bind.

Two seconds are reported, under names that cannot be swapped.
``service_seconds`` is the summed duration of a question's model calls.
``elapsed_seconds`` is the wall clock that question took. One measured run
recorded 3,130.5 service seconds against 613.2 elapsed, a factor of 5.1, because
eight questions were answered at once. Reading either number as the other
overstates a run by about the worker count.

A call that reports no tokens is counted as such. A streamed response carries
usage only when the request sets ``stream_options={"include_usage": True}``, and
says nothing at all when it does not, so ``calls_missing_usage`` counts the
calls whose tokens are unknown and :attr:`QuestionUsage.tokens_complete` says
whether a token figure is whole.

Prompt tokens include the provider's prompt cache, whatever the provider's own
convention. Through the proxy's ``/v1/chat/completions``, an Anthropic model
reports ``cache_creation_input_tokens``, ``cache_creation_1h_input_tokens``, and
``cache_read_input_tokens`` beside a ``prompt_tokens`` and ``total_tokens`` that
leave all three out: a call that read 10,803 cached tokens reported 19 prompt
tokens. An OpenAI model reports ``prompt_tokens_details.cached_tokens`` and
``cache_write_tokens`` inside a ``prompt_tokens`` that already counts them. A
Gemini model reports ``cache_read_input_tokens`` at the top level, as Anthropic
does, but counts it inside ``prompt_tokens``, as OpenAI does: the same 8,595-token
prompt reported 8,595 prompt tokens on a cold call and on a call that read 6,117
cached tokens. Gemini never reports ``cache_creation_input_tokens``, and an
Anthropic model always does, so that field tells the two apart.
:func:`usage_of` adds Anthropic's counts back, so ``prompt_tokens`` means every
input token on every family, and keeps the read and written shares apart for
:mod:`aieng.advanced_rag.spend` to price at the cache rates. These counts are
the provider's prompt cache. ``cache_hits`` and ``cached_prompt_tokens`` are
this package's own response cache, which answers a repeated request without
calling the model.

Usage is also totalled per model. Every report names the model requested and,
when the response says, the model it reported, and :meth:`UsageMeter.by_model`
returns the totals that :mod:`aieng.advanced_rag.spend` prices. A cache hit can
carry the tokens its stored response cost, kept apart from live tokens, so a
replay can say what a cold run would have cost.

Three injection points, because the libraries differ:

* :class:`MeteredCompleter` wraps the ``complete`` callable every generating
  stage in this package already takes.
* :class:`MeteredClient` and :func:`meter_client` wrap an OpenAI-compatible
  client, for a library such as LightRAG or atlas-rag that builds its own
  requests and accepts no callable.
* :func:`register_metered_completion` registers a metered provider with
  Microsoft GraphRAG, which builds its models from its own configuration file.

A tool call is a model call. The proxy answers one with ``content`` of null and
``tool_calls`` populated, and every completer in this package reads null content
as an abstention, so the meter looks at ``tool_calls`` before deciding that a
response was empty.
"""

from __future__ import annotations

import inspect
import threading
import time
import warnings
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


# The provider name a GraphRAG model configuration selects with ``type:``.
METERED_PROVIDER = "metered_litellm"


class MissingUsageWarning(UserWarning):
    """A response reported no tokens, so its cost is unknown rather than zero."""


@dataclass(frozen=True)
class CallUsage:
    """The token counts one response reported.

    Parameters
    ----------
    prompt_tokens : int, optional
        Tokens the request spent.
    completion_tokens : int, optional
        Tokens the response spent, including any reasoning the model billed as
        completion tokens.
    total_tokens : int, optional
        What the response called the total, falling back to the sum of the other
        two when the response omits it. A provider total that disagrees with the
        sum is kept as the provider reported it.
    reported : bool, optional
        Whether the response carried a usage block at all. One that carried none
        leaves every count at zero, which is why a meter counts those calls
        separately instead of adding zero tokens and moving on.
    prompt_cache_read_tokens : int, optional
        Prompt tokens the provider read from its prompt cache, counted inside
        ``prompt_tokens``.
    prompt_cache_write_tokens : int, optional
        Prompt tokens the provider wrote to its prompt cache, at any lifetime,
        counted inside ``prompt_tokens``.
    prompt_cache_write_1h_tokens : int, optional
        The share of ``prompt_cache_write_tokens`` written with Anthropic's
        one-hour lifetime, which bills at a higher rate than the five-minute
        default.

    Examples
    --------
    >>> CallUsage(10, 2).total_tokens
    12
    >>> CallUsage(10, 2, 15).total_tokens
    15
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reported: bool = False
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    prompt_cache_write_1h_tokens: int = 0

    @property
    def uncached_prompt_tokens(self) -> int:
        """Return the prompt tokens billed at the plain input rate."""
        cached = self.prompt_cache_read_tokens + self.prompt_cache_write_tokens
        return max(self.prompt_tokens - cached, 0)

    def __post_init__(self) -> None:
        """Fall back to the sum, so a hand-built record is not short by its total.

        The field stays a field rather than becoming a property because a
        provider total can disagree with the sum, and the provider's figure is
        the one to report. Only an absent total is filled in, which is the
        fallback the parameter documents and what :func:`usage_of` already does
        for a response that omits one.
        """
        if not self.total_tokens:
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.completion_tokens
            )


@dataclass(frozen=True)
class ParsedCompletion:
    """What one chat completion says about its shape and what it spent.

    Parameters
    ----------
    text : str
        The message content, or ``""`` when the response carried none.
    usage : CallUsage
        The token counts.
    tool_calls : int
        How many tool calls the message asked for. A tool call returns null
        content, so this is what separates one from an abstention.
    empty : bool
        Whether the response carried neither content nor a tool call. Every
        parser in this package reads that as an abstention.
    finish_reason : str or None
        The first choice's finish reason, when it has one.
    model : str or None, optional
        The model the response says produced it. A proxy can resolve an alias to
        a dated snapshot, so this can differ from the name requested.
    """

    text: str
    usage: CallUsage
    tool_calls: int
    empty: bool
    finish_reason: str | None
    model: str | None = None


def _attr(source: Any, name: str) -> Any:
    """Read one field off an object or a mapping, or ``None`` when it is absent."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _count(source: Any, name: str) -> int | None:
    """Read one token count, or ``None`` when it is absent or not a number."""
    value = _attr(source, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


#: The cache counts an Anthropic model reports at the top level of ``usage``,
#: each left out of ``prompt_tokens``. The one-hour write is its own field, not a
#: share of ``cache_creation_input_tokens``: a one-hour write reported 0 there and
#: 17,511 in ``cache_creation_1h_input_tokens``.
_ANTHROPIC_CACHE_FIELDS = (
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
)


def usage_of(response: Any) -> CallUsage:
    """Read the token counts off one response or streamed chunk.

    ``prompt_tokens`` comes back counting every input token, prompt-cache reads
    and writes included, on every family; the module docstring gives the two
    conventions this reconciles. The proxy returns ``prompt_tokens_details`` as
    null for the on-prem and Anthropic models, which then report no OpenAI-style
    breakdown.

    Parameters
    ----------
    response : Any
        Anything carrying a ``usage`` field, as an attribute or a mapping key.

    Returns
    -------
    CallUsage
        The counts, with ``reported`` false when the response carried no usage.

    Examples
    --------
    >>> usage = usage_of(
    ...     {
    ...         "usage": {
    ...             "prompt_tokens": 19,
    ...             "completion_tokens": 5,
    ...             "total_tokens": 24,
    ...             "cache_creation_input_tokens": 0,
    ...             "cache_read_input_tokens": 10803,
    ...         }
    ...     }
    ... )
    >>> usage.prompt_tokens, usage.prompt_cache_read_tokens, usage.total_tokens
    (10822, 10803, 10827)
    """
    usage = _attr(response, "usage")
    if usage is None:
        return CallUsage()
    prompt = _count(usage, "prompt_tokens")
    completion = _count(usage, "completion_tokens")
    total = _count(usage, "total_tokens")
    if prompt is None and completion is None and total is None:
        return CallUsage()
    prompt = prompt or 0
    completion = completion or 0
    total = prompt + completion if total is None else total
    anthropic = [_count(usage, name) for name in _ANTHROPIC_CACHE_FIELDS]
    if _count(usage, "cache_creation_input_tokens") is not None:
        read, write_5m, write_1h = (value or 0 for value in anthropic)
        added = read + write_5m + write_1h
        return CallUsage(
            prompt_tokens=prompt + added,
            completion_tokens=completion,
            total_tokens=total + added,
            reported=True,
            prompt_cache_read_tokens=read,
            prompt_cache_write_tokens=write_5m + write_1h,
            prompt_cache_write_1h_tokens=write_1h,
        )
    details = _attr(usage, "prompt_tokens_details")
    # Gemini puts its read at the top level, OpenAI inside the details.
    read = _count(details, "cached_tokens") or anthropic[0] or 0
    write = _count(details, "cache_write_tokens") or 0
    # Counted inside prompt_tokens, so a share can never exceed it.
    read = min(read, prompt)
    write = min(write, prompt - read)
    return CallUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reported=True,
        prompt_cache_read_tokens=read,
        prompt_cache_write_tokens=write,
    )


def read_completion(response: Any) -> ParsedCompletion:
    """Read one chat completion's text, tokens, and shape.

    Parameters
    ----------
    response : Any
        A chat completion that was not streamed.

    Returns
    -------
    ParsedCompletion
        What the response carried, with ``empty`` set only when it carried
        neither content nor a tool call.
    """
    choices: Any = _attr(response, "choices") or ()
    choice = choices[0] if choices else None
    message = _attr(choice, "message")
    content = _attr(message, "content")
    tool_calls = len(_attr(message, "tool_calls") or ())
    reason = _attr(choice, "finish_reason")
    model = _attr(response, "model")
    return ParsedCompletion(
        text="" if content is None else str(content),
        usage=usage_of(response),
        tool_calls=tool_calls,
        empty=content is None and tool_calls == 0,
        finish_reason=None if reason is None else str(reason),
        model=str(model) if model else None,
    )


@dataclass
class _Totals:
    """Accumulators for one question, for a run, or for the unattributed calls.

    The meter's lock guards every field, so nothing here takes one of its own.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    cache_hits: int = 0
    empty_responses: int = 0
    calls_missing_usage: int = 0
    service_seconds: float = 0.0
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    prompt_cache_write_1h_tokens: int = 0

    def add(
        self, usage: CallUsage, seconds: float, tool_calls: int, empty: bool
    ) -> None:
        """Add one answered model call."""
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.prompt_cache_read_tokens += usage.prompt_cache_read_tokens
        self.prompt_cache_write_tokens += usage.prompt_cache_write_tokens
        self.prompt_cache_write_1h_tokens += usage.prompt_cache_write_1h_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.tool_calls += tool_calls
        self.service_seconds += seconds
        if empty:
            self.empty_responses += 1
        if not usage.reported:
            self.calls_missing_usage += 1

    def as_dict(self) -> dict[str, Any]:
        """Return the counters, with seconds rounded for a written record."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "cache_hits": self.cache_hits,
            "empty_responses": self.empty_responses,
            "calls_missing_usage": self.calls_missing_usage,
            "service_seconds": round(self.service_seconds, 3),
            "prompt_cache_read_tokens": self.prompt_cache_read_tokens,
            "prompt_cache_write_tokens": self.prompt_cache_write_tokens,
            "prompt_cache_write_1h_tokens": self.prompt_cache_write_1h_tokens,
        }


#: The key per-model totals use for a call that named no model.
UNKNOWN_MODEL = "unknown"


@dataclass
class _ModelTotals(_Totals):
    """One model's accumulators, with what its cache hits would have cost."""

    cached_prompt_tokens: int = 0
    cached_completion_tokens: int = 0
    response_models: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ModelUsage:
    """What one model spent over a run, and what its cache hits saved.

    Parameters
    ----------
    model : str
        The model name as requested, which is the name a price table lists.
        :data:`UNKNOWN_MODEL` for calls that named none.
    calls : int
        Model calls that returned a response.
    prompt_tokens : int
        Prompt tokens over those calls.
    completion_tokens : int
        Completion tokens over those calls.
    total_tokens : int
        Total tokens as the responses reported them.
    cache_hits : int
        Calls a cache answered for this model.
    cached_prompt_tokens : int
        Prompt tokens the cached responses cost when they were made, so a replay
        can say what it would have cost. Zero for a cache that stores no tokens.
    cached_completion_tokens : int
        Completion tokens the cached responses cost when they were made.
    calls_missing_usage : int
        Calls that reported no tokens.
    service_seconds : float
        Summed duration of the calls.
    response_models : tuple of str
        The model names the responses reported, which differ from ``model``
        when the proxy resolved an alias.
    prompt_cache_read_tokens : int, optional
        Prompt tokens the provider read from its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_tokens : int, optional
        Prompt tokens the provider wrote to its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_1h_tokens : int, optional
        The share of the writes made with a one-hour lifetime.
    """

    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hits: int
    cached_prompt_tokens: int
    cached_completion_tokens: int
    calls_missing_usage: int
    service_seconds: float
    response_models: tuple[str, ...] = ()
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    prompt_cache_write_1h_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_completion_tokens": self.cached_completion_tokens,
            "calls_missing_usage": self.calls_missing_usage,
            "service_seconds": round(self.service_seconds, 3),
            "response_models": list(self.response_models),
            "prompt_cache_read_tokens": self.prompt_cache_read_tokens,
            "prompt_cache_write_tokens": self.prompt_cache_write_tokens,
            "prompt_cache_write_1h_tokens": self.prompt_cache_write_1h_tokens,
        }


def _model_usage(model: str, totals: _ModelTotals) -> ModelUsage:
    """Build one model's record. The meter's lock is held."""
    return ModelUsage(
        model=model,
        calls=totals.calls,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        total_tokens=totals.total_tokens,
        cache_hits=totals.cache_hits,
        cached_prompt_tokens=totals.cached_prompt_tokens,
        cached_completion_tokens=totals.cached_completion_tokens,
        calls_missing_usage=totals.calls_missing_usage,
        service_seconds=totals.service_seconds,
        response_models=tuple(sorted(totals.response_models)),
        prompt_cache_read_tokens=totals.prompt_cache_read_tokens,
        prompt_cache_write_tokens=totals.prompt_cache_write_tokens,
        prompt_cache_write_1h_tokens=totals.prompt_cache_write_1h_tokens,
    )


@dataclass(frozen=True)
class QuestionUsage:
    """What one question spent.

    The two seconds answer different questions and are named so they cannot be
    swapped. ``service_seconds`` is what the model service spent on this
    question, which is the figure to compare against a token bill.
    ``elapsed_seconds`` is how long the question took, which is the figure a
    reader experiences. A question that fans out spends more service seconds than
    it takes; a run that answers eight questions at once spends far more service
    seconds in total than its own wall clock.

    Parameters
    ----------
    question_id : str
        Identifier of the question these figures belong to.
    calls : int
        Model calls that returned a response.
    prompt_tokens : int
        Prompt tokens over those calls.
    completion_tokens : int
        Completion tokens over those calls.
    total_tokens : int
        Total tokens as the responses reported them.
    tool_calls : int
        Tool calls the model asked for, counted inside ``calls``.
    cache_hits : int
        Calls a cache answered, counted outside ``calls`` so that a re-run keeps
        reporting what a cold run cost.
    empty_responses : int
        Responses that carried neither content nor a tool call.
    calls_missing_usage : int
        Calls that reported no tokens, which makes every token figure here a
        lower bound.
    service_seconds : float
        Summed duration of the calls.
    elapsed_seconds : float
        Wall clock of the question, summed over the spans it was measured for.
    prompt_cache_read_tokens : int, optional
        Prompt tokens the provider read from its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_tokens : int, optional
        Prompt tokens the provider wrote to its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_1h_tokens : int, optional
        The share of the writes made with a one-hour lifetime.
    """

    question_id: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: int
    cache_hits: int
    empty_responses: int
    calls_missing_usage: int
    service_seconds: float
    elapsed_seconds: float
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    prompt_cache_write_1h_tokens: int = 0

    @property
    def tokens_complete(self) -> bool:
        """Whether every counted call reported its tokens.

        Returns
        -------
        bool
            False when at least one call reported none, which makes the token
            counts a lower bound rather than the cost.
        """
        return self.calls_missing_usage == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of what this question spent.

        Returns
        -------
        dict
            Every counter, with both seconds under their own names and rounded
            to milliseconds.
        """
        return {
            "question_id": self.question_id,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "cache_hits": self.cache_hits,
            "empty_responses": self.empty_responses,
            "calls_missing_usage": self.calls_missing_usage,
            "service_seconds": round(self.service_seconds, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "prompt_cache_read_tokens": self.prompt_cache_read_tokens,
            "prompt_cache_write_tokens": self.prompt_cache_write_tokens,
            "prompt_cache_write_1h_tokens": self.prompt_cache_write_1h_tokens,
        }


@dataclass(frozen=True)
class RunUsage:
    """What a whole run spent, over its questions and everything unattributed.

    This record carries no elapsed figure. Elapsed seconds are measured per
    question, and adding them up across questions answered at once counts the
    same wall clock several times, so a run's wall clock is measured around the
    run itself.

    Parameters
    ----------
    questions : int
        Questions the meter has opened.
    calls : int
        Model calls that returned a response.
    prompt_tokens : int
        Prompt tokens over those calls.
    completion_tokens : int
        Completion tokens over those calls.
    total_tokens : int
        Total tokens as the responses reported them.
    tool_calls : int
        Tool calls the model asked for, counted inside ``calls``.
    cache_hits : int
        Calls a cache answered, counted outside ``calls``.
    empty_responses : int
        Responses that carried neither content nor a tool call.
    calls_missing_usage : int
        Calls that reported no tokens.
    service_seconds : float
        Summed duration of the calls, across every worker.
    prompt_cache_read_tokens : int, optional
        Prompt tokens the provider read from its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_tokens : int, optional
        Prompt tokens the provider wrote to its prompt cache, inside
        ``prompt_tokens``.
    prompt_cache_write_1h_tokens : int, optional
        The share of the writes made with a one-hour lifetime.
    """

    questions: int
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: int
    cache_hits: int
    empty_responses: int
    calls_missing_usage: int
    service_seconds: float
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    prompt_cache_write_1h_tokens: int = 0

    @property
    def tokens_complete(self) -> bool:
        """Whether every counted call reported its tokens.

        Returns
        -------
        bool
            False when at least one call reported none.
        """
        return self.calls_missing_usage == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of what the run spent.

        Returns
        -------
        dict
            Every counter, with service seconds rounded to milliseconds.
        """
        return {"questions": self.questions, **_shared_fields(self)}


def _shared_fields(usage: RunUsage) -> dict[str, Any]:
    """Return the counters a run record shares with a question record."""
    return {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "tool_calls": usage.tool_calls,
        "cache_hits": usage.cache_hits,
        "empty_responses": usage.empty_responses,
        "calls_missing_usage": usage.calls_missing_usage,
        "service_seconds": round(usage.service_seconds, 3),
        "prompt_cache_read_tokens": usage.prompt_cache_read_tokens,
        "prompt_cache_write_tokens": usage.prompt_cache_write_tokens,
        "prompt_cache_write_1h_tokens": usage.prompt_cache_write_1h_tokens,
    }


@dataclass(frozen=True)
class _ReportedCall:
    """One finished model call, handed over by whatever made it."""

    usage: CallUsage
    seconds: float
    tool_calls: int = 0
    empty: bool = False
    cached: bool = False
    model: str | None = None
    response_model: str | None = None


@dataclass
class ReportScope:
    """Counts the reports made inside one block. See :func:`report_scope`."""

    reports: int = 0


_REPORT_SCOPE: ContextVar[ReportScope | None] = ContextVar(
    "aieng_advanced_rag_report_scope", default=None
)


@contextmanager
def report_scope() -> Iterator[ReportScope]:
    """Count the call and cache-hit reports made inside the block.

    A completer that reports its own calls wraps the client call in this, and
    skips its own report when something inside already made one. That is how a
    :class:`~aieng.advanced_rag.pipeline.generate.ChatCompleter` over a client
    governed by :func:`aieng.advanced_rag.graphrag.proxy.govern`, which reports
    every call and every cache hit itself, counts each call once.

    Yields
    ------
    ReportScope
        Its ``reports`` field counts the reports made in the block.
    """
    scope = ReportScope()
    token = _REPORT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _REPORT_SCOPE.reset(token)


def _count_report() -> None:
    scope = _REPORT_SCOPE.get()
    if scope is not None:
        scope.reports += 1


_ACTIVE_QUESTION: ContextVar[QuestionMeter | None] = ContextVar(
    "aieng_advanced_rag_question", default=None
)
_PENDING_CALLS: ContextVar[list[_ReportedCall] | None] = ContextVar(
    "aieng_advanced_rag_pending_calls", default=None
)


def current_question() -> QuestionMeter | None:
    """Return the question meter bound to the current context.

    Returns
    -------
    QuestionMeter or None
        The question bound by :meth:`UsageMeter.question` or
        :meth:`QuestionMeter.bind` in this thread or task, or ``None`` when
        nothing is metering here. A new thread starts with an empty context, so
        this is ``None`` inside a pool worker until the worker binds a question.
    """
    return _ACTIVE_QUESTION.get()


def report_call(
    usage: CallUsage,
    seconds: float,
    *,
    tool_calls: int = 0,
    empty: bool = False,
    model: str | None = None,
    response_model: str | None = None,
) -> None:
    """Hand one finished model call to whatever is metering this context.

    A completer deep inside a pipeline does not know which question it is
    answering, and nothing obliges it to hold a meter, so it reports here
    instead. A :class:`MeteredCompleter` around the call takes the report, and
    failing that the question bound to the current context records it. Recording
    nothing is a normal outcome: with no meter in the call path this returns
    without doing anything, which keeps the reporting side free for callers who
    never meter. Those dropped reports are counted, so
    :func:`unmetered_calls` separates a run that cost nothing from a run nothing
    was measuring.

    Parameters
    ----------
    usage : CallUsage
        Tokens the response reported.
    seconds : float
        How long the call took. This is one call's service time, which a meter
        adds into ``service_seconds``.
    tool_calls : int, optional
        Tool calls the response asked for. A tool call is a model call, so
        passing this is what stops null content being read as an abstention.
    empty : bool, optional
        Whether the response carried neither content nor a tool call.
    model : str or None, optional
        The model requested, which is what per-model totals and prices key on.
    response_model : str or None, optional
        The model the response reported. Recorded so an alias that moved to a
        new snapshot shows up beside the name requested.
    """
    if model and response_model:
        _observe_response_model(model, response_model)
    _count_report()
    try:
        _record_reported(usage, seconds, tool_calls, empty, model, response_model)
    finally:
        _run_call_hooks(model, usage)


def _record_reported(
    usage: CallUsage,
    seconds: float,
    tool_calls: int,
    empty: bool,
    model: str | None,
    response_model: str | None,
) -> None:
    """Hand one reported call to the wrapper or question metering this context."""
    pending = _PENDING_CALLS.get()
    if pending is not None:
        pending.append(
            _ReportedCall(
                usage,
                seconds,
                tool_calls,
                empty,
                model=model,
                response_model=response_model,
            )
        )
        return
    question = _ACTIVE_QUESTION.get()
    if question is not None:
        question.record(
            usage,
            seconds,
            tool_calls=tool_calls,
            empty=empty,
            model=model,
            response_model=response_model,
        )
        return
    _count_unmetered()


#: A function called after every live model call with the model requested and
#: the tokens reported. The spend guard is one: it raises
#: ``spend.SpendLimitReached`` once a run passes its ceiling.
CallHook = Callable[[str | None, CallUsage], None]


@dataclass
class _Hooks:
    """The call hooks installed in this process."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    hooks: list[CallHook] = field(default_factory=list)


_HOOKS = _Hooks()


def add_call_hook(hook: CallHook) -> None:
    """Call ``hook`` after every live model call this package reports.

    The hook sees every call that reaches :func:`report_call` or a metered
    client, whether or not a meter is collecting. An exception it raises
    propagates to the caller of the model call, after the call is recorded,
    which is how a spending ceiling stops a run.

    Parameters
    ----------
    hook : callable
        Takes the model requested (or None) and the call's :class:`CallUsage`.
    """
    with _HOOKS.lock:
        if hook not in _HOOKS.hooks:
            _HOOKS.hooks.append(hook)


def remove_call_hook(hook: CallHook) -> None:
    """Stop calling a hook installed with :func:`add_call_hook`."""
    with _HOOKS.lock:
        if hook in _HOOKS.hooks:
            _HOOKS.hooks.remove(hook)


def _run_call_hooks(model: str | None, usage: CallUsage) -> None:
    with _HOOKS.lock:
        hooks = list(_HOOKS.hooks)
    for hook in hooks:
        hook(model, usage)


class RunStopped(RuntimeError):  # noqa: N818 - names what happened, as the guards do
    """A spend guard stopped the run before a model call was sent.

    The base of :class:`aieng.advanced_rag.spend.SpendLimitReached` and
    :class:`aieng.advanced_rag.runs.ConfirmationRequiredError`. Code that
    catches each call's errors so one failure does not lose the rest re-raises
    this one and sends nothing more, because every later call would be refused
    the same way.
    """


#: A function called before every live model call with the model requested. The
#: spend guard is one: once a run is past its ceiling it raises
#: ``spend.SpendLimitReached`` before the request leaves, so calls already queued
#: on a thread pool are refused instead of sent and billed.
PreCallHook = Callable[[str | None], None]


@dataclass
class _PreCallHooks:
    """The pre-call hooks installed in this process."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    hooks: list[PreCallHook] = field(default_factory=list)


_PRE_HOOKS = _PreCallHooks()


def add_pre_call_hook(hook: PreCallHook) -> None:
    """Call ``hook`` before every live model call a metered or governed client makes.

    An exception the hook raises propagates to the caller and the request is not
    sent. A cache hit is not a live call and does not reach the hook.

    Parameters
    ----------
    hook : callable
        Takes the model requested, or None.
    """
    with _PRE_HOOKS.lock:
        if hook not in _PRE_HOOKS.hooks:
            _PRE_HOOKS.hooks.append(hook)


def remove_pre_call_hook(hook: PreCallHook) -> None:
    """Stop calling a hook installed with :func:`add_pre_call_hook`."""
    with _PRE_HOOKS.lock:
        if hook in _PRE_HOOKS.hooks:
            _PRE_HOOKS.hooks.remove(hook)


def before_call(model: str | None) -> None:
    """Run the pre-call hooks for one live call about to be sent.

    A client that sends its own requests calls this after its cache misses and
    before the request, so a spending ceiling can stop the call.

    Parameters
    ----------
    model : str or None
        The model about to be asked.

    Raises
    ------
    Exception
        Whatever a hook raises, such as ``spend.SpendLimitReached``.
    """
    with _PRE_HOOKS.lock:
        hooks = list(_PRE_HOOKS.hooks)
    for hook in hooks:
        hook(model)


@dataclass
class _Observed:
    """The last response model seen for each requested name, for the process."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    names: dict[str, str] = field(default_factory=dict)


_OBSERVED = _Observed()


def _observe_response_model(model: str, response_model: str) -> None:
    with _OBSERVED.lock:
        _OBSERVED.names[model] = response_model


def observed_response_model(model: str) -> str | None:
    """Return the model name the last live response for ``model`` reported.

    A proxy can resolve an alias such as ``claude-haiku-5-5`` to a dated
    snapshot and later move it. A cached result stores the name its response
    reported, and comparing that with this value is how a reader notices the
    move.

    Parameters
    ----------
    model : str
        The model name as requested.

    Returns
    -------
    str or None
        The reported name, or None when no live call for ``model`` has been
        reported in this process.
    """
    with _OBSERVED.lock:
        return _OBSERVED.names.get(model)


@dataclass
class _Dropped:
    """Reports that found no meter, counted for the process.

    A record rather than a module counter, so counting one needs no ``global``.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    calls: int = 0


_DROPPED = _Dropped()


def _count_unmetered() -> None:
    """Count one report that found nothing to record it."""
    with _DROPPED.lock:
        _DROPPED.calls += 1


def unmetered_calls() -> int:
    """Return how many reported calls found no meter in their call path.

    A run that never opened a question reports ``RunUsage.calls == 0``, which
    reads as free rather than as unmeasured.
    :attr:`RunUsage.tokens_complete` makes that distinction for tokens and
    nothing made it for the calls themselves, so this counts the reports that
    were dropped. It is process-wide, counted across every meter, and a run that
    wants a figure of its own zeroes it with :func:`reset_unmetered_calls`
    first.

    Returns
    -------
    int
        Calls reported through :func:`report_call` with no wrapper collecting
        and no question bound. Anything above zero beside a run of zero calls
        means the run was not metered rather than free.
    """
    with _DROPPED.lock:
        return _DROPPED.calls


def reset_unmetered_calls() -> int:
    """Zero the dropped-report count and return what it was.

    Returns
    -------
    int
        The count before it was zeroed.
    """
    with _DROPPED.lock:
        was = _DROPPED.calls
        _DROPPED.calls = 0
    return was


def report_cache_hit(
    *, model: str | None = None, usage: CallUsage | None = None
) -> None:
    """Hand one cache hit to whatever is metering this context.

    A hit costs nothing, so it is counted apart from the calls. That keeps a
    re-run comparable with the cold run it repeats instead of reporting a
    pipeline that suddenly became free.

    Parameters
    ----------
    model : str or None, optional
        The model the cached response came from, when the cache stores it.
    usage : CallUsage or None, optional
        The tokens the cached response cost when it was made, when the cache
        stores them. Per-model totals keep these apart from live tokens, so a
        replay can say what it would have cost.
    """
    _count_report()
    stored = usage or CallUsage()
    pending = _PENDING_CALLS.get()
    if pending is not None:
        pending.append(_ReportedCall(stored, 0.0, cached=True, model=model))
        return
    question = _ACTIVE_QUESTION.get()
    if question is not None:
        question.record_cache_hit(model=model, usage=stored)


class QuestionMeter:
    """The running cost of one question, wherever that question's calls run.

    Open one with :meth:`UsageMeter.question` when the question is answered in the
    thread or task that opens it, which covers the asyncio arms and any worker
    that takes a whole question. Take one from :meth:`UsageMeter.for_question`
    and hand it over when the calls happen somewhere else: a new thread starts
    with an empty context, so a meter opened in the parent thread is invisible
    inside a ``ThreadPoolExecutor`` worker until the worker binds it or passes it
    to an injection point.

    Parameters
    ----------
    meter : UsageMeter
        The meter these calls also count towards.
    question_id : str
        Identifier for the question, used as the key in
        :meth:`UsageMeter.per_question`.
    """

    def __init__(self, meter: UsageMeter, question_id: str) -> None:
        self.question_id = question_id
        self._meter = meter
        self._totals = _Totals()
        self._elapsed = 0.0

    @property
    def meter(self) -> UsageMeter:
        """Return the meter this question reports to.

        Returns
        -------
        UsageMeter
            The meter that created the question.
        """
        return self._meter

    def record(
        self,
        usage: CallUsage,
        seconds: float,
        *,
        tool_calls: int = 0,
        empty: bool = False,
        model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        """Add one answered model call to this question and to the run.

        Parameters
        ----------
        usage : CallUsage
            Tokens the response reported.
        seconds : float
            How long the call took, added to ``service_seconds``.
        tool_calls : int, optional
            Tool calls the response asked for.
        empty : bool, optional
            Whether the response carried neither content nor a tool call.
        model : str or None, optional
            The model requested, for the run's per-model totals.
        response_model : str or None, optional
            The model the response reported.
        """
        self._meter._add_call(
            self,
            usage,
            seconds,
            tool_calls=tool_calls,
            empty=empty,
            model=model,
            response_model=response_model,
        )

    def record_cache_hit(
        self, *, model: str | None = None, usage: CallUsage | None = None
    ) -> None:
        """Add one call a cache answered, so a cold run's cost stays visible."""
        self._meter._add_cache_hit(self, model=model, usage=usage)

    @contextmanager
    def bind(self) -> Iterator[QuestionMeter]:
        """Bind this question to the current context for the block.

        Use this in a worker that was handed a meter and answers the whole
        question, so anything it calls meters against that question without being
        passed the meter itself.

        Yields
        ------
        QuestionMeter
            This question.
        """
        token = _ACTIVE_QUESTION.set(self)
        try:
            yield self
        finally:
            _ACTIVE_QUESTION.reset(token)

    @contextmanager
    def measure(self) -> Iterator[QuestionMeter]:
        """Bind this question and time the wall clock of the block.

        Yields
        ------
        QuestionMeter
            This question.

        Raises
        ------
        RuntimeError
            When the question is already being measured somewhere else. Two open
            measurements would count the same wall clock twice, and a question
            measured from two threads at once is a wiring mistake rather than a
            figure worth reporting.
        """
        self._meter._begin_measure(self)
        started = time.perf_counter()
        token = _ACTIVE_QUESTION.set(self)
        try:
            yield self
        finally:
            _ACTIVE_QUESTION.reset(token)
            self._meter._end_measure(self, time.perf_counter() - started)

    def totals(self) -> QuestionUsage:
        """Return what this question has cost so far.

        Returns
        -------
        QuestionUsage
            A snapshot, safe to read while other workers are still recording.
        """
        return self._meter._snapshot(self)


class UsageMeter:
    """Model calls, tokens, and seconds, counted for each question and the run.

    One meter is shared by every worker. A question's figures live on the meter
    that question opened rather than on a field the next worker overwrites, which
    is what makes a per-question figure survive eight-way concurrency.

    Calls made with no question bound land in an unattributed bucket, which is
    where a library's own background thread and any indexing work end up. They
    still count towards the run.

    Examples
    --------
    >>> meter = UsageMeter()
    >>> with meter.question("q1") as question:
    ...     question.record(CallUsage(10, 2, 12, reported=True), 0.5)
    >>> meter.per_question()["q1"].total_tokens
    12
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run = _Totals()
        self._unattributed = _Totals()
        self._questions: dict[str, QuestionMeter] = {}
        self._measuring: set[str] = set()
        self._by_model: dict[str, _ModelTotals] = {}

    def for_question(self, question_id: str) -> QuestionMeter:
        """Return the meter for one question, creating it on first use.

        Use this when the question is answered in a different thread from the one
        that opens it: hand the meter to the worker, which binds it with
        :meth:`QuestionMeter.measure` or passes it to an injection point.

        Parameters
        ----------
        question_id : str
            Identifier for the question.

        Returns
        -------
        QuestionMeter
            The meter for that question, the same object on every call.
        """
        with self._lock:
            question = self._questions.get(question_id)
            if question is None:
                question = QuestionMeter(self, question_id)
                self._questions[question_id] = question
            return question

    #: Deprecated alias for :meth:`for_question`, kept for callers of the old
    #: name. Deprecated in 0.1.1; use ``for_question``, because a scope on this
    #: package's retrieval side is a corpus filter.
    scope = for_question

    @contextmanager
    def question(self, question_id: str) -> Iterator[QuestionMeter]:
        """Open one question, timing it and binding it to the current context.

        Parameters
        ----------
        question_id : str
            Identifier for the question.

        Yields
        ------
        QuestionMeter
            The question every call inside the block counts towards.
        """
        with self.for_question(question_id).measure() as question:
            yield question

    def record(
        self,
        usage: CallUsage,
        seconds: float,
        *,
        tool_calls: int = 0,
        empty: bool = False,
        model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        """Record one answered model call against the question bound here.

        Parameters
        ----------
        usage : CallUsage
            Tokens the response reported.
        seconds : float
            How long the call took, added to ``service_seconds``.
        tool_calls : int, optional
            Tool calls the response asked for.
        empty : bool, optional
            Whether the response carried neither content nor a tool call.
        model : str or None, optional
            The model requested, for the per-model totals.
        response_model : str or None, optional
            The model the response reported.
        """
        self._add_call(
            self._active_question(),
            usage,
            seconds,
            tool_calls=tool_calls,
            empty=empty,
            model=model,
            response_model=response_model,
        )

    def record_cache_hit(
        self, *, model: str | None = None, usage: CallUsage | None = None
    ) -> None:
        """Record one call a cache answered, against the question bound here."""
        self._add_cache_hit(self._active_question(), model=model, usage=usage)

    def _add_call(
        self,
        question: QuestionMeter | None,
        usage: CallUsage,
        seconds: float,
        *,
        tool_calls: int = 0,
        empty: bool = False,
        model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        """Record one answered model call against a question you already hold.

        Parameters
        ----------
        question : QuestionMeter or None
            The question to charge, or ``None`` for the unattributed bucket.
        usage : CallUsage
            Tokens the response reported.
        seconds : float
            How long the call took, added to ``service_seconds``.
        tool_calls : int, optional
            Tool calls the response asked for.
        empty : bool, optional
            Whether the response carried neither content nor a tool call.
        model : str or None, optional
            The model requested.
        response_model : str or None, optional
            The model the response reported.
        """
        with self._lock:
            self._run.add(usage, seconds, tool_calls, empty)
            self._target(question).add(usage, seconds, tool_calls, empty)
            per_model = self._model_totals(model or response_model)
            per_model.add(usage, seconds, tool_calls, empty)
            if response_model:
                per_model.response_models.add(response_model)

    def _add_cache_hit(
        self,
        question: QuestionMeter | None,
        *,
        model: str | None = None,
        usage: CallUsage | None = None,
    ) -> None:
        """Record one cache hit against a question you already hold.

        Parameters
        ----------
        question : QuestionMeter or None
            The question to charge, or ``None`` for the unattributed bucket.
        model : str or None, optional
            The model the cached response came from.
        usage : CallUsage or None, optional
            What the cached response cost when it was made.
        """
        with self._lock:
            self._run.cache_hits += 1
            self._target(question).cache_hits += 1
            per_model = self._model_totals(model)
            per_model.cache_hits += 1
            if usage is not None:
                per_model.cached_prompt_tokens += usage.prompt_tokens
                per_model.cached_completion_tokens += usage.completion_tokens

    def _model_totals(self, model: str | None) -> _ModelTotals:
        """Return one model's accumulator, creating it. The lock is held."""
        name = model or UNKNOWN_MODEL
        totals = self._by_model.get(name)
        if totals is None:
            totals = _ModelTotals()
            self._by_model[name] = totals
        return totals

    def by_model(self) -> dict[str, ModelUsage]:
        """Return what each model spent, keyed by the model name requested.

        Calls that named no model, such as those from a plain ``complete``
        callable that reports nothing, are under :data:`UNKNOWN_MODEL`.

        Returns
        -------
        dict
            One :class:`ModelUsage` per model, in first-use order.
        """
        with self._lock:
            return {
                name: _model_usage(name, totals)
                for name, totals in self._by_model.items()
            }

    def _begin_measure(self, question: QuestionMeter) -> None:
        """Take the elapsed-time measurement for one question.

        Parameters
        ----------
        question : QuestionMeter
            The question about to be measured.

        Raises
        ------
        RuntimeError
            When that question is already being measured.
        """
        with self._lock:
            if question.question_id in self._measuring:
                raise RuntimeError(
                    f"question {question.question_id!r} is already being measured: "
                    "two open measurements would count its elapsed seconds twice"
                )
            self._measuring.add(question.question_id)

    def _end_measure(self, question: QuestionMeter, seconds: float) -> None:
        """Add one measured span to a question's elapsed seconds.

        Parameters
        ----------
        question : QuestionMeter
            The question that was measured.
        seconds : float
            Wall clock of the span.
        """
        with self._lock:
            question._elapsed += seconds
            self._measuring.discard(question.question_id)

    def _snapshot(self, question: QuestionMeter) -> QuestionUsage:
        """Return a snapshot of one question's cost.

        Parameters
        ----------
        question : QuestionMeter
            The question to read.

        Returns
        -------
        QuestionUsage
            The figures so far.
        """
        with self._lock:
            return _question_usage(question)

    def per_question(self) -> dict[str, QuestionUsage]:
        """Return what each question cost, keyed by question identifier.

        Returns
        -------
        dict
            One :class:`QuestionUsage` per question, in the order the questions
            were opened.
        """
        with self._lock:
            return {
                question_id: _question_usage(question)
                for question_id, question in self._questions.items()
            }

    def totals(self) -> RunUsage:
        """Return what the run cost, over every call this meter recorded.

        Returns
        -------
        RunUsage
            The run figures, including the calls no question was bound for.
        """
        with self._lock:
            return _run_usage(len(self._questions), self._run)

    def unattributed(self) -> RunUsage:
        """Return the cost of the calls no question was bound for.

        Returns
        -------
        RunUsage
            The unattributed figures, with ``questions`` zero.
        """
        with self._lock:
            return _run_usage(0, self._unattributed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the run and every question.

        Returns
        -------
        dict
            ``run`` holds the totals over every call, ``unattributed`` the calls
            no question was bound for, and ``questions`` one record per question,
            keyed by identifier.
        """
        with self._lock:
            return {
                "run": _run_usage(len(self._questions), self._run).to_dict(),
                "unattributed": self._unattributed.as_dict(),
                "questions": {
                    question_id: _question_usage(question).to_dict()
                    for question_id, question in self._questions.items()
                },
                "by_model": {
                    name: _model_usage(name, totals).to_dict()
                    for name, totals in self._by_model.items()
                },
            }

    def reset(self) -> None:
        """Drop every total, so an index run and a query run report separately.

        Call this between runs. A meter handed out before the reset keeps
        recording into the run totals, but its question is no longer listed.
        """
        with self._lock:
            self._run = _Totals()
            self._unattributed = _Totals()
            self._questions = {}
            self._measuring = set()
            self._by_model = {}

    def _target(self, question: QuestionMeter | None) -> _Totals:
        """Return the accumulator one call belongs to. The lock is held."""
        return self._unattributed if question is None else question._totals

    def _active_question(self) -> QuestionMeter | None:
        """Return the bound question, if it is this meter's."""
        question = _ACTIVE_QUESTION.get()
        # The context variable is process-wide, so a second meter must not
        # inherit the question a first one opened.
        if question is not None and question.meter is self:
            return question
        return None


def _question_usage(question: QuestionMeter) -> QuestionUsage:
    """Build one question's record. The meter's lock is held."""
    totals = question._totals
    return QuestionUsage(
        question_id=question.question_id,
        calls=totals.calls,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        total_tokens=totals.total_tokens,
        tool_calls=totals.tool_calls,
        cache_hits=totals.cache_hits,
        empty_responses=totals.empty_responses,
        calls_missing_usage=totals.calls_missing_usage,
        service_seconds=totals.service_seconds,
        elapsed_seconds=question._elapsed,
        prompt_cache_read_tokens=totals.prompt_cache_read_tokens,
        prompt_cache_write_tokens=totals.prompt_cache_write_tokens,
        prompt_cache_write_1h_tokens=totals.prompt_cache_write_1h_tokens,
    )


def _run_usage(questions: int, totals: _Totals) -> RunUsage:
    """Build a run record. The meter's lock is held."""
    return RunUsage(
        questions=questions,
        calls=totals.calls,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        total_tokens=totals.total_tokens,
        tool_calls=totals.tool_calls,
        cache_hits=totals.cache_hits,
        empty_responses=totals.empty_responses,
        calls_missing_usage=totals.calls_missing_usage,
        service_seconds=totals.service_seconds,
        prompt_cache_read_tokens=totals.prompt_cache_read_tokens,
        prompt_cache_write_tokens=totals.prompt_cache_write_tokens,
        prompt_cache_write_1h_tokens=totals.prompt_cache_write_1h_tokens,
    )


@dataclass
class _Ambient:
    """The meter an injection point reaches for when it is passed none.

    A record rather than a module global, so :func:`using` swaps the default
    without a ``global`` statement.
    """

    meter: UsageMeter


_AMBIENT = _Ambient(UsageMeter())


def default_meter() -> UsageMeter:
    """Return the process-wide meter the injection points use by default.

    Returns
    -------
    UsageMeter
        The shared meter. Pass a meter of your own wherever two runs must not
        add up together.
    """
    return _AMBIENT.meter


def reset_default_meter() -> UsageMeter:
    """Zero the process-wide meter and return it.

    Two runs in one process merge unless one of them zeroes the meter, and
    anything already holding this meter keeps recording into the run that
    follows. To keep a run's figures to itself instead of zeroing a shared
    meter, use ``with using(UsageMeter()) as meter:``.

    Returns
    -------
    UsageMeter
        The same shared meter, emptied, so anything already holding it keeps
        recording into the run that follows.
    """
    _AMBIENT.meter.reset()
    return _AMBIENT.meter


@contextmanager
def using(meter: UsageMeter) -> Iterator[UsageMeter]:
    """Make one meter the default for the block, then restore the previous one.

    The scoped alternative to zeroing the shared meter. Every injection point
    built inside the block resolves :func:`default_meter` to this meter, so two
    runs in one process keep their figures apart without either one emptying the
    other's.

    An injection point built before the block keeps the meter it captured, and
    the swap is process-wide rather than a :class:`~contextvars.ContextVar`, so
    this belongs in a run's single-threaded setup rather than inside a worker.

    Parameters
    ----------
    meter : UsageMeter
        The meter to install for the block.

    Yields
    ------
    UsageMeter
        The same meter, so the block can read its totals.

    Examples
    --------
    >>> with using(UsageMeter()) as meter:
    ...     complete = MeteredCompleter(lambda prompt: "ok")
    ...     with meter.question("q1"):
    ...         complete("say ok")
    'ok'
    >>> meter.totals().calls
    1
    """
    previous = _AMBIENT.meter
    _AMBIENT.meter = meter
    try:
        yield meter
    finally:
        _AMBIENT.meter = previous


class MeteredCompleter:
    """A metered ``complete`` callable, wrapping the one a stage already takes.

    Every generating stage in this package takes a ``complete`` callable rather
    than a client, so wrapping that callable meters query analysis, grading,
    generation, rewriting, and answer judging at once, without changing how any
    of them calls the model.

    Tokens come from the wrapped callable.
    :class:`~aieng.advanced_rag.pipeline.generate.ChatCompleter` reports each
    response through :func:`report_call`, and this wrapper collects the reports
    made during one call, so a completer that retried has every attempt it paid
    for counted. A callable that reports nothing still has its call and its
    seconds counted, and lands in ``calls_missing_usage``, so no token figure
    silently reads as zero. A call that raises is not counted at all: a transport
    failure is not a model call.

    Parameters
    ----------
    complete : callable
        Takes a prompt, returns text.
    meter : UsageMeter or None, optional
        Where the calls land. :func:`default_meter` when omitted.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the current context.
        Pass one when the question was opened in another thread.

    Examples
    --------
    >>> meter = UsageMeter()
    >>> complete = MeteredCompleter(lambda prompt: "ok", meter)
    >>> with meter.question("q1"):
    ...     complete("say ok")
    'ok'
    >>> meter.per_question()["q1"].calls
    1

    Notes
    -----
    One wrapper per completer. Wrapping a wrapper is refused rather than
    counted, because the inner one drains the reports and the outer one then
    sees an empty list and books a second call it has no tokens for. Wrap at the
    outermost point that should be metered.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        meter: UsageMeter | None = None,
        *,
        charge_to: QuestionMeter | None = None,
    ) -> None:
        self._complete = complete
        self._meter = meter or default_meter()
        self._charge_to = charge_to

    def __call__(self, prompt: str) -> str:
        """Send one prompt and return its text, counting what it cost.

        Parameters
        ----------
        prompt : str
            The prompt to send.

        Returns
        -------
        str
            Whatever the wrapped callable returned.

        Raises
        ------
        RuntimeError
            When another wrapper is already collecting reports in this context.
            Both would count the same call, and the outer one would count it
            with no tokens.
        """
        if _PENDING_CALLS.get() is not None:
            raise RuntimeError(
                "MeteredCompleter is not reentrant: another one is already "
                "collecting reports in this context, so the same call would be "
                "counted twice and the outer count would carry no tokens. Wrap "
                "the completer once, at the outermost point to be metered."
            )
        pending: list[_ReportedCall] = []
        token = _PENDING_CALLS.set(pending)
        started = time.perf_counter()
        try:
            text = self._complete(prompt)
        finally:
            seconds = time.perf_counter() - started
            _PENDING_CALLS.reset(token)
        if pending:
            for call in pending:
                self._record(call)
            return text
        # Nothing reported, so the tokens are unknown and the only signal for an
        # abstention is the text the callable returned.
        self._record(_ReportedCall(CallUsage(), seconds, empty=not text.strip()))
        return text

    def _record(self, call: _ReportedCall) -> None:
        """Charge one reported call to the question it holds, or to the meter."""
        if call.cached:
            target = self._charge_to if self._charge_to is not None else self._meter
            target.record_cache_hit(model=call.model, usage=call.usage)
            return
        recorder = self._charge_to if self._charge_to is not None else self._meter
        recorder.record(
            call.usage,
            call.seconds,
            tool_calls=call.tool_calls,
            empty=call.empty,
            model=call.model,
            response_model=call.response_model,
        )


def _ask_for_usage(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a streamed request that asks for its usage.

    A streamed response reports usage only when the request sets
    ``stream_options={"include_usage": True}``, and says nothing when it does
    not, so the default here is to ask. A caller who set ``include_usage``
    already keeps their own value.
    """
    options = kwargs.get("stream_options")
    if options is None:
        return {**kwargs, "stream_options": {"include_usage": True}}
    if "include_usage" in options:
        return kwargs
    return {**kwargs, "stream_options": {**options, "include_usage": True}}


@dataclass
class _StreamState:
    """What a streamed response said while its chunks went past."""

    usage: CallUsage = field(default_factory=CallUsage)
    tool_calls: int = 0
    text: bool = False

    def read(self, chunk: Any) -> None:
        """Take the usage, the tool calls, and any text out of one chunk."""
        usage = usage_of(chunk)
        if usage.reported:
            self.usage = usage
        for choice in _attr(chunk, "choices") or ():
            delta = _attr(choice, "delta")
            if _attr(delta, "content"):
                self.text = True
            for call in _attr(delta, "tool_calls") or ():
                # Tool calls arrive as deltas against an index, so the highest
                # index seen is how many the model asked for.
                index = _count(call, "index") or 0
                self.tool_calls = max(self.tool_calls, index + 1)


class _CallRecorder:
    """Times one client call, reads what came back, and records it.

    Shared by the client wrapper, the in-place patch, and the Microsoft GraphRAG
    provider, so all three count a response the same way. Holds no per-call
    state, because one recorder serves every thread using the client.
    """

    def __init__(
        self, meter: UsageMeter | None, charge_to: QuestionMeter | None
    ) -> None:
        self._meter = meter
        self._charge_to = charge_to

    def run(self, create: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Make one call and record what it cost, streamed or not."""
        streaming = bool(kwargs.get("stream"))
        if streaming:
            kwargs = _ask_for_usage(kwargs)
        model = kwargs.get("model")
        requested = str(model) if model else None
        before_call(requested)
        started = time.perf_counter()
        result = create(**kwargs)
        if inspect.isawaitable(result):
            return self._awaited(result, started, streaming, requested)
        return self._finish(result, started, streaming, requested)

    async def _awaited(
        self, awaitable: Any, started: float, streaming: bool, model: str | None
    ) -> Any:
        """Await an asynchronous client's result, then record it."""
        return self._finish(await awaitable, started, streaming, model)

    def _finish(
        self, result: Any, started: float, streaming: bool, model: str | None
    ) -> Any:
        """Record a completed call, or wrap a stream so it records as it goes."""
        if not streaming:
            facts = read_completion(result)
            self._record(
                facts.usage,
                time.perf_counter() - started,
                facts.tool_calls,
                facts.empty,
                model=model,
                response_model=facts.model,
            )
            return result
        if hasattr(result, "__aiter__"):
            return self._astream(result, started, model)
        return self._stream(result, started, model)

    def _stream(
        self, chunks: Iterable[Any], started: float, model: str | None
    ) -> Iterator[Any]:
        """Pass a stream through, recording the call once it ends."""
        state = _StreamState()
        try:
            for chunk in chunks:
                state.read(chunk)
                yield chunk
        finally:
            self._record_stream(state, time.perf_counter() - started, model)

    async def _astream(
        self, chunks: AsyncIterable[Any], started: float, model: str | None
    ) -> AsyncIterator[Any]:
        """Pass an asynchronous stream through, recording the call once it ends."""
        state = _StreamState()
        try:
            async for chunk in chunks:
                state.read(chunk)
                yield chunk
        finally:
            self._record_stream(state, time.perf_counter() - started, model)

    def _record_stream(
        self, state: _StreamState, seconds: float, model: str | None = None
    ) -> None:
        """Record one streamed call, saying so when its tokens never arrived."""
        if not state.usage.reported:
            warnings.warn(
                "a streamed response reported no usage, so its tokens are "
                "unknown rather than zero: the request needs "
                'stream_options={"include_usage": True}, and a cancelled stream '
                "can drop the usage chunk even then",
                MissingUsageWarning,
                stacklevel=2,
            )
        self._record(
            state.usage,
            seconds,
            state.tool_calls,
            not state.text and state.tool_calls == 0,
            model=model,
        )

    def _record(
        self,
        usage: CallUsage,
        seconds: float,
        tool_calls: int,
        empty: bool,
        *,
        model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        """Charge one call to the meter it holds, or to the active question."""
        if model and response_model:
            _observe_response_model(model, response_model)
        try:
            if self._charge_to is not None:
                self._charge_to.record(
                    usage,
                    seconds,
                    tool_calls=tool_calls,
                    empty=empty,
                    model=model,
                    response_model=response_model,
                )
                return
            meter = self._meter or default_meter()
            meter.record(
                usage,
                seconds,
                tool_calls=tool_calls,
                empty=empty,
                model=model,
                response_model=response_model,
            )
        finally:
            _run_call_hooks(model, usage)


class _MeteredCompletions:
    """Meters every ``create`` call, streamed or not, awaited or not."""

    def __init__(self, inner: Any, recorder: _CallRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        """Call the wrapped client and record what the call cost."""
        return self._recorder.run(self._inner.create, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _MeteredChat:
    """Hands out metered completions and passes everything else through."""

    def __init__(self, inner: Any, recorder: _CallRecorder) -> None:
        self._inner = inner
        self.completions = _MeteredCompletions(inner.completions, recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class MeteredClient:
    """An OpenAI-compatible client that meters every completion it makes.

    The injection point for a library that builds its own requests and takes no
    completion callable, such as LightRAG or atlas-rag. Wraps rather than
    subclasses, so any object exposing ``chat.completions.create`` works, and
    every other attribute passes through untouched.

    Streamed calls are handled rather than dropped: the request asks for usage
    through ``stream_options``, the chunks are read as the caller consumes them,
    and the call is recorded when the stream ends. A library that runs its calls
    on its own threads records against the unattributed bucket, because a thread
    starts with no question bound; pass ``charge_to`` to charge one anyway.

    Parameters
    ----------
    client : Any
        The client to wrap, normally ``openai.OpenAI`` or ``openai.AsyncOpenAI``
        pointed at the proxy.
    meter : UsageMeter or None, optional
        Where the calls land. :func:`default_meter` when omitted.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the calling context.

    Examples
    --------
    >>> from openai import OpenAI  # doctest: +SKIP
    >>> meter = UsageMeter()  # doctest: +SKIP
    >>> client = MeteredClient(OpenAI(), meter)  # doctest: +SKIP
    """

    def __init__(
        self,
        client: Any,
        meter: UsageMeter | None = None,
        *,
        charge_to: QuestionMeter | None = None,
    ) -> None:
        self._client = client
        self.chat = _MeteredChat(client.chat, _CallRecorder(meter, charge_to))

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute this wrapper does not define to the client."""
        return getattr(self._client, name)


def meter_client(
    client: Any,
    meter: UsageMeter | None = None,
    *,
    charge_to: QuestionMeter | None = None,
) -> Any:
    """Meter a client in place, returning the same object.

    Prefer this over :class:`MeteredClient` when the consuming library
    type-checks its client. ``atlas_rag.llm_generator.LLMGenerator`` raises
    ``ValueError: Unsupported client type`` for anything that is not an
    ``OpenAI`` or ``AzureOpenAI`` instance, so no wrapper reaches it however
    transparent it is. Idempotent, and it composes with
    :func:`~aieng.advanced_rag.reasoning.disable_thinking` in either order.

    Parameters
    ----------
    client : Any
        An OpenAI-compatible client. Mutated in place.
    meter : UsageMeter or None, optional
        Where the calls land. :func:`default_meter` when omitted.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the calling context.

    Returns
    -------
    Any
        The same client, so this reads as a decorator at the call site.
    """
    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_metering_installed", False):
        return client
    recorder = _CallRecorder(meter, charge_to)

    def create(**kwargs: Any) -> Any:
        return recorder.run(original, kwargs)

    create._metering_installed = True  # type: ignore[attr-defined]
    completions.create = create
    return client


def _metered_completion_class(base: Any, recorder: _CallRecorder) -> Any:
    """Return ``base`` with every completion it makes recorded.

    Taking the base class as an argument keeps this testable without building a
    real LiteLLM model, which needs a full model configuration.
    """

    class MeteredCompletion(base):  # type: ignore[misc]
        """Completions with the cost of every response recorded."""

        def completion(self, /, **kwargs: Any) -> Any:
            """Complete one request, recording what it cost."""
            return recorder.run(super().completion, kwargs)

        async def completion_async(self, /, **kwargs: Any) -> Any:
            """Complete one request asynchronously, recording what it cost."""
            return await recorder.run(super().completion_async, kwargs)

    return MeteredCompletion


def register_metered_completion(
    meter: UsageMeter | None = None,
    provider: str = METERED_PROVIDER,
) -> str:
    """Register a metered completion provider with Microsoft GraphRAG.

    Microsoft GraphRAG builds its own model objects from its configuration file,
    so neither a completion callable nor a patched client reaches it. A
    registered provider does, and a model selects it with
    ``type: metered_litellm``. Its calls are charged to the question bound to
    whichever context the library calls from, and to the unattributed bucket when
    the library uses threads of its own, which indexing does.

    Idempotent, because the indexer and the query path both build models and
    either may run first.

    Parameters
    ----------
    meter : UsageMeter or None, optional
        Where the calls land. :func:`default_meter` when omitted, read at call
        time, so :func:`reset_default_meter` between runs needs no second
        registration.
    provider : str, optional
        Name to register under.

    Returns
    -------
    str
        The provider name, for writing into a model configuration.

    Raises
    ------
    ImportError
        When the ``graphrag`` extra is not installed.
    """
    from graphrag_llm.completion.completion_factory import (
        completion_factory,
        register_completion,
    )
    from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion

    if provider in completion_factory:
        return provider
    register_completion(
        completion_type=provider,
        completion_initializer=_metered_completion_class(
            LiteLLMCompletion, _CallRecorder(meter, None)
        ),
        scope="singleton",
    )
    return provider


__all__ = [
    "METERED_PROVIDER",
    "UNKNOWN_MODEL",
    "CallUsage",
    "MeteredClient",
    "MeteredCompleter",
    "MissingUsageWarning",
    "ModelUsage",
    "ParsedCompletion",
    "QuestionMeter",
    "CallHook",
    "PreCallHook",
    "QuestionUsage",
    "ReportScope",
    "RunUsage",
    "UsageMeter",
    "add_call_hook",
    "add_pre_call_hook",
    "RunStopped",
    "before_call",
    "current_question",
    "default_meter",
    "meter_client",
    "observed_response_model",
    "read_completion",
    "report_scope",
    "register_metered_completion",
    "remove_call_hook",
    "remove_pre_call_hook",
    "report_cache_hit",
    "report_call",
    "reset_default_meter",
    "reset_unmetered_calls",
    "unmetered_calls",
    "usage_of",
    "using",
]

#: Deprecated alias for :class:`UsageMeter`, kept for callers of the old name.
#: Deprecated in 0.1.1; use ``UsageMeter``, which
#: is what it reports: no price, no currency, no model rate.
CostMeter = UsageMeter

#: Deprecated alias for :class:`QuestionUsage`. Deprecated in 0.1.1; use
#: ``QuestionUsage``.
QuestionCost = QuestionUsage

#: Deprecated alias for :class:`RunUsage`. Deprecated in 0.1.1; use ``RunUsage``.
RunCost = RunUsage

#: Deprecated alias for :class:`ParsedCompletion`. Deprecated in 0.1.1; use
#: ``ParsedCompletion``.
CompletionFacts = ParsedCompletion

#: Deprecated alias for :class:`QuestionMeter`. Deprecated in 0.1.1; use
#: ``QuestionMeter``: a scope on this package's retrieval side is the corpus's
#: own filter, and this is a usage accumulator.
QuestionScope = QuestionMeter
