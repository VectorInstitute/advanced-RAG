"""What one agent event cost, handed to the shared cost meter.

ADK reports token usage per model call. ``Event`` subclasses ``LlmResponse``, so
``event.usage_metadata`` carries the provider's own counts off the ``run_async``
stream with no extra plumbing, and an event without that field is not a model
response: a tool result is an event too.

The names differ from the ones a chat completion uses, which is the only reason
this module exists. ADK reports ``prompt_token_count``,
``candidates_token_count``, and ``total_token_count``, while
:class:`~aieng.advanced_rag.metering.CallUsage` holds ``prompt_tokens``,
``completion_tokens``, and ``total_tokens``. Nothing else is translated: the
counts are the provider's, and the meter does the accounting.

Nothing here imports ADK. The fields are read by name off whatever the runner
yields, the way :mod:`aieng.advanced_rag.metering` reads a chat completion, so
this module is importable without the ``agents`` extra and testable without a
runner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..metering import CallUsage, QuestionMeter, report_call


@dataclass(frozen=True)
class ParsedEvent:
    """What one model call reported, read off the event that carried it.

    Parameters
    ----------
    usage : CallUsage
        The token counts, with ``reported`` false when the event carried a usage
        block holding no numbers.
    text : str
        The answer text this response carried, or ``""``.
    tool_calls : tuple of str
        Tool names this response asked for, in the order asked. A tool call is a
        model call, so this is what stops a response with no text being read as
        an abstention.
    empty : bool
        Whether the response carried neither answer text nor a tool call.
    """

    usage: CallUsage
    text: str
    tool_calls: tuple[str, ...]
    empty: bool


def _field(source: Any, name: str) -> Any:
    """Read one field off an object or a mapping, or ``None`` when absent."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _count(source: Any, name: str) -> int | None:
    """Read one token count, or ``None`` when absent or not a number."""
    value = _field(source, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def usage_of_event(event: Any) -> CallUsage | None:
    """Read the token counts one event reported.

    ``candidates_token_count`` is taken as the whole completion count, and
    ``thoughts_token_count`` is not added to it. The proxy bills reasoning as
    completion tokens and returns ``completion_tokens_details`` as null, so the
    thought count is either absent or a subset of a figure already counted.

    Parameters
    ----------
    event : Any
        An event from ``Runner.run_async``.

    Returns
    -------
    CallUsage or None
        The counts, or ``None`` when the event carried no usage block at all,
        which is how a tool-result event differs from a model response.
    """
    metadata = _field(event, "usage_metadata")
    if metadata is None:
        return None
    prompt = _count(metadata, "prompt_token_count")
    completion = _count(metadata, "candidates_token_count")
    total = _count(metadata, "total_token_count")
    if prompt is None and completion is None and total is None:
        return CallUsage()
    prompt = prompt or 0
    completion = completion or 0
    return CallUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        reported=True,
    )


def read_event(event: Any) -> ParsedEvent | None:
    """Read one event's tokens and shape, or ``None`` when it is not a call.

    Parameters
    ----------
    event : Any
        An event from ``Runner.run_async``.

    Returns
    -------
    ParsedEvent or None
        What the model call reported, or ``None`` when the event carried no
        usage block and so records no call.
    """
    usage = usage_of_event(event)
    if usage is None:
        return None
    text, tool_calls = _parts(event)
    return ParsedEvent(
        usage=usage,
        text=text,
        tool_calls=tool_calls,
        empty=not text.strip() and not tool_calls,
    )


def sum_usage(usages: Iterable[CallUsage]) -> CallUsage:
    """Add up what several calls reported.

    Parameters
    ----------
    usages : iterable of CallUsage
        One entry per model call.

    Returns
    -------
    CallUsage
        The totals, with ``reported`` true only when there was at least one call
        and every one of them reported its tokens. A total that is missing a
        call's usage is not a whole figure, which is the distinction
        ``QuestionUsage.tokens_complete`` makes on the meter's side.
    """
    items = list(usages)
    return CallUsage(
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
        reported=bool(items) and all(item.reported for item in items),
    )


def _parts(event: Any) -> tuple[str, tuple[str, ...]]:
    """Return one event's answer text and the tools it asked for.

    A part marked ``thought`` is not answer text. With thinking on, a call that
    spends its whole completion budget reasoning comes back as thought parts and
    nothing else, and reading that as an answer would hide the truncation that
    :mod:`aieng.advanced_rag.reasoning` exists to avoid.
    """
    content = _field(event, "content")
    parts = _field(content, "parts") or ()
    text = ""
    tool_calls: list[str] = []
    for part in parts:
        call = _field(part, "function_call")
        if call is not None:
            tool_calls.append(str(_field(call, "name") or ""))
        elif not _field(part, "thought"):
            text += _field(part, "text") or ""
    return text, tuple(tool_calls)


def report_event(
    event: Any,
    seconds: float,
    *,
    charge_to: QuestionMeter | None = None,
) -> ParsedEvent | None:
    """Charge one event's model call to whatever is metering this run.

    With no ``charge_to``, the call goes to
    :func:`~aieng.advanced_rag.metering.report_call`, which hands it to the
    question bound to the current context. Asyncio copies that context into each
    task, so a caller that opens ``meter.question(...)`` around the run gets its
    calls attributed with nothing passed down. Pass ``charge_to`` when the run
    happens in a thread that never bound one.

    Parameters
    ----------
    event : Any
        An event from ``Runner.run_async``.
    seconds : float
        How long this call took, which the meter adds to ``service_seconds``.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the current context.

    Returns
    -------
    ParsedEvent or None
        What was recorded, or ``None`` when the event was not a model call and
        nothing was recorded.
    """
    facts = read_event(event)
    if facts is None:
        return None
    if charge_to is None:
        report_call(
            facts.usage,
            seconds,
            tool_calls=len(facts.tool_calls),
            empty=facts.empty,
        )
    else:
        charge_to.record(
            facts.usage,
            seconds,
            tool_calls=len(facts.tool_calls),
            empty=facts.empty,
        )
    return facts


__all__ = [
    "ParsedEvent",
    "read_event",
    "report_event",
    "sum_usage",
    "usage_of_event",
]

#: Deprecated alias for :class:`ParsedEvent`. Deprecated in 0.1.1; use
#: ``ParsedEvent``.
EventFacts = ParsedEvent
