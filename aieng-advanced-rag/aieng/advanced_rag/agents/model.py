"""The model side of an ADK agent: one governed LiteLLM client per run.

ADK's ``LiteLlm`` hands every request to an ``LiteLLMClient`` it holds as
``llm_client``. This module's :class:`GovernedLlmClient` is one, and does for
the ADK path what :func:`aieng.advanced_rag.graphrag.proxy.govern` does for an
OpenAI client, except caching, which happens a layer up in
:class:`~aieng.advanced_rag.agents.plugins.ResponseCachePlugin` so that a hit
never reaches the client:

* The request goes through :func:`~aieng.advanced_rag.models.adapt_request`, so
  a paid model gets the parameters its family accepts whatever the agent's
  configuration asked for.
* :func:`~aieng.advanced_rag.metering.before_call` runs first, so a spending
  ceiling stops the call before it is sent.
* A shared :class:`~aieng.advanced_rag.graphrag.proxy.RateLimiter` admits it.
* A transient error (HTTP 429, a connection failure, a 5xx) is retried on
  :func:`~aieng.advanced_rag.graphrag.proxy.backoff_seconds`; anything else is
  raised on the first attempt.
* The response is reported to :mod:`aieng.advanced_rag.metering` with the model
  requested, the model reported, and the call's own seconds.

Reach this module by path. It needs the ``agents`` extra.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

from .. import metering
from ..graphrag.protocol import MissingExtraError
from ..graphrag.proxy import RateLimiter, backoff_seconds, transient_errors
from ..metering import QuestionMeter
from ..models import adapt_request


try:
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
except ImportError as error:  # pragma: no cover - import guard
    raise MissingExtraError("agents", "google-adk") from error


def _parse_arguments(raw: str) -> Any:
    """Parse tool-call arguments with the parser LiteLlm applies to a response.

    Using the same one means an argument string is replaced exactly when
    LiteLlm would have raised on it.
    """
    try:
        from google.adk.models.lite_llm import (  # noqa: PLC0415
            _parse_tool_call_arguments,
        )
    except ImportError:  # pragma: no cover - a later ADK may rename it
        return json.loads(raw)
    return _parse_tool_call_arguments(raw)


#: The single argument a tool call carries when its arguments did not parse.
#: :class:`~aieng.advanced_rag.agents.plugins.QuestionGuardPlugin` answers such
#: a call with an error instead of running the tool.
UNREADABLE_ARGUMENTS = "__unreadable_arguments__"


def mark_unreadable_arguments(response: Any) -> int:
    """Replace tool-call arguments that do not parse to an object, in place.

    LiteLlm raises ``JSONDecodeError`` on such a call, which ends the whole
    run. The chat-completions loop answered it with an error the model could
    read, and this keeps that behaviour: the arguments become
    ``{UNREADABLE_ARGUMENTS: "<reason>"}``, which the question guard turns into
    a tool error.

    Parameters
    ----------
    response : litellm.ModelResponse
        A non-streamed response.

    Returns
    -------
    int
        How many tool calls were replaced.
    """
    replaced = 0
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            raw = getattr(function, "arguments", None)
            if function is None or not raw:
                continue
            try:
                parsed = _parse_arguments(raw)
            except (ValueError, SyntaxError) as error:
                reason = f"could not read the arguments: {error}"
            else:
                if isinstance(parsed, dict):
                    continue
                reason = "could not read the arguments: they are not a JSON object"
            function.arguments = json.dumps({UNREADABLE_ARGUMENTS: reason})
            replaced += 1
    return replaced


def _retryable() -> tuple[type[Exception], ...]:
    """Return the errors worth another attempt, in LiteLLM's classes too.

    LiteLLM's exceptions subclass the OpenAI client's, so the classes
    :func:`transient_errors` names catch them, except a 503, which LiteLLM
    raises as ``ServiceUnavailableError`` under ``APIStatusError``.
    """
    errors = transient_errors()
    try:
        from litellm.exceptions import ServiceUnavailableError  # noqa: PLC0415
    except ImportError:  # pragma: no cover - LiteLLM comes with ADK's extra
        return errors
    return (*errors, ServiceUnavailableError)


def bare_model(model: str) -> str:
    """Return a model name without LiteLLM's provider prefix.

    Parameters
    ----------
    model : str
        A name such as ``openai/Qwen3_6-35B-A3B``.

    Returns
    -------
    str
        ``Qwen3_6-35B-A3B``. A name with no prefix comes back unchanged.
    """
    return model.split("/", 1)[1] if "/" in model else model


class GovernedLlmClient(LiteLLMClient):
    """A LiteLLM client that adapts, guards, throttles, retries, and meters.

    Build one per agent run, because :attr:`waited` and ``charge_to`` belong to
    the run.

    Parameters
    ----------
    limiter : RateLimiter or None, optional
        The request budget shared by every caller of this model. None sends
        without waiting.
    max_retries : int, optional
        Attempts at a call that keeps raising a transient error.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the current context.

    Attributes
    ----------
    waited : float
        Seconds this client spent waiting for the limiter, so a caller can take
        them off a run's time.
    calls : int
        Responses this client received.
    """

    def __init__(
        self,
        *,
        limiter: RateLimiter | None = None,
        max_retries: int = 16,
        charge_to: QuestionMeter | None = None,
    ) -> None:
        super().__init__()
        self.limiter = limiter
        self.max_retries = max_retries
        self.charge_to = charge_to
        self.waited = 0.0
        self.calls = 0

    async def acompletion(
        self, model: Any, messages: Any, tools: Any, **kwargs: Any
    ) -> Any:
        """Send one chat completion through the guards.

        Parameters
        ----------
        model : str
            The LiteLLM model name, with its provider prefix.
        messages : list
            The chat messages ``LiteLlm`` built.
        tools : list or None
            The tool declarations.
        **kwargs : Any
            The rest of the request, adapted for the model's family before it
            is sent.

        Returns
        -------
        litellm.ModelResponse
            The response.

        Raises
        ------
        ValueError
            For a streaming request, which this client does not meter.
        RuntimeError
            When every attempt raised a transient error.
        Exception
            Whatever a pre-call hook raises, such as
            ``spend.SpendLimitReached``, or any non-transient client error.
        """
        if kwargs.get("stream"):
            raise ValueError("GovernedLlmClient does not support streaming responses")
        name = bare_model(str(model))
        params = adapt_request(name, kwargs)
        # The OpenAI client inside LiteLLM retries on its own minute-long
        # schedule; retries happen here instead, on the proxy's cadence.
        params.setdefault("max_retries", 0)
        metering.before_call(name)
        retryable = _retryable()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            if self.limiter is not None:
                started = time.perf_counter()
                await self.limiter.aacquire()
                self.waited += time.perf_counter() - started
            started = time.perf_counter()
            try:
                response = await super().acompletion(
                    model=model, messages=messages, tools=tools, **params
                )
            except retryable as error:
                last = error
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            self.calls += 1
            self._report(name, response, time.perf_counter() - started)
            mark_unreadable_arguments(response)
            return response
        raise RuntimeError(f"model call failed after {self.max_retries} tries: {last}")

    def _report(self, model: str, response: Any, seconds: float) -> None:
        """Hand one response to the meter, charging ``charge_to`` when given."""
        usage = metering.usage_of(response)
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", None) if message is not None else None
        calls = len(getattr(message, "tool_calls", None) or []) if message else 0
        reported = getattr(response, "model", None) or None
        fields: dict[str, Any] = {
            "tool_calls": calls,
            "empty": not content and not calls,
            "model": model,
            "response_model": reported,
        }
        if self.charge_to is None:
            metering.report_call(usage, seconds, **fields)
            return
        # Binding routes the report to this question and still runs the call
        # hooks, which is where a spending ceiling sees the call.
        with self.charge_to.bind():
            metering.report_call(usage, seconds, **fields)


def governed_model(
    model: str,
    *,
    base_url: str,
    api_key: str,
    extra_body: Mapping[str, Any] | None = None,
    client: GovernedLlmClient | None = None,
) -> Any:
    """Return a ``LiteLlm`` for an OpenAI-compatible endpoint, on a governed client.

    Parameters
    ----------
    model : str
        The bare model name the endpoint serves.
    base_url : str
        The endpoint.
    api_key : str
        The key, sent as a bearer token.
    extra_body : Mapping or None, optional
        Fields merged into the request body, such as the thinking switch.
    client : GovernedLlmClient or None, optional
        The client to send through. A new one with default settings when
        omitted.

    Returns
    -------
    LiteLlm
        The model object for an ``LlmAgent``.
    """
    return LiteLlm(
        # The provider prefix is what routes this to LiteLLM's
        # OpenAI-compatible path, which is the path that copies an
        # unrecognised parameter into the request body.
        model=f"openai/{model}",
        api_base=base_url,
        api_key=api_key,
        extra_body=dict(extra_body or {}),
        llm_client=client or GovernedLlmClient(),
    )


__all__ = [
    "UNREADABLE_ARGUMENTS",
    "GovernedLlmClient",
    "bare_model",
    "governed_model",
    "mark_unreadable_arguments",
]
