"""Turning a reasoning model's thinking off, for libraries that will not.

The default bulk model on the Vector proxy, ``Qwen3_6-35B-A3B``, is a reasoning
model, and its reasoning tokens are billed as completion tokens. On short
structured calls, the kind graph construction makes thousands of, the reasoning
is almost the entire cost: measured at 1,312 completion tokens against 16 for
the same answer.

Worse than the cost is the failure mode. When ``max_tokens`` runs out during
reasoning, the API returns ``content=None`` with a normal finish reason. A
library that assumes a string gets ``AttributeError: 'NoneType' object has no
attribute 'find'`` rather than anything that names the problem. ``atlas-rag``
fails exactly this way on every triple-extraction call.

The switch is ``chat_template_kwargs={"enable_thinking": false}`` in
``extra_body``. ``reasoning_effort`` is ignored by this deployment. A library
that constructs its own requests has nowhere to put that, so the switch is
injected at the client instead.

The switch belongs to the on-prem models only. Whether the proxy drops
``chat_template_kwargs`` for a paid model or rejects it is unknown, so the
injected body is chosen per call from the request's ``model`` through
:func:`aieng.advanced_rag.models.thinking_body`: the on-prem switch for Qwen and
any unrecognised self-hosted name, and each paid model's own field, such as
``reasoning_effort`` for OpenAI models and the ``thinking`` field for Claude.

Two ways to do that, because libraries differ:

* :func:`disable_thinking` patches the client in place and returns it. Use this
  with any library that type-checks its client, which ``atlas-rag`` does:
  it raises ``Unsupported client type`` for anything that is not literally an
  ``OpenAI`` instance.
* :class:`ThinkingOffClient` wraps instead of mutating, which is cleaner when
  the caller owns the client and nothing inspects its type.

Both also pass every call through :func:`aieng.advanced_rag.models.adapt_request`
before the switch is merged, because a library that will not turn thinking off
also sends parameters some paid models reject: atlas-rag sends ``max_tokens``,
``temperature``, and ``response_format={"type": "json_object"}`` to any model.
On-prem requests are unchanged.
"""

from __future__ import annotations

from typing import Any

from aieng.advanced_rag.models import adapt_request, thinking_body


THINKING_OFF: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def _adapted(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the call's arguments in the shape its model accepts."""
    model = kwargs.get("model")
    if model is None:
        return kwargs
    rest = {key: value for key, value in kwargs.items() if key != "model"}
    return {"model": model, **adapt_request(str(model), rest)}


def _switch_for(kwargs: dict[str, Any], fixed: dict[str, Any] | None) -> dict[str, Any]:
    """Return the body to inject: a caller's fixed one, or the model's own."""
    if fixed is not None:
        return fixed
    body: dict[str, Any] = thinking_body(str(kwargs.get("model", "")))
    return body


class _Completions:
    """Injects the thinking switch into every ``create`` call."""

    def __init__(self, inner: Any, extra_body: dict[str, Any] | None) -> None:
        self._inner = inner
        self._extra_body = extra_body

    def create(self, **kwargs: Any) -> Any:
        """Call the wrapped client, merging the thinking switch into extra_body.

        A caller's own ``extra_body`` wins on conflicting keys, so this can be
        overridden per call rather than being a lock. The arguments are first
        adapted to the model with :func:`~aieng.advanced_rag.models.adapt_request`.
        """
        kwargs = _adapted(kwargs)
        switch = _switch_for(kwargs, self._extra_body)
        extra_body = {**switch, **(kwargs.pop("extra_body", None) or {})}
        if not extra_body:
            return self._inner.create(**kwargs)
        return self._inner.create(extra_body=extra_body, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: Any, extra_body: dict[str, Any] | None) -> None:
        self._inner = inner
        self.completions = _Completions(inner.completions, extra_body)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ThinkingOffClient:
    """An OpenAI-compatible client with reasoning disabled on every call.

    Wraps rather than subclasses, so it works with any client object exposing
    ``chat.completions.create``, and passes everything else through untouched.

    Parameters
    ----------
    client : Any
        The client to wrap, normally ``openai.OpenAI`` pointed at the proxy.
    extra_body : dict, optional
        What to inject on every call. By default the body is chosen per call
        from the request's model, as the module docstring describes.

    Examples
    --------
    >>> from openai import OpenAI  # doctest: +SKIP
    >>> client = ThinkingOffClient(OpenAI())  # doctest: +SKIP
    >>> generator = LLMGenerator(
    ...     client=client, model_name="Qwen3_6-35B-A3B"
    ... )  # doctest: +SKIP
    """

    def __init__(self, client: Any, extra_body: dict[str, Any] | None = None) -> None:
        self._client = client
        self.chat = _Chat(client.chat, extra_body)

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute this wrapper does not define to the client."""
        return getattr(self._client, name)


def disable_thinking(client: Any, extra_body: dict[str, Any] | None = None) -> Any:
    """Disable reasoning on a client in place, returning the same object.

    Prefer this over :class:`ThinkingOffClient` when the consuming library
    type-checks its client. ``atlas_rag.llm_generator.LLMGenerator`` raises
    ``ValueError: Unsupported client type`` for anything that is not an
    ``OpenAI`` or ``AzureOpenAI`` instance, so a wrapper cannot be passed to it
    however transparent the wrapper is.

    Parameters
    ----------
    client : Any
        An OpenAI-compatible client. Mutated in place.
    extra_body : dict, optional
        What to inject on every call. By default the body is chosen per call
        from the request's model.

    Returns
    -------
    Any
        The same client, so this reads as a decorator at the call site.
    """
    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_thinking_disabled", False):
        return client

    def create(**kwargs: Any) -> Any:
        kwargs = _adapted(kwargs)
        switch = _switch_for(kwargs, extra_body)
        merged = {**switch, **(kwargs.pop("extra_body", None) or {})}
        if not merged:
            return original(**kwargs)
        return original(extra_body=merged, **kwargs)

    create._thinking_disabled = True  # type: ignore[attr-defined]
    completions.create = create
    return client
