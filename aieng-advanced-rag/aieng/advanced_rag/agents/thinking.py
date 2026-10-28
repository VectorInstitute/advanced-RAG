"""The switch that turns this deployment's reasoning off, for the agent loop.

One function, because the two routes that look like they should work do not, and
the reason belongs next to the value rather than in a commit message.

The default bulk model on the proxy, ``Qwen3_6-35B-A3B``, is a reasoning model
whose reasoning is billed as completion tokens: 1,312 of them against 16 for the
same answer. :mod:`aieng.advanced_rag.reasoning` injects the same switch into an
OpenAI-compatible client for libraries that build their own requests. An agent
loop hands it to the model object instead, so it lives here.

Nothing in this module imports ADK, so the shape of the switch is checkable
without the ``agents`` extra installed.
"""

from __future__ import annotations

from typing import Any


def thinking_extra_body(enable_thinking: bool) -> dict[str, Any]:
    """Return the request body that turns this deployment's reasoning on or off.

    The switch has to travel in ``extra_body``, because both first-class routes
    fail on the LiteLLM path.

    ADK's own thinking configuration is dropped in silence. ``ThinkingConfig``,
    whether set through ``generate_content_config.thinking_config`` or a
    ``BuiltInPlanner``, lands on the request object, and the LiteLLM adapter
    copies only eight allowlisted generation parameters out of that object:
    temperature, max output tokens, top-p, top-k, seed, stop sequences, presence
    penalty, and frequency penalty. The thinking configuration is not one of
    them and no warning says so, which makes the silence easy to read as
    success.

    ``reasoning_effort`` is worse than ignored. LiteLLM raises
    ``UnsupportedParamsError`` for it on an OpenAI-compatible provider before
    any request leaves the process, so the proxy's documented habit of ignoring
    that parameter is never reached. Leave it out.

    Parameters
    ----------
    enable_thinking : bool
        Whether the model reasons before answering. Reasoning is billed as
        completion tokens here, measured at 1,312 against 16 for the same
        answer, so a bulk run leaves it off.

    Returns
    -------
    dict
        The ``extra_body`` to hand to ``LiteLlm``. LiteLLM copies what it does
        not recognise into the request body, and an OpenAI-compatible client
        merges ``extra_body`` into the top level of the JSON it sends, which is
        where vLLM reads ``chat_template_kwargs``.
    """
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


__all__ = ["thinking_extra_body"]
