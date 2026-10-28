"""Run one question over a tool session on ADK, and return an :class:`AgentRun`.

:class:`~aieng.advanced_rag.agents.loop.AgentLoop` runs the session's tools
with the arm's call budget and a forced final answer, the read-before-answer
guard, and the prompt-size guard, with every response cached through
:class:`~aieng.advanced_rag.agents.plugins.ResponseCachePlugin`. The record it
returns has the fields of the chat-completions loop it replaced, so stored
records read the same.

ADK's ``FunctionTool`` declares the tools from each function's signature and
docstring (:func:`~aieng.advanced_rag.agents.toolset.tool_declarations` returns
them), so the requests differ from that loop's and its cache entries do not
hit. LiteLLM repairs tool-call arguments written as Python literals or with
unquoted keys, which that loop did not; arguments it cannot read go back to the
model as an error, as they did there.

Reach this module by path. It needs the ``agents`` extra.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .corpus_tools import ToolSession
from .loop import AgentLoop
from .records import DEFAULT_INSTRUCTION, AgentRun, parse_answer


def session_loop(
    session: ToolSession,
    *,
    model: str,
    corpus_description: str = "a corpus of documents",
    instruction: str = DEFAULT_INSTRUCTION,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_prompt_tokens: int | None = 60_000,
    **loop_options: Any,
) -> AgentLoop:
    """Return the agent loop for one question's session, configured as its arm says.

    Parameters
    ----------
    session : ToolSession
        The tools and trace for one question.
    model : str
        The bare model name.
    corpus_description : str, optional
        What the instruction calls the corpus.
    instruction : str, optional
        The system instruction, with ``{corpus}`` filled from
        ``corpus_description``.
    max_tokens : int, optional
        Completion cap per call.
    temperature : float, optional
        Sampling temperature, for families that accept one.
    max_prompt_tokens : int or None, optional
        Make the model answer once a request passes this estimate, before the
        endpoint refuses an over-long request.
    **loop_options : Any
        Any other :class:`~aieng.advanced_rag.agents.loop.AgentLoop` argument,
        such as ``cache``, ``limiter``, ``charge_to``, or ``plugins``.

    Returns
    -------
    AgentLoop
        A loop with ``max_model_calls`` from the arm, ``on_limit="answer"``,
        and the read-first guard on when the arm asks for it and the session
        offers a tool that reads.
    """
    arm = session.arm
    read_first = (
        (lambda: session.opened_any)
        if arm.read_before_answer and session.offers_reading
        else None
    )
    return AgentLoop(
        session.tools,
        model=model,
        instruction=instruction.format(corpus=corpus_description),
        max_tokens=max_tokens,
        temperature=temperature,
        max_model_calls=arm.max_model_calls,
        on_limit="answer",
        max_prompt_tokens=max_prompt_tokens,
        read_first=read_first,
        **loop_options,
    )


async def arun_session(
    question: str, session: ToolSession, *, model: str, **options: Any
) -> AgentRun:
    """Answer one question with the session's tools, on ADK.

    Parameters
    ----------
    question : str
        The question, unmodified.
    session : ToolSession
        The tools and trace for this question, from
        :meth:`~aieng.advanced_rag.agents.corpus_tools.CorpusToolServer.session`.
    model : str
        The bare model name.
    **options : Any
        Passed to :func:`session_loop`.

    Returns
    -------
    AgentRun
        The answer, the cost, the trace, and the transcript.
    """
    answer = await session_loop(session, model=model, **options).arun(question)
    return AgentRun(
        answer=parse_answer(answer.final_text),
        final_text=answer.final_text,
        model=model,
        model_calls=answer.model_calls,
        tool_calls=answer.tools_run,
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
        seconds=answer.seconds,
        exhausted=answer.exhausted,
        nudged=answer.nudged,
        trace=session.trace.to_list(),
        read_ids=session.trace.read_ids,
        seen_ids=session.trace.seen_ids,
        transcript=list(answer.transcript),
    )


def run_session(
    question: str, session: ToolSession, *, model: str, **options: Any
) -> AgentRun:
    """Answer one question from synchronous code; see :func:`arun_session`.

    Parameters
    ----------
    question : str
        The question, unmodified.
    session : ToolSession
        The tools and trace for this question.
    model : str
        The bare model name.
    **options : Any
        Passed to :func:`session_loop`.

    Returns
    -------
    AgentRun
        What :func:`arun_session` returned.

    Raises
    ------
    RuntimeError
        When called from inside a running event loop. Await
        :func:`arun_session` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun_session(question, session, model=model, **options))
    raise RuntimeError(
        "run_session() cannot run inside an active event loop; await arun_session()"
    )


__all__ = ["arun_session", "run_session", "session_loop"]
