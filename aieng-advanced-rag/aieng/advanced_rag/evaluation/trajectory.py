"""Score an agent run on its trajectory and its answer, and record both in Langfuse.

An agent's final answer says little about how it got there. Two runs that both
answer correctly can differ tenfold in model calls, and one can reach the answer
without calling the retrieval tool at all. The scores here cover the path:
how many tool calls and model calls the run took, what it cost, whether the tools
it was expected to use appear in order, and whether a judge finds the answer
correct.

The judge sees no retrieved evidence, because the run carries none, and on an empty
evidence block it marks an abstention correct against a specific reference. A guard in
:func:`score_agent_run` overrides that, so ``correct`` is false for an abstention unless
the reference itself says the information is absent.

What this does not cover. The trajectory is the tool names and counts that
:class:`~aieng.advanced_rag.agents.loop.AgentLoop` returns. It holds no tool
arguments or results, and the run's individual model calls are not spans in the
Langfuse trace unless an ADK instrumentor is installed, so a trace shows one
root observation with scores and metadata attached. Grounding is not scored,
because the run carries no retrieved evidence to judge it against.

Nothing here imports ``agents``, so the module works with any loop that returns
an object with the same fields.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..pipeline.generate import looks_like_abstention
from ..tracing.langfuse import langfuse_trace
from ..tracing.scores import record_scores
from .answers import AnswerJudgement, judge_answer


class AgentRun(Protocol):
    """The fields of an agent run that the scores read."""

    text: str
    tool_calls: Sequence[str]
    model_calls: int
    elapsed_seconds: float

    @property
    def abstained(self) -> bool:
        """Whether the run ended without an answer."""
        ...

    @property
    def usage(self) -> Any:
        """Token counts, with ``prompt_tokens`` and ``completion_tokens``."""
        ...


class Loop(Protocol):
    """Anything that answers one question with an :class:`AgentRun`."""

    def run(self, question: str) -> AgentRun:
        """Answer one question."""
        ...


def tools_in_order(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Return the share of expected tools that appear in order among the calls.

    Parameters
    ----------
    expected : sequence of str
        Tool names the run should have used, in the order it should have used
        them. Other calls may fall between them.
    actual : sequence of str
        Tool names the run called.

    Returns
    -------
    float
        The longest prefix of ``expected`` found as a subsequence of ``actual``,
        divided by the length of ``expected``. 1.0 when ``expected`` is empty.
    """
    if not expected:
        return 1.0
    matched = 0
    for name in actual:
        if matched < len(expected) and name == expected[matched]:
            matched += 1
    return matched / len(expected)


def trajectory_scores(
    run: AgentRun, *, expected_tools: Sequence[str] = ()
) -> dict[str, float]:
    """Return the numbers that describe how a run got to its answer.

    Parameters
    ----------
    run : AgentRun
        The finished run.
    expected_tools : sequence of str, optional
        Tools the run should have used, in order. Adds ``tools_in_order``.

    Returns
    -------
    dict
        ``tool_calls``, ``distinct_tools``, ``repeated_tool_calls`` (calls past
        the first to a tool the run already used), ``model_calls``,
        ``prompt_tokens``, ``completion_tokens``, ``elapsed_seconds``, and
        ``abstained``, plus ``tools_in_order`` when ``expected_tools`` is given.
    """
    calls = list(run.tool_calls)
    scores: dict[str, float] = {
        "tool_calls": float(len(calls)),
        "distinct_tools": float(len(set(calls))),
        "repeated_tool_calls": float(len(calls) - len(set(calls))),
        "model_calls": float(run.model_calls),
        "prompt_tokens": float(run.usage.prompt_tokens or 0),
        "completion_tokens": float(run.usage.completion_tokens or 0),
        "elapsed_seconds": round(float(run.elapsed_seconds), 3),
        "abstained": float(run.abstained),
    }
    if expected_tools:
        scores["tools_in_order"] = tools_in_order(expected_tools, calls)
    return scores


@dataclass(frozen=True)
class ScoredRun:
    """One agent run, its scores, and where its trace is.

    Parameters
    ----------
    run : AgentRun
        What the loop returned.
    scores : dict
        Every score computed, trajectory and judge together.
    judgement : AnswerJudgement or None
        The judge's verdict, None when no reference was given.
    trace_id : str or None
        The Langfuse trace, None when nothing was traced.
    trace_url : str or None
        A link to it.
    sent : int
        How many scores reached Langfuse.
    """

    run: AgentRun
    scores: dict[str, float]
    judgement: AnswerJudgement | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    sent: int = 0


def _correct(judgement: AnswerJudgement, reference: str) -> bool:
    """Return the judge's verdict, except that an abstention cannot be correct.

    The run carries no retrieved evidence, so the judge sees an empty evidence
    block. On that input the judge reads an abstention as consistent with the
    evidence and marks it correct against a reference that states a specific
    answer, with a reason such as "matching the reference's implication of no
    content". Shown real evidence, it marks the same abstention incorrect. An
    abstention is correct only when the reference itself says the information is
    absent.
    """
    if (
        judgement.abstained
        and reference.strip()
        and not looks_like_abstention(reference)
    ):
        return False
    return judgement.correct


def score_agent_run(
    loop: Loop,
    question: str,
    *,
    name: str,
    query_id: str = "q",
    reference: str | None = None,
    complete: Callable[[str], str] | None = None,
    expected_tools: Sequence[str] = (),
    tags: Sequence[str] = ("advanced-rag",),
) -> ScoredRun:
    """Run an agent once inside a Langfuse trace, then score and record it.

    Parameters
    ----------
    loop : Loop
        The agent loop to run.
    question : str
        The question to ask.
    name : str
        The trace name.
    query_id : str, optional
        Identifier passed to the judge.
    reference : str, optional
        The ground-truth answer. With ``complete`` it enables the judge's
        ``correct`` score.
    complete : callable, optional
        Takes a prompt, returns text. The judge's model access, which must be
        the proxy's chat endpoint and never a paid embedding API.
    expected_tools : sequence of str, optional
        Tools the run should have used, in order.
    tags : sequence of str, optional
        Langfuse tags.

    Returns
    -------
    ScoredRun
        The run, every score, and the trace link when one was recorded.
    """
    with langfuse_trace(
        name, tags=list(tags), metadata={"query_id": query_id}
    ) as trace:
        run = loop.run(question)

    scores = trajectory_scores(run, expected_tools=expected_tools)
    judgement = None
    comment = None
    if reference is not None and complete is not None:
        judgement = judge_answer(query_id, question, run.text, reference, (), complete)
        scores["correct"] = float(_correct(judgement, reference))
        comment = judgement.reason or None

    sent = record_scores(trace.trace_id, scores, comment=comment)
    return ScoredRun(
        run=run,
        scores=scores,
        judgement=judgement,
        trace_id=trace.trace_id,
        trace_url=trace.trace_url,
        sent=sent,
    )


__all__ = [
    "AgentRun",
    "Loop",
    "ScoredRun",
    "score_agent_run",
    "tools_in_order",
    "trajectory_scores",
]
