"""What an agent run over a tool session records.

:class:`AgentRun` is the record the RI #5 arms and the mixed-source router
store for each question, and :func:`parse_answer` reads the short answer out
of a final reply. :mod:`aieng.advanced_rag.agents.session_loop` produces it,
with the fields the earlier chat-completions loop wrote, so records stored by
either read the same.

Nothing here imports ADK.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graphrag.proxy import RateLimiter


#: What the agent is told. The corpus description is filled in by the caller.
DEFAULT_INSTRUCTION = """You answer questions from {corpus} using the tools.
A question can need several steps: find the first entity, read about it, then look up the next
one. Read the documents your answer relies on. When you know the answer, reply with one line:
Answer: <the answer, a short phrase of a few words>
If the documents do not contain it, reply: Answer: unknown"""

#: Sent with the last call of a run that reached its budget.
BUDGET_MESSAGE = (
    "You have used your tool budget. Answer now from what you have read, in the form "
    "'Answer: <short answer>'."
)

#: Sent once when the model answers without opening anything.
READ_FIRST_MESSAGE = (
    "Before you answer, open the documents your answer relies on with the open tool, "
    "then answer in the form 'Answer: <short answer>'."
)

_ANSWER = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class MeasuredLimiter(RateLimiter):
    """A rate limiter that records how long each thread waited.

    Seconds per question are reported without the time spent waiting for the
    shared rate limit, because that wait depends on what else runs on the key,
    not on the arm. :meth:`aacquire` records too, so the ADK path, which waits
    inside an event loop, reports the same way.
    """

    def __init__(self, per_minute: int = 40, window: float = 60.0) -> None:
        super().__init__(per_minute=per_minute, window=window)
        self._local = threading.local()

    def acquire(self) -> None:
        """Block until the request fits, adding the wait to this thread's total."""
        started = time.perf_counter()
        super().acquire()
        self._local.waited = self.waited() + time.perf_counter() - started

    async def aacquire(self) -> None:
        """Await until the request fits, adding the wait to this thread's total."""
        started = time.perf_counter()
        await super().aacquire()
        self._local.waited = self.waited() + time.perf_counter() - started

    def waited(self) -> float:
        """Return this thread's total wait, in seconds."""
        return float(getattr(self._local, "waited", 0.0))


@dataclass(frozen=True)
class AgentRun:
    """One question's run: the answer, its cost, and every tool call.

    Parameters
    ----------
    answer : str
        The short answer parsed from the final reply, or ``""``.
    final_text : str
        The final reply as the model wrote it.
    model : str
        The model requested.
    model_calls : int
        Responses the run produced, the forced final answer included. A
        response served from the response cache counts, so a replay reports
        the same figure as the run it repeats.
    tool_calls : int
        Tool calls executed.
    prompt_tokens, completion_tokens : int
        Summed over the run's calls, as the endpoint reported them.
    seconds : float
        Model and tool time, without rate-limit waits.
    exhausted : bool
        Whether the run reached its call budget and was made to answer.
    nudged : bool
        Whether the read-before-answer guard fired.
    trace : list of dict
        :meth:`~aieng.advanced_rag.agents.corpus_tools.ToolTrace.to_list`.
    read_ids, seen_ids : list of str
        Documents read in full, and documents named in any result.
    transcript : list of dict
        The messages sent and received, in chat-completions form. Holds corpus
        text, so it is kept in the artifact cache and never committed.
    """

    answer: str
    final_text: str
    model: str
    model_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    exhausted: bool
    nudged: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    read_ids: list[str] = field(default_factory=list)
    seen_ids: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> AgentRun:
        """Read a record written by :meth:`to_dict`."""
        return cls(**dict(record))


def parse_answer(text: str) -> str:
    """Return the short answer from a final reply.

    The last ``Answer:`` line wins. A reply with none returns its last
    non-empty line, so a model that forgot the prefix still scores.
    """
    clean = _THINK.sub("", text or "").strip()
    found = _ANSWER.findall(clean)
    if found:
        return str(found[-1]).strip().strip("*").strip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return lines[-1] if lines else ""


__all__ = [
    "BUDGET_MESSAGE",
    "DEFAULT_INSTRUCTION",
    "READ_FIRST_MESSAGE",
    "AgentRun",
    "MeasuredLimiter",
    "parse_answer",
]
