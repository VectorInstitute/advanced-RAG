"""The agent loop against the live proxy. Marked, because it spends tokens.

The stub twin in ``test_loop`` proves the thinking switch reaches the request
body. What only the real endpoint can show is the half that is server side: that
it accepts the body, emits an OpenAI-compatible tool call for a question it
cannot answer on its own, takes the tool result back, and reports token counts
the meter can read.

Run it by hand, once:

    pytest -m integration_test tests/aieng/advanced_rag/agents

It issues its model calls one at a time. Nothing here fans out, which keeps it
well under the 8 concurrent requests above which this proxy's throughput
collapses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


REASON = "the agents extra is not installed: pip install 'aieng-advanced-rag[agents]'"

pytest.importorskip("google.adk", reason=REASON)

from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: E402
from aieng.advanced_rag.metering import UsageMeter  # noqa: E402
from aieng.advanced_rag.proxy import MissingProxyKeyError, resolve_api_key  # noqa: E402


pytestmark = pytest.mark.integration_test

CLAUSE = "Either party may terminate this agreement on ninety days written notice."
QUESTION = "What notice does clause 7.2 require for termination?"


def proxy_key() -> str:
    """Return the proxy key, or skip.

    The lookup is the package's own, the environment and then the repository
    ``.env``. This test skips rather than fails because it is a hand-run check of
    the loop; the onboarding gate at the repository root's
    ``tests/test_integration.py`` is the one that fails on a missing key.
    """
    try:
        return resolve_api_key(start=Path(__file__).parent)
    except MissingProxyKeyError:
        pytest.skip("no proxy key in the environment or the repository .env")


def read_clause(clause: str) -> dict[str, Any]:
    """Return the text of one numbered clause of the supply agreement.

    Parameters
    ----------
    clause : str
        The clause number, such as ``7.2``.

    Returns
    -------
    dict
        The clause number and its text.
    """
    return {"clause": clause, "text": CLAUSE}


def test_the_loop_answers_through_the_proxy_and_the_meter_reads_the_cost():
    meter = UsageMeter()
    loop = AgentLoop(
        [read_clause],
        api_key=proxy_key(),
        enable_thinking=False,
        max_tokens=256,
        max_model_calls=4,
    )

    with meter.question("integration"):
        answer = loop.run(QUESTION)

    cost = meter.per_question()["integration"]
    assert answer.model_calls >= 1
    assert cost.calls == answer.model_calls
    # The proxy returns prompt, completion, and total counts on every response,
    # so a run that reports nothing means the usage plumbing broke rather than
    # that the model was cheap.
    assert cost.tokens_complete is True
    assert cost.total_tokens > 0
    assert answer.usage.reported is True
    # A question about a clause the model cannot know is what makes the tool
    # call the expected path rather than a coin toss.
    assert "read_clause" in answer.tool_calls
    assert answer.abstained is False
