"""The agent loop, driven against a local endpoint that records what it is sent.

This is the free half of the thinking-toggle smoke test the plan asks for. It
asserts on the outgoing HTTP request rather than on a mock's call arguments,
because the chain being checked is four layers deep: ``AgentLoop`` to
``LiteLlm``, which forwards an unrecognised keyword argument to LiteLLM, which
copies it into ``extra_body``, which an OpenAI-compatible client merges into the
top level of the JSON body. A mock at any one of those layers would pass while
the proxy received nothing.

No key, no network, and no spend: the endpoint is a stub on the loopback
interface. The integration-marked twin in ``test_proxy_integration`` is what
proves the live proxy honours the same body.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest


REASON = "the agents extra is not installed: pip install 'aieng-advanced-rag[agents]'"

pytest.importorskip("google.adk", reason=REASON)

from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: E402
from aieng.advanced_rag.metering import UsageMeter  # noqa: E402


FAKE_KEY = "test-proxy-key-not-real"
QUESTION = "How many people live in Toronto?"
ANSWER = "Toronto has 2.8 million people."
PROMPT_TOKENS = 41
COMPLETION_TOKENS = 7


def lookup_population(city: str) -> dict[str, Any]:
    """Return the population of a city.

    Parameters
    ----------
    city : str
        The city to look up.

    Returns
    -------
    dict
        The city and its population.
    """
    return {"city": city, "population": 2_800_000}


def lookup_area(city: str) -> dict[str, Any]:
    """Return the area of a city in square kilometres.

    Parameters
    ----------
    city : str
        The city to look up.

    Returns
    -------
    dict
        The city and its area.
    """
    return {"city": city, "area_km2": 630}


class StubProxy:
    """An OpenAI-compatible endpoint on the loopback interface.

    Records every request it receives, answers the first turn with a tool call
    and the turn after the tool result with text, which is the two-call shape the
    real proxy returned when this was checked.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                """Keep the test output clean."""

            def do_POST(self):  # noqa: N802 - the handler API's own spelling
                """Record one request and answer it."""
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                payload = json.dumps(stub.reply(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return a tool call, or the answer once a tool result has arrived."""
        answered = any(
            message.get("role") == "tool" for message in body.get("messages", [])
        )
        if answered:
            message = {"role": "assistant", "content": ANSWER}
            finish = "stop"
        else:
            # A tool call comes back with null content, which is the shape the
            # existing completer would have read as an abstention.
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_population",
                            "arguments": json.dumps({"city": "Toronto"}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }

    def close(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def stub():
    proxy = StubProxy()
    try:
        yield proxy
    finally:
        proxy.close()


def build_loop(
    stub: StubProxy, model: str = "qwen3-35b-a3b", **kwargs: Any
) -> AgentLoop:
    return AgentLoop(
        [lookup_population, lookup_area],
        model=model,
        base_url=stub.base_url,
        api_key=FAKE_KEY,
        **kwargs,
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_the_thinking_flag_reaches_every_request_body(stub, thinking):
    build_loop(stub, enable_thinking=thinking).run(QUESTION)

    assert stub.requests, "the loop sent no request"
    for request in stub.requests:
        assert request["body"]["chat_template_kwargs"] == {"enable_thinking": thinking}


def test_claude_gets_no_temperature_and_its_own_thinking_switch(stub):
    build_loop(stub, model="claude-haiku-5-5").run(QUESTION)

    for request in stub.requests:
        body = request["body"]
        assert "temperature" not in body
        assert "chat_template_kwargs" not in body
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_completion_tokens"] == 512


def test_openai_gets_max_completion_tokens_and_no_vllm_switch(stub):
    build_loop(stub, model="gpt-6-luna").run(QUESTION)

    for request in stub.requests:
        body = request["body"]
        assert "max_tokens" not in body and body["max_completion_tokens"] == 512
        assert "temperature" not in body
        assert "chat_template_kwargs" not in body
        assert body["reasoning_effort"] == "none"


def test_gemini_keeps_the_temperature(stub):
    build_loop(stub, model="gemini-3.5-flash-lite").run(QUESTION)

    for request in stub.requests:
        body = request["body"]
        assert body["temperature"] == 0.0
        assert body["reasoning_effort"] == "minimal"
        assert "chat_template_kwargs" not in body


def test_the_loop_reaches_the_endpoint_it_was_given_with_the_key_as_a_bearer(stub):
    build_loop(stub).run(QUESTION)

    assert stub.requests[0]["path"] == "/v1/chat/completions"
    assert stub.requests[0]["authorization"] == f"Bearer {FAKE_KEY}"


def test_both_plain_functions_are_advertised_as_tools(stub):
    build_loop(stub).run(QUESTION)

    declared = stub.requests[0]["body"]["tools"]
    assert [tool["function"]["name"] for tool in declared] == [
        "lookup_population",
        "lookup_area",
    ]


def test_the_answer_carries_the_tool_the_model_called(stub):
    answer = build_loop(stub).run(QUESTION)

    assert answer.text == ANSWER
    assert answer.tool_calls == ("lookup_population",)
    assert answer.model_calls == 2
    assert answer.abstained is False


def test_a_tool_call_is_metered_as_a_model_call_and_not_an_abstention(stub):
    meter = UsageMeter()
    loop = build_loop(stub)

    with meter.question("q1"):
        answer = loop.run(QUESTION)

    cost = meter.per_question()["q1"]
    assert cost.calls == 2
    assert cost.tool_calls == 1
    assert cost.empty_responses == 0
    assert cost.prompt_tokens == 2 * PROMPT_TOKENS
    assert cost.completion_tokens == 2 * COMPLETION_TOKENS
    assert cost.tokens_complete is True
    assert answer.usage.total_tokens == cost.total_tokens


def test_a_meter_handed_over_is_charged_when_no_question_is_bound(stub):
    meter = UsageMeter()
    charged = meter.for_question("q2")

    build_loop(stub, charge_to=charged).run(QUESTION)

    assert meter.per_question()["q2"].calls == 2
    assert meter.unattributed().calls == 0


def test_the_old_toolset_name_still_imports():
    from aieng.advanced_rag.agents.toolset import FunctionToolset, PythonToolset

    assert PythonToolset is FunctionToolset
