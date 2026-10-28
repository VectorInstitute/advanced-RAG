"""The ADK response cache and question guard, against a loopback endpoint.

The endpoint is a scripted OpenAI-compatible stub, so every assertion is about
the HTTP request ADK and LiteLLM actually sent, or that nothing was sent. No
key, no network, and no spend.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest


pytest.importorskip("google.adk", reason="the agents extra is not installed")

from google.adk.agents.invocation_context import (
    LlmCallsLimitExceededError,  # noqa: E402
)
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.plugins.base_plugin import BasePlugin  # noqa: E402
from google.genai import types  # noqa: E402

from aieng.advanced_rag import metering  # noqa: E402
from aieng.advanced_rag.agents import model as model_module  # noqa: E402
from aieng.advanced_rag.agents.corpus_tools import (  # noqa: E402
    STANDARD_ARMS,
    Corpus,
    CorpusToolServer,
)
from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: E402
from aieng.advanced_rag.agents.plugins import request_key  # noqa: E402
from aieng.advanced_rag.agents.records import (  # noqa: E402
    BUDGET_MESSAGE,
    READ_FIRST_MESSAGE,
    AgentRun,
)
from aieng.advanced_rag.agents.session_loop import run_session  # noqa: E402
from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: E402
from aieng.advanced_rag.metering import UsageMeter  # noqa: E402


FAKE_KEY = "test-proxy-key-not-real"
MODEL = "Qwen3_6-35B-A3B"


def lookup(city: str) -> dict[str, Any]:
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


def broken(city: str) -> dict[str, Any]:
    """Return nothing, by raising.

    Parameters
    ----------
    city : str
        Ignored.

    Returns
    -------
    dict
        Never returned.
    """
    raise KeyError(city)


Reply = Callable[[dict[str, Any], int], dict[str, Any]]


def tool_call(name: str, arguments: dict[str, Any], call_id: str | None) -> dict:
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    if call_id is not None:
        call["id"] = call_id
    return call


def message(content: str | None = None, calls: list[dict] | None = None) -> dict:
    found: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        found["tool_calls"] = calls
    return found


def has_tool_result(body: dict[str, Any]) -> bool:
    return any(m.get("role") == "tool" for m in body.get("messages", []))


def ask_then_answer(call_id: str | None) -> Reply:
    """Ask for ``lookup`` once, then answer once a tool result is in."""

    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        if has_tool_result(body):
            return message("Answer: 2.8 million")
        return message(calls=[tool_call("lookup", {"city": "Toronto"}, call_id)])

    return reply


class StubProxy:
    """A loopback endpoint that answers from a reply function and records requests."""

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.fail_first: int = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                """Keep the test output clean."""

            def do_POST(self):  # noqa: N802 - the handler API's own spelling
                """Record one request and answer it."""
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if stub.fail_first > 0:
                    stub.fail_first -= 1
                    payload = json.dumps({"error": {"message": "slow down"}}).encode()
                    self.send_response(429)
                else:
                    stub.requests.append(body)
                    payload = json.dumps(stub.respond(body)).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        found = self.reply(body, len(self.requests) - 1)
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": found,
                    "finish_reason": "tool_calls"
                    if found.get("tool_calls")
                    else "stop",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46},
        }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_stub():
    stubs: list[StubProxy] = []

    def make(reply: Reply) -> StubProxy:
        stubs.append(StubProxy(reply))
        return stubs[-1]

    yield make
    for stub in stubs:
        stub.close()


@pytest.fixture
def cache(tmp_path) -> ResponseCache:
    return ResponseCache(tmp_path / "responses.sqlite")


def loop_on(stub: StubProxy, cache: ResponseCache | bool, **kwargs: Any) -> AgentLoop:
    options: dict[str, Any] = {"model": MODEL}
    options.update(kwargs)
    return AgentLoop(
        options.pop("tools", [lookup]),
        base_url=stub.base_url,
        api_key=FAKE_KEY,
        cache=cache,
        **options,
    )


@pytest.mark.parametrize("call_id", [None, "chatcmpl-tool-a1b2"])
def test_a_second_run_is_served_from_the_cache(make_stub, cache, call_id):
    # None makes ADK assign a random ``adk-`` id, which the key must not see.
    stub = make_stub(ask_then_answer(call_id))
    meter = UsageMeter()

    with meter.question("cold"):
        cold = loop_on(stub, cache).run("How many people live in Toronto?")
    sent = len(stub.requests)
    with meter.question("warm"):
        warm = loop_on(stub, cache).run("How many people live in Toronto?")

    assert sent == 2
    assert len(stub.requests) == sent
    assert warm.text == cold.text == "Answer: 2.8 million"
    assert warm.model_calls == cold.model_calls == 2
    assert warm.usage == cold.usage
    assert warm.cache_hits == 2 and cold.cache_hits == 0
    usage = meter.per_question()
    assert usage["cold"].calls == 2 and usage["cold"].cache_hits == 0
    assert usage["warm"].calls == 0 and usage["warm"].cache_hits == 2


def test_live_calls_are_metered_with_the_model_name(make_stub, cache):
    stub = make_stub(ask_then_answer("call_1"))
    meter = UsageMeter()
    with meter.question("q"):
        loop_on(stub, cache).run("q")

    totals = meter.per_question()["q"]
    assert totals.calls == 2 and totals.tool_calls == 1 and totals.empty_responses == 0
    assert totals.prompt_tokens == 80
    assert set(meter.by_model()) == {MODEL}


def test_a_meter_handed_over_is_charged_hits_and_calls(make_stub, cache):
    stub = make_stub(ask_then_answer("call_1"))
    meter = UsageMeter()
    question = meter.for_question("handed")
    loop_on(stub, cache, charge_to=question).run("q")
    loop_on(stub, cache, charge_to=question).run("q")

    totals = question.totals()
    assert totals.calls == 2 and totals.cache_hits == 2


def test_no_cache_sends_every_request(make_stub):
    stub = make_stub(ask_then_answer("call_1"))
    loop_on(stub, False).run("q")
    loop_on(stub, False).run("q")
    assert len(stub.requests) == 4


def test_request_key_ignores_labels_and_call_ids():
    def request(call_id: str, label: str) -> LlmRequest:
        return LlmRequest(
            model="openai/x",
            contents=[
                types.Content(role="user", parts=[types.Part(text="q")]),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id=call_id, name="lookup", args={"city": "T"}
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=call_id, name="lookup", response={"p": 1}
                            )
                        )
                    ],
                ),
            ],
            config=types.GenerateContentConfig(labels={"adk_agent_name": label}),
        )

    one = request("adk-1111", "a")
    assert request_key(one) == request_key(request("adk-2222", "b"))
    assert request_key(one) != request_key(one, salt={"thinking": True})


def always_tool(call_id: str | None = "call_1") -> Reply:
    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        if body.get("tool_choice") == "none":
            return message("Answer: 2.8 million")
        return message(calls=[tool_call("lookup", {"city": f"c{index}"}, call_id)])

    return reply


def test_the_last_allowed_call_is_forced_to_answer(make_stub, cache):
    stub = make_stub(always_tool())
    answer = loop_on(stub, cache, max_model_calls=3, on_limit="answer").run("q")

    assert len(stub.requests) == 3
    last = stub.requests[-1]
    assert last["tool_choice"] == "none"
    assert last["messages"][-1] == {"role": "user", "content": BUDGET_MESSAGE}
    assert all(
        "tool_choice" not in r or r["tool_choice"] != "none" for r in stub.requests[:2]
    )
    assert answer.exhausted and answer.final_text == "Answer: 2.8 million"
    assert answer.model_calls == 3 and answer.tools_run == 2

    again = loop_on(stub, cache, max_model_calls=3, on_limit="answer").run("q")
    assert len(stub.requests) == 3
    assert again.exhausted and again.cache_hits == 3
    assert again.transcript == answer.transcript


def test_tools_asked_for_by_the_final_call_are_not_run(make_stub, cache):
    ran: list[str] = []

    def counted(city: str) -> dict[str, Any]:
        """Return a city.

        Parameters
        ----------
        city : str
            The city.

        Returns
        -------
        dict
            The city.
        """
        ran.append(city)
        return {"city": city}

    def stubborn(body: dict[str, Any], index: int) -> dict[str, Any]:
        return message(
            "partial", calls=[tool_call("counted", {"city": f"c{index}"}, f"id{index}")]
        )

    stub = make_stub(stubborn)
    answer = loop_on(
        stub, cache, tools=[counted], max_model_calls=2, on_limit="answer"
    ).run("q")
    assert len(stub.requests) == 2
    assert ran == ["c0"]
    assert answer.final_text == "partial" and answer.model_calls == 2


def test_raise_mode_stops_a_replay_where_the_cold_run_stopped(make_stub, cache):
    stub = make_stub(always_tool())
    with pytest.raises(LlmCallsLimitExceededError):
        loop_on(stub, cache, max_model_calls=2).run("q")
    assert len(stub.requests) == 2
    with pytest.raises(LlmCallsLimitExceededError):
        loop_on(stub, cache, max_model_calls=2).run("q")
    assert len(stub.requests) == 2


def test_a_long_prompt_is_made_final(make_stub, cache):
    stub = make_stub(always_tool())
    answer = loop_on(
        stub, cache, max_model_calls=10, on_limit="answer", max_prompt_tokens=1
    ).run("q")
    assert len(stub.requests) == 1
    assert stub.requests[0]["tool_choice"] == "none"
    assert not answer.exhausted


def test_the_read_first_guard_asks_once(make_stub, cache):
    def answers_at_once(body: dict[str, Any], index: int) -> dict[str, Any]:
        return message(f"Answer: guess {index}")

    stub = make_stub(answers_at_once)
    answer = loop_on(stub, cache, read_first=lambda: False).run("q")

    assert len(stub.requests) == 2
    assert stub.requests[1]["messages"][-1]["content"] == READ_FIRST_MESSAGE
    assert answer.nudged and answer.final_text == "Answer: guess 1"

    again = loop_on(stub, cache, read_first=lambda: False).run("q")
    assert len(stub.requests) == 2 and again.nudged and again.cache_hits == 2


def test_a_tool_error_goes_back_to_the_model(make_stub, cache):
    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        tool = [m for m in body["messages"] if m.get("role") == "tool"]
        if tool:
            return message(f"Answer: saw {json.loads(tool[0]['content'])['error'][:6]}")
        return message(calls=[tool_call("broken", {"city": "T"}, "c1")])

    stub = make_stub(reply)
    answer = loop_on(stub, cache, tools=[broken]).run("q")
    assert answer.text == "Answer: saw broken"
    assert answer.tools_run == 1


def test_the_spend_guard_stops_a_live_call_and_not_a_hit(make_stub, cache):
    stub = make_stub(ask_then_answer("call_1"))
    loop_on(stub, cache).run("q")

    class StopError(Exception):
        pass

    def refuse(model: str | None) -> None:
        raise StopError(model)

    metering.add_pre_call_hook(refuse)
    try:
        assert loop_on(stub, cache).run("q").cache_hits == 2
        with pytest.raises(StopError, match=MODEL):
            loop_on(stub, cache).run("a question not in the cache")
    finally:
        metering.remove_pre_call_hook(refuse)
    assert len(stub.requests) == 2


def test_a_rate_limited_call_is_retried(make_stub, cache, monkeypatch):
    monkeypatch.setattr(model_module, "backoff_seconds", lambda attempt: 0.0)
    stub = make_stub(ask_then_answer("call_1"))
    stub.fail_first = 2
    answer = loop_on(stub, cache).run("q")
    assert answer.text == "Answer: 2.8 million"
    assert len(stub.requests) == 2


def test_a_paid_model_request_goes_through_adapt_request(make_stub, cache):
    stub = make_stub(ask_then_answer("call_1"))
    loop_on(
        stub,
        cache,
        model="claude-haiku-5-5",
        agent_options={
            "generate_content_config": types.GenerateContentConfig(
                temperature=0.7, max_output_tokens=64
            )
        },
    ).run("q")
    for body in stub.requests:
        assert "temperature" not in body
        assert "chat_template_kwargs" not in body
        assert "max_retries" not in body


def test_caller_plugins_and_callbacks_run_alongside(make_stub, cache):
    seen: list[str] = []

    class Watch(BasePlugin):
        async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
            seen.append(f"plugin:{tool.name}")

    def before_model(callback_context, llm_request):
        seen.append("callback")

    stub = make_stub(ask_then_answer("call_1"))
    loop_on(
        stub,
        cache,
        plugins=[Watch(name="watch")],
        agent_options={"before_model_callback": before_model},
    ).run("q")
    assert seen == ["callback", "plugin:lookup", "callback"]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str


CORPUS = Corpus.from_passages(
    [
        Passage("p0", "Lothair II", "Lothair II's mother was Ermengarde of Tours."),
        Passage("p1", "Ermengarde of Tours", "Ermengarde of Tours died in 851."),
    ]
)


def test_a_session_run_returns_tool_loops_record(make_stub, cache):
    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        if index == 0:
            return message(calls=[tool_call("grep", {"pattern": "Lothair"}, "a")])
        if index == 1:
            return message(calls=[tool_call("open", {"doc_id": "p0"}, "b")])
        return message("Answer: Ermengarde of Tours")

    stub = make_stub(reply)
    server = CorpusToolServer(CORPUS)

    def run() -> AgentRun:
        return run_session(
            "Who was Lothair II's mother?",
            server.session(STANDARD_ARMS["C5"]),
            model=MODEL,
            corpus_description="test passages",
            base_url=stub.base_url,
            api_key=FAKE_KEY,
            cache=cache,
        )

    cold = run()
    warm = run()
    assert len(stub.requests) == 3
    assert cold.answer == "Ermengarde of Tours"
    assert cold.model_calls == 3 and cold.tool_calls == 2
    assert [e["name"] for e in cold.trace] == ["grep", "open"]
    assert cold.read_ids == ["p0"]
    assert "test passages" in cold.transcript[0]["content"]
    roles = [m["role"] for m in cold.transcript]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]

    def timeless(run: AgentRun) -> dict[str, Any]:
        record = run.to_dict()
        record["seconds"] = 0
        record["trace"] = [{**event, "seconds": 0} for event in record["trace"]]
        return record

    assert timeless(warm) == timeless(cold)
    assert AgentRun.from_dict(json.loads(json.dumps(cold.to_dict()))) == cold


def test_threads_share_one_cache_and_limiter(make_stub, cache):
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from aieng.advanced_rag.agents.records import MeasuredLimiter  # noqa: PLC0415

    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        if has_tool_result(body):
            return message("Answer: done")
        question = body["messages"][1]["content"]
        return message(calls=[tool_call("lookup", {"city": question}, f"id-{index}")])

    stub = make_stub(reply)
    limiter = MeasuredLimiter(per_minute=1000)
    meter = UsageMeter()

    def one(index: int) -> str:
        with meter.question(f"q{index}"):
            return loop_on(stub, cache, limiter=limiter).run(f"city {index}").text

    for _ in range(2):
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(one, range(8))) == ["Answer: done"] * 8
    assert len(stub.requests) == 16
    totals = meter.totals()
    assert totals.calls == 16 and totals.cache_hits == 16


def test_unreadable_arguments_go_back_to_the_model_as_an_error(make_stub, cache):
    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        tool = [m for m in body["messages"] if m.get("role") == "tool"]
        if tool:
            return message(f"Answer: {json.loads(tool[0]['content'])['error'][:24]}")
        call = tool_call("lookup", {}, "c1")
        call["function"]["arguments"] = '{"city": "Toronto"'
        return message(calls=[call])

    stub = make_stub(reply)
    answer = loop_on(stub, cache).run("q")
    assert answer.text == "Answer: could not read the argum"
    assert answer.tools_run == 1
    assert loop_on(stub, cache).run("q").text == answer.text
    assert len(stub.requests) == 2


def test_the_key_and_endpoint_can_come_from_a_named_env_file(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "elsewhere.env"
    env.write_text(
        f"VECTOR_PROXY_API_KEY={FAKE_KEY}\nOPENAI_BASE_URL=http://192.0.2.1/v1\n"
    )
    loop = AgentLoop([lookup], env_file=env, cache=False)
    assert loop.api_key == FAKE_KEY
    assert loop.base_url == "http://192.0.2.1/v1"


def test_tool_declarations_match_what_the_agent_sends(make_stub, cache):
    from aieng.advanced_rag.agents.toolset import tool_declarations  # noqa: PLC0415

    declared = tool_declarations([lookup, broken])
    assert [d["name"] for d in declared] == ["lookup", "broken"]
    stub = make_stub(ask_then_answer("call_1"))
    loop_on(stub, cache, tools=[lookup, broken]).run("q")
    sent = [tool["function"]["name"] for tool in stub.requests[0]["tools"]]
    assert sent == ["lookup", "broken"]
    assert declared == tool_declarations([lookup, broken])
