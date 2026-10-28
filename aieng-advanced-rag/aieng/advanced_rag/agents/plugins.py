"""ADK plugins that cache model responses and hold a question to its budget.

Two plugins, registered on the runner in this order:

1. :class:`QuestionGuardPlugin` counts every model response a question
   produces, whether the model or the cache gave it, and enforces the call
   cap, the prompt-size cap, and the forced final answer. It also records the
   transcript and the run's time.
2. :class:`ResponseCachePlugin` answers a request it has seen from the
   package's :class:`~aieng.advanced_rag.graphrag.proxy.ResponseCache`, and
   stores the response to one it has not.

The order matters because ADK runs plugin callbacks in registration order and
stops at the first that returns a value. The guard returns nothing from
``before_model_callback``, so it always sees the request and can change it
before the cache keys it. A cache hit returns a response, so ADK sends nothing
and calls no ``after_model_callback``; the guard reads every response, served
or sent, from ``on_event_callback`` instead.

Reach this module by path. It needs the ``agents`` extra.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import metering
from ..graphrag.protocol import MissingExtraError
from ..graphrag.proxy import ResponseCache
from ..metering import CallUsage, QuestionMeter
from .model import UNREADABLE_ARGUMENTS, bare_model
from .records import BUDGET_MESSAGE
from .usage import usage_of_event


try:
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types
except ImportError as error:  # pragma: no cover - import guard
    raise MissingExtraError("agents", "google-adk") from error


#: Request fields that change from run to run without changing the answer.
#: ``tools_dict`` holds the tool objects, which ``config.tools`` declares.
_VOLATILE_REQUEST = frozenset(
    {
        "live_connect_config",
        "tools_dict",
        "cache_config",
        "cache_metadata",
        "cacheable_contents_token_count",
        "previous_interaction_id",
    }
)
#: ``labels`` carries ADK's agent-name billing label and ``http_options`` the
#: transport settings; neither reaches the model as content.
_VOLATILE_CONFIG = ("labels", "http_options")

#: Marks the response the guard returns in place of a call after the final one.
_SYNTHETIC = "aieng_guard_final"


def _renumber_call_ids(contents: list[Any]) -> None:
    """Replace every tool-call id in serialised contents with its position.

    ADK gives a tool call that came back without an id a random
    ``adk-<uuid>``, and LiteLLM-backed models send that id back on the next
    turn, so the next request differs on every run. A provider's own id is
    stable on a replay, because it comes back inside the cached response, but
    renumbering both keeps one rule. The pairing between a call and its result
    survives, because the same id gets the same number.
    """
    numbers: dict[str, str] = {}
    for content in contents:
        for part in content.get("parts") or []:
            for kind in ("function_call", "function_response"):
                record = part.get(kind)
                if isinstance(record, dict) and record.get("id"):
                    original = str(record["id"])
                    numbers.setdefault(original, f"call-{len(numbers)}")
                    record["id"] = numbers[original]


def request_key(llm_request: Any, salt: Mapping[str, Any] | None = None) -> str:
    """Return the response-cache key for one ADK model request.

    The key covers the model name, the contents, the system instruction, the
    tool declarations, and the generation settings, and leaves out what
    changes between runs of the same question: ``config.labels``,
    ``config.http_options``, the request's caching and interaction handles, and
    the tool-call ids, which are renumbered in order of appearance.

    Parameters
    ----------
    llm_request : LlmRequest
        The request as the ``before_model_callback`` sees it.
    salt : Mapping or None, optional
        Settings that change the response but never reach the request object,
        such as the ``extra_body`` a ``LiteLlm`` adds. Two agents that differ
        only there must pass different salts.

    Returns
    -------
    str
        A SHA-256 hex digest, in the key space of
        :class:`~aieng.advanced_rag.graphrag.proxy.ResponseCache`, tagged so it
        cannot collide with a chat-completions key.
    """
    data = llm_request.model_dump(
        mode="json", exclude_none=True, exclude=set(_VOLATILE_REQUEST)
    )
    config = data.get("config") or {}
    for name in _VOLATILE_CONFIG:
        config.pop(name, None)
    _renumber_call_ids(data.get("contents") or [])
    return ResponseCache.key(
        {"kind": "adk-llm-request", "request": data, "salt": dict(salt or {})}
    )


def _usage_figures(response: Any) -> CallUsage:
    """Return a response's token counts, or an unreported zero."""
    return usage_of_event(response) or CallUsage()


class ResponseCachePlugin(BasePlugin):
    """Serve model responses from the package's response cache.

    A request seen before is answered from the cache and never sent; ADK does
    not count it against ``RunConfig.max_llm_calls`` either. The hit is
    reported to :mod:`aieng.advanced_rag.metering` with the tokens the stored
    response cost, as :func:`~aieng.advanced_rag.graphrag.proxy.govern` reports
    one. A response from the model is stored unless it is a streamed partial.

    Parameters
    ----------
    cache : ResponseCache
        Where responses live. Share one file across runs to replay them.
    salt : Mapping or None, optional
        Passed to :func:`request_key`.
        :class:`~aieng.advanced_rag.agents.loop.AgentLoop` passes the model's
        ``extra_body``.
    charge_to : QuestionMeter or None, optional
        Charge hits to this question rather than the one bound to the context.
    name : str, optional
        The plugin's name, unique within a runner.

    Attributes
    ----------
    hits : int
        Responses served from the cache.
    stored : int
        Responses written to it.
    """

    def __init__(
        self,
        cache: ResponseCache,
        *,
        salt: Mapping[str, Any] | None = None,
        charge_to: QuestionMeter | None = None,
        name: str = "response_cache",
    ) -> None:
        super().__init__(name=name)
        self.cache = cache
        self.salt = dict(salt or {})
        self.charge_to = charge_to
        self.hits = 0
        self.stored = 0
        self._pending: dict[str, tuple[str, str]] = {}

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> Any:
        """Return the stored response for this request, or None to send it."""
        model = bare_model(str(llm_request.model or ""))
        key = request_key(llm_request, self.salt)
        entry = self.cache.entry(key)
        if entry is None:
            self._pending[callback_context.invocation_id] = (key, model)
            return None
        self.hits += 1
        if self.charge_to is None:
            metering.report_cache_hit(model=entry.model or model, usage=entry.usage)
        else:
            self.charge_to.record_cache_hit(
                model=entry.model or model, usage=entry.usage
            )
        return LlmResponse.model_validate_json(entry.payload)

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> Any:
        """Store a response from the model under the key its request had."""
        pending = self._pending.pop(callback_context.invocation_id, None)
        if pending is None or llm_response.partial:
            return None
        key, model = pending
        usage = _usage_figures(llm_response)
        self.cache.put(
            key,
            llm_response.model_dump_json(exclude_none=True),
            usage.prompt_tokens,
            usage.completion_tokens,
            model=model or None,
            response_model=llm_response.model_version,
        )
        self.stored += 1
        return None


@dataclass
class GuardState:
    """What the guard knows about one session, across its invocations.

    Parameters
    ----------
    model_calls : int
        Responses produced, served or sent.
    tool_calls : int
        Tools executed.
    final_sent : bool
        Whether the forced final call has been made.
    exhausted : bool
        Whether the budget message was sent.
    final_text : str
        The text of the last response that asked for no tool, or of the
        forced final response.
    prompt_tokens, completion_tokens : int
        Summed over every response, served or sent.
    seconds : float
        Model and tool time, without rate-limit waits.
    system : str
        The system instruction of the last request.
    request : list of Content
        The contents of the last request.
    response : Content or None
        The last response.
    """

    model_calls: int = 0
    tool_calls: int = 0
    final_sent: bool = False
    exhausted: bool = False
    final_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    system: str = ""
    request: list[Any] = field(default_factory=list)
    response: Any = None
    _model_started: float | None = None
    _wait_mark: float = 0.0
    _tool_started: dict[str, float] = field(default_factory=dict)

    def transcript(self) -> list[dict[str, Any]]:
        """Return the last request and its response as chat-completions messages."""
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for content in self.request:
            messages.extend(content_messages(content))
        if self.response is not None:
            messages.extend(content_messages(self.response))
        return messages


def _text_of(content: Any) -> str:
    """Return a content's answer text, leaving out thought parts."""
    parts = getattr(content, "parts", None) or []
    return "".join(
        part.text for part in parts if part.text and not getattr(part, "thought", False)
    )


def content_messages(content: Any) -> list[dict[str, Any]]:
    """Return one ADK content as the chat-completions messages it stands for.

    A model turn becomes one assistant message with its text and tool calls. A
    turn of tool results becomes one ``tool`` message per result, and a user
    turn one user message. Thought parts are left out.

    Parameters
    ----------
    content : google.genai.types.Content
        One turn.

    Returns
    -------
    list of dict
        Zero or more messages.
    """
    parts = getattr(content, "parts", None) or []
    results = [part.function_response for part in parts if part.function_response]
    if results:
        return [
            {
                "role": "tool",
                "tool_call_id": result.id or "",
                "content": json.dumps(result.response, ensure_ascii=False, default=str),
            }
            for result in results
        ]
    text = _text_of(content)
    if content.role == "model":
        calls = [
            {
                "id": part.function_call.id or "",
                "type": "function",
                "function": {
                    "name": part.function_call.name,
                    "arguments": json.dumps(
                        part.function_call.args or {}, ensure_ascii=False
                    ),
                },
            }
            for part in parts
            if part.function_call
        ]
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = calls
        return [message]
    return [{"role": "user", "content": text}]


def _instruction_text(llm_request: Any) -> str:
    """Return the system instruction of a request as plain text."""
    instruction = getattr(llm_request.config, "system_instruction", None)
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    return _text_of(instruction)


def _ends_with_tool_results(contents: list[Any]) -> bool:
    """Return whether the last turn of a request is tool results."""
    if not contents:
        return False
    return any(part.function_response for part in contents[-1].parts or [])


class QuestionGuardPlugin(BasePlugin):
    """Hold each question to a model-call budget and record its transcript.

    Every response counts, whether the model or a cache gave it, so a replay
    stops where the run it repeats stopped. ADK's own ``max_llm_calls`` counts
    only calls that reach the model.

    With ``on_limit="raise"`` the call after the last allowed one raises
    ``LlmCallsLimitExceededError``, as ADK's own limit does. With
    ``on_limit="answer"`` the last allowed call is made with tools switched off
    (``tool_choice="none"``) and, when the turn before it was a tool result,
    with ``budget_message`` added as a user turn, so the run still answers.
    A request whose estimated size passes ``max_prompt_tokens`` is made final
    the same way, before the endpoint refuses it. A tool the model asks for
    after the final call is not run.

    A tool that raises returns ``{"error": ...}`` to the model instead of
    ending the run, and so does a tool call whose arguments
    :class:`~aieng.advanced_rag.agents.model.GovernedLlmClient` could not
    parse.

    Parameters
    ----------
    max_model_calls : int
        Responses allowed per session.
    on_limit : {"raise", "answer"}, optional
        What the cap does.
    max_prompt_tokens : int or None, optional
        Make the call final once the request passes this estimate, at four
        characters a token over its serialised contents and instruction.
    budget_message : str, optional
        The user turn added to a forced final call after a tool result.
    wait_clock : callable or None, optional
        Returns the seconds the model client has spent waiting for its rate
        limiter, such as ``lambda: client.waited``. Those seconds come off the
        recorded time.
    name : str, optional
        The plugin's name, unique within a runner.

    Raises
    ------
    LlmCallsLimitExceededError
        From the model call past the cap, with ``on_limit="raise"``. ADK wraps
        an error raised in a plugin in a ``RuntimeError``;
        :class:`~aieng.advanced_rag.agents.loop.AgentLoop` unwraps it.
    """

    def __init__(
        self,
        *,
        max_model_calls: int,
        on_limit: Literal["raise", "answer"] = "raise",
        max_prompt_tokens: int | None = None,
        budget_message: str = BUDGET_MESSAGE,
        wait_clock: Callable[[], float] | None = None,
        name: str = "question_guard",
    ) -> None:
        if on_limit not in ("raise", "answer"):
            raise ValueError(f"on_limit must be 'raise' or 'answer', not {on_limit!r}")
        super().__init__(name=name)
        self.max_model_calls = max_model_calls
        self.on_limit = on_limit
        self.max_prompt_tokens = max_prompt_tokens
        self.budget_message = budget_message
        self.wait_clock = wait_clock or (lambda: 0.0)
        self._states: dict[str, GuardState] = {}

    def state(self, session_id: str) -> GuardState:
        """Return the guard's record of one session.

        Parameters
        ----------
        session_id : str
            The ADK session id.

        Returns
        -------
        GuardState
            A fresh record when the session has made no call.
        """
        return self._states.setdefault(session_id, GuardState())

    def _estimate(self, llm_request: Any) -> int:
        """Return a request's size in tokens, at four characters a token."""
        contents = [content.model_dump(mode="json") for content in llm_request.contents]
        size = len(json.dumps(contents, default=str))
        return (size + len(_instruction_text(llm_request))) // 4

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> Any:
        """Count the call, and make it final at the cap or the size limit."""
        state = self.state(callback_context.session.id)
        if state.final_sent:
            # The forced final response asked for tools anyway. They were not
            # run; end on its text without another call, as the earlier loop did.
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=state.final_text)]
                ),
                custom_metadata={_SYNTHETIC: True},
            )
        if state.model_calls >= self.max_model_calls:
            raise LlmCallsLimitExceededError(
                f"Max number of llm calls limit of {self.max_model_calls} exceeded"
            )
        state.model_calls += 1
        last = self.on_limit == "answer" and state.model_calls == self.max_model_calls
        long = (
            self.max_prompt_tokens is not None
            and self._estimate(llm_request) > self.max_prompt_tokens
        )
        if last or long:
            state.final_sent = True
            if llm_request.config.tools:
                llm_request.config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                )
            if state.model_calls > 1 and _ends_with_tool_results(llm_request.contents):
                state.exhausted = True
                llm_request.contents.append(
                    types.Content(
                        role="user", parts=[types.Part(text=self.budget_message)]
                    )
                )
        state.system = _instruction_text(llm_request)
        state.request = list(llm_request.contents)
        state._model_started = time.perf_counter()
        state._wait_mark = self.wait_clock()
        return None

    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> Any:
        """Read each model response, whether the model or the cache gave it."""
        content = event.content
        if content is None or content.role != "model" or event.partial:
            return None
        if (event.custom_metadata or {}).get(_SYNTHETIC):
            return None
        state = self.state(invocation_context.session.id)
        if state._model_started is not None:
            elapsed = time.perf_counter() - state._model_started
            waited = self.wait_clock() - state._wait_mark
            state.seconds += max(elapsed - waited, 0.0)
            state._model_started = None
        usage = usage_of_event(event)
        if usage is not None:
            state.prompt_tokens += usage.prompt_tokens
            state.completion_tokens += usage.completion_tokens
        state.response = content
        if state.final_sent or not event.get_function_calls():
            state.final_text = _text_of(content)
        return None

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        """Refuse a tool asked for by the final call; start the tool's clock."""
        state = self.state(tool_context.session.id)
        if state.final_sent:
            return {"error": "the tool budget is used; answer from what you have read"}
        if UNREADABLE_ARGUMENTS in tool_args:
            # Counted as a tool call, as the chat-completions loop counted it.
            state.tool_calls += 1
            return {"error": str(tool_args[UNREADABLE_ARGUMENTS])}
        state._tool_started[tool_context.function_call_id or tool.name] = (
            time.perf_counter()
        )
        return None

    def _stop_clock(self, tool: Any, tool_context: Any) -> None:
        state = self.state(tool_context.session.id)
        started = state._tool_started.pop(
            tool_context.function_call_id or tool.name, None
        )
        if started is not None:
            state.seconds += time.perf_counter() - started
            state.tool_calls += 1

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Count the tool and add its time."""
        self._stop_clock(tool, tool_context)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Return a tool's error to the model as its result."""
        self._stop_clock(tool, tool_context)
        if isinstance(error, ValueError) and str(error).startswith(
            f"Tool '{tool.name}' not found"
        ):
            # ADK's own reply to an unknown tool lists the tools there are.
            return None
        return {"error": f"{tool.name} failed: {type(error).__name__}: {error}"}


__all__ = [
    "GuardState",
    "QuestionGuardPlugin",
    "ResponseCachePlugin",
    "content_messages",
    "request_key",
]
