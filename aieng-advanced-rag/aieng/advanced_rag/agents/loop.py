"""An agent loop on ADK, over plain Python tools, pointed at the proxy.

The binding is three objects. ``LiteLlm`` carries the endpoint, the key, and the
request body; ``LlmAgent`` carries the instruction and the tools; and
``InMemoryRunner`` drives the turns, yielding one event per model response and
one per tool result. Everything a caller reads back is in this package's own
types, so replacing the framework means replacing this file and
:mod:`aieng.advanced_rag.agents.toolset`.

This loop has its own call path rather than reusing
:class:`~aieng.advanced_rag.pipeline.generate.ChatCompleter`, and the reason is
not tidiness. That completer sends no ``tools`` array, so an agent could not ask
for a tool through it, and it converts a response with null ``content`` into an
empty string and counts it as an abstention. A tool call is exactly that shape:
checked against the proxy on 2026-09-28, ``message.content`` is null while
``tool_calls`` is populated. Routing an agent through the old completer would
record every tool call as a refusal to answer and throw the call away.

What a call cost still goes to the one meter. The model's client,
:class:`~aieng.advanced_rag.agents.model.GovernedLlmClient`, reports each live
call to :mod:`aieng.advanced_rag.metering` with the model's own token counts,
after the spend guard's pre-call hooks have passed it, and a response served
from the response cache is reported as a cache hit by
:class:`~aieng.advanced_rag.agents.plugins.ResponseCachePlugin`, so an agent run
and a pipeline run add up the same way.

Every run gets two ADK plugins by default: the question guard, which holds the
run to ``max_model_calls`` counting cached responses too, and the response
cache, which makes a repeated question cost nothing. Add your own tools,
callbacks, and plugins through the constructor; :meth:`AgentLoop.build_agent`
and :meth:`AgentLoop.build_plugins` give the same pieces to a runner of your
own.

Reach this module by path. It needs the ``agents`` extra, so the package's
``__init__`` does not import it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..cache.paths import root as cache_root
from ..graphrag.protocol import MissingExtraError
from ..graphrag.proxy import RateLimiter, ResponseCache
from ..metering import CallUsage, QuestionMeter
from ..models import family_of, request_params
from ..proxy import PROXY_BASE_URL, resolve_api_key, resolve_base_url
from .records import BUDGET_MESSAGE, READ_FIRST_MESSAGE
from .thinking import thinking_extra_body
from .tools import ToolFunction
from .usage import read_event, sum_usage


try:
    from google.adk.agents import LlmAgent
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError
    from google.adk.agents.run_config import RunConfig
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.base_toolset import BaseToolset
    from google.genai import types
except ImportError as error:  # pragma: no cover - import guard
    raise MissingExtraError("agents", "google-adk") from error

from .model import GovernedLlmClient, governed_model
from .plugins import QuestionGuardPlugin, ResponseCachePlugin
from .toolset import FunctionToolset


#: The default bulk model on that endpoint. It is a reasoning model, which is
#: why thinking is off unless a caller asks for it.
DEFAULT_MODEL = "Qwen3_6-35B-A3B"

#: What the agent is told to do. Short on purpose: every token here is paid for
#: on every call of every turn.
DEFAULT_INSTRUCTION = (
    "Answer the question from the corpus. Search before you answer, cite the "
    "passages you used, and say that the corpus does not say so when it does not."
)


def default_response_cache_path() -> Path:
    """Return where :class:`AgentLoop` caches responses unless told otherwise.

    Returns
    -------
    Path
        ``agents/responses.sqlite`` under
        :func:`~aieng.advanced_rag.cache.paths.root`, so
        ``ADVANCED_RAG_CACHE_DIR`` moves it with every other cache.
    """
    return cache_root() / "agents" / "responses.sqlite"


def proxy_api_key() -> str:
    """Return the proxy key from the environment, then the repository ``.env``.

    LiteLLM cannot find this on its own. Its fallbacks are provider-scoped,
    ``OPENAI_API_KEY`` for the ``openai`` provider and ``HOSTED_VLLM_API_KEY``
    for ``hosted_vllm``, so a key held under this project's own name has to be
    passed to the constructor explicitly. The lookup is
    :func:`aieng.advanced_rag.proxy.resolve_api_key`.

    Returns
    -------
    str
        The key, stripped.

    Raises
    ------
    aieng.advanced_rag.proxy.MissingProxyKeyError
        A ``RuntimeError``, when neither ``OPENAI_API_KEY`` nor
        ``VECTOR_PROXY_API_KEY`` is set in either place. Pass ``api_key`` to
        :class:`AgentLoop` to skip the lookup.
    """
    return resolve_api_key()


@dataclass(frozen=True)
class AgentAnswer:
    """What one invocation produced and what it cost.

    Parameters
    ----------
    text : str
        The answer, or ``""`` when the model produced none. Empty is what every
        parser in this package reads as an abstention.
    tool_calls : tuple of str
        Tool names the model asked for, in the order it asked.
    usage : CallUsage
        The invocation's tokens, summed over its model calls, with ``reported``
        true only when every call reported its own.
    model_calls : int
        How many model responses the run produced. A tool call is one of them.
    elapsed_seconds : float
        Wall clock for the whole invocation. The meter holds the service seconds
        separately, because the two differ by the worker count under
        concurrency.
    final_text : str, optional
        The text of the run's last response, which is ``text`` unless the
        last response carried none.
    exhausted : bool, optional
        Whether the run reached its call budget and was made to answer, with
        ``on_limit="answer"``.
    nudged : bool, optional
        Whether the read-first guard asked the model to read before answering.
    cache_hits : int, optional
        Responses served from the response cache. They count in
        ``model_calls`` and ``usage`` with the tokens they cost when made.
    seconds : float, optional
        Model and tool time, without rate-limit waits.
    transcript : tuple of dict, optional
        The last request and its response as chat-completions messages.
    tools_run : int, optional
        Tool calls executed. A tool asked for by a forced final call is not.
    """

    text: str
    tool_calls: tuple[str, ...]
    usage: CallUsage
    model_calls: int
    elapsed_seconds: float
    final_text: str = ""
    exhausted: bool = False
    nudged: bool = False
    cache_hits: int = 0
    seconds: float = 0.0
    transcript: tuple[dict[str, Any], ...] = ()
    tools_run: int = 0

    @property
    def abstained(self) -> bool:
        """Return whether the run ended without an answer.

        Returns
        -------
        bool
            ``True`` when the final model response carried no text.
        """
        return not self.text.strip()


@dataclass
class _Seen:
    """What the model responses of one run carried, across its turns."""

    texts: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    usages: list[CallUsage] = field(default_factory=list)


class AgentLoop:
    """An ADK agent that answers one question at a time, over plain functions.

    Parameters
    ----------
    tools : iterable of ToolFunction
        Framework-neutral functions, wrapped for ADK by
        :class:`~aieng.advanced_rag.agents.toolset.FunctionToolset`.
    model : str, optional
        Model name as the endpoint serves it, without a provider prefix.
    base_url : str or None, optional
        The endpoint. ``OPENAI_BASE_URL`` when set, then the repository
        ``.env``, then :data:`PROXY_BASE_URL`.
    api_key : str or None, optional
        The key. Read from the environment when omitted.
    instruction : str, optional
        The agent's system instruction.
    name : str, optional
        Agent name, which ADK requires to be a valid identifier.
    enable_thinking : bool, optional
        Whether the model reasons before answering. Off by default.
    temperature : float, optional
        Zero by default to remove sampling variance only. Serving backends may
        remain nondeterministic. Left out for models that reject it, as
        :meth:`request_fields` describes.
    max_tokens : int, optional
        Completion cap per call. With thinking on, a low cap truncates during
        reasoning and the response comes back with no content at all.
    max_model_calls : int, optional
        A cost guard. A loop that will not converge fails here instead of
        billing until it does.
    app_name : str, optional
        The name sessions are filed under.
    charge_to : QuestionMeter or None, optional
        Charge this question rather than the one bound to the current context.
        Pass one when the run happens in a thread that never bound one.
    cache : ResponseCache or bool, optional
        Where responses are cached. True, the default, uses
        :func:`default_response_cache_path`; False sends every request.
    limiter : RateLimiter or None, optional
        A request budget shared with other runs on the same key.
    max_retries : int, optional
        Attempts at a call that keeps raising a transient error.
    on_limit : {"raise", "answer"}, optional
        What ``max_model_calls`` does. ``"raise"`` raises
        ``LlmCallsLimitExceededError`` from the call past the cap. ``"answer"``
        makes the last allowed call with tools off, so the run answers.
    max_prompt_tokens : int or None, optional
        Make the call final once a request passes this estimate.
    read_first : callable or None, optional
        Returns whether the agent has read a source. When it returns False
        after the model answers, the loop asks once more with
        ``read_first_message``.
    read_first_message : str, optional
        The user turn the read-first guard sends.
    budget_message : str, optional
        The user turn a forced final call adds after a tool result.
    plugins : sequence of BasePlugin, optional
        Your own plugins, registered after the guard and the cache.
    agent_options : Mapping or None, optional
        More ``LlmAgent`` arguments, such as ``before_model_callback`` or
        ``after_tool_callback``.
    env_file : Path or None, optional
        The ``.env`` to read the key and endpoint from when the environment
        sets neither. By default the nearest ``.env`` up to the repository
        root, which in a git worktree is the worktree's own root, not the main
        checkout's.

    Examples
    --------
    >>> loop = AgentLoop(retrieval_tools(retriever))  # doctest: +SKIP
    >>> with meter.question("q1"):  # doctest: +SKIP
    ...     answer = await loop.arun("What is the notice period?")
    """

    def __init__(
        self,
        tools: Iterable[ToolFunction],
        *,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
        name: str = "rag_agent",
        enable_thinking: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_model_calls: int = 10,
        app_name: str = "advanced_rag",
        charge_to: QuestionMeter | None = None,
        cache: ResponseCache | bool = True,
        limiter: RateLimiter | None = None,
        max_retries: int = 16,
        on_limit: Literal["raise", "answer"] = "raise",
        max_prompt_tokens: int | None = None,
        read_first: Callable[[], bool] | None = None,
        read_first_message: str = READ_FIRST_MESSAGE,
        budget_message: str = BUDGET_MESSAGE,
        plugins: Sequence[Any] = (),
        agent_options: Mapping[str, Any] | None = None,
        env_file: Path | None = None,
    ) -> None:
        self.tools = tuple(tools)
        self.model = model
        self.base_url = base_url or resolve_base_url(env_file)
        self.api_key = api_key or resolve_api_key(env_file)
        self.instruction = instruction
        self.name = name
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_model_calls = max_model_calls
        self.app_name = app_name
        self.charge_to = charge_to
        self._cache = cache
        self.limiter = limiter
        self.max_retries = max_retries
        self.on_limit = on_limit
        self.max_prompt_tokens = max_prompt_tokens
        self.read_first = read_first
        self.read_first_message = read_first_message
        self.budget_message = budget_message
        self.plugins = tuple(plugins)
        self.agent_options = dict(agent_options or {})

    @property
    def cache(self) -> ResponseCache | None:
        """Return the response cache runs read and write, or None when off.

        The default file is opened on first use, so building a loop or an
        agent writes nothing to disk.

        Returns
        -------
        ResponseCache or None
            The cache passed in, the default one, or None for ``cache=False``.
        """
        if self._cache is True:
            self._cache = ResponseCache(default_response_cache_path())
        # Not ``self._cache or None``: an empty ResponseCache has length zero.
        return None if self._cache is False else self._cache

    def request_fields(self) -> tuple[float | None, int | None, dict[str, Any]]:
        """Return the temperature, completion cap, and body this model accepts.

        On-prem and unrecognised models get the loop's temperature and the vLLM
        thinking switch. A paid model gets the temperature and thinking switch
        :func:`~aieng.advanced_rag.models.request_params` gives its family, so
        a Claude 5.5 model gets no temperature. The switch travels in the body,
        because LiteLLM refuses a top-level ``reasoning_effort`` on this route.
        LiteLLM sends the cap as ``max_completion_tokens`` for every model,
        which every family on the proxy accepts.

        Returns
        -------
        tuple
            ``(temperature, max_tokens, extra_body)``. A None temperature
            leaves the field out of the request.
        """
        if family_of(self.model) in (None, "onprem"):
            return (
                self.temperature,
                self.max_tokens,
                thinking_extra_body(self.enable_thinking),
            )
        params = request_params(
            self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            thinking=self.enable_thinking,
        )
        temperature = params.pop("temperature", None)
        for cap in ("max_tokens", "max_completion_tokens"):
            params.pop(cap, None)
        body: dict[str, Any] = dict(params.pop("extra_body", None) or {})
        body.update(params)
        return temperature, self.max_tokens, body

    def build_agent(self, client: GovernedLlmClient | None = None) -> Any:
        """Build the ADK agent, ready for a runner of your own.

        Pair it with :meth:`build_plugins` to keep the cache and the guard.

        Parameters
        ----------
        client : GovernedLlmClient or None, optional
            The model client. A new one with this loop's limiter, retries, and
            ``charge_to`` when omitted.

        Returns
        -------
        LlmAgent
            An agent whose model points at the endpoint through a governed
            client, and whose tools are this loop's functions, behind one
            toolset, followed by any ADK tools passed in.
        """
        temperature, max_tokens, extra_body = self.request_fields()
        functions = [
            tool for tool in self.tools if not isinstance(tool, BaseTool | BaseToolset)
        ]
        native = [
            tool for tool in self.tools if isinstance(tool, BaseTool | BaseToolset)
        ]
        tools: list[Any] = (
            [FunctionToolset(functions)] if functions or not native else []
        )
        options: dict[str, Any] = {
            "name": self.name,
            "model": governed_model(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                extra_body=extra_body,
                client=client or self._client(),
            ),
            "instruction": self.instruction,
            "generate_content_config": types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            "tools": tools + native,
        }
        options.update(self.agent_options)
        return LlmAgent(**options)

    def _client(self) -> GovernedLlmClient:
        return GovernedLlmClient(
            limiter=self.limiter,
            max_retries=self.max_retries,
            charge_to=self.charge_to,
        )

    def build_plugins(self, client: GovernedLlmClient | None = None) -> list[Any]:
        """Return the plugins a run registers, in the order it registers them.

        Parameters
        ----------
        client : GovernedLlmClient or None, optional
            The client the agent sends through, so the guard can take its
            rate-limit waits off the recorded time.

        Returns
        -------
        list of BasePlugin
            The :class:`~aieng.advanced_rag.agents.plugins.QuestionGuardPlugin`,
            the :class:`~aieng.advanced_rag.agents.plugins.ResponseCachePlugin`
            when caching is on, then the plugins passed in.
        """
        guard = QuestionGuardPlugin(
            max_model_calls=self.max_model_calls,
            on_limit=self.on_limit,
            max_prompt_tokens=self.max_prompt_tokens,
            budget_message=self.budget_message,
            wait_clock=(lambda: client.waited) if client is not None else None,
        )
        plugins: list[Any] = [guard]
        cache = self.cache
        if cache is not None:
            plugins.append(
                ResponseCachePlugin(
                    cache,
                    salt={"extra_body": self.request_fields()[2]},
                    charge_to=self.charge_to,
                )
            )
        return plugins + list(self.plugins)

    async def arun(self, question: str, *, user_id: str = "participant") -> AgentAnswer:
        """Answer one question, metering every model call the loop makes.

        Parameters
        ----------
        question : str
            The question, unmodified.
        user_id : str, optional
            The session's user, which ADK requires and nothing here reads.

        Returns
        -------
        AgentAnswer
            The answer, the tools it called, and what it cost.

        Raises
        ------
        LlmCallsLimitExceededError
            With ``on_limit="raise"``, when the run passes ``max_model_calls``.
            Responses served from the cache count, so a replay stops where the
            run it repeats stopped. The calls made before it are already
            metered.
        """
        client = self._client()
        plugins = self.build_plugins(client)
        guard = plugins[0]
        cache_plugin = next(
            (plugin for plugin in plugins if isinstance(plugin, ResponseCachePlugin)),
            None,
        )
        runner = InMemoryRunner(
            app=App(
                name=self.app_name, root_agent=self.build_agent(client), plugins=plugins
            )
        )
        # ADK does not open a session for you. ``Runner`` takes
        # ``auto_create_session`` and it defaults to false, ``InMemoryRunner``
        # does not expose the flag, and from 2.10.0 the in-memory session
        # service raises ``SessionNotFoundError`` when an event is appended to a
        # session it does not hold.
        session = await runner.session_service.create_session(
            app_name=self.app_name, user_id=user_id
        )
        message = types.Content(role="user", parts=[types.Part(text=question)])
        seen = _Seen()
        nudged = False
        started = time.perf_counter()
        try:
            while True:
                await self._invoke(
                    runner, message, user_id=user_id, session_id=session.id, seen=seen
                )
                state = guard.state(session.id)
                if (
                    self.read_first is not None
                    and not nudged
                    and not state.final_sent
                    and state.model_calls < self.max_model_calls
                    and not self.read_first()
                ):
                    # A second turn on the same session, so the model sees its
                    # own answer and the request to read before giving it.
                    nudged = True
                    message = types.Content(
                        role="user", parts=[types.Part(text=self.read_first_message)]
                    )
                    continue
                break
        finally:
            await runner.close()
        state = guard.state(session.id)
        return AgentAnswer(
            # The last text the model produced is the answer. An earlier turn
            # can carry text alongside the tool call that follows it.
            text=seen.texts[-1] if seen.texts else "",
            tool_calls=tuple(seen.names),
            usage=sum_usage(seen.usages),
            model_calls=len(seen.usages),
            elapsed_seconds=time.perf_counter() - started,
            final_text=state.final_text,
            exhausted=state.exhausted,
            nudged=nudged,
            cache_hits=cache_plugin.hits if cache_plugin is not None else 0,
            seconds=round(state.seconds, 3),
            transcript=tuple(state.transcript()),
            tools_run=state.tool_calls,
        )

    async def _invoke(
        self,
        runner: Any,
        message: Any,
        *,
        user_id: str,
        session_id: str,
        seen: _Seen,
    ) -> None:
        """Run one turn, reading what each model response carried."""
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
                # ADK's own cap counts only calls that reach the model; the
                # guard counts cached responses too and stops first.
                run_config=RunConfig(max_llm_calls=self.max_model_calls),
            ):
                facts = read_event(event)
                if facts is None:
                    continue
                seen.usages.append(facts.usage)
                seen.names.extend(facts.tool_calls)
                if facts.text.strip():
                    seen.texts.append(facts.text)
        except RuntimeError as error:
            # ADK wraps an error raised inside a plugin callback.
            if isinstance(error.__cause__, LlmCallsLimitExceededError):
                raise error.__cause__ from None
            raise

    def run(self, question: str, *, user_id: str = "participant") -> AgentAnswer:
        """Answer one question from synchronous code.

        Parameters
        ----------
        question : str
            The question, unmodified.
        user_id : str, optional
            The session's user.

        Returns
        -------
        AgentAnswer
            What :meth:`arun` returned.

        Raises
        ------
        RuntimeError
            When called from inside a running event loop, where starting
            another one would deadlock. Await :meth:`arun` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(question, user_id=user_id))
        raise RuntimeError(
            "run() cannot run inside an active event loop; await arun() instead"
        )


__all__ = [
    "DEFAULT_INSTRUCTION",
    "DEFAULT_MODEL",
    "PROXY_BASE_URL",
    "AgentAnswer",
    "AgentLoop",
    "default_response_cache_path",
    "proxy_api_key",
]
