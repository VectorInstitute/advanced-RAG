"""Bulk model calls through infermesh, with this package's cache, meter, and guards.

A graph build, a synthetic question set, a judging pass, and a contextual index
are each thousands of independent calls to one model. :class:`ProxyBatch` sends
them through `infermesh <https://github.com/VectorInstitute/infermesh>`_,
Vector's batching client on LiteLLM, and keeps the parts of this package that
make a bulk run safe to repeat:

* infermesh runs the batch. It keeps a bounded window of requests in flight
  (``max_parallel``), holds them to a token-bucket rate limit (``rpm``, ``tpm``)
  that re-syncs from any ``x-ratelimit-*`` headers the endpoint returns, retries
  only transient errors with exponential backoff and ``Retry-After``, and
  returns one result or one error per item, so a failed item never stops the
  rest. Its sync methods run on a background event loop, so a notebook needs no
  ``asyncio`` setup.
* The response cache answers first. Every request is keyed with
  :func:`aieng.advanced_rag.graphrag.proxy.request_key`, the key ``govern``
  uses, so a cache written by either path answers the other, and the committed
  reference caches still hit. A batch whose every item is cached makes no call
  and builds no client, so a replay needs no key.
* Each result is written to the cache as it settles, through infermesh's
  ``on_result`` callback. A run that is killed loses only the requests in
  flight, and the next run sends only what is missing. That is the resume.
* The request is shaped for the model. The base URL and key come from
  :mod:`aieng.advanced_rag.proxy`, paid models get
  :func:`aieng.advanced_rag.models.adapt_request`, and every model gets its own
  thinking-off switch from :func:`aieng.advanced_rag.models.thinking_body`.
* Cost is counted and capped. Every call and cache hit reaches
  :mod:`aieng.advanced_rag.metering`, so a question meter or ``load_or_run``
  sees it. A live batch runs :func:`aieng.advanced_rag.metering.before_call`
  first, so a run past ``ADVANCED_RAG_SPEND_LIMIT_USD`` sends nothing, and the
  call that crosses the limit raises once it returns, which stops the batch.
  Before a live batch on a paid model, the estimate is checked against the
  key's budget with :func:`aieng.advanced_rag.runs.confirmation_needed`, as
  ``load_or_run`` does. The estimate counts only the requests the cache cannot
  answer, so a fully cached batch never asks and a partly cached one is
  judged on its misses.

Retries cover a rate limit (429), a connection failure, a timeout, and a server
error (500, 502, 503, 504). Anything else, such as a 400 for a parameter the
model rejects, fails its item on the first attempt.

The layer needs the ``batch`` extra (``pip install 'aieng-advanced-rag[batch]'``).

Examples
--------
>>> from aieng.advanced_rag.batch import ProxyBatch  # doctest: +SKIP
>>> with ProxyBatch(
...     "Qwen3_6-35B-A3B", cache=Path("responses.sqlite")
... ) as batch:  # doctest: +SKIP
...     outcomes = batch.complete_batch(prompts, max_tokens=200, temperature=0.0)
>>> [o.text for o in outcomes if o.ok]  # doctest: +SKIP
"""

from __future__ import annotations

import contextvars
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aieng.advanced_rag import metering, runs
from aieng.advanced_rag.graphrag.proxy import CallLedger, ResponseCache, request_key
from aieng.advanced_rag.metering import CallUsage
from aieng.advanced_rag.models import adapt_request, thinking_body


if TYPE_CHECKING:
    from infermesh import GenerationResult, LMClient
    from infermesh.types import RequestMetrics

    from aieng.advanced_rag.spend import Budget, ModelPrice


#: Requests per minute for a model with no measured cap. It matches the
#: package's :class:`~aieng.advanced_rag.graphrag.proxy.RateLimiter` default.
DEFAULT_RPM = 80

#: Requests per minute for each model whose cap on the Vector proxy was
#: measured, set under the cap so a second job on the same key still fits.
#: ``Qwen3_6-35B-A3B`` answered HTTP 429 above 90 requests a minute.
PROXY_RPM: Mapping[str, int] = {"Qwen3_6-35B-A3B": 80}

#: Requests in flight at once. Throughput through the proxy rose from 5.3 to
#: 15.4 items a second between 4 and 32 in flight on a short extraction call;
#: the proxy is shared, so the default stays low and a job raises it on purpose.
DEFAULT_MAX_PARALLEL = 8

#: Retries of a transient error per request. A graph build loses a document
#: when one extraction call fails, so the budget matches ``govern``'s.
DEFAULT_MAX_RETRIES = 16

#: Input to one request: a prompt, sent as one user message, or chat messages.
Prompt = str | Sequence[Mapping[str, Any]]

#: A caller's own cache key: ``key(messages, params) -> str``, from the chat
#: messages and the request arguments as the caller wrote them.
KeyFunction = Callable[[list[dict[str, Any]], Mapping[str, Any]], str]


def proxy_rpm(model: str) -> int:
    """Return the request rate a batch on ``model`` is held to by default.

    Parameters
    ----------
    model : str
        A bare model name.

    Returns
    -------
    int
        The measured setting from :data:`PROXY_RPM`, or :data:`DEFAULT_RPM`.
    """
    return PROXY_RPM.get(model, DEFAULT_RPM)


@dataclass(frozen=True)
class BatchOutcome:
    """One request's result: the reply, what it cost, or why it failed.

    Parameters
    ----------
    text : str or None
        The message content, ``""`` for a response with none (which also sets
        ``error``), and None when the request failed.
    response : Any or None
        The full response as an ``openai`` ``ChatCompletion``, the same object
        ``govern`` returns, or None when the request failed or a text
        cache answered it.
    usage : CallUsage
        The tokens the response reported. A cache hit carries what the cold
        call cost.
    cached : bool
        Whether the response cache answered.
    error : BaseException or None
        The error the request ended on, after any retries, or
        :class:`EmptyReplyError` for a reply with no content.
    metrics : RequestMetrics or None
        infermesh's queue wait, service time, and retry count for a live call.
        None for a cache hit or a failure.
    """

    text: str | None
    response: Any | None = field(default=None, repr=False)
    usage: CallUsage = field(default_factory=CallUsage)
    cached: bool = False
    error: BaseException | None = None
    metrics: RequestMetrics | None = None

    @property
    def ok(self) -> bool:
        """Return whether the request produced a response."""
        return self.error is None


class EmptyReplyError(RuntimeError):
    """A reply that carried no text and no tool call.

    It is not cached, so a rerun asks the model again. The outcome keeps the
    reply's usage, because the call was still made and billed.
    """


def _messages(prompt: Prompt, system: str | None) -> list[dict[str, Any]]:
    """Return the chat messages for one prompt."""
    if isinstance(prompt, str):
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        return messages
    if system is not None:
        raise ValueError("pass a system prompt or chat messages, not both")
    return [dict(message) for message in prompt]


def _has_cache_control(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and "cache_control" in part for part in content
        ):
            return True
    return False


def _as_completion(raw: Any) -> Any:
    """Return a LiteLLM response as the ``ChatCompletion`` the cache stores."""
    from openai.types.chat import ChatCompletion

    data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    return ChatCompletion.model_validate(data)


def _new_client(**kwargs: Any) -> LMClient:
    """Return an infermesh client that also retries a 502."""
    try:
        from infermesh import LMClient
        from litellm.exceptions import BadGatewayError
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "ProxyBatch needs infermesh: pip install 'aieng-advanced-rag[batch]'"
        ) from error
    client = LMClient(**kwargs)
    # LiteLLM raises BadGatewayError for a 502, which infermesh does not retry,
    # and infermesh takes no retry set as an argument. govern retries every
    # 5xx, so adding it here keeps the two paths alike.
    client._retryable_exceptions = (*client._retryable_exceptions, BadGatewayError)
    return client


class ProxyBatch:
    """Cached, metered, rate-limited calls to one model through infermesh.

    Parameters
    ----------
    model : str
        The bare model name the proxy lists, such as ``Qwen3_6-35B-A3B``.
    cache : ResponseCache, Path, or None, optional
        The response cache, or a SQLite path to open one at. None caches
        nothing and resumes nothing.
    rpm : int or None, optional
        Requests per minute. :func:`proxy_rpm` when omitted.
    tpm : int or None, optional
        Tokens per minute. No token limit when omitted.
    max_parallel : int, optional
        Requests in flight at once.
    max_retries : int, optional
        Retries of a transient error per request.
    timeout : float, optional
        Seconds before one request times out, which is itself retried.
    thinking : bool, optional
        Leave the model's reasoning on. It is off by default, because on a
        short call it is most of the cost.
    ledger : CallLedger or None, optional
        Where calls, tokens, hits, and failures accumulate. One is created
        when omitted.
    api_key : str or None, optional
        Overrides :func:`aieng.advanced_rag.proxy.resolve_api_key`.
    base_url : str or None, optional
        Overrides :func:`aieng.advanced_rag.proxy.resolve_base_url`.
    env_path : Path or None, optional
        A ``.env`` file for the resolvers to read.
    table : mapping or None, optional
        Rates for the pre-batch estimate.
        :func:`aieng.advanced_rag.spend.price_table` when omitted.
    key : callable or None, optional
        ``key(messages, params) -> str``, a cache key of the caller's own, for a
        cache whose keys a committed result records and so cannot change.
        :func:`aieng.advanced_rag.graphrag.proxy.request_key` when omitted.
    payload : {"completion", "text"}, optional
        What the cache stores: the whole response as ``ChatCompletion`` JSON,
        as ``govern`` stores it, or the reply text alone, as ``async_llm`` and
        the older text caches store it. A text hit carries no ``response``.
        Under either payload, a reply with no text and no tool call is not
        cached and comes back with ``error`` set to :class:`EmptyReplyError`,
        so the next run asks again.

    Notes
    -----
    The infermesh client is built on the first request the cache cannot answer,
    so a fully cached run never reads a key. Call :meth:`close`, or use the
    batch as a context manager, to stop the client's background loop.
    """

    def __init__(
        self,
        model: str,
        *,
        cache: ResponseCache | Path | None = None,
        rpm: int | None = None,
        tpm: int | None = None,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 600.0,
        thinking: bool = False,
        ledger: CallLedger | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env_path: Path | None = None,
        table: Mapping[str, ModelPrice] | None = None,
        key: KeyFunction | None = None,
        payload: Literal["completion", "text"] = "completion",
    ) -> None:
        if payload not in ("completion", "text"):
            raise ValueError(f"payload must be 'completion' or 'text', not {payload!r}")
        self.model = model
        self.table = table
        self.cache = ResponseCache(cache) if isinstance(cache, Path) else cache
        self.rpm = rpm if rpm is not None else proxy_rpm(model)
        self.tpm = tpm
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking = thinking
        self.ledger = ledger if ledger is not None else CallLedger(label=model)
        self._api_key = api_key
        self._base_url = base_url
        self._env_path = env_path
        self._client: LMClient | None = None
        self._lock = threading.Lock()
        self._warned_cache_control = False
        self._key_function = key
        self.payload = payload

    # -- the infermesh client -------------------------------------------------

    def _switch(self) -> dict[str, Any]:
        return {} if self.thinking else thinking_body(self.model)

    def client(self) -> LMClient:
        """Return the infermesh client, building it on first use.

        Returns
        -------
        infermesh.LMClient
            Pointed at the proxy, with the rate limit, retry budget, timeout,
            and thinking switch set.
        """
        with self._lock:
            if self._client is None:
                from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url

                switch = self._switch()
                self._client = _new_client(
                    # LiteLLM routes an OpenAI-compatible endpoint by this
                    # prefix and strips it before sending.
                    model=f"openai/{self.model}",
                    api_base=self._base_url or resolve_base_url(self._env_path),
                    api_key=self._api_key or resolve_api_key(self._env_path),
                    max_parallel_requests=self.max_parallel,
                    rpm=self.rpm,
                    tpm=self.tpm,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                    default_request_kwargs={"extra_body": switch} if switch else None,
                )
            return self._client

    def close(self) -> None:
        """Stop the infermesh client's background loop, if one was built."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def __enter__(self) -> ProxyBatch:
        """Return the batch."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client."""
        self.close()

    # -- requests ---------------------------------------------------------------

    def _request_kwargs(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return what infermesh sends for ``params``: adapted, with the switch.

        The switch travels in ``default_request_kwargs``. A caller's own
        ``extra_body`` replaces that default, so it is merged here with the
        caller's keys winning, as :func:`~aieng.advanced_rag.reasoning.disable_thinking`
        merges them.
        """
        if params.get("stream"):
            raise ValueError("ProxyBatch does not support streaming responses")
        sent: dict[str, Any] = adapt_request(self.model, params)
        caller = sent.pop("extra_body", None)
        if caller:
            sent["extra_body"] = {**self._switch(), **caller}
        return sent

    def _key(self, messages: list[dict[str, Any]], params: Mapping[str, Any]) -> str:
        if self._key_function is not None:
            return self._key_function(messages, params)
        key: str = request_key(self.model, messages, params, thinking=self.thinking)
        return key

    def _hit(self, key: str) -> BatchOutcome | None:
        """Return the cached outcome for ``key``, reporting the hit."""
        if self.cache is None:
            return None
        cached = self.cache.entry(key)
        if cached is None:
            return None
        from openai.types.chat import ChatCompletion

        self.ledger.hit()
        metering.report_cache_hit(model=self.model, usage=cached.usage)
        if self.payload == "text":
            return BatchOutcome(cached.payload, None, cached.usage, cached=True)
        response = ChatCompletion.model_validate_json(cached.payload)
        parsed = metering.read_completion(response)
        return BatchOutcome(parsed.text, response, cached.usage, cached=True)

    def _settle(
        self, key: str, result: GenerationResult | None, error: BaseException | None
    ) -> BatchOutcome:
        """Store, count, and report one live result.

        The response is cached before it is reported, because reporting is
        where a spend ceiling raises, and a paid response should not be lost.
        """
        if error is not None or result is None:
            self.ledger.failure()
            return BatchOutcome(None, error=error or RuntimeError("no result"))
        response = _as_completion(result.raw_response)
        parsed = metering.read_completion(response)
        metrics = result.metrics
        seconds = metrics.service_time_s if metrics is not None else 0.0
        for _ in range(metrics.retries if metrics is not None else 0):
            self.ledger.failure()
        self.ledger.record(parsed.usage, seconds)
        # A reply with no text and no tool call is not stored, so the next run
        # asks again instead of replaying it, and it is returned as a failure.
        empty = parsed.tool_calls == 0 and not (parsed.text or "").strip()
        if empty:
            self.ledger.empty()
        stored = (
            response.model_dump_json() if self.payload == "completion" else parsed.text
        )
        if self.cache is not None and not empty:
            self.cache.put(
                key,
                stored,
                parsed.usage.prompt_tokens,
                parsed.usage.completion_tokens,
                model=self.model,
                response_model=parsed.model,
            )
        metering.report_call(
            parsed.usage,
            seconds,
            tool_calls=parsed.tool_calls,
            empty=empty,
            model=self.model,
            response_model=parsed.model,
        )
        return BatchOutcome(
            parsed.text,
            response,
            parsed.usage,
            metrics=metrics,
            error=EmptyReplyError(f"{self.model} returned no content")
            if empty
            else None,
        )

    def _check_messages(self, messages: list[dict[str, Any]]) -> None:
        if not self._warned_cache_control and _has_cache_control(messages):
            self._warned_cache_control = True
            warnings.warn(
                "LiteLLM removes cache_control from messages on its OpenAI-"
                "compatible route, so the provider's prompt cache is not used "
                "for these requests; the response cache still applies",
                stacklevel=3,
            )

    def create(
        self, prompt: Prompt, *, system: str | None = None, **params: Any
    ) -> BatchOutcome:
        """Send one request, or answer it from the cache.

        Safe to call from many threads at once: every call shares the one
        client's rate limit and in-flight window.

        Parameters
        ----------
        prompt : str or sequence of mapping
            A prompt, sent as one user message, or chat messages.
        system : str or None, optional
            A system message to put before a string prompt.
        **params
            Other ``chat.completions.create`` arguments, such as ``max_tokens``
            and ``temperature``, as written for an on-prem model. They are
            adapted for a paid model before sending; the cache key uses them
            as written.

        Returns
        -------
        BatchOutcome
            The reply, with its response and usage.

        Raises
        ------
        Exception
            The request's error after any retries, such as
            ``litellm.BadRequestError`` on the first attempt, or
            :class:`~aieng.advanced_rag.spend.SpendLimitReached` from the spend
            guard.
        """
        messages = _messages(prompt, system)
        key = self._key(messages, params)
        hit = self._hit(key)
        if hit is not None:
            return hit
        kwargs = self._request_kwargs(params)
        self._check_messages(messages)
        metering.before_call(self.model)
        try:
            result = self.client().generate(messages, **kwargs)
        except Exception as error:
            self._settle(key, None, error)
            raise
        return self._settle(key, result, None)

    async def acreate(
        self, prompt: Prompt, *, system: str | None = None, **params: Any
    ) -> BatchOutcome:
        """Send one request on the caller's event loop, or answer it from the cache.

        The same as :meth:`create`, for code that is already async, such as a
        model function LightRAG awaits. Raises as :meth:`create` does.
        """
        messages = _messages(prompt, system)
        key = self._key(messages, params)
        hit = self._hit(key)
        if hit is not None:
            return hit
        kwargs = self._request_kwargs(params)
        self._check_messages(messages)
        metering.before_call(self.model)
        try:
            result = await self.client().agenerate(messages, **kwargs)
        except Exception as error:
            self._settle(key, None, error)
            raise
        return self._settle(key, result, None)

    def complete(
        self, prompt: Prompt, *, system: str | None = None, **params: Any
    ) -> str:
        """Return the reply text for one request. See :meth:`create`."""
        return self.create(prompt, system=system, **params).text or ""

    def _plan(
        self, prompts: Sequence[Prompt], system: str | None, params: Mapping[str, Any]
    ) -> tuple[list[BatchOutcome | None], list[tuple[int, str, list[dict[str, Any]]]]]:
        """Answer what the cache can and list the rest as (index, key, messages)."""
        outcomes: list[BatchOutcome | None] = [None] * len(prompts)
        misses: list[tuple[int, str, list[dict[str, Any]]]] = []
        for index, prompt in enumerate(prompts):
            messages = _messages(prompt, system)
            key = self._key(messages, params)
            hit = self._hit(key)
            if hit is not None:
                outcomes[index] = hit
            else:
                misses.append((index, key, messages))
        return outcomes, misses

    def _confirm(
        self,
        misses: Sequence[tuple[int, str, list[dict[str, Any]]]],
        params: Mapping[str, Any],
        confirm: bool,
        budget: Budget | None,
    ) -> None:
        """Stop a live batch whose cache misses need ``confirm=True``.

        Only the misses are priced: a request the cache answers costs nothing.
        """
        from aieng.advanced_rag import spend

        if confirm or runs.step_checked() or not spend.is_paid(self.model):
            return
        dollars = self.estimate_usd([messages for _, _, messages in misses], params)
        if not dollars:
            return
        print(
            f"batch of {len(misses):,} calls to {self.model}: est. ${dollars:,.2f}",
            flush=True,
        )
        known = budget if budget is not None else spend.read_budget()
        stop = runs.confirmation_needed(dollars, known)
        if stop is not None:
            raise runs.ConfirmationRequiredError(stop)

    def estimate_usd(
        self, prompts: Sequence[Prompt], params: Mapping[str, Any]
    ) -> float | None:
        """Return what sending ``prompts`` would cost at most, in dollars.

        Prompt tokens are counted with LiteLLM's tokenizer, and every reply is
        taken to use its full completion cap, so the figure is an upper bound.

        Parameters
        ----------
        prompts : sequence
            Prompts or chat messages, as :meth:`complete_batch` takes them.
        params : Mapping
            The request arguments; ``max_tokens`` sets the completion cap.

        Returns
        -------
        float or None
            Dollars, or None for a model with no price.
        """
        import litellm

        from aieng.advanced_rag import spend

        cap = int(params.get("max_tokens") or params.get("max_completion_tokens") or 0)
        prompt_tokens = 0
        for prompt in prompts:
            messages = _messages(prompt, None)
            try:
                prompt_tokens += int(
                    litellm.token_counter(model=self.model, messages=messages)
                )
            except Exception:  # noqa: BLE001 - fall back to a character estimate
                prompt_tokens += (
                    sum(len(str(m.get("content", ""))) for m in messages) // 4
                )
        dollars: float | None = spend.estimate_tokens(
            self.model, prompt_tokens, cap * len(prompts), table=self.table
        )
        return dollars

    def complete_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        system: str | None = None,
        confirm: bool = False,
        budget: Budget | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        **params: Any,
    ) -> list[BatchOutcome]:
        """Answer every prompt, from the cache where it can and live otherwise.

        Parameters
        ----------
        prompts : sequence
            Prompts, each sent as one user message, or chat-message lists.
        system : str or None, optional
            A system message to put before every string prompt.
        confirm : bool, optional
            Send a paid batch even when its estimate is over 10% of the key's
            budget left, or over $1 while the key reports no budget. Inside a
            ``load_or_run`` step, which has already checked, no check runs.
        budget : Budget or None, optional
            The key's budget, for that check. Read from the proxy when omitted.
        on_progress : callable or None, optional
            Called as ``on_progress(done, total)``, counting cache hits first.
        **params
            Request arguments shared by every prompt; see :meth:`create`.

        Returns
        -------
        list of BatchOutcome
            One per prompt, in order. A failed request is an outcome with
            ``error`` set; the batch carries on without it.

        Raises
        ------
        ConfirmationRequiredError
            Before any call, when the estimate needs ``confirm=True``.
        SpendLimitReached
            From the spend guard. The responses that settled before it are in
            the cache.
        """
        outcomes, misses = self._plan(prompts, system, params)
        total, hits = len(prompts), len(prompts) - len(misses)
        if on_progress is not None and hits:
            on_progress(hits, total)
        if not misses:
            return [outcome for outcome in outcomes if outcome is not None]
        kwargs = self._request_kwargs(params)
        self._check_messages(misses[0][2])
        self._confirm(misses, params, confirm, budget)
        with runs._waive_deferred_stop():
            return self._send_batch(misses, outcomes, hits, total, kwargs, on_progress)

    def _send_batch(
        self,
        misses: list[tuple[int, str, list[dict[str, Any]]]],
        outcomes: list[BatchOutcome | None],
        hits: int,
        total: int,
        kwargs: dict[str, Any],
        on_progress: Callable[[int, int], None] | None,
    ) -> list[BatchOutcome]:
        """Send the cache misses of a batch that passed its check."""
        metering.before_call(self.model)
        # infermesh calls on_result on its own event-loop thread. Running each
        # settle in the caller's context keeps the reports on the caller's
        # question meter, and inside the waiver the caller holds.
        context = contextvars.copy_context()

        def settled(
            index: int, result: GenerationResult | None, error: BaseException | None
        ) -> None:
            slot, key, _ = misses[index]
            outcomes[slot] = context.run(self._settle, key, result, error)

        def progress(done: int, _: int) -> None:
            if on_progress is not None:
                on_progress(hits + done, total)

        self.client().generate_batch(
            [messages for _, _, messages in misses],
            on_result=settled,
            on_progress=progress,
            **kwargs,
        )
        return [
            outcome
            if outcome is not None
            else BatchOutcome(None, error=RuntimeError("unsettled"))
            for outcome in outcomes
        ]

    async def acomplete_batch(
        self,
        prompts: Sequence[Prompt],
        *,
        system: str | None = None,
        confirm: bool = False,
        budget: Budget | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        **params: Any,
    ) -> list[BatchOutcome]:
        """Answer every prompt on the caller's event loop.

        The same as :meth:`complete_batch`, for code that is already async.
        The cache is read and written synchronously; each read is one SQLite
        lookup.
        """
        outcomes, misses = self._plan(prompts, system, params)
        total, hits = len(prompts), len(prompts) - len(misses)
        if on_progress is not None and hits:
            on_progress(hits, total)
        if not misses:
            return [outcome for outcome in outcomes if outcome is not None]
        kwargs = self._request_kwargs(params)
        self._check_messages(misses[0][2])
        self._confirm(misses, params, confirm, budget)
        with runs._waive_deferred_stop():
            return await self._asend_batch(
                misses, outcomes, hits, total, kwargs, on_progress
            )

    async def _asend_batch(
        self,
        misses: list[tuple[int, str, list[dict[str, Any]]]],
        outcomes: list[BatchOutcome | None],
        hits: int,
        total: int,
        kwargs: dict[str, Any],
        on_progress: Callable[[int, int], None] | None,
    ) -> list[BatchOutcome]:
        """Send the cache misses of a batch that passed its check, async."""
        metering.before_call(self.model)

        def settled(
            index: int, result: GenerationResult | None, error: BaseException | None
        ) -> None:
            slot, key, _ = misses[index]
            outcomes[slot] = self._settle(key, result, error)

        def progress(done: int, _: int) -> None:
            if on_progress is not None:
                on_progress(hits + done, total)

        await self.client().agenerate_batch(
            [messages for _, _, messages in misses],
            on_result=settled,
            on_progress=progress,
            **kwargs,
        )
        return [
            outcome
            if outcome is not None
            else BatchOutcome(None, error=RuntimeError("unsettled"))
            for outcome in outcomes
        ]

    def completer(self, *, system: str | None = None, **params: Any) -> BatchCompleter:
        """Return a ``complete(prompt) -> str`` callable that can also batch.

        The builders in this package that take a ``complete`` callable, such as
        :func:`aieng.advanced_rag.graphrag.extraction.build_graph` and
        :func:`aieng.advanced_rag.evaluation.answers.judge_answers`, look for its
        ``batch`` method and send their prompts as one batch when it is there.

        Parameters
        ----------
        system : str or None, optional
            A system message to put before every prompt.
        **params
            Request arguments for every call; see :meth:`create`.

        Returns
        -------
        BatchCompleter
            The callable.
        """
        return BatchCompleter(self, system, dict(params))


class BatchCompleter:
    """A ``complete(prompt) -> str`` callable backed by a :class:`ProxyBatch`.

    Calling it sends one request. :meth:`batch` sends many.

    Parameters
    ----------
    batch : ProxyBatch
        Where requests go.
    system : str or None
        A system message for every prompt.
    params : dict
        Request arguments for every call.
    """

    def __init__(
        self, batch: ProxyBatch, system: str | None, params: dict[str, Any]
    ) -> None:
        self.proxy = batch
        self.system = system
        self.params = params

    def __call__(self, prompt: str) -> str:
        """Return the reply to one prompt, raising if the request fails."""
        return self.proxy.complete(prompt, system=self.system, **self.params)

    def batch(
        self,
        prompts: Sequence[str],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[str | BaseException]:
        """Return each prompt's reply, or the error its request ended on.

        Parameters
        ----------
        prompts : sequence of str
            The prompts, in order.
        on_progress : callable or None, optional
            Called as ``on_progress(done, total)``.

        Returns
        -------
        list
            One reply text or one exception per prompt, in order.
        """
        outcomes = self.proxy.complete_batch(
            prompts, system=self.system, on_progress=on_progress, **self.params
        )
        replies: list[str | BaseException] = []
        for outcome in outcomes:
            if outcome.error is not None:
                replies.append(outcome.error)
            else:
                replies.append(outcome.text or "")
        return replies


__all__ = [
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RPM",
    "PROXY_RPM",
    "BatchCompleter",
    "BatchOutcome",
    "EmptyReplyError",
    "KeyFunction",
    "Prompt",
    "ProxyBatch",
    "proxy_rpm",
]
