"""A rate-limited, cached, metered path to an OpenAI-compatible endpoint.

It serves long indexing runs.

Indexing a corpus with a graph method is tens of thousands of model calls made by
a library that builds its own requests. Three things have to hold for that to be
safe, and all of them live at the client rather than in the library.

* **A request budget.** The endpoint enforces a requests-per-minute cap per model and
  key, and answers an over-rate request with HTTP 429. LightRAG treats a failed
  extraction call as a failed document and writes no graph for it, so an
  unthrottled run does not slow down, it quietly produces an empty store.
  :class:`RateLimiter` holds every caller to one shared budget.
* **A response cache.** :class:`ResponseCache` keys on the whole request, so an
  interrupted run resumes at no cost and a repeated run reports the cold cost
  through its hit count instead of looking free.
* **Cost that is counted.** :class:`CallLedger` accumulates calls and tokens where
  the call is made, so a figure covers whatever the library did inside its own
  code path. Each call is also reported to :mod:`aieng.advanced_rag.metering`,
  and each live call first passes :func:`aieng.advanced_rag.metering.before_call`,
  where a run past its ``ADVANCED_RAG_SPEND_LIMIT_USD`` ceiling stops.

:func:`govern` applies all of it to an OpenAI client in place, because ``atlas-rag``
rejects anything that is not literally an ``OpenAI`` instance. :func:`async_llm`
builds the model function LightRAG takes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag import metering
from aieng.advanced_rag.metering import CallUsage
from aieng.advanced_rag.models import adapt_request, family_of, request_params
from aieng.advanced_rag.reasoning import disable_thinking


@dataclass(frozen=True)
class CachedResponse:
    """One stored response, what it cost, and the models behind it."""

    payload: str
    usage: CallUsage
    model: str | None
    response_model: str | None


class ResponseCache:
    """Content-addressed store of proxy responses, keyed on the full request.

    A re-run costs nothing and stays comparable with the run it repeats. Hits
    are reported separately from calls, so a cost figure is always the cost of
    a cold run.

    Each row also stores the model requested and the model the response
    reported, so a hit can be priced and an alias that moved to a new snapshot
    can be noticed. A cache file written before those columns existed gains
    them on open, with nulls for its existing rows, and so does a file with
    only ``key`` and ``payload``, which gains the two token columns too.

    Parameters
    ----------
    path : Path
        SQLite file. Parent directories are created.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "prompt_tokens INT, completion_tokens INT)"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(responses)")}
        for column, kind in (
            ("prompt_tokens", "INT"),
            ("completion_tokens", "INT"),
            ("model", "TEXT"),
            ("response_model", "TEXT"),
        ):
            if column not in columns:
                self._db.execute(f"ALTER TABLE responses ADD COLUMN {column} {kind}")
        self._db.commit()

    @staticmethod
    def key(request: Mapping[str, Any]) -> str:
        """Return the cache key for one request."""
        blob = json.dumps(request, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> str | None:
        """Return the cached response text, or ``None``."""
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row else None

    def entry(self, key: str) -> CachedResponse | None:
        """Return the cached response with its tokens and models, or ``None``."""
        with self._lock:
            row = self._db.execute(
                "SELECT payload, prompt_tokens, completion_tokens, model, "
                "response_model FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            payload=str(row[0]),
            usage=CallUsage(
                int(row[1] or 0), int(row[2] or 0), reported=row[1] is not None
            ),
            model=row[3],
            response_model=row[4],
        )

    def put(
        self,
        key: str,
        payload: str,
        prompt: int,
        completion: int,
        *,
        model: str | None = None,
        response_model: str | None = None,
    ) -> None:
        """Store one response, the tokens it cost, and the models behind it."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, payload, prompt_tokens, "
                "completion_tokens, model, response_model) VALUES (?, ?, ?, ?, ?, ?)",
                (key, payload, prompt, completion, model, response_model),
            )
            self._db.commit()

    def __len__(self) -> int:
        """Return how many responses are stored."""
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0])


class RateLimiter:
    """A sliding-window cap on requests per minute, shared by every caller.

    Some endpoints limit requests per minute for each model on a key. The
    limit has differed by model, so measure it (an over-rate response names it)
    and set the cap below it so a second job on the same key still fits.

    Parameters
    ----------
    per_minute : int, optional
        Requests allowed in any window.
    window : float, optional
        Window width in seconds.
    """

    def __init__(self, per_minute: int = 80, window: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window = window
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()

    def _wait_for(self) -> float:
        """Return seconds to wait, admitting the request when that is zero."""
        with self._lock:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= self.window:
                self._sent.popleft()
            if len(self._sent) < self.per_minute:
                self._sent.append(now)
                return 0.0
            return self.window - (now - self._sent[0]) + 0.01

    def acquire(self) -> None:
        """Block until the request fits inside the window."""
        while True:
            delay = self._wait_for()
            if delay <= 0.0:
                return
            time.sleep(delay)

    async def aacquire(self) -> None:
        """Await until the request fits inside the window."""
        while True:
            delay = self._wait_for()
            if delay <= 0.0:
                return
            await asyncio.sleep(delay)


def backoff_seconds(attempt: int) -> float:
    """Return the wait before retrying a failed proxy call.

    Short at first and capped well under a minute. The proxy's cap is a sliding
    window shared with anything else running on the key, so capacity frees up
    continuously and a long fixed wait spends a worker's time for nothing. The
    OpenAI client's own retry defaults to a full minute, which is why callers
    construct it with retries off and handle them at this cadence instead.

    Parameters
    ----------
    attempt : int
        Zero-based attempt number that just failed.

    Returns
    -------
    float
        Seconds to wait.
    """
    return float(min(30.0, 2.0**attempt))


def transient_errors() -> tuple[type[Exception], ...]:
    """Return the client errors a governed call retries.

    A rate limit (HTTP 429), a connection failure or timeout, and a server
    error (HTTP 5xx) can succeed on the next attempt. Anything else, such as a
    400 for a parameter the model rejects or a 401 for a bad key, fails the
    same way every time, so it is raised on the first attempt instead of
    spending the retry budget on it.

    Returns
    -------
    tuple of type
        ``openai.RateLimitError``, ``openai.APIConnectionError`` (which
        ``openai.APITimeoutError`` subclasses), and ``openai.InternalServerError``.
    """
    import openai  # noqa: PLC0415

    return (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )


@dataclass
class CallLedger:
    """Thread-safe call and token totals for one long run.

    Parameters
    ----------
    label : str, optional
        What the run is, for the report.
    """

    label: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_read_tokens: int = 0
    prompt_cache_write_tokens: int = 0
    cache_hits: int = 0
    empty_responses: int = 0
    failed_attempts: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage: CallUsage, seconds: float) -> None:
        """Add one uncached call."""
        with self._lock:
            self.calls += 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.prompt_cache_read_tokens += usage.prompt_cache_read_tokens
            self.prompt_cache_write_tokens += usage.prompt_cache_write_tokens
            self.seconds += seconds

    def hit(self) -> None:
        """Add one call that the cache answered."""
        with self._lock:
            self.cache_hits += 1

    def empty(self) -> None:
        """Add one response that carried no content."""
        with self._lock:
            self.empty_responses += 1

    def failure(self) -> None:
        """Add one attempt that raised, whether or not it was then retried."""
        with self._lock:
            self.failed_attempts += 1

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record.

        ``call_seconds`` is the sum of per-call durations across every thread, not
        elapsed wall time.
        """
        record: dict[str, Any] = {
            "label": self.label,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "empty_responses": self.empty_responses,
            "failed_attempts": self.failed_attempts,
            "call_seconds": round(self.seconds, 1),
        }
        # Provider prompt-cache counts sit inside prompt_tokens. They are left
        # out when zero so an on-prem record keeps its committed shape.
        if self.prompt_cache_read_tokens:
            record["prompt_cache_read_tokens"] = self.prompt_cache_read_tokens
        if self.prompt_cache_write_tokens:
            record["prompt_cache_write_tokens"] = self.prompt_cache_write_tokens
        return record


def _request(kind: str, model: str, messages: Any, params: Mapping[str, Any]) -> str:
    """Return the cache key for one request, tagged so shapes never collide."""
    return ResponseCache.key(
        {"kind": kind, "model": model, "messages": messages, "params": dict(params)}
    )


def request_key(
    model: str, messages: Any, params: Mapping[str, Any], *, thinking: bool = False
) -> str:
    """Return the cache key :func:`govern` stores one chat completion under.

    The key is built from the request as the caller wrote it, before
    :func:`~aieng.advanced_rag.models.adapt_request` and the thinking switch
    change it, so the same call keeps its key whichever client sends it.
    :class:`aieng.advanced_rag.batch.ProxyBatch` uses it too, which is what lets
    a cache written by one path answer the other.

    Parameters
    ----------
    model : str
        The model name the caller asked for.
    messages : Any
        The chat messages, as sent.
    params : Mapping
        Every other ``chat.completions.create`` argument, without ``model`` and
        ``messages``.
    thinking : bool, optional
        Whether the model's reasoning was left on.

    Returns
    -------
    str
        The SHA-256 hex digest.
    """
    return _request(
        "chat-completion", model, messages, {**params, "thinking": thinking}
    )


def _text_key_params(model: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the request fields an ``async_llm`` cache key names.

    The reference LightRAG indexes were built with keys holding only
    ``max_tokens`` and ``temperature``, when the on-prem thinking switch was
    the same on every call. Those keys are kept for on-prem and unrecognised
    models so the reference caches still hit. Any other family adds the full
    request, because its switch and parameter names differ.
    """
    legacy = {"max_tokens": params.get("max_tokens"), "temperature": 0.0}
    family = family_of(model)
    if family in (None, "onprem"):
        return legacy
    return {**legacy, "request": dict(params)}


def govern(
    client: Any,
    *,
    cache: ResponseCache,
    limiter: RateLimiter,
    ledger: CallLedger | None = None,
    max_retries: int = 16,
    thinking: bool = False,
) -> Any:
    """Cache, throttle, retry, and meter a synchronous client in place.

    The client's ``chat.completions.create`` is replaced, and the same object is
    returned, so a library that type-checks its client still accepts it.
    Reasoning is switched off through
    :func:`~aieng.advanced_rag.reasoning.disable_thinking`, which picks the
    switch per call from the request's model, unless ``thinking`` is true. The
    library builds the rest of the request, and
    :func:`~aieng.advanced_rag.models.adapt_request` reshapes it for the model's
    family before it is sent: a Claude model gets no ``temperature`` and no
    ``json_object`` response format, an OpenAI model gets
    ``max_completion_tokens``. On-prem requests are sent as the library wrote
    them. The cache key is built from the request as the library wrote it, so
    existing cache entries still hit. Streaming is not supported.

    Parameters
    ----------
    client : Any
        An ``openai.OpenAI`` client, built with retries off so the proxy's 429
        is handled here on a short schedule.
    cache : ResponseCache
        Where responses are stored. A hit returns the stored completion without a
        call.
    limiter : RateLimiter
        The request budget shared by every caller of this model.
    ledger : CallLedger or None, optional
        Where cost accumulates. One is created when omitted, and returned on
        ``client.ledger``.
    max_retries : int, optional
        Attempts at a call that keeps hitting a transient error (see
        :func:`transient_errors`) before it is treated as failed.
    thinking : bool, optional
        Leave the model's reasoning on.

    Returns
    -------
    Any
        The same client, with ``ledger`` set.

    Raises
    ------
    RuntimeError
        From a call whose every attempt raised a transient error. A transport
        failure never becomes an empty answer.
    openai.APIError
        Any other error from the client, such as ``openai.BadRequestError``,
        re-raised from the first attempt without a retry.
    """
    from openai.types.chat import ChatCompletion  # noqa: PLC0415

    if not thinking:
        disable_thinking(client)
    ledger = ledger if ledger is not None else CallLedger()
    retryable = transient_errors()
    completions = client.chat.completions
    original = completions.create

    def create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise ValueError("govern does not support streaming responses")
        model = str(kwargs["model"])
        messages = kwargs.get("messages")
        params = {k: v for k, v in kwargs.items() if k not in ("model", "messages")}
        key = request_key(model, messages, params, thinking=thinking)
        cached = cache.entry(key)
        if cached is not None:
            ledger.hit()
            metering.report_cache_hit(model=model, usage=cached.usage)
            return ChatCompletion.model_validate_json(cached.payload)

        kwargs = {"model": model, "messages": messages, **adapt_request(model, params)}
        metering.before_call(model)
        last: Exception | None = None
        for attempt in range(max_retries):
            limiter.acquire()
            started = time.perf_counter()
            try:
                response = original(**kwargs)
            except retryable as error:
                last = error
                ledger.failure()
                time.sleep(backoff_seconds(attempt))
                continue
            except Exception:
                ledger.failure()
                raise
            elapsed = time.perf_counter() - started
            usage = metering.usage_of(response)
            ledger.record(usage, elapsed)
            content = response.choices[0].message.content if response.choices else None
            if content is None:
                ledger.empty()
            reported = getattr(response, "model", None) or None
            metering.report_call(
                usage,
                elapsed,
                empty=content is None,
                model=model,
                response_model=reported,
            )
            cache.put(
                key,
                response.model_dump_json(),
                usage.prompt_tokens,
                usage.completion_tokens,
                model=model,
                response_model=reported,
            )
            return response
        raise RuntimeError(f"proxy call failed after {max_retries} tries: {last}")

    completions.create = create
    client.ledger = ledger
    return client


def async_llm(
    client: Any,
    model: str,
    *,
    cache: ResponseCache,
    limiter: RateLimiter,
    ledger: CallLedger | None = None,
    max_retries: int = 16,
    default_max_tokens: int = 4000,
    key: Callable[[list[dict[str, Any]], Mapping[str, Any]], str] | None = None,
) -> Callable[..., Awaitable[str]]:
    """Return the cached, throttled, metered model function LightRAG calls.

    LightRAG builds its own requests, so this function sends them: its
    parameters come from :func:`~aieng.advanced_rag.models.request_params`, so
    the completion cap, temperature, and reasoning switch are the ones the
    model's family accepts, and the accounting happens here. A retry budget
    this large is deliberate: LightRAG marks a document ``failed`` when one
    extraction call fails, so an unretried rate limit empties a run. Only a
    transient error (see :func:`transient_errors`) is retried.

    Parameters
    ----------
    client : Any
        An ``openai.AsyncOpenAI`` client built with retries off.
    model : str
        Model name to send.
    cache : ResponseCache
        Response cache. Only the response text is stored.
    limiter : RateLimiter
        The request budget shared by every caller of this model.
    ledger : CallLedger or None, optional
        Where cost accumulates. Read it from the returned function's ``ledger``.
    max_retries : int, optional
        Attempts at a call that keeps hitting a transient error before it raises.
    default_max_tokens : int, optional
        Completion cap when the caller names none.
    key : callable or None, optional
        ``key(messages, params) -> str``, a cache key of the caller's own, given
        the chat messages and the parameters sent, for a cache whose keys a
        committed result records. The ``chat-text`` key when omitted.

    Returns
    -------
    callable
        ``async (prompt, system_prompt=None, history_messages=None, **kwargs) -> str``.
        Calling it raises ``RuntimeError`` when every attempt hit a transient
        error, and re-raises any other client error, such as
        ``openai.BadRequestError``, from the first attempt.
    """
    patched = disable_thinking(client)
    ledger = ledger if ledger is not None else CallLedger()
    custom_key = key
    retryable = transient_errors()

    async def llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        for drop in ("hashing_kv", "keyword_extraction", "mode", "enable_cot"):
            kwargs.pop(drop, None)
        max_tokens = int(kwargs.pop("max_tokens", default_max_tokens))
        params = request_params(model, max_tokens=max_tokens, temperature=0.0)
        key = (
            custom_key(messages, params)
            if custom_key is not None
            else _request("chat-text", model, messages, _text_key_params(model, params))
        )
        cached = cache.entry(key)
        if cached is not None:
            ledger.hit()
            metering.report_cache_hit(model=model, usage=cached.usage)
            return cached.payload

        metering.before_call(model)
        last: Exception | None = None
        for attempt in range(max_retries):
            await limiter.aacquire()
            started = time.perf_counter()
            try:
                response = await patched.chat.completions.create(
                    model=model, messages=messages, **params
                )
            except retryable as error:
                last = error
                ledger.failure()
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            except Exception:
                ledger.failure()
                raise
            elapsed = time.perf_counter() - started
            usage = metering.usage_of(response)
            ledger.record(usage, elapsed)
            content = response.choices[0].message.content
            reported = getattr(response, "model", None) or None
            metering.report_call(
                usage,
                elapsed,
                empty=content is None,
                model=model,
                response_model=reported,
            )
            if content is None:
                ledger.empty()
                return ""
            cache.put(
                key,
                str(content),
                usage.prompt_tokens,
                usage.completion_tokens,
                model=model,
                response_model=reported,
            )
            return str(content)
        raise RuntimeError(f"proxy call failed after {max_retries} tries: {last}")

    llm.ledger = ledger  # type: ignore[attr-defined]
    return llm


__all__ = [
    "CachedResponse",
    "CallLedger",
    "RateLimiter",
    "ResponseCache",
    "async_llm",
    "backoff_seconds",
    "govern",
    "request_key",
    "transient_errors",
]
