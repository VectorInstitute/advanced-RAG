"""Chat completions through the Vector proxy: cached, metered, thinking off.

Three things this run depends on.

Reasoning is disabled. Calls go through
:class:`aieng.advanced_rag.batch.ProxyBatch`, which adds each model family's
thinking-off switch, holds every worker to one rate limit, and retries only
transient errors. The default bulk model on the proxy bills reasoning as
completion tokens, at 1,312 of them against 16 for the same answer, and it
returns no content when ``max_tokens`` runs out mid-reasoning. A sweep makes
tens of thousands of short structured calls, so leaving it on would make the
run a measurement of reasoning overhead.

Every response is cached on the exact request. A harvest is the expensive step
and it is run more than once while the sweep is being built, so keying on the
request means a re-run costs nothing and stays comparable. Cached hits are
counted as logical calls, because the trace records what a cold run would spend,
and they are excluded from the network totals, because that is the real bill.
The batch keeps the cache under this run's own key shape, so caches written by
earlier sweeps still hit.

There is deliberately no embedding path here. Embeddings run locally on
sentence-transformers. The proxy hosts no embedding model of its own, so an
embedding call through it bills an external provider.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.batch import KeyFunction, ProxyBatch
from aieng.advanced_rag.graphrag.proxy import ResponseCache
from aieng.advanced_rag.models import REFERENCE_MODELS, choose_models, request_params
from aieng.advanced_rag.proxy import PROXY_BASE_URL, resolve_api_key, resolve_base_url


DEFAULT_BASE_URL = PROXY_BASE_URL
DEFAULT_MODEL = "Qwen3_6-35B-A3B"


def load_api_key(env_path: Path | None = None) -> str:
    """Read the proxy key from the environment, falling back to a repo ``.env``.

    The lookup is :func:`aieng.advanced_rag.proxy.resolve_api_key`, which every
    runner shares.

    Parameters
    ----------
    env_path : Path or None, optional
        An explicit ``.env`` to read. ``None`` walks up from this file to the
        repository root.

    Returns
    -------
    str
        The key.

    Raises
    ------
    RuntimeError
        If no key is set and no ``.env`` holds one.
    """
    key: str = resolve_api_key(env_path, start=Path(__file__).parent)
    return key


def sweep_key(model: str) -> KeyFunction:
    """Return the cache key function earlier sweeps wrote their caches with.

    Parameters
    ----------
    model : str
        The model the key is for.

    Returns
    -------
    callable
        ``key(messages, params) -> str``, as :class:`ProxyBatch` takes it.
    """

    def key(messages: list[dict[str, Any]], params: Mapping[str, Any]) -> str:
        request: dict[str, Any] = {
            "model": model,
            "prompt": messages[-1]["content"],
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
        }
        if model != REFERENCE_MODELS.generation.model:
            request["request_params"] = request_params(
                model, max_tokens=params["max_tokens"], temperature=params["temperature"], thinking=False
            )
        digest: str = ResponseCache.key(request)
        return digest

    return key


class CachedCompleter:
    """One worker's prompt-to-text callable, counting what it asks for.

    Each worker holds its own instance so the per-stage call attribution in
    ``harvest_traces`` stays correct under concurrency: a shared counter would
    be read across threads and price the wrong stage.

    Parameters
    ----------
    batch : ProxyBatch
        The shared proxy connection and cache. Workers share its rate limit.
    max_tokens : int, optional
        Completion cap.
    temperature : float, optional
        Sampling temperature.

    Attributes
    ----------
    calls : int
        Logical calls, cached or not. This is what a cold run would cost and
        what the trace records.
    network_calls : int
        Calls that actually reached the proxy. This is the bill.
    empty_responses : int
        Calls that came back with no content.
    failures : int
        Transient errors the batch retried.
    cache_hits : int
        Calls the cache answered.
    """

    def __init__(self, batch: ProxyBatch, max_tokens: int = 512, temperature: float = 0.0) -> None:
        self.batch = batch
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls = 0
        self.network_calls = 0
        self.empty_responses = 0
        self.failures = 0
        self.cache_hits = 0

    def __call__(self, prompt: str) -> str:
        """Send one prompt and return its text.

        Raises
        ------
        RuntimeError
            When the response has no content, because a sweep that scored an
            empty answer would be incomplete.
        """
        self.calls += 1
        outcome = self.batch.create(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        text = outcome.text or ""
        if outcome.cached:
            if not text.strip():
                raise ValueError("cached completion is empty; sweep is incomplete")
            # A hit reaches no model; the batch reports it to the cost meter
            # apart from the calls, so a re-run does not look free.
            self.cache_hits += 1
            return text
        self.network_calls += 1
        self.failures += outcome.metrics.retries if outcome.metrics is not None else 0
        if not text.strip():
            self.empty_responses += 1
            raise RuntimeError("proxy returned an empty completion; sweep is incomplete")
        return text


class ProxyClient:
    """The shared proxy connection, and a completer factory over it.

    Parameters
    ----------
    model : str, optional
        Model name to send. Bare, with no provider prefix.
    base_url : str, optional
        Proxy endpoint.
    api_key : str or None, optional
        ``None`` reads the key from the environment or the repository ``.env``.
    cache_path : Path or None, optional
        SQLite cache file. ``None`` disables caching.
    max_tokens : int, optional
        Completion cap.
    temperature : float, optional
        Zero removes sampling variance only. Serving backends may remain
        nondeterministic, and some providers restrict this setting.
    timeout : float, optional
        Per-request timeout in seconds. A stalled request holds a worker, and a
        sweep runs many of them, so it times out and is retried instead.
    """

    def __init__(  # noqa: PLR0917 - retain the existing public positional API
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_path: Path | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> None:
        spend.session()
        self.model = model or choose_models().generation.model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = ResponseCache(cache_path) if cache_path else None
        self.batch = ProxyBatch(
            self.model,
            cache=self.cache,
            key=sweep_key(self.model),
            payload="text",
            base_url=base_url or resolve_base_url(start=Path(__file__).parent),
            api_key=api_key or load_api_key(),
            timeout=timeout,
        )
        self._completers: list[CachedCompleter] = []
        self._lock = threading.Lock()

    def completer(self) -> CachedCompleter:
        """Return a fresh per-worker completer over the shared connection."""
        completer = CachedCompleter(self.batch, self.max_tokens, self.temperature)
        with self._lock:
            self._completers.append(completer)
        return completer

    def usage(self) -> dict[str, int]:
        """Return the totals across every completer this client handed out."""
        return {
            "logical_calls": sum(c.calls for c in self._completers),
            "network_calls": sum(c.network_calls for c in self._completers),
            "empty_responses": sum(c.empty_responses for c in self._completers),
            "retried_failures": sum(c.failures for c in self._completers),
            "cache_hits": sum(c.cache_hits for c in self._completers),
        }
