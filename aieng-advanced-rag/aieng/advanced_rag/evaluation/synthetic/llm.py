"""The model calls the generator makes, as plain callables with a name attached.

Every stage takes a :class:`Model`: a name, which is recorded next to every
question it touched, and a ``complete(system, prompt) -> str`` callable. Tests
pass a stub; a live run passes :func:`proxy_model`, which sends each call through
:class:`aieng.advanced_rag.batch.ProxyBatch`: infermesh's rate limit, in-flight
window, and transient-only retries, with the shared response cache and ledger in
front, so a re-run costs nothing and a cost figure is always a cold-run figure.
The stages run candidates on a thread pool, and every thread shares the one
client's rate limit.

The generator, the checker, and the judge are separate :class:`Model` values.
Published work on synthetic test sets (Rahmani et al., SIGIR 2024; van Elburg et
al., 2025) reports that a model favours its own family, so the funnel records
whether the checker shares the generator's family.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.models import family_of, request_params


#: ``(system, prompt) -> text``. The one signature every stage calls.
Complete = Callable[[str, str], str]


@dataclass(frozen=True)
class Model:
    """A named model callable.

    Parameters
    ----------
    name : str
        The model identifier, as the endpoint knows it. Recorded on results.
    complete : Complete
        Takes a system prompt and a user prompt and returns the reply text.
    ledger : Any, optional
        Whatever accumulates this model's calls and tokens, read by
        :func:`usage_of`. ``None`` for a stub.
    """

    name: str
    complete: Complete = field(repr=False, compare=False)
    ledger: Any = field(default=None, repr=False, compare=False)

    def __call__(self, system: str, prompt: str) -> str:
        """Return the model's reply."""
        return self.complete(system, prompt)


def family(model_name: str) -> str:
    """Return a coarse model family for the same-family check.

    The rule is the name's leading letters, lowercased: ``Qwen3_8-27B`` and
    ``Qwen3_6-35B-A3B`` are both ``qwen``, ``claude-haiku-5-5`` is ``claude``,
    and ``gpt-5.4-mini`` is ``gpt``.

    Parameters
    ----------
    model_name : str
        A model identifier.

    Returns
    -------
    str
        The family label, or the whole lowercased name when it has no letters.
    """
    match = re.match(r"[A-Za-z]+", model_name)
    return match.group(0).lower() if match else model_name.lower()


def parse_json(raw: str) -> dict[str, Any]:
    """Pull one JSON object out of a model reply, tolerating fences and prose.

    Parameters
    ----------
    raw : str
        The reply text.

    Returns
    -------
    dict
        The first object that parses, or an empty dict when none does. A reply
        that parses to a list or a scalar also returns an empty dict.
    """
    text = (raw or "").strip()
    candidates: list[str] = []
    if "```" in text:
        for part in text.split("```"):
            body = part[4:] if part.lower().startswith("json") else part
            candidates.append(body.strip())
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


class RequestTally:
    """Requests and tokens a build asked for, whether or not the cache answered.

    The ledger counts only uncached calls, so a re-run reports zero. This
    counts every request with the token counts its response carries, which is
    the cost a cold run on the same inputs would pay.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Count one request."""
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def to_dict(self) -> dict[str, int]:
        """Return the totals under ``cold_`` names."""
        return {
            "cold_requests": self.requests,
            "cold_prompt_tokens": self.prompt_tokens,
            "cold_completion_tokens": self.completion_tokens,
        }


def usage_of(model: Model) -> dict[str, Any]:
    """Return a model's call and token totals, for the results file.

    Parameters
    ----------
    model : Model
        A model whose ``ledger`` is a list of objects with ``to_dict``, one
        such object, or ``None``.

    Returns
    -------
    dict
        Every record merged, with the model name added.
    """
    record: dict[str, Any] = {"model": model.name}
    parts = model.ledger if isinstance(model.ledger, (list, tuple)) else [model.ledger]
    for part in parts:
        if part is not None and hasattr(part, "to_dict"):
            record.update(part.to_dict())
    return record


def proxy_model(
    name: str,
    *,
    cache_path: Path,
    per_minute: int = 18,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
    env_path: Path | None = None,
    timeout: float = 180.0,
    max_parallel: int = 8,
) -> Model:
    """Return a cached, throttled, metered model on the proxy.

    Thinking is switched off because reasoning tokens bill as completion tokens
    and make up most of the cost of a short structured call. The key comes from
    :func:`aieng.advanced_rag.proxy.resolve_api_key` when the first uncached call
    is made, and is never printed, so a fully cached run needs no key.

    Parameters
    ----------
    name : str
        The model identifier on the proxy, such as ``Qwen3_8-27B`` or
        ``claude-haiku-5-5``.
    cache_path : Path
        SQLite response cache. Share one file across runs so a re-run is free.
    per_minute : int, optional
        Request cap for this model. Other jobs share the proxy's per-model
        limit, so keep it under what the proxy allows.
    max_tokens : int, optional
        Completion cap per call.
    temperature : float, optional
        Sampling temperature. Zero keeps a cached run reproducible.
    api_key : str or None, optional
        Overrides the resolver.
    base_url : str or None, optional
        Overrides the resolver.
    env_path : Path or None, optional
        A ``.env`` file for the resolver to read when the environment holds no
        key, such as the main checkout's when running from a worktree.
    timeout : float, optional
        Per-request timeout in seconds.
    max_parallel : int, optional
        Requests to this model in flight at once, across every thread.

    Returns
    -------
    Model
        The named callable, with a
        :class:`~aieng.advanced_rag.graphrag.proxy.CallLedger`.
    """
    from aieng.advanced_rag.batch import ProxyBatch
    from aieng.advanced_rag.graphrag.proxy import CallLedger, ResponseCache

    ledger = CallLedger(label=name)
    batch = ProxyBatch(
        name,
        cache=ResponseCache(cache_path),
        rpm=per_minute,
        max_parallel=max_parallel,
        timeout=timeout,
        ledger=ledger,
        api_key=api_key,
        base_url=base_url,
        env_path=env_path,
    )

    # On-prem keeps the classic parameters, so the committed reference caches
    # still hit. A paid family names its completion cap and temperature
    # differently (gpt-6-luna rejects max_tokens), which request_params adapts.
    if family_of(name) in (None, "onprem"):
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    else:
        params = request_params(name, max_tokens=max_tokens, temperature=temperature)

    def complete(system: str, prompt: str) -> str:
        outcome = batch.create(prompt, system=system, **params)
        # The response's own counts, as the cold call reported them, so the
        # tally matches what earlier runs recorded for every family.
        usage = getattr(outcome.response, "usage", None)
        tally.add(
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return outcome.text or ""

    tally = RequestTally()
    return Model(name=name, complete=complete, ledger=[ledger, tally])


__all__ = [
    "Complete",
    "Model",
    "RequestTally",
    "family",
    "parse_json",
    "proxy_model",
    "usage_of",
]
