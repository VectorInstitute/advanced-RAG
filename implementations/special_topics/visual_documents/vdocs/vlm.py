"""Text and image requests through the Vector proxy, throttled, cached, and metered.

Every model step in this topic sends one chat completion with a text part and
zero or more image parts. Images travel as OpenAI-style ``image_url`` parts
holding a JPEG ``data:`` URI, which the proxy passes to every family whose
models accept images.

Each model gets its own governed client, so each has its own request budget
(:class:`~aieng.advanced_rag.graphrag.proxy.RateLimiter`), while all of them
share one response cache. A repeated request, image bytes included, is a cache
hit that costs nothing and is still counted, so a warm re-run reports what the
cold run cost. Every call and every hit reaches
:mod:`aieng.advanced_rag.metering`, which is how
:func:`aieng.advanced_rag.runs.load_or_run` records a step's tokens.

The first request to a paid model opens the process's spend session
(:func:`aieng.advanced_rag.spend.session`), so ``ADVANCED_RAG_SPEND_LIMIT_USD``
holds for every paid call, including those a script sends from a thread pool:
once the estimate passes the ceiling, queued calls raise
:class:`~aieng.advanced_rag.spend.SpendLimitReached` before they are sent.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.graphrag.proxy import RateLimiter, ResponseCache, govern
from aieng.advanced_rag.metering import usage_of
from aieng.advanced_rag.models import request_params


#: Requests per minute for each model. The reference build ran at this rate on
#: the on-prem models, which other jobs share.
DEFAULT_RPM = 20

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def response_cache_path() -> Path:
    """Return the SQLite file this topic's responses are cached in."""
    return Path(cache_root()) / "responses" / "visual_documents.sqlite"


@dataclass(frozen=True)
class Reply:
    """One completion's text and what it cost.

    Parameters
    ----------
    text : str
        The content, empty when the model returned none.
    prompt_tokens : int
        Input tokens the response reported, image tokens included.
    completion_tokens : int
        Output tokens the response reported.
    seconds : float
        Wall time of the call, near zero for a cache hit.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def parse_json(text: str) -> Any:
    """Return the first JSON object in a reply, tolerating fences and prose around it.

    Raises
    ------
    ValueError
        If the reply holds no JSON object.
    """
    match = _JSON.search(text or "")
    if match is None:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(match.group(0))


class ProxyModels:
    """One governed client per model, sharing one response cache.

    Parameters
    ----------
    api_key : str
        The proxy key. It is held by the client and never printed or stored.
    base_url : str
        The proxy endpoint.
    cache_path : Path or None, optional
        The response cache. :func:`response_cache_path` when omitted.
    rpm : int, optional
        Requests per minute for each model.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        cache_path: Path | None = None,
        rpm: int = DEFAULT_RPM,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._cache = ResponseCache(cache_path or response_cache_path())
        self._rpm = rpm
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls,
        *,
        rpm: int = DEFAULT_RPM,
        cache_path: Path | None = None,
        env_path: Path | None = None,
    ) -> ProxyModels:
        """Build from the key and endpoint the onboarding resolver finds.

        ``env_path`` names a ``.env`` file to read when the environment holds no
        key; by default the resolver looks for one above the working directory.
        """
        from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415

        return cls(resolve_api_key(env_path), resolve_base_url(env_path), rpm=rpm, cache_path=cache_path)

    def _client(self, model: str) -> Any:
        with self._lock:
            if model not in self._clients:
                from openai import OpenAI  # noqa: PLC0415

                if spend.is_paid(model):
                    # A script that never opens the spend session would otherwise
                    # send paid calls with no ceiling; opening it here applies
                    # ADVANCED_RAG_SPEND_LIMIT_USD from the first paid call on.
                    spend.session()

                client = OpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0, timeout=300)
                self._clients[model] = govern(client, cache=self._cache, limiter=RateLimiter(self._rpm))
            return self._clients[model]

    def complete(
        self,
        model: str,
        prompt: str,
        images: Sequence[str] = (),
        *,
        max_tokens: int = 600,
        system: str | None = None,
    ) -> Reply:
        """Send one request and return its reply.

        Parameters
        ----------
        model : str
            The model to ask.
        prompt : str
            The text part, which comes after the images.
        images : sequence of str, optional
            ``data:`` URIs, sent in order before the text.
        max_tokens : int, optional
            Completion cap.
        system : str or None, optional
            A system message.

        Returns
        -------
        Reply
            The text and its token counts.
        """
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": uri}} for uri in images]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content if images else prompt})
        params = request_params(model, max_tokens=max_tokens, temperature=0.0)
        # govern switches reasoning off per call from the model's family, so the
        # body is left to it and the cache key stays the same for every family.
        params.pop("extra_body", None)
        started = time.perf_counter()
        response = self._client(model).chat.completions.create(model=model, messages=messages, **params)
        elapsed = time.perf_counter() - started
        usage = usage_of(response)
        text = response.choices[0].message.content if response.choices else None
        return Reply(str(text or ""), usage.prompt_tokens, usage.completion_tokens, elapsed)


__all__ = ["DEFAULT_RPM", "ProxyModels", "Reply", "parse_json", "response_cache_path"]
