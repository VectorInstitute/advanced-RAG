"""What model calls cost in dollars, and how much of the key's budget is left.

:mod:`aieng.advanced_rag.metering` counts calls and tokens per model and holds
no price. This module prices them, in two ways that a notebook shows side by
side.

The estimate is per call. :func:`price` multiplies a meter's per-model totals by
the key's rates. The rates come from the proxy's ``/admin/me/models``, which
reports ``cost_per_input_token`` and ``cost_per_output_token`` for each model
the key can use, and that is the rate the key is billed at. When the proxy does
not answer, :data:`OFFLINE_PRICES` stands in, and a model neither source lists
is priced as unknown rather than as zero.

The bill comes from the proxy. A :class:`SpendSession` reads
``/admin/me/key-usage`` and ``/admin/me/key-budget`` when it opens and again in
:meth:`SpendSession.report`, and prints the difference beside the estimate. The
platform issues one key per team and the proxy reports usage per key, so the
difference includes teammates' calls in the same window, and the report says so.
A personal key or another endpoint has no such routes, and the report then
shows the estimate only.

Prompt-cache tokens are priced at the provider's cache rates. A meter's
``prompt_tokens`` counts every input token, and
:attr:`~aieng.advanced_rag.metering.CallUsage.prompt_cache_read_tokens` and
``prompt_cache_write_tokens`` say how many of them the provider read from or
wrote to its cache. Reads bill at :attr:`ModelPrice.cached_input_per_mtok`,
writes at :attr:`ModelPrice.cache_write_per_mtok` (Anthropic's one-hour writes
at :attr:`ModelPrice.cache_write_1h_per_mtok`), and the rest at the input rate.
The proxy's ``/admin/me/models`` lists no cache rates, so they come from
:data:`OFFLINE_PRICES` even when the input and output rates come from the proxy.

Two guards follow. At 50% and at 80% of the key budget spent, :func:`describe`
and :meth:`SpendSession.report` warn with a :class:`BudgetWarning`. The proxy
has no alert endpoint, so the workspace has to. A key that reports no budget is
treated as having :data:`ASSUMED_BUDGET_USD` ($300, a participant team's budget
for the bootcamp), or the dollars in ``ADVANCED_RAG_KEY_BUDGET_USD``, and every
line and warning that uses the figure says it is assumed.
``ADVANCED_RAG_SPEND_LIMIT_USD`` sets a ceiling for one run: the call that takes
the estimate past it raises :class:`SpendLimitReached` once it returns, and every
later call a metered or governed client makes raises it before the request is
sent. The second check is what stops calls already queued on a thread pool.

Every proxy read is a short GET with the participant's bearer key, and the key
is never printed or stored.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeAlias

from aieng.advanced_rag import metering
from aieng.advanced_rag.metering import CallUsage, ModelUsage, UsageMeter
from aieng.advanced_rag.models import ModelSet, family_of


#: Variable that sets a dollar ceiling for one run.
SPEND_LIMIT_ENV = "ADVANCED_RAG_SPEND_LIMIT_USD"

#: Variable that replaces :data:`ASSUMED_BUDGET_USD` for a key that reports no
#: budget. It has no effect when the key reports one.
KEY_BUDGET_ENV = "ADVANCED_RAG_KEY_BUDGET_USD"

#: The budget assumed for a key that reports none: what each participant team's
#: key holds for the whole bootcamp.
ASSUMED_BUDGET_USD = 300.0

#: Fractions of the key budget at which a warning is printed.
WARN_AT: tuple[float, ...] = (0.5, 0.8)

#: The date the offline list prices were read from the providers' pricing
#: pages. Provider prices change, and two in this table are introductory:
#: ``gemini-4-argon`` doubles to $4 / $20 per 1M tokens on 2026-12-01 and
#: ``gemini-3.8-flash`` to $1.50 / $7.50 on 2027-01-01. The proxy's own rates,
#: read at run time, take precedence over this table.
PRICES_AS_OF = "2026-10-20"

_MILLION = 1_000_000


class SpendLimitReached(RuntimeError):  # noqa: N818 - the name the design uses
    """The run's estimated spend passed ``ADVANCED_RAG_SPEND_LIMIT_USD``."""


class BudgetWarning(UserWarning):
    """The key has spent at least 50% or 80% of its budget."""


@dataclass(frozen=True)
class ModelPrice:
    """One model's rate, in US dollars per million tokens.

    Parameters
    ----------
    model : str
        The bare model name.
    input_per_mtok : float
        Price of one million prompt tokens.
    output_per_mtok : float
        Price of one million completion tokens, reasoning included.
    cached_input_per_mtok : float or None, optional
        Price of one million prompt tokens read from the provider's prompt
        cache. None bills reads at the input rate.
    source : str, optional
        ``proxy`` for a rate read from ``/admin/me/models``, ``offline`` for
        :data:`OFFLINE_PRICES`.
    note : str, optional
        A pricing condition the totals cannot apply, such as a surcharge on
        long prompts, which depends on each call's size.
    cache_write_per_mtok : float or None, optional
        Price of one million prompt tokens written to the provider's prompt
        cache, at the default lifetime (five minutes on Anthropic). None bills
        writes at the input rate, which is what a provider with no write fee
        charges.
    cache_write_1h_per_mtok : float or None, optional
        Price of one million tokens written with Anthropic's one-hour lifetime.
        None bills them at ``cache_write_per_mtok``.

    Examples
    --------
    The second of two Anthropic calls that share a 10,803-token system prompt
    reads it from the cache, which costs a tenth of the input rate on Haiku 5.5:

    >>> rate = OFFLINE_PRICES["claude-haiku-5-5"]
    >>> round(rate.cost(10_822, 5, cache_read_tokens=10_803) * 1e6, 2)
    112.43
    """

    model: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    source: str = "offline"
    note: str = ""
    cache_write_per_mtok: float | None = None
    cache_write_1h_per_mtok: float | None = None

    @property
    def free(self) -> bool:
        """Whether both rates are zero, as on Vector's on-prem models."""
        return self.input_per_mtok == 0 and self.output_per_mtok == 0

    def cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        """Return the dollar cost of a token count at this rate.

        Parameters
        ----------
        prompt_tokens : int
            Every input token, cache reads and writes included.
        completion_tokens : int
            Output tokens, reasoning included.
        cache_read_tokens : int, optional
            The share of ``prompt_tokens`` read from the prompt cache.
        cache_write_tokens : int, optional
            The share of ``prompt_tokens`` written to the prompt cache, at any
            lifetime.
        cache_write_1h_tokens : int, optional
            The share of ``cache_write_tokens`` written with a one-hour lifetime.

        Returns
        -------
        float
            US dollars.
        """
        read = min(cache_read_tokens, prompt_tokens)
        write = min(cache_write_tokens, prompt_tokens - read)
        write_1h = min(cache_write_1h_tokens, write)
        plain = prompt_tokens - read - write
        read_rate = _or(self.cached_input_per_mtok, self.input_per_mtok)
        write_rate = _or(self.cache_write_per_mtok, self.input_per_mtok)
        write_1h_rate = _or(self.cache_write_1h_per_mtok, write_rate)
        return (
            plain * self.input_per_mtok
            + read * read_rate
            + (write - write_1h) * write_rate
            + write_1h * write_1h_rate
            + completion_tokens * self.output_per_mtok
        ) / _MILLION

    def cost_of(self, usage: CallUsage | ModelUsage) -> float:
        """Return the dollar cost of one call's or one model's live tokens."""
        return self.cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            cache_read_tokens=usage.prompt_cache_read_tokens,
            cache_write_tokens=usage.prompt_cache_write_tokens,
            cache_write_1h_tokens=usage.prompt_cache_write_1h_tokens,
        )

    def label(self) -> str:
        """Return the rate as one short line."""
        if self.free:
            return "$0 (Vector cluster)"
        return (
            f"${self.input_per_mtok:.2f} in / ${self.output_per_mtok:.2f} out "
            "per 1M tokens"
        )


def _or(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _offline(
    model: str,
    inp: float,
    out: float,
    cached: float | None,
    note: str = "",
    *,
    write: float | None = None,
    write_1h: float | None = None,
) -> tuple[str, ModelPrice]:
    return model, ModelPrice(model, inp, out, cached, "offline", note, write, write_1h)


def _anthropic(
    model: str, inp: float, out: float, cached: float, note: str = ""
) -> tuple[str, ModelPrice]:
    # Anthropic bills a five-minute cache write at 1.25x input and a one-hour
    # write at 2x, on every model.
    return _offline(model, inp, out, cached, note, write=inp * 1.25, write_1h=inp * 2.0)


#: List prices per million tokens (input, output, cache read, and cache write
#: where the provider charges one), read on :data:`PRICES_AS_OF`. The input and
#: output rates are a fallback for when the proxy does not answer; the cache
#: rates are used always, because the proxy lists none. Google charges no cache
#: write fee. OpenAI bills a write at 1.25x input from GPT-5.6 on, and reports
#: it as ``prompt_tokens_details.cache_write_tokens``. Update this table and
#: :data:`aieng.advanced_rag.models.DEFAULT_MODELS` together.
OFFLINE_PRICES: Mapping[str, ModelPrice] = dict(
    [
        _anthropic(
            "claude-haiku-5-5",
            0.10,
            0.50,
            0.01,
            "5x on any request whose prompt exceeds 100K tokens",
        ),
        _anthropic("claude-sonnet-5-5", 2.00, 10.00, 0.10),
        _anthropic("claude-opus-5-5", 4.00, 20.00, 0.20),
        _anthropic("claude-haiku-4-5", 1.00, 5.00, 0.10),
        _anthropic("claude-sonnet-4-6", 3.00, 15.00, 0.30),
        _offline("gemini-3.5-flash-lite", 0.30, 2.50, 0.03),
        _offline("gemini-3.1-flash-lite", 0.25, 1.50, 0.025),
        _offline("gemini-3.8-flash", 0.75, 3.75, 0.075, "introductory price"),
        _offline("gemini-4-argon", 2.00, 10.00, 0.10, "introductory price"),
        _offline(
            "gpt-6-luna",
            0.10,
            0.50,
            0.01,
            "2x input and 1.5x output above 272K prompt tokens",
            write=0.125,
        ),
        _offline("gpt-6.1-sol", 2.00, 10.00, 0.10, write=2.50),
        _offline("gpt-6-astra", 10.00, 50.00, 1.00, write=12.50),
        _offline("Qwen3_6-35B-A3B", 0.0, 0.0, None),
        _offline("Qwen3_8-27B", 0.0, 0.0, None),
        _offline("Qwen3_8-Flash-Next", 0.0, 0.0, None),
        _offline("gpt-oss-120b", 0.0, 0.0, None),
        _offline("Nemotron-3-Nano-Omni-30B-A3B", 0.0, 0.0, None),
    ]
)


def _admin_root(base_url: str) -> str:
    """Return the proxy root that ``/admin`` sits under, from the ``/v1`` URL."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _decimal(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


Fetch = Callable[[str, Mapping[str, str]], Any]


def _http_get_json(url: str, headers: Mapping[str, str], timeout: float) -> Any:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


class ProxyAccount:
    """The three self-service routes of the proxy, read with one key.

    Every method returns None instead of raising when the route does not
    answer: no network, a timeout, an HTTP error, a non-JSON body, or an
    endpoint that is not the Vector proxy. That is what lets every report
    degrade to the estimate alone.

    Parameters
    ----------
    api_key : str or None
        The bearer key. None makes every read return None.
    base_url : str
        The OpenAI-compatible endpoint, normally the proxy's ``/v1`` URL.
    timeout : float, optional
        Seconds per request.
    fetch : callable or None, optional
        ``fetch(url, headers)`` returning parsed JSON, for tests. The default
        uses :mod:`urllib`.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        *,
        timeout: float = 5.0,
        fetch: Fetch | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self._timeout = timeout
        self._fetch = fetch
        self.errors: list[str] = []

    @property
    def reachable(self) -> bool:
        """Whether the account has a key to read with and a proxy endpoint."""
        from aieng.advanced_rag.proxy import points_at_proxy  # noqa: PLC0415

        return bool(self._api_key) and (
            self._fetch is not None or points_at_proxy(self.base_url)
        )

    def _get(self, path: str, query: Mapping[str, str] | None = None) -> Any:
        if not self.reachable:
            return None
        url = _admin_root(self.base_url) + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        try:
            if self._fetch is not None:
                return self._fetch(url, headers)
            return _http_get_json(url, headers, self._timeout)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            # The message names the route and the error class only. An
            # HTTPError's text can echo request details, so it is not kept.
            self.errors.append(f"{path}: {type(error).__name__}")
            return None

    def models(self) -> dict[str, ModelPrice] | None:
        """Return the key's per-model rates from ``/admin/me/models``.

        Returns
        -------
        dict or None
            Model name to :class:`ModelPrice` with ``source="proxy"``, for each
            model whose rates the proxy lists, or None when the route did not
            answer.
        """
        rows = self._get("/admin/me/models")
        if not isinstance(rows, list):
            return None
        prices: dict[str, ModelPrice] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("name"):
                continue
            per_input = _decimal(row.get("cost_per_input_token"))
            per_output = _decimal(row.get("cost_per_output_token"))
            if per_input is None or per_output is None:
                continue
            name = str(row["name"])
            prices[name] = ModelPrice(
                name, per_input * _MILLION, per_output * _MILLION, source="proxy"
            )
        return prices

    def key_usage(self) -> dict[str, Any] | None:
        """Return the key's usage summary from ``/admin/me/key-usage``.

        Returns
        -------
        dict or None
            ``summary`` (with ``total_cost_usd`` and token totals) and
            ``by_model`` as the proxy returns them, or None.
        """
        payload = self._get("/admin/me/key-usage", {"exclude": "daily_by_model"})
        return payload if isinstance(payload, dict) else None

    def key_budget(self) -> float | None:
        """Return the key's budget in dollars, or None when unset or unreachable."""
        payload = self._get("/admin/me/key-budget")
        if not isinstance(payload, Mapping):
            return None
        return _decimal(payload.get("budget_usd"))


def spent_usd(usage: Mapping[str, Any] | None) -> float | None:
    """Return the dollars a ``/admin/me/key-usage`` payload reports spent."""
    if not usage:
        return None
    summary = usage.get("summary")
    if not isinstance(summary, Mapping):
        return None
    return _decimal(summary.get("total_cost_usd"))


@dataclass
class _AccountCache:
    lock: threading.Lock = field(default_factory=threading.Lock)
    account: ProxyAccount | None = None
    prices: dict[str, ModelPrice] | None = None
    prices_read: bool = False


_CACHE = _AccountCache()


def account() -> ProxyAccount:
    """Return the process's proxy account, built from the resolved key once.

    A missing key gives an account whose reads all return None, so every
    report still prints its estimate.

    Returns
    -------
    ProxyAccount
        The shared account.
    """
    with _CACHE.lock:
        if _CACHE.account is None:
            from aieng.advanced_rag.proxy import (  # noqa: PLC0415
                MissingProxyKeyError,
                resolve_api_key,
                resolve_base_url,
            )

            try:
                key: str | None = resolve_api_key()
            except MissingProxyKeyError:
                key = None
            _CACHE.account = ProxyAccount(key, resolve_base_url())
        return _CACHE.account


def use_account(replacement: ProxyAccount | None) -> None:
    """Replace the process's proxy account, or reset it with None.

    Tests and offline runs install a :class:`ProxyAccount` with a stub fetch
    or no key, so nothing reaches the network.
    """
    with _CACHE.lock:
        _CACHE.account = replacement
        _CACHE.prices = None
        _CACHE.prices_read = False


def price_table(source: ProxyAccount | None = None) -> dict[str, ModelPrice]:
    """Return every known rate: the proxy's, over the offline table.

    The proxy's rates are read once per process for the shared account.

    Parameters
    ----------
    source : ProxyAccount or None, optional
        The account to read. :func:`account` when omitted.

    Returns
    -------
    dict
        Model name to :class:`ModelPrice`.
    """
    if source is not None:
        live = source.models()
    else:
        shared = account()
        with _CACHE.lock:
            read = _CACHE.prices_read
            live = _CACHE.prices
        if not read:
            live = shared.models()
            with _CACHE.lock:
                _CACHE.prices, _CACHE.prices_read = live, True
    table = dict(OFFLINE_PRICES)
    for name, rate in (live or {}).items():
        offline = OFFLINE_PRICES.get(name)
        table[name] = ModelPrice(
            name,
            rate.input_per_mtok,
            rate.output_per_mtok,
            offline.cached_input_per_mtok if offline else None,
            "proxy",
            offline.note if offline else "",
            offline.cache_write_per_mtok if offline else None,
            offline.cache_write_1h_per_mtok if offline else None,
        )
    return table


def price_of(
    model: str, table: Mapping[str, ModelPrice] | None = None
) -> ModelPrice | None:
    """Return one model's rate, or None when no source lists it.

    A dated snapshot such as ``claude-haiku-4-5-20251001`` falls back to its
    undated name.
    """
    prices = table if table is not None else price_table()
    if model in prices:
        return prices[model]
    for name, rate in prices.items():
        if model.startswith(name + "-") and model[len(name) + 1 :].isdigit():
            return rate
    return None


@dataclass(frozen=True)
class Estimate:
    """Dollar estimates per model and in total.

    Parameters
    ----------
    by_model : dict
        Model name to dollars, or None for a model with no known rate.
    calls : dict
        Model name to live calls counted.
    """

    by_model: Mapping[str, float | None]
    calls: Mapping[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Return the dollars over every priced model."""
        return sum(value for value in self.by_model.values() if value is not None)

    @property
    def unpriced(self) -> tuple[str, ...]:
        """Return the models no source has a rate for."""
        return tuple(name for name, value in self.by_model.items() if value is None)

    def line(self) -> str:
        """Return the estimate as one line, naming any unpriced model."""
        models = len(self.by_model)
        text = (
            f"est. ${self.total:,.2f} over {models} model{'s' if models != 1 else ''}"
        )
        if self.unpriced:
            text += f" (no price for {', '.join(self.unpriced)})"
        return text


UsageSource: TypeAlias = UsageMeter | Mapping[str, ModelUsage] | Iterable[ModelUsage]


def _usages(source: UsageSource) -> list[ModelUsage]:
    if isinstance(source, UsageMeter):
        return list(source.by_model().values())
    if isinstance(source, Mapping):
        return list(source.values())
    return list(source)


def price(
    source: UsageSource,
    *,
    table: Mapping[str, ModelPrice] | None = None,
    include_cached: bool = False,
) -> Estimate:
    """Price per-model usage at the key's rates.

    Long-prompt surcharges depend on each call's size and totals do not keep
    it, so models with one (see :attr:`ModelPrice.note`) are priced at their
    base rate, which makes the estimate a lower bound for them.

    Parameters
    ----------
    source : UsageMeter, mapping, or iterable of ModelUsage
        A meter, the result of :meth:`UsageMeter.by_model`, or the records.
    table : mapping or None, optional
        Rates to use. :func:`price_table` when omitted.
    include_cached : bool, optional
        Price what the cache hits cost when they were made instead of the live
        calls, which answers "what would a cold run cost". The response cache
        stores prompt and completion totals only, so a replayed call's prompt
        tokens are all priced at the input rate.

    Returns
    -------
    Estimate
        Dollars per model, None for a model with no rate.
    """
    prices = table if table is not None else price_table()
    by_model: dict[str, float | None] = {}
    calls: dict[str, int] = {}
    for usage in _usages(source):
        rate = price_of(usage.model, prices)
        if rate is None:
            by_model[usage.model] = None
        elif include_cached:
            by_model[usage.model] = rate.cost(
                usage.cached_prompt_tokens, usage.cached_completion_tokens
            )
        else:
            by_model[usage.model] = rate.cost_of(usage)
        calls[usage.model] = usage.cache_hits if include_cached else usage.calls
    return Estimate(by_model, calls)


def usage_line(
    source: UsageSource, *, table: Mapping[str, ModelPrice] | None = None
) -> str:
    """Return the one-line cost summary a cell prints after calling models.

    Examples
    --------
    ``592 calls, 434,921 in + 18,000 out tokens on gemini-3.5-flash-lite, est. $0.18``
    """
    usages = [u for u in _usages(source) if u.calls or u.cache_hits]
    if not usages:
        return "no model calls"
    prices = table if table is not None else price_table()
    estimate = price(usages, table=prices)
    calls = sum(u.calls for u in usages)
    prompt = sum(u.prompt_tokens for u in usages)
    completion = sum(u.completion_tokens for u in usages)
    names = ", ".join(u.model for u in usages)
    read = sum(u.prompt_cache_read_tokens for u in usages)
    written = sum(u.prompt_cache_write_tokens for u in usages)
    cache = (
        f" ({read:,} read from and {written:,} written to the prompt cache)"
        if read or written
        else ""
    )
    text = (
        f"{calls:,} calls, {prompt:,} in{cache} + {completion:,} out tokens on "
        f"{names}, {estimate.line()}"
    )
    hits = sum(u.cache_hits for u in usages)
    if hits:
        saved = price(usages, table=prices, include_cached=True)
        text += f"; {hits:,} answered from cache (would have cost ${saved.total:,.2f})"
    return text


def estimate_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    table: Mapping[str, ModelPrice] | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float | None:
    """Return the dollars a token count would cost on ``model``, or None if unpriced.

    The cache counts are shares of ``prompt_tokens``; see :meth:`ModelPrice.cost`.
    """
    rate = price_of(model, table)
    if rate is None:
        return None
    return rate.cost(
        prompt_tokens,
        completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
    )


@dataclass(frozen=True)
class Budget:
    """The key's budget and what it has spent, as the proxy reports them.

    Parameters
    ----------
    budget_usd : float or None
        The key's budget. None when the key inherits its project's limit or the
        proxy did not answer.
    spent_usd : float or None
        Dollars spent on the key over its lifetime.
    assumed : bool, optional
        True when ``budget_usd`` is the assumed figure from :meth:`assuming`
        and the key itself reports no budget.
    """

    budget_usd: float | None
    spent_usd: float | None
    assumed: bool = False

    def assuming(self) -> Budget:
        """Return this budget, or the assumed one when the key reports none.

        The assumed budget is ``ADVANCED_RAG_KEY_BUDGET_USD`` when set, and
        :data:`ASSUMED_BUDGET_USD` otherwise.

        Raises
        ------
        ValueError
            When ``ADVANCED_RAG_KEY_BUDGET_USD`` is not a number of dollars.
        """
        if self.budget_usd is not None:
            return self
        return Budget(_assumed_budget(), self.spent_usd, assumed=True)

    @property
    def left_usd(self) -> float | None:
        """Return the dollars left, or None when either figure is unknown.

        An assumed budget with no reported spend counts the spend as $0, so the
        confirmation guard still has a figure to compare against.
        """
        if self.budget_usd is None:
            return None
        if self.spent_usd is None:
            return self.budget_usd if self.assumed else None
        return self.budget_usd - self.spent_usd

    @property
    def fraction_spent(self) -> float | None:
        """Return the share of the budget spent, or None when it has no meaning.

        A $0 budget means free models only, so it has no share.
        """
        if not self.budget_usd or self.spent_usd is None:
            return None
        return self.spent_usd / self.budget_usd

    def line(self) -> str:
        """Return the budget as one line."""
        if self.assumed and self.budget_usd is not None:
            assumed = (
                f"assuming ${self.budget_usd:,.2f} (set {KEY_BUDGET_ENV} to override)"
            )
            if self.spent_usd is None:
                return f"key budget unknown (the proxy did not answer); {assumed}"
            left = self.budget_usd - self.spent_usd
            return (
                f"key budget not set on the key; {assumed}, spent "
                f"${self.spent_usd:,.2f}, left ${left:,.2f}"
            )
        if self.budget_usd is None:
            if self.spent_usd is None:
                return "key budget unknown (the proxy did not answer)"
            return f"key budget not set on the key, spent ${self.spent_usd:,.2f}"
        if self.budget_usd == 0:
            return "key budget $0.00: on-prem models only, at no charge"
        spent = self.spent_usd or 0.0
        left = self.budget_usd - spent
        share = left / self.budget_usd
        return (
            f"key budget ${self.budget_usd:,.2f}, spent ${spent:,.2f}, "
            f"left ${left:,.2f} ({share:.0%})"
        )


def read_budget(source: ProxyAccount | None = None) -> Budget:
    """Return the key's budget and lifetime spend.

    A key that reports no budget gets the assumed one from
    :meth:`Budget.assuming`. The spend is None when the proxy is silent.
    """
    reader = source if source is not None else account()
    return Budget(reader.key_budget(), spent_usd(reader.key_usage())).assuming()


def budget_warnings(budget: Budget, already: set[float] | None = None) -> list[str]:
    """Return a warning for each threshold in :data:`WARN_AT` the key has crossed.

    Parameters
    ----------
    budget : Budget
        What the proxy reports, or an assumed budget from
        :meth:`Budget.assuming`.
    already : set of float or None, optional
        Thresholds warned about before. Crossed thresholds are added to it, so
        a session warns once per threshold.

    Returns
    -------
    list of str
        One message per newly crossed threshold, highest first.
    """
    fraction = budget.fraction_spent
    if fraction is None:
        return []
    seen = already if already is not None else set()
    messages = []
    of_budget = (
        f"an assumed ${budget.budget_usd:,.2f} budget (the key reports none; set "
        f"{KEY_BUDGET_ENV} to override)"
        if budget.assumed
        else f"its ${budget.budget_usd:,.2f} budget"
    )
    for threshold in sorted(WARN_AT, reverse=True):
        if fraction >= threshold and threshold not in seen:
            messages.append(
                f"the key has spent {fraction:.0%} of {of_budget}, past the "
                f"{threshold:.0%} mark; check estimates before rebuilding an "
                "artifact"
            )
    seen.update(threshold for threshold in WARN_AT if fraction >= threshold)
    return messages[:1]


def _warn(messages: Iterable[str]) -> None:
    for message in messages:
        warnings.warn(message, BudgetWarning, stacklevel=3)


def describe(
    models: ModelSet,
    *,
    source: ProxyAccount | None = None,
    table: Mapping[str, ModelPrice] | None = None,
) -> str:
    """Return the model set with each model's price and the key's budget.

    Warns with :class:`BudgetWarning` past 50% or 80% of the budget, or of
    the assumed budget when the key reports none.

    Parameters
    ----------
    models : ModelSet
        The set from :func:`aieng.advanced_rag.models.choose_models`.
    source : ProxyAccount or None, optional
        The account to read. :func:`account` when omitted.
    table : mapping or None, optional
        Rates to use. :func:`price_table` when omitted.

    Returns
    -------
    str
        A short table, one line per role.
    """
    prices = table if table is not None else price_table(source)
    width = max([len(choice.model) for choice in models] + [len(models.embeddings)])
    lines = [f"family {models.family}"]
    for choice in models:
        rate = price_of(choice.model, prices)
        cost = (
            "price unknown (not listed by the proxy)" if rate is None else rate.label()
        )
        if rate is not None and rate.note:
            cost += f"; {rate.note}"
        lines.append(f"  {choice.role:<11} {choice.model:<{width}}  {cost}")
    lines.append(f"  {'embeddings':<11} {models.embeddings:<{width}}  local")
    budget = read_budget(source)
    lines.append(budget.line())
    _warn(budget_warnings(budget))
    return "\n".join(lines)


class SpendSession:
    """One notebook run's spend: the estimate, the proxy's bill, and the guards.

    Open one in the model cell and call :meth:`report` in the last cell. The
    session snapshots the key's usage when it opens, so the report can print
    the proxy's figure for this run beside the estimate. It also installs the
    spend ceiling from ``ADVANCED_RAG_SPEND_LIMIT_USD`` or ``limit_usd``.

    Parameters
    ----------
    meter : UsageMeter or None, optional
        The meter whose per-model totals are priced.
        :func:`~aieng.advanced_rag.metering.default_meter` when omitted.
    limit_usd : float or None, optional
        Ceiling for the run's estimate. Read from ``ADVANCED_RAG_SPEND_LIMIT_USD``
        when omitted; no ceiling when neither is set.
    source : ProxyAccount or None, optional
        The account to read. :func:`account` when omitted.
    table : mapping or None, optional
        Rates to use. :func:`price_table` when omitted.
    """

    def __init__(
        self,
        meter: UsageMeter | None = None,
        *,
        limit_usd: float | None = None,
        source: ProxyAccount | None = None,
        table: Mapping[str, ModelPrice] | None = None,
    ) -> None:
        self.meter = meter if meter is not None else metering.default_meter()
        self._source = source
        self._table = dict(table) if table is not None else None
        self.opened_at = datetime.now(timezone.utc)
        self._warned: set[float] = set()
        self._lock = threading.Lock()
        self._live_usd = 0.0
        self.limit_usd = limit_usd if limit_usd is not None else _limit_from_env()
        self._start_spent = spent_usd(self._account().key_usage())
        if self.limit_usd is not None:
            metering.add_call_hook(self._guard)
            metering.add_pre_call_hook(self._refuse)

    def _account(self) -> ProxyAccount:
        return self._source if self._source is not None else account()

    def _prices(self) -> Mapping[str, ModelPrice]:
        if self._table is None:
            self._table = price_table(self._source)
        return self._table

    def _guard(self, model: str | None, usage: CallUsage) -> None:
        """Add one live call's price, raising once the ceiling is passed."""
        if model is None or self.limit_usd is None:
            return
        rate = price_of(model, self._prices())
        cost = None if rate is None else rate.cost_of(usage)
        with self._lock:
            self._live_usd += cost or 0.0
            spent = self._live_usd
        if spent > self.limit_usd:
            raise SpendLimitReached(
                f"this run's estimate is ${spent:,.2f}, past the "
                f"${self.limit_usd:,.2f} set by {SPEND_LIMIT_ENV}; raise the "
                "limit or unset it to continue"
            )

    def _refuse(self, model: str | None) -> None:
        """Refuse a call before it is sent once the ceiling has been passed."""
        if self.limit_usd is None:
            return
        with self._lock:
            spent = self._live_usd
        if spent > self.limit_usd:
            raise SpendLimitReached(
                f"this run's estimate is ${spent:,.2f}, past the "
                f"${self.limit_usd:,.2f} set by {SPEND_LIMIT_ENV}; the call to "
                f"{model or 'the model'} was not sent"
            )

    @property
    def estimated_usd(self) -> float:
        """Return the guard's running estimate of live calls since opening."""
        with self._lock:
            return self._live_usd

    def close(self) -> None:
        """Remove the spend ceiling."""
        metering.remove_call_hook(self._guard)
        metering.remove_pre_call_hook(self._refuse)

    def __enter__(self) -> SpendSession:
        """Return the session, so it can scope a block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove the spend ceiling at the end of the block."""
        self.close()

    def report(self) -> str:
        """Return the run's estimate beside the proxy's figure and the budget.

        Warns with :class:`BudgetWarning` past 50% or 80% of the budget, or
        of the assumed budget when the key reports none but its spend.

        Returns
        -------
        str
            Three short lines, or the estimate alone when the proxy is silent.
        """
        usages = [u for u in self.meter.by_model().values() if u.calls or u.cache_hits]
        estimate = price(usages, table=self._prices())
        lines = [f"this run, estimated   {estimate.line()}"]
        source = self._account()
        usage = source.key_usage()
        now_spent = spent_usd(usage)
        if now_spent is not None and self._start_spent is not None:
            lines.append(
                f"this run, proxy usage ${now_spent - self._start_spent:,.2f} "
                "(team key: includes teammates' calls in the same window)"
            )
        reported = Budget(source.key_budget(), now_spent)
        if reported.budget_usd is None and now_spent is None:
            # A silent proxy gives no spend to set against an assumed budget.
            return "\n".join(lines)
        budget = reported.assuming()
        lines.append(budget.line())
        _warn(budget_warnings(budget, self._warned))
        return "\n".join(lines)


def _assumed_budget() -> float:
    raw = os.environ.get(KEY_BUDGET_ENV, "").strip()
    if not raw:
        return ASSUMED_BUDGET_USD
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(
            f"{KEY_BUDGET_ENV} must be a number of dollars, got {raw!r}"
        ) from error


def _limit_from_env() -> float | None:
    raw = os.environ.get(SPEND_LIMIT_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(
            f"{SPEND_LIMIT_ENV} must be a number of dollars, got {raw!r}"
        ) from error


@dataclass
class _Session:
    lock: threading.Lock = field(default_factory=threading.Lock)
    session: SpendSession | None = None


_SESSION = _Session()


def session(**kwargs: Any) -> SpendSession:
    """Return the process's spend session, opening it on first use.

    Parameters
    ----------
    **kwargs
        Passed to :class:`SpendSession` when the session opens. Ignored after.

    Returns
    -------
    SpendSession
        The shared session.
    """
    with _SESSION.lock:
        if _SESSION.session is None:
            _SESSION.session = SpendSession(**kwargs)
        return _SESSION.session


def reset_session() -> None:
    """Close and forget the process's spend session."""
    with _SESSION.lock:
        if _SESSION.session is not None:
            _SESSION.session.close()
        _SESSION.session = None


def report() -> str:
    """Return the shared session's report: estimate, proxy usage, and budget.

    The last cell of a notebook that calls a model prints this.
    """
    return session().report()


def is_paid(model: str) -> bool:
    """Return whether a model is in a paid family."""
    family = family_of(model)
    return family is not None and family != "onprem"


__all__ = [
    "ASSUMED_BUDGET_USD",
    "KEY_BUDGET_ENV",
    "OFFLINE_PRICES",
    "PRICES_AS_OF",
    "SPEND_LIMIT_ENV",
    "WARN_AT",
    "Budget",
    "BudgetWarning",
    "Estimate",
    "ModelPrice",
    "ProxyAccount",
    "SpendLimitReached",
    "SpendSession",
    "account",
    "budget_warnings",
    "describe",
    "estimate_tokens",
    "is_paid",
    "price",
    "price_of",
    "price_table",
    "read_budget",
    "report",
    "reset_session",
    "session",
    "spent_usd",
    "usage_line",
    "use_account",
]
