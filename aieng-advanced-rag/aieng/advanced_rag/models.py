"""Which model answers, judges, and extracts, and how to ask each family for it.

A participant picks a provider family once, at the top of a notebook, and every
model-produced step reads its model from the set that comes back::

    from aieng.advanced_rag.models import choose_models

    MODELS = choose_models()  # or choose_models("anthropic")
    MODELS.generation.model

Three roles, because their call patterns differ. Extraction builds graphs and
indexes from tens of thousands of short structured calls, so it takes the
family's cheapest model. Generation answers one question per call and takes the
same cheap tier. Judging is one call per question, and every paid family
shares one mid-tier judge, :data:`SHARED_JUDGE`, so scores compare across teams
whichever family generated the answers. The frontier tier of each family is
opt-in only: name it as a per-role override.
Embeddings stay local in every family.

:data:`DEFAULT_MODELS` is the one table of default IDs. A pricing review changes
that table and nothing else. Prices live in :mod:`aieng.advanced_rag.spend`.

:func:`request_params` is the per-family request adapter, the one place the
parameter differences between providers live. Every completer in the package
asks it for the keyword arguments of a chat completion, so a change to how the
proxy translates a field is a one-line change here. A family's models can
differ, so :data:`MODEL_PROFILES` overrides :data:`REQUEST_PROFILES` per model.
An entry marked confirmed has passed a request through the proxy's
``/v1/chat/completions``; the Google entry has not, because the proxy has no
Google provider key configured for the development project.

:func:`cacheable_system` builds a system message that the provider's prompt
cache can reuse. Anthropic caches through the proxy only when the system
message is a list of text parts carrying ``cache_control``. A plain string does
not cache, and neither does ``cache_control`` on a user content part. OpenAI
caches a repeated prefix on its own, so every other family gets a plain string.

Prior art and why this design: LiteLLM's ``completion`` and LangChain's chat
models translate one request shape into each provider's fields, and the
Anthropic SDK and LangChain's ``ChatAnthropic`` take ``cache_control`` on text
parts in the list form :func:`cacheable_system` writes. The Vector proxy already
does the translation on its side (it maps ``reasoning_effort`` to Anthropic's
``output_config.effort``), and the package talks to it with the OpenAI SDK, so
adding a translation library would translate twice. What is custom is the
per-model table, because the accepted values differ inside one family and no
library records what this proxy accepts.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from aieng.advanced_rag.proxy import resolve_setting


Family = Literal["anthropic", "google", "openai", "onprem"]
Role = Literal["generation", "judging", "extraction"]

FAMILIES: tuple[Family, ...] = ("anthropic", "google", "openai", "onprem")
ROLES: tuple[Role, ...] = ("generation", "judging", "extraction")

#: The family used when neither an argument nor the environment names one. The
#: candidates are ``google``, the only paid family proven on the proxy over two
#: earlier cohorts, and ``openai``, the cheapest. This stays ``google`` until the
#: per-family smoke test passes for OpenAI and a choice is made.
DEFAULT_FAMILY: Family = "google"

#: The local embedding model every family uses. No family sends embeddings to a
#: paid API.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: Variable that picks the family.
FAMILY_ENV = "ADVANCED_RAG_MODEL_FAMILY"

#: Variables that override one role each.
ROLE_ENV: Mapping[Role, str] = {
    "generation": "ADVANCED_RAG_GENERATION_MODEL",
    "judging": "ADVANCED_RAG_JUDGE_MODEL",
    "extraction": "ADVANCED_RAG_EXTRACTION_MODEL",
}

#: The judge every paid family defaults to. One judge for every team keeps scores
#: comparable across families. A team that judges with another model compares
#: only with teams that made the same choice.
SHARED_JUDGE = "gemini-3.8-flash"

#: The default model for each role in each family. Generation and extraction use
#: the family's cheap tier, and every paid family judges with :data:`SHARED_JUDGE`.
#: ``onprem`` is the set the reference artifacts were built with on Vector's
#: cluster, at no charge.
DEFAULT_MODELS: Mapping[Family, Mapping[Role, str]] = {
    "anthropic": {
        "generation": "claude-haiku-5-5",
        "judging": SHARED_JUDGE,
        "extraction": "claude-haiku-5-5",
    },
    "google": {
        "generation": "gemini-3.5-flash-lite",
        "judging": SHARED_JUDGE,
        "extraction": "gemini-3.5-flash-lite",
    },
    "openai": {
        "generation": "gpt-6-luna",
        "judging": SHARED_JUDGE,
        "extraction": "gpt-6-luna",
    },
    "onprem": {
        "generation": "Qwen3_6-35B-A3B",
        "judging": "Qwen3_6-35B-A3B",
        "extraction": "Qwen3_6-35B-A3B",
    },
}

#: The frontier model of each paid family. No role defaults to one; pass it as
#: an override, for example ``choose_models(judging="claude-opus-5-5")``.
FRONTIER_MODELS: Mapping[Family, str] = {
    "anthropic": "claude-opus-5-5",
    "google": "gemini-4-argon",
    "openai": "gpt-6-astra",
}

#: Other on-prem models the proxy serves at no charge. ``Qwen3_8-27B`` built the
#: ServerFault reference graph.
ONPREM_MODELS: tuple[str, ...] = (
    "Qwen3_6-35B-A3B",
    "Qwen3_8-27B",
    "Qwen3_8-Flash-Next",
    "gpt-oss-120b",
    "Nemotron-3-Nano-Omni-30B-A3B",
)

Endpoint = Literal["vector-cluster", "vector-proxy-paid", "unknown"]


class UnlistedModelWarning(UserWarning):
    """A model outside the default sets was chosen; it works if the proxy grants it."""


# On-prem names are matched first, because ``gpt-oss-120b`` would otherwise read
# as an OpenAI model.
_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], Family], ...] = (
    (re.compile(r"^(qwen|gpt-oss|nemotron)", re.IGNORECASE), "onprem"),
    (re.compile(r"^claude-", re.IGNORECASE), "anthropic"),
    (re.compile(r"^gemini-", re.IGNORECASE), "google"),
    (re.compile(r"^(gpt-|o\d)", re.IGNORECASE), "openai"),
)


def family_of(model: str) -> Family | None:
    """Return the provider family a model name belongs to.

    Parameters
    ----------
    model : str
        A bare model name as the proxy lists it.

    Returns
    -------
    Family or None
        The family, or None for a name no pattern matches, such as a model on
        a self-hosted endpoint.

    Examples
    --------
    >>> family_of("claude-haiku-5-5")
    'anthropic'
    >>> family_of("gpt-oss-120b")
    'onprem'
    >>> family_of("my-local-model") is None
    True
    """
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.match(model.strip()):
            return family
    return None


def endpoint_of(model: str) -> Endpoint:
    """Return the endpoint class a model runs on, for cache keys and labels.

    One name can route to different deployments, so the class is part of every
    model-produced cache key.

    Parameters
    ----------
    model : str
        A bare model name.

    Returns
    -------
    str
        ``vector-cluster`` for the free on-prem models, ``vector-proxy-paid``
        for a paid family, and ``unknown`` otherwise.
    """
    family = family_of(model)
    if family is None:
        return "unknown"
    return "vector-cluster" if family == "onprem" else "vector-proxy-paid"


@dataclass(frozen=True)
class RequestProfile:
    """How one family wants a chat completion's parameters sent.

    Parameters
    ----------
    token_param : str
        Name of the completion cap: ``max_tokens``, or ``max_completion_tokens``
        for models that reject the older name.
    send_temperature : bool
        Whether to send ``temperature``. Families whose reasoning models reject a
        non-default value leave it out, so the provider default applies.
    thinking_off : dict
        Keyword arguments that turn reasoning off. ``extra_body`` is merged into
        the request body by the OpenAI client.
    thinking_on : dict
        Keyword arguments that leave reasoning on. Empty means the model default.
    cache_control : bool
        Whether the provider caches a prompt prefix only when the request marks
        it with ``cache_control``. :func:`cacheable_system` reads this.
    confirmed : bool
        Whether a request with this profile has passed through the proxy.
    """

    token_param: str = "max_tokens"
    send_temperature: bool = True
    thinking_off: Mapping[str, Any] | None = None
    thinking_on: Mapping[str, Any] | None = None
    cache_control: bool = False
    confirmed: bool = False


#: The per-family request differences, which apply to any model of the family
#: that :data:`MODEL_PROFILES` does not list. ``onprem`` is measured: vLLM
#: honours ``chat_template_kwargs`` and ignores ``reasoning_effort``. The Google
#: entry follows what the forecasting bootcamp found on this proxy: ``disable``
#: and ``low`` return 400 for Flash-Lite, ``minimal`` passes. The Anthropic and
#: OpenAI entries are the fields older models in each family document; the
#: models the defaults use are in :data:`MODEL_PROFILES`.
REQUEST_PROFILES: Mapping[Family, RequestProfile] = {
    "onprem": RequestProfile(
        thinking_off={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
        confirmed=True,
    ),
    "google": RequestProfile(
        thinking_off={"extra_body": {"reasoning_effort": "minimal"}},
    ),
    "anthropic": RequestProfile(
        # Claude models from 4.7 on reject a temperature other than the default.
        # The version check is in `_sends_temperature`.
        thinking_off={"extra_body": {"thinking": {"type": "disabled"}}},
        cache_control=True,
    ),
    "openai": RequestProfile(
        token_param="max_completion_tokens",
        send_temperature=False,
        thinking_off={"reasoning_effort": "none"},
    ),
}

_OPENAI = REQUEST_PROFILES["openai"]
_ANTHROPIC = REQUEST_PROFILES["anthropic"]

#: Per-model profiles, for models whose family profile does not fit. Each entry
#: was confirmed with a request through the proxy's ``/v1/chat/completions``.
#: A dated snapshot such as ``claude-haiku-5-5-20261001`` takes its undated
#: name's entry.
MODEL_PROFILES: Mapping[str, RequestProfile] = {
    # With no thinking field Haiku 5.5 thinks by default, and at max_tokens=10
    # its whole budget went to thinking and the text came back empty.
    "claude-haiku-5-5": replace(_ANTHROPIC, confirmed=True),
    # Sonnet 5.5 rejects "disabled" and names "between_tools" as its off
    # switch: no thinking before the answer, only short updates between tool
    # calls, which come back as thinking blocks.
    "claude-sonnet-5-5": replace(
        _ANTHROPIC,
        thinking_off={"extra_body": {"thinking": {"type": "between_tools"}}},
        confirmed=True,
    ),
    # Opus 5.5 rejects "disabled" and takes adaptive thinking with an effort.
    # The proxy maps `reasoning_effort` to Anthropic's `output_config.effort`,
    # whose values are low, medium, high, xhigh, and max. An `output_config`
    # sent in `extra_body` is dropped without an error, so it cannot be used.
    "claude-opus-5-5": replace(
        _ANTHROPIC,
        thinking_off={
            "reasoning_effort": "low",
            "extra_body": {"thinking": {"type": "adaptive"}},
        },
        confirmed=True,
    ),
    "gpt-6-luna": replace(_OPENAI, confirmed=True),
    # Sol and Astra reject "none"; they take low, medium, high, and xhigh. On a
    # trivial prompt "low" spent 0 reasoning tokens on both.
    "gpt-6.1-sol": replace(
        _OPENAI, thinking_off={"reasoning_effort": "low"}, confirmed=True
    ),
    "gpt-6-astra": replace(
        _OPENAI, thinking_off={"reasoning_effort": "low"}, confirmed=True
    ),
}

# A name no family matches is most likely a self-hosted vLLM model, which takes
# the on-prem switch and the classic parameters.
_UNKNOWN_PROFILE = RequestProfile(
    thinking_off=REQUEST_PROFILES["onprem"].thinking_off,
)

_CLAUDE_VERSION = re.compile(r"^claude-[a-z]+-(\d+)(?:-(\d+))?", re.IGNORECASE)


def profile_for(model: str) -> RequestProfile:
    """Return the request profile for a model.

    Parameters
    ----------
    model : str
        A bare model name.

    Returns
    -------
    RequestProfile
        The model's own entry in :data:`MODEL_PROFILES`, else its family's
        profile, else a vLLM-shaped default for an unknown name.

    Examples
    --------
    >>> profile_for("claude-sonnet-5-5").thinking_off
    {'extra_body': {'thinking': {'type': 'between_tools'}}}
    >>> profile_for("claude-haiku-4-5-20251001").confirmed
    False
    """
    name = model.strip()
    if name in MODEL_PROFILES:
        return MODEL_PROFILES[name]
    for listed, profile in MODEL_PROFILES.items():
        suffix = name[len(listed) + 1 :]
        if name.startswith(listed + "-") and suffix.isdigit():
            return profile
    family = family_of(name)
    return _UNKNOWN_PROFILE if family is None else REQUEST_PROFILES[family]


def _sends_temperature(model: str, profile: RequestProfile) -> bool:
    if family_of(model) != "anthropic":
        return profile.send_temperature
    match = _CLAUDE_VERSION.match(model)
    if match is None:
        return False
    major = int(match.group(1))
    # A date suffix such as -20251001 is not a minor version.
    minor = int(match.group(2)) if match.group(2) and len(match.group(2)) <= 2 else 0
    return (major, minor) < (4, 7)


def request_params(
    model: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    thinking: bool = False,
) -> dict[str, Any]:
    """Return the keyword arguments of one chat completion for this model's family.

    The result goes straight into ``client.chat.completions.create(model=...,
    messages=..., **params)``. It is also exactly what was sent, so a cache key
    built from it names the thinking switch and the effort as sent.

    Parameters
    ----------
    model : str
        A bare model name.
    max_tokens : int
        Completion cap, sent under the family's parameter name.
    temperature : float, optional
        Sampling temperature, left out for families that reject it.
    thinking : bool, optional
        Leave reasoning on. Off by default, because reasoning is billed as
        completion tokens and on short structured calls it is most of the bill.

    Returns
    -------
    dict
        Keyword arguments, with ``extra_body`` only when the family needs one.

    Examples
    --------
    >>> request_params("Qwen3_6-35B-A3B", max_tokens=64)["extra_body"]
    {'chat_template_kwargs': {'enable_thinking': False}}
    >>> sorted(request_params("gpt-6-luna", max_tokens=64))
    ['max_completion_tokens', 'reasoning_effort']
    """
    profile = profile_for(model)
    params: dict[str, Any] = {profile.token_param: max_tokens}
    if _sends_temperature(model, profile):
        params["temperature"] = temperature
    switch = profile.thinking_on if thinking else profile.thinking_off
    for name, value in (switch or {}).items():
        params[name] = dict(value) if isinstance(value, Mapping) else value
    return params


def thinking_body(model: str) -> dict[str, Any]:
    """Return the request-body fields that turn reasoning off for this model.

    :func:`aieng.advanced_rag.reasoning.disable_thinking` injects this per call
    as ``extra_body``, so a client shared between an on-prem and a paid model
    sends each model its own switch. A top-level keyword argument and an
    ``extra_body`` key land in the same JSON body, so a switch that
    :func:`request_params` sends at the top level, such as ``reasoning_effort``,
    is folded in here.

    Parameters
    ----------
    model : str
        A bare model name.

    Returns
    -------
    dict
        The thinking-off fields, or ``{}`` for a profile with none.

    Examples
    --------
    >>> thinking_body("gpt-6.1-sol")
    {'reasoning_effort': 'low'}
    >>> thinking_body("claude-opus-5-5")
    {'thinking': {'type': 'adaptive'}, 'reasoning_effort': 'low'}
    """
    switch = profile_for(model).thinking_off or {}
    fields = {**switch.get("extra_body", {})}
    fields.update({key: value for key, value in switch.items() if key != "extra_body"})
    return {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in fields.items()
    }


#: The marker Anthropic reads as the end of a cacheable prefix. The default
#: lifetime is five minutes.
EPHEMERAL_CACHE: Mapping[str, str] = {"type": "ephemeral"}


def cacheable_system(model: str, *parts: str) -> dict[str, Any]:
    """Return a system message whose text the provider's prompt cache can reuse.

    Put everything that repeats across calls in ``parts``, such as the
    instructions and a whole document, and keep what varies per call in the user
    message. For a family whose profile sets ``cache_control`` (Anthropic), the
    parts become a list of text parts with a cache breakpoint on the last one,
    which is the only form the proxy caches. Every other family gets the parts
    joined by a blank line as a plain string, because OpenAI caches a repeated
    prefix without a marker and the on-prem models do not cache.

    A prefix shorter than the provider's minimum is sent and billed as usual and
    is not cached.

    Parameters
    ----------
    model : str
        A bare model name.
    *parts : str
        The system text, in order. Empty parts are dropped.

    Returns
    -------
    dict
        One chat message with ``role`` ``system``.

    Examples
    --------
    >>> cacheable_system("gpt-6-luna", "Rules.", "Document.")["content"].split()
    ['Rules.', 'Document.']
    >>> cacheable_system("claude-haiku-5-5", "Rules.")["content"]
    [{'type': 'text', 'text': 'Rules.', 'cache_control': {'type': 'ephemeral'}}]
    """
    texts = [part for part in parts if part]
    if not profile_for(model).cache_control or not texts:
        return {"role": "system", "content": "\n\n".join(texts)}
    content: list[dict[str, Any]] = [{"type": "text", "text": text} for text in texts]
    content[-1]["cache_control"] = dict(EPHEMERAL_CACHE)
    return {"role": "system", "content": content}


def strip_additional_properties(schema: Any) -> Any:
    """Return a JSON schema with every ``additionalProperties`` removed.

    The proxy rejects ``additionalProperties`` in Gemini response schemas.

    Parameters
    ----------
    schema : Any
        A JSON schema, or any JSON value inside one.

    Returns
    -------
    Any
        A copy without the field at any depth.
    """
    if isinstance(schema, Mapping):
        return {
            key: strip_additional_properties(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [strip_additional_properties(value) for value in schema]
    return schema


def adapt_response_format(model: str, response_format: Any) -> Any:
    """Return a ``response_format`` the model's family accepts.

    Parameters
    ----------
    model : str
        A bare model name.
    response_format : Any
        The ``response_format`` argument as the caller wrote it.

    Returns
    -------
    Any
        The same value, with ``additionalProperties`` stripped for Google models.
    """
    if family_of(model) == "google":
        return strip_additional_properties(response_format)
    return response_format


#: Request fields only a vLLM server reads. OpenAI models reject them with
#: "Unknown parameter"; the other paid families ignore them.
_VLLM_ONLY_FIELDS = ("chat_template_kwargs",)


def adapt_request(model: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a library's chat-completion arguments in the shape the model accepts.

    :func:`request_params` covers code that builds its own requests. A library
    such as atlas-rag builds the request itself and sends ``max_tokens``, a
    ``temperature``, and a ``response_format`` whatever the model, so the
    clients this package hands to libraries pass every call through here. The
    on-prem models and unrecognised names get the arguments back unchanged.

    The changes, each measured through the proxy on 2026-10-26:

    * ``max_tokens`` is renamed to the family's completion cap, because
      ``gpt-6-luna`` rejects ``max_tokens``.
    * ``temperature`` is dropped where :func:`request_params` would leave it
      out. Claude 5.5 models reject any value but the default with
      "`temperature` is deprecated for this model".
    * ``response_format={"type": "json_object"}`` is dropped for Anthropic
      models, which the proxy rejects because Anthropic needs an explicit
      schema. A schema with no properties is accepted but makes the model
      return ``{}``, so the request goes without one and relies on the prompt
      asking for JSON, which every library caller here does.
    * A Google ``response_format`` loses ``additionalProperties``, as in
      :func:`adapt_response_format`.
    * ``chat_template_kwargs`` is removed from ``extra_body`` for paid models.

    Parameters
    ----------
    model : str
        A bare model name.
    params : Mapping
        The keyword arguments of one ``chat.completions.create`` call, without
        ``model`` and ``messages``.

    Returns
    -------
    dict
        A new dictionary. The input is not changed.

    Examples
    --------
    >>> adapt_request("claude-haiku-5-5", {"max_tokens": 64, "temperature": 0.0})
    {'max_tokens': 64}
    >>> adapt_request("gpt-6-luna", {"max_tokens": 64, "temperature": 0.0})
    {'max_completion_tokens': 64}
    >>> adapt_request("Qwen3_6-35B-A3B", {"max_tokens": 64, "temperature": 0.0})
    {'max_tokens': 64, 'temperature': 0.0}
    """
    adapted = dict(params)
    family = family_of(model)
    if family in (None, "onprem"):
        return adapted
    profile = profile_for(model)
    if (
        "max_tokens" in adapted
        and profile.token_param != "max_tokens"
        and profile.token_param not in adapted
    ):
        adapted[profile.token_param] = adapted.pop("max_tokens")
    if "temperature" in adapted and not _sends_temperature(model, profile):
        del adapted["temperature"]
    response_format = adapted.get("response_format")
    if isinstance(response_format, Mapping):
        if family == "anthropic" and response_format.get("type") == "json_object":
            del adapted["response_format"]
        else:
            adapted["response_format"] = adapt_response_format(model, response_format)
    extra_body = adapted.get("extra_body")
    if isinstance(extra_body, Mapping) and any(
        name in extra_body for name in _VLLM_ONLY_FIELDS
    ):
        adapted["extra_body"] = {
            key: value
            for key, value in extra_body.items()
            if key not in _VLLM_ONLY_FIELDS
        }
    return adapted


@dataclass(frozen=True)
class ModelChoice:
    """One role's model, its family, and where the choice came from.

    Parameters
    ----------
    role : str
        ``generation``, ``judging``, or ``extraction``.
    model : str
        The bare model name to send.
    family : str or None
        The family the name belongs to, or None for an unrecognised name.
    source : str
        ``argument``, ``environment``, or ``default``.
    """

    role: Role
    model: str
    family: Family | None
    source: str = "default"

    @property
    def endpoint(self) -> Endpoint:
        """Return the endpoint class this model runs on."""
        return endpoint_of(self.model)

    def request_params(
        self, *, max_tokens: int, temperature: float = 0.0, thinking: bool = False
    ) -> dict[str, Any]:
        """Return the chat-completion keyword arguments for this model.

        See :func:`request_params`.
        """
        return request_params(
            self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
        )

    def config(self) -> dict[str, Any]:
        """Return the fields a model-produced cache key must include."""
        return {"model": self.model, "endpoint": self.endpoint}


@dataclass(frozen=True)
class ModelSet:
    """The models a notebook uses, one per role.

    Parameters
    ----------
    family : str
        The family the defaults came from.
    generation : ModelChoice
        Answers questions.
    judging : ModelChoice
        Grades answers.
    extraction : ModelChoice
        Builds graphs, indexes, and contextual chunks.
    """

    family: Family
    generation: ModelChoice
    judging: ModelChoice
    extraction: ModelChoice
    embeddings: str = EMBEDDING_MODEL

    def __iter__(self) -> Iterator[ModelChoice]:
        """Iterate over the three roles in a fixed order."""
        return iter((self.generation, self.judging, self.extraction))

    def for_role(self, role: Role) -> ModelChoice:
        """Return the choice for one role."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        return cast(ModelChoice, getattr(self, role))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the set."""
        return {
            "family": self.family,
            "embeddings": self.embeddings,
            **{choice.role: choice.model for choice in self},
        }


def _known_models() -> set[str]:
    known = {model for roles in DEFAULT_MODELS.values() for model in roles.values()}
    return known | set(FRONTIER_MODELS.values()) | set(ONPREM_MODELS)


def _resolve_family(family: str | None) -> Family:
    value = family if family is not None else resolve_setting(FAMILY_ENV)
    value = (value or DEFAULT_FAMILY).lower()
    if value not in FAMILIES:
        raise ValueError(
            f"unknown model family {value!r}; expected one of {', '.join(FAMILIES)}"
        )
    return value


def choose_models(
    family: str | None = None,
    *,
    generation: str | None = None,
    judging: str | None = None,
    extraction: str | None = None,
    show: bool = False,
) -> ModelSet:
    """Return the model set for a provider family, with per-role overrides.

    Each role resolves in this order: the argument, then the role's variable
    (:data:`ROLE_ENV`), then the family's default. The family resolves from the
    argument, then ``ADVANCED_RAG_MODEL_FAMILY``, then :data:`DEFAULT_FAMILY`.
    Each variable is read from the environment first and the repository
    ``.env`` second, as :func:`aieng.advanced_rag.proxy.resolve_setting` does.

    A model outside every default set is accepted with an
    :class:`UnlistedModelWarning`, so you can try any model the proxy grants
    your key.

    Parameters
    ----------
    family : str or None, optional
        ``anthropic``, ``google``, ``openai``, or ``onprem``.
    generation, judging, extraction : str or None, optional
        Override one role's model.
    show : bool, optional
        Print the set with each model's price and the key's budget, read from
        the proxy when it answers, and open the spend session that
        :func:`aieng.advanced_rag.spend.report` reads. See
        :func:`aieng.advanced_rag.spend.describe`.

    Returns
    -------
    ModelSet
        One :class:`ModelChoice` per role.

    Raises
    ------
    ValueError
        When the family is not one of :data:`FAMILIES`.

    Examples
    --------
    >>> choose_models("onprem").generation.model
    'Qwen3_6-35B-A3B'
    >>> choose_models("anthropic", judging="claude-opus-5-5").judging.model
    'claude-opus-5-5'
    """
    chosen = _resolve_family(family)
    overrides: dict[Role, str | None] = {
        "generation": generation,
        "judging": judging,
        "extraction": extraction,
    }
    known = _known_models()
    choices: dict[Role, ModelChoice] = {}
    for role in ROLES:
        argument = overrides[role]
        variable = resolve_setting(ROLE_ENV[role])
        if argument:
            model, source = argument.strip(), "argument"
        elif variable:
            model, source = variable, "environment"
        else:
            model, source = DEFAULT_MODELS[chosen][role], "default"
        if model not in known:
            warnings.warn(
                f"{model} is not in any default model set; it works if the proxy "
                "grants it to your key, and its price comes from the proxy",
                UnlistedModelWarning,
                stacklevel=2,
            )
        choices[role] = ModelChoice(role, model, family_of(model), source)
    models = ModelSet(
        family=chosen,
        generation=choices["generation"],
        judging=choices["judging"],
        extraction=choices["extraction"],
    )
    if show:
        from aieng.advanced_rag import spend  # noqa: PLC0415

        print(spend.describe(models))
        # The session snapshots the key's usage now, so the notebook's last
        # cell can report what the proxy billed for this run.
        spend.session()
    return models


#: The models the reference artifacts were built with, on Vector's cluster.
REFERENCE_MODELS: ModelSet = ModelSet(
    family="onprem",
    generation=ModelChoice("generation", "Qwen3_6-35B-A3B", "onprem"),
    judging=ModelChoice("judging", "Qwen3_6-35B-A3B", "onprem"),
    extraction=ModelChoice("extraction", "Qwen3_6-35B-A3B", "onprem"),
)


__all__ = [
    "DEFAULT_FAMILY",
    "DEFAULT_MODELS",
    "EPHEMERAL_CACHE",
    "MODEL_PROFILES",
    "SHARED_JUDGE",
    "EMBEDDING_MODEL",
    "FAMILIES",
    "FAMILY_ENV",
    "FRONTIER_MODELS",
    "ONPREM_MODELS",
    "REFERENCE_MODELS",
    "REQUEST_PROFILES",
    "ROLES",
    "ROLE_ENV",
    "ModelChoice",
    "ModelSet",
    "RequestProfile",
    "UnlistedModelWarning",
    "adapt_request",
    "adapt_response_format",
    "cacheable_system",
    "choose_models",
    "endpoint_of",
    "family_of",
    "profile_for",
    "request_params",
    "strip_additional_properties",
    "thinking_body",
]
