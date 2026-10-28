"""One call per model step, so a live run and a reused artifact read as one path.

The reference artifacts (graphs, indexes, sweeps, judged answers) were built on
Vector's on-prem Qwen models at no charge. You choose a paid model with
:func:`aieng.advanced_rag.models.choose_models`. :func:`load_or_run` decides,
for one model-produced step, which of the two you get, and prints a label that
says so every time:

1. Your own earlier result for your model, when the cache holds one.
2. With ``live`` off and no result of your own, the reference result, labelled
   with the model that built it and what rebuilding it on your model would
   cost.
3. With ``live`` on and no result of your own, the estimate first, then a run on
   your model, stored in your local cache tier.

The cache key names the model and its endpoint class through
:meth:`Step.key`, so a result built on one model is never returned under
another's key. The stored :class:`Provenance` also keeps the model name the
responses reported. A proxy can resolve an alias to a dated snapshot and later
move it, so when a live call in this process has reported a different snapshot
for the same requested name, your cached result is treated as stale and rebuilt
(or, with ``live`` off, reported as stale and returned with that warning).

A live run also records :func:`code_hash` of the producer's source in the
provenance. The hash is not part of the cache key, so a published reference
stays reachable after an edit. When a stored result's hash differs from the
code on disk, your own result is rebuilt with ``live`` on, as a changed
snapshot is. With ``live`` off, or for the reference, the result is returned
and the label says the code changed and how to rebuild. A result stored
without a hash replays with no such line.

Downstream steps put the returned :attr:`StepResult.digest` into their own
``upstream``, so answers over the reference graph and answers over a rebuilt
graph never share a key.

Before a live run, :func:`confirmation_needed` checks the estimate against the
key's budget. A step that would spend more than :data:`CONFIRM_SHARE` of the
budget left stops until the call passes ``confirm=True``. A key that reports no
budget is checked against the assumed budget from
:meth:`aieng.advanced_rag.spend.Budget.assuming` ($300 unless
``ADVANCED_RAG_KEY_BUDGET_USD`` says otherwise), and while the budget is
assumed, any step estimated at over :data:`ASSUMED_CONFIRM_USD` also stops for
``confirm=True``. Neither check runs for a cached or reference result, or with
``live`` off, so a replay never stops.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import inspect
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from aieng.advanced_rag import metering, spend
from aieng.advanced_rag.cache import (
    CacheKey,
    JsonValue,
    TieredResultCache,
    fetch_directory,
    store_directory,
)
from aieng.advanced_rag.metering import UsageMeter, using
from aieng.advanced_rag.models import (
    REFERENCE_MODELS,
    ModelChoice,
    ModelSet,
    Role,
    endpoint_of,
    family_of,
)


Origin = Literal["yours", "reference", "live"]

#: A live step whose estimate exceeds this share of the budget left stops
#: unless the call passes ``confirm=True``.
CONFIRM_SHARE = 0.10

#: While the key's budget is assumed, a live step whose estimate exceeds this
#: many dollars stops unless the call passes ``confirm=True``.
ASSUMED_CONFIRM_USD = 1.0


# Set while ``load_or_run`` runs a step it has already checked against the
# budget, so a batch inside the step does not stop for the same estimate twice.
_STEP_CHECKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aieng_step_checked", default=False
)


def step_checked() -> bool:
    """Return whether the current context runs a step ``load_or_run`` checked.

    :class:`aieng.advanced_rag.batch.ProxyBatch` makes its own estimate before
    a live batch and stops for ``confirm=True`` as :func:`load_or_run` does.
    Inside a step that :func:`load_or_run` has already estimated and let run,
    that second stop would ask again for a decision already made, so the batch
    skips it.

    Returns
    -------
    bool
        True inside the ``compute`` call of a live :func:`load_or_run` step.
    """
    return _STEP_CHECKED.get()


@contextlib.contextmanager
def _checked_step() -> Iterator[None]:
    """Mark the block as a step :func:`load_or_run` has checked."""
    token = _STEP_CHECKED.set(True)
    try:
        yield
    finally:
        _STEP_CHECKED.reset(token)


class ReferenceMissingError(LookupError):
    """Neither your result nor the reference artifact is in the cache."""


class ConfirmationRequiredError(RuntimeError):
    """A live step would spend more than 10% of the budget left.

    Also raised while the budget is assumed, for a step estimated at over
    :data:`ASSUMED_CONFIRM_USD`.
    """


@dataclass(frozen=True)
class Provenance:
    """What produced one model-produced artifact.

    Parameters
    ----------
    model : str
        The model requested.
    response_models : tuple of str, optional
        The model names the responses reported. Differs from ``model`` when the
        proxy resolved an alias.
    family : str or None, optional
        The model's family.
    endpoint : str, optional
        ``vector-cluster``, ``vector-proxy-paid``, or ``unknown``.
    producer : str, optional
        What built the artifact, such as a script or a function name.
    usage : dict, optional
        Model name to ``{"calls", "prompt_tokens", "completion_tokens"}`` for
        the build. This is what a rebuild estimate scales.
    note : str, optional
        Anything a reader needs, such as the corpus slice.
    code_hash : str, optional
        :func:`code_hash` of the producer's source at build time. Empty for an
        artifact built before hashes were recorded.
    """

    model: str
    response_models: tuple[str, ...] = ()
    family: str | None = None
    endpoint: str = "unknown"
    producer: str = ""
    usage: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    note: str = ""
    code_hash: str = ""

    @classmethod
    def for_model(cls, model: str, **fields: Any) -> Provenance:
        """Build a record with the family and endpoint filled in from the name."""
        return cls(
            model=model, family=family_of(model), endpoint=endpoint_of(model), **fields
        )

    @property
    def calls(self) -> int:
        """Return the build's live calls over every model."""
        return sum(int(row.get("calls", 0)) for row in self.usage.values())

    @property
    def prompt_tokens(self) -> int:
        """Return the build's prompt tokens over every model."""
        return sum(int(row.get("prompt_tokens", 0)) for row in self.usage.values())

    @property
    def completion_tokens(self) -> int:
        """Return the build's completion tokens over every model."""
        return sum(int(row.get("completion_tokens", 0)) for row in self.usage.values())

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-serialisable record, the ``produced_by`` block."""
        return {
            "model": self.model,
            "response_models": list(self.response_models),
            "family": self.family,
            "endpoint": self.endpoint,
            "producer": self.producer,
            "usage": {name: dict(row) for name, row in self.usage.items()},
            "note": self.note,
            "code_hash": self.code_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Provenance:
        """Read a ``produced_by`` block written by :meth:`to_dict`."""
        return cls(
            model=str(payload["model"]),
            response_models=tuple(payload.get("response_models") or ()),
            family=payload.get("family"),
            endpoint=str(payload.get("endpoint") or "unknown"),
            producer=str(payload.get("producer") or ""),
            usage={
                str(name): {str(k): int(v) for k, v in row.items()}
                for name, row in (payload.get("usage") or {}).items()
            },
            note=str(payload.get("note") or ""),
            code_hash=str(payload.get("code_hash") or ""),
        )


def code_hash(sources: Iterable[object]) -> str | None:
    """Return a short digest of the source files that define ``sources``.

    Each item is a function, class, or module. Its file is read whole, so an
    edit to a prompt constant or a helper in the same file changes the digest.
    The digest ignores file paths and order, so it matches across checkouts,
    and treats CRLF as LF, so a Windows checkout matches too.

    Parameters
    ----------
    sources : iterable
        Functions, classes, or modules. A :func:`functools.partial` counts as
        the function it wraps.

    Returns
    -------
    str or None
        Sixteen hex characters, or None when any source has no readable file,
        such as a builtin or a function defined in a notebook cell.
    """
    digests: set[str] = set()
    for item in sources:
        source = item
        while isinstance(source, functools.partial):
            source = source.func
        try:
            path = inspect.getsourcefile(source)  # type: ignore[arg-type]
            data = Path(path).read_bytes() if path else None
        except (TypeError, OSError):
            data = None
        if data is None:
            return None
        digests.add(hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest())
    if not digests:
        return None
    return hashlib.sha256("".join(sorted(digests)).encode()).hexdigest()[:16]


def _usage_of(meter: UsageMeter) -> dict[str, dict[str, int]]:
    """Return what the build would cost cold: live calls plus cached ones.

    A step answered partly from a response cache still records the full
    build, so its rebuild estimate does not shrink with every warm re-run.
    """
    return {
        name: {
            "calls": row.calls + row.cache_hits,
            "prompt_tokens": row.prompt_tokens + row.cached_prompt_tokens,
            "completion_tokens": row.completion_tokens + row.cached_completion_tokens,
        }
        for name, row in meter.by_model().items()
        if row.calls or row.cache_hits
    }


@dataclass(frozen=True)
class Step:
    """One model-produced step: what it is, which role runs it, and its inputs.

    Parameters
    ----------
    ri_id : str
        The reference implementation, such as ``ri4``.
    artifact_type : str
        What the step produces, such as ``knowledge-graph`` or ``answers``.
    producer_id : str
        The method or script that produces it.
    config : dict
        Every input other than the model that changes the output: prompt
        digest, dataset revision and slice, seed, evidence budget, request
        parameters.
    role : str, optional
        The role whose model runs the step.
    reference_model : str or None, optional
        The model the reference artifact was built with. Defaults to the role's
        model in :data:`aieng.advanced_rag.models.REFERENCE_MODELS`.
    upstream : dict, optional
        Name to digest of every artifact the step reads, such as a graph.
    kind : str, optional
        ``json`` for a JSON value, ``directory`` for a directory such as an index.
    reference : Provenance or None, optional
        Provenance to show for a reference artifact stored without its own,
        such as an index built before this module existed.
    code : tuple, optional
        Functions, classes, or modules whose files :func:`code_hash` covers.
        Defaults to the file that defines ``compute``. Name them when the
        step's logic lives in other files than the one that passes
        ``compute``. Not part of the cache key.
    """

    ri_id: str
    artifact_type: str
    producer_id: str
    config: Mapping[str, JsonValue]
    role: Role = "generation"
    reference_model: str | None = None
    upstream: Mapping[str, str] = field(default_factory=dict)
    kind: Literal["json", "directory"] = "json"
    reference: Provenance | None = None
    code: tuple[object, ...] = ()

    def code_hash(self, compute: Callable[..., Any]) -> str | None:
        """Return :func:`code_hash` of :attr:`code`, or of ``compute``'s file."""
        return code_hash(self.code or (compute,))

    @property
    def reference_choice(self) -> ModelChoice:
        """Return the model the reference artifact was built with."""
        if self.reference_model is not None:
            return ModelChoice(
                self.role, self.reference_model, family_of(self.reference_model)
            )
        return REFERENCE_MODELS.for_role(self.role)

    def key(self, choice: ModelChoice) -> CacheKey:
        """Return the cache key for this step run on one model.

        The key holds the model as sent and its endpoint class, beside the
        step's own configuration and upstream digests.
        """
        config: dict[str, JsonValue] = dict(self.config)
        config.update(choice.config())
        if self.upstream:
            config["upstream"] = dict(self.upstream)
        return CacheKey(self.ri_id, self.artifact_type, self.producer_id, config)

    def provenance_key(self, choice: ModelChoice) -> CacheKey:
        """Return the key a directory artifact's provenance is stored under."""
        key = self.key(choice)
        return CacheKey(
            key.ri_id, f"{key.artifact_type}.provenance", key.producer_id, key.config
        )


@dataclass(frozen=True)
class StepResult:
    """What :func:`load_or_run` returned, and where it came from.

    Parameters
    ----------
    value : Any
        The JSON value, or the directory path for a ``directory`` step.
    origin : str
        ``yours``, ``reference``, or ``live``.
    provenance : Provenance or None
        What produced it, when known.
    digest : str
        The digest of the cache key it is stored under. Downstream steps put
        this in their ``upstream``.
    label : str
        The lines :func:`load_or_run` printed.
    estimate_usd : float or None
        The estimated cost of rebuilding the reference on your model.
    code_changed : bool
        True when the result was replayed and its recorded code hash differs
        from the producer's code on disk.
    """

    value: Any
    origin: Origin
    provenance: Provenance | None
    digest: str
    label: str
    estimate_usd: float | None = None
    code_changed: bool = False


def rebuild_estimate(
    provenance: Provenance | None,
    model: str,
    *,
    table: Mapping[str, spend.ModelPrice] | None = None,
) -> tuple[int, float | None]:
    """Return the calls and dollars a rebuild on ``model`` would take, roughly.

    The reference build's token counts are priced at ``model``'s rate. Two
    reasons make this approximate: token counts differ across tokenizers, and
    completion length is a property of the model, so a model with thinking off
    can write fewer completion tokens than the one that built the reference.

    Returns
    -------
    tuple
        Calls, and dollars or None when ``model`` has no known rate or the
        reference recorded no tokens.
    """
    if provenance is None or not provenance.usage:
        return 0, None
    cost = spend.estimate_tokens(
        model, provenance.prompt_tokens, provenance.completion_tokens, table=table
    )
    return provenance.calls, cost


def _built_with(provenance: Provenance | None, fallback: str) -> str:
    model = provenance.model if provenance is not None else fallback
    where = (
        "on Vector's cluster at $0"
        if endpoint_of(model) == "vector-cluster"
        else "through the proxy"
    )
    return f"built with {model} {where}"


def _tokens(provenance: Provenance) -> str:
    if not provenance.usage:
        return ""
    return (
        f"{provenance.calls:,} calls, {provenance.prompt_tokens:,} prompt + "
        f"{provenance.completion_tokens:,} completion tokens"
    )


def _label(
    step: Step,
    origin: Origin,
    provenance: Provenance | None,
    choice: ModelChoice,
    estimate: tuple[int, float | None] | None,
) -> str:
    name = step.artifact_type
    pad = " " * (len(name) + 1)
    if origin == "reference":
        lines = [
            f"{name} reference artifact, {_built_with(provenance, step.reference_choice.model)}"
        ]
        if provenance is not None and provenance.usage:
            lines.append(pad + _tokens(provenance))
        if choice.model != step.reference_choice.model and estimate is not None:
            calls, dollars = estimate
            if dollars is not None:
                cost = f"est. ${dollars:,.2f}"
            elif provenance is None or not provenance.usage:
                cost = "no estimate: the reference recorded no token counts"
            else:
                cost = "no estimate: no price for this model"
            calls_text = f"about {calls:,} calls, " if calls else ""
            lines.append(
                pad + f"rebuilding with {choice.model}: {calls_text}{cost}"
                " (set RUN_EXPENSIVE = True to rebuild)"
            )
        return "\n".join(lines)
    if origin == "yours":
        lines = [f"{name} your cached result, {_built_with(provenance, choice.model)}"]
    else:
        lines = [f"{name} your run: {choice.model}"]
    if provenance is not None and provenance.usage:
        lines.append(pad + _tokens(provenance))
    if step.upstream:
        lines.append(
            pad + "over " + ", ".join(f"{k} {v[:12]}" for k, v in step.upstream.items())
        )
    return "\n".join(lines)


def _stale(provenance: Provenance | None) -> str | None:
    """Return the snapshot a live call reported, when it differs from the stored one."""
    if provenance is None or not provenance.response_models:
        return None
    seen: str | None = metering.observed_response_model(provenance.model)
    if seen is None or seen in provenance.response_models:
        return None
    return seen


def _code_changed(
    step: Step, provenance: Provenance | None, current: str | None
) -> str | None:
    """Return the label line for a changed producer, or None.

    An artifact with no recorded hash, or a producer whose source is not on
    disk, gives None, so results built before hashes existed replay silently.
    """
    if provenance is None or not provenance.code_hash or current is None:
        return None
    if provenance.code_hash == current:
        return None
    return (
        " "
        * (len(step.artifact_type) + 1)
        + f"code changed: built from {provenance.code_hash}, the producer is now "
        f"{current}, so this result does not reflect the code on disk "
        "(set RUN_EXPENSIVE = True to rebuild it)"
    )


def _flag(result: StepResult, changed: str | None) -> StepResult:
    """Return ``result`` with the changed-code line added, when there is one."""
    if changed is None:
        return result
    return dataclasses.replace(
        result, label=result.label + "\n" + changed, code_changed=True
    )


class _Store:
    """Reads and writes one step's artifacts, JSON or directory."""

    def __init__(
        self, cache: TieredResultCache, step: Step, workdir: Path | None
    ) -> None:
        self.cache = cache
        self.step = step
        self.workdir = workdir

    def load(self, choice: ModelChoice) -> tuple[Any, Provenance | None] | None:
        key = self.step.key(choice)
        if self.step.kind == "json":
            found = self.cache.get(key)
            if found is None:
                return None
            return _unwrap(found.value)
        destination = self._destination(choice)
        path = fetch_directory(self.cache, key, destination, replace=True)
        if path is None:
            return None
        record = self.cache.get(self.step.provenance_key(choice))
        provenance = (
            Provenance.from_dict(cast(Mapping[str, Any], record.value))
            if record is not None and isinstance(record.value, dict)
            else None
        )
        return path, provenance

    def save(self, choice: ModelChoice, value: Any, provenance: Provenance) -> Any:
        key = self.step.key(choice)
        if self.step.kind == "json":
            wrapped: JsonValue = {"result": value, "produced_by": provenance.to_dict()}
            self.cache.load_or_compute(key, lambda: wrapped, force_refresh=True)
            return value
        store_directory(self.cache, key, Path(value), force_refresh=True)
        self.cache.load_or_compute(
            self.step.provenance_key(choice), provenance.to_dict, force_refresh=True
        )
        return Path(value)

    def _destination(self, choice: ModelChoice) -> Path:
        root = self.workdir or Path(tempfile.gettempdir()) / "aieng-advanced-rag-runs"
        digest: str = self.step.key(choice).digest
        return Path(root, self.step.ri_id, self.step.artifact_type, digest[:16])


def _unwrap(value: JsonValue) -> tuple[Any, Provenance | None]:
    if isinstance(value, dict) and "produced_by" in value and "result" in value:
        produced = value["produced_by"]
        provenance = (
            Provenance.from_dict(cast(Mapping[str, Any], produced))
            if isinstance(produced, dict)
            else None
        )
        return value["result"], provenance
    return value, None


def load_or_run(
    step: Step,
    compute: Callable[..., Any],
    *,
    models: ModelSet | None = None,
    live: bool = False,
    confirm: bool = False,
    cache: TieredResultCache | None = None,
    workdir: Path | None = None,
    show: bool = True,
    table: Mapping[str, spend.ModelPrice] | None = None,
    budget: spend.Budget | None = None,
) -> StepResult:
    """Return your result, the labelled reference, or a live run on your model.

    Parameters
    ----------
    step : Step
        The step and everything but the model that changes its output.
    compute : callable
        Runs the step on one model. Called as ``compute(choice)`` for a
        ``json`` step, returning a JSON value, and ``compute(choice, directory)``
        for a ``directory`` step, filling ``directory``. Every model call it
        makes through this package's completers, or through a client metered
        with the default meter, is recorded in the stored provenance.
    models : ModelSet or None, optional
        Your model set. The reference set when omitted.
    live : bool, optional
        Run on your model when no result of yours exists. Pass the notebook's
        ``RUN_EXPENSIVE``.
    confirm : bool, optional
        Run even when the estimate is over 10% of the key's budget left, or
        over $1 while the key reports no budget.
    cache : TieredResultCache or None, optional
        The cache. The default tiers when omitted.
    workdir : Path or None, optional
        Where a ``directory`` step's artifact is extracted.
    show : bool, optional
        Print the label.
    table : mapping or None, optional
        Rates to price with. :func:`aieng.advanced_rag.spend.price_table` when
        omitted.
    budget : Budget or None, optional
        The key's budget, for the confirmation guard. Read from the proxy when
        omitted and the step runs live. A budget of None is replaced with the
        assumed one, as :meth:`aieng.advanced_rag.spend.Budget.assuming` does.

    Returns
    -------
    StepResult
        The value, where it came from, its provenance, and its digest.

    Raises
    ------
    ReferenceMissingError
        With ``live`` off, when neither your result nor the reference is cached.
    ConfirmationRequiredError
        With ``live`` on and ``confirm`` false, when the estimate is over 10%
        of the budget left, or over $1 while the budget is assumed.
    """
    chosen = (models or REFERENCE_MODELS).for_role(step.role)
    reference = step.reference_choice
    store = _Store(cache or TieredResultCache(), step, workdir)

    def done(result: StepResult) -> StepResult:
        if show:
            print(result.label)
        return result

    current = step.code_hash(compute)
    own = store.load(chosen)
    stale = _stale(own[1]) if own is not None else None
    changed = _code_changed(step, own[1], current) if own is not None else None
    if own is not None and not ((stale or changed) and live):
        value, provenance = own
        provenance = provenance or (
            step.reference if chosen.model == reference.model else None
        )
        origin: Origin = "reference" if chosen.model == reference.model else "yours"
        label = _label(step, origin, provenance, chosen, None)
        if stale:
            label += (
                f"\n  stale: {chosen.model} now reports {stale}; run live to rebuild"
            )
        return done(
            _flag(
                StepResult(value, origin, provenance, step.key(chosen).digest, label),
                changed,
            )
        )

    found = store.load(reference) if chosen.model != reference.model else None
    reference_provenance = None
    if found is not None:
        reference_provenance = found[1] or step.reference
    estimate = rebuild_estimate(
        reference_provenance or step.reference, chosen.model, table=table
    )

    if not live:
        if found is None:
            raise ReferenceMissingError(
                f"no {step.artifact_type} for {chosen.model} or for the reference "
                f"model {reference.model} in the cache; fetch the reference "
                "artifacts or set RUN_EXPENSIVE = True to build it"
            )
        value, _ = found
        label = _label(step, "reference", reference_provenance, chosen, estimate)
        return done(
            _flag(
                StepResult(
                    value,
                    "reference",
                    reference_provenance,
                    step.key(reference).digest,
                    label,
                    estimate[1],
                ),
                _code_changed(step, reference_provenance, current),
            )
        )

    _guard(step, chosen, estimate, confirm, budget)
    meter = UsageMeter()
    scratch: Path | None = None
    try:
        # Binding a question is what makes a bare ChatCompleter's reports land on
        # this meter; without one they are counted as unmetered and dropped.
        with (
            using(meter),
            meter.question(f"{step.ri_id}/{step.producer_id}"),
            _checked_step(),
        ):
            if step.kind == "json":
                value = compute(chosen)
            else:
                scratch = Path(tempfile.mkdtemp(prefix="aieng-step-"))
                compute(chosen, scratch)
                value = scratch
        responses = sorted(
            {name for row in meter.by_model().values() for name in row.response_models}
        )
        provenance = Provenance.for_model(
            chosen.model,
            response_models=tuple(responses),
            producer=f"{step.ri_id}/{step.producer_id}",
            usage=_usage_of(meter),
            code_hash=current or "",
        )
        stored = store.save(chosen, value, provenance)
        if step.kind == "directory":
            stored = store.load(chosen)[0]  # type: ignore[index]
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    label = _label(step, "live", provenance, chosen, None)
    label += (
        "\n"
        + " " * (len(step.artifact_type) + 1)
        + spend.usage_line(meter, table=table)
    )
    return done(StepResult(stored, "live", provenance, step.key(chosen).digest, label))


def _guard(
    step: Step,
    chosen: ModelChoice,
    estimate: tuple[int, float | None],
    confirm: bool,
    budget: spend.Budget | None,
) -> None:
    calls, dollars = estimate
    if dollars is None:
        reason = (
            "no reference build to scale" if not calls else "no price for this model"
        )
        print(f"{step.artifact_type} running on {chosen.model}: no estimate ({reason})")
        return
    print(
        f"{step.artifact_type} running on {chosen.model}: about {calls:,} calls, "
        f"est. ${dollars:,.2f}"
    )
    if dollars == 0:
        return
    known = (budget if budget is not None else spend.read_budget()).assuming()
    if known.assumed:
        print(" " * (len(step.artifact_type) + 1) + known.line())
    if confirm:
        return
    stop = confirmation_needed(dollars, known)
    if stop is not None:
        raise ConfirmationRequiredError(stop)


def confirmation_needed(dollars: float, budget: spend.Budget) -> str | None:
    """Return why a live step estimated at ``dollars`` needs ``confirm=True``.

    Parameters
    ----------
    dollars : float
        The step's estimate.
    budget : Budget
        The key's budget. A budget of None is replaced with the assumed one
        from :meth:`aieng.advanced_rag.spend.Budget.assuming`.

    Returns
    -------
    str or None
        The message for :class:`ConfirmationRequiredError`, or None when the
        step can run: the estimate is at most :data:`CONFIRM_SHARE` of the
        budget left and, for an assumed budget, at most
        :data:`ASSUMED_CONFIRM_USD`.
    """
    known = budget.assuming()
    left = known.left_usd
    assumed = (
        f" (the key reports no budget, so ${known.budget_usd:,.2f} is assumed; "
        f"set {spend.KEY_BUDGET_ENV} to override)"
        if known.assumed
        else ""
    )
    if left is not None and dollars > CONFIRM_SHARE * left:
        return (
            f"the estimate ${dollars:,.2f} is over {CONFIRM_SHARE:.0%} of the "
            f"${left:,.2f} left on the key{assumed}; pass confirm=True to run it"
        )
    if known.assumed and dollars > ASSUMED_CONFIRM_USD:
        return (
            f"the estimate ${dollars:,.2f} is over ${ASSUMED_CONFIRM_USD:,.2f}"
            f"{assumed}; pass confirm=True to run it"
        )
    return None


__all__ = [
    "ASSUMED_CONFIRM_USD",
    "CONFIRM_SHARE",
    "ConfirmationRequiredError",
    "Provenance",
    "ReferenceMissingError",
    "Step",
    "StepResult",
    "code_hash",
    "confirmation_needed",
    "load_or_run",
    "rebuild_estimate",
    "step_checked",
]
