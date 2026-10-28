"""Measurement pieces for the GraphRAG null-answer arm.

Everything here is testable without a network, a model, or a GraphRAG library.
The wiring that needs those lives in ``run_null_arm.py``.

Three things this module owns.

**Cost is counted, not estimated.** :class:`Meter` accumulates calls and tokens
at the client, so a figure covers whatever a third-party library chose to do
inside its own code path. That accounting disqualified Microsoft GraphRAG
on cost, and it only works if nothing bypasses the meter.

**Cached calls are not counted.** :class:`ResponseCache` makes a re-run free
while keeping the reported number the cost of a cold run. ContractNLI asks the
same 17 hypotheses of every contract, so a query-only call such as LightRAG's
keyword extraction repeats verbatim across documents; the cache records those
as hits rather than pretending they were never made.

**Evidence reaching the reader is scored by text, not by identifier.** A
GraphRAG library chunks a document its own way, so its chunk identifiers cannot
be compared with the harness judgment set. :func:`evidence_covers_spans` asks
the question the identifiers were a proxy for: did any gold span actually reach
the generator?
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.batch import KeyFunction, ProxyBatch
from aieng.advanced_rag.evaluation.metrics import NullReport, score_null_detection
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.graphrag.proxy import ResponseCache
from aieng.advanced_rag.models import REFERENCE_MODELS, request_params
from aieng.advanced_rag.pipeline.spine import Retrieve
from aieng.advanced_rag.provenance import Provenance
from aieng.advanced_rag.proxy import PROXY_BASE_URL, resolve_api_key


DEFAULT_BASE_URL = PROXY_BASE_URL
DEFAULT_MODEL = "Qwen3_6-35B-A3B"

_WHITESPACE = re.compile(r"\s+")


def load_api_key(env_path: Path) -> str:
    """Read the proxy key from the environment, then from a ``.env`` file.

    An exported ``VECTOR_PROXY_API_KEY`` wins, and so does an exported
    ``OPENAI_API_KEY`` when ``OPENAI_BASE_URL`` points at the proxy, which is
    what a Coder workspace provides, because ``onboard`` exports both and
    writes no file. An ``OPENAI_API_KEY`` exported without the proxy endpoint
    may be a credential for OpenAI's API and is never sent. The file is the
    fallback, and its ``NAME = value`` lines, with spaces around the ``=``,
    parse there. The lookup is :func:`aieng.advanced_rag.proxy.resolve_api_key`.

    Parameters
    ----------
    env_path : Path
        The ``.env`` file to read when the environment holds no key. It need
        not exist.

    Returns
    -------
    str
        The key.

    Raises
    ------
    RuntimeError
        If neither the environment nor the file holds a recognised key.
    """
    key: str = resolve_api_key(env_path)
    return key


@dataclass
class Meter:
    """Running call and token totals, split by the phase that spent them.

    Parameters
    ----------
    phase : str, optional
        Label applied to calls recorded from now on.
    """

    phase: str = "unassigned"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    empty_responses: int = 0
    seconds: float = 0.0
    by_phase: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Give the meter its own lock, so threads can share one instance."""
        self._lock = threading.Lock()

    def record(self, prompt: int, completion: int, seconds: float) -> None:
        """Add one uncached call to the totals."""
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.seconds += seconds
            bucket = self._bucket(self.phase)
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["seconds"] += seconds

    def record_cache_hit(self) -> None:
        """Add one call that a cache answered, so cold cost stays visible."""
        with self._lock:
            self.cache_hits += 1
            self._bucket(self.phase)["cache_hits"] += 1

    def record_empty(self) -> None:
        """Add one response that carried no content."""
        with self._lock:
            self.empty_responses += 1
            self._bucket(self.phase)["empty_responses"] += 1

    def _bucket(self, phase: str) -> dict[str, float]:
        """Return the accumulator for one phase, creating it if needed."""
        return self.by_phase.setdefault(
            phase,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_hits": 0,
                "empty_responses": 0,
                "seconds": 0.0,
            },
        )

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def phase_totals(self, phase: str) -> dict[str, int | float]:
        """Return one phase's totals, zeroed when the phase spent nothing.

        ``call_seconds`` is the sum of per-call durations across every thread,
        not elapsed wall time. Concurrent arms make the two differ by the
        worker count, and reading one as the other overstates a run by an order
        of magnitude.
        """
        bucket = self.by_phase.get(phase)
        if bucket is None:
            return {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hits": 0,
                "empty_responses": 0,
                "call_seconds": 0.0,
            }
        return {
            "calls": int(bucket["calls"]),
            "prompt_tokens": int(bucket["prompt_tokens"]),
            "completion_tokens": int(bucket["completion_tokens"]),
            "total_tokens": int(bucket["prompt_tokens"] + bucket["completion_tokens"]),
            "cache_hits": int(bucket["cache_hits"]),
            "empty_responses": int(bucket["empty_responses"]),
            "call_seconds": round(bucket["seconds"], 1),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of every phase."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "empty_responses": self.empty_responses,
            "call_seconds": round(self.seconds, 1),
            "by_phase": {phase: self.phase_totals(phase) for phase in sorted(self.by_phase)},
        }


def completion_key(model: str) -> KeyFunction:
    """Return the cache key function this run's completion caches were written with.

    The key names the prompt, the completion cap, and the temperature, and,
    for any model but the reference generator, the request parameters the
    model's family is sent. The committed replay caches hold these keys, so
    the shape does not change.

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
            "thinking": False,
        }
        if model != REFERENCE_MODELS.generation.model:
            request["request_params"] = request_params(
                model, max_tokens=params["max_tokens"], temperature=params["temperature"], thinking=False
            )
        digest: str = ResponseCache.key(request)
        return digest

    return key


class ReplayBatch(ProxyBatch):  # type: ignore[misc]
    """A proxy batch that answers only from its cache and never builds a client."""

    def client(self) -> Any:
        """Refuse, because a replay must not reach a model."""
        raise FileNotFoundError("cache-only replay is missing a model response")


def proxy_batch(
    model: str,
    cache: ResponseCache | None,
    *,
    cache_only: bool = False,
    **connection: Any,
) -> ProxyBatch:
    """Return the proxy connection for ``model``, caching under :func:`completion_key`.

    Parameters
    ----------
    model : str
        Model name to send.
    cache : ResponseCache or None
        Response cache, storing the reply text.
    cache_only : bool, optional
        Answer only from the cache: a miss raises ``FileNotFoundError`` and no
        key or endpoint is read.
    **connection
        Other :class:`ProxyBatch` arguments, such as ``rpm``, ``timeout``,
        ``api_key``, and ``base_url``.

    Returns
    -------
    ProxyBatch
        The batch. Share one among every caller on ``model``.

    Raises
    ------
    ValueError
        If ``cache_only`` is set without a cache.
    """
    if cache_only and cache is None:
        raise ValueError("cache-only completion requires a response cache")
    kind = ReplayBatch if cache_only else ProxyBatch
    return kind(model, cache=cache, key=completion_key(model), payload="text", **connection)


class MeteredCompleter:
    """A prompt-to-text callable that meters every call and cache hit.

    This is the ``complete`` argument every generating stage of the pipeline
    takes, so grading, generation, and answer judging all flow through one
    meter. Calls go through a :class:`aieng.advanced_rag.batch.ProxyBatch`
    from :func:`proxy_batch`, which answers from the cache first, switches
    reasoning off, holds calls to its rate limit, and retries only transient
    errors. The bulk model on the Vector proxy bills reasoning as completion
    tokens and returns no content when ``max_tokens`` runs out mid-reasoning.

    Thread-safe, because the arms run their queries through a thread pool.

    Parameters
    ----------
    batch : ProxyBatch
        The proxy connection, from :func:`proxy_batch`.
    meter : Meter or None, optional
        Where to accumulate cost. One is created when omitted.
    max_tokens : int, optional
        Completion cap.
    temperature : float, optional
        Sampling temperature. Zero removes sampling variance only; model serving
        can still be nondeterministic, and some providers restrict this setting.

    Raises
    ------
    RuntimeError
        From a call whose response has no content. A failed call must not
        become an empty response: every parser in the pipeline reads an empty
        response as an abstention, so a rate-limited run would quietly report
        an arm that abstains rather than a run that did not finish.
    """

    def __init__(
        self,
        batch: ProxyBatch,
        meter: Meter | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.batch = batch
        self.model = batch.model
        self.meter = meter or Meter()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        """Return a nonempty completion; failed calls never become abstentions."""
        outcome = self.batch.create(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        text = outcome.text or ""
        if outcome.cached:
            if not text.strip():
                raise ValueError("cached completion is empty; the run is incomplete")
            self.meter.record_cache_hit()
            return text
        seconds = outcome.metrics.service_time_s if outcome.metrics is not None else 0.0
        self.meter.record(outcome.usage.prompt_tokens, outcome.usage.completion_tokens, seconds)
        if not text.strip():
            self.meter.record_empty()
            raise RuntimeError("proxy returned an empty completion")
        return text


def normalise(text: str) -> str:
    """Return text with whitespace collapsed and case folded, for matching."""
    return _WHITESPACE.sub(" ", text or "").strip().casefold()


def evidence_covers_spans(evidence: Sequence[Evidence], spans: Iterable[str], min_chars: int = 30) -> bool:
    """Whether any gold span reached the generator inside a retrieved passage.

    The identifier-based judgment set cannot cross a method boundary: LightRAG
    chunks a document its own way, so its chunk identifiers are not the
    harness's. Containment of the span text answers the question the
    identifiers were standing in for, and it applies to every arm alike.

    A span shorter than ``min_chars`` is skipped. Short spans match by accident
    in contract language, where phrases such as "the Receiving Party" recur on
    every page.

    Parameters
    ----------
    evidence : sequence of Evidence
        The passages the generator was shown.
    spans : iterable of str
        Gold span texts for this question.
    min_chars : int, optional
        Shortest span trusted to match.

    Returns
    -------
    bool
        True when at least one usable span is contained in a passage.
    """
    haystack = " || ".join(normalise(item.text) for item in evidence)
    if not haystack:
        return False
    for span in spans:
        needle = normalise(span)
        if len(needle) >= min_chars and needle in haystack:
            return True
    return False


@dataclass(frozen=True)
class QueryOutcome:
    """What one arm did with one question.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    doc_id : str
        Contract the question is scoped to.
    is_null : bool
        Whether the dataset marks the question unanswerable.
    abstained : bool
        Whether the pipeline declined to answer.
    reason : str
        Why it abstained, empty on an answered question.
    verdict : str
        The grader's reading of the final attempt.
    score : float
        The final attempt's pooled grade.
    retrieved : int
        Passages the retriever returned.
    evidence_chars : int
        Characters of passage text placed in the generation prompt.
    span_reached : bool
        Whether a gold span reached the generator. Always False on a null
        question, which has no gold span.
    answer : str
        What a reader is shown.
    """

    query_id: str
    doc_id: str
    is_null: bool
    abstained: bool
    reason: str
    verdict: str
    score: float
    retrieved: int
    evidence_chars: int
    span_reached: bool
    answer: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the outcome."""
        return {
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "is_null": self.is_null,
            "abstained": self.abstained,
            "reason": self.reason,
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "retrieved": self.retrieved,
            "evidence_chars": self.evidence_chars,
            "span_reached": self.span_reached,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class AnswerableCost:
    """What an arm gives up on the questions that do have an answer.

    An arm can win :func:`~aieng.advanced_rag.evaluation.metrics.score_null_detection`
    by abstaining on everything, so a null report is unreadable without this
    beside it.

    Parameters
    ----------
    answerable : int
        Questions the dataset marks answerable.
    abstained : int
        Answerable questions the arm declined.
    span_reached : int
        Answerable questions where a gold span reached the generator.
    answered_with_span : int
        Answerable questions the arm answered and where a gold span had
        reached the generator. The best available proxy for a correct answer
        without a judge.
    """

    answerable: int
    abstained: int
    span_reached: int
    answered_with_span: int

    @property
    def abstention_rate(self) -> float:
        """Return the share of answerable questions the arm declined."""
        return round(self.abstained / self.answerable, 4) if self.answerable else 0.0

    @property
    def span_reach_rate(self) -> float:
        """Return the share of answerable questions whose span was retrieved."""
        return round(self.span_reached / self.answerable, 4) if self.answerable else 0.0

    @property
    def grounded_answer_rate(self) -> float:
        """Return the share answered with a gold span in front of the model."""
        if not self.answerable:
            return 0.0
        return round(self.answered_with_span / self.answerable, 4)

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary."""
        return {
            "answerable_queries": self.answerable,
            "abstained_on_answerable": self.abstained,
            "answerable_abstention_rate": self.abstention_rate,
            "span_reached": self.span_reached,
            "span_reach_rate": self.span_reach_rate,
            "answered_with_span": self.answered_with_span,
            "grounded_answer_rate": self.grounded_answer_rate,
        }


def answerable_cost(outcomes: Sequence[QueryOutcome]) -> AnswerableCost:
    """Price an arm on the questions the dataset says have an answer.

    Parameters
    ----------
    outcomes : sequence of QueryOutcome
        Every question the arm ran, null and answerable together.

    Returns
    -------
    AnswerableCost
        The aggregate over the answerable ones.
    """
    answerable = [outcome for outcome in outcomes if not outcome.is_null]
    return AnswerableCost(
        answerable=len(answerable),
        abstained=sum(1 for outcome in answerable if outcome.abstained),
        span_reached=sum(1 for outcome in answerable if outcome.span_reached),
        answered_with_span=sum(1 for outcome in answerable if outcome.span_reached and not outcome.abstained),
    )


def null_report(outcomes: Sequence[QueryOutcome]) -> NullReport:
    """Score an arm's abstentions against the dataset's null labels.

    Parameters
    ----------
    outcomes : sequence of QueryOutcome
        Every question the arm ran.

    Returns
    -------
    NullReport
        The aggregate ``score_null_detection`` returns.
    """
    return score_null_detection(
        {outcome.query_id: outcome.abstained for outcome in outcomes},
        [outcome.query_id for outcome in outcomes if outcome.is_null],
    )


def constant_outcomes(outcomes: Sequence[QueryOutcome], abstained: bool) -> list[QueryOutcome]:
    """Return the same questions decided by a fixed rule, for the controls.

    Always-answer and always-abstain are the floor every other arm is read
    against. At this corpus's null rate, always abstaining scores an abstention
    recall of 1.0 and a precision equal to the null rate, which is the number
    that makes the rest of the table legible.

    Parameters
    ----------
    outcomes : sequence of QueryOutcome
        Any arm's outcomes, used for the question set and its labels.
    abstained : bool
        The fixed decision.

    Returns
    -------
    list of QueryOutcome
        One outcome per question, with retrieval and grading zeroed.
    """
    label = "control: always abstain" if abstained else "control: always answer"
    return [
        QueryOutcome(
            query_id=outcome.query_id,
            doc_id=outcome.doc_id,
            is_null=outcome.is_null,
            abstained=abstained,
            reason=label if abstained else "",
            verdict="not graded",
            score=0.0,
            retrieved=0,
            evidence_chars=0,
            span_reached=False,
            answer="",
        )
        for outcome in outcomes
    ]


def evidence_to_dict(item: Evidence) -> dict[str, Any]:
    """Return a JSON-serialisable record of one retrieved passage."""
    return {
        "text": item.text,
        "doc_id": item.provenance.doc_id,
        "chunk_id": item.provenance.chunk_id,
        "start": item.provenance.start,
        "end": item.provenance.end,
        "score": item.score,
        "retriever": item.retriever,
    }


def evidence_from_dict(payload: Mapping[str, Any]) -> Evidence:
    """Rebuild one retrieved passage from :func:`evidence_to_dict` output."""
    return Evidence(
        text=str(payload["text"]),
        provenance=Provenance(
            doc_id=str(payload["doc_id"]),
            chunk_id=payload.get("chunk_id"),
            start=payload.get("start"),
            end=payload.get("end"),
        ),
        score=payload.get("score"),
        retriever=str(payload.get("retriever", "")),
    )


def save_evidence(path: Path, table: Mapping[str, Sequence[Evidence]]) -> None:
    """Write a retrieval table to disk, so an arm re-runs without re-retrieving."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {query_id: [evidence_to_dict(item) for item in items] for query_id, items in table.items()}
    # A table cut off mid-write would pass an existence check, so it is
    # renamed into place only once complete.
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_evidence(path: Path) -> dict[str, list[Evidence]]:
    """Read a retrieval table written by :func:`save_evidence`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {query_id: [evidence_from_dict(item) for item in items] for query_id, items in payload.items()}


def table_retrieve(
    table: Mapping[str, Sequence[Evidence]],
    query_ids: Mapping[tuple[str, str], str],
) -> Retrieve:
    """Adapt a precomputed retrieval table to the pipeline's retrieval callable.

    Retrieval runs once per arm and is stored, then replayed here. That keeps
    the expensive half of a GraphRAG arm out of the generation loop, lets every
    arm be re-scored without re-retrieving, and makes the evidence auditable
    after the fact.

    The pipeline hands a retriever ``(question, scope, k)`` rather than an
    identifier, so the table is reached through the ``(scope, question)`` pair.
    ContractNLI asks each hypothesis once per contract, which makes that pair
    unique. A question the table does not hold retrieves nothing, which the
    pipeline reads as an abstention rather than as a silent empty result.

    Parameters
    ----------
    table : mapping
        Query identifier to ranked evidence.
    query_ids : mapping
        ``(scope, question)`` to query identifier.

    Returns
    -------
    callable
        A ``(question, scope, k)`` callable the spine accepts.
    """

    def retrieve(question: str, scope: str | None = None, k: int = 10) -> list[Evidence]:
        query_id = query_ids.get((scope or "", question), "")
        return list(table.get(query_id, ()))[:k]

    return retrieve
