"""Project Langfuse traces onto the inputs of the scoring harness.

The harness scores two mappings: ranked chunk identifiers per query for
:func:`~aieng.advanced_rag.evaluation.metrics.score_retrieval`, and an
abstention flag per query for
:func:`~aieng.advanced_rag.evaluation.metrics.score_null_detection`. A Langfuse
trace holds both in free-form JSON, so reading them back is a convention the
trace has to follow. This module states the convention and reads it.

A trace projects when it carries three things:

* A query identifier, as ``metadata["query_id"]`` or ``input["query_id"]``.
* At least one observation of type ``RETRIEVER``. Each one's output is a list of
  ranked passages, written as identifier strings or as mappings with ``id``,
  ``chunk_id``, or ``provenance.chunk_id``. The last retriever observation by
  start time is the final attempt, which is what
  :func:`~aieng.advanced_rag.pipeline.spine.retrieval_runs` scores.
* A trace output that is a mapping with a boolean ``abstained``, or a string that
  :func:`~aieng.advanced_rag.pipeline.generate.looks_like_abstention` reads.

Judgments come from the dataset, not the trace, so this projection never
produces ``qrels``.

The projection is lossy by design. :data:`DROPPED` lists what it discards.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aieng.advanced_rag.pipeline.generate import looks_like_abstention


#: What a projected trace no longer carries, and where to look for it instead.
DROPPED = (
    "Passage scores: the harness scores rank only, so a score in a retriever "
    "output is read for order and discarded.",
    "Earlier attempts: only the last retriever observation reaches `runs`. "
    "`first_runs` keeps the first, to price a corrective loop.",
    "Grader verdicts, rewrite text, and the answer text itself: the harness reads "
    "an abstention flag, not the reasoning behind it.",
    "Tokens, calls, and latency: read those from the meter or from the "
    "generation observations, which this projection does not open.",
)


@dataclass(frozen=True)
class Projection:
    """Scoring inputs read from a set of traces.

    Parameters
    ----------
    runs : dict
        Query identifier to the final attempt's ranked chunk identifiers, for
        ``score_retrieval``.
    first_runs : dict
        Query identifier to the first attempt's ranked chunk identifiers.
    abstained : dict
        Query identifier to whether the pipeline declined, for
        ``score_null_detection``.
    skipped : dict
        Trace identifier to the reason that trace did not project. A trace
        lands here rather than raising, so one malformed trace does not stop a
        batch, and the count is visible.
    """

    runs: dict[str, list[str]] = field(default_factory=dict)
    first_runs: dict[str, list[str]] = field(default_factory=dict)
    abstained: dict[str, bool] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def project_traces(traces: Iterable[Mapping[str, Any]]) -> Projection:
    """Project Langfuse traces onto ``score_retrieval`` and ``score_null_detection``.

    Parameters
    ----------
    traces : iterable of mapping
        Traces in the Langfuse API shape: ``id``, ``input``, ``output``,
        ``metadata``, and an ``observations`` list whose items carry ``type``,
        ``output``, and ``startTime``. Anything with a ``dict()`` or
        ``model_dump()`` method must be converted first.

    Returns
    -------
    Projection
        The mappings the two scorers take, and the traces that did not project.
        When two traces carry the same query identifier, the later one wins and
        the earlier is recorded in ``skipped``, because averaging two runs of
        one question into a single row would hide the repeat.
    """
    projection = Projection()
    owner: dict[str, str] = {}
    for trace in traces:
        trace_id = str(trace.get("id", "<no id>"))
        try:
            query_id, first, last, abstained = _read(trace)
        except _UnprojectableError as reason:
            projection.skipped[trace_id] = str(reason)
            continue
        if query_id in owner:
            projection.skipped[owner[query_id]] = (
                f"query {query_id!r} reappears in trace {trace_id!r}, which replaced it"
            )
        owner[query_id] = trace_id
        projection.runs[query_id] = last
        projection.first_runs[query_id] = first
        projection.abstained[query_id] = abstained
    return projection


class _UnprojectableError(Exception):
    """A trace lacks something the convention requires."""


def _read(trace: Mapping[str, Any]) -> tuple[str, list[str], list[str], bool]:
    query_id = _query_id(trace)
    retrievals = _retrievals(trace)
    if not retrievals:
        raise _UnprojectableError("no RETRIEVER observation")
    return (
        query_id,
        _ids(retrievals[0].get("output")),
        _ids(retrievals[-1].get("output")),
        _abstained(trace.get("output")),
    )


def _query_id(trace: Mapping[str, Any]) -> str:
    for source in (trace.get("metadata"), trace.get("input")):
        if isinstance(source, Mapping) and source.get("query_id"):
            return str(source["query_id"])
    raise _UnprojectableError("no query_id in metadata or input")


def _retrievals(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations = [
        obs
        for obs in trace.get("observations") or ()
        if str(obs.get("type", "")).upper() == "RETRIEVER"
    ]
    # ISO timestamps from one server sort as strings. A missing one keeps the
    # list order, which Langfuse returns oldest first.
    return sorted(observations, key=lambda obs: str(obs.get("startTime") or ""))


def _ids(output: Any) -> list[str]:
    if isinstance(output, Mapping):
        output = output.get("evidence") or output.get("documents") or []
    ids: list[str] = []
    for item in output or ():
        found = _chunk_id(item)
        if found is None:
            raise _UnprojectableError(f"passage without an identifier: {item!r:.80}")
        ids.append(found)
    return ids


def _chunk_id(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("id", "chunk_id"):
            if item.get(key):
                return str(item[key])
        provenance = item.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get("chunk_id"):
            return str(provenance["chunk_id"])
    return None


def _abstained(output: Any) -> bool:
    if isinstance(output, Mapping):
        if isinstance(output.get("abstained"), bool):
            return bool(output["abstained"])
        output = output.get("answer")
    if isinstance(output, str):
        return bool(looks_like_abstention(output))
    raise _UnprojectableError("trace output has no `abstained` flag or answer text")


__all__ = ["DROPPED", "Projection", "project_traces"]
