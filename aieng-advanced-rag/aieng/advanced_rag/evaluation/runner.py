"""Run one configuration of the pipeline over a query set, and vary one part at a time.

A notebook that compares two retrievers, two graders, or two loop policies has
to hold everything else fixed, and rebuilding the spine by hand for each arm is
where that slips. :class:`Experiment` holds the fixed parts once.
:meth:`Experiment.with_` returns a copy with some of them replaced, and
:meth:`Experiment.run` executes it and scores it with the shared metrics.

    experiment = Experiment(
        queries, qrels, retrieve=dense, grader=grader, complete=chat
    )
    baseline = experiment.run("dense")
    hybrid = experiment.with_(retrieve=fused).run("hybrid")
    print(baseline.null, hybrid.null)

The runner owns no model access and no index. Every part arrives as a callable
or an object the caller built, so it runs offline in a test.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from aieng.advanced_rag.metering import RunUsage, UsageMeter, default_meter
from aieng.advanced_rag.pipeline.grading import Grader
from aieng.advanced_rag.pipeline.spine import (
    AdvancedRagSpine,
    Retrieve,
    SpineResult,
    abstentions,
    retrieval_runs,
)

from .metrics import (
    NullReport,
    RetrievalReport,
    score_null_detection,
    score_retrieval,
)


# Everything ``AdvancedRagSpine`` accepts, so a misspelt knob fails here rather
# than becoming an unused entry in a dict.
_SPINE_KNOBS = frozenset(
    {
        "retrieve",
        "grader",
        "complete",
        "prepare",
        "rewriter",
        "config",
        "template",
        "max_chars",
        "analysis",
        "policy",
        "prompt",
    }
)


class RunnableQuery(Protocol):
    """The part of a corpus query the runner reads.

    ``doc_id`` or ``contract_id``, whichever the query has, becomes the
    retrieval scope. The corpus ``Query`` type carries ``doc_id``.
    """

    query_id: str
    question: str

    @property
    def is_answerable(self) -> bool:
        """Whether the dataset records a gold span for this question."""
        ...


def scope_of(query: object) -> str | None:
    """Return the retrieval scope a query carries, or None.

    Parameters
    ----------
    query : object
        A corpus query. ``doc_id`` is read first and ``contract_id`` second.

    Returns
    -------
    str or None
        The scope, or None when the query names neither.
    """
    for name in ("doc_id", "contract_id"):
        value = getattr(query, name, None)
        if value:
            return str(value)
    return None


@dataclass(frozen=True)
class ArmResult:
    """What one configuration produced and how it scored.

    Parameters
    ----------
    name : str
        The label the run was given.
    results : tuple of SpineResult
        One per query, in the order the queries were given.
    null : NullReport
        Abstention scored against the queries that are not answerable.
    retrieval : RetrievalReport or None
        The final attempt's ranking scored against the judgments, or None when
        no query has a judgment set (a null-only slice).
    usage : RunUsage
        What the run spent on the meter, as the difference across the run.
    """

    name: str
    results: tuple[SpineResult, ...]
    null: NullReport
    retrieval: RetrievalReport | None
    usage: RunUsage

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, without the per-query results."""
        return {
            "name": self.name,
            "queries": len(self.results),
            "null": dataclasses.asdict(self.null),
            "retrieval": (
                dataclasses.asdict(self.retrieval) if self.retrieval else None
            ),
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True)
class Experiment:
    """A query set, its judgments, and one configuration of the pipeline.

    Parameters
    ----------
    queries : sequence of RunnableQuery
        The questions to run, each with ``query_id``, ``question``, and
        ``is_answerable``.
    qrels : mapping
        Query identifier to its relevant chunk identifiers.
    retrieve : Retrieve
        Turns a question into ranked evidence.
    grader : Grader
        Scores the evidence.
    complete : callable
        Takes a prompt, returns text.
    spine : mapping, optional
        Any other :class:`~aieng.advanced_rag.pipeline.spine.AdvancedRagSpine`
        keyword: ``config``, ``analysis``, ``policy``, ``prompt``,
        ``rewriter``, ``prepare``, ``template``, ``max_chars``.
    k : int, optional
        Depth to score retrieval at.
    """

    queries: Sequence[RunnableQuery]
    qrels: Mapping[str, set[str]]
    retrieve: Retrieve
    grader: Grader
    complete: Callable[[str], str]
    spine: Mapping[str, Any] = field(default_factory=dict)
    k: int = 10

    def __post_init__(self) -> None:
        """Reject an unknown spine keyword and an empty query set."""
        unknown = set(self.spine) - (_SPINE_KNOBS - {"retrieve", "grader", "complete"})
        if unknown:
            raise ValueError(
                f"unknown spine option(s) {sorted(unknown)}: "
                f"expected a subset of {sorted(_SPINE_KNOBS)}"
            )
        if not self.queries:
            raise ValueError("an experiment needs at least one query")

    def with_(self, **changes: Any) -> "Experiment":
        """Return a copy with some parts replaced and the rest held fixed.

        Parameters
        ----------
        **changes
            ``queries``, ``qrels``, ``k``, ``retrieve``, ``grader``, or
            ``complete`` replace that field. Any other
            :class:`~aieng.advanced_rag.pipeline.spine.AdvancedRagSpine`
            keyword is merged into ``spine``, and ``None`` removes it.

        Returns
        -------
        Experiment
            The variant.

        Raises
        ------
        ValueError
            If a name is neither a field nor a spine keyword.
        """
        own = {f.name for f in dataclasses.fields(self)} - {"spine"}
        spine = dict(self.spine)
        fields: dict[str, Any] = {}
        for name, value in changes.items():
            if name in own:
                fields[name] = value
            elif name in _SPINE_KNOBS:
                if value is None:
                    spine.pop(name, None)
                else:
                    spine[name] = value
            else:
                raise ValueError(
                    f"unknown knob {name!r}: expected one of "
                    f"{sorted(own | _SPINE_KNOBS)}"
                )
        return dataclasses.replace(self, spine=spine, **fields)

    def build(self) -> AdvancedRagSpine:
        """Return the pipeline this configuration describes."""
        return AdvancedRagSpine(self.retrieve, self.grader, self.complete, **self.spine)

    def run(self, name: str, *, meter: UsageMeter | None = None) -> ArmResult:
        """Execute the configuration over every query and score it.

        Parameters
        ----------
        name : str
            Label for the result.
        meter : UsageMeter or None, optional
            The meter the pipeline's model client reports to. The result's usage
            is the change on this meter across the run, so it is the right one
            only if the ``complete`` callable and the grader were built against
            it. ``None`` reads the process-wide default.

        Returns
        -------
        ArmResult
            Results, null-detection and retrieval scores, and usage.
        """
        meter = meter if meter is not None else default_meter()
        spine = self.build()
        before = meter.totals()
        results = tuple(_answer(spine, meter, query) for query in self.queries)
        usage = _difference(meter.totals(), before)
        truly_null = [q.query_id for q in self.queries if not q.is_answerable]
        runs = retrieval_runs(results)
        judged = {qid for qid in runs if self.qrels.get(qid)}
        return ArmResult(
            name=name,
            results=results,
            null=score_null_detection(abstentions(results), truly_null),
            retrieval=(score_retrieval(runs, self.qrels, k=self.k) if judged else None),
            usage=usage,
        )

    def compare(
        self,
        arms: Mapping[str, Mapping[str, Any]],
        *,
        meter: UsageMeter | None = None,
    ) -> dict[str, ArmResult]:
        """Run several variants of this configuration and return them by name.

        Parameters
        ----------
        arms : mapping
            Arm name to the knobs :meth:`with_` takes. An empty mapping runs
            this configuration unchanged.
        meter : UsageMeter or None, optional
            Passed to every :meth:`run`.

        Returns
        -------
        dict
            Arm name to result, in the order given.
        """
        return {
            name: self.with_(**dict(changes)).run(name, meter=meter)
            for name, changes in arms.items()
        }


def _answer(
    spine: AdvancedRagSpine, meter: UsageMeter, query: RunnableQuery
) -> SpineResult:
    with meter.question(query.query_id):
        return spine.answer(query.query_id, query.question, scope_of(query))


def _difference(after: RunUsage, before: RunUsage) -> RunUsage:
    """Return what the meter gained between two snapshots."""
    gained = {
        f.name: getattr(after, f.name) - getattr(before, f.name)
        for f in dataclasses.fields(RunUsage)
    }
    return RunUsage(**gained)


__all__ = ["ArmResult", "Experiment", "RunnableQuery", "scope_of"]
