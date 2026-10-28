"""Sweeping the corrective loop's thresholds without re-running the model.

``GradingThresholds`` turns a pooled grade into a verdict, and the verdict into
an action. Nothing upstream of that turn depends on the thresholds: the grader
scores each passage against the question, and the generator reads a fixed
evidence list. So one pass that records those scores and those generation
outcomes answers an entire threshold grid, and the grid costs nothing beyond
the first pass.

That is what this module is. :func:`harvest_traces` runs retrieve, grade,
rewrite, and generate once per attempt and records the threshold-free parts.
:func:`replay` then reproduces :meth:`AdvancedRagSpine.answer` over a recorded
trace under any replayable loop policy, and ``tests/.../test_sweep.py`` asserts
the two agree, because a replay that drifts from the spine would report an
operating point the pipeline does not have. A grid point is a policy built at
that point's thresholds, which is how :func:`sweep` walks the grid.

Every row reports both sides. Abstention scored on its own is maximised by
abstaining on everything, so :class:`SweepRow` carries what the same thresholds
cost on the answerable questions next to what they buy on the null ones, and
:attr:`SweepRow.balanced_accuracy` is the single number on which both trivial
controls score 0.5.

The decision itself is a :class:`~aieng.advanced_rag.pipeline.answering.LoopPolicy`,
the same object the spine drives, so the two loops cannot drift apart on what a
grade means. A policy that calls a model inside its own decision cannot be
answered from a recording at all, so it declares itself unreplayable and
:func:`replay` refuses it rather than reporting a row for a loop nobody ran.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, Protocol

from ..graphrag.evidence import Evidence, to_runs
from ..pipeline.answering import (
    CorrectivePolicy,
    FixedPromptSelector,
    LoopDecision,
    LoopPolicy,
    LoopState,
    PassthroughAnalysis,
    PromptSelector,
    QueryAnalysis,
    QueryPlan,
    RetrievalStep,
    merge_steps,
    step_bound,
)
from ..pipeline.generate import generate
from ..pipeline.grading import Action, Grader, GradingThresholds
from ..pipeline.rewrite import Rewriter
from .metrics import NullReport, score_null_detection


class Judged(Protocol):
    """The part of a corpus query the harvest reads."""

    query_id: str
    question: str

    @property
    def is_answerable(self) -> bool:
        """Whether the dataset records a gold span for this question."""
        ...


@dataclass(frozen=True)
class AttemptTrace:
    """One retrieve-and-grade pass, recorded before any threshold is applied.

    Parameters
    ----------
    question : str
        The question sent to the retriever, which after a rewrite is not the
        question the reader asked.
    strategy : str
        Which rewriter produced that question, or ``"original"``.
    evidence_ids : tuple of str
        Ranked chunk identifiers this retrieval returned.
    scores : tuple of float
        The grader's per-passage scores, in the same rank order. Empty when
        retrieval returned nothing.
    reason : str
        The grader's reason, which is what the rewrite was aimed at.
    generated_abstained : bool
        What :func:`~aieng.advanced_rag.pipeline.generate.generate` did on this
        evidence. Recorded for every attempt because which attempt reaches the
        generator is exactly what the thresholds decide.
    generated_reason : str, optional
        Why generation declined, empty when it answered. Four paths abstain and
        they are not interchangeable: the model's own ``not_stated`` flag is a
        judgment about the evidence, while the citation gate is a formatting
        contract. A sweep that cannot tell them apart cannot say which one its
        null detection rests on.
    rewrite_available : bool
        Whether the rewriter produced a question to retry with. False makes a
        retry impossible, and the spine then settles on the rule its final
        attempt would have applied.
    grader_calls : int, optional
        Model calls the grader made here. Zero for ``HeuristicGrader``.
    rewrite_calls : int, optional
        Model calls the rewriter made after this attempt.
    generation_calls : int, optional
        Model calls generation made on this evidence, which is zero when
        retrieval returned nothing.
    scope : str or None, optional
        The corpus filter this retrieval ran under. A decomposition varies it
        per sub-question, so it is recorded per attempt rather than per query.
    parent : str, optional
        The question this attempt's question was derived from, empty when it is
        the question as asked.
    rewrite_outcome : str, optional
        What the rewriter that produced this attempt's question did, from
        :attr:`~aieng.advanced_rag.pipeline.rewrite.RewritePlan.outcome`. Empty
        when no rewriter ran. ``rewrite_available`` says whether there was
        anything to retry with; this says why there was not, which is the
        difference between a rewriter that never fires on a corpus and one whose
        output is always discarded.
    """

    question: str
    strategy: str
    evidence_ids: tuple[str, ...]
    scores: tuple[float, ...]
    reason: str
    generated_abstained: bool
    rewrite_available: bool
    generated_reason: str = ""
    grader_calls: int = 0
    rewrite_calls: int = 0
    generation_calls: int = 0
    scope: str | None = None
    parent: str = ""
    rewrite_outcome: str = ""

    def step(self) -> RetrievalStep:
        """Return the plan step this attempt retrieved with.

        Returns
        -------
        RetrievalStep
            The recorded question, scope, strategy, and parent, which is what a
            replay hands a loop policy in place of the live step.
        """
        return RetrievalStep(
            question=self.question,
            scope=self.scope,
            strategy=self.strategy,
            parent=self.parent,
        )

    @property
    def score(self) -> float:
        """Return the pooled grade, the maximum, as ``pooled_score`` computes it."""
        return max(self.scores, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the attempt."""
        return {
            "question": self.question,
            "strategy": self.strategy,
            "evidence_ids": list(self.evidence_ids),
            "scores": [round(score, 4) for score in self.scores],
            "score": round(self.score, 4),
            "reason": self.reason,
            "generated_abstained": self.generated_abstained,
            "generated_reason": self.generated_reason,
            "rewrite_available": self.rewrite_available,
            "grader_calls": self.grader_calls,
            "rewrite_calls": self.rewrite_calls,
            "generation_calls": self.generation_calls,
            "scope": self.scope,
            "parent": self.parent,
            "rewrite_outcome": self.rewrite_outcome,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttemptTrace":
        """Rebuild an attempt from :meth:`to_dict`."""
        return cls(
            question=str(payload["question"]),
            strategy=str(payload["strategy"]),
            evidence_ids=tuple(payload["evidence_ids"]),
            scores=tuple(float(score) for score in payload["scores"]),
            reason=str(payload["reason"]),
            generated_abstained=bool(payload["generated_abstained"]),
            generated_reason=str(payload.get("generated_reason", "")),
            rewrite_available=bool(payload["rewrite_available"]),
            grader_calls=int(payload.get("grader_calls", 0)),
            rewrite_calls=int(payload.get("rewrite_calls", 0)),
            generation_calls=int(payload.get("generation_calls", 0)),
            scope=payload.get("scope") or None,
            parent=str(payload.get("parent", "")),
            rewrite_outcome=str(payload.get("rewrite_outcome", "")),
        )


@dataclass(frozen=True)
class QueryTrace:
    """Every attempt one question produced, with the labels needed to score it.

    Parameters
    ----------
    query_id : str
        Identifier the null scorer keys on.
    question : str
        The question as asked.
    is_null : bool
        Whether the dataset marks the question unanswerable.
    gold_chunk_ids : frozenset of str
        The judgment set, empty on a null query. Used to say whether an answered
        question was answered from the right passage.
    attempts : tuple of AttemptTrace
        Attempts in order, as far as the harvest went.
    step_questions : tuple of str, optional
        The questions the analysis stage planned for the first attempt, in
        order. A replay rebuilds the plan from these, so a policy that reads
        ``state.plan.steps`` decides on the same plan in both loops. Empty on a
        trace harvested before this was recorded, which replays as the plan with
        no steps that the spine falls back to anyway.
    """

    query_id: str
    question: str
    is_null: bool
    gold_chunk_ids: frozenset[str]
    attempts: tuple[AttemptTrace, ...]
    step_questions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the trace."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "is_null": self.is_null,
            "gold_chunk_ids": sorted(self.gold_chunk_ids),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "step_questions": list(self.step_questions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryTrace":
        """Rebuild a trace from :meth:`to_dict`."""
        return cls(
            query_id=str(payload["query_id"]),
            question=str(payload["question"]),
            is_null=bool(payload["is_null"]),
            gold_chunk_ids=frozenset(payload["gold_chunk_ids"]),
            attempts=tuple(
                AttemptTrace.from_dict(attempt) for attempt in payload["attempts"]
            ),
            step_questions=tuple(
                str(question) for question in payload.get("step_questions", ())
            ),
        )


@dataclass(frozen=True)
class Outcome:
    """What one trace does at one set of thresholds.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    abstained : bool
        Whether the pipeline declined to answer. This is what
        ``score_null_detection`` reads.
    abstained_by : str
        ``"grader"`` when the corrective loop abstained without generating,
        ``"generator"`` when generation declined, and ``""`` on an answer. The
        thresholds move the first of those directly and the second only by
        changing which evidence reaches the generator.
    attempts_used : int
        Retrievals spent.
    final_evidence_ids : tuple of str
        The ranked identifiers the last attempt returned.
    model_calls : int
        Calls one production run at these thresholds would make. Grading on the
        attempts actually taken, the rewriter on every attempt the loop asked
        for a retry on, whether or not the rewrite came back, plus generation
        when the pipeline generated.
    """

    query_id: str
    abstained: bool
    abstained_by: str
    attempts_used: int
    final_evidence_ids: tuple[str, ...]
    model_calls: int


def _priced_move(
    loop: LoopPolicy,
    state: LoopState,
    move: LoopDecision,
    attempt: AttemptTrace,
    spent: int,
) -> tuple[Action, int]:
    """Price a retry the way the spine spends on one, and settle it if it cannot run.

    Parameters
    ----------
    loop : LoopPolicy
        The policy that produced ``move``, re-asked when the retry cannot run.
    state : LoopState
        The state ``move`` was decided from.
    move : LoopDecision
        What the policy asked for.
    attempt : AttemptTrace
        The recorded attempt, which carries what the rewriter cost.
    spent : int
        Model calls charged so far.

    Returns
    -------
    tuple of (Action, int)
        The action the loop takes and the running charge.

    Raises
    ------
    ValueError
        If the policy asks for another retrieval with no attempts left.
    """
    if move.action is not Action.REWRITE:
        return move.action, spent
    # The rewriter runs before the loop knows whether it found anything, so the
    # spine pays for the call either way. Charging it only when a rewrite came
    # back under-reported a run by one call on every attempt whose rewriter
    # returned nothing.
    spent += attempt.rewrite_calls
    if attempt.rewrite_available:
        return Action.REWRITE, spent
    # Nothing new to retrieve with, so settle on the rule the final attempt
    # would have applied.
    settled = loop.next_move(replace(state, attempts_left=0))
    if settled.action is Action.REWRITE:
        raise ValueError(
            f"loop policy {loop.name!r} asked for another retrieval with no "
            "attempts left, so the loop cannot settle."
        )
    return settled.action, spent


def _recorded_plan(trace: QueryTrace) -> QueryPlan:
    """Rebuild the plan the analysis stage produced for a recorded trace.

    The plan used to be built with no steps at all. ``CorrectivePolicy`` reads
    neither the steps nor the outcome, so the two loops agreed, and a policy
    that chose its next retrieval from ``state.plan.steps`` would have seen a
    decomposition on the spine and an empty plan on the replay. That
    disagreement would have surfaced as a sweep row rather than as an error.

    Parameters
    ----------
    trace : QueryTrace
        A harvested trace.

    Returns
    -------
    QueryPlan
        The question as asked with one step per recorded step question. The
        scope and the labels come from the first attempt, which is where the
        harvest merged them from the same plan.
    """
    first = trace.attempts[0] if trace.attempts else None
    return QueryPlan(
        question=trace.question,
        steps=tuple(
            RetrievalStep(
                question=question,
                scope=first.scope if first is not None else None,
                strategy=first.strategy if first is not None else "original",
                parent=first.parent if first is not None else "",
            )
            for question in trace.step_questions
        ),
        rewrite_outcome=first.rewrite_outcome if first is not None else "",
    )


def replay(
    trace: QueryTrace,
    policy: LoopPolicy,
    max_attempts: int = 2,
) -> Outcome:
    """Reproduce the spine's decision over a recorded trace under ``policy``.

    This drives the same :class:`~aieng.advanced_rag.pipeline.answering.LoopPolicy`
    :meth:`AdvancedRagSpine.answer` drives, over a
    :class:`~aieng.advanced_rag.pipeline.answering.LoopState` built from the
    recording instead of from live work. One decision, two drivers: this one
    reads recorded values and prices them, and the spine's does the work. The
    agreement is the load-bearing part, so it is asserted against the spine in
    the tests rather than assumed.

    The policy is the only decision source. A grid point is replayed by handing
    it to the policy that reads it, as
    ``replay(trace, CorrectivePolicy(thresholds))``, so a row's threshold label
    can only name the thresholds the decision was made under.

    Parameters
    ----------
    trace : QueryTrace
        A harvested trace. It must hold at least ``max_attempts`` attempts, or
        an attempt whose rewrite was unavailable, since either ends the loop.
    policy : LoopPolicy
        The loop to replay, carrying its own verdict boundaries.
        :class:`~aieng.advanced_rag.pipeline.answering.CorrectivePolicy` at a
        set of thresholds is what the spine ships.
    max_attempts : int, optional
        Retrievals allowed. One disables the corrective loop.

    Returns
    -------
    Outcome
        The decision and what it cost.

    Raises
    ------
    ValueError
        If ``max_attempts`` is below one, the policy cannot be replayed, the
        trace ran out of attempts while the loop still wanted another one, or
        the policy asked for a retry with no attempts left.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not policy.replayable:
        raise ValueError(
            f"loop policy {policy.name!r} is not replayable, so a recorded trace "
            "cannot answer it. A policy that calls a model inside its own "
            "decision has to be run, not replayed: measure it with "
            "AdvancedRagSpine and report that run instead of a sweep row."
        )

    plan = _recorded_plan(trace)
    notes: list[str] = []
    spent = 0
    for number in range(1, max_attempts + 1):
        if number > len(trace.attempts):
            raise ValueError(
                f"trace {trace.query_id} holds {len(trace.attempts)} attempts "
                f"but the loop asked for attempt {number}. Harvest at least "
                "max_attempts attempts before replaying."
            )
        attempt = trace.attempts[number - 1]
        spent += attempt.grader_calls
        state = LoopState(
            plan=plan,
            step=attempt.step(),
            score=attempt.score,
            had_evidence=bool(attempt.evidence_ids),
            attempts_left=max_attempts - number,
            rewrite_available=attempt.rewrite_available,
            notes=tuple(notes),
        )
        move = policy.next_move(state)
        if move.note:
            notes.append(move.note)
        action, spent = _priced_move(policy, state, move, attempt, spent)
        if action is Action.ANSWER:
            return Outcome(
                query_id=trace.query_id,
                abstained=attempt.generated_abstained,
                abstained_by="generator" if attempt.generated_abstained else "",
                attempts_used=number,
                final_evidence_ids=attempt.evidence_ids,
                model_calls=spent + attempt.generation_calls,
            )
        if action is Action.ABSTAIN:
            return Outcome(
                query_id=trace.query_id,
                abstained=True,
                abstained_by="grader",
                attempts_used=number,
                final_evidence_ids=attempt.evidence_ids,
                model_calls=spent,
            )

    raise ValueError(  # pragma: no cover - decide never rewrites with none left
        f"trace {trace.query_id} did not settle within {max_attempts} attempts"
    )


def always_answer(trace: QueryTrace) -> Outcome:
    """Return the control that never abstains, whatever the evidence.

    The floor on the answerable side and the worst case on the null side. Its
    false-answer rate is 1.0 by construction, which is what makes a 43.2% null
    rate legible: a system this control beats has not detected anything.
    """
    attempt = trace.attempts[0]
    return Outcome(
        query_id=trace.query_id,
        abstained=False,
        abstained_by="",
        attempts_used=1,
        final_evidence_ids=attempt.evidence_ids,
        model_calls=attempt.generation_calls,
    )


def generator_only(trace: QueryTrace) -> Outcome:
    """Return the control with no grader at all: retrieve once, generate once.

    This is the number the corrective loop has to beat, and the one the two
    trivial controls cannot supply. ``generate`` already has its own abstention
    path, so a pipeline with the grading verdict removed still declines
    questions; what it does not do is decline them before generating. Anything
    the thresholds buy has to be measured against this, not against always
    answering.
    """
    attempt = trace.attempts[0]
    return Outcome(
        query_id=trace.query_id,
        abstained=attempt.generated_abstained,
        abstained_by="generator" if attempt.generated_abstained else "",
        attempts_used=1,
        final_evidence_ids=attempt.evidence_ids,
        model_calls=attempt.generation_calls,
    )


def always_abstain(trace: QueryTrace) -> Outcome:
    """Return the control that never answers.

    Its abstention recall is 1.0 and its precision is the corpus null rate, so
    it is the number any threshold has to beat on both sides at once.
    """
    return Outcome(
        query_id=trace.query_id,
        abstained=True,
        abstained_by="grader",
        attempts_used=1,
        final_evidence_ids=trace.attempts[0].evidence_ids,
        model_calls=0,
    )


@dataclass(frozen=True)
class AnswerableReport:
    """What an operating point costs on the questions that do have an answer.

    ``score_null_detection`` prices abstention. It cannot price the other side
    on its own: abstention precision falls when answerable questions are
    refused, but it moves with the corpus null rate too, so it does not say how
    many answerable questions were lost. These rates do.

    Parameters
    ----------
    answerable_queries : int
        Questions with at least one gold span.
    abstain_rate : float
        Share of them the pipeline declined. This is the cost of the operating
        point, stated directly.
    grounded_answer_rate : float
        Share of them answered with a gold chunk among the passages the
        generator was shown. An answer built from the wrong passage is not a
        recovered question, so this is the stricter reading of the same side.
    """

    answerable_queries: int
    abstain_rate: float
    grounded_answer_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary."""
        return {
            "answerable_queries": self.answerable_queries,
            "answerable_abstain_rate": self.abstain_rate,
            "grounded_answer_rate": self.grounded_answer_rate,
        }


def score_answerable(
    outcomes: Iterable[Outcome],
    traces: Iterable[QueryTrace],
    *,
    evidence_budget: int = 5,
) -> AnswerableReport:
    """Score what an operating point did to the answerable questions.

    Parameters
    ----------
    outcomes : iterable of Outcome
        One per query, from :func:`replay` or a control.
    traces : iterable of QueryTrace
        The traces those outcomes came from, supplying the labels.
    evidence_budget : int, optional
        Passages the generator was shown, matching ``SpineConfig``. Only these
        count towards ``grounded_answer_rate``, because the rest were graded and
        never read.

    Returns
    -------
    AnswerableReport
        The aggregate over the answerable queries alone.

    Raises
    ------
    ValueError
        If no answerable query was scored.
    """
    by_id = {trace.query_id: trace for trace in traces}
    abstained: list[float] = []
    grounded: list[float] = []
    for outcome in outcomes:
        trace = by_id[outcome.query_id]
        if trace.is_null:
            continue
        abstained.append(1.0 if outcome.abstained else 0.0)
        read = set(outcome.final_evidence_ids[:evidence_budget])
        hit = bool(read & trace.gold_chunk_ids) and not outcome.abstained
        grounded.append(1.0 if hit else 0.0)

    if not abstained:
        raise ValueError("No answerable queries to score")

    return AnswerableReport(
        answerable_queries=len(abstained),
        abstain_rate=round(mean(abstained), 4),
        grounded_answer_rate=round(mean(grounded), 4),
    )


@dataclass(frozen=True)
class SweepRow:
    """One arm of the sweep, with both sides of the trade-off on it.

    Parameters
    ----------
    label : str
        How the arm is named in a table.
    grader : str
        Which grader produced the scores replayed here.
    sufficient : float or None
        The answer threshold, or ``None`` on a control that has none.
    insufficient : float or None
        The rewrite threshold, or ``None`` on a control.
    max_attempts : int
        Retrievals allowed per question.
    null : NullReport
        The null-detection side, from ``score_null_detection``.
    answerable : AnswerableReport
        What the same setting cost on the answerable questions.
    model_calls : int
        Calls one production run at this operating point would make over the
        slice. The sweep's own harvest costs more, once, for the whole grid.
    mean_attempts : float
        Retrievals spent per question.
    grader_abstentions : int, optional
        Abstentions the corrective loop made without generating. The rest came
        from the generator's own "not stated" path, which the thresholds reach
        only by changing which evidence the generator is shown, so this says
        how much of the result the thresholds are responsible for.
    """

    label: str
    grader: str
    sufficient: float | None
    insufficient: float | None
    max_attempts: int
    null: NullReport
    answerable: AnswerableReport
    model_calls: int
    mean_attempts: float
    grader_abstentions: int = 0

    @property
    def balanced_accuracy(self) -> float:
        """Return the mean of abstention recall and answerable answer rate.

        The one number that orders the grid honestly. Abstention recall alone is
        maximised by abstaining on everything and the answer rate alone by
        answering everything, so both trivial controls land on 0.5 here and a
        threshold has to move both sides to beat them. It also ignores the base
        rate, so ContractNLI at 43.2% null and CUAD at 70% are read on the same
        scale.
        """
        return round(
            (self.null.abstain_recall + (1.0 - self.answerable.abstain_rate)) / 2, 4
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the row."""
        return {
            "label": self.label,
            "grader": self.grader,
            "sufficient": self.sufficient,
            "insufficient": self.insufficient,
            "max_attempts": self.max_attempts,
            "balanced_accuracy": self.balanced_accuracy,
            **self.null.to_dict(),
            **self.answerable.to_dict(),
            "grader_abstentions": self.grader_abstentions,
            "model_calls": self.model_calls,
            "mean_attempts": self.mean_attempts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SweepRow":
        """Rebuild a row from :meth:`to_dict`.

        A written sweep is the record every reported number traces to, so the
        tables in a memo are rendered from the file rather than copied out of a
        terminal.
        """
        return cls(
            label=str(payload["label"]),
            grader=str(payload["grader"]),
            sufficient=payload["sufficient"],
            insufficient=payload["insufficient"],
            max_attempts=int(payload["max_attempts"]),
            null=NullReport(
                null_queries=int(payload["null_queries"]),
                answerable_queries=int(payload["answerable_queries"]),
                abstain_precision=float(payload["abstain_precision"]),
                abstain_recall=float(payload["abstain_recall"]),
                false_answer_rate=float(payload["false_answer_rate"]),
            ),
            answerable=AnswerableReport(
                answerable_queries=int(payload["answerable_queries"]),
                abstain_rate=float(payload["answerable_abstain_rate"]),
                grounded_answer_rate=float(payload["grounded_answer_rate"]),
            ),
            model_calls=int(payload["model_calls"]),
            mean_attempts=float(payload["mean_attempts"]),
            grader_abstentions=int(payload.get("grader_abstentions", 0)),
        )

    def table_row(self) -> str:
        """Render one Markdown row under :data:`TABLE_HEADER`."""
        return (
            f"| {self.label} | {self.max_attempts} | "
            f"{self.null.abstain_recall:.3f} | "
            f"{self.null.abstain_precision:.3f} | "
            f"{self.null.false_answer_rate:.3f} | "
            f"{self.answerable.abstain_rate:.3f} | "
            f"{self.answerable.grounded_answer_rate:.3f} | "
            f"{self.balanced_accuracy:.3f} |"
        )


TABLE_HEADER = (
    "| Thresholds | Attempts | Abstain recall | Abstain precision "
    "| False-answer rate | Answerable lost | Grounded answers | Balanced acc. |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def score_arm(
    traces: Sequence[QueryTrace],
    outcomes: Sequence[Outcome],
    *,
    label: str,
    grader: str,
    sufficient: float | None,
    insufficient: float | None,
    max_attempts: int,
    evidence_budget: int = 5,
) -> SweepRow:
    """Turn one set of outcomes into a row reporting both sides."""
    abstained = {outcome.query_id: outcome.abstained for outcome in outcomes}
    null_ids = [trace.query_id for trace in traces if trace.is_null]
    return SweepRow(
        label=label,
        grader=grader,
        sufficient=sufficient,
        insufficient=insufficient,
        max_attempts=max_attempts,
        null=score_null_detection(abstained, null_ids),
        answerable=score_answerable(outcomes, traces, evidence_budget=evidence_budget),
        model_calls=sum(outcome.model_calls for outcome in outcomes),
        mean_attempts=round(mean([outcome.attempts_used for outcome in outcomes]), 3),
        grader_abstentions=sum(
            1 for outcome in outcomes if outcome.abstained_by == "grader"
        ),
    )


def threshold_grid(
    sufficient: Sequence[float],
    insufficient: Sequence[float],
) -> list[GradingThresholds]:
    """Return every valid pairing of the two axes, in a stable order.

    Pairs where the rewrite threshold exceeds the answer threshold are dropped
    rather than raised on: they cannot produce three verdicts, so they are not
    points on the grid.
    """
    return [
        GradingThresholds(sufficient=high, insufficient=low)
        for high in sufficient
        for low in insufficient
        if low <= high
    ]


def sweep(
    traces: Sequence[QueryTrace],
    grid: Sequence[GradingThresholds],
    attempts: Sequence[int] = (1, 2),
    *,
    grader: str = "",
    evidence_budget: int = 5,
    policy_factory: Callable[[GradingThresholds], LoopPolicy] = CorrectivePolicy,
) -> list[SweepRow]:
    """Score every threshold pair at every attempt budget over one trace set.

    Parameters
    ----------
    traces : sequence of QueryTrace
        Harvested traces, one per query.
    grid : sequence of GradingThresholds
        Threshold pairs to score, from :func:`threshold_grid`.
    attempts : sequence of int, optional
        ``max_attempts`` values. The rewrite path only exists above one, so a
        sweep that omits two has not measured the corrective loop.
    grader : str, optional
        Label recorded on each row.
    evidence_budget : int, optional
        Passages the generator was shown.
    policy_factory : callable, optional
        Built once per grid point with that point's thresholds, so a custom
        loop is swept rather than flattened to one measurement under every
        label. A policy that is not replayable is refused, because a grid row
        for a loop nobody ran is worse than no row.

    Returns
    -------
    list of SweepRow
        One row per pairing, thresholds varying fastest.
    """
    rows: list[SweepRow] = []
    for max_attempts in attempts:
        for thresholds in grid:
            policy = policy_factory(thresholds)
            outcomes = [replay(trace, policy, max_attempts) for trace in traces]
            rows.append(
                score_arm(
                    traces,
                    outcomes,
                    label=(
                        f"{thresholds.sufficient:.2f} / {thresholds.insufficient:.2f}"
                    ),
                    grader=grader,
                    sufficient=thresholds.sufficient,
                    insufficient=thresholds.insufficient,
                    max_attempts=max_attempts,
                    evidence_budget=evidence_budget,
                )
            )
    return rows


def null_only_sweep(
    traces: Sequence[QueryTrace],
    grid: Sequence[GradingThresholds],
    attempts: Sequence[int] = (1, 2),
    *,
    grader: str = "",
    policy_factory: Callable[[GradingThresholds], LoopPolicy] = CorrectivePolicy,
) -> list[dict[str, Any]]:
    """Score a query set where every question is null, over the same grid.

    The out-of-domain control has no answerable questions by construction, so
    the answerable-side cost and the balanced accuracy are undefined for it and
    a :class:`SweepRow` would have to invent both. What the control can say is
    the share of far nulls the pipeline declined, and which stage declined
    them, so that is what this returns.

    Parameters
    ----------
    traces : sequence of QueryTrace
        Harvested traces, every one of them null.
    grid : sequence of GradingThresholds
        Threshold pairs to score.
    attempts : sequence of int, optional
        ``max_attempts`` values.
    grader : str, optional
        Label recorded on each row.
    policy_factory : callable, optional
        Built once per grid point, as :func:`sweep` builds it.

    Returns
    -------
    list of dict
        One record per pairing.

    Raises
    ------
    ValueError
        If any trace is answerable, which means this is the wrong scorer.
    """
    answerable = [trace.query_id for trace in traces if not trace.is_null]
    if answerable:
        raise ValueError(
            f"{len(answerable)} answerable queries in a null-only control, "
            f"e.g. {answerable[:3]}. Use sweep() instead."
        )

    rows: list[dict[str, Any]] = []
    for max_attempts in attempts:
        for thresholds in grid:
            policy = policy_factory(thresholds)
            outcomes = [replay(trace, policy, max_attempts) for trace in traces]
            declined = [outcome for outcome in outcomes if outcome.abstained]
            rows.append(
                {
                    "label": (
                        f"{thresholds.sufficient:.2f} / {thresholds.insufficient:.2f}"
                    ),
                    "grader": grader,
                    "sufficient": thresholds.sufficient,
                    "insufficient": thresholds.insufficient,
                    "max_attempts": max_attempts,
                    "null_queries": len(traces),
                    "abstain_rate": round(len(declined) / len(traces), 4),
                    "false_answer_rate": round(1 - len(declined) / len(traces), 4),
                    "grader_abstentions": sum(
                        1 for o in declined if o.abstained_by == "grader"
                    ),
                    "generator_abstentions": sum(
                        1 for o in declined if o.abstained_by == "generator"
                    ),
                    "model_calls": sum(o.model_calls for o in outcomes),
                    "mean_attempts": round(
                        mean([o.attempts_used for o in outcomes]), 3
                    ),
                }
            )
    return rows


def control_rows(
    traces: Sequence[QueryTrace],
    *,
    grader: str = "control",
    evidence_budget: int = 5,
) -> list[SweepRow]:
    """Return the always-answer and always-abstain floors.

    None of them reads a threshold, so all three are the same on every arm of
    the sweep and all three belong in every table. Always answering and always
    abstaining are the floor, and both score 0.5 on
    :attr:`SweepRow.balanced_accuracy` by construction, which is what makes a
    corpus's null rate legible. The generator-only row is the one a threshold
    actually has to beat.
    """
    rows = []
    for label, control in (
        ("no grader", generator_only),
        ("always answer", always_answer),
        ("always abstain", always_abstain),
    ):
        outcomes = [control(trace) for trace in traces]
        rows.append(
            score_arm(
                traces,
                outcomes,
                label=label,
                grader=grader,
                sufficient=None,
                insufficient=None,
                max_attempts=1,
                evidence_budget=evidence_budget,
            )
        )
    return rows


def harvest_traces(
    queries: Iterable[Judged],
    retrieve: Callable[[str, str | None, int], Sequence[Evidence]],
    grader: Grader,
    complete: Callable[[str], str],
    qrels: dict[str, set[str]],
    *,
    analysis: QueryAnalysis | None = None,
    rewriter: Rewriter | None = None,
    prompt: PromptSelector | None = None,
    top_k: int = 10,
    evidence_budget: int = 5,
    attempts: int = 2,
    max_chars: int | None = 1200,
    require_citations: bool = True,
    count_calls: Callable[[], int] | None = None,
    on_query: Callable[[QueryTrace], None] | None = None,
) -> list[QueryTrace]:
    """Run the pipeline once per attempt and record the threshold-free parts.

    The loop here is :meth:`AdvancedRagSpine.answer` with the decision removed.
    It always takes ``attempts`` retrievals, whatever the grades say, because
    the point is to record what every threshold pair would have seen. It also
    generates on every attempt's evidence for the same reason. A harvest is
    therefore more expensive than one pipeline run and cheaper than a grid.

    The rewriter is not called on the final attempt. ``decide`` never returns
    ``REWRITE`` with no attempts left, so that call would be paid for and never
    read.

    Parameters
    ----------
    queries : iterable
        Anything carrying ``query_id``, ``question``, ``is_answerable``, and
        optionally ``doc_id`` for the retrieval scope.
    retrieve : callable
        Takes ``(question, scope, k)``, returns ranked evidence. The depth is
        passed rather than applied afterwards, because truncating a shallower
        result cannot deepen it, and a harvest that retrieved less than the
        spine would replay chunk identifiers the pipeline never saw.
    grader : Grader
        Scores the evidence.
    complete : callable
        Takes a prompt, returns text. Used for generation and rewriting.
    qrels : dict
        Query identifier to its relevant chunk identifiers.
    analysis : QueryAnalysis or None, optional
        The pre-retrieval stage, which has to be the one the spine runs. A
        harvest that cannot represent it records the question as asked while
        the spine retrieves something else, and the replay then reports chunk
        identifiers the pipeline never returned.
    rewriter : Rewriter or None, optional
        Corrective rewriter. Without one there is no second attempt, and
        ``attempts`` above one records nothing extra.
    prompt : PromptSelector or None, optional
        Where the answer prompt's template comes from, which also has to be the
        one the spine uses. ``None`` is the shipped
        :data:`~aieng.advanced_rag.pipeline.generate.GENERATION_TEMPLATE`. A
        harvest fixed on one template while the spine renders another records
        abstentions the pipeline does not make, because the two formats trip
        different abstention paths.
    top_k : int, optional
        Passages kept from each retrieval.
    evidence_budget : int, optional
        Passages placed in the generation prompt.
    attempts : int, optional
        Retrievals to record per question.
    max_chars : int or None, optional
        Per-passage truncation in the generation prompt.
    require_citations : bool, optional
        Treat an uncited answer as an abstention, matching ``SpineConfig``.
    count_calls : callable or None, optional
        Returns the running total of model calls the shared client has made.
        Every stage is priced by the difference across it, so a grader that
        skips a call, as ``LLMGrader`` does past ``max_chunks``, is counted
        honestly rather than assumed. ``None`` counts grading and rewriting as
        free and generation as one call, which is right for a model-free
        grader and wrong for anything else.
    on_query : callable or None, optional
        Called with each finished trace, for progress reporting or checkpoints.

    Returns
    -------
    list of QueryTrace
        One trace per query, in the order given.
    """
    stage = analysis if analysis is not None else PassthroughAnalysis()
    # The corrective retry keeps as many candidates as the analysis stage would,
    # which is what the spine does. A harvest that kept fewer would record a
    # narrower retry than the pipeline runs.
    max_steps = step_bound(stage)
    prompts = prompt if prompt is not None else FixedPromptSelector()
    traces: list[QueryTrace] = []
    for query in queries:
        query_id = str(query.query_id)
        question = str(query.question)
        scope = getattr(query, "doc_id", None) or None
        plan = stage.analyse(question, scope)
        steps = list(plan.retrieval_steps())
        outcome = plan.rewrite_outcome
        recorded: list[AttemptTrace] = []

        for number in range(attempts):
            evidence: list[Evidence] = []
            for step in steps:
                # Ask for the configured depth rather than slicing a default-depth
                # result, which the spine also does, so both retrieve alike.
                got = list(retrieve(step.question, step.scope, top_k))
                evidence.extend(got[:top_k])
            asked = merge_steps(steps)

            before = count_calls() if count_calls else 0
            grade = grader.grade(asked.question, evidence, query_id=query_id)
            grader_calls = count_calls() - before if count_calls else 0

            before = count_calls() if count_calls else 0
            answer = generate(
                query_id,
                plan.question,
                evidence,
                complete,
                template=prompts.get_template(plan, evidence),
                budget=evidence_budget,
                max_chars=max_chars,
                require_citations=require_citations,
            )
            generation_calls = (
                count_calls() - before if count_calls else int(bool(evidence))
            )

            next_steps: list[RetrievalStep] = []
            next_outcome = ""
            rewrite_calls = 0
            # Nothing to rewrite for on the final attempt: no policy retries
            # with no attempts left, so the call would be paid for and never
            # read.
            if rewriter is not None and number < attempts - 1:
                before = count_calls() if count_calls else 0
                rewritten = rewriter.rewrite(question, feedback=grade.reason)
                rewrite_calls = count_calls() - before if count_calls else 1
                next_outcome = rewritten.outcome
                candidates = rewritten.questions(include_original=False)[:max_steps]
                next_steps = [
                    RetrievalStep(
                        question=candidate,
                        scope=scope,
                        strategy=rewriter.name,
                        parent=question,
                    )
                    for candidate in candidates
                ]

            recorded.append(
                AttemptTrace(
                    question=asked.question,
                    strategy=asked.strategy,
                    evidence_ids=tuple(to_runs(evidence)),
                    scores=tuple(item.score for item in grade.chunk_grades),
                    reason=grade.reason,
                    generated_abstained=answer.abstained,
                    generated_reason=answer.reason,
                    rewrite_available=bool(next_steps),
                    grader_calls=grader_calls,
                    rewrite_calls=rewrite_calls,
                    generation_calls=generation_calls,
                    scope=asked.scope,
                    parent=asked.parent,
                    rewrite_outcome=outcome,
                )
            )
            if not next_steps:
                break
            steps = next_steps
            outcome = next_outcome

        trace = QueryTrace(
            query_id=query_id,
            question=question,
            is_null=not query.is_answerable,
            gold_chunk_ids=frozenset(qrels.get(query_id, set())),
            attempts=tuple(recorded),
            step_questions=tuple(step.question for step in plan.retrieval_steps()),
        )
        traces.append(trace)
        if on_query is not None:
            on_query(trace)
    return traces


__all__ = [
    "TABLE_HEADER",
    "AnswerableReport",
    "AttemptTrace",
    "Outcome",
    "QueryTrace",
    "SweepRow",
    "always_abstain",
    "always_answer",
    "control_rows",
    "generator_only",
    "harvest_traces",
    "null_only_sweep",
    "replay",
    "score_answerable",
    "score_arm",
    "sweep",
    "threshold_grid",
]
