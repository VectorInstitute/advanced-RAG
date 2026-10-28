"""The composed pipeline: retrieve, grade, correct, answer or abstain.

Each stage in this package is useful on its own and none of them knows about the
others. This module is where they are wired into one loop, and it is deliberately
short: if the composition needs more than a page, the stages are doing too
little.

The loop is retrieve, grade, act. Acting on a poor grade means retrieving again
with a rewritten query, and running out of attempts on a poor grade means
abstaining without a generation call. Every pass records an :class:`Attempt`
holding the question used, the passages it found, the grade, and the action, so
a participant can read why a query ended where it did instead of inferring it
from the answer.

The three decisions in that loop live in :mod:`aieng.advanced_rag.pipeline.answering`,
not here. What the question turns into before retrieval is a query analysis,
what an attempt's outcome means is a loop policy, and which template the answer
prompt is rendered from is a prompt source. This module composes them and does
the work. ``evaluation.sweep`` drives the same policy over a recording, which is
what makes a swept operating point one the pipeline has.

Retrieval enters as a callable. That is what keeps this module independent of
which retriever is behind it, testable with a fake, and free of any import from
the retrieval half of the harness. Every retriever in the package satisfies that
callable itself, through
:class:`~aieng.advanced_rag.retrieval.interface.RetrieverBase`, so an indexed
retriever is passed in as it is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from ..graphrag.evidence import Evidence, to_runs
from ..provenance import Provenance
from .answering import (
    CorrectivePolicy,
    FixedPromptSelector,
    LoopDecision,
    LoopPolicy,
    LoopState,
    PassthroughAnalysis,
    PromptSelector,
    QueryAnalysis,
    RetrievalStep,
    RewriterAnalysis,
    merge_steps,
    step_bound,
)
from .generate import GENERATION_TEMPLATE, GroundedAnswer, abstention_flags, generate
from .grading import (
    Action,
    EvidenceGrade,
    Grader,
    GradingThresholds,
    Verdict,
)
from .hierarchical import Hierarchy, small_to_big
from .rewrite import Rewriter


if TYPE_CHECKING:  # pragma: no cover - annotations only, to stay uncoupled
    from aieng.advanced_rag.retrieval.corpus import Chunk
    from aieng.advanced_rag.retrieval.retrievers import Retrieved


class Retrieve(Protocol):
    """Anything that turns a question into ranked evidence.

    ``scope`` is the corpus's own filter. On CUAD it is the contract identifier,
    because the 41 questions repeat verbatim across contracts and carry no
    signal about which one they belong to, so retrieval has to be scoped to be
    meaningful at all.

    ``k`` is the depth to retrieve at, defaulted so a caller that has no opinion
    can leave it out. :class:`AdvancedRagSpine` always passes
    ``SpineConfig.top_k``, because a depth can only be honoured by the retriever:
    truncating a finished ranking caps the configured depth at whatever the
    retriever chose for itself.
    """

    def __call__(
        self, question: str, scope: str | None = None, k: int = 10
    ) -> Sequence[Evidence]:
        """Return up to ``k`` ranked passages for one question."""
        ...


class Askable(Protocol):
    """The part of a corpus query this pipeline reads."""

    query_id: str
    question: str


@dataclass(frozen=True)
class SpineConfig:
    """The pipeline's tunable numbers, in one place.

    Parameters
    ----------
    top_k : int, optional
        Passages kept from each retrieval, passed to the retriever as its ``k``.
    evidence_budget : int, optional
        Passages placed in the generation prompt. Smaller than ``top_k`` on
        purpose: the grader sees the whole ranking, the reader sees the top of
        it.
    max_attempts : int, optional
        Retrievals allowed per question. One means no corrective loop, which is
        the baseline to measure the loop against.
    require_citations : bool, optional
        Treat an uncited answer as an abstention.
    thresholds : GradingThresholds, optional
        Where the pooled grade turns into a verdict, and the verdict into an
        action. The default loop policy is built from this, so it is the setting
        a sweep turns. A policy of your own carries its own thresholds, so
        ``AdvancedRagSpine`` refuses both rather than letting this field go dead.
        The grader keeps a third copy for the verdict it records, which is
        advisory and does not drive the action.
    """

    top_k: int = 10
    evidence_budget: int = 5
    max_attempts: int = 2
    require_citations: bool = True
    thresholds: GradingThresholds = field(default_factory=GradingThresholds)

    def __post_init__(self) -> None:
        """Reject settings that would silently answer from nothing."""
        if min(self.top_k, self.evidence_budget, self.max_attempts) < 1:
            raise ValueError(
                "top_k, evidence_budget, and max_attempts must all be at least 1"
            )


@dataclass(frozen=True)
class Attempt:
    """One pass through retrieve and grade.

    Parameters
    ----------
    number : int
        1 for the first retrieval.
    question : str
        The question sent to the retriever, which after a rewrite is not the
        question the reader asked.
    strategy : str
        Which rewriter produced that question, or ``"original"``.
    evidence_ids : tuple of str
        Ranked chunk identifiers, the form the retrieval metrics score.
    verdict : Verdict
        The grader's reading, which is advisory: the loop policy owns the
        verdict the action was taken on, and it reads the pooled score against
        ``SpineConfig.thresholds``. The two agree whenever the grader was given
        the same thresholds.
    score : float
        The pooled grade, which is what the policy reads.
    action : Action
        What the pipeline did next, which is the policy's move.
    reason : str
        The grader's reason, which is also what a rewrite was aimed at.
    rewrite_outcome : str, optional
        What the rewriter that produced this attempt's question did, from
        :attr:`~aieng.advanced_rag.pipeline.rewrite.RewritePlan.outcome`. Empty
        when no rewriter ran. An attempt that retrieved the question as asked
        reads the same either way, so without this the trace cannot say whether
        the rewriter found nothing to change or returned nothing usable.
    """

    number: int
    question: str
    strategy: str
    evidence_ids: tuple[str, ...]
    verdict: Verdict
    score: float
    action: Action
    reason: str
    rewrite_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the attempt."""
        return {
            "attempt": self.number,
            "question": self.question,
            "strategy": self.strategy,
            "evidence_ids": list(self.evidence_ids),
            "verdict": str(self.verdict),
            "score": round(self.score, 4),
            "action": str(self.action),
            "reason": self.reason,
            "rewrite_outcome": self.rewrite_outcome,
        }


@dataclass(frozen=True)
class SpineResult:
    """Everything one question produced, answer and trace together.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    question : str
        The question as asked.
    answer : GroundedAnswer
        The answer or the abstention.
    attempts : tuple of Attempt, optional
        One entry per retrieval, in order.
    evidence : tuple of Evidence, optional
        The passages the final attempt found.
    """

    query_id: str
    question: str
    answer: GroundedAnswer
    attempts: tuple[Attempt, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    @property
    def abstained(self) -> bool:
        """Whether the pipeline declined to answer."""
        return self.answer.abstained

    @property
    def retrieved_ids(self) -> list[str]:
        """Return the final attempt's ranked identifiers."""
        return to_runs(list(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the answer and its trace."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "retrieved_ids": self.retrieved_ids,
            **self.answer.to_dict(),
        }


def _analysis_for(prepare: Rewriter | None) -> QueryAnalysis:
    """Return the analysis stage a ``prepare`` rewriter stands for."""
    if prepare is None:
        return PassthroughAnalysis()
    return RewriterAnalysis(prepare)


class AdvancedRagSpine:
    """Retrieval, grading, one corrective retry, and grounded generation.

    Parameters
    ----------
    retrieve : Retrieve
        Ranked evidence for a question. Inject a fake in tests, and an indexed
        retriever otherwise: every retriever in the package is callable this
        way, at its own default depth of 10.
    grader : Grader
        Scores what came back. ``HeuristicGrader`` runs offline; ``LLMGrader``
        is the one to report from.
    complete : callable
        Takes a prompt, returns text, used for generation only.
    prepare : Rewriter or None, optional
        Rewrites the question before the first retrieval. ``TemplateRewriter``
        belongs here. Shorthand for ``analysis=RewriterAnalysis(prepare)``, so
        passing both raises.
    rewriter : Rewriter or None, optional
        Rewrites in response to a grade, on a corrective retry. With no
        rewriter, a retry has nothing new to try, so the loop settles on the
        same rule it would have used on its last attempt.
    config : SpineConfig or None, optional
        The tunable numbers.
    template : str or None, optional
        Generation prompt template. Shorthand for
        ``prompt=FixedPromptSelector(template)``, so passing both raises. ``None``
        renders from the package template.
    max_chars : int or None, optional
        Per-passage truncation in the generation prompt.
    analysis : QueryAnalysis or None, optional
        Turns the question into a retrieval plan. ``None`` uses ``prepare``
        when one is given and retrieves the question as asked otherwise. A
        decomposition goes here, and every attempt retrieves each of its steps
        and grades the pool. A stage carrying a ``max_steps`` also bounds how
        many candidates the corrective retry retrieves with.
    policy : LoopPolicy or None, optional
        Decides what each attempt's outcome means. ``None`` is the corrective
        loop at ``config.thresholds``. ``SingleShotPolicy`` drops the loop.
    prompt : PromptSelector or None, optional
        Chooses the answer prompt's template. ``None`` renders every answer
        from ``template``.

    Raises
    ------
    ValueError
        If ``prepare`` and ``analysis`` are both given, if ``template`` and
        ``prompt`` are both given, or if ``policy`` is given alongside a
        ``config.thresholds`` of your own. Each pair says the same thing two
        ways, and one used to win in silence.
    """

    def __init__(
        self,
        retrieve: Retrieve,
        grader: Grader,
        complete: Callable[[str], str],
        *,
        prepare: Rewriter | None = None,
        rewriter: Rewriter | None = None,
        config: SpineConfig | None = None,
        template: str | None = None,
        max_chars: int | None = 1200,
        analysis: QueryAnalysis | None = None,
        policy: LoopPolicy | None = None,
        prompt: PromptSelector | None = None,
    ) -> None:
        if prepare is not None and analysis is not None:
            raise ValueError(
                "pass prepare or analysis, not both: prepare is shorthand for "
                "analysis=RewriterAnalysis(prepare)"
            )
        if template is not None and prompt is not None:
            raise ValueError(
                "pass template or prompt, not both: template is shorthand for "
                "prompt=FixedPromptSelector(template)"
            )
        settings = config or SpineConfig()
        if policy is not None and settings.thresholds != GradingThresholds():
            raise ValueError(
                "pass config.thresholds or policy, not both: a policy carries "
                "the thresholds it decides on, so config.thresholds would be "
                "read by nothing while a sweep still labelled rows with it"
            )
        self.retrieve = retrieve
        self.grader = grader
        self.complete = complete
        self.prepare = prepare
        self.rewriter = rewriter
        self.config = settings
        self.max_chars = max_chars
        self.analysis = analysis if analysis is not None else _analysis_for(prepare)
        self.max_steps = step_bound(self.analysis)
        self.policy = (
            policy if policy is not None else CorrectivePolicy(self.config.thresholds)
        )
        # No ``self.template``: it reported the losing value of the pair, saying
        # one template while the prompt rendered another.
        self.prompt = (
            prompt
            if prompt is not None
            else FixedPromptSelector(
                GENERATION_TEMPLATE if template is None else template
            )
        )

    def answer(
        self, query_id: str, question: str, scope: str | None = None
    ) -> SpineResult:
        """Answer one question, or abstain, recording every decision on the way.

        Parameters
        ----------
        query_id : str
            Identifier, carried through to the answer so the null scorer can
            key on it.
        question : str
            The question as asked.
        scope : str or None, optional
            Corpus filter passed to the retriever, the contract identifier on
            CUAD.

        Returns
        -------
        SpineResult
            The answer and the full trace.
        """
        plan = self.analysis.analyse(question, scope)
        steps = list(plan.retrieval_steps())
        outcome = plan.rewrite_outcome
        attempts: list[Attempt] = []
        evidence: list[Evidence] = []
        notes: list[str] = []

        for number in range(1, self.config.max_attempts + 1):
            evidence = []
            for step in steps:
                found = self.retrieve(step.question, step.scope, self.config.top_k)
                # The slice holds a retriever that returns more than it was
                # asked for to top_k. It cannot raise the depth, which is why
                # the configured k is passed to the retriever as well.
                evidence.extend(list(found)[: self.config.top_k])
            asked = merge_steps(steps)
            grade = self.grader.grade(asked.question, evidence, query_id=query_id)
            state = LoopState(
                plan=plan,
                step=asked,
                score=grade.score,
                had_evidence=bool(evidence),
                attempts_left=self.config.max_attempts - number,
                rewrite_available=self.rewriter is not None,
                evidence=tuple(evidence),
                grade=grade,
                notes=tuple(notes),
            )
            move = self.policy.next_move(state)
            if move.note:
                notes.append(move.note)

            next_steps: list[RetrievalStep] = []
            next_outcome = ""
            if move.action is Action.REWRITE:
                next_steps, next_outcome = self._next_steps(
                    move, question, grade, scope
                )
                if not next_steps:
                    move = self._settled(state)

            attempts.append(
                Attempt(
                    number=number,
                    question=asked.question,
                    strategy=asked.strategy,
                    evidence_ids=tuple(to_runs(evidence)),
                    verdict=grade.verdict,
                    score=grade.score,
                    action=move.action,
                    reason=grade.reason,
                    rewrite_outcome=outcome,
                )
            )
            if move.action is Action.ANSWER:
                break
            if move.action is Action.ABSTAIN:
                return SpineResult(
                    query_id=query_id,
                    question=question,
                    answer=GroundedAnswer.abstain(
                        query_id,
                        reason=(
                            f"evidence graded {grade.verdict} at "
                            f"{grade.score:.2f}: {grade.reason}"
                        ),
                    ),
                    attempts=tuple(attempts),
                    evidence=tuple(evidence),
                )
            steps = next_steps
            outcome = next_outcome

        # The loop always breaks or returns: on the last pass no attempts are
        # left, and a settled move is never another retrieval.
        return SpineResult(
            query_id=query_id,
            question=question,
            answer=generate(
                query_id,
                plan.question,
                evidence,
                self.complete,
                template=self.prompt.get_template(plan, evidence),
                budget=self.config.evidence_budget,
                max_chars=self.max_chars,
                require_citations=self.config.require_citations,
            ),
            attempts=tuple(attempts),
            evidence=tuple(evidence),
        )

    def _next_steps(
        self,
        move: LoopDecision,
        question: str,
        grade: EvidenceGrade,
        scope: str | None,
    ) -> tuple[list[RetrievalStep], str]:
        """Return what a retry retrieves with, which may be nothing, and why.

        A policy that chose its own next retrieval carries it on the move. Every
        other policy falls back to the corrective rewriter, which is asked why
        the last attempt failed and is paid for whether or not it answers.

        The retry keeps as many candidates as the analysis stage's own bound
        allows, so it retrieves as widely as the first attempt did. Capping it
        at one while the first attempt planned three would narrow the loop on
        the pass that needed it most.

        Returns
        -------
        tuple
            The steps to retrieve with, and the rewriter's ``outcome``, which is
            empty when no rewriter ran.
        """
        if move.step is not None:
            return [move.step], ""
        if self.rewriter is None:
            return [], ""
        plan = self.rewriter.rewrite(question, feedback=grade.reason)
        candidates = plan.questions(include_original=False)[: self.max_steps]
        if not candidates:
            return [], plan.outcome
        return [
            RetrievalStep(
                question=candidate,
                scope=scope,
                strategy=self.rewriter.name,
                parent=question,
            )
            for candidate in candidates
        ], plan.outcome

    def _settled(self, state: LoopState) -> LoopDecision:
        """Settle a retry that has nothing to retrieve with.

        The rewriter has already run and been paid for, so the loop asks the
        policy again with no attempts left, which is the rule its final attempt
        would have applied.
        """
        move = self.policy.next_move(replace(state, attempts_left=0))
        if move.action is Action.REWRITE:
            raise ValueError(
                f"loop policy {self.policy.name!r} asked for another retrieval "
                "with no attempts left, so the loop cannot settle."
            )
        return move


def run_spine(spine: AdvancedRagSpine, queries: Iterable[Askable]) -> list[SpineResult]:
    """Run the pipeline over a set of queries.

    Accepts anything carrying ``query_id`` and ``question``; a ``contract_id``,
    if present, becomes the retrieval scope. That is the harness query type
    without importing it.
    """
    return [
        spine.answer(
            query.query_id, query.question, getattr(query, "contract_id", None)
        )
        for query in queries
    ]


def abstentions(results: Iterable[SpineResult]) -> dict[str, bool]:
    """Return the mapping ``evaluation.metrics.score_null_detection`` scores."""
    return abstention_flags(result.answer for result in results)


def retrieval_runs(results: Iterable[SpineResult]) -> dict[str, list[str]]:
    """Return the runs ``evaluation.metrics.score_retrieval`` scores.

    The final attempt's ranking, since that is what the pipeline delivered. To
    price the corrective loop, score ``result.attempts[0].evidence_ids`` the
    same way and compare.
    """
    return {result.query_id: result.retrieved_ids for result in results}


def evidence_from_hits(
    hits: Sequence["Retrieved"],
    chunks: Mapping[str, "Chunk"],
    retriever: str = "",
) -> list[Evidence]:
    """Turn harness retriever hits into the shared evidence type.

    Hits carry an identifier and a score; the text and the provenance come from
    the chunk. A hit whose chunk is not in the mapping is dropped, which keeps a
    stale index from putting text-free evidence in front of a reader.
    """
    evidence: list[Evidence] = []
    for hit in hits:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            continue
        evidence.append(
            Evidence(
                text=chunk.text,
                provenance=Provenance.from_chunk(chunk),
                score=hit.score,
                retriever=retriever,
            )
        )
    return evidence


def with_small_to_big(
    retrieve: Retrieve, hierarchy: Hierarchy, limit: int | None = None
) -> Retrieve:
    """Wrap a retriever so retrieved children are read as their parents.

    Composed at the retrieval boundary rather than inside the pipeline, so the
    grading and generation stages never learn that chunking is hierarchical.
    """

    def expanded(
        question: str, scope: str | None = None, k: int = 10
    ) -> list[Evidence]:
        return small_to_big(hierarchy, retrieve(question, scope, k), limit=limit)

    return expanded
