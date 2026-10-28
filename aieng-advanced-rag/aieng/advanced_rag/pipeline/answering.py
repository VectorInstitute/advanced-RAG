"""The answering interface: query analysis, a loop policy, and a prompt switch.

Three things the composed pipeline used to decide for itself are objects here.
What a question turns into before retrieval is a :class:`QueryPlan`, produced by
a :class:`QueryAnalysis`. What one attempt's outcome means is a :class:`LoopDecision`,
produced by a :class:`LoopPolicy` from a :class:`LoopState`. Which template the
answer prompt is rendered from is a :class:`PromptSelector`.

The split is what lets two loops share one decision.
:meth:`~aieng.advanced_rag.pipeline.spine.AdvancedRagSpine.answer` retrieves,
grades, and generates; :func:`~aieng.advanced_rag.evaluation.sweep.replay`
reads a recording and prices it. Both build a :class:`LoopState` and both ask
the same policy what to do, so an operating point the sweep reports is one the
pipeline has.

The state carries more than the verdict and the attempts left, because a policy
that interleaves retrieval with reasoning chooses its next retrieval from the
reasoning derived so far (Trivedi et al., arXiv:2212.10509). A policy typed on
a verdict and a count could not express that, and interleaving is one of the
capabilities that justifies having a policy at all.

Not every policy can be replayed. One that calls a model inside its own
decision cannot be driven from a recording at any thresholds, so
:class:`LoopPolicy` carries a ``replayable`` flag and the sweep refuses a
policy that sets it false rather than reporting a grid row for a loop it never
ran.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..graphrag.evidence import Evidence
from .generate import GENERATION_TEMPLATE
from .grading import Action, EvidenceGrade, GradingThresholds, Verdict, decide
from .rewrite import Rewriter


if TYPE_CHECKING:
    # The spine imports this module, so the return type is an annotation only.
    # ``from __future__ import annotations`` is what makes that cost nothing.
    from .spine import SpineResult


@dataclass(frozen=True)
class RetrievalStep:
    """One retrieval the plan asks for.

    Parameters
    ----------
    question : str
        The text sent to the retriever.
    scope : str or None, optional
        Corpus filter for this step. A decomposition can vary it per
        sub-question, which is why it lives on the step and not on the plan.
    strategy : str, optional
        Which stage produced this text, or ``"original"``.
    parent : str, optional
        The step this one was derived from, empty at the top level. A
        decomposition records the question it split here, so a trace shows the
        hop structure.
    """

    question: str
    scope: str | None = None
    strategy: str = "original"
    parent: str = ""


@dataclass(frozen=True)
class QueryPlan:
    """What analysis decided to do with one question.

    Parameters
    ----------
    question : str
        The question as asked, which is what the generator is shown. A rewrite
        is a retrieval device, so answering it would answer something the
        reader did not ask.
    steps : tuple of RetrievalStep, optional
        Retrievals to run, in order. One step is the shipped behaviour; several
        is a decomposition. Empty means retrieve the question as asked.
    rewrite_outcome : str, optional
        What the pre-retrieval rewriter did, carried from
        :attr:`~aieng.advanced_rag.pipeline.rewrite.RewritePlan.outcome`. Empty
        when no rewriter ran. The spine records it on every
        :class:`~aieng.advanced_rag.pipeline.spine.Attempt`, so a run that
        retrieved the question as asked says whether that was the rewriter
        finding nothing or the rewriter failing.
    """

    question: str
    steps: tuple[RetrievalStep, ...] = ()
    rewrite_outcome: str = ""

    def retrieval_steps(self) -> tuple[RetrievalStep, ...]:
        """Return the steps to retrieve with, falling back to the question.

        Returns
        -------
        tuple of RetrievalStep
            ``steps`` when analysis produced any, otherwise one step carrying
            the question as asked.
        """
        return self.steps or (RetrievalStep(question=self.question),)


class QueryAnalysis(Protocol):
    """Anything that turns a question into a retrieval plan."""

    name: str

    def analyse(self, question: str, scope: str | None = None) -> QueryPlan:
        """Return the plan for one question.

        Parameters
        ----------
        question : str
            The question as asked.
        scope : str or None, optional
            The corpus filter the caller supplied, carried onto each step
            unless analysis narrows it.

        Returns
        -------
        QueryPlan
            The plan to retrieve and answer with.
        """
        ...


class PassthroughAnalysis:
    """Retrieve the question as asked, which is the shipped behaviour.

    Named for what it does, the way scikit-learn's ``Pipeline`` takes the literal
    ``"passthrough"`` and LangChain has ``RunnablePassthrough``. The ``name`` is a
    label recorded on every attempt and step, so it stays ``"original"``.
    """

    name = "original"

    def analyse(self, question: str, scope: str | None = None) -> QueryPlan:
        """Return a one-step plan over the question as asked.

        Parameters
        ----------
        question : str
            The question as asked.
        scope : str or None, optional
            Carried onto the single step.

        Returns
        -------
        QueryPlan
            A plan whose one step is the question itself.
        """
        return QueryPlan(
            question=question,
            steps=(RetrievalStep(question=question, scope=scope),),
        )


class RewriterAnalysis:
    """Analysis backed by a :class:`~aieng.advanced_rag.pipeline.rewrite.Rewriter`.

    This is how ``TemplateRewriter`` and ``LLMRewriter`` satisfy
    :class:`QueryAnalysis` without either of them learning about plans. The
    rewriter's candidates become the plan's steps, up to ``max_steps``; a
    rewriter that finds nothing to change leaves the question as asked, which is
    the fallback the spine has always applied.

    ``max_steps`` defaults to 1 because raising it moves measured numbers.
    :func:`~aieng.advanced_rag.pipeline.grading.pooled_score` pools by maximum,
    so every extra candidate can only raise an attempt's score, and a score
    that rises crosses the sufficient boundary sooner. That changes the
    abstention rate, which is the number the sweep reports. Reading a candidate
    count off a configuration default rather than off the row would make two
    runs incomparable, so the default retrieves exactly what it always did and
    a wider plan is asked for in the open.

    Widening the plan widens the grader's pool and gives the reader nothing.
    The spine concatenates each step's ranking in step-then-rank order and
    :func:`~aieng.advanced_rag.pipeline.generate.generate` takes the first
    ``evidence_budget`` entries of that list, so with three candidates and a
    budget of five the generator sees the first candidate's top five and none
    of the rest. The asymmetry is the shipped behaviour and not an oversight:
    interleaving the rankings, or fusing them by score, is a retrieval-side
    decision with its own measurement, and it is deliberately not a seam here.
    Fuse before the pipeline, in the ``retrieve`` callable, where
    :mod:`aieng.advanced_rag.retrieval` already scores fusion.

    Parameters
    ----------
    rewriter : Rewriter
        The stage to run before retrieval.
    max_steps : int, optional
        How many of the rewriter's candidates to retrieve. 1 is the shipped
        behaviour.

    Raises
    ------
    ValueError
        If ``max_steps`` is below one, which would plan no retrieval at all.
    """

    def __init__(self, rewriter: Rewriter, max_steps: int = 1) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.rewriter = rewriter
        self.name = rewriter.name
        self.max_steps = max_steps

    def analyse(self, question: str, scope: str | None = None) -> QueryPlan:
        """Return one step per rewrite candidate, up to ``max_steps``.

        Parameters
        ----------
        question : str
            The question as asked.
        scope : str or None, optional
            Carried onto every step; a rewrite does not change which documents
            the question belongs to.

        Returns
        -------
        QueryPlan
            A plan over the candidates in the order the rewriter produced them,
            or a one-step plan over the question as asked when there are none.
            Either way the plan carries the rewriter's outcome.
        """
        rewritten = self.rewriter.rewrite(question)
        candidates = rewritten.questions(include_original=False)[: self.max_steps]
        if not candidates:
            return QueryPlan(
                question=question,
                steps=(RetrievalStep(question=question, scope=scope),),
                rewrite_outcome=rewritten.outcome,
            )
        return QueryPlan(
            question=question,
            steps=tuple(
                RetrievalStep(
                    question=candidate,
                    scope=scope,
                    strategy=self.rewriter.name,
                    parent=question,
                )
                for candidate in candidates
            ),
            rewrite_outcome=rewritten.outcome,
        )


class DecompositionAnalysis:
    """Analysis that retrieves one step per sub-question.

    Multi-step by construction, which is what separates it from
    :class:`RewriterAnalysis`: a rewriter proposes several phrasings of one
    question and the plan keeps as many as its bound allows, while a
    decomposition's sub-questions *are* the plan. A splitter composes with this
    class rather than growing a second analysis stage of its own.

    The generator is still asked ``plan.question``, the question as asked, so a
    sub-question is a retrieval device and never the thing answered. The single
    step a trace row records comes from :func:`merge_steps`, which names every
    sub-question retrieved, so the record does not pass one sub-question off as
    the attempt.

    Both facts in :class:`RewriterAnalysis` apply here and bite harder.
    Pooling by maximum means more sub-questions can only raise the grade, and
    the generation budget is spent on the first sub-question's ranking, so four
    sub-questions under a budget of five leave the reader with one
    sub-question's evidence. Decide what the reader should see in the
    ``retrieve`` callable, not here.

    Parameters
    ----------
    decompose : callable
        Takes the question as asked and returns its sub-questions in retrieval
        order. Returning nothing leaves a one-step plan over the question as
        asked, which is how a splitter declines a question it cannot split.
    name : str, optional
        Strategy label recorded on every step and attempt.
    max_steps : int or None, optional
        Cap on the sub-questions retrieved. ``None`` retrieves all of them.
        Unlike the rewriter's bound this has no shipped number to preserve,
        because nothing measured has run through this stage.

    Raises
    ------
    ValueError
        If ``max_steps`` is below one, which would plan no retrieval at all.
    """

    def __init__(
        self,
        decompose: Callable[[str], Sequence[str]],
        name: str = "decomposition",
        max_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.decompose = decompose
        self.name = name
        self.max_steps = max_steps

    def analyse(self, question: str, scope: str | None = None) -> QueryPlan:
        """Return one step per sub-question, up to ``max_steps``.

        Parameters
        ----------
        question : str
            The question as asked.
        scope : str or None, optional
            Carried onto every step. A splitter that needs a scope per
            sub-question builds the plan itself; this one holds the caller's.

        Returns
        -------
        QueryPlan
            A plan over the sub-questions, de-duplicated in order, or a one-step
            plan over the question as asked when the splitter returned none.
        """
        parts: dict[str, None] = {}
        for part in self.decompose(question):
            if part:
                parts.setdefault(part, None)
        sub_questions = list(parts)[: self.max_steps]
        if not sub_questions:
            return QueryPlan(
                question=question,
                steps=(RetrievalStep(question=question, scope=scope),),
            )
        return QueryPlan(
            question=question,
            steps=tuple(
                RetrievalStep(
                    question=sub_question,
                    scope=scope,
                    strategy=self.name,
                    parent=question,
                )
                for sub_question in sub_questions
            ),
        )


def step_bound(analysis: QueryAnalysis) -> int | None:
    """Return how many retrievals one attempt of ``analysis`` may ask for.

    The corrective retry reads this so it honours the bound the first attempt
    honoured. A retry capped at one candidate under an analysis that planned
    three would retrieve less than the attempt it replaces, and the trace would
    show that only as a shorter evidence list.

    Parameters
    ----------
    analysis : QueryAnalysis
        The pre-retrieval stage. One without a ``max_steps`` is the shipped
        single-step kind.

    Returns
    -------
    int or None
        The bound, or ``None`` when the stage sets no bound.
    """
    bound = getattr(analysis, "max_steps", 1)
    return None if bound is None else int(bound)


def merge_steps(steps: Sequence[RetrievalStep]) -> RetrievalStep:
    """Return the one step a trace records for an attempt that retrieved several.

    An attempt is one row of the trace and carries one question, while a
    decomposition retrieves several. Naming all of them keeps the record honest
    about what was retrieved instead of picking one of the sub-questions.

    Parameters
    ----------
    steps : sequence of RetrievalStep
        The steps one attempt retrieved with, at least one.

    Returns
    -------
    RetrievalStep
        ``steps[0]`` when there is one, otherwise a step whose question lists
        every question retrieved, carrying the first step's scope and parent.

    Raises
    ------
    ValueError
        If ``steps`` is empty, which means the attempt retrieved nothing.
    """
    if not steps:
        raise ValueError("an attempt needs at least one step to retrieve with")
    if len(steps) == 1:
        return steps[0]
    return RetrievalStep(
        question=" | ".join(step.question for step in steps),
        scope=steps[0].scope,
        strategy=steps[0].strategy,
        parent=steps[0].parent,
    )


@dataclass(frozen=True)
class LoopState:
    """Everything a policy may read, buildable from live work or a recording.

    Parameters
    ----------
    plan : QueryPlan
        The plan analysis produced. ``plan.question`` is the question as asked.
    step : RetrievalStep
        The step this attempt retrieved with, merged by :func:`merge_steps`
        when the attempt retrieved several.
    score : float
        The pooled grade for this attempt.
    had_evidence : bool
        Whether retrieval returned anything. Distinct from ``score == 0.0``: an
        empty result set is absent evidence, which the shipped ``insufficient``
        of 0.0 cannot express as a score. A replay reads it off the recorded
        identifiers, which is faithful because ``to_runs`` returns a non-empty
        list for any non-empty input.
    attempts_left : int
        Retrievals still available after this one.
    rewrite_available : bool
        Whether there is a next step to retry with.
    evidence : tuple of Evidence, optional
        The passages pooled for this attempt, across every step it retrieved.
        Empty on a replay, which holds identifiers rather than passages, so a
        policy that reads this one is not replayable.
    grade : EvidenceGrade or None, optional
        The full grade when one is in hand, ``None`` on a replay. The grade's
        own verdict is advisory; the policy owns the verdict that drives the
        action.
    notes : tuple of str, optional
        Reasoning carried forward from earlier moves, oldest first. This is
        what an interleaved retrieve-and-reason policy reads to choose its next
        retrieval.
    """

    plan: QueryPlan
    step: RetrievalStep
    score: float
    had_evidence: bool
    attempts_left: int
    rewrite_available: bool
    evidence: tuple[Evidence, ...] = ()
    grade: EvidenceGrade | None = None
    notes: tuple[str, ...] = ()

    @property
    def question(self) -> str:
        """Return the question as asked, which the plan carries."""
        return self.plan.question


@dataclass(frozen=True)
class LoopDecision:
    """What the policy decided to do next.

    Parameters
    ----------
    action : Action
        Answer, retrieve again, or abstain.
    step : RetrievalStep or None, optional
        What to retrieve next. ``None`` on :attr:`Action.REWRITE` asks the loop
        to use its corrective rewriter, which is the shipped behaviour.
    note : str, optional
        A reasoning sentence to carry into the next state.
    """

    action: Action
    step: RetrievalStep | None = None
    note: str = ""


class LoopPolicy(Protocol):
    """Anything that decides what one attempt's outcome means.

    A policy with ``replayable`` true is a pure function of :class:`LoopState`
    and makes no model calls, which is what lets
    :mod:`aieng.advanced_rag.evaluation.sweep` drive it over recorded traces at
    any thresholds. A policy that calls a model, or that reads
    ``state.evidence``, must set it false, and the sweep refuses such a policy
    rather than reporting an operating point it never measured.
    """

    name: str
    replayable: bool

    def next_move(self, state: LoopState) -> LoopDecision:
        """Return the next move for one attempt's outcome.

        Parameters
        ----------
        state : LoopState
            What the attempt found and what the loop has left.

        Returns
        -------
        LoopDecision
            The action to take, with the step to retrieve next when the action
            is a retry.
        """
        ...


class CorrectivePolicy:
    """The corrective loop: answer, retry on a weak grade, or abstain.

    This reproduces the decisions
    :meth:`~aieng.advanced_rag.pipeline.spine.AdvancedRagSpine.answer` made
    when they were inlined, and it agrees with them on one condition: the
    verdict is ``thresholds.verdict`` applied to the pooled maximum, with an
    empty result set short-circuited to :attr:`Verdict.INSUFFICIENT`. Both
    shipped graders are that shape. A grader that returned a label independent
    of the score, or pooled by the mean where the trace keeps the maximum,
    breaks this policy, and it already breaks the sweep's replay the same way,
    which reads the verdict off the recorded pooled score.

    The agreement was measured at all 160 points of the sweep's threshold grid
    and at the 32 empty-evidence points beside them, with no disagreement, and
    the grid test drives both loops through this policy on every point it walks.

    Parameters
    ----------
    thresholds : GradingThresholds or None, optional
        Where the pooled score turns into a verdict. This is the loop's own
        setting, taken from ``SpineConfig.thresholds`` on the spine and from
        the swept pair on a replay. ``None`` applies the thresholds the grade
        was made with, and the shipped defaults when there is no grade.
    """

    name = "corrective"
    replayable = True

    def __init__(self, thresholds: GradingThresholds | None = None) -> None:
        self.thresholds = thresholds

    def verdict_for(self, state: LoopState) -> Verdict:
        """Return the verdict this policy reads from one attempt's outcome.

        Parameters
        ----------
        state : LoopState
            What the attempt found.

        Returns
        -------
        Verdict
            :attr:`Verdict.INSUFFICIENT` when retrieval returned nothing,
            otherwise the thresholds applied to the pooled score.
        """
        if not state.had_evidence:
            return Verdict.INSUFFICIENT
        return self._thresholds(state).verdict(state.score)

    def next_move(self, state: LoopState) -> LoopDecision:
        """Return the corrective loop's move for one attempt's outcome.

        A retry this policy asks for can still turn out to be impossible,
        because the corrective rewriter runs after the decision and may return
        nothing. The loop driver settles that case by asking again with no
        attempts left, and it does so after charging the rewriter's call, since
        the call is paid for whether or not a rewrite comes back.

        Parameters
        ----------
        state : LoopState
            What the attempt found and what the loop has left.

        Returns
        -------
        LoopDecision
            The action :func:`~aieng.advanced_rag.pipeline.grading.decide`
            gives for the verdict at the attempts remaining.
        """
        return LoopDecision(decide(self.verdict_for(state), state.attempts_left))

    def _thresholds(self, state: LoopState) -> GradingThresholds:
        """Return the thresholds to read this state's score against."""
        if self.thresholds is not None:
            return self.thresholds
        if state.grade is not None:
            return state.grade.thresholds
        return GradingThresholds()


class SingleShotPolicy:
    """No corrective loop: grade for the record, then answer.

    This is what ``max_attempts=1`` already does, which the spine's
    configuration calls the baseline to measure the loop against. The generator
    keeps its own abstention path, so a question with nothing behind it is
    still declined; what this policy does not do is decline it before
    generating.
    """

    name = "single-shot"
    replayable = True

    def next_move(self, state: LoopState) -> LoopDecision:
        """Return :attr:`Action.ANSWER` whatever the grade said.

        Parameters
        ----------
        state : LoopState
            Read for nothing, and accepted so the protocol holds.

        Returns
        -------
        LoopDecision
            Always an answer.
        """
        return LoopDecision(Action.ANSWER)


class PromptSelector(Protocol):
    """Anything that chooses the answer prompt for one result set."""

    name: str

    def get_template(self, plan: QueryPlan, evidence: Sequence[Evidence]) -> str:
        """Return the template to render the answer prompt with.

        Parameters
        ----------
        plan : QueryPlan
            The plan the evidence was retrieved under.
        evidence : sequence of Evidence
            The passages the generator is about to be shown.

        Returns
        -------
        str
            A template carrying ``{query}`` and ``{sources}``.
        """
        ...


class FixedPromptSelector:
    """One template for every answer, which is the shipped behaviour.

    Parameters
    ----------
    template : str, optional
        The template to render every answer prompt with.
    """

    name = "fixed"

    def __init__(self, template: str = GENERATION_TEMPLATE) -> None:
        self.template = template

    def get_template(self, plan: QueryPlan, evidence: Sequence[Evidence]) -> str:
        """Return the one template, whatever was retrieved.

        Parameters
        ----------
        plan : QueryPlan
            Read for nothing, and accepted so the protocol holds.
        evidence : sequence of Evidence
            Read for nothing, and accepted so the protocol holds.

        Returns
        -------
        str
            The template this source was built with.
        """
        return self.template

    #: Deprecated alias for :meth:`get_template`. Deprecated in 0.1.1; use
    #: ``get_template``, because what this returns is a template and what the
    #: class is is a selector.
    template_for = get_template


class RetrieverPromptSelector:
    """A retrieval method's own answer prompt, falling back to the default.

    Two answer formats already ship, one asking for a JSON object and one for
    prose with ``[1]`` markers, and which one a method wants is a property of
    the method rather than of the question.
    :attr:`~aieng.advanced_rag.graphrag.evidence.Evidence.retriever` is the
    only method label a passage carries, so that is the key.

    Parameters
    ----------
    default : str, optional
        The project's template, used when no single method owns the result set.
    by_retriever : mapping or None, optional
        Retriever name to the template that method answers with. A name absent
        here falls back to ``default``.
    """

    name = "by-method"

    def __init__(
        self,
        default: str = GENERATION_TEMPLATE,
        by_retriever: Mapping[str, str] | None = None,
    ) -> None:
        self.default = default
        self.by_retriever = dict(by_retriever or {})

    def get_template(self, plan: QueryPlan, evidence: Sequence[Evidence]) -> str:
        """Return the template the result set's own method answers with.

        Parameters
        ----------
        plan : QueryPlan
            Read for nothing, and accepted so the protocol holds.
        evidence : sequence of Evidence
            The passages the generator is about to be shown.

        Returns
        -------
        str
            The owning method's template, or ``default`` when the passages came
            from more than one method or from none that named itself. A
            decomposition pools several retrievals, so that fallback is the
            ordinary case rather than an edge one.
        """
        names = {item.retriever for item in evidence if item.retriever}
        if len(names) != 1:
            return self.default
        return self.by_retriever.get(names.pop(), self.default)

    #: Deprecated alias for :meth:`get_template`. Deprecated in 0.1.1; use
    #: ``get_template``, because what this returns is a template and what the
    #: class is is a selector.
    template_for = get_template


class Answerer(Protocol):
    """Anything that turns a question into a grounded answer and its trace."""

    def answer(
        self, query_id: str, question: str, scope: str | None = None
    ) -> SpineResult:
        """Answer one question, or abstain, recording every decision.

        Parameters
        ----------
        query_id : str
            Identifier carried through to the answer.
        question : str
            The question as asked.
        scope : str or None, optional
            Corpus filter for retrieval.

        Returns
        -------
        SpineResult
            The answer and its trace.
        """
        ...


__all__ = [
    "Answerer",
    "CorrectivePolicy",
    "DecompositionAnalysis",
    "FixedPromptSelector",
    "LoopDecision",
    "LoopPolicy",
    "LoopState",
    "PassthroughAnalysis",
    "PromptSelector",
    "QueryAnalysis",
    "QueryPlan",
    "RetrievalStep",
    "RetrieverPromptSelector",
    "RewriterAnalysis",
    "SingleShotPolicy",
    "merge_steps",
    "step_bound",
]


#: Deprecated aliases. ``LoopDecision`` and ``RetrievalStep`` say what they are
#: where ``Move`` competed with ``Action`` and ``Step`` was too generic to
#: export, and the three selectors are named for what they choose.
#:
#: Deprecated in 0.1.1, removed in 0.2.0. They are kept for one minor version
#: because 0.1.1 is published. They stay importable and stay out of
#: ``__all__``, so ``import *`` never binds a retired name, which is the policy
#: ``tests/aieng/advanced_rag/test_exports.py`` encodes.
Move = LoopDecision
Step = RetrievalStep
PromptSource = PromptSelector
FixedPrompt = FixedPromptSelector
MethodPrompt = RetrieverPromptSelector
NoAnalysis = PassthroughAnalysis
