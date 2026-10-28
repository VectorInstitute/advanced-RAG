"""Grading retrieved evidence, and the decision the grade drives.

This is the corrective half of CRAG-style generation. Retrieval always returns
its top-k, whether or not the corpus holds an answer, so something has to look
at what came back and say whether it is worth generating from. That judgment is
the whole mechanism, and it is only teachable if it is visible: the grade is an
object with a score, a verdict, and a reason, and the action follows from a
stated threshold rather than from an ``if`` buried in a loop.

Three verdicts and three actions, because two of each is not enough. Evidence
that is plainly on topic but does not state the answer is neither a green light
nor an abstention; it is the case where rewriting the query is worth a second
retrieval. :func:`decide` maps a verdict and the attempts remaining onto one of
:class:`Action`, and that table is the pipeline's control flow in six lines.

Prior art: Corrective RAG (Yan et al., 2024, arXiv:2401.15884) trains a retrieval
evaluator that returns correct, ambiguous, or incorrect and routes each to a
different action. Self-RAG (Asai et al., 2023, arXiv:2310.11511) has the
generator emit reflection tokens that grade its own passages. The LangGraph
corrective-RAG tutorial wires the same grade-then-act loop as graph nodes. This
module keeps the grade-then-act shape and replaces the trained evaluator with a
prompted grader or a heuristic, so it runs on any model and replays from cache.

Two graders ship here. :class:`HeuristicGrader` needs no model and runs in
tests and offline sweeps. :class:`LLMGrader` is the one to report numbers from,
and it costs one call per passage graded, which ``max_chunks`` bounds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..graphrag.evidence import Evidence


class Verdict(StrEnum):
    """How well the retrieved evidence supports answering the question."""

    SUFFICIENT = "sufficient"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


class Action(StrEnum):
    """What the pipeline does next about a verdict."""

    ANSWER = "answer"
    REWRITE = "rewrite"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class GradingThresholds:
    """The two numbers the corrective loop turns.

    Measured on ContractNLI, and the sweep behind these values is in
    ``implementations/regulatory_policy_qa/THRESHOLDS.md``. Raising
    ``insufficient`` buys abstention recall on the null cases and costs answers
    on the answerable ones, and on that corpus the exchange rate is poor for
    both graders: against 0.0, the previous default of 0.3 caught 20 more of
    322 null questions and refused 43 more of 358 answerable ones. It ships at
    0.0, where the grader leaves abstention to the generator's own "not stated"
    path and :attr:`Verdict.INSUFFICIENT` is reached only through
    :func:`no_evidence_grade`, which fires when retrieval returned nothing.

    ``sufficient`` was flat from 0.2 to 1.0 on the same sweep, within its
    sampling error, so it stays at 0.6 for want of a reason to move it.

    A deployment that weighs a false answer above roughly twice a refused
    answerable question wants ``insufficient`` raised again. The memo carries
    the curve, and ``evaluation.metrics.score_null_detection`` with
    ``evaluation.sweep`` prices any other point on it.

    Parameters
    ----------
    sufficient : float, optional
        At or above this pooled score the evidence is treated as answerable.
    insufficient : float, optional
        Below this the evidence is treated as absent. At the default of 0.0 no
        score is below it, so the grader itself never returns
        :attr:`Verdict.INSUFFICIENT`.

    Raises
    ------
    ValueError
        If the thresholds are out of range or inverted.
    """

    sufficient: float = 0.6
    insufficient: float = 0.0

    def __post_init__(self) -> None:
        """Reject threshold pairs that cannot produce three verdicts."""
        if not 0.0 <= self.insufficient <= self.sufficient <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= insufficient <= sufficient <= 1"
            )

    def verdict(self, score: float) -> Verdict:
        """Return the verdict for a pooled score."""
        if score >= self.sufficient:
            return Verdict.SUFFICIENT
        if score < self.insufficient:
            return Verdict.INSUFFICIENT
        return Verdict.AMBIGUOUS


@dataclass(frozen=True)
class ChunkGrade:
    """One passage's support for the question.

    Parameters
    ----------
    chunk_id : str
        Identifier of the graded passage.
    score : float
        Support in ``[0, 1]``.
    reason : str, optional
        One-line justification, kept for auditing and fed to the rewriter.
    """

    chunk_id: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class EvidenceGrade:
    """The grade for one attempt's evidence, with the decision inputs attached.

    Parameters
    ----------
    query_id : str
        Identifier of the question graded.
    question : str
        The question as it was sent to the retriever, which on a retry is the
        rewrite rather than the original.
    verdict : Verdict
        The threshold's reading of ``score``.
    score : float
        The pooled per-passage score.
    reason : str
        Why the evidence scored as it did. This text is what the rewriter is
        given on a retry, so it must name what is missing.
    chunk_grades : tuple of ChunkGrade, optional
        Per-passage scores, in the order the passages were ranked.
    thresholds : GradingThresholds, optional
        The thresholds this verdict came from, recorded so a report is
        self-describing after a sweep changes them.
    """

    query_id: str
    question: str
    verdict: Verdict
    score: float
    reason: str
    chunk_grades: tuple[ChunkGrade, ...] = ()
    thresholds: GradingThresholds = field(default_factory=GradingThresholds)

    @property
    def supporting(self) -> tuple[ChunkGrade, ...]:
        """Return the passages that scored at or above the sufficient threshold."""
        return tuple(
            grade
            for grade in self.chunk_grades
            if grade.score >= self.thresholds.sufficient
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the grade."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "verdict": str(self.verdict),
            "score": round(self.score, 4),
            "reason": self.reason,
            "sufficient": self.thresholds.sufficient,
            "insufficient": self.thresholds.insufficient,
            "chunk_grades": [
                {
                    "chunk_id": grade.chunk_id,
                    "score": round(grade.score, 4),
                    "reason": grade.reason,
                }
                for grade in self.chunk_grades
            ],
        }


def decide(verdict: Verdict, attempts_left: int) -> Action:
    """Map a verdict and the retries remaining onto the next action.

    The whole control flow of the corrective loop is this table:

    ==============  ================  =========
    Verdict         Attempts left     Action
    ==============  ================  =========
    sufficient      any               answer
    ambiguous       one or more       rewrite
    ambiguous       none              answer
    insufficient    one or more       rewrite
    insufficient    none              abstain
    ==============  ================  =========

    Ambiguous evidence on the last attempt is answered rather than abstained on,
    because the generator has its own abstention path and can see the passages
    the grader only scored. Insufficient evidence on the last attempt abstains
    without a generation call, which is both cheaper and the safer default on a
    corpus where most questions have no answer.

    Parameters
    ----------
    verdict : Verdict
        The grader's reading of the evidence.
    attempts_left : int
        Retrieval attempts still available after this one.

    Returns
    -------
    Action
        What the pipeline does next.
    """
    if verdict is Verdict.SUFFICIENT:
        return Action.ANSWER
    if attempts_left > 0:
        return Action.REWRITE
    return Action.ANSWER if verdict is Verdict.AMBIGUOUS else Action.ABSTAIN


# A model asked for JSON returns it wrapped in prose or a code fence often
# enough that failing the call would throw away completions already paid for.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def first_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object in a model response, or ``None``.

    Parameters
    ----------
    raw : str
        The raw response text. ``None``-safe: an empty string yields ``None``.

    Returns
    -------
    dict or None
        The parsed object, or ``None`` when the response held no parsable one.
    """
    match = _JSON_RE.search(raw or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def pooled_score(grades: Sequence[ChunkGrade]) -> float:
    """Pool per-passage scores into the number the verdict is read from.

    The maximum, not the mean. One passage that states the answer is enough to
    answer from, and averaging it against the four filler passages that came
    back with it would score a successful retrieval as a failure.
    """
    return max((grade.score for grade in grades), default=0.0)


def no_evidence_grade(
    query_id: str, question: str, thresholds: GradingThresholds
) -> EvidenceGrade:
    """Return the grade for an empty result set, with no model calls made."""
    return EvidenceGrade(
        query_id=query_id,
        question=question,
        verdict=Verdict.INSUFFICIENT,
        score=0.0,
        reason="retrieval returned no passages",
        thresholds=thresholds,
    )


class Grader(Protocol):
    """Anything that scores retrieved evidence against a question."""

    name: str

    def grade(
        self, question: str, evidence: Sequence[Evidence], query_id: str = ""
    ) -> EvidenceGrade:
        """Score the evidence and return an inspectable grade."""
        ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Enough to strip the function words out of a question. A full stop-word list
# would need a dependency and would not change which passage wins.
_STOPWORD_TEXT = """
a an and any are as at be by do does for from has have how in is it its of on
or that the there this to was were what when where which who why will with
"""

STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _stem(token: str) -> str:
    """Strip the common English suffixes, crudely.

    Deliberately not a real stemmer. "governing law" has to match "governed by
    the laws of", and that needs suffix stripping and nothing more; a proper
    stemmer would be a dependency for no measurable gain at this stage.
    """
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> set[str]:
    """Return the stemmed content words of a piece of text."""
    return {
        _stem(match.group(0).lower())
        for match in _TOKEN_RE.finditer(text)
        if match.group(0).lower() not in STOPWORDS and len(match.group(0)) > 2
    }


class HeuristicGrader:
    """Scores a passage by how much of the question's vocabulary it covers.

    Model-free and deterministic, which is what makes it the grader for tests
    and for offline threshold sweeps. It is a floor: term coverage cannot tell a
    passage that states the answer from one that merely repeats the question's
    nouns, so numbers reported from it understate what the pipeline can do.

    Parameters
    ----------
    thresholds : GradingThresholds, optional
        The verdict boundaries.
    name : str, optional
        Label recorded on traces.
    """

    def __init__(
        self, thresholds: GradingThresholds | None = None, name: str = "heuristic"
    ) -> None:
        self.thresholds = thresholds or GradingThresholds()
        self.name = name

    def grade(
        self, question: str, evidence: Sequence[Evidence], query_id: str = ""
    ) -> EvidenceGrade:
        """Score each passage by term coverage and pool the results."""
        if not evidence:
            return no_evidence_grade(query_id, question, self.thresholds)

        wanted = content_terms(question)
        grades: list[ChunkGrade] = []
        for item in evidence:
            found = wanted & content_terms(item.text)
            score = len(found) / len(wanted) if wanted else 0.0
            grades.append(
                ChunkGrade(
                    chunk_id=item.chunk_id or item.doc_id,
                    score=score,
                    reason=f"covers {len(found)} of {len(wanted)} question terms",
                )
            )
        score = pooled_score(grades)
        best = max(grades, key=lambda grade: grade.score)
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=self.thresholds.verdict(score),
            score=score,
            reason=f"best passage {best.chunk_id} {best.reason}",
            chunk_grades=tuple(grades),
            thresholds=self.thresholds,
        )


GRADE_PROMPT = """Rate how well one passage answers one question.

Question:
{question}

Passage:
{passage}

Score from 0.0 to 1.0:
  1.0  the passage states the answer.
  0.5  the passage is on the right topic but does not state the answer.
  0.0  the passage is unrelated to the question.

Judge this passage alone. Do not assume what other passages might contain, and
do not use anything you know outside the passage.

Return one JSON object and nothing else:
{{"score": <number>, "reason": "<at most 20 words, naming what is missing>"}}
"""


def parse_chunk_grade(chunk_id: str, raw: str) -> ChunkGrade:
    """Parse one grading response, treating anything unreadable as zero support.

    An unparsable grade is scored 0.0 rather than raised on. The failure it
    usually signals is a truncated completion, and a truncated completion is not
    evidence that the passage supports the question.

    Parameters
    ----------
    chunk_id : str
        Identifier of the graded passage.
    raw : str
        The grader's raw response.

    Returns
    -------
    ChunkGrade
        The parsed grade, clamped to ``[0, 1]``.
    """
    payload = first_json_object(raw)
    if payload is None:
        return ChunkGrade(chunk_id=chunk_id, score=0.0, reason="unreadable grade")
    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        return ChunkGrade(chunk_id=chunk_id, score=0.0, reason="non-numeric score")
    return ChunkGrade(
        chunk_id=chunk_id,
        score=min(max(score, 0.0), 1.0),
        reason=str(payload.get("reason", ""))[:200],
    )


class LLMGrader:
    """Asks a model how well each passage answers the question.

    One call per passage. Grading them together would be cheaper and is how this
    was first written, but the question-generator verifier in this project
    already showed what batching does to a judge: asked about several items at
    once, one bad item dragged the verdicts on the rest down with it, and its
    acceptance rate fell to 5%. ``max_chunks`` is the cost lever instead.

    Parameters
    ----------
    complete : callable
        Takes a prompt, returns text. Supply a cached, metered client.
    thresholds : GradingThresholds, optional
        The verdict boundaries.
    max_chunks : int, optional
        Passages graded, taken from the top of the ranking. The rest are
        recorded with a zero score and no call made, so the pooled score can
        only be driven by a passage that was actually looked at.
    max_chars : int or None, optional
        Per-passage truncation in the prompt.
    prompt : str, optional
        Template with ``{question}`` and ``{passage}``.
    name : str, optional
        Label recorded on traces.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        thresholds: GradingThresholds | None = None,
        max_chunks: int = 5,
        max_chars: int | None = 1200,
        prompt: str = GRADE_PROMPT,
        name: str = "llm",
    ) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        self.complete = complete
        self.thresholds = thresholds or GradingThresholds()
        self.max_chunks = max_chunks
        self.max_chars = max_chars
        self.prompt = prompt
        self.name = name

    def grade(
        self, question: str, evidence: Sequence[Evidence], query_id: str = ""
    ) -> EvidenceGrade:
        """Grade the top passages and pool them into one verdict."""
        if not evidence:
            return no_evidence_grade(query_id, question, self.thresholds)

        grades: list[ChunkGrade] = []
        for position, item in enumerate(evidence):
            chunk_id = item.chunk_id or item.doc_id
            if position >= self.max_chunks:
                grades.append(
                    ChunkGrade(chunk_id=chunk_id, score=0.0, reason="not graded")
                )
                continue
            passage = (
                item.text if self.max_chars is None else item.text[: self.max_chars]
            )
            raw = self.complete(self.prompt.format(question=question, passage=passage))
            grades.append(parse_chunk_grade(chunk_id, raw))

        score = pooled_score(grades)
        best = max(grades, key=lambda grade: grade.score)
        reason = best.reason or "no passage scored above zero"
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=self.thresholds.verdict(score),
            score=score,
            reason=reason,
            chunk_grades=tuple(grades),
            thresholds=self.thresholds,
        )
