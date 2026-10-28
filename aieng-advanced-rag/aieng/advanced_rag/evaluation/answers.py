"""Answer scoring for a fixed evidence budget.

This is the generation half of the shared harness. The retrieval half scores
ranked identifiers against judgments derived from a dataset, so it needs no
model. Answers cannot be scored that way, so a model judges them, and the design
here is shaped by two things that went wrong earlier in this project.

**Fields are judged independently.** An earlier verifier in the question
generator collapsed several questions into one and its acceptance rate fell to
5%: asked whether a question was "good", the model let one bad property poison
its verdict on the others. The judge here returns four fields and the prompt
asks for each one separately.

**No embeddings.** RAGAS ``answer_correctness`` is a weighted sum of an
LLM-judged factual overlap and an embedding similarity, and its default
embedding model is a paid API model. This judge uses no embeddings at all, so
it needs no embedding configuration to be correct. RAGAS is still usable for
standard-RAG evaluation, and it has to be constructed with an embedding model
you have chosen deliberately.

The completion function is a parameter, so this module depends on no client and
can be driven by a cached, metered one. :func:`judge_answers` judges many answers
at once, and sends every prompt as one batch when the completion function has a
``batch`` method, as :meth:`aieng.advanced_rag.batch.ProxyBatch.completer` does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

from ..graphrag.evidence import Evidence
from ..graphrag.synthesis import build_prompt


JUDGE_PROMPT = """You are grading one answer produced by a retrieval system.

Judge each field independently. A failure on one field is not a reason to fail
another. Judge only what each field asks about.

Question:
{question}

Reference answer (the ground truth):
{reference}

Evidence the system was given:
{evidence}

The system's answer:
{answer}

Return one JSON object, no other text, with exactly these keys:

  "correct": true if the answer states the same thing as the reference answer.
    Wording, length, and extra correct detail do not matter. Missing a required
    part of the reference, or contradicting it, is false.
  "grounded": true if every factual claim in the answer is supported by the
    evidence above. Judge this against the evidence only, ignoring whether the
    answer is correct.
  "abstained": true if the answer declines to answer or says the information is
    not available, rather than attempting an answer.
  "reason": one sentence, at most 30 words, explaining the "correct" verdict.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class AnswerJudgement:
    """One judged answer.

    Parameters
    ----------
    query_id : str
        Identifier of the question judged.
    correct : bool
        Whether the answer states what the reference answer states.
    grounded : bool
        Whether every claim is supported by the evidence supplied.
    abstained : bool
        Whether the answer declined rather than attempted.
    reason : str
        One-sentence justification of ``correct``, kept for auditing.
    """

    query_id: str
    correct: bool
    grounded: bool
    abstained: bool
    reason: str = ""


@dataclass(frozen=True)
class AnswerReport:
    """Aggregate answer quality at a fixed evidence budget.

    Parameters
    ----------
    method : str
        Name of the retrieval method the answers came from.
    evidence_budget : int
        Number of passages every method was given. Reported because a
        comparison of answers is only fair when this is held constant.
    answered : int
        Questions where the system attempted an answer.
    judgements : list of AnswerJudgement
        The per-question verdicts these rates are computed from.
    """

    method: str
    evidence_budget: int
    answered: int
    judgements: list[AnswerJudgement] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Return the number of questions judged."""
        return len(self.judgements)

    @property
    def correctness(self) -> float:
        """Return the share of questions answered correctly."""
        return self._rate(lambda j: j.correct)

    @property
    def grounding(self) -> float:
        """Return the share of answers fully supported by their evidence."""
        return self._rate(lambda j: j.grounded)

    @property
    def abstention(self) -> float:
        """Return the share of questions the system declined to answer."""
        return self._rate(lambda j: j.abstained)

    @property
    def hallucination(self) -> float:
        """Return the share of answers that attempted and were ungrounded.

        This is the number that matters for a grounded system: an abstention is
        not a hallucination, so abstentions are excluded rather than counted as
        successes.
        """
        attempted = [j for j in self.judgements if not j.abstained]
        if not attempted:
            return 0.0
        return sum(1 for j in attempted if not j.grounded) / len(attempted)

    def _rate(self, predicate: Callable[[AnswerJudgement], bool]) -> float:
        if not self.judgements:
            return 0.0
        return sum(1 for j in self.judgements if predicate(j)) / len(self.judgements)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary including per-question verdicts."""
        return {
            "method": self.method,
            "evidence_budget": self.evidence_budget,
            "questions": self.total,
            "answered": self.answered,
            "correctness": round(self.correctness, 4),
            "grounding": round(self.grounding, 4),
            "abstention": round(self.abstention, 4),
            "hallucination": round(self.hallucination, 4),
            "judgements": [asdict(j) for j in self.judgements],
        }

    def table_row(self, label: str | None = None) -> str:
        """Render one Markdown table row, matching the retrieval harness style."""
        return (
            f"| {label or self.method} | {self.correctness:.4f} | "
            f"{self.grounding:.4f} | {self.hallucination:.4f} | "
            f"{self.abstention:.4f} | {self.total} |"
        )


TABLE_HEADER = (
    "| Method | Correctness | Grounding | Hallucination | Abstention | Questions |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: |"
)


def parse_judgement(query_id: str, raw: str) -> AnswerJudgement:
    """Parse a judge response, tolerating text around the JSON object.

    A model asked for JSON sometimes wraps it in prose or a code fence. Failing
    the whole run on that would lose the judgements already paid for, so the
    first JSON object in the response is used and a missing field is read as
    false rather than raising.

    Parameters
    ----------
    query_id : str
        Identifier to attach to the parsed verdict.
    raw : str
        The judge's raw response text.

    Returns
    -------
    AnswerJudgement
        The parsed verdict.

    Raises
    ------
    ValueError
        If the response contains no JSON object at all.
    """
    match = _JSON_RE.search(raw or "")
    if match is None:
        raise ValueError(f"no JSON object in judge response for {query_id}: {raw!r}")
    payload = json.loads(match.group(0))
    return AnswerJudgement(
        query_id=query_id,
        correct=bool(payload.get("correct", False)),
        grounded=bool(payload.get("grounded", False)),
        abstained=bool(payload.get("abstained", False)),
        reason=str(payload.get("reason", ""))[:300],
    )


def judge_answer(  # noqa: PLR0917 - preserve the shared harness's positional API
    query_id: str,
    question: str,
    answer_text: str,
    reference: str,
    evidence: Sequence[Evidence],
    complete: Callable[[str], str],
    max_chars: int | None = 1200,
    *,
    template: str = JUDGE_PROMPT,
) -> AnswerJudgement:
    """Judge one answer on correctness, grounding, and abstention.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    question : str
        The question asked.
    answer_text : str
        The system's answer.
    reference : str
        Ground-truth answer text.
    evidence : sequence of Evidence
        The passages the system was given, used to judge grounding.
    complete : callable
        Takes a prompt, returns text. Supply a cached, metered client.
    max_chars : int or None, optional
        Per-passage truncation in the rendered evidence block.
    template : str, optional
        Judge instructions using the question, reference, evidence, and answer
        placeholders. Custom templates must return the same verdict fields.

    Returns
    -------
    AnswerJudgement
        The verdict for this question.
    """
    prompt = judge_prompt(
        question, answer_text, reference, evidence, max_chars, template
    )
    return parse_judgement(query_id, complete(prompt))


def judge_prompt(
    question: str,
    answer_text: str,
    reference: str,
    evidence: Sequence[Evidence],
    max_chars: int | None = 1200,
    template: str = JUDGE_PROMPT,
) -> str:
    """Return the judge prompt :func:`judge_answer` sends for one answer.

    Parameters
    ----------
    question, answer_text, reference, evidence, max_chars, template
        As for :func:`judge_answer`.

    Returns
    -------
    str
        The rendered prompt.
    """
    from ..graphrag.synthesis import format_sources

    return template.format(
        question=question,
        reference=reference,
        evidence=format_sources(list(evidence), max_chars) or "(no evidence retrieved)",
        answer=answer_text,
    )


class AnswerCase(NamedTuple):
    """One answer to judge: the arguments :func:`judge_answer` takes for it.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    question : str
        The question asked.
    answer_text : str
        The system's answer.
    reference : str
        Ground-truth answer text.
    evidence : sequence of Evidence
        The passages the system was given.
    """

    query_id: str
    question: str
    answer_text: str
    reference: str
    evidence: Sequence[Evidence]


def judge_answers(
    cases: Iterable[AnswerCase],
    complete: Callable[[str], str],
    max_chars: int | None = 1200,
    *,
    template: str = JUDGE_PROMPT,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AnswerJudgement | None]:
    """Judge many answers, as one batch when ``complete`` can send one.

    Every verdict is the one :func:`judge_answer` returns for the same case,
    because the prompt is the same. A call that fails, or a response with no
    JSON object, gives None for that case instead of stopping the run, so the
    verdicts already paid for are kept.

    Parameters
    ----------
    cases : iterable of AnswerCase
        The answers to judge.
    complete : callable
        Takes a prompt, returns text. When it has a
        ``batch(prompts, on_progress=None)`` method returning one reply or one
        exception per prompt, as :meth:`aieng.advanced_rag.batch.ProxyBatch.completer`
        does, every prompt is sent through it at once. Otherwise each prompt is
        sent in turn.
    max_chars : int or None, optional
        Per-passage truncation in the rendered evidence block.
    template : str, optional
        Judge instructions, as for :func:`judge_answer`.
    on_progress : callable or None, optional
        Called as ``on_progress(done, total)``.

    Returns
    -------
    list of AnswerJudgement or None
        One per case, in order.
    """
    items = list(cases)
    prompts = [
        judge_prompt(
            c.question, c.answer_text, c.reference, c.evidence, max_chars, template
        )
        for c in items
    ]
    batch = getattr(complete, "batch", None)
    replies: list[str | BaseException] = []
    if callable(batch):
        replies = list(batch(prompts, on_progress=on_progress))
    else:
        for done, prompt in enumerate(prompts, start=1):
            try:
                replies.append(complete(prompt))
            except Exception as error:  # noqa: BLE001 - one failed call is one missing verdict
                replies.append(error)
            if on_progress is not None:
                on_progress(done, len(prompts))
    verdicts: list[AnswerJudgement | None] = []
    for case, reply in zip(items, replies, strict=True):
        if isinstance(reply, BaseException):
            verdicts.append(None)
            continue
        try:
            verdicts.append(parse_judgement(case.query_id, reply))
        except ValueError:
            verdicts.append(None)
    return verdicts


def answer_with_budget(
    question: str,
    evidence: Sequence[Evidence],
    complete: Callable[[str], str],
    budget: int,
    max_chars: int | None = 1200,
) -> tuple[str, list[Evidence]]:
    """Answer a question from the top ``budget`` passages.

    The budget is enforced here rather than at retrieval time so that every
    method is compared on the same amount of context. A method that returns
    fewer passages than the budget is not padded; it answers with what it found,
    which is itself part of what is being compared.

    Parameters
    ----------
    question : str
        The question.
    evidence : sequence of Evidence
        Retrieved passages in rank order.
    complete : callable
        Takes a prompt, returns text.
    budget : int
        Maximum passages to place in the prompt.
    max_chars : int or None, optional
        Per-passage truncation.

    Returns
    -------
    tuple of (str, list of Evidence)
        The answer and the passages it was given.
    """
    if budget < 1:
        raise ValueError("evidence budget must be at least 1")
    used = list(evidence)[:budget]
    return complete(build_prompt(question, used, max_chars=max_chars)), used


COVERAGE_PROMPT = """You are grading an answer to a question that requires \
information from several documents.

Question:
{question}

Each document below contributes something the answer is required to state. For \
each one, decide whether the candidate answer actually states that \
contribution. Judge the substance, not the wording, and do not credit a vague \
gesture at the topic.

Required contributions:
{contributions}

Candidate answer:
{answer}

Reply with JSON only, in this form:
{{"verdicts": [{{"document": "<document id>", "covered": true or false}}]}}
"""

COVERAGE_TABLE_HEADER = (
    "| Method | Mean contribution coverage | Complete coverage |\n| --- | ---: | ---: |"
)


def parse_coverage_verdicts(raw: str, gold: Sequence[str]) -> dict[str, bool]:
    """Read the coverage judge's JSON, defaulting anything it omitted to False.

    A judge asked for one verdict per document sometimes returns fewer, or wraps
    the object in prose. Failing the run on that would lose the verdicts already
    paid for, so a missing document is read as not covered.

    Parameters
    ----------
    raw : str
        The judge's raw response text.
    gold : sequence of str
        The documents a verdict is required for.

    Returns
    -------
    dict
        Document identifier to whether the answer stated its contribution.
    """
    verdicts = dict.fromkeys(gold, False)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return verdicts
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return verdicts
    for item in payload.get("verdicts", []):
        doc_id = str(item.get("document", "")).strip()
        if doc_id in verdicts:
            verdicts[doc_id] = bool(item.get("covered"))
    return verdicts


def judge_coverage(
    question: str,
    contributions: Mapping[str, str],
    answer_text: str,
    complete: Callable[[str], str],
    max_answer_chars: int = 6000,
) -> dict[str, bool]:
    """Judge one answer against the per-document contributions of one question.

    This is the fan-out counterpart to ``judge_answer``. A retrieval comparison
    scores a document list, and for a global-search method that list has to be
    recovered from citations rather than returned by the method. Scoring the
    answer instead removes that interpretation: a fan-out question records what
    each gold document has to contribute, so the answer can be scored on how
    many of those contributions it states.

    One judge call returns a verdict per gold document.

    Parameters
    ----------
    question : str
        The question asked.
    contributions : mapping
        Document identifier to what that document has to contribute. Its keys
        are the gold set.
    answer_text : str
        The system's answer.
    complete : callable
        Takes a prompt, returns text. Supply a cached, metered client.
    max_answer_chars : int, optional
        Truncation applied to the answer before it reaches the judge.

    Returns
    -------
    dict
        Document identifier to whether the answer stated its contribution.
    """
    gold = list(contributions)
    rendered = "\n".join(
        f"- {doc_id}: {contributions.get(doc_id) or '(not recorded)'}"
        for doc_id in gold
    )
    reply = complete(
        COVERAGE_PROMPT.format(
            question=question,
            contributions=rendered,
            answer=answer_text[:max_answer_chars],
        ),
    )
    return parse_coverage_verdicts(reply, gold)


@dataclass(frozen=True)
class CoverageReport:
    """Aggregate contribution coverage over a fan-out question set.

    Parameters
    ----------
    method : str
        Name of the method the answers came from.
    verdicts : list of dict
        One mapping per question, from document identifier to whether the
        answer stated that document's contribution.
    """

    method: str
    verdicts: list[dict[str, bool]] = field(default_factory=list)

    @property
    def questions(self) -> int:
        """Return the number of questions judged."""
        return len(self.verdicts)

    @property
    def mean_coverage(self) -> float:
        """Return the mean share of required contributions the answers state."""
        scored = [v for v in self.verdicts if v]
        if not scored:
            return 0.0
        return sum(sum(v.values()) / len(v) for v in scored) / len(scored)

    @property
    def complete_coverage_rate(self) -> float:
        """Return the share of questions whose every contribution was stated.

        The harsher reading of the same verdicts: a fan-out answer that states
        three of four required contributions is still an incomplete answer.
        """
        scored = [v for v in self.verdicts if v]
        if not scored:
            return 0.0
        return sum(1 for v in scored if all(v.values())) / len(scored)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary including per-question verdicts."""
        return {
            "method": self.method,
            "questions": self.questions,
            "mean_coverage": round(self.mean_coverage, 4),
            "complete_coverage_rate": round(self.complete_coverage_rate, 4),
            "verdicts": self.verdicts,
        }

    def table_row(self, label: str | None = None) -> str:
        """Render one Markdown row under ``COVERAGE_TABLE_HEADER``."""
        return (
            f"| {label or self.method} | {self.mean_coverage:.4f} "
            f"| {self.complete_coverage_rate:.4f} |"
        )
