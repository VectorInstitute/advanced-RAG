"""Score pages and answers, split retrieval loss from reading loss, and price the arms.

Two things compare one to one across arms here. Page-level retrieval does,
because every arm returns ranked page identifiers over one page universe.
Answer accuracy does, when one reader model answers for every arm. Chunk
metrics and raw cost figures do not, and nothing here ranks them across arms.

An answer is correct when it states the reference value within 1% relative,
or, when no number in it matches, when a judge from another model says it
states the same thing as the reference. A null question is answered correctly
by abstaining.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aieng.advanced_rag.evaluation.answers import judge_answer
from aieng.advanced_rag.evaluation.metrics import ndcg_at_k, recall_at_k, retrieval_failure


#: The reply every reader and checker gives when the evidence lacks the answer.
NOT_STATED = "NOT STATED"

_NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")
_ABSTAIN = re.compile(
    r"not stated|not (?:provided|shown|given|available|included|mentioned|specified)|"
    r"does not (?:state|provide|show|include|contain|mention|specify)|cannot be determined|no information",
    re.IGNORECASE,
)


def numbers_in(text: str) -> list[float]:
    """Return every number in a text, reading accounting parentheses as negative.

    Thousands separators are removed. A trailing percent sign, a currency
    sign, and units such as "million" are ignored, so "$2,868 million" reads as
    2868 and "(1.5)" as -1.5.
    """
    found: list[float] = []
    for match in _NUMBER.finditer(text or ""):
        token = match.group(0)
        negative = token.startswith("(") and token.endswith(")") or "-" in token
        digits = re.sub(r"[^\d.]", "", token)
        if not digits or digits == ".":
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        found.append(-value if negative else value)
    return found


def numeric_match(answer: str, value: float, tolerance: float = 0.01) -> bool:
    """Return whether a number in the answer is within ``tolerance`` of ``value``.

    The sign is ignored, because "a decrease of 5" and "-5" state one change.
    """
    target = abs(value)
    for number in numbers_in(answer):
        found = abs(number)
        if target == 0:
            if found == 0:
                return True
        elif abs(found - target) <= tolerance * target:
            return True
    return False


def abstained(answer: str) -> bool:
    """Return whether an answer declines to answer."""
    return bool(_ABSTAIN.search(answer or "")) or not (answer or "").strip()


@dataclass(frozen=True)
class Verdict:
    """How one answer was scored.

    Parameters
    ----------
    correct : bool
        The verdict.
    method : str
        ``numeric``, ``abstention``, ``judge``, or ``abstained`` (an answerable
        question the reader declined).
    """

    correct: bool
    method: str


def score_answer(
    question: Mapping[str, Any],
    answer: str,
    judge: Callable[[str], str] | None,
) -> Verdict:
    """Score one answer against a question's reference.

    Parameters
    ----------
    question : mapping
        A question record with ``group``, ``question``, ``answer``, and
        optionally ``value``.
    answer : str
        The reader's answer.
    judge : callable or None
        A completion function for the tiebreak, or None to mark unmatched
        answers incorrect.

    Returns
    -------
    Verdict
        Correct or not, and how that was decided.
    """
    if question["group"] == "null":
        return Verdict(abstained(answer), "abstention")
    if abstained(answer):
        return Verdict(False, "abstained")
    value = question.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and numeric_match(answer, float(value)):
        return Verdict(True, "numeric")
    if judge is None:
        return Verdict(False, "numeric")
    verdict = judge_answer(
        str(question["query_id"]),
        str(question["question"]),
        answer,
        str(question["answer"]),
        [],
        judge,
    )
    return Verdict(verdict.correct, "judge")


def page_metrics(
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, Iterable[str]],
    k: int = 5,
) -> dict[str, float]:
    """Return mean recall@k, nDCG@k, and failure@k over questions with gold pages.

    Parameters
    ----------
    rankings : mapping
        Question identifier to ranked page identifiers.
    gold : mapping
        Question identifier to its gold pages. Questions with none, the nulls,
        are skipped.
    k : int, optional
        Depth.

    Returns
    -------
    dict
        ``recall``, ``ndcg``, ``failure``, and ``n``.
    """
    recalls: list[float] = []
    ndcgs: list[float] = []
    failures: list[float] = []
    for query_id, pages in gold.items():
        relevant = set(pages)
        if not relevant or query_id not in rankings:
            continue
        ranked = list(rankings[query_id])
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        failures.append(float(retrieval_failure(ranked, relevant, k)))
    n = len(recalls)
    if not n:
        return {"recall": math.nan, "ndcg": math.nan, "failure": math.nan, "n": 0}
    return {
        "recall": round(sum(recalls) / n, 4),
        "ndcg": round(sum(ndcgs) / n, 4),
        "failure": round(sum(failures) / n, 4),
        "n": n,
    }


@dataclass(frozen=True)
class Scenario:
    """A corpus, a question load, and an update rate: the frame arm costs compare in.

    Parameters
    ----------
    pages : int
        Pages in the corpus.
    pictures_per_page : float
        Described pictures per page.
    tables_per_page : float
        Summarised tables per page.
    questions : int
        Questions answered between re-indexes.
    reindexes : int
        Times the corpus is re-indexed, counting the first build.
    """

    pages: int
    pictures_per_page: float
    tables_per_page: float
    questions: int
    reindexes: int = 1


def scenario_cost(
    scenario: Scenario,
    *,
    enrichment_usd_per_element: float,
    reader_usd_per_question: Mapping[str, float],
    machine_seconds_per_page: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Return indexing and query cost per arm under one scenario.

    Parameters
    ----------
    scenario : Scenario
        The corpus and load.
    enrichment_usd_per_element : float
        Measured cost of one description or summary on the chosen model.
    reader_usd_per_question : mapping
        Arm to measured reader cost per question on the chosen model.
    machine_seconds_per_page : mapping
        Arm to indexing machine time per page on the stated hardware.

    Returns
    -------
    dict
        Arm to ``index_usd``, ``query_usd``, ``total_usd``, and ``index_hours``.
    """
    elements = scenario.pages * (scenario.pictures_per_page + scenario.tables_per_page)
    out: dict[str, dict[str, float]] = {}
    for arm, per_question in reader_usd_per_question.items():
        enriched = arm not in {"T", "VV"}
        index_usd = elements * enrichment_usd_per_element * scenario.reindexes if enriched else 0.0
        query_usd = per_question * scenario.questions * scenario.reindexes
        hours = machine_seconds_per_page.get(arm, 0.0) * scenario.pages * scenario.reindexes / 3600
        out[arm] = {
            "index_usd": round(index_usd, 2),
            "query_usd": round(query_usd, 2),
            "total_usd": round(index_usd + query_usd, 2),
            "index_hours": round(hours, 2),
        }
    return out


def break_even_questions(index_usd_a: float, per_question_a: float, index_usd_b: float, per_question_b: float) -> float:
    """Return the question count at which two arms cost the same, or infinity."""
    slope = per_question_b - per_question_a
    if slope <= 0:
        return math.inf
    return max(0.0, (index_usd_a - index_usd_b) / slope)


__all__ = [
    "NOT_STATED",
    "Scenario",
    "Verdict",
    "abstained",
    "break_even_questions",
    "numbers_in",
    "numeric_match",
    "page_metrics",
    "scenario_cost",
    "score_answer",
]
