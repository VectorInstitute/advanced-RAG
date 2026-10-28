"""Scoring for multi-hop question answering: every supporting passage, and the answer.

A multi-hop question is answered only if every hop's passage reaches the reader, so
the retrieval measure here asks for all of them. ``metrics.retrieval_failure`` asks
whether any gold passage appears, which a two-hop question passes by finding half
its evidence. The answer measures are the SQuAD-style exact match and token F1 that
MuSiQue reports, taken as the best over the gold answer and its aliases.

The answer measures follow the SQuAD evaluation script. The McNemar test agrees
with ``scipy.stats.binomtest`` and ``statsmodels``' ``mcnemar(exact=True)``, and
the bootstrap is the percentile method (Efron, 1979) that
``scipy.stats.bootstrap`` also offers. The tests check the McNemar agreement.

The comparison helpers are paired, because two arms answer the same questions: a
bootstrap interval on a mean difference, and an exact McNemar test on which
questions one arm gets right and the other does not.
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean


def all_gold_hit(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> bool:
    """Return True when every relevant passage is in the top ``k``.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked passage identifiers.
    relevant : iterable of str
        The question's supporting passages.
    k : int
        Depth the reader sees.

    Returns
    -------
    bool
        True when none is missing. A question with no relevant passage is False,
        because there is nothing to find.
    """
    gold = set(relevant)
    return bool(gold) and gold <= set(retrieved[:k])


def all_gold_recall_at_k(
    runs: Mapping[str, Sequence[str]], qrels: Mapping[str, Iterable[str]], k: int
) -> float:
    """Return the share of questions whose every supporting passage is in the top ``k``.

    Parameters
    ----------
    runs : mapping
        Question identifier to its ranked passage identifiers.
    qrels : mapping
        Question identifier to its supporting passages.
    k : int
        Depth to score at.

    Returns
    -------
    float
        The share, over the questions in ``runs`` that have supporting passages.

    Raises
    ------
    ValueError
        If no question in ``runs`` has supporting passages.
    """
    scored = [qid for qid in runs if list(qrels.get(qid, ()))]
    if not scored:
        raise ValueError("no question has supporting passages to score")
    return mean(1.0 if all_gold_hit(runs[q], qrels[q], k) else 0.0 for q in scored)


def supporting_recall_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int
) -> float:
    """Return the share of one question's supporting passages in the top ``k``."""
    gold = set(relevant)
    return len(gold & set(retrieved[:k])) / len(gold) if gold else 0.0


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalise_answer(text: str) -> str:
    """Return an answer in the form the SQuAD and MuSiQue scorers compare.

    Lowercased, with punctuation and the articles ``a``, ``an``, and ``the``
    removed and whitespace collapsed.
    """
    lowered = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def exact_match(prediction: str, golds: Iterable[str]) -> float:
    """Return 1.0 when the prediction equals any gold answer after normalising."""
    normal = normalise_answer(prediction)
    return 1.0 if any(normal == normalise_answer(g) for g in golds) else 0.0


def _f1(prediction: str, gold: str) -> float:
    predicted, expected = (
        normalise_answer(prediction).split(),
        normalise_answer(gold).split(),
    )
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: Iterable[str]) -> float:
    """Return the best token-overlap F1 between the prediction and any gold answer."""
    return max((_f1(prediction, gold) for gold in golds), default=0.0)


@dataclass(frozen=True)
class BootstrapResult:
    """A paired bootstrap of the mean difference between two arms.

    Parameters
    ----------
    mean_difference : float
        Mean of ``a - b`` over the questions.
    low : float
        Lower end of the percentile interval.
    high : float
        Upper end.
    confidence : float
        The interval's coverage.
    """

    mean_difference: float
    low: float
    high: float
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """Return whether the whole interval lies on one side of zero."""
        return self.low > 0 or self.high < 0


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap the mean of ``a - b`` over questions both arms answered.

    Parameters
    ----------
    a, b : sequence of float
        One score per question, in the same question order.
    samples : int, optional
        Bootstrap resamples.
    seed : int, optional
        Seed, so the interval is reproducible.
    confidence : float, optional
        Coverage of the percentile interval.

    Raises
    ------
    ValueError
        If the arms have different lengths or no questions.
    """
    if len(a) != len(b) or not a:
        raise ValueError("arms must score the same, non-empty set of questions")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    return BootstrapResult(
        mean_difference=mean(diffs),
        low=means[int(tail * samples)],
        high=means[min(samples - 1, int((1 - tail) * samples))],
        confidence=confidence,
    )


@dataclass(frozen=True)
class McNemarResult:
    """An exact McNemar test between two arms on the same questions.

    Parameters
    ----------
    only_a : int
        Questions arm A got right and arm B got wrong.
    only_b : int
        Questions arm B got right and arm A got wrong.
    p_value : float
        Two-sided exact binomial p-value under the null that the two arms are equally
        likely to win a disagreement. 1.0 when the arms never disagree.
    """

    only_a: int
    only_b: int
    p_value: float


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Return the exact two-sided McNemar p-value for the discordant counts.

    This is the two-sided binomial tail at probability one half over the
    ``only_a + only_b`` questions the arms disagree on. It equals
    ``scipy.stats.binomtest(min(only_a, only_b), only_a + only_b).pvalue`` and
    ``statsmodels.stats.contingency_tables.mcnemar(table, exact=True).pvalue``,
    which the tests check. The package base install has no dependencies, so the
    20 lines stay here instead of importing either one. Every script that
    reports a McNemar p-value calls this function.

    Parameters
    ----------
    only_a, only_b : int
        Questions only arm A got right, and only arm B got right.

    Returns
    -------
    float
        The p-value, or 1.0 when the arms never disagree.
    """
    disagreements = only_a + only_b
    if disagreements == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = (
        sum(math.comb(disagreements, i) for i in range(smaller + 1)) / 2**disagreements
    )
    return float(min(1.0, 2 * tail))


def mcnemar_exact(
    a_correct: Sequence[bool], b_correct: Sequence[bool]
) -> McNemarResult:
    """Test whether two arms differ on the questions they disagree about.

    Parameters
    ----------
    a_correct, b_correct : sequence of bool
        Whether each arm got each question right, in the same question order.

    Raises
    ------
    ValueError
        If the arms have different lengths.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("arms must score the same questions")
    only_a = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    return McNemarResult(only_a, only_b, mcnemar_p(only_a, only_b))


__all__ = [
    "BootstrapResult",
    "McNemarResult",
    "all_gold_hit",
    "all_gold_recall_at_k",
    "exact_match",
    "mcnemar_exact",
    "mcnemar_p",
    "normalise_answer",
    "paired_bootstrap",
    "supporting_recall_at_k",
    "token_f1",
]
