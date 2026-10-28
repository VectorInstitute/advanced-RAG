"""Per-hop summaries and paired comparisons for the MuSiQue graph test.

Everything here is a pure function of the per-question rows, so it is tested
without a model, an index, or a network. The rows carry passage identifiers and
scores and no question or passage text, which is what lets the results be
committed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from aieng.advanced_rag.evaluation.multihop import (
    all_gold_hit,
    mcnemar_exact,
    paired_bootstrap,
    supporting_recall_at_k,
)


DEPTHS = (5, 10)


@dataclass(frozen=True)
class Row:
    """One arm's result on one question.

    Parameters
    ----------
    question_id : str
        The MuSiQue identifier.
    hops : int
        2, 3, or 4.
    ranked : tuple of str
        Passage identifiers the arm retrieved, best first.
    em : float
        Exact match of the generated answer, 0.0 or 1.0.
    f1 : float
        Token F1 of the generated answer.
    """

    question_id: str
    hops: int
    ranked: tuple[str, ...]
    em: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, with no question or passage text."""
        record = asdict(self)
        record["ranked"] = list(self.ranked)
        return record


def groups(rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Split rows by hop count, and add ``all``."""
    found: dict[str, list[Row]] = {}
    for hops in sorted({r.hops for r in rows}):
        found[f"{hops}hop"] = [r for r in rows if r.hops == hops]
    found["all"] = list(rows)
    return found


def summarise(rows: Sequence[Row], gold: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    """Summarise one arm's rows by hop count.

    Parameters
    ----------
    rows : sequence of Row
        One arm's rows.
    gold : mapping
        Question identifier to its supporting passage identifiers.

    Returns
    -------
    dict
        For each hop group and ``all``: the question count, the share with every
        supporting passage in the top 5 and top 10, the mean share of supporting
        passages found at each depth, and mean exact match and F1.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, members in groups(rows).items():
        entry: dict[str, Any] = {"questions": len(members)}
        for k in DEPTHS:
            entry[f"all_supporting_recall_at_{k}"] = round(
                mean(1.0 if all_gold_hit(r.ranked, gold[r.question_id], k) else 0.0 for r in members),
                4,
            )
            entry[f"mean_supporting_recall_at_{k}"] = round(
                mean(supporting_recall_at_k(r.ranked, gold[r.question_id], k) for r in members), 4
            )
        entry["exact_match"] = round(mean(r.em for r in members), 4)
        entry["f1"] = round(mean(r.f1 for r in members), 4)
        out[name] = entry
    return out


def compare(
    candidate: Sequence[Row],
    baseline: Sequence[Row],
    gold: Mapping[str, Sequence[str]],
    *,
    samples: int = 10_000,
) -> dict[str, dict[str, Any]]:
    """Compare two arms on the same questions, by hop group.

    Parameters
    ----------
    candidate, baseline : sequence of Row
        The two arms' rows. They must cover the same question identifiers.
    gold : mapping
        Question identifier to its supporting passages.
    samples : int, optional
        Bootstrap resamples.

    Returns
    -------
    dict
        For each hop group and ``all``, the paired bootstrap interval on the mean
        difference of F1, and exact McNemar tests on all-supporting recall at 10
        and on exact match. Positive differences favour ``candidate``.

    Raises
    ------
    ValueError
        If the arms do not cover the same questions.
    """
    if {r.question_id for r in candidate} != {r.question_id for r in baseline}:
        raise ValueError("the arms must cover the same questions")
    base = {r.question_id: r for r in baseline}
    out: dict[str, dict[str, Any]] = {}
    for name, members in groups(candidate).items():
        paired = [(r, base[r.question_id]) for r in members]
        a_hit = [all_gold_hit(x.ranked, gold[x.question_id], 10) for x, _ in paired]
        b_hit = [all_gold_hit(y.ranked, gold[y.question_id], 10) for _, y in paired]
        a_em = [x.em == 1.0 for x, _ in paired]
        b_em = [y.em == 1.0 for _, y in paired]
        f1 = paired_bootstrap([x.f1 for x, _ in paired], [y.f1 for _, y in paired], samples=samples)
        hit_test, em_test = mcnemar_exact(a_hit, b_hit), mcnemar_exact(a_em, b_em)
        out[name] = {
            "questions": len(paired),
            "f1_difference": round(f1.mean_difference, 4),
            "f1_interval": [round(f1.low, 4), round(f1.high, 4)],
            "f1_interval_excludes_zero": f1.excludes_zero,
            "all_supporting_at_10": {
                "only_candidate": hit_test.only_a,
                "only_baseline": hit_test.only_b,
                "p_value": round(hit_test.p_value, 4),
            },
            "exact_match": {
                "only_candidate": em_test.only_a,
                "only_baseline": em_test.only_b,
                "p_value": round(em_test.p_value, 4),
            },
        }
    return out
