"""Rank agreement: whether generated labels order retrieval arms as real labels do.

A generated set is useful for choosing between retrieval configurations only if
it ranks them as the real labels would. Absolute scores on synthetic sets run
high (van Elburg et al., 2025; Rahmani et al., SIGIR 2024), so the comparison
here is of orderings and of paired differences, never of absolute recall.

Two measures:

* :func:`kendall_tau`, tau-b over the arms' mean scores under the two label
  sets.
* :func:`pairwise_agreement`, for every arm pair, the paired bootstrap of the
  per-question difference under each label set, and whether the generated
  labels give the same sign. A pair whose real interval excludes zero and whose
  generated interval excludes zero on the other side is a significant
  reversal, which is the failure that matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from statistics import mean
from typing import Any

from aieng.advanced_rag.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from aieng.advanced_rag.evaluation.multihop import paired_bootstrap


Metric = Callable[[Sequence[str], set[str]], float]


def _reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    return float(reciprocal_rank(ranked, relevant))


def metric(name: str, k: int = 10) -> Metric:
    """Return a per-question metric by name.

    Parameters
    ----------
    name : str
        ``recall``, ``ndcg``, ``mrr``, or ``hit``.
    k : int, optional
        Depth, ignored by ``mrr``.

    Returns
    -------
    callable
        ``(ranked ids, relevant ids) -> float``.

    Raises
    ------
    ValueError
        On an unknown name.
    """
    if name == "recall":
        return lambda ranked, relevant: recall_at_k(ranked, relevant, k)
    if name == "ndcg":
        return lambda ranked, relevant: ndcg_at_k(ranked, relevant, k)
    if name == "mrr":
        return _reciprocal_rank
    if name == "hit":
        return lambda ranked, relevant: 1.0 if set(ranked[:k]) & relevant else 0.0
    raise ValueError(f"unknown metric {name!r}")


def per_question(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    score: Metric,
) -> dict[str, float]:
    """Return one score per judged question, for one arm.

    Questions with an empty judgment set (nulls) are skipped.
    """
    return {qid: score(runs[qid], qrels[qid]) for qid in runs if qrels.get(qid)}


def kendall_tau(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Kendall's tau-b between two score lists over the same items.

    Parameters
    ----------
    left, right : sequence of float
        One score per arm, in the same arm order.

    Returns
    -------
    float
        Tau-b in [-1, 1]. ``nan`` when either list is constant.

    Raises
    ------
    ValueError
        If the lists differ in length or have fewer than two items.
    """
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("need two equal-length lists of at least two items")
    concordant = discordant = ties_left = ties_right = 0
    for i, j in combinations(range(len(left)), 2):
        a = (left[i] > left[j]) - (left[i] < left[j])
        b = (right[i] > right[j]) - (right[i] < right[j])
        if a == 0 and b == 0:
            continue
        if a == 0:
            ties_left += 1
        elif b == 0:
            ties_right += 1
        elif a == b:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def ranking(scores: Mapping[str, float]) -> list[str]:
    """Return arm names from best to worst, ties broken on name."""
    return sorted(scores, key=lambda arm: (-scores[arm], arm))


def pairwise_agreement(
    real: Mapping[str, Mapping[str, float]],
    generated: Mapping[str, Mapping[str, float]],
    *,
    samples: int = 2000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compare every arm pair's paired difference under both label sets.

    Parameters
    ----------
    real, generated : mapping
        Arm name to question id to score, one per label set. Each arm must score
        the same questions within a label set.
    samples : int, optional
        Bootstrap resamples.
    seed : int, optional
        Bootstrap seed.

    Returns
    -------
    list of dict
        One record per pair: the mean difference and 95% interval under each
        label set, whether the real difference is significant, whether the
        signs agree, and whether it is a significant reversal.
    """
    records: list[dict[str, Any]] = []
    for a, b in combinations(sorted(real), 2):
        pair: dict[str, Any] = {"a": a, "b": b}
        for label, scores in (("real", real), ("generated", generated)):
            ids = sorted(scores[a])
            result = paired_bootstrap(
                [scores[a][q] for q in ids],
                [scores[b][q] for q in ids],
                samples=samples,
                seed=seed,
            )
            pair[label] = {
                "mean_difference": round(result.mean_difference, 4),
                "low": round(result.low, 4),
                "high": round(result.high, 4),
                "significant": result.excludes_zero,
            }
        real_sign = _sign(pair["real"]["mean_difference"])
        generated_sign = _sign(pair["generated"]["mean_difference"])
        pair["same_sign"] = real_sign == generated_sign
        pair["significant_reversal"] = (
            pair["real"]["significant"]
            and pair["generated"]["significant"]
            and real_sign != generated_sign
        )
        records.append(pair)
    return records


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def compare_rankings(
    real: Mapping[str, Mapping[str, float]],
    generated: Mapping[str, Mapping[str, float]],
    *,
    samples: int = 2000,
    seed: int = 0,
    pairs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarise how far the generated labels agree with the real ones.

    Parameters
    ----------
    real, generated : mapping
        Arm name to question id to score, as for :func:`pairwise_agreement`.
    samples, seed : int, optional
        Passed to the bootstrap.
    pairs : sequence of dict or None, optional
        Records from :func:`pairwise_agreement` over a superset of these arms,
        reused instead of bootstrapping again. Only pairs of these arms count.

    Returns
    -------
    dict
        Mean score per arm under each label set, both rankings, Kendall's tau,
        the share of pairs with the same sign, the same share over pairs the
        real labels separate significantly, the significant reversals, the mean
        absolute gap (generated minus real), and the pair records.
    """
    arms = sorted(real)
    real_means = {arm: round(mean(real[arm].values()), 4) for arm in arms}
    generated_means = {arm: round(mean(generated[arm].values()), 4) for arm in arms}
    if pairs is None:
        pairs = pairwise_agreement(real, generated, samples=samples, seed=seed)
    else:
        pairs = [p for p in pairs if p["a"] in real and p["b"] in real]
    significant = [p for p in pairs if p["real"]["significant"]]
    return {
        "arms": arms,
        "real_mean": real_means,
        "generated_mean": generated_means,
        "real_ranking": ranking(real_means),
        "generated_ranking": ranking(generated_means),
        "kendall_tau": round(
            kendall_tau(
                [real_means[a] for a in arms], [generated_means[a] for a in arms]
            ),
            4,
        ),
        "pairs": len(pairs),
        "sign_agreement": round(mean(p["same_sign"] for p in pairs), 4)
        if pairs
        else None,
        "real_significant_pairs": len(significant),
        "sign_agreement_on_significant": (
            round(mean(p["same_sign"] for p in significant), 4) if significant else None
        ),
        "significant_reversals": [
            f"{p['a']} vs {p['b']}" for p in pairs if p["significant_reversal"]
        ],
        "mean_gap": round(mean(generated_means[a] - real_means[a] for a in arms), 4),
        "pair_records": pairs,
    }


__all__ = [
    "Metric",
    "compare_rankings",
    "kendall_tau",
    "metric",
    "pairwise_agreement",
    "per_question",
    "ranking",
]
