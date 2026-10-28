"""The shared retrieval metrics.

Retrieval-failure rate is the number that matters for method selection. It is
the share of answerable questions where nothing relevant reached the reader at
all, so it isolates the retrieval stage from the generator. A pipeline cannot
answer above its retrieval ceiling, and the failure rate is that ceiling stated
directly. The ranking metrics next to it, recall@k, MRR, and nDCG@k, are what
make a result comparable with published retrieval benchmarks.

Recall@k, hit rate, reciprocal rank, and binary-gain nDCG@k follow the
definitions in trec_eval, and ``ir_measures``, ``ranx``, and ``pytrec_eval``
compute the same values on untied rankings; the tests check this against
``ir_measures`` where it is installed. These functions stay here because they
take ranked identifiers without scores and import nothing, which keeps the base
install free of compiled dependencies. trec_eval re-sorts tied scores by
document identifier, descending, so convert a ranking to strictly decreasing
scores before handing it to a library. Retrieval-failure rate, the all-gold
measures, null detection, and the character budget have no library counterpart.

Everything here takes ranked chunk IDs and a judgment set. Nothing imports a
vector store, a model, or a corpus, so the same functions score a dense
retriever, a graph traversal, or an agent's tool trace.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean


def retrieval_failure(retrieved: Sequence[str], relevant: set[str], k: int) -> bool:
    """Return True when the top-k holds nothing relevant.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked chunk identifiers.
    relevant : set of str
        The judgment set for this query.
    k : int
        Depth the reader sees.

    Returns
    -------
    bool
        True on a failure.
    """
    return not (set(retrieved[:k]) & relevant)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the share of the relevant chunks that appear in the top-k.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked chunk identifiers.
    relevant : set of str
        The judgment set for this query.
    k : int
        Depth.

    Returns
    -------
    float
        Recall, or 0.0 when there is nothing to recall.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return 1.0 when at least one relevant chunk is in the top-k.

    The per-query complement of the failure rate.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked chunk identifiers.
    relevant : set of str
        The judgment set for this query.
    k : int
        Depth.

    Returns
    -------
    float
        1.0 or 0.0.
    """
    return 0.0 if retrieval_failure(retrieved, relevant, k) else 1.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return 1/rank of the first relevant chunk, or 0.0 if none was retrieved.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked chunk identifiers, at full depth.
    relevant : set of str
        The judgment set for this query.

    Returns
    -------
    float
        The reciprocal rank.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return binary-gain nDCG@k.

    Binary because span-derived judgments are binary: a chunk either overlaps a
    gold answer or it does not. The ideal ranking puts every relevant chunk, up
    to k, at the top.

    Parameters
    ----------
    retrieved : sequence of str
        Ranked chunk identifiers.
    relevant : set of str
        The judgment set for this query.
    k : int
        Depth.

    Returns
    -------
    float
        nDCG@k, or 0.0 when there is nothing relevant.
    """
    if not relevant:
        return 0.0
    # A relevant identifier counts once, at its first rank. Without this a
    # repeated identifier gains twice and nDCG can pass 1: a ranking of
    # ["a", "a"] against {"a"} would score 1.63. trec_eval and ranx see one
    # document per identifier, so this matches them.
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant and chunk_id not in retrieved[: rank - 1]
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """Scores for one retriever over one set of answerable queries.

    Parameters
    ----------
    k : int
        The depth every rate was measured at.
    queries : int
        Queries scored. Only those with a non-empty judgment set count.
    retrieval_failure_rate : float
        Share of queries where nothing relevant reached the top-k.
    hit_rate : float
        The complement of the failure rate.
    recall_at_k : float
        Mean recall at k.
    mrr : float
        Mean reciprocal rank, at full depth.
    ndcg_at_k : float
        Mean binary-gain nDCG at k.
    """

    k: int
    queries: int
    retrieval_failure_rate: float
    hit_rate: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary, with k in the metric names."""
        return {
            "k": self.k,
            "queries": self.queries,
            "retrieval_failure_rate": self.retrieval_failure_rate,
            "hit_rate": self.hit_rate,
            f"recall_at_{self.k}": self.recall_at_k,
            "mrr": self.mrr,
            f"ndcg_at_{self.k}": self.ndcg_at_k,
        }

    def table_row(self, label: str) -> str:
        """Render one Markdown table row under ``TABLE_HEADER``."""
        return (
            f"| {label} | {self.retrieval_failure_rate:.4f} | {self.recall_at_k:.4f} "
            f"| {self.mrr:.4f} | {self.ndcg_at_k:.4f} | {self.queries} |"
        )


TABLE_HEADER = (
    "| Retriever | Retrieval-failure rate | Recall@k | MRR | nDCG@k | Queries |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: |"
)


def score_retrieval(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    *,
    k: int = 10,
) -> RetrievalReport:
    """Score ranked results for many queries.

    Only queries with a non-empty judgment set are scored: the null cases have
    no relevant chunk by construction, and averaging them in would report a
    failure rate of 1.0 for questions that are correctly unanswerable.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked chunk identifiers.
    qrels : mapping
        Query identifier to the set of relevant chunk identifiers.
    k : int, optional
        Depth to score at.

    Returns
    -------
    RetrievalReport
        The aggregate.

    Raises
    ------
    ValueError
        If every query had an empty judgment set.
    """
    judged = [qid for qid in runs if qrels.get(qid)]
    if not judged:
        raise ValueError("No scored queries: every query had an empty judgment set")

    failures, hits, recalls, rrs, ndcgs = [], [], [], [], []
    for qid in judged:
        retrieved, relevant = runs[qid], qrels[qid]
        failures.append(1.0 if retrieval_failure(retrieved, relevant, k) else 0.0)
        hits.append(hit_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))

    return RetrievalReport(
        k=k,
        queries=len(judged),
        retrieval_failure_rate=round(mean(failures), 4),
        hit_rate=round(mean(hits), 4),
        recall_at_k=round(mean(recalls), 4),
        mrr=round(mean(rrs), 4),
        ndcg_at_k=round(mean(ndcgs), 4),
    )


@dataclass(frozen=True)
class NullReport:
    """How well a pipeline abstains on questions with no answer in the corpus.

    CUAD is 70% unanswerable, so a system that always answers scores well on
    retrieval and still fails a "not stated" requirement. Precision and recall
    are over the abstain decision, so recall is the share of genuinely
    unanswerable questions the system declined to answer.

    Parameters
    ----------
    null_queries : int
        Questions the dataset marks unanswerable.
    answerable_queries : int
        The rest.
    abstain_precision : float
        Share of abstentions that were correct.
    abstain_recall : float
        Share of unanswerable questions the system abstained on.
    false_answer_rate : float
        Share of unanswerable questions the system answered anyway.
    """

    null_queries: int
    answerable_queries: int
    abstain_precision: float
    abstain_recall: float
    false_answer_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary."""
        return {
            "null_queries": self.null_queries,
            "answerable_queries": self.answerable_queries,
            "abstain_precision": self.abstain_precision,
            "abstain_recall": self.abstain_recall,
            "false_answer_rate": self.false_answer_rate,
        }


def score_null_detection(
    abstained: Mapping[str, bool],
    truly_null: Iterable[str],
) -> NullReport:
    """Score abstention against the dataset's unanswerable labels.

    Parameters
    ----------
    abstained : mapping
        Query identifier to whether the system declined to answer.
    truly_null : iterable of str
        Identifiers of the genuinely unanswerable questions.

    Returns
    -------
    NullReport
        The aggregate.

    Raises
    ------
    ValueError
        If there are no queries to score.
    """
    null_ids = set(truly_null)
    scored = list(abstained)
    if not scored:
        raise ValueError("No queries to score")

    true_abstain = sum(1 for q in scored if abstained[q] and q in null_ids)
    all_abstain = sum(1 for q in scored if abstained[q])
    total_null = sum(1 for q in scored if q in null_ids)
    total_answerable = len(scored) - total_null
    false_answers = sum(1 for q in scored if not abstained[q] and q in null_ids)

    return NullReport(
        null_queries=total_null,
        answerable_queries=total_answerable,
        abstain_precision=round(true_abstain / all_abstain, 4) if all_abstain else 0.0,
        abstain_recall=round(true_abstain / total_null, 4) if total_null else 0.0,
        false_answer_rate=round(false_answers / total_null, 4) if total_null else 0.0,
    )


def retrieved_chars(
    runs: Mapping[str, Sequence[str]],
    chars_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Return the mean characters a reader receives from the top-k.

    The number that makes a chunk-size comparison fair. Ten 4,000-character
    chunks put four times as much text in front of the reader as ten
    1,000-character ones, so a failure rate at a fixed k rewards large chunks for
    retrieving more of the document rather than for locating the answer.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked chunk identifiers.
    chars_by_id : mapping
        Chunk identifier to its length in characters.
    k : int
        Depth.

    Returns
    -------
    float
        Mean characters across queries.
    """
    totals = [
        sum(chars_by_id.get(cid, 0) for cid in ranked[:k]) for ranked in runs.values()
    ]
    return sum(totals) / max(len(totals), 1)


def budget_k(
    runs: Mapping[str, Sequence[str]],
    chars_by_id: Mapping[str, int],
    budget: float,
    depth: int,
) -> int:
    """Return the smallest k whose top-k holds at least ``budget`` characters.

    Parameters
    ----------
    runs : mapping
        Query identifier to ranked chunk identifiers.
    chars_by_id : mapping
        Chunk identifier to its length in characters.
    budget : float
        Characters the reader is allowed.
    depth : int
        Largest k to consider, and the fallback when the budget is unreachable.

    Returns
    -------
    int
        The k to compare at.
    """
    for k in range(1, depth + 1):
        if retrieved_chars(runs, chars_by_id, k) >= budget:
            return k
    return depth
