"""MuSiQue dense retrieval and single-pass answering through the shared harness."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import MusiqueQuestion, Passage
from aieng.advanced_rag.evaluation.metrics import score_retrieval
from aieng.advanced_rag.evaluation.multihop import (
    all_gold_recall_at_k,
    exact_match,
    token_f1,
)
from aieng.advanced_rag.evaluation.runner import ArmResult, Experiment
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.pipeline.answering import SingleShotPolicy
from aieng.advanced_rag.pipeline.grading import HeuristicGrader
from aieng.advanced_rag.pipeline.spine import SpineConfig
from aieng.advanced_rag.provenance import Provenance
from aieng.advanced_rag.retrieval.index import DenseIndex
from aieng.advanced_rag.retrieval.interface import Hints, ScopeUnsupported


ANSWER_TEMPLATE = """Answer the question using only the numbered sources.
Give the shortest answer that identifies the requested entity, date, or value.
Combine facts across sources when necessary. Do not explain your reasoning.
If the sources do not support an answer, set not_stated to true.
Return one JSON object: {{"answer": "short answer", "sources": [1, 2], "not_stated": false}}.

Question: {query}

Sources:
{sources}
"""


def passage_text(passage: Passage) -> str:
    """Return the same title-plus-text representation for embedding and answering."""
    return f"{passage.title}\n{passage.text}"


@dataclass
class HarnessQuestion:
    """A gold-free question record for corpus-wide harness execution."""

    query_id: str
    question: str

    @property
    def is_answerable(self) -> bool:
        """Return True for HippoRAG 2's answerable MuSiQue subset."""
        return True


class DensePassageRetriever:
    """Serve precomputed dense rankings through the shared retrieval interface.

    Only ranked IDs and passage text reach this adapter. Gold supports, answers,
    and decompositions stay with the scorer. Query vectors are encoded locally
    before ranking, so concurrent answering does not share a model forward pass.
    """

    name = "dense-bge-small"

    def __init__(self, passages: Sequence[Passage], rankings: Mapping[str, Sequence[str]]) -> None:
        self.passages = {p.passage_id: p for p in passages}
        if len(self.passages) != len(passages):
            raise ValueError("duplicate corpus passage IDs")
        self.rankings = {question: tuple(ids) for question, ids in rankings.items()}
        for ids in self.rankings.values():
            if len(set(ids)) != len(ids) or not set(ids) <= self.passages.keys():
                raise ValueError("rankings contain duplicate or unknown passage IDs")

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return ranked corpus passages; reject document scope for this baseline."""
        if scope is not None:
            raise ScopeUnsupported("MuSiQue baseline searches the entire corpus")
        if k < 1:
            raise ValueError("k must be positive")
        ranking = self.rankings[question]
        if k > len(ranking):
            raise ValueError("requested depth exceeds the precomputed ranking")
        return [
            Evidence(
                text=passage_text(self.passages[pid]),
                provenance=Provenance(doc_id=pid, chunk_id=pid),
                retriever=self.name,
            )
            for pid in ranking[:k]
        ]

    def __call__(self, question: str, scope: str | None = None, k: int = 10) -> list[Evidence]:
        """Implement the shared spine's callable retrieval contract."""
        return self.retrieve(question, k, scope=scope)

    async def aretrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Expose the same ranked sources through the asynchronous interface."""
        return self.retrieve(question, k, scope=scope, hints=hints)


def rank_questions(
    passages: Sequence[Passage],
    questions: Sequence[MusiqueQuestion],
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> dict[str, list[str]]:
    """Rank every original question against the full corpus with shared DenseIndex."""
    if k < 1 or k > len(passages):
        raise ValueError("k must be within the corpus size")
    if (
        document_vectors.ndim != 2
        or query_vectors.ndim != 2
        or document_vectors.shape[0] != len(passages)
        or query_vectors.shape != (len(questions), document_vectors.shape[1])
    ):
        raise ValueError("vector shapes do not match passages and questions")
    if not np.isfinite(document_vectors).all() or not np.isfinite(query_vectors).all():
        raise ValueError("vectors must be finite")
    index = DenseIndex([p.passage_id for p in passages], document_vectors)
    rankings = index.search(query_vectors, k=k)
    return {q.question_id: ids for q, ids in zip(questions, rankings, strict=True)}


def score_by_hops(
    questions: Sequence[MusiqueQuestion],
    runs: Mapping[str, Sequence[str]],
    depths: Sequence[int],
    predictions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score all selected questions, overall and by hop count, using shared metrics."""
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("questions must be non-empty and unique")
    expected = {q.question_id for q in questions}
    if set(runs) != expected or (predictions is not None and set(predictions) != expected):
        raise ValueError("runs and predictions must cover exactly the selected questions")
    if not depths or min(depths) < 1 or any(len(ids) < max(depths) for ids in runs.values()):
        raise ValueError("rankings must cover every positive scoring depth")
    groups = {"overall": list(questions)}
    groups.update({str(h): [q for q in questions if q.hops == h] for h in sorted({q.hops for q in questions})})
    report = {}
    for label, selected in groups.items():
        selected_runs = {q.question_id: runs[q.question_id] for q in selected}
        qrels = {q.question_id: set(q.supporting_ids) for q in selected}
        metrics: dict[str, Any] = {"questions": len(selected), "retrieval": {}}
        for k in depths:
            retrieval = score_retrieval(selected_runs, qrels, k=k).to_dict()
            retrieval[f"all_supports_at_{k}"] = all_gold_recall_at_k(selected_runs, qrels, k)
            metrics["retrieval"][str(k)] = retrieval
        if predictions is not None:
            metrics["answers"] = {
                "exact_match": mean(
                    exact_match(predictions[q.question_id], (q.answer, *q.answer_aliases)) for q in selected
                ),
                "token_f1": mean(token_f1(predictions[q.question_id], (q.answer, *q.answer_aliases)) for q in selected),
            }
        report[label] = metrics
    return report


def answer_baseline(
    questions: Sequence[MusiqueQuestion],
    retriever: DensePassageRetriever,
    complete: Callable[[str], str],
    meter: UsageMeter,
    evidence_budget: int = 5,
) -> ArmResult:
    """Run single-pass generation in Experiment, with no corrective retries.

    The heuristic grade is recorded only: SingleShotPolicy always generates from
    available evidence. No MuSiQue quality claim is made about that grade.
    """
    experiment = Experiment(
        queries=[HarnessQuestion(q.question_id, q.question) for q in questions],
        qrels={q.question_id: set(q.supporting_ids) for q in questions},
        retrieve=retriever,
        grader=HeuristicGrader(),
        complete=complete,
        k=evidence_budget,
        spine={
            "config": SpineConfig(top_k=evidence_budget, evidence_budget=evidence_budget, max_attempts=1),
            "policy": SingleShotPolicy(),
            "template": ANSWER_TEMPLATE,
            "max_chars": None,
        },
    )
    return experiment.run("dense-single-pass", meter=meter)
