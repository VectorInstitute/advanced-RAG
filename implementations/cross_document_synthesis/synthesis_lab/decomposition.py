"""Bounded decomposition behind Answerer, with local retrieval and pooled evidence."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.pipeline.answering import DecompositionAnalysis, SingleShotPolicy
from aieng.advanced_rag.pipeline.grading import HeuristicGrader, first_json_object
from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig, SpineResult
from aieng.advanced_rag.retrieval.index import DenseIndex
from aieng.advanced_rag.retrieval.interface import ScopeUnsupported

from .dense_baseline import ANSWER_TEMPLATE, DensePassageRetriever


DECOMPOSITION_TEMPLATE = """Split the question into two to four short evidence-search questions.
Return only JSON: {{"subquestions": ["question", "question"]}}.
Each question must be standalone, at most 20 words, and use only entities and
relations in the original question. Never invent an intermediate answer.
For an unknown entity, retain its descriptive phrase so retrieval can find it.
Do not use numbered placeholders such as #1 or refer to another subquestion.
If splitting would not help, return {{"subquestions": []}}.
Do not answer the question or explain your reasoning.

Question: {question}
"""


class ModelDecomposer:
    """One completion per plan; malformed plans fall back explicitly to the original."""

    def __init__(self, complete: Callable[[str], str]) -> None:
        self.complete = complete
        self.outcome = ""
        self.raw = ""

    def __call__(self, question: str) -> list[str]:
        """Return validated standalone queries without access to corpus annotations."""
        self.raw = self.complete(DECOMPOSITION_TEMPLATE.format(question=question))
        payload = first_json_object(self.raw)
        parts = payload.get("subquestions") if payload is not None else None
        self.outcome = "invalid-plan"
        if not isinstance(parts, list):
            return []
        if not parts:
            self.outcome = "declined"
            return []
        if not 2 <= len(parts) <= 4 or any(
            not isinstance(part, str) or not part.strip() or len(part.split()) > 40 or re.search(r"#\s*\d+", part)
            for part in parts
        ):
            return []
        unique = list(dict.fromkeys(part.strip() for part in parts))
        if len(unique) < 2:
            return []
        self.outcome = "decomposed"
        return unique


def pool_rankings(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Interleave by rank and query order, deduplicating before the evidence budget."""
    if k < 1:
        raise ValueError("k must be positive")
    pooled: dict[str, None] = {}
    for rank in range(max((len(ids) for ids in rankings), default=0)):
        for ids in rankings:
            if rank < len(ids):
                pooled.setdefault(ids[rank], None)
                if len(pooled) == k:
                    return list(pooled)
    return list(pooled)


class DecompositionAnswerer:
    """Implement the shared Answerer contract without changing the default spine.

    Shared DecompositionAnalysis produces the plan. Retrieval pools each plan
    step's ranking plus the original question before the shared spine spends its
    five-passage generation budget. No intermediate answers or corrective loop
    run here; evidence-dependent query substitution belongs to IRCoT.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        vectors: np.ndarray,
        encode_queries: Callable[[Sequence[str]], list[list[float]]],
        decompose: Callable[[str], str],
        complete: Callable[[str], str],
        *,
        retrieval_depth: int = 20,
        evidence_budget: int = 5,
    ) -> None:
        if not 1 <= evidence_budget <= retrieval_depth <= len(passages):
            raise ValueError("budgets must satisfy 1 <= evidence <= retrieval <= corpus size")
        if vectors.ndim != 2 or len(vectors) != len(passages) or not np.isfinite(vectors).all():
            raise ValueError("document vectors must be finite and match the corpus")
        if len({p.passage_id for p in passages}) != len(passages):
            raise ValueError("duplicate passage IDs")
        self.passages = passages
        self.index = DenseIndex([p.passage_id for p in passages], vectors)
        self.encode_queries = encode_queries
        self.decomposer = ModelDecomposer(decompose)
        self.analysis = DecompositionAnalysis(self.decomposer, max_steps=4)
        self.complete = complete
        self.retrieval_depth = retrieval_depth
        self.evidence_budget = evidence_budget
        self.traces: dict[str, dict[str, Any]] = {}

    def answer(self, query_id: str, question: str, scope: str | None = None) -> SpineResult:
        """Plan once, search locally, and generate once for the original question."""
        if scope is not None:
            raise ScopeUnsupported("MuSiQue decomposition searches the entire corpus")
        plan = self.analysis.analyse(question)
        queries = list(dict.fromkeys([step.question for step in plan.retrieval_steps()] + [question]))
        vectors = np.asarray(self.encode_queries(queries), dtype=np.float32)
        if vectors.shape != (len(queries), self.index.dimension) or not np.isfinite(vectors).all():
            raise ValueError("query vectors must be finite and match the plan and index")
        rankings = self.index.search(vectors, k=self.retrieval_depth)
        pooled = pool_rankings(rankings, self.retrieval_depth)
        retriever = DensePassageRetriever(self.passages, {question: pooled})
        spine = AdvancedRagSpine(
            retrieve=retriever,
            grader=HeuristicGrader(),
            complete=self.complete,
            config=SpineConfig(top_k=self.retrieval_depth, evidence_budget=self.evidence_budget, max_attempts=1),
            policy=SingleShotPolicy(),
            template=ANSWER_TEMPLATE,
            max_chars=None,
        )
        result = spine.answer(query_id, question)
        self.traces[query_id] = {
            "plan": asdict(plan),
            "plan_outcome": self.decomposer.outcome,
            "raw_plan": self.decomposer.raw,
            "retrieval_queries": queries,
            "step_rankings": rankings,
            "pooled_ids": pooled,
            "generation_ids": pooled[: self.evidence_budget],
            "result": result.to_dict(),
            "raw_answer": result.answer.raw,
        }
        return result
