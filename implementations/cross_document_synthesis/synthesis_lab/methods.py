"""Common cited generation and the optional RI #1 corrective policy comparison."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline.answering import CorrectivePolicy, SingleShotPolicy
from aieng.advanced_rag.pipeline.grading import GradingThresholds, HeuristicGrader, LLMGrader
from aieng.advanced_rag.pipeline.rewrite import LLMRewriter
from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig, SpineResult
from aieng.advanced_rag.retrieval.index import DenseIndex
from aieng.advanced_rag.retrieval.interface import ScopeUnsupported

from .dense_baseline import ANSWER_TEMPLATE, DensePassageRetriever


class DenseAnswerer:
    """Search locally; use the same spine for the single-pass and corrective arms."""

    def __init__(
        self,
        passages: Sequence[Passage],
        vectors: np.ndarray,
        encode: Callable[[Sequence[str]], list[list[float]]],
        complete: Callable[[str], str],
        *,
        corrective: bool = False,
        graded_control: bool = False,
        grade: Callable[[str], str] | None = None,
        rewrite: Callable[[str], str] | None = None,
    ) -> None:
        self.passages, self.encode = passages, encode
        self.index = DenseIndex([p.passage_id for p in passages], vectors)
        self.traces: dict[str, dict[str, Any]] = {}
        thresholds = GradingThresholds(sufficient=0.6, insufficient=0.0)
        self.spine = AdvancedRagSpine(
            self.retrieve,
            LLMGrader(grade if grade is not None else complete, thresholds=thresholds, max_chunks=5, max_chars=None)
            if corrective or graded_control
            else HeuristicGrader(),
            complete,
            rewriter=LLMRewriter(rewrite if rewrite is not None else complete, variants=1) if corrective else None,
            config=SpineConfig(top_k=min(20, len(passages)), evidence_budget=5, max_attempts=2 if corrective else 1),
            policy=CorrectivePolicy(thresholds) if corrective else SingleShotPolicy(),
            template=ANSWER_TEMPLATE,
            max_chars=None,
        )

    def retrieve(self, question: str, scope: str | None = None, k: int = 20) -> Sequence[Evidence]:
        """Rank a new original or corrective query without seeing annotations."""
        if scope is not None:
            raise ScopeUnsupported("dense retrieval searches the full corpus")
        vectors = np.asarray(self.encode([question]), dtype=np.float32)
        if vectors.shape != (1, self.index.dimension) or not np.isfinite(vectors).all():
            raise ValueError("query vectors must match the index")
        ranking = self.index.search(vectors, k=min(k, len(self.passages)))[0]
        return DensePassageRetriever(self.passages, {question: ranking})(question, k=len(ranking))

    def answer(self, query_id: str, question: str, scope: str | None = None) -> SpineResult:
        """Return the shared answer result and explicit generation evidence."""
        result = self.spine.answer(query_id, question, scope)
        self.traces[query_id] = {
            "generation_ids": [] if result.abstained and not result.answer.raw else result.retrieved_ids[:5],
            "raw_answer": result.answer.raw,
            "stop_reason": result.answer.reason or "answered",
        }
        return result
