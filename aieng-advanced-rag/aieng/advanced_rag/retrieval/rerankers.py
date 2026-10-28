"""Cross-encoder reranking over a first-stage retriever, on CPU.

A reranker is the cheapest quality lever in a pipeline that has already chosen
its retriever: it does not change what can be found, only what reaches the
reader. That makes retrieval-failure rate at a fixed k the right way to score
it, which is the metric ``evaluation.metrics`` already produces.

Two constraints shape the candidates. A reranker has to run in the bootcamp
container, which is CPU, so model size is latency. And it has to be local:
Cohere's reranker is an API that bills per search, and an exercise that needs a
paid key participants do not have is a worse exercise.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankReport:
    """Latency and throughput for one reranker over one workload.

    Parameters
    ----------
    model : str
        The checkpoint that produced these numbers.
    queries : int
        Queries reranked.
    pairs : int
        Query-candidate pairs scored.
    seconds : float
        Wall time spent scoring.
    load_seconds : float
        Wall time spent loading the checkpoint.
    parameters : int or None
        Parameter count, when the module exposes one.
    """

    model: str
    queries: int = 0
    pairs: int = 0
    seconds: float = 0.0
    load_seconds: float = 0.0
    parameters: int | None = None

    @property
    def pairs_per_second(self) -> float:
        """Return scoring throughput, which decides whether it fits a lab."""
        return self.pairs / self.seconds if self.seconds else 0.0

    @property
    def ms_per_query(self) -> float:
        """Return the added latency per query at this candidate depth."""
        return 1000 * self.seconds / self.queries if self.queries else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "model": self.model,
            "queries": self.queries,
            "pairs": self.pairs,
            "seconds": round(self.seconds, 1),
            "load_seconds": round(self.load_seconds, 1),
            "parameters": self.parameters,
            "pairs_per_second": round(self.pairs_per_second, 1),
            "ms_per_query": round(self.ms_per_query, 1),
        }


class CrossEncoderReranker:
    """A local cross-encoder that reorders first-stage candidates.

    Parameters
    ----------
    model_name : str
        A sentence-transformers ``CrossEncoder`` checkpoint.
    device : str, optional
        Kept as ``cpu``: the bootcamp container has no GPU, and a number
        measured on one would not transfer.
    batch_size : int, optional
        Pairs scored per forward pass.
    trust_remote_code : bool, optional
        Required by some checkpoints, notably Jina's. Worth knowing before it
        goes in front of participants, since it executes code from the hub.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        trust_remote_code: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        began = time.perf_counter()
        kwargs: dict[str, Any] = {"device": device}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        self._model = CrossEncoder(model_name, **kwargs)
        self.load_seconds = time.perf_counter() - began
        self.report = RerankReport(
            model=model_name,
            load_seconds=self.load_seconds,
            parameters=self._count_parameters(),
        )

    def _count_parameters(self) -> int | None:
        module = getattr(self._model, "model", None)
        if module is None or not hasattr(module, "parameters"):
            return None
        return int(sum(p.numel() for p in module.parameters()))

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score one query against candidate texts."""
        if not texts:
            return []
        began = time.perf_counter()
        scores = self._model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        self.report.seconds += time.perf_counter() - began
        self.report.pairs += len(texts)
        self.report.queries += 1
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        k: int,
    ) -> list[str]:
        """Reorder ``(chunk_id, text)`` candidates and return the top ``k`` ids.

        Ties break on the identifier, so a run is reproducible.
        """
        if not candidates:
            return []
        scores = self.score(query, [text for _, text in candidates])
        ranked = sorted(
            zip(scores, [chunk_id for chunk_id, _ in candidates]),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [chunk_id for _, chunk_id in ranked[:k]]


@dataclass
class Candidate:
    """One reranker under consideration, with why it is here.

    Parameters
    ----------
    name : str
        Short label for a report.
    model : str
        The checkpoint.
    note : str
        Why this candidate is worth measuring.
    trust_remote_code : bool, optional
        Whether loading it executes code from the hub.
    tags : tuple of str, optional
        Free-form markers, such as ``cpu`` or ``remote-code``.
    """

    name: str
    model: str
    note: str
    trust_remote_code: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="MiniLM-L6",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        note="The small, fast baseline. If a larger model cannot beat this by "
        "more than it costs, the larger model is not worth the container.",
        tags=("cpu", "small"),
    ),
    Candidate(
        name="BGE reranker base",
        model="BAAI/bge-reranker-base",
        note="The middle of the BGE family, and the same lineage as the "
        "embedding model already used for first-stage retrieval.",
        tags=("cpu",),
    ),
    Candidate(
        name="BGE reranker v2-m3",
        model="BAAI/bge-reranker-v2-m3",
        note="Large and multilingual; check whether its size is affordable on CPU.",
        tags=("cpu", "large"),
    ),
    Candidate(
        name="Jina reranker v2",
        model="jinaai/jina-reranker-v2-base-multilingual",
        note="The third option worth checking. Needs trust_remote_code, which "
        "is a supply-chain decision as much as a quality one.",
        trust_remote_code=True,
        tags=("cpu", "remote-code"),
    ),
)

# Named and excluded rather than silently skipped: Cohere Rerank is an API that
# bills per search, and a bootcamp exercise that needs a paid key participants
# do not have is a worse exercise.
EXCLUDED = {
    "Cohere Rerank": (
        "API only, billed per search, needs a key participants do not have"
    ),
}
