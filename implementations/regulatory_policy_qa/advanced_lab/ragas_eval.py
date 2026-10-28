"""RAGAS evaluation with explicit Vector chat and CPU-only local embeddings."""

from __future__ import annotations

import asyncio
import importlib.metadata
import math
from pathlib import Path
from typing import Any

from .common import EMBEDDING, OUTPUT, completer, provenance, require_compute, write_json
from .regulatory_inputs import input_fingerprints, regulatory_inputs


def measure_ragas(*, input_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate answerables with RAGAS; null handling remains a separate report."""
    require_compute()
    source_dir = input_dir if input_dir is not None else OUTPUT
    source = regulatory_inputs(source_dir)
    rows = [{k: v for k, v in row.items() if k not in {"is_null", "query_id"}} for row in source if not row["is_null"]]
    if not rows:
        raise ValueError("RAGAS requires a nonempty answerable evaluation set")
    # Optional evaluation dependencies are unnecessary for saved-result replay.
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
    from langchain_core.outputs import Generation, LLMResult  # noqa: PLC0415
    from ragas import EvaluationDataset, evaluate  # noqa: PLC0415
    from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: PLC0415
    from ragas.llms.base import BaseRagasLLM  # noqa: PLC0415
    from ragas.metrics import Faithfulness, ResponseRelevancy, SemanticSimilarity  # noqa: PLC0415
    from ragas.run_config import RunConfig  # noqa: PLC0415

    complete = completer(max_tokens=2048, role="judging")

    class VectorJudge(BaseRagasLLM):
        """Adapt the cached, metered Vector completer to RAGAS' evaluation API."""

        def generate_text(
            self, prompt: Any, n: int = 1, temperature: float | None = None, stop: Any = None, callbacks: Any = None
        ) -> LLMResult:
            """Return cached responses under the fixed zero-temperature protocol."""
            return LLMResult(generations=[[Generation(text=complete(prompt.to_string())) for _ in range(n)]])

        async def agenerate_text(
            self, prompt: Any, n: int = 1, temperature: float | None = None, stop: Any = None, callbacks: Any = None
        ) -> LLMResult:
            """Keep synchronous model I/O out of RAGAS' event loop."""
            return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

        def is_finished(self, response: LLMResult) -> bool:
            """Reject empty generations instead of scoring a transport failure."""
            return all(g.text.strip() for group in response.generations for g in group)

    class LocalEmbeddings(BaseRagasEmbeddings):
        """Run every evaluation embedding on a local CPU model."""

        def __init__(self) -> None:
            """Load the local encoder once for all metrics."""
            super().__init__()
            self.encoder = SentenceTransformerEmbedder(EMBEDDING, device="cpu")

        def embed_query(self, text: str) -> list[float]:
            """Encode a query locally."""
            return list(self.encoder.encode_queries([text])[0])

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Encode metric documents locally; never call an embedding API."""
            return [list(vector) for vector in self.encoder.encode(texts)]

        async def aembed_query(self, text: str) -> list[float]:
            """Encode a query without blocking the evaluation event loop."""
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            """Encode evaluation documents locally in a worker thread."""
            return await asyncio.to_thread(self.embed_documents, texts)

    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), ResponseRelevancy(), SemanticSimilarity()],
        llm=VectorJudge(),
        embeddings=LocalEmbeddings(),
        raise_exceptions=True,
        run_config=RunConfig(max_workers=1, max_retries=2, timeout=240),
    )
    scores = result.to_pandas().to_dict(orient="records")
    public_scores = []
    for index, row in enumerate(scores):
        metrics = {name: float(row[name]) for name in ("faithfulness", "answer_relevancy", "semantic_similarity")}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("RAGAS returned an incomplete/nonfinite metric; no report published")
        public_scores.append({"row": index, **metrics})
    report = {
        **provenance(),
        "model": complete.model,
        "ragas_version": importlib.metadata.version("ragas"),
        "local_embedding_model": EMBEDDING,
        "hosted_embedding_calls": 0,
        "answerable_rows": len(rows),
        "input_sha256": input_fingerprints(source_dir),
        "generation_protocol": "Fixed temperature zero; repeated identical prompts reuse the response cache, not independent samples.",
        "metrics": public_scores,
        "means": {
            name: sum(r[name] for r in public_scores) / len(public_scores)
            for name in ("faithfulness", "answer_relevancy", "semantic_similarity")
        },
        "usage": complete.meter.to_dict(),
        "limitations": [
            "Synthetic references; scores are model judgments, not independently adjudicated accuracy.",
            "Null controls are excluded from answer relevance/similarity and reported in the regulatory experiment.",
        ],
    }
    write_json(OUTPUT / "ragas.json", report)
    return report
