"""Verified, read-only graph handoffs; index construction belongs to RI #4."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import CacheKey, TieredResultCache, fetch_directory, tree_digest
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.pipeline.answering import SingleShotPolicy
from aieng.advanced_rag.pipeline.grading import HeuristicGrader
from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig, SpineResult
from aieng.advanced_rag.retrieval.interface import ScopeUnsupported

from .corpora import Corpus
from .dense_baseline import ANSWER_TEMPLATE, DensePassageRetriever
from .graph_release import native_metadata, verify_release


class PendingGraphArtifact(RuntimeError):  # noqa: N818 - dependency status
    """The verified producer handoff is unavailable; never rebuild implicitly."""


def resolve_index(
    cache: TieredResultCache, key: CacheKey, corpus: Corpus, dest: Path, *, allow_incomplete: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Verify the archive, full passage map, corpus pin, and producer settings."""
    if allow_incomplete and not (
        key.ri_id == "ri4" and key.artifact_type == "graph-index" and key.producer_id == "lightrag"
    ):
        raise ValueError("incomplete-index mode requires the pinned LightRAG release")
    artifact = cache.get_artifact(key)
    if artifact is None:
        raise PendingGraphArtifact("RI #4 prebuilt index is absent from the artifact cache")
    released = (
        key.ri_id == "ri4" and key.artifact_type == "graph-index" and key.producer_id in {"hipporag2", "lightrag"}
    )
    entry = verify_release(artifact, key, corpus) if released else None
    directory: Path | None
    if dest.exists():
        if not isinstance(artifact.metadata, dict) or tree_digest(dest) != artifact.metadata.get("tree_digest"):
            raise ValueError("unpacked graph index differs from its artifact")
        directory = dest
    else:
        directory = fetch_directory(cache, key, dest)
    if directory is None:
        raise PendingGraphArtifact("graph directory payload is unavailable")
    if entry is not None:
        return directory, native_metadata(directory, key, corpus, entry, allow_incomplete=allow_incomplete)
    return directory, handoff_metadata(directory, corpus)


def handoff_metadata(directory: Path, corpus: Corpus) -> dict[str, Any]:
    """Validate explicitly versioned custom artifacts using their in-archive record."""
    handoff = directory / "ri2_handoff.json"
    if not handoff.exists():
        raise PendingGraphArtifact("use the pinned RI4 release or an explicit verified handoff")
    metadata: dict[str, Any] = json.loads(handoff.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("graph artifact lacks provenance metadata")
    if (
        corpus.name != "musique"
        or metadata.get("corpus_revision") != corpus.provenance["revision"]
        or metadata.get("corpus_sha256") != corpus.provenance["corpus_sha256"]
        or metadata.get("passage_ids") != [p.passage_id for p in corpus.passages]
    ):
        raise ValueError("graph index does not match the complete pinned corpus")
    if metadata.get("backend") not in {"hipporag2", "lightrag", "atlas-rag"} or not isinstance(
        metadata.get("settings"), dict
    ):
        raise ValueError("graph artifact needs an explicit backend and retrieval settings")
    if not metadata.get("producer_revision"):
        raise ValueError("graph producer revision is missing")
    mapping = metadata.get("retriever_id_map", {})
    known = {p.passage_id for p in corpus.passages}
    if (
        not isinstance(mapping, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())
        or not set(mapping.values()) <= known
    ):
        raise ValueError("graph passage mapping must contain known original passage IDs")
    if metadata["backend"] == "hipporag2" and metadata["settings"].get("implementation") != "hipporag2-paper":
        raise ValueError("paper HippoRAG 2 label requires its own producer configuration")
    return metadata


class GraphAnswerer:
    """Use an injected RI #4 ranking arm with the same cited generation budget."""

    def __init__(self, passages: Sequence[Passage], rank: Callable[..., Any], complete: Callable[[str], str]) -> None:
        self.passages = passages
        self.known = {p.passage_id for p in passages}
        self.rank, self.complete = rank, complete
        self.traces: dict[str, dict[str, Any]] = {}

    def answer(self, query_id: str, question: str, scope: str | None = None) -> SpineResult:
        """Run the shared synchronous Answerer interface outside an event loop."""
        return asyncio.run(self.aanswer(query_id, question, scope))

    async def aanswer(self, query_id: str, question: str, scope: str | None = None) -> SpineResult:
        """Resolve graph hits to original corpus passages before generation."""
        if scope is not None:
            raise ScopeUnsupported("MuSiQue graph searches the complete corpus")
        ranking: Sequence[str] = await self.rank(question, 20)
        if len(set(ranking)) != len(ranking) or not set(ranking) <= self.known:
            raise ValueError("graph returned duplicate or unknown passage IDs")
        if not ranking:
            from aieng.advanced_rag.pipeline.generate import GroundedAnswer  # noqa: PLC0415

            result = SpineResult(query_id, question, GroundedAnswer.abstain(query_id, "graph_no_evidence"))
        else:
            spine = AdvancedRagSpine(
                DensePassageRetriever(self.passages, {question: ranking}),
                HeuristicGrader(),
                self.complete,
                config=SpineConfig(top_k=len(ranking), evidence_budget=5, max_attempts=1),
                policy=SingleShotPolicy(),
                template=ANSWER_TEMPLATE,
                max_chars=None,
            )
            result = await asyncio.to_thread(spine.answer, query_id, question)
        self.traces[query_id] = {
            "retrieved_ids": list(ranking),
            "generation_ids": list(ranking[:5]),
            "stop_reason": result.answer.reason or "answered",
            "raw_answer": result.answer.raw,
        }
        return result
