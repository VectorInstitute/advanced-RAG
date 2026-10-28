"""Pinned local embeddings, Vector proxy completions, and experiment IO."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from aieng.advanced_rag.batch import KeyFunction, ProxyBatch
from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_CORPUS,
    MUSIQUE_QUESTIONS,
    MUSIQUE_REVISION,
    MusiqueQuestion,
    Passage,
)
from aieng.advanced_rag.graphrag.proxy import ResponseCache
from aieng.advanced_rag.models import REFERENCE_MODELS, request_params
from aieng.advanced_rag.proxy import resolve_base_url
from aieng.advanced_rag.retrieval.embedders import BGE_QUERY, SentenceTransformerEmbedder

from .dense_baseline import passage_text


MODEL = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PROXY_URL = "https://proxy.vectorinstitute.ai/v1"
ANSWER_MODEL = REFERENCE_MODELS.generation.model  # Model used by the recorded reference audits.
LAB_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = LAB_DIR.parents[1]
DEFAULT_CACHE = REPOSITORY_ROOT / "Arm Store/musique-dense"


def write_json(path: Path, payload: Any) -> None:
    """Write a complete JSON file via an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def answer_key(model: str, base_url: str) -> KeyFunction:
    """Return the cache key function the recorded results were written with.

    The committed results record these keys as ``cache_key``, and
    ``benchmarks/full_evaluation.py`` rebuilds them to audit each stored
    answer, so the shape does not change.

    Parameters
    ----------
    model : str
        The model the key is for.
    base_url : str
        The proxy endpoint, which the key names.

    Returns
    -------
    callable
        ``key(messages, params) -> str``, as :class:`ProxyBatch` takes it.
    """

    def key(messages: list[dict[str, Any]], params: Mapping[str, Any]) -> str:
        max_tokens = params["max_tokens"]
        request = {
            "prompt": messages[-1]["content"],
            "model": model,
            "base_url": base_url,
            "request_params": request_params(model, max_tokens=max_tokens, temperature=0, thinking=False),
            "temperature": 0,
            "max_tokens": max_tokens,
            "thinking": False,
        }
        if params.get("reasoning_effort") is not None:
            request["reasoning_effort"] = params["reasoning_effort"]
        digest: str = ResponseCache.key(request)
        return digest

    return key


def answer_batch(cache_dir: Path, model: str) -> ProxyBatch:
    """Return the proxy connection and response cache for ``model``.

    Share one among every completer on a model so they share its rate limit.
    """
    base_url = resolve_base_url()
    return ProxyBatch(
        model,
        cache=ResponseCache(cache_dir / "answers.sqlite"),
        key=answer_key(model, base_url),
        payload="text",
        base_url=base_url,
        timeout=90,
    )


class CachedProxyCompletion:
    """Answer prompts from the run's response cache, or live through the proxy.

    Calls go through :class:`aieng.advanced_rag.batch.ProxyBatch`, which
    answers from the cache first, holds calls to the rate limit, retries only
    transient errors, switches reasoning off, and reports each call to the
    meter and the spend guard. The cache keys come from :func:`answer_key`.

    Parameters
    ----------
    cache_dir : Path
        Directory holding ``answers.sqlite``.
    model : str
        Model name to send.
    max_tokens : int, optional
        Completion cap.
    reasoning_effort : str or None, optional
        Sent with each request and recorded in its key when set.
    batch : ProxyBatch or None, optional
        The connection from :func:`answer_batch`, to share one rate limit
        across completers. One is created for ``model`` when omitted.

    Raises
    ------
    RuntimeError
        From a call whose response has no content. An empty answer is never
        cached or scored.
    """

    def __init__(
        self,
        cache_dir: Path,
        model: str,
        *,
        max_tokens: int = 256,
        reasoning_effort: str | None = None,
        batch: ProxyBatch | None = None,
    ) -> None:
        self.batch = batch if batch is not None else answer_batch(cache_dir, model)
        self.base_url = resolve_base_url()
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.cache = self.batch.cache
        self.completed = 0
        self.requests: list[dict[str, Any]] = []

    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": 0}
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def key(self, prompt: str) -> str:
        """Return the cache key for ``prompt``, as the committed results record it."""
        key: str = answer_key(self.model, self.base_url)([{"role": "user", "content": prompt}], self._params())
        return key

    def __call__(self, prompt: str) -> str:
        """Return a cached or live completion."""
        outcome = self.batch.create(prompt, **self._params())
        text = outcome.text or ""
        if not text:
            raise RuntimeError("Vector proxy returned no answer content")
        self.requests.append({"cache_key": self.key(prompt), "cache_hit": outcome.cached})
        self.completed += 1
        if self.completed % 25 == 0:
            print(f"Completed {self.completed} model requests", flush=True)
        return text


def local_embedder(cache_dir: Path) -> SentenceTransformerEmbedder:
    """Load the pinned BGE model for both original questions and agent subqueries."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - local extra

    snapshot = snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        cache_dir=cache_dir / "models",
        allow_patterns=["*.json", "model.safetensors", "vocab.txt", "1_Pooling/*"],
    )
    return SentenceTransformerEmbedder(
        snapshot, device="cpu", query_prompt=BGE_QUERY, document_prompt="", batch_size=32
    )


def embeddings(
    cache_dir: Path,
    passages: Sequence[Passage],
    questions: Sequence[MusiqueQuestion],
    *,
    embedder: SentenceTransformerEmbedder | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or verify pinned local vectors and record actual encoder settings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder if embedder is not None else local_embedder(cache_dir)
    description = embedder.describe() | {"hf_id": MODEL, "label": "bge-small-en-v1.5", "revision": MODEL_REVISION}
    identity = {
        "corpus_revision": MUSIQUE_REVISION,
        "corpus_sha256": MUSIQUE_CORPUS.sha256,
        "questions_sha256": MUSIQUE_QUESTIONS.sha256,
        "encoder": description,
        "passage_format": "title + newline + text",
        "normalised": True,
        "dtype": "float32",
    }
    vectors_path = cache_dir / "embeddings.npz"
    manifest_path = cache_dir / "embeddings.json"
    if vectors_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("embedding cache configuration differs; use a new --cache-dir")
        if hashlib.sha256(vectors_path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("embedding cache checksum mismatch")
        with np.load(vectors_path, allow_pickle=False) as stored:
            docs, queries = stored["documents"], stored["queries"]
        return docs, queries, manifest | {"cache_hit": True}
    print(f"Embedding {len(passages)} passages locally", flush=True)
    started = time.perf_counter()
    docs = np.asarray(embedder.encode([passage_text(p) for p in passages]), dtype=np.float32)
    document_seconds = time.perf_counter() - started
    print(f"Embedding {len(questions)} questions locally", flush=True)
    started = time.perf_counter()
    queries = np.asarray(embedder.encode_queries([q.question for q in questions]), dtype=np.float32)
    query_seconds = time.perf_counter() - started
    np.savez(vectors_path, documents=docs, queries=queries)
    manifest = {
        "identity": identity,
        "sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "document_embedding_seconds": document_seconds,
        "batch_query_embedding_seconds": query_seconds,
        "document_vector_bytes": docs.nbytes,
        "query_vector_bytes": queries.nbytes,
    }
    write_json(manifest_path, manifest)
    return docs, queries, manifest | {"cache_hit": False}
