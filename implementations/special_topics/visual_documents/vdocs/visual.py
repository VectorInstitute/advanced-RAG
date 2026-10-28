"""Embed page images with ColModernVBERT and search them by late interaction.

ColModernVBERT writes one 128-dimensional vector per image patch, about 1,150
for a US letter page rendered at :data:`~.pages.PAGE_SCALE`, and one per query
token. A page scores the sum, over query tokens, of each token's best cosine
match among the page's vectors (MaxSim). The search here is exact, in numpy,
because a thousand pages is well inside exact-scan range.

The model is fastembed's ONNX export. The index stores vectors in float16 to
halve its size; the search casts back to float32.

On a CPU, a page takes tens of seconds to embed and so does a query, which is
why the reference index and the reference question embeddings ship through the
artifact cache. With ``onnxruntime-gpu`` installed, the same export runs on a
GPU through ``providers=["CUDAExecutionProvider"]`` and writes the same
vectors to within float precision.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from aieng.advanced_rag.cache.paths import root as cache_root


Array = npt.NDArray[Any]


#: fastembed's ONNX export of ModernVBERT's late-interaction model.
MODEL = "Qdrant/colmodernvbert"
DIM = 128


def model_cache() -> Path:
    """Return where the model weights are downloaded, about 1 GB."""
    return Path(cache_root()) / "models" / "fastembed"


def load_model(providers: Sequence[str] | None = None, threads: int | None = None) -> Any:
    """Load ColModernVBERT through fastembed, on the given ONNX providers."""
    from fastembed import LateInteractionMultimodalEmbedding  # noqa: PLC0415

    kwargs: dict[str, Any] = {"cache_dir": str(model_cache())}
    if providers:
        kwargs["providers"] = list(providers)
    if threads:
        kwargs["threads"] = threads
    return LateInteractionMultimodalEmbedding(MODEL, **kwargs)


def _normalise(matrix: Array) -> Array:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised: Array = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalised


def embed_images(model: Any, images: Sequence[Any], batch_size: int = 1) -> tuple[list[Array], float]:
    """Embed page images, returning one matrix per page and the seconds per page."""
    started = time.perf_counter()
    vectors = [
        _normalise(np.asarray(v, dtype=np.float32)) for v in model.embed_image(list(images), batch_size=batch_size)
    ]
    seconds = (time.perf_counter() - started) / max(1, len(vectors))
    return vectors, seconds


def embed_queries(model: Any, texts: Sequence[str], batch_size: int = 8) -> tuple[list[Array], float]:
    """Embed query texts, returning one matrix per query and the seconds per query."""
    started = time.perf_counter()
    vectors = [
        _normalise(np.asarray(v, dtype=np.float32)) for v in model.embed_text(list(texts), batch_size=batch_size)
    ]
    seconds = (time.perf_counter() - started) / max(1, len(vectors))
    return vectors, seconds


@dataclass
class PageIndex:
    """Page multivectors, stored end to end with an offset per page.

    Parameters
    ----------
    page_ids : list of str
        The pages, in storage order.
    vectors : numpy.ndarray
        Every page's vectors, shape ``(total, 128)``, float16 on disk.
    offsets : numpy.ndarray
        Start row of each page, with the total appended, shape ``(pages + 1,)``.
    meta : dict
        Model, render scale, hardware, and seconds per page.
    """

    page_ids: list[str]
    vectors: Array
    offsets: Array
    meta: dict[str, Any]

    @classmethod
    def build(cls, page_ids: Sequence[str], matrices: Sequence[Array], meta: dict[str, Any]) -> PageIndex:
        """Stack per-page matrices into one index."""
        sizes = [m.shape[0] for m in matrices]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        stacked = np.concatenate(list(matrices), axis=0).astype(np.float16)
        return cls(list(page_ids), stacked, offsets, dict(meta))

    def save(self, directory: Path) -> None:
        """Write the index as ``vectors.npy``, ``offsets.npy``, and ``index.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        np.save(directory / "offsets.npy", self.offsets)
        (directory / "index.json").write_text(
            json.dumps({"page_ids": self.page_ids, "meta": self.meta}, indent=1) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> PageIndex:
        """Read an index :meth:`save` wrote."""
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        return cls(
            list(record["page_ids"]),
            np.load(directory / "vectors.npy"),
            np.load(directory / "offsets.npy"),
            dict(record["meta"]),
        )

    @property
    def megabytes(self) -> float:
        """Return the vectors' size in megabytes."""
        return round(self.vectors.nbytes / 1e6, 1)

    def search(self, query: Array, k: int = 10) -> list[tuple[str, float]]:
        """Return the top ``k`` pages for one query matrix, by MaxSim."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[Array], k: int = 10, block: int = 8192) -> list[list[tuple[str, float]]]:
        """Return the top ``k`` pages for each query matrix, by MaxSim.

        Every query's tokens are scored against a block of page vectors at once,
        so the float16 vectors are cast to float32 one block at a time.
        """
        stacked = np.concatenate(list(queries), axis=0).astype(np.float32)
        bounds = np.concatenate([[0], np.cumsum([q.shape[0] for q in queries])])
        best = np.full((len(self.page_ids), stacked.shape[0]), -np.inf, dtype=np.float32)
        owner = np.repeat(np.arange(len(self.page_ids)), np.diff(self.offsets))
        for start in range(0, self.vectors.shape[0], block):
            stop = min(start + block, self.vectors.shape[0])
            scores = self.vectors[start:stop].astype(np.float32) @ stacked.T
            pages = owner[start:stop]
            # Rows of one page are contiguous, so the block splits into runs.
            cuts = np.flatnonzero(np.diff(pages)) + 1
            firsts = np.concatenate([[0], cuts])
            maxima = np.maximum.reduceat(scores, firsts, axis=0)
            run_pages = pages[firsts]
            best[run_pages] = np.maximum(best[run_pages], maxima)
        results: list[list[tuple[str, float]]] = []
        for index in range(len(queries)):
            totals = best[:, bounds[index] : bounds[index + 1]].sum(axis=1)
            top = np.argsort(-totals, kind="stable")[:k]
            results.append([(self.page_ids[i], float(totals[i])) for i in top])
        return results


def build_index(
    page_ids: Sequence[str],
    render: Callable[[str], Any],
    model: Any,
    *,
    meta: dict[str, Any],
    batch: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> PageIndex:
    """Render and embed every page, timing the embedding alone.

    Parameters
    ----------
    page_ids : sequence of str
        The pages.
    render : callable
        Page identifier to a ``PIL.Image``.
    model : Any
        From :func:`load_model`.
    meta : dict
        Recorded with the index; the measured seconds per page are added.
    batch : int, optional
        Pages rendered and embedded together.
    progress : callable or None, optional
        Called with pages done and pages in total.

    Returns
    -------
    PageIndex
        The index.
    """
    matrices: list[Array] = []
    seconds = 0.0
    for start in range(0, len(page_ids), batch):
        images = [render(p) for p in page_ids[start : start + batch]]
        vectors, per_page = embed_images(model, images)
        matrices.extend(vectors)
        seconds += per_page * len(images)
        if progress:
            progress(len(matrices), len(page_ids))
    meta = {**meta, "model": MODEL, "seconds_per_page": round(seconds / max(1, len(page_ids)), 3)}
    return PageIndex.build(page_ids, matrices, meta)


def save_queries(directory: Path, query_ids: Sequence[str], matrices: Sequence[Array], meta: dict[str, Any]) -> None:
    """Write query matrices as one ``queries.npz`` with ``queries.json`` beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {f"q{i}": m.astype(np.float16) for i, m in enumerate(matrices)}
    np.savez_compressed(directory / "queries.npz", **arrays)
    (directory / "queries.json").write_text(
        json.dumps({"query_ids": list(query_ids), "meta": meta}, indent=1) + "\n",
        encoding="utf-8",
    )


def load_queries(directory: Path) -> tuple[dict[str, Array], dict[str, Any]]:
    """Read query matrices :func:`save_queries` wrote, keyed by question identifier."""
    record = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    with np.load(directory / "queries.npz") as stored:
        matrices = {qid: stored[f"q{i}"].astype(np.float32) for i, qid in enumerate(record["query_ids"])}
    return matrices, dict(record["meta"])


__all__ = [
    "DIM",
    "MODEL",
    "PageIndex",
    "build_index",
    "embed_images",
    "embed_queries",
    "load_model",
    "load_queries",
    "model_cache",
    "save_queries",
]
