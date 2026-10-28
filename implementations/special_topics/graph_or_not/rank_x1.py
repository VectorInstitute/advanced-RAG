#!/usr/bin/env python3
"""Rank X1's MuSiQue questions with local models; write each arm's top 10.

Needs only ``torch``, ``sentence-transformers``, ``fastembed``, ``scipy``, and
``numpy``, not the workspace package, so it runs in the bootcamp environment on a
CPU or in any environment with a GPU. ``run_graph_or_not.py x1`` writes the
inputs file (question and passage text, in the cache folder, never in the
repository) and calls this when no rankings are cached.

On a CPU the 11,656-passage SPLADE pass dominates (hours); on one GPU the whole
run takes minutes. ColBERT runs through fastembed's ONNX export, as the package
does, whenever its document cache exists (over an hour to build on a CPU); on a
GPU with no such cache it runs through :class:`gon.x1.TorchColbert` instead.

Example
-------
::

    $ python rank_x1.py --inputs <work>/x1-inputs-300.json --out
    <work>/x1-rankings-300.json --device cuda

where ``<work>`` is ``$ADVANCED_RAG_CACHE/special_topics/graph-or-not``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from gon.x1 import (  # noqa: E402
    BGE_QUERY,
    DENSE,
    LIGHT_RERANKER,
    RERANKER,
    ColbertIndex,
    SentenceTransformersSplade,
    SpladeIndex,
    StrongStack,
    TorchColbert,
    top_k,
)


def rank(
    inputs: dict[str, Any], work: Path, device: str, partial: Path | None = None, colbert_device: str = "cpu"
) -> dict[str, dict[str, list[str]]]:
    """Return every arm's top 10 per question."""
    from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: PLC0415

    ids = [p["id"] for p in inputs["passages"]]
    texts = [p["text"] for p in inputs["passages"]]
    by_id = dict(zip(ids, texts, strict=True))
    matrix = np.load(inputs["dense_matrix"]).astype(np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    dense_model = SentenceTransformer(DENSE, device=device)

    def dense_scores(query: str) -> np.ndarray:
        q = dense_model.encode([BGE_QUERY + query], normalize_embeddings=True)[0].astype(np.float32)
        return np.asarray(matrix @ q)

    splade = SpladeIndex(texts, work / f"splade-{len(ids)}.npz", SentenceTransformersSplade(device=device))
    # MaxSim scoring runs on the CPU in both cases, which keeps GPU memory for the
    # encoders.
    if device.startswith("cuda") and not (work / f"colbert-{len(ids)}.npz").is_file():
        colbert = ColbertIndex(texts, work / f"colbert-torch-{len(ids)}.npz", TorchColbert(device=device), device="cpu")
    else:
        colbert = ColbertIndex(texts, work / f"colbert-{len(ids)}.npz", device="cpu")
    import torch  # noqa: PLC0415

    half = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
    heavy = CrossEncoder(RERANKER, device=device, model_kwargs=half)
    light = CrossEncoder(LIGHT_RERANKER, device=device, model_kwargs=half)

    def rerank_with(model: CrossEncoder) -> Callable[[str, Sequence[str], int], list[str]]:
        def rerank(query: str, candidates: Sequence[str], k: int) -> list[str]:
            if not candidates:
                return []
            scores = model.predict([(query, by_id[c]) for c in candidates], batch_size=16, show_progress_bar=False)
            ordered = sorted(zip(scores.tolist(), candidates, strict=True), key=lambda x: (-x[0], x[1]))
            return [c for _, c in ordered[:k]]

        return rerank

    stack = StrongStack(
        ids,
        by_id.__getitem__,
        dense_scores=dense_scores,
        splade_scores=splade.scores,
        colbert_scores=colbert.scores,
        rerank=rerank_with(heavy),
    )
    # A partial file lets an interrupted run resume where it stopped.
    out: dict[str, dict[str, list[str]]] = {}
    if partial is not None and partial.is_file():
        out = json.loads(partial.read_text(encoding="utf-8"))
    started = time.perf_counter()
    for n, question in enumerate(inputs["questions"], start=1):
        if question["id"] in out:
            continue
        arms = stack.arms(question["text"], inputs["hippo"][question["id"]])
        candidates = arms.pop("_candidates")
        arms["hybrid3-rerank-minilm"] = rerank_with(light)(question["text"], candidates, 10)
        arms["dense"] = top_k(dense_scores(question["text"]), ids, 10)
        out[question["id"]] = arms
        if n % 25 == 0:
            print(f"ranked {n}/{len(inputs['questions'])} in {time.perf_counter() - started:.0f}s", flush=True)
            if partial is not None:
                partial.write_text(json.dumps(out), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Read the inputs file, rank, and write the rankings."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--colbert-device", default="cpu", help="where ColBERT MaxSim scoring runs")
    args = parser.parse_args(argv)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    partial = args.out.with_suffix(".partial.json")
    rankings = rank(inputs, args.inputs.parent, args.device, partial, args.colbert_device)
    args.out.write_text(json.dumps(rankings), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
