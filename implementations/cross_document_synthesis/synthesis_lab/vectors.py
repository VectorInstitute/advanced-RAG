"""Corpus-keyed local vectors; annotation changes cannot alter method inputs."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique

from .corpora import Corpus
from .dense_baseline import passage_text
from .runtime import DEFAULT_CACHE, MODEL, MODEL_REVISION, embeddings, local_embedder, write_json


def corpus_vectors(corpus: Corpus, cache: Path) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Return documents, a gold-free query encoder, and checked vector metadata."""
    encoder = local_embedder(DEFAULT_CACHE)
    texts = [q.question for q in corpus.questions]
    identity = {
        "corpus": corpus.provenance["corpus_sha256"],
        "model": MODEL,
        "revision": MODEL_REVISION,
        "encoder": encoder.describe(),
        "format": "title + newline + text",
    }
    folder = cache / "vectors" / corpus.name
    folder.mkdir(parents=True, exist_ok=True)
    matrix_path, manifest_path = folder / "documents.npy", folder / "manifest.json"
    if matrix_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["identity"] != identity
            or hashlib.sha256(matrix_path.read_bytes()).hexdigest() != manifest["sha256"]
        ):
            raise ValueError("vector identity or digest mismatch")
        docs = np.load(matrix_path, allow_pickle=False)
    else:
        if corpus.name == "musique" and corpus.questions[0].wording == "original":
            # Reuse the already verified reference vectors without changing its cache.
            passages, questions = load_musique(fetch_musique(dest=DEFAULT_CACHE / "corpus"))
            docs, _, _ = embeddings(DEFAULT_CACHE, passages, questions, embedder=encoder)
        else:
            docs = np.asarray(encoder.encode([passage_text(p) for p in corpus.passages]), dtype=np.float32)
        with matrix_path.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, docs)
        matrix_path.with_suffix(".tmp").replace(matrix_path)
        manifest = {"identity": identity, "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()}
        write_json(manifest_path, manifest)
    if docs.shape != (len(corpus.passages), 384) or not np.isfinite(docs).all():
        raise ValueError("invalid document vectors")
    # Cache original queries in memory. Follow-up queries still use this exact encoder.
    vectors = encoder.encode_queries(texts)
    by_text = dict(zip(texts, vectors, strict=True))
    lock = threading.Lock()

    def encode(queries: Sequence[str]) -> list[list[float]]:
        with lock:
            missing = list(dict.fromkeys(q for q in queries if q not in by_text))
            if missing:
                by_text.update(zip(missing, encoder.encode_queries(missing), strict=True))
            return [by_text[q] for q in queries]

    return docs, encode, manifest
