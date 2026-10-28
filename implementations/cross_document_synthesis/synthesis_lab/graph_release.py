"""Read RI4's checksum-pinned release without changing its native indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import ArtifactResult, CacheKey

from .corpora import Corpus
from .runtime import MODEL, MODEL_REVISION


#: RI #4's own manifest pins the graph indexes, so a republished RI #4 release
#: repins RI #2's graph runs too.
MANIFEST = Path(__file__).resolve().parents[2] / "graphrag_technical_support" / "reference-artifacts.json"
CONTRACT_REVISION = "97997198e1819fde9a2a7d16d4007991180e1a43"


def release_entry(backend: str) -> dict[str, Any]:
    """Return the immutable public release descriptor for one producer."""
    entries = json.loads(MANIFEST.read_text())["artifacts"]
    found = [
        entry
        for entry in entries
        if entry["key"]["artifact_type"] == "graph-index" and entry["key"]["producer_id"] == backend
    ]
    if len(found) != 1:
        raise ValueError("select one released backend: hipporag2 or lightrag")
    entry: dict[str, Any] = found[0]
    return entry


def verify_release(artifact: ArtifactResult, key: CacheKey, corpus: Corpus) -> dict[str, Any]:
    """Check release bytes and corpus identity before loading native stores."""
    entry = release_entry(key.producer_id)
    if key != CacheKey(**entry["key"]):
        raise ValueError("graph key differs from the pinned RI4 release")
    for path, expected in (
        (artifact.path, entry["payload_sha256"]),
        (artifact.envelope_path, entry["sha256"]),
    ):
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError("graph artifact differs from the pinned release checksum")
    from aieng.advanced_rag.corpora.musique import MUSIQUE_CORPUS  # noqa: PLC0415

    config = key.config
    if (
        corpus.name != "musique"
        or len(corpus.passages) != config["corpus_passages"]
        or config["passages"] != config["corpus_passages"]
        or corpus.provenance["revision"] != config["dataset_revision"]
        or corpus.provenance["corpus_sha256"] != MUSIQUE_CORPUS.sha256
    ):
        raise ValueError("graph release does not match the complete pinned corpus")
    return entry


def hippo_precompute_files() -> tuple[str, ...]:
    """Files atlas-rag must load rather than recompute under the build settings."""
    prefix = "musique_eventFalse_conceptFalse_bge-small-en-v1.5"
    return (
        f"{prefix}_node_faiss.index",
        f"{prefix}_edge_faiss.index",
        f"{prefix}_node_embeddings.pkl",
        f"{prefix}_edge_embeddings.pkl",
        "musique_bge-small-en-v1.5_text_embeddings.pkl",
        "musique_text_faiss.index",
    )


def native_metadata(
    directory: Path, key: CacheKey, corpus: Corpus, entry: dict[str, Any], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    """Validate native document mappings and describe the actual implementation."""
    settings: dict[str, Any] = dict(key.config) | {
        "embedding_model": MODEL,
        # RI4's cache key names the encoder, but does not pin its HF revision.
        "embedding_revision": None,
        "query_embedding_revision": MODEL_REVISION,
    }
    if key.producer_id == "hipporag2":
        mapping = hippo_mapping(directory, corpus)
        settings["implementation"] = "atlas-rag"
        backend = "atlas-rag"
    else:
        mapping = light_mapping(directory, corpus, allow_incomplete=allow_incomplete)
        settings.update(implementation="LightRAG", mode="mix", top_k=40)
        backend = "lightrag"
    return {
        "index_coverage": {
            "complete": set(mapping.values()) == {p.passage_id for p in corpus.passages},
            "represented_passages": len(set(mapping.values())),
            "missing_passage_ids": sorted({p.passage_id for p in corpus.passages} - set(mapping.values())),
            "allow_incomplete": allow_incomplete,
        },
        "backend": backend,
        "producer_backend": key.producer_id,
        "contract_revision": CONTRACT_REVISION,
        "producer_revision": None,
        "corpus_revision": corpus.provenance["revision"],
        "corpus_sha256": corpus.provenance["corpus_sha256"],
        "passage_ids": [p.passage_id for p in corpus.passages],
        "retriever_id_map": mapping,
        "settings": settings,
        "release": entry,
    }


def hippo_mapping(directory: Path, corpus: Corpus) -> dict[str, str]:
    """Resolve every native graph passage node using its original indexed text."""
    known = {p.passage_id for p in corpus.passages}
    import networkx as nx  # noqa: PLC0415

    required = [directory / "precompute" / name for name in hippo_precompute_files()]
    graph_path = directory / "kg_graphml/musique_graph.graphml"
    if not graph_path.is_file() or any(not p.is_file() for p in required):
        raise ValueError("HippoRAG atlas-rag release lacks prebuilt graph/vector stores")
    by_text: dict[str, list[str]] = {}
    for passage in corpus.passages:
        by_text.setdefault(f"{passage.title}\n{passage.text}", []).append(passage.passage_id)
    mapping = {}
    for node, data in nx.read_graphml(graph_path).nodes(data=True):
        if data.get("type") != "passage":
            continue
        matches = by_text.get(str(data.get("id", "")))
        if not matches or len(matches) != 1:
            raise ValueError("graph passage text is unknown or ambiguous")
        mapping[str(node)] = matches[0]
    if set(mapping.values()) != known or len(mapping) != len(known):
        raise ValueError("graph passage nodes do not cover the complete corpus")
    return mapping


def light_mapping(directory: Path, corpus: Corpus, *, allow_incomplete: bool = False) -> dict[str, str]:
    """Resolve and verify original documents and every LightRAG chunk."""
    known = {p.passage_id for p in corpus.passages}
    required_names = (
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
    )
    if any(not (directory / name).is_file() for name in required_names):
        raise ValueError("LightRAG release lacks prebuilt graph/vector stores")
    statuses = json.loads((directory / "kv_store_doc_status.json").read_text())
    docs = json.loads((directory / "kv_store_full_docs.json").read_text())
    expected = {p.passage_id: f"{p.title}\n{p.text}" for p in corpus.passages}
    if set(statuses) != known or set(docs) != known:
        raise ValueError("LightRAG document IDs do not cover the complete corpus")
    if any(row.get("status") != "processed" for row in statuses.values()):
        raise ValueError("LightRAG contains unprocessed documents")
    if any(docs[pid].get("content") != text for pid, text in expected.items()):
        raise ValueError("LightRAG document bytes differ from the corpus")
    chunks = json.loads((directory / "kv_store_text_chunks.json").read_text())
    mapping = {}
    for chunk_id, chunk in chunks.items():
        pid = chunk.get("full_doc_id")
        if pid not in known or chunk.get("file_path") != pid:
            raise ValueError("LightRAG chunk does not resolve to an original corpus passage")
        mapping[chunk_id] = pid
    if not mapping or (set(mapping.values()) != known and not allow_incomplete):
        raise ValueError("LightRAG chunks do not cover the complete corpus")
    return mapping
