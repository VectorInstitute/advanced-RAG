"""Check that a LightRAG working directory holds every document it marks processed.

LightRAG 1.5.6 writes a document's status to ``kv_store_doc_status.json`` as
soon as it changes, and writes chunks, vectors, and the graph only when its
storages flush. A process stopped between the two leaves documents marked
``processed`` that the index does not hold. A count of statuses cannot see
that, so these checks compare the status store with the chunk store, the chunk
vector store, and the entity and relation tracking stores.

The checks read LightRAG's JSON files and do not import LightRAG, so they run
on an unpacked index without the ``lightrag`` extra::

    from aieng.advanced_rag.graphrag.lightrag_index import index_problems

    problems = index_problems(directory, expected_document_ids)
    if problems:
        raise ValueError("; ".join(problems))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


__all__ = ["incomplete_documents", "index_problems"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else {}


def _relation_key(source: str, target: str) -> str:
    # LightRAG keys relation tracking by the sorted pair; see
    # lightrag.utils.make_relation_chunk_key.
    return "<SEP>".join(sorted((source, target)))


def incomplete_documents(working_dir: Path) -> dict[str, str]:
    """Return the processed documents the index does not hold in full, with the reason.

    A document is complete when every chunk its status lists is in the chunk store
    and the chunk vector store, and every entity and relation its extraction listed
    in ``full_entities`` and ``full_relations`` is tracked against one of its chunks.
    Reads LightRAG's JSON stores only, so it runs before or after LightRAG loads.
    """
    statuses = _read_json(working_dir / "kv_store_doc_status.json")
    chunks = _read_json(working_dir / "kv_store_text_chunks.json")
    vectors = {
        row["__id__"]
        for row in _read_json(working_dir / "vdb_chunks.json").get("data", [])
    }
    entities = _read_json(working_dir / "kv_store_full_entities.json")
    relations = _read_json(working_dir / "kv_store_full_relations.json")
    entity_chunks = _read_json(working_dir / "kv_store_entity_chunks.json")
    relation_chunks = _read_json(working_dir / "kv_store_relation_chunks.json")
    found: dict[str, str] = {}
    for doc_id, record in statuses.items():
        if record.get("status") != "processed":
            continue
        listed = set(record.get("chunks_list") or [])
        if not listed:
            found[doc_id] = "no chunks listed"
        elif not listed <= set(chunks):
            found[doc_id] = "chunk missing from kv_store_text_chunks.json"
        elif not listed <= vectors:
            found[doc_id] = "chunk missing from vdb_chunks.json"
        elif any(
            not listed & set(entity_chunks.get(name, {}).get("chunk_ids", []))
            for name in entities.get(doc_id, {}).get("entity_names", [])
        ):
            found[doc_id] = "entity not linked to the document's chunks"
        elif any(
            not listed
            & set(relation_chunks.get(_relation_key(*pair), {}).get("chunk_ids", []))
            for pair in relations.get(doc_id, {}).get("relation_pairs", [])
        ):
            found[doc_id] = "relation not linked to the document's chunks"
    return found


def index_problems(working_dir: Path, expected: set[str]) -> list[str]:
    """Return every way the stores disagree with each other or with ``expected``.

    Empty means each expected document is processed and complete, every chunk
    belongs to a document with a status, and the chunk and vector stores hold the
    same chunks.
    """
    statuses = _read_json(working_dir / "kv_store_doc_status.json")
    chunks = _read_json(working_dir / "kv_store_text_chunks.json")
    vectors = {
        row["__id__"]
        for row in _read_json(working_dir / "vdb_chunks.json").get("data", [])
    }
    processed = {d for d, r in statuses.items() if r.get("status") == "processed"}
    problems = []
    if missing := sorted(expected - processed):
        problems.append(f"{len(missing)} documents not processed: {missing[:10]}")
    for doc_id, reason in sorted(incomplete_documents(working_dir).items()):
        problems.append(f"{doc_id} is processed but incomplete: {reason}")
    orphans = sorted(
        c for c, row in chunks.items() if row.get("full_doc_id") not in statuses
    )
    if orphans:
        problems.append(f"{len(orphans)} chunks belong to no document: {orphans[:10]}")
    if vectors != set(chunks):
        problems.append(
            f"chunk and vector stores differ: {len(set(chunks) - vectors)} chunks without a vector, "
            f"{len(vectors - set(chunks))} vectors without a chunk"
        )
    return problems
