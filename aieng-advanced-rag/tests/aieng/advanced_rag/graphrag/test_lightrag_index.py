"""Test the LightRAG store checks on hand-written JSON stores."""

import json
from pathlib import Path
from typing import Any

import pytest

from aieng.advanced_rag.graphrag import lightrag_index


def write_stores(directory: Path, **stores: Any) -> None:
    """Write LightRAG JSON stores by short name, for example ``doc_status``."""
    files = {
        "doc_status": "kv_store_doc_status.json",
        "text_chunks": "kv_store_text_chunks.json",
        "full_entities": "kv_store_full_entities.json",
        "full_relations": "kv_store_full_relations.json",
        "entity_chunks": "kv_store_entity_chunks.json",
        "relation_chunks": "kv_store_relation_chunks.json",
    }
    for name, value in stores.items():
        if name == "vectors":
            (directory / "vdb_chunks.json").write_text(
                json.dumps({"data": [{"__id__": i} for i in value]})
            )
        else:
            (directory / files[name]).write_text(json.dumps(value))


def complete_index(directory: Path) -> None:
    """Write complete documents a and b: one chunk, two entities, one relation each."""
    write_stores(
        directory,
        doc_status={
            d: {"status": "processed", "chunks_list": [f"{d}-c0"]} for d in ("a", "b")
        },
        text_chunks={f"{d}-c0": {"full_doc_id": d} for d in ("a", "b")},
        vectors=["a-c0", "b-c0"],
        full_entities={
            "a": {"entity_names": ["X", "Y"]},
            "b": {"entity_names": ["Y", "Z"]},
        },
        full_relations={
            "a": {"relation_pairs": [["Y", "X"]]},
            "b": {"relation_pairs": [["Y", "Z"]]},
        },
        entity_chunks={
            "X": {"chunk_ids": ["a-c0"]},
            "Y": {"chunk_ids": ["a-c0", "b-c0"]},
            "Z": {"chunk_ids": ["b-c0"]},
        },
        relation_chunks={
            "X<SEP>Y": {"chunk_ids": ["a-c0"]},
            "Y<SEP>Z": {"chunk_ids": ["b-c0"]},
        },
    )


def test_a_complete_index_has_no_problems(tmp_path: Path) -> None:
    complete_index(tmp_path)
    assert lightrag_index.incomplete_documents(tmp_path) == {}
    assert lightrag_index.index_problems(tmp_path, {"a", "b"}) == []


@pytest.mark.parametrize(
    ("stores", "reason"),
    [
        (
            {"text_chunks": {"a-c0": {"full_doc_id": "a"}}},
            "chunk missing from kv_store_text_chunks.json",
        ),
        ({"vectors": ["a-c0"]}, "chunk missing from vdb_chunks.json"),
        (
            {
                "entity_chunks": {
                    "X": {"chunk_ids": ["a-c0"]},
                    "Y": {"chunk_ids": ["a-c0"]},
                    "Z": {"chunk_ids": []},
                }
            },
            "entity not linked to the document's chunks",
        ),
        (
            {"relation_chunks": {"X<SEP>Y": {"chunk_ids": ["a-c0"]}}},
            "relation not linked to the document's chunks",
        ),
    ],
)
def test_a_processed_document_missing_from_a_store_is_incomplete(
    tmp_path: Path, stores: dict[str, Any], reason: str
) -> None:
    complete_index(tmp_path)
    write_stores(tmp_path, **stores)
    assert lightrag_index.incomplete_documents(tmp_path) == {"b": reason}
    assert any(
        line.startswith("b is processed but incomplete")
        for line in lightrag_index.index_problems(tmp_path, {"a", "b"})
    )


def test_documents_still_in_the_pipeline_are_not_incomplete(tmp_path: Path) -> None:
    complete_index(tmp_path)
    write_stores(
        tmp_path,
        doc_status={
            "a": {"status": "processed", "chunks_list": ["a-c0"]},
            "b": {"status": "pending", "chunks_list": []},
        },
        text_chunks={"a-c0": {"full_doc_id": "a"}},
        vectors=["a-c0"],
    )
    assert lightrag_index.incomplete_documents(tmp_path) == {}
    assert lightrag_index.index_problems(tmp_path, {"a", "b"}) == [
        "1 documents not processed: ['b']"
    ]


def test_problems_name_orphan_chunks_and_stray_vectors(tmp_path: Path) -> None:
    complete_index(tmp_path)
    write_stores(
        tmp_path,
        text_chunks={
            "a-c0": {"full_doc_id": "a"},
            "b-c0": {"full_doc_id": "b"},
            "gone-c0": {"full_doc_id": "gone"},
        },
        vectors=["a-c0", "b-c0", "gone-c0", "stray"],
    )
    problems = lightrag_index.index_problems(tmp_path, {"a", "b"})
    assert problems == [
        "1 chunks belong to no document: ['gone-c0']",
        "chunk and vector stores differ: 0 chunks without a vector, 1 vectors without a chunk",
    ]
