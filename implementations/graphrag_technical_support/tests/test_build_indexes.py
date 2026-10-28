"""Test the LightRAG build: flush throttle, record cleanup, and lost documents."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import build_indexes
import numpy as np
import pytest


class FakeRag:
    """Stand in for LightRAG, counting how often it flushes."""

    def __init__(self) -> None:
        self.flushed: list[int] = []
        self._stores = ["a", "b", "c"]

    def _index_storages(self) -> list[str]:
        return self._stores

    async def _flush_storages(self, storages: list[str]) -> None:
        self.flushed.append(len(storages))


def test_throttle_skips_full_flushes_inside_the_interval() -> None:
    rag = FakeRag()
    flush_all = build_indexes.throttle_flushes(rag, interval=3600.0)

    async def run() -> None:
        for _ in range(5):
            await rag._flush_storages(rag._index_storages())
        await rag._flush_storages(["a"])
        await flush_all()

    asyncio.run(run())
    # Five gated full flushes are skipped, the partial one passes, the forced one runs.
    assert rag.flushed == [1, 3]


def test_throttle_flushes_again_after_the_interval() -> None:
    rag = FakeRag()
    build_indexes.throttle_flushes(rag, interval=0.0)

    async def run() -> None:
        await rag._flush_storages(rag._index_storages())
        await rag._flush_storages(rag._index_storages())

    asyncio.run(run())
    assert rag.flushed == [3, 3]


def test_known_ids_drop_duplicate_name_records(tmp_path: Path) -> None:
    status = {
        "musique::00001": {"status": "processed"},
        "dup-1": {"status": "failed", "error_msg": "File name already exists. Original doc_id: musique::00001"},
        "musique::00002": {"status": "failed", "error_msg": "Cancelled by internal error"},
    }
    (tmp_path / "kv_store_doc_status.json").write_text(json.dumps(status))
    assert build_indexes.known_document_ids(tmp_path) == {"musique::00001", "musique::00002"}
    assert set(json.loads((tmp_path / "kv_store_doc_status.json").read_text())) == {"musique::00001", "musique::00002"}


def test_known_ids_of_an_empty_directory(tmp_path: Path) -> None:
    assert build_indexes.known_document_ids(tmp_path) == set()


def test_a_second_build_cannot_take_the_directory(tmp_path: Path) -> None:
    directory = tmp_path / "lightrag"
    with (
        build_indexes.exclusive(directory),
        pytest.raises(SystemExit, match="another build"),
        build_indexes.exclusive(directory),
    ):
        pass
    with build_indexes.exclusive(directory):
        pass
    assert not directory.exists()


# The rest drives LightRAG itself, with a scripted model and a hashing embedder,
# through the failure the reconcile exists for.

DOCUMENTS = {"doc-a": "Alpha met Beta in the valley.", "doc-b": "Gamma met Delta at the river."}
EXTRACTIONS = {
    "Alpha met Beta": ("Alpha", "Beta"),
    "Gamma met Delta": ("Gamma", "Delta"),
}


async def scripted_llm(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
    """Return one relation between the two people a document names, or nothing."""
    for marker, (first, second) in EXTRACTIONS.items():
        if marker in prompt or marker in (system_prompt or ""):
            return (
                f"entity<|#|>{first}<|#|>person<|#|>{first} is a person.\n"
                f"entity<|#|>{second}<|#|>person<|#|>{second} is a person.\n"
                f"relation<|#|>{first}<|#|>{second}<|#|>meeting<|#|>{first} met {second}.\n"
                "<|COMPLETE|>"
            )
    return "<|COMPLETE|>"


def hashing_embedder() -> Any:
    lightrag_utils = pytest.importorskip("lightrag.utils")

    async def embed(texts: list[str]) -> np.ndarray:
        rows = [np.frombuffer(hashlib.sha256(t.encode()).digest()[:16], dtype=np.uint8) for t in texts]
        return np.array(rows, dtype=np.float32) + 1.0

    return lightrag_utils.EmbeddingFunc(embedding_dim=16, max_token_size=512, func=embed)


@pytest.fixture
def lightrag_process_state() -> Any:
    """Clear LightRAG's process-wide storage state, which outlives an event loop."""
    shared = pytest.importorskip("lightrag.kg.shared_storage")
    shared.finalize_share_data()
    yield
    shared.finalize_share_data()


async def open_rag(directory: Path) -> Any:
    lightrag = pytest.importorskip("lightrag")
    rag = lightrag.LightRAG(
        working_dir=str(directory),
        llm_model_func=scripted_llm,
        embedding_func=hashing_embedder(),
        entity_extract_max_gleaning=0,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag


async def insert(directory: Path, doc_ids: list[str]) -> None:
    rag = await open_rag(directory)
    await rag.ainsert([DOCUMENTS[d] for d in doc_ids], ids=doc_ids, file_paths=doc_ids)
    await rag.finalize_storages()


GRAPH_FILES = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
)


@pytest.mark.parametrize(
    ("lost", "reason"),
    [
        # A kill before the next flush keeps doc-b's status, which LightRAG writes
        # at once, and loses everything else it wrote for doc-b.
        (None, "chunk missing from kv_store_text_chunks.json"),
        # A flush that caught doc-b's chunks but not its graph merge.
        (GRAPH_FILES, "entity not linked to the document's chunks"),
    ],
)
def test_a_document_lost_between_flushes_is_found_and_indexed_again(
    tmp_path: Path, lost: tuple[str, ...] | None, reason: str, lightrag_process_state: None
) -> None:
    directory = tmp_path / "index"
    asyncio.run(insert(directory, ["doc-a"]))
    before = tmp_path / "before"
    shutil.copytree(directory, before)
    asyncio.run(insert(directory, ["doc-b"]))
    for path in before.iterdir():
        if path.name != "kv_store_doc_status.json" and (lost is None or path.name in lost):
            shutil.copy(path, directory / path.name)

    assert build_indexes.incomplete_documents(directory) == {"doc-b": reason}

    async def repair() -> Any:
        rag = await open_rag(directory)
        await build_indexes.forget_documents(rag, ["doc-b"])
        await rag.ainsert([DOCUMENTS["doc-b"]], ids=["doc-b"], file_paths=["doc-b"])
        nodes = await rag.chunk_entity_relation_graph.get_all_labels()
        await rag.finalize_storages()
        return nodes

    nodes = asyncio.run(repair())
    assert build_indexes.index_problems(directory, set(DOCUMENTS)) == []
    assert {"Gamma", "Delta"} <= set(nodes)
    statuses = json.loads((directory / "kv_store_doc_status.json").read_text())
    # No duplicate-name record from the second insert.
    assert set(statuses) == set(DOCUMENTS)
