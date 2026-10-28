"""Tests for packaging the null-arm store so another machine can score from it.

Covers what the bundle holds and omits, that an exported bundle verifies
against its own manifest, and that a tampered byte is caught. Nothing here
touches a network or a model.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import tarfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "aieng-advanced-rag"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from export_store import (  # noqa: E402
    MANIFEST_NAME,
    bundle_members,
    read_manifest,
    unpacked_mismatches,
    verify_bundle,
    write_bundle,
)


def build_store(root: Path) -> Path:
    """Return a store laid out like a real one, with a file in every part."""
    store = root / "graphrag-null"
    (store / "evidence").mkdir(parents=True)
    (store / "evidence" / "dense.json").write_text('{"q1": []}', encoding="utf-8")
    (store / "evidence" / "lightrag-mix.json").write_text('{"q1": []}', encoding="utf-8")
    (store / "lightrag" / "doc-1").mkdir(parents=True)
    (store / "lightrag" / "doc-1" / "graph.graphml").write_text("<graphml/>", encoding="utf-8")
    (store / "corpus").mkdir()
    (store / "corpus" / "test.json").write_text("[]", encoding="utf-8")
    with sqlite3.connect(str(store / "proxy_cache.sqlite")) as cache:
        cache.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, payload TEXT)")
    return store


class TestWhatTheBundleCarries:
    """The score-only bundle is the default, because scoring is the use."""

    def test_the_default_bundle_holds_the_tables_and_the_cache(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        members = {str(path) for path in bundle_members(store, full=False)}
        assert members == {
            "evidence/dense.json",
            "evidence/lightrag-mix.json",
            "proxy_cache.sqlite",
        }

    def test_the_default_bundle_omits_the_indexes(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        members = {str(path) for path in bundle_members(store, full=False)}
        assert not [name for name in members if name.startswith("lightrag/")]

    def test_the_full_bundle_adds_the_indexes(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        members = {str(path) for path in bundle_members(store, full=True)}
        assert "lightrag/doc-1/graph.graphml" in members

    def test_no_bundle_ships_the_corpus(self, tmp_path: Path) -> None:
        # The corpus re-downloads through the digest-pinned loader, so a copy
        # here would only be a second thing to keep honest.
        store = build_store(tmp_path)
        for full in (False, True):
            members = {str(path) for path in bundle_members(store, full=full)}
            assert not [name for name in members if name.startswith("corpus/")]

    def test_a_directory_that_is_not_a_store_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="not a null-arm store"):
            bundle_members(tmp_path / "empty", full=False)


class TestTheManifestTravels:
    """A bundle states what it is, so the receiver need not ask."""

    def test_the_manifest_names_every_packed_file_with_a_digest(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        manifest = write_bundle(store, tmp_path / "bundle.tar.gz")
        assert len(manifest["files"]) == 3
        assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])

    def test_the_manifest_records_what_it_omits_and_why(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        manifest = write_bundle(store, tmp_path / "bundle.tar.gz")
        assert set(manifest["omits"]) == {"corpus", "lightrag"}
        assert "re-downloads" in manifest["omitted_because"]["corpus"]

    def test_the_corpus_attribution_travels_with_the_text(self, tmp_path: Path) -> None:
        # The tables and the prompt cache quote contract text, so the licence
        # the corpus ships under has to reach whoever unpacks this.
        store = build_store(tmp_path)
        manifest = write_bundle(store, tmp_path / "bundle.tar.gz")
        assert manifest["corpus"]["licence"]
        assert manifest["corpus"]["attribution"]

    def test_the_manifest_is_inside_the_archive(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
            assert MANIFEST_NAME in archive.getnames()
        assert read_manifest(tmp_path / "bundle.tar.gz")["bundle"] == "score-only"


class TestVerification:
    """A bundle crossing machines is checked, not trusted."""

    def test_a_fresh_bundle_verifies(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        assert verify_bundle(tmp_path / "bundle.tar.gz") == []

    def test_an_altered_file_is_caught(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")

        # Repack with one byte of one table changed and the manifest untouched,
        # which is what a damaged transfer looks like from the receiving end.
        unpacked = tmp_path / "unpacked"
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
            archive.extractall(unpacked, filter="data")
        (unpacked / "evidence" / "dense.json").write_text('{"q1": ["x"]}', encoding="utf-8")
        with tarfile.open(tmp_path / "tampered.tar.gz", "w:gz") as archive:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(unpacked)))

        assert verify_bundle(tmp_path / "tampered.tar.gz") == ["evidence/dense.json"]

    def test_a_link_member_is_caught(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as source:
            members = [(m, source.extractfile(m).read() if m.isfile() else None) for m in source.getmembers()]
        with tarfile.open(tmp_path / "linked.tar.gz", "w:gz") as archive:
            for member, data in members:
                archive.addfile(member, io.BytesIO(data) if data is not None else None)
            link = tarfile.TarInfo("evidence/outside")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside"
            archive.addfile(link)
        assert verify_bundle(tmp_path / "linked.tar.gz") == ["evidence/outside"]

    def test_an_edit_after_unpacking_is_caught(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        unpacked = tmp_path / "unpacked"
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
            archive.extractall(unpacked, filter="data")
        assert unpacked_mismatches(unpacked) == []
        (unpacked / "evidence" / "dense.json").write_text('{"q1": ["x"]}', encoding="utf-8")
        assert unpacked_mismatches(unpacked) == ["evidence/dense.json"]

    def test_the_export_replaces_the_destination_in_one_step(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == ["bundle.tar.gz"]

    def test_a_bundle_without_a_manifest_cannot_be_verified(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        with tarfile.open(tmp_path / "bare.tar.gz", "w:gz") as archive:
            archive.add(store / "proxy_cache.sqlite", arcname="proxy_cache.sqlite")
        with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
            verify_bundle(tmp_path / "bare.tar.gz")


class TestTheBundleScoresWithoutReindexing:
    """The claim the bundle exists to make, checked rather than asserted."""

    def test_an_unpacked_bundle_carries_a_table_for_every_arm_it_names(self, tmp_path: Path) -> None:
        # `run_null_arm.py` skips indexing when every requested arm has a
        # table, so this is the on-disk condition that makes the skip fire.
        store = build_store(tmp_path)
        write_bundle(store, tmp_path / "bundle.tar.gz")
        unpacked = tmp_path / "unpacked"
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
            archive.extractall(unpacked, filter="data")

        arms = ["dense", "lightrag-mix"]
        assert all((unpacked / "evidence" / f"{arm}.json").exists() for arm in arms)
        assert not (unpacked / "lightrag").exists()

    def test_the_manifest_says_so(self, tmp_path: Path) -> None:
        store = build_store(tmp_path)
        manifest = write_bundle(store, tmp_path / "bundle.tar.gz")
        assert manifest["scores_without_reindexing"] is True
        assert json.loads(json.dumps(manifest))  # the manifest round-trips
