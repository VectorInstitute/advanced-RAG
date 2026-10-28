"""Tests for the reference-artifact publisher and the setup-time installer.

The release is played by a local directory (``file://`` URLs) or by a small HTTP
server that redirects ``/releases/download/<tag>/<name>`` the way GitHub does.
Nothing here talks to GitHub.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from aieng.advanced_rag.cache import (
    CacheKey,
    TieredResultCache,
    fetch_directory,
    store_directory,
)


SCRIPTS = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publish = _load("publish_artifacts")
fetch = _load("fetch_reference_artifacts")


POLICY = {
    "licences": {},
    "rules": [
        {
            "match": {"ri_id": "ri1", "artifact_type": "scores"},
            "corpora": ["cuad"],
            "corpus_text": False,
        },
        {"match": {"ri_id": "ri1"}, "corpora": ["cuad"], "corpus_text": True},
        {"match": {"ri_id": "ri3"}, "corpora": ["financebench"], "corpus_text": True},
    ],
    "allowlist": [],
}


def _write_policy(path: Path, **changes: Any) -> Path:
    path.write_text(json.dumps({**POLICY, **changes}), encoding="utf-8")
    return path


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    """Return a local tier holding one JSON and one directory artifact for RI #1."""
    root = tmp_path / "cache"
    cache = TieredResultCache(local_dir=root, global_dir=tmp_path / "empty")
    cache.load_or_compute(CacheKey("ri1", "scores", "bm25", {"k": 10}), lambda: {"recall": 0.5})
    tree = tmp_path / "index"
    (tree / "sub").mkdir(parents=True)
    (tree / "chunks.json").write_text('["clause text"]', encoding="utf-8")
    (tree / "sub" / "vectors.bin").write_bytes(bytes(range(256)) * 64)
    store_directory(cache, CacheKey("ri1", "vector-collection", "bge", {"n": 2}), tree)
    return root


def _run(*args: Any) -> int:
    return int(publish.run([str(a) for a in args]))


def _publish(tmp_path: Path, cache_root: Path, base_url: str, *extra: Any) -> Path:
    manifest = tmp_path / "out" / "reference-artifacts.json"
    status = _run(
        "1",
        "--cache-root",
        cache_root,
        "--policy",
        _write_policy(tmp_path / "policy.json"),
        "--stage-dir",
        tmp_path / "stage",
        "--manifest",
        manifest,
        "--base-url",
        base_url,
        *extra,
    )
    assert status == 0
    return manifest


def test_collect_verifies_and_orders(cache_root: Path) -> None:
    items, problems, missing = publish.collect([cache_root], ["ri1"])
    assert [i.key.artifact_type for i in items] == ["scores", "vector-collection"]
    assert problems == [] and missing == []
    assert items[0].payload is None and items[1].payload is not None
    assert items[1].assets == 2


def test_collect_reports_a_corrupt_payload(cache_root: Path) -> None:
    payload = next(cache_root.rglob("*.artifact"))
    payload.write_bytes(payload.read_bytes() + b"x")
    items, problems, _ = publish.collect([cache_root], ["ri1"])
    assert [i.key.artifact_type for i in items] == ["scores"]
    assert "does not match" in problems[0].message


def test_collect_filters_by_key_and_reports_missing(cache_root: Path) -> None:
    digest = CacheKey("ri1", "scores", "bm25", {"k": 10}).digest
    items, _, missing = publish.collect([cache_root], ["ri1"], keys=[digest[:12], "deadbeefdead"])
    assert [i.key.digest for i in items] == [digest]
    assert missing == ["deadbeefdead"]


def test_publish_round_trips_through_a_file_release(tmp_path: Path, cache_root: Path) -> None:
    base = (tmp_path / "stage").as_uri() + "/"
    manifest = _publish(tmp_path, cache_root, base)
    first = manifest.read_bytes()
    # Packing is deterministic: a second run writes the same manifest bytes.
    _publish(tmp_path, cache_root, base)
    assert manifest.read_bytes() == first

    entries = json.loads(first)["artifacts"]
    assert {"payload_url", "payload_sha256"} <= set(entries[1])
    assert "payload_url" not in entries[0]
    sums = (tmp_path / "stage" / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 3
    notes = (tmp_path / "stage" / "RELEASE_NOTES.md").read_text()
    assert "Contract Understanding Atticus Dataset" in notes

    installed = TieredResultCache(local_dir=tmp_path / "participant", global_dir=tmp_path / "none")
    assert len(installed.fetch_manifest(manifest)) == 2
    key = CacheKey("ri1", "vector-collection", "bge", {"n": 2})
    restored = fetch_directory(installed, key, tmp_path / "restored")
    assert restored is not None
    assert (restored / "sub" / "vectors.bin").read_bytes() == bytes(range(256)) * 64
    hit = installed.get(CacheKey("ri1", "scores", "bm25", {"k": 10}))
    assert hit is not None and hit.value == {"recall": 0.5} and hit.tier == "global"


def test_default_urls_point_at_the_release(tmp_path: Path, cache_root: Path) -> None:
    manifest = tmp_path / "m.json"
    status = _run(
        "1",
        "--cache-root",
        cache_root,
        "--policy",
        _write_policy(tmp_path / "policy.json"),
        "--stage-dir",
        tmp_path / "stage",
        "--manifest",
        manifest,
    )
    assert status == 0
    url = json.loads(manifest.read_text())["artifacts"][0]["url"]
    assert url.startswith("https://github.com/VectorInstitute/advanced-RAG/releases/download/reference-ri1-")
    assert url.endswith(".json")


def test_dry_run_writes_nothing(tmp_path: Path, cache_root: Path) -> None:
    report = tmp_path / "report.json"
    status = _run(
        "1",
        "--dry-run",
        "--cache-root",
        cache_root,
        "--policy",
        _write_policy(tmp_path / "policy.json"),
        "--stage-dir",
        tmp_path / "stage",
        "--manifest",
        tmp_path / "m.json",
        "--report",
        report,
    )
    assert status == 0
    assert not (tmp_path / "stage").exists() and not (tmp_path / "m.json").exists()
    summary = json.loads(report.read_text())["targets"][0]
    assert summary["publishable_artifacts"] == 2 and summary["publishable_assets"] == 3


def test_dry_run_reports_manifest_gaps(tmp_path: Path, cache_root: Path) -> None:
    manifest = _publish(tmp_path, cache_root, "file:///release/")
    target = publish.Target("ri1", "x", ("ri1",))
    items, _, _ = publish.collect([cache_root], ["ri1"])
    target_path = tmp_path / "impl" / "reference-artifacts.json"
    target_path.parent.mkdir()
    target_path.write_bytes(manifest.read_bytes())
    original = publish.Target.manifest_path
    try:
        publish.Target.manifest_path = property(lambda self: target_path)  # type: ignore[assignment]
        assert publish.manifest_gaps(target, items) == []
        assert len(publish.manifest_gaps(target, items[:1])) == 1
    finally:
        publish.Target.manifest_path = original  # type: ignore[assignment]


def _financebench_root(tmp_path: Path) -> Path:
    root = tmp_path / "fb"
    cache = TieredResultCache(local_dir=root, global_dir=tmp_path / "empty")
    cache.load_or_compute(CacheKey("ri3", "chunks", "docling", {"doc": "A"}), lambda: ["filing text"])
    return root


def test_licence_guard_refuses_non_commercial_text(tmp_path: Path) -> None:
    root = _financebench_root(tmp_path)
    policy = publish.Policy.load(_write_policy(tmp_path / "p.json"))
    items, _, _ = publish.collect([root], ["ri3"])
    verdict = policy.decide(items[0])
    assert verdict.status == "refused" and "CC BY-NC 4.0" in verdict.reason
    status = _run(
        "3",
        "--cache-root",
        root,
        "--policy",
        tmp_path / "p.json",
        "--manifest",
        tmp_path / "m.json",
        "--stage-dir",
        tmp_path / "s",
    )
    assert status == 1 and not (tmp_path / "m.json").exists()


def test_allowlist_entry_with_a_reason_lets_text_through(tmp_path: Path) -> None:
    root = _financebench_root(tmp_path)
    allow = [{"match": {"ri_id": "ri3"}, "reason": "the licensor granted permission"}]
    policy = publish.Policy.load(_write_policy(tmp_path / "p.json", allowlist=allow))
    items, _, _ = publish.collect([root], ["ri3"])
    verdict = policy.decide(items[0])
    assert verdict.status == "allowlisted"
    assert verdict.reason == "the licensor granted permission"


def test_allowlist_entry_without_a_reason_is_rejected(tmp_path: Path) -> None:
    path = _write_policy(tmp_path / "p.json", allowlist=[{"match": {"ri_id": "ri3"}}])
    with pytest.raises(ValueError, match="reason"):
        publish.Policy.load(path)


def test_unmatched_artifact_is_unclassified(tmp_path: Path) -> None:
    root = tmp_path / "c"
    TieredResultCache(local_dir=root, global_dir=tmp_path / "e").load_or_compute(
        CacheKey("ri2", "notes", "x", {}), lambda: 1
    )
    policy = publish.Policy.load(_write_policy(tmp_path / "p.json"))
    items, _, _ = publish.collect([root], ["ri2"])
    assert policy.decide(items[0]).status == "unclassified"


def test_probe_catches_text_in_a_text_free_rule(tmp_path: Path) -> None:
    root = tmp_path / "c"
    TieredResultCache(local_dir=root, global_dir=tmp_path / "e").load_or_compute(
        CacheKey("ri1", "scores", "leaky", {}), lambda: {"chunk": "clause " * 100}
    )
    policy = publish.Policy.load(_write_policy(tmp_path / "p.json"))
    items, _, _ = publish.collect([root], ["ri1"])
    verdict = policy.decide(items[0])
    assert verdict.status == "refused" and "700-character" in verdict.reason


def test_asset_limit_stops_a_publish(tmp_path: Path, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publish, "MAX_ASSETS", 4)
    status = _run(
        "1",
        "--cache-root",
        cache_root,
        "--policy",
        _write_policy(tmp_path / "policy.json"),
        "--manifest",
        tmp_path / "m.json",
        "--stage-dir",
        tmp_path / "s",
    )
    assert status == 1 and not (tmp_path / "m.json").exists()


def test_nothing_runs_without_upload(tmp_path: Path, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(publish.subprocess, "run", lambda command, **_: calls.append(command))
    _publish(tmp_path, cache_root, "file:///release/")
    assert calls == []


def test_upload_runs_only_upload_commands(tmp_path: Path, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(publish.subprocess, "run", lambda command, **_: calls.append(command))
    monkeypatch.setattr(publish, "verify_release", lambda *args: [])
    _publish(tmp_path, cache_root, "file:///release/", "--upload")
    assert calls and all(c[:3] == ["gh", "release", "upload"] for c in calls)
    assert sum(len(c) - 6 for c in calls) == 5  # three assets, sums, notes


def test_shipped_policy_allows_serverfault_and_refuses_financebench() -> None:
    policy = publish.Policy.load(publish.DEFAULT_POLICY)
    assert {"financebench", "serverfault", "click", "repository"} <= set(policy.licences)

    def verdict(key: CacheKey) -> Any:
        item = publish.Found(key, Path(), Path("e.json"), "0" * 64, 2, {})
        return policy.decide(item)

    graph = CacheKey("ri4", "knowledge-graph", "direct-extraction", {"documents": 9999})
    runs = CacheKey("ri5", "agent-runs", "c0-errors-serverfault", {"corpus": "serverfault-errors"})
    finance = CacheKey("ri5", "agent-runs", "c0-financebench", {"corpus": "financebench"})
    assert verdict(graph).status == "allowlisted"
    assert verdict(runs).status == "allowlisted"
    assert verdict(finance).status == "refused"


def test_release_notes_give_each_corpus_of_one_type_its_own_row() -> None:
    policy = publish.Policy.load(publish.DEFAULT_POLICY)
    keys = [
        CacheKey("ri5", "agent-runs", "c0-shared", {"corpus": "musique-11656"}),
        CacheKey("ri5", "agent-runs", "c0-errors-serverfault", {"corpus": "serverfault-errors"}),
        CacheKey("ri5", "agent-runs", "c0-folder-docs", {"corpus": "folder-docs"}),
    ]
    items = [publish.Found(key, Path(), Path("e.json"), "0" * 64, 2, {}) for key in keys]
    verdicts = {item.key.digest: policy.decide(item) for item in items}
    notes = publish.release_notes(publish.resolve_target("5"), items, verdicts, policy)
    rows = [line for line in notes.splitlines() if line.startswith("| `ri5/agent-runs`")]
    assert [row.split(" | ")[3] for row in rows] == ["musique", "serverfault", "repository"]
    assert "CC BY-SA 4.0" in notes


SLICE = {
    "dump_published": "2024-04-06",
    "questions": [
        {
            "question_id": 9708,
            "link": "https://serverfault.com/questions/9708",
            "content_license": "CC BY-SA 3.0",
            "owner": {"display_name": "user2882", "user_id": 2882, "link": "https://serverfault.com/users/2882"},
            "answers": [
                {
                    "answer_id": 9717,
                    "is_accepted": True,
                    "content_license": "CC BY-SA 4.0",
                    "owner": {"display_name": "A [pipe] | name"},
                }
            ],
        },
        {
            "question_id": 1,
            "link": "https://serverfault.com/questions/1",
            "content_license": "CC BY-SA 2.5",
            "owner": {"display_name": "asker", "user_id": 5, "link": "https://serverfault.com/users/5"},
            "answers": [],
        },
    ],
}


def test_serverfault_attribution_links_every_post_and_author() -> None:
    text = publish.serverfault_attribution(SLICE, {2882: "Ada"})
    rows = [line for line in text.splitlines() if line.startswith("| [")]
    assert rows[0] == (
        "| [9708](https://serverfault.com/questions/9708) | [Ada](https://serverfault.com/users/2882) "
        "| CC BY-SA 3.0 | [9717](https://serverfault.com/a/9717) | A \\[pipe\\] \\| name | CC BY-SA 4.0 |"
    )
    assert rows[1].endswith("| CC BY-SA 2.5 | none kept |  |  |")
    assert "Stack Exchange network" in text and "2024-04-06" in text


def test_a_serverfault_release_ships_its_attribution_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache"
    cache = TieredResultCache(local_dir=root, global_dir=tmp_path / "empty")
    cache.load_or_compute(
        CacheKey("ri4", "knowledge-graph", "direct-extraction", {"documents": 2}), lambda: {"nodes": ["x"]}
    )
    rules = [{"match": {"ri_id": "ri4"}, "corpora": ["serverfault"], "corpus_text": True}]
    allow = [{"match": {"ri_id": "ri4"}, "reason": "redistributed with attribution"}]
    monkeypatch.setitem(publish.ATTRIBUTION_FILES, "serverfault", (publish.SERVERFAULT_ATTRIBUTION, lambda: "posts\n"))
    status = _run(
        "4",
        "--cache-root",
        root,
        "--policy",
        _write_policy(tmp_path / "p.json", rules=rules, allowlist=allow),
        "--stage-dir",
        tmp_path / "stage",
        "--manifest",
        tmp_path / "m.json",
    )
    assert status == 0
    assert (tmp_path / "stage" / publish.SERVERFAULT_ATTRIBUTION).read_text() == "posts\n"
    assert publish.SERVERFAULT_ATTRIBUTION in (tmp_path / "stage" / "SHA256SUMS").read_text()
    notes = (tmp_path / "stage" / "RELEASE_NOTES.md").read_text()
    assert publish.SERVERFAULT_ATTRIBUTION in notes and "redistributed with attribution" in notes


def test_targets_resolve_numbers_and_folders() -> None:
    assert publish.resolve_target("4").folder == "graphrag_technical_support"
    assert publish.resolve_target("ri5").name == "ri5"
    assert publish.resolve_target("implementations/agent_native_retrieval/").name == "ri5"
    assert publish.resolve_target("visual_documents").name == "visual-documents"
    with pytest.raises(ValueError):
        publish.resolve_target("9")


# ---------------------------------------------------------------------------
# Installer against a fake release server


class _Release(http.server.BaseHTTPRequestHandler):
    """Serve ``/<repo>/releases/download/<tag>/<name>`` through a redirect."""

    stage: Path
    tag = "reference-ri1-test"
    failures: dict[str, int] = {}  # noqa: RUF012
    requests: list[str] = []  # noqa: RUF012

    def do_GET(self) -> None:  # noqa: N802
        """Redirect a download URL, or serve the object it redirects to."""
        self.requests.append(self.path)
        prefix = f"/VectorInstitute/advanced-RAG/releases/download/{self.tag}/"
        if self.path.startswith(prefix):
            self.send_response(302)
            self.send_header("Location", "/objects/" + self.path[len(prefix) :])
            self.end_headers()
            return
        if self.path.startswith("/objects/"):
            name = self.path[len("/objects/") :]
            if self.failures.get(name, 0) > 0:
                self.failures[name] -= 1
                self.send_error(503)
                return
            path = self.stage / name
            if path.is_file():
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_error(404)

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def release(tmp_path: Path) -> Iterator[str]:
    """Run the fake release server and return its base URL."""
    _Release.stage = tmp_path / "stage"
    _Release.failures = {}
    _Release.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Release)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def _download_base(release: str, tag: str = _Release.tag) -> str:
    return f"{release}/VectorInstitute/advanced-RAG/releases/download/{tag}/"


def test_installer_fetches_then_skips(tmp_path: Path, cache_root: Path, release: str) -> None:
    manifest = _publish(tmp_path, cache_root, _download_base(release))
    cache = TieredResultCache(local_dir=tmp_path / "p", global_dir=tmp_path / "n")
    assert fetch.install_manifest(cache, manifest, retries=0) == (2, 0, [])
    before = len(_Release.requests)
    assert fetch.install_manifest(cache, manifest, retries=0) == (0, 2, [])
    assert len(_Release.requests) == before


def test_installer_retries_a_transient_failure(tmp_path: Path, cache_root: Path, release: str) -> None:
    manifest = _publish(tmp_path, cache_root, _download_base(release))
    name = json.loads(manifest.read_text())["artifacts"][0]["url"].rsplit("/", 1)[-1]
    _Release.failures[name] = 2
    waits: list[float] = []
    cache = TieredResultCache(local_dir=tmp_path / "p", global_dir=tmp_path / "n")
    assert fetch.install_manifest(cache, manifest, retries=3, sleep=waits.append) == (
        2,
        0,
        [],
    )
    assert waits == [1, 2]


def test_installer_stops_a_manifest_on_404(tmp_path: Path, cache_root: Path, release: str) -> None:
    manifest = _publish(tmp_path, cache_root, _download_base(release, "no-such-tag"))
    waits: list[float] = []
    cache = TieredResultCache(local_dir=tmp_path / "p", global_dir=tmp_path / "n")
    installed, present, failures = fetch.install_manifest(cache, manifest, retries=3, sleep=waits.append)
    assert (installed, present) == (0, 0)
    assert "not published" in failures[-1] and waits == []


def test_installer_cli_exit_status(
    tmp_path: Path, cache_root: Path, release: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADVANCED_RAG_CACHE_DIR", str(tmp_path / "participant"))
    good = _publish(tmp_path, cache_root, _download_base(release))
    assert fetch.main([str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert fetch.main([str(bad), "--retries", "0"]) == 1
