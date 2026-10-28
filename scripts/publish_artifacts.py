#!/usr/bin/env python3
"""Pack an RI's reference artifacts for a GitHub release and write its fetch manifest.

Participants install reference artifacts with
``python -m aieng.advanced_rag.cache implementations/<ri>/reference-artifacts.json``,
which calls :meth:`TieredResultCache.fetch_manifest`. This script produces that
manifest and the files it points at:

1. It collects every artifact of the RI from the cache tiers it is given
   (``~/.cache/aieng-advanced-rag/artifacts`` by default), validating each
   envelope against its key and each payload against the digest in its envelope.
2. It checks each artifact against the licence policy in
   ``scripts/reference-artifact-licences.json`` and refuses corpus text under a
   CC BY-NC or CC BY-SA licence unless an allowlist entry gives the reason.
3. It stages the exact cached bytes under content-addressed asset names, with a
   ``SHA256SUMS`` file and release notes that carry each corpus's attribution.
4. It writes ``implementations/<ri>/reference-artifacts.json`` and prints the
   ``gh release`` commands. Nothing is uploaded unless you pass ``--upload``, and
   the script never creates a release.

Examples
--------
Report what the local cache holds for every RI, writing nothing::

    $ python scripts/publish_artifacts.py all --dry-run

Stage RI #4 and write its manifest::

    $ python scripts/publish_artifacts.py 4

Prior art and why this design
-----------------------------
pooch pins each file with a ``name hash`` registry and retries a failed
download with a wait that grows by one second up to ten. torch.hub puts a hash
prefix in the file name (``resnet18-f37072fd.pth``) and checks it on download.
huggingface_hub and DVC address blobs by content hash and keep a pointer file
in git. gdown fetches Google Drive links and pins nothing. GitHub allows up to
1,000 assets per release, each under 2 GiB, with no limit on total size or
bandwidth, and records a SHA-256 ``digest`` for every uploaded asset.

The manifest format is fixed by the installer, which rejects unknown fields, so
this script writes that format and adds nothing to it. From the prior art it
takes content-addressed asset names (the torch.hub idea, with the full key
digest), a ``SHA256SUMS`` file in the ``sha256sum`` format that pooch's
registry also uses, and GitHub's recorded digests as a check after upload
(``--verify-release``). The retry schedule lives in
``scripts/fetch_reference_artifacts.py``, which ``setup.sh`` runs. None of the
libraries above is a dependency: pooch and DVC would add a second cache layout
beside the tiered cache, and the packing, hashing, and checks here use the
package's own key and envelope code.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import (
    CacheIntegrityError,
    CacheKey,
    JsonValue,
    TieredResultCache,
)
from aieng.advanced_rag.cache import paths as cache_paths
from aieng.advanced_rag.corpora import (
    contract_nli,
    cuad,
    financebench,
    multihop_rag,
    musique,
    serverfault,
)
from aieng.advanced_rag.corpora.sources import Licence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "VectorInstitute/advanced-RAG"
DEFAULT_POLICY = REPO_ROOT / "scripts" / "reference-artifact-licences.json"
MANIFEST_NAME = "reference-artifacts.json"

#: GitHub's documented limits: each file under 2 GiB, 1,000 assets per release.
MAX_ASSET_BYTES = 2 * 1024**3 - 1
MAX_ASSETS = 1000
#: ``SHA256SUMS`` and ``RELEASE_NOTES.md`` are assets too.
EXTRA_ASSETS = 2
#: Files per ``gh release upload`` command, to keep each command line short.
UPLOAD_BATCH = 40

#: A string at least this long in an artifact the policy calls text-free is
#: treated as possible corpus text. MuSiQue questions and model answers stay
#: well under it; a passage or a chunk does not.
LONG_STRING = 400
#: Members of a text-free payload with these suffixes are refused outright.
TEXT_SUFFIXES = (".txt", ".md", ".jsonl", ".csv", ".tsv", ".graphml", ".html")
#: JSON members of a text-free payload larger than this are refused unscanned.
SCAN_LIMIT_BYTES = 64 * 1024**2

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RESTRICTED = re.compile(r"\bBY-(?:NC|SA)|-NC\b|-SA\b")

#: Licences of the corpora the package pins, read from the package itself.
PACKAGE_LICENCES: dict[str, Licence] = {
    "contract_nli": contract_nli.CONTRACT_NLI_LICENCE,
    "cuad": cuad.CUAD_LICENCE,
    "financebench": financebench.FINANCEBENCH_LICENCE,
    "multihop_rag": multihop_rag.MULTIHOP_LICENCE,
    "musique": musique.MUSIQUE_LICENCE,
    "serverfault": serverfault.SERVERFAULT_LICENCE,
}


@dataclass(frozen=True)
class Target:
    """One release: an implementation folder and the cache ids it owns.

    Parameters
    ----------
    name : str
        Short name, used in the release tag.
    folder : str
        Folder under ``implementations/`` that receives the manifest.
    ri_ids : tuple of str
        Cache ``ri_id`` values whose artifacts belong to this release.
    exclude_types : tuple of str, optional
        Artifact types left out by default, with ``--include-type`` to override.
    """

    name: str
    folder: str
    ri_ids: tuple[str, ...]
    exclude_types: tuple[str, ...] = ()

    @property
    def manifest_path(self) -> Path:
        """Return where this target's manifest lives in the repository."""
        return REPO_ROOT / "implementations" / self.folder / MANIFEST_NAME


TARGETS: dict[str, Target] = {
    "1": Target("ri1", "regulatory_policy_qa", ("ri1",)),
    "2": Target("ri2", "cross_document_synthesis", ("ri2",)),
    "3": Target("ri3", "table_structured_reasoning", ("ri3",)),
    "4": Target("ri4", "graphrag_technical_support", ("ri4",)),
    # agent-question entries are the per-question resume cache behind the
    # agent-runs steps, which already hold every result. There are more than
    # 1,600 of them, past the 1,000-asset limit on a release.
    "5": Target("ri5", "agent_native_retrieval", ("ri5",), exclude_types=("agent-question",)),
    "visual-documents": Target(
        "visual-documents",
        "special_topics/visual_documents",
        ("visual-documents", "visual-documents-own"),
    ),
    "citations": Target("citations", "special_topics/citations", ("citations",)),
}


def resolve_target(raw: str) -> Target:
    """Return the target an RI number, ``riN``, folder name, or name refers to.

    Raises
    ------
    ValueError
        If ``raw`` names no target.
    """
    text = raw.strip().rstrip("/").split("/")[-1].lower()
    text = text.removeprefix("#").removeprefix("ri")
    for number, target in TARGETS.items():
        if text in (number, target.name, target.folder.split("/")[-1]):
            return target
    known = ", ".join(sorted(TARGETS))
    raise ValueError(f"unknown RI {raw!r}: use one of {known}, an RI folder, or all")


def _slug(value: str) -> str:
    """Return the cache's path slug, which is also safe as a GitHub asset name."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Collection


@dataclass
class Found:
    """One artifact found in a cache tier, with its checked digests."""

    key: CacheKey
    root: Path
    envelope: Path
    envelope_sha256: str
    envelope_bytes: int
    value: JsonValue
    payload: Path | None = None
    payload_sha256: str | None = None
    payload_bytes: int = 0
    duplicates: list[Path] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Return the bytes this artifact adds to a release."""
        return self.envelope_bytes + self.payload_bytes

    @property
    def asset_base(self) -> str:
        """Return the content-addressed asset name, without its suffix."""
        key = self.key
        return ".".join(
            (
                _slug(key.ri_id),
                _slug(key.artifact_type),
                _slug(key.producer_id),
                key.digest,
            )
        )

    @property
    def envelope_asset(self) -> str:
        """Return the envelope's asset name."""
        return f"{self.asset_base}.json"

    @property
    def payload_asset(self) -> str | None:
        """Return the payload's asset name, for a binary artifact."""
        return f"{self.asset_base}.artifact" if self.payload is not None else None

    @property
    def assets(self) -> int:
        """Return how many release assets this artifact needs."""
        return 1 if self.payload is None else 2


@dataclass
class Problem:
    """Something that stops an artifact from being published."""

    path: Path
    message: str


def _envelope_paths(root: Path, ri_ids: Iterable[str]) -> Iterator[Path]:
    """Yield the envelope files of the given RIs under one tier root."""
    for ri_id in ri_ids:
        base = root / _slug(ri_id)
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/*/*.json")):
            if _DIGEST.match(path.stem):
                yield path


def _key_from_envelope(fields: dict[str, Any]) -> CacheKey:
    return CacheKey(
        ri_id=str(fields["ri_id"]),
        artifact_type=str(fields["artifact_type"]),
        producer_id=str(fields["producer_id"]),
        config=dict(fields["config"]),
    )


def _is_binary(value: JsonValue) -> bool:
    return isinstance(value, dict) and isinstance(value.get("artifact"), dict)


def inspect(path: Path, root: Path) -> Found:
    """Read and verify one envelope, and its payload when it has one.

    Raises
    ------
    CacheIntegrityError
        If the envelope does not match its key or path, or the payload does
        not match the digest and size its envelope records.
    """
    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
        key = _key_from_envelope(fields)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CacheIntegrityError(f"unreadable envelope: {exc}") from exc
    if path.relative_to(root) != key.relative_path:
        raise CacheIntegrityError("envelope is not at the path its key gives")
    # A cache with only this root as its local tier runs the package's own
    # envelope checks: digest, metadata, schema version, and unknown fields.
    missing = root / ".no-such-tier"
    cache = TieredResultCache(local_dir=root, global_dir=missing, fetched_global_dir=missing)
    result = cache.get(key)
    if result is None or result.path != path:
        raise CacheIntegrityError("the cache does not resolve this envelope")
    found = Found(
        key=key,
        root=root,
        envelope=path,
        envelope_sha256=sha256_file(path),
        envelope_bytes=path.stat().st_size,
        value=result.value,
    )
    if _is_binary(result.value):
        descriptor = result.value["artifact"]
        payload = path.with_suffix(".artifact")
        if not payload.is_file():
            raise CacheIntegrityError(f"payload {payload.name} is missing")
        size = payload.stat().st_size
        digest = sha256_file(payload)
        if size != descriptor.get("bytes") or digest != descriptor.get("sha256"):
            raise CacheIntegrityError("payload does not match the digest and size in its envelope")
        found.payload, found.payload_sha256, found.payload_bytes = (
            payload,
            digest,
            size,
        )
    return found


def collect(
    roots: Sequence[Path],
    ri_ids: Sequence[str],
    *,
    include_types: Sequence[str] = (),
    exclude_types: Sequence[str] = (),
    keys: Sequence[str] = (),
) -> tuple[list[Found], list[Problem], list[str]]:
    """Collect verified artifacts from cache tier roots, first root first.

    Parameters
    ----------
    roots : sequence of Path
        Tier roots in precedence order. A digest found in two roots is taken
        from the first and the later copy is recorded as a duplicate.
    ri_ids : sequence of str
        Cache ``ri_id`` values to collect.
    include_types, exclude_types : sequence of str, optional
        Artifact type patterns. With ``include_types``, only matching types are
        kept; ``exclude_types`` then removes matches.
    keys : sequence of str, optional
        Key digests or unique prefixes of at least eight characters. When given,
        only these artifacts are collected.

    Returns
    -------
    tuple
        The artifacts, the problems found, and the requested keys that no root
        holds.
    """
    found: dict[str, Found] = {}
    problems: list[Problem] = []
    for root in roots:
        for path in _envelope_paths(root, ri_ids):
            artifact_type = path.parent.parent.name
            if include_types and not any(fnmatch.fnmatchcase(artifact_type, p) for p in include_types):
                continue
            if any(fnmatch.fnmatchcase(artifact_type, p) for p in exclude_types):
                continue
            if keys and not any(path.stem.startswith(k) for k in keys):
                continue
            try:
                item = inspect(path, root)
            except CacheIntegrityError as exc:
                problems.append(Problem(path, str(exc)))
                continue
            first = found.get(item.key.digest)
            if first is None:
                found[item.key.digest] = item
                continue
            first.duplicates.append(path)
            if (first.envelope_sha256, first.payload_sha256) != (
                item.envelope_sha256,
                item.payload_sha256,
            ):
                problems.append(
                    Problem(
                        path,
                        f"same key as {first.envelope} with different bytes; the first root's copy is used",
                    )
                )
    missing = [k for k in keys if not any(d.startswith(k) for d in found)]
    ordered = sorted(
        found.values(),
        key=lambda f: (
            f.key.ri_id,
            f.key.artifact_type,
            f.key.producer_id,
            f.key.digest,
        ),
    )
    return ordered, problems, missing


# ---------------------------------------------------------------------------
# Licence policy


@dataclass(frozen=True)
class Verdict:
    """The licence decision for one artifact."""

    status: str  # "ok", "allowlisted", "refused", or "unclassified"
    corpora: tuple[str, ...]
    licences: tuple[str, ...]
    corpus_text: bool | None
    reason: str

    @property
    def publishable(self) -> bool:
        """Return whether the artifact may go in a release."""
        return self.status in ("ok", "allowlisted")


@dataclass
class Policy:
    """Licence rules and allowlist read from the policy file."""

    licences: dict[str, Licence]
    rules: list[dict[str, Any]]
    allowlist: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> Policy:
        """Read a policy file.

        Raises
        ------
        ValueError
            If the file names an unknown corpus, redefines a corpus the package
            pins, or has an allowlist entry without a reason.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        licences = dict(PACKAGE_LICENCES)
        for name, entry in data.get("licences", {}).items():
            if name in licences:
                raise ValueError(f"{path}: licence for {name!r} comes from the package; do not redefine it")
            licences[name] = Licence(
                name=entry["name"],
                url=entry["url"],
                permits_commercial_use=bool(entry.get("permits_commercial_use", True)),
                attribution=entry["attribution"],
                notes=entry.get("notes", ""),
            )
        rules = list(data.get("rules", []))
        for rule in rules:
            for corpus in rule.get("corpora", []):
                if corpus not in licences:
                    raise ValueError(f"{path}: rule names unknown corpus {corpus!r}")
        allowlist = list(data.get("allowlist", []))
        for entry in allowlist:
            if not str(entry.get("reason", "")).strip():
                raise ValueError(f"{path}: every allowlist entry needs a reason")
        return cls(licences, rules, allowlist)

    @staticmethod
    def _matches(match: dict[str, Any], key: CacheKey) -> bool:
        for field_name in ("ri_id", "artifact_type", "producer_id"):
            pattern = match.get(field_name)
            if pattern is not None and not fnmatch.fnmatchcase(getattr(key, field_name), pattern):
                return False
        if "key" in match and not key.digest.startswith(str(match["key"])):
            return False
        for name, pattern in match.get("config", {}).items():
            if name not in key.config:
                return False
            if not fnmatch.fnmatchcase(str(key.config[name]), str(pattern)):
                return False
        return True

    def decide(self, item: Found) -> Verdict:
        """Return the licence verdict for one artifact."""
        rule = next(
            (r for r in self.rules if self._matches(r.get("match", {}), item.key)),
            None,
        )
        if rule is None:
            return Verdict(
                "unclassified",
                (),
                (),
                None,
                "no rule in the licence policy matches this artifact",
            )
        corpora = tuple(rule.get("corpora", []))
        licences = tuple(self.licences[c].name for c in corpora)
        corpus_text = bool(rule.get("corpus_text", True))
        if not corpus_text:
            finding = probe_text(item)
            if finding is not None:
                return Verdict(
                    "refused",
                    corpora,
                    licences,
                    False,
                    f"the policy says no corpus text, but {finding}",
                )
            return Verdict("ok", corpora, licences, False, "no corpus text")
        restricted = [f"{c} ({self.licences[c].name})" for c in corpora if _RESTRICTED.search(self.licences[c].name)]
        if not restricted:
            return Verdict("ok", corpora, licences, True, "corpus text, licence allows")
        entry = next(
            (a for a in self.allowlist if self._matches(a.get("match", {}), item.key)),
            None,
        )
        if entry is not None:
            return Verdict("allowlisted", corpora, licences, True, str(entry["reason"]).strip())
        return Verdict(
            "refused",
            corpora,
            licences,
            True,
            f"holds text under {', '.join(restricted)} and no allowlist entry",
        )


def _long_string(value: Any) -> int:
    """Return the length of the longest string in a JSON value."""
    stack, longest = [value], 0
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            longest = max(longest, len(current))
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return longest


def probe_text(item: Found) -> str | None:
    """Look for text in an artifact the policy calls text-free.

    The probe catches a rule that matches more than its author meant. It reads
    the envelope's value and, for a tar payload, refuses text-like members and
    scans the JSON ones.

    Returns
    -------
    str or None
        What the probe found, or ``None`` when it found nothing.
    """
    longest = _long_string(item.value)
    if longest >= LONG_STRING:
        return f"its value holds a {longest:,}-character string"
    if item.payload is None:
        return None
    try:
        return _probe_tar(item.payload)
    except (tarfile.TarError, ValueError) as exc:
        return f"its payload cannot be read as a tar archive of JSON ({exc})"


def _probe_tar(payload: Path) -> str | None:
    """Return what a text probe finds in a tar payload's members, if anything."""
    with tarfile.open(payload, mode="r:*") as tar:
        for member in tar.getmembers():
            name = member.name.lower()
            if not member.isfile() or not name.endswith((".json", *TEXT_SUFFIXES)):
                continue
            if name.endswith(TEXT_SUFFIXES):
                return f"its payload holds the text file {member.name}"
            if member.size > SCAN_LIMIT_BYTES:
                return f"its payload's {member.name} is too large to scan"
            handle = tar.extractfile(member)
            longest = 0 if handle is None else _long_string(json.loads(handle.read()))
            if longest >= LONG_STRING:
                return f"its payload's {member.name} holds a {longest:,}-character string"
    return None


# ---------------------------------------------------------------------------
# Manifest, staging, and commands


def release_tag(target: Target, items: Sequence[Found]) -> str:
    """Return a tag derived from the artifacts, so the same set gives the same tag."""
    digest = hashlib.sha256()
    for item in items:
        line = f"{item.key.digest} {item.envelope_sha256} {item.payload_sha256 or '-'}"
        digest.update(line.encode() + b"\n")
    return f"reference-{target.name}-{digest.hexdigest()[:12]}"


def download_base(repo: str, tag: str) -> str:
    """Return the URL prefix GitHub serves a release's assets from."""
    return f"https://github.com/{repo}/releases/download/{tag}/"


def manifest_entries(items: Sequence[Found], base_url: str) -> list[dict[str, Any]]:
    """Return manifest entries in the installer's format, in a fixed order."""
    entries: list[dict[str, Any]] = []
    for item in items:
        entry: dict[str, Any] = {
            "key": {
                "ri_id": item.key.ri_id,
                "artifact_type": item.key.artifact_type,
                "producer_id": item.key.producer_id,
                "config": item.key.config,
            },
            "url": base_url + item.envelope_asset,
            "sha256": item.envelope_sha256,
        }
        if item.payload is not None:
            entry["payload_url"] = base_url + str(item.payload_asset)
            entry["payload_sha256"] = item.payload_sha256
        entries.append(entry)
    return entries


def render_manifest(entries: list[dict[str, Any]]) -> str:
    """Render a manifest so the same entries always give the same bytes."""
    return json.dumps({"artifacts": entries}, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _place(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination``, or copy it across filesystems."""
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and sha256_file(destination) == sha256_file(source):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def release_notes(target: Target, items: Sequence[Found], verdicts: dict[str, Verdict], policy: Policy) -> str:
    """Return the release notes: what the release holds and whose text it carries."""
    total = sum(i.total_bytes for i in items)
    assets = sum(i.assets for i in items)
    lines = [
        f"# Reference artifacts for {target.folder}",
        "",
        f"{len(items)} artifacts in {assets} files, {_size(total)}. Install them with:",
        "",
        "```bash",
        f"python -m aieng.advanced_rag.cache implementations/{target.folder}/{MANIFEST_NAME}",
        "```",
        "",
        "Each file name ends in the digest of the cache key it is installed under, and "
        "`SHA256SUMS` lists the SHA-256 of every file.",
        "",
        "| Artifact type | Artifacts | Size | Corpora | Licence decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    # One row per type and licence decision: artifacts of one type can come from
    # different corpora, and a row that took its first member's verdict would
    # understate what the release carries.
    groups: dict[tuple[str, str, tuple[str, ...], str], list[Found]] = {}
    for item in items:
        verdict = verdicts[item.key.digest]
        group = (item.key.ri_id, item.key.artifact_type, verdict.corpora, verdict.reason)
        groups.setdefault(group, []).append(item)
    for (ri_id, artifact_type, corpora_used, reason), members in groups.items():
        corpora = ", ".join(corpora_used) or "none"
        lines.append(
            f"| `{ri_id}/{artifact_type}` | {len(members)} | "
            f"{_size(sum(m.total_bytes for m in members))} | {corpora} | {reason} |"
        )
    # Only the corpora of what ships: a refused artifact's corpus is not in the release.
    used = sorted({c for i in items for c in verdicts[i.key.digest].corpora})
    if used:
        lines += ["", "## Attribution", ""]
        for corpus in used:
            licence = policy.licences[corpus]
            lines.append(f"- {licence.attribution} Licence: {licence.name}, {licence.url}.")
            if corpus in ATTRIBUTION_NOTES:
                lines.append(f"  {ATTRIBUTION_NOTES[corpus]}")
    return "\n".join(lines) + "\n"


SERVERFAULT_ATTRIBUTION = "ATTRIBUTION-serverfault.md"


def _md_link(text: str, url: str | None) -> str:
    # Escape what Markdown would read as table, link, or emphasis syntax in a name.
    label = re.sub(r"([\\`*_\[\]|<>])", r"\\\1", " ".join(text.split()))
    return f"[{label}]({url})" if url else label


def serverfault_attribution(payload: Mapping[str, Any], names: Mapping[int, str] | None = None) -> str:
    """Return the attribution file for posts from a ServerFault slice.

    Stack Exchange asks reusers to say the content comes from the network, link
    each question, name the author of every question and answer, and link each
    name to the author's profile (https://stackoverflow.blog/2009/06/25/attribution-required/).
    One row per question names both posts with their licence versions, which
    Stack Exchange sets by contribution date (https://stackoverflow.com/help/licensing).
    ``names`` maps user ids to display names from ``Users.xml``; a post whose author
    has no entry keeps the name the slice recorded.
    """
    names = names or {}

    def author(owner: Mapping[str, Any]) -> str:
        user_id = owner.get("user_id")
        name = names.get(int(user_id)) if user_id is not None else None
        return _md_link(name or str(owner.get("display_name", "")), owner.get("link"))

    questions = payload["questions"]
    lines = [
        "# ServerFault attribution",
        "",
        "The ServerFault-derived artifacts in this release are built from questions and "
        f"accepted answers posted on [Server Fault]({serverfault.SITE_URL}), a site of the "
        "[Stack Exchange network](https://stackexchange.com), taken from the Stack Exchange "
        f"data dump published {payload.get('dump_published', serverfault.DUMP_PUBLISHED)} "
        "(https://archive.org/details/stackexchange). Each post is licensed by its author "
        "under the Creative Commons Attribution-ShareAlike version in the table "
        "([CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/), "
        "[3.0](https://creativecommons.org/licenses/by-sa/3.0/), or "
        "[4.0](https://creativecommons.org/licenses/by-sa/4.0/)). The artifacts change the "
        "posts: they split them into chunks, extract entities and relations, and quote them "
        "in model transcripts. They are shared under CC BY-SA 4.0, which each of those "
        "versions allows for adaptations.",
        "",
        f"The table lists all {len(questions):,} questions in the slice the artifacts were "
        "built from, with the accepted answer kept for each.",
        "",
        "| Question | Question author | Question licence | Accepted answer | Answer author | Answer licence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for question in questions:
        owner = question.get("owner") or {}
        row = [
            _md_link(str(question["question_id"]), question["link"]),
            author(owner),
            str(question.get("content_license", "")),
        ]
        answers = [a for a in question.get("answers", []) if a.get("is_accepted")]
        if answers:
            answer = answers[0]
            row += [
                _md_link(str(answer["answer_id"]), f"{serverfault.SITE_URL}/a/{answer['answer_id']}"),
                author(answer.get("owner") or {}),
                str(answer.get("content_license", "")),
            ]
        else:
            row += ["none kept", "", ""]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _serverfault_file() -> str:
    # Every ServerFault-derived artifact so far is built from the 10,000-question
    # slice; fetch_slice checks it against its recorded digest.
    payload = serverfault.fetch_slice()
    return serverfault_attribution(payload, serverfault.fetch_user_names(payload))


#: Per-post attribution files a release carries when it holds a corpus's text.
ATTRIBUTION_FILES: dict[str, tuple[str, Callable[[], str]]] = {
    "serverfault": (SERVERFAULT_ATTRIBUTION, _serverfault_file),
}
ATTRIBUTION_NOTES = {
    "serverfault": (
        f"`{SERVERFAULT_ATTRIBUTION}` links every post and names and links its author, as "
        "Stack Exchange requires. Artifacts derived from the posts are shared under CC BY-SA 4.0."
    ),
    "financebench": (
        "Artifacts derived from the FinanceBench filings, page indexes included, are shared "
        "under CC BY-NC 4.0 and only for non-commercial use."
    ),
}


def attribution_files(items: Sequence[Found], verdicts: dict[str, Verdict]) -> dict[str, str]:
    """Return the attribution files ``items`` need, by file name."""
    used = {c for i in items for c in verdicts[i.key.digest].corpora}
    return {name: write() for corpus, (name, write) in ATTRIBUTION_FILES.items() if corpus in used}


def stage(
    stage_dir: Path,
    items: Sequence[Found],
    notes: str,
    extras: Mapping[str, str] | None = None,
) -> list[Path]:
    """Place every asset in ``stage_dir`` and write ``SHA256SUMS`` and the notes.

    ``extras`` maps further file names, such as attribution files, to their text.

    Returns
    -------
    list of Path
        The files to upload, in upload order.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    sums: list[str] = []
    for item in items:
        pairs = [(item.envelope, item.envelope_asset, item.envelope_sha256)]
        if item.payload is not None:
            pairs.append((item.payload, str(item.payload_asset), str(item.payload_sha256)))
        for source, name, digest in pairs:
            destination = stage_dir / name
            _place(source, destination)
            files.append(destination)
            sums.append(f"{digest}  {name}")
    for name, text in (extras or {}).items():
        destination = stage_dir / name
        destination.write_text(text, encoding="utf-8")
        files.append(destination)
        sums.append(f"{sha256_file(destination)}  {name}")
    sums_path = stage_dir / "SHA256SUMS"
    sums_path.write_text("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n")
    notes_path = stage_dir / "RELEASE_NOTES.md"
    notes_path.write_text(notes, encoding="utf-8")
    return [*files, sums_path, notes_path]


def upload_commands(repo: str, tag: str, files: Sequence[Path]) -> list[list[str]]:
    """Return ``gh release upload`` commands in batches, run from the stage directory.

    The asset name is the file's base name, so each command names files
    relative to the directory they were staged in.
    """
    names = [path.name for path in files]
    return [
        ["gh", "release", "upload", tag, "--repo", repo, *names[i : i + UPLOAD_BATCH]]
        for i in range(0, len(names), UPLOAD_BATCH)
    ]


def create_command(repo: str, tag: str, target: Target, notes: Path) -> list[str]:
    """Return the ``gh release create`` command, which this script only prints."""
    return [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        repo,
        "--draft",
        "--title",
        f"Reference artifacts: {target.folder}",
        "--notes-file",
        str(notes),
    ]


def verify_release(repo: str, tag: str, manifest: Path) -> list[str]:
    """Compare GitHub's recorded asset digests with a manifest, reading only.

    Returns
    -------
    list of str
        One line per asset that is missing or whose digest differs.
    """
    # The tag endpoint returns 404 for a draft, which is when this check runs, so
    # find the release in the list and page through its assets separately.
    release_id = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases", "--paginate", "--jq", f'.[] | select(.tag_name == "{tag}") | .id'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not release_id:
        return [f"no release tagged {tag} on {repo}"]
    output = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/{release_id[0]}/assets", "--paginate", "--jq", ".[]"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assets = {a["name"]: a for a in map(json.loads, output.splitlines())}
    errors: list[str] = []
    for entry in json.loads(manifest.read_text(encoding="utf-8"))["artifacts"]:
        pairs = [(entry["url"], entry["sha256"])]
        if "payload_url" in entry:
            pairs.append((entry["payload_url"], entry["payload_sha256"]))
        for url, digest in pairs:
            name = url.rsplit("/", 1)[-1]
            asset = assets.get(name)
            if asset is None:
                errors.append(f"missing on the release: {name}")
            elif asset.get("digest") not in (None, f"sha256:{digest}"):
                errors.append(f"digest differs: {name} is {asset.get('digest')}")
    return errors


# ---------------------------------------------------------------------------
# Reporting


def _size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1000
    return f"{value:,.1f} GB"


@dataclass
class TargetReport:
    """What one target's run found, for printing and for ``--report``."""

    target: Target
    items: list[Found]
    verdicts: dict[str, Verdict]
    problems: list[Problem]
    missing_keys: list[str]
    manifest_gaps: list[str]
    limit_errors: list[str]

    @property
    def publishable(self) -> list[Found]:
        """Return the artifacts the licence policy lets through."""
        return [i for i in self.items if self.verdicts[i.key.digest].publishable]

    @property
    def refused(self) -> list[Found]:
        """Return the artifacts the licence policy stops."""
        return [i for i in self.items if not self.verdicts[i.key.digest].publishable]

    def as_json(self) -> dict[str, Any]:
        """Return a summary with no corpus text in it."""
        groups: dict[str, dict[str, Any]] = {}
        for item in self.items:
            verdict = self.verdicts[item.key.digest]
            name = f"{item.key.ri_id}/{item.key.artifact_type}"
            group = groups.setdefault(
                name,
                {
                    "artifacts": 0,
                    "assets": 0,
                    "bytes": 0,
                    "statuses": {},
                    "corpora": sorted(verdict.corpora),
                    "licences": sorted(set(verdict.licences)),
                },
            )
            group["artifacts"] += 1
            group["assets"] += item.assets
            group["bytes"] += item.total_bytes
            group["statuses"][verdict.status] = group["statuses"].get(verdict.status, 0) + 1
            group["corpora"] = sorted({*group["corpora"], *verdict.corpora})
            group["licences"] = sorted({*group["licences"], *verdict.licences})
            group["reasons"] = sorted({*group.get("reasons", []), verdict.reason})
        publishable = self.publishable
        return {
            "target": self.target.name,
            "folder": self.target.folder,
            "ri_ids": list(self.target.ri_ids),
            "excluded_types": list(self.target.exclude_types),
            "artifacts": len(self.items),
            "assets": sum(i.assets for i in self.items),
            "bytes": sum(i.total_bytes for i in self.items),
            "publishable_artifacts": len(publishable),
            "publishable_assets": sum(i.assets for i in publishable),
            "publishable_bytes": sum(i.total_bytes for i in publishable),
            "refused_artifacts": len(self.refused),
            "types": groups,
            "problems": [f"{p.path}: {p.message}" for p in self.problems],
            "missing_keys": self.missing_keys,
            "manifest_gaps": self.manifest_gaps,
            "limit_errors": self.limit_errors,
        }


def print_report(report: TargetReport) -> None:
    """Print one target's summary."""
    data = report.as_json()
    print(
        f"== {report.target.name} ({report.target.folder}): "
        f"{data['artifacts']} artifacts, {data['assets']} assets, {_size(data['bytes'])}; "
        f"publishable {data['publishable_artifacts']} artifacts, "
        f"{data['publishable_assets']} assets, {_size(data['publishable_bytes'])}"
    )
    if report.target.exclude_types:
        print(f"   left out by default: {', '.join(report.target.exclude_types)}")
    for name, group in data["types"].items():
        statuses = ", ".join(f"{k} {v}" for k, v in sorted(group["statuses"].items()))
        licences = "; ".join(group["licences"]) or "no corpus"
        print(
            f"   {name}: {group['artifacts']} artifacts, {group['assets']} assets, "
            f"{_size(group['bytes'])} [{statuses}] {licences}"
        )
        for reason in group["reasons"]:
            print(f"      {reason}")
    for line in data["problems"]:
        print(f"   PROBLEM {line}")
    for key in report.missing_keys:
        print(f"   MISSING key {key}")
    for line in report.manifest_gaps:
        print(f"   MANIFEST {line}")
    for line in report.limit_errors:
        print(f"   LIMIT {line}")


def check_limits(items: Sequence[Found]) -> list[str]:
    """Return every way ``items`` would break GitHub's release limits."""
    errors = [
        f"{path.name} is {_size(size)}, over the 2 GiB asset limit"
        for item in items
        for path, size in (
            (item.envelope, item.envelope_bytes),
            (item.payload, item.payload_bytes),
        )
        if path is not None and size > MAX_ASSET_BYTES
    ]
    assets = sum(i.assets for i in items) + EXTRA_ASSETS
    if assets > MAX_ASSETS:
        errors.append(
            f"{assets} assets, over the {MAX_ASSETS}-asset limit on a release; "
            "narrow it with --include-type or --exclude-type"
        )
    return errors


def manifest_gaps(target: Target, items: Sequence[Found]) -> list[str]:
    """Compare an existing manifest with what the cache holds."""
    path = target.manifest_path
    if not path.is_file():
        return []
    present = {i.key.digest: i for i in items}
    gaps: list[str] = []
    for entry in json.loads(path.read_text(encoding="utf-8")).get("artifacts", []):
        key = CacheKey(**entry["key"])
        item = present.get(key.digest)
        if item is None:
            gaps.append(f"in the manifest, not in the cache: {key.relative_path}")
        elif item.envelope_sha256 != entry["sha256"] or item.payload_sha256 != entry.get("payload_sha256"):
            gaps.append(f"cache bytes differ from the manifest: {key.relative_path}")
    return gaps


# ---------------------------------------------------------------------------
# Command line


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/publish_artifacts.py",
        description=(
            "Stage an RI's reference artifacts as GitHub release assets and write "
            "its checksum-pinned fetch manifest. Uploads only with --upload."
        ),
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="RI numbers (1 to 5), RI folder names, visual-documents, or all.",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Cache key digest, or a unique prefix of 8 or more characters. Repeatable.",
    )
    parser.add_argument(
        "--cache-root",
        action="append",
        type=Path,
        default=[],
        help="Tier root to collect from, in precedence order. Repeatable. Default: the package's local tier.",
    )
    parser.add_argument("--include-type", action="append", default=[])
    parser.add_argument("--exclude-type", action="append", default=[])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", help="Release tag. Default: derived from the artifacts.")
    parser.add_argument(
        "--base-url",
        help="URL prefix for the manifest. Default: the release's download URL.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="Where to stage assets. Default: <cache root>/publish/<tag>.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Where to write the manifest. Default: implementations/<ri>/reference-artifacts.json.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check presence, integrity, licences, and limits, and write nothing.",
    )
    parser.add_argument(
        "--skip-refused",
        action="store_true",
        help="Leave refused artifacts out instead of stopping.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Run the gh release upload commands. The release must already exist.",
    )
    parser.add_argument(
        "--verify-release",
        action="store_true",
        help="Compare GitHub's asset digests with the manifest, reading only.",
    )
    parser.add_argument("--report", type=Path, help="Write the summary as JSON here.")
    arguments = parser.parse_args(argv)
    if not arguments.targets and not arguments.key:
        parser.error("name an RI or pass --key")
    if arguments.dry_run and arguments.upload:
        parser.error("--dry-run and --upload cannot be combined")
    return arguments


def _targets(arguments: argparse.Namespace, roots: Sequence[Path]) -> list[Target]:
    if any(t.lower() == "all" for t in arguments.targets):
        return list(TARGETS.values())
    if arguments.targets:
        return [resolve_target(t) for t in arguments.targets]
    # Keys alone: find the RI each one belongs to.
    owners: set[str] = set()
    for target in TARGETS.values():
        for root in roots:
            for path in _envelope_paths(root, target.ri_ids):
                if any(path.stem.startswith(k) for k in arguments.key):
                    owners.add(target.name)
    return [t for t in TARGETS.values() if t.name in owners] or [resolve_target(arguments.key[0])]


def _report_target(
    arguments: argparse.Namespace, policy: Policy, roots: Sequence[Path], target: Target
) -> tuple[TargetReport, bool]:
    """Collect and judge one target's artifacts, print them, and say if blocked."""
    excluded = [*arguments.exclude_type]
    if not arguments.include_type:
        excluded += list(target.exclude_types)
    items, problems, missing = collect(
        roots,
        target.ri_ids,
        include_types=arguments.include_type,
        exclude_types=excluded,
        keys=arguments.key,
    )
    report = TargetReport(
        target=target,
        items=items,
        verdicts={i.key.digest: policy.decide(i) for i in items},
        problems=problems,
        missing_keys=missing,
        manifest_gaps=manifest_gaps(target, items),
        limit_errors=[],
    )
    report.limit_errors = check_limits(report.publishable if arguments.skip_refused else items)
    print_report(report)
    blocked = bool(problems or missing or report.limit_errors)
    if report.refused and not arguments.skip_refused:
        print(
            f"   {len(report.refused)} artifacts refused by the licence policy; "
            "add allowlist entries or pass --skip-refused"
        )
        blocked = True
    return report, blocked


def _publish_target(arguments: argparse.Namespace, policy: Policy, target: Target, report: TargetReport) -> int:
    """Stage one target, write its manifest, and print or run the upload commands."""
    selected = report.publishable
    if not selected:
        print(f"   nothing to publish for {target.name}")
        return 0
    tag = arguments.tag or release_tag(target, selected)
    base_url = arguments.base_url or download_base(arguments.repo, tag)
    if not base_url.endswith("/"):
        base_url += "/"
    manifest_path = arguments.manifest or target.manifest_path
    stage_dir = arguments.stage_dir or cache_paths.root() / "publish" / tag
    files = stage(
        stage_dir,
        selected,
        release_notes(target, selected, report.verdicts, policy),
        attribution_files(selected, report.verdicts),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(render_manifest(manifest_entries(selected, base_url)), encoding="utf-8")
    print(f"   staged {len(files)} files in {stage_dir}")
    print(f"   wrote {manifest_path}")
    print("   Create the release (this script never does):")
    print("     " + shlex.join(create_command(arguments.repo, tag, target, files[-1])))
    commands = upload_commands(arguments.repo, tag, files)
    if not arguments.upload:
        print("   Upload commands (pass --upload to run them):")
        print("     cd " + shlex.quote(str(stage_dir)))
        for command in commands:
            print("     " + shlex.join(command))
    else:
        for command in commands:
            print(f"   running gh release upload {tag}: {len(command) - 6} files")
            subprocess.run(command, check=True, cwd=stage_dir)
    if not (arguments.upload or arguments.verify_release):
        return 0
    errors = verify_release(arguments.repo, tag, manifest_path)
    for line in errors:
        print(f"   VERIFY {line}")
    return 1 if errors else 0


def run(argv: Sequence[str] | None = None) -> int:
    """Run the command line, returning the exit status."""
    arguments = _parse_args(argv)
    roots = arguments.cache_root or [cache_paths.artifacts_dir()]
    if any(len(k) < 8 for k in arguments.key):
        print("--key needs at least 8 characters of the digest", file=sys.stderr)
        return 2
    policy = Policy.load(arguments.policy)
    try:
        targets = _targets(arguments, roots)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if (
        len(targets) > 1
        and not arguments.dry_run
        and (arguments.tag or arguments.manifest or arguments.stage_dir or arguments.base_url)
    ):
        print("--tag, --manifest, --stage-dir, and --base-url take one RI", file=sys.stderr)
        return 2

    status = 0
    summaries: list[dict[str, Any]] = []
    for target in targets:
        report, blocked = _report_target(arguments, policy, roots, target)
        summaries.append(report.as_json())
        if blocked:
            status = 1
        elif not arguments.dry_run:
            status = max(status, _publish_target(arguments, policy, target, report))
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps({"roots": [str(r) for r in roots], "targets": summaries}, indent=2) + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    raise SystemExit(run())
