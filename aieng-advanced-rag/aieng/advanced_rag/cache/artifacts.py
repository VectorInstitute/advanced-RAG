"""Two-tier, content-addressed artifact cache for Advanced RAG RIs.

Tier 1 contains committed artifacts for the standard notebook configuration.
Tier 2 contains user-local results produced after a configuration change. Both
tiers use the same key: RI id, artifact type, producer/predictor id, and a
canonical spec. Atomic writes and item-level keys make interrupted multi-item
runs resumable: every completed item is durable before the next one starts.

Lookup runs local, then fetched-global, then shipped-global:

- ``local``, under :func:`aieng.advanced_rag.cache.paths.artifacts_dir`, holds
  what this machine computed after a configuration change or an explicit
  refresh.
- ``fetched-global``, under ``local/fetched-global``, holds checksum-pinned
  standard artifacts too large to commit, installed by
  :meth:`TieredResultCache.fetch_manifest`.
- ``global``, the ``precomputed`` directory beside this module, holds standard
  artifacts distributed with the package. It ships empty.

The tier is not part of the digest, so a local artifact overrides the
corresponding standard artifact without changing its logical identity, and a
caller who changed nothing still gets the standard artifact immediately.

Every resolved hit is reported to :mod:`aieng.advanced_rag.metering` through
:func:`~aieng.advanced_rag.metering.report_cache_hit`, so a warm re-run reads as
comparable with the cold run it repeats rather than as a pipeline that became
free. Outside a metered context the report does nothing.

Cache invalidation is explicit: a caller is responsible for naming in its
configuration every input that would change the result. Nothing expires and
nothing is evicted. Atomic replacement stops a partial file being read as a
hit, but it does not stop two processes computing the same missing key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, cast

from aieng.advanced_rag import metering
from aieng.advanced_rag.cache.paths import artifacts_dir


# A recursive alias, quoted because the name is not bound until the statement
# finishes. PEP 695 `type JsonValue = ...` would say this without the quotes,
# but it needs 3.12 and this package supports 3.11.
JsonValue: TypeAlias = (
    "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
)
CacheTier: TypeAlias = Literal["global", "local"]
_SCHEMA_VERSION: Literal[1] = 1
_PACKAGE_GLOBAL_DIR = Path(__file__).parent / "precomputed"
_SENSITIVE_KEYS = {"api_key", "apikey", "password", "secret", "token"}
_SENSITIVE_SUFFIXES = ("_api_key", "_password", "_secret", "_token")


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_secret_keys(value: JsonValue, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in _SENSITIVE_KEYS or normalized_key.endswith(
                _SENSITIVE_SUFFIXES
            ):
                raise ValueError(
                    f"cache configuration must not contain secret field {path}.{key}"
                )
            _reject_secret_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_keys(child, f"{path}[{index}]")


def digest_mismatch(payload: bytes, expected_sha256: str) -> str | None:
    """Return the digest of ``payload`` when it is not the expected one.

    `corpora.sources.verify` does the same comparison and raises
    `ChecksumMismatchError`, a `RuntimeError`, while the fetch paths below raise
    `CacheIntegrityError`, a `ValueError`. Only the comparison is shared. The
    two error types stay distinct because their bases differ, so collapsing
    them would change which `except` clause catches a bad download, and
    `tests/aieng/advanced_rag/corpora/test_sources.py` catches the corpus one by
    name.

    Parameters
    ----------
    payload : bytes
        The bytes that were downloaded or read.
    expected_sha256 : str
        The digest those bytes are pinned to.

    Returns
    -------
    str or None
        The actual SHA-256 hex digest when it differs from
        ``expected_sha256``, and ``None`` when the bytes match.
    """
    actual = hashlib.sha256(payload).hexdigest()
    return None if actual == expected_sha256 else actual


_HASH_BLOCK_BYTES = 1 << 20


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read one block at a time.

    Payloads can be hundreds of megabytes, so the file is never held whole.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()
    if not slug:
        raise ValueError("cache key labels must contain at least one letter or number")
    return slug


@dataclass(frozen=True)
class CacheKey:
    """Stable identity for one artifact produced from one RI configuration."""

    ri_id: str
    artifact_type: str
    producer_id: str
    config: dict[str, JsonValue]

    def __post_init__(self) -> None:
        """Freeze a canonical JSON copy and reject accidental credentials."""
        normalized = cast(
            dict[str, JsonValue], json.loads(_canonical_json(self.config))
        )
        _reject_secret_keys(normalized)
        object.__setattr__(self, "config", normalized)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the complete canonical key."""
        identity: dict[str, JsonValue] = {
            "artifact_type": self.artifact_type,
            "config": self.config,
            "producer_id": self.producer_id,
            "ri_id": self.ri_id,
            "schema_version": _SCHEMA_VERSION,
        }
        return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()

    @property
    def relative_path(self) -> Path:
        """Return the common path used by the global and local tiers."""
        return (
            Path(_slug(self.ri_id))
            / _slug(self.artifact_type)
            / _slug(self.producer_id)
            / f"{self.digest}.json"
        )


class CacheIntegrityError(ValueError):
    """Raised when a cache file does not match its content-addressed key."""


@dataclass(frozen=True)
class CacheEnvelope:
    """On-disk representation of one cached value.

    Parameters
    ----------
    cache_key : str
        Digest of the key the value was stored under.
    ri_id : str
        RI the value belongs to.
    artifact_type : str
        What kind of artifact the value is.
    producer_id : str
        Producer or predictor that computed the value.
    config : dict
        Canonical configuration the value was computed from.
    created_at : str
        ISO-8601 timestamp of the write.
    value : JsonValue
        The cached value, or the descriptor of an opaque payload.
    schema_version : int, optional
        On-disk format version. Only ``1`` is readable.
    """

    cache_key: str
    ri_id: str
    artifact_type: str
    producer_id: str
    config: dict[str, JsonValue]
    created_at: str
    value: JsonValue
    schema_version: Literal[1] = _SCHEMA_VERSION


@dataclass(frozen=True)
class GlobalArtifact:
    """One oversized Tier-1 artifact available from a verified remote URL.

    Parameters
    ----------
    key : CacheKey
        Identity the fetched artifact is installed under.
    url : str
        Location of the JSON envelope.
    sha256 : str
        Digest of the bytes at ``url``.
    payload_url : str or None, optional
        Location of the opaque payload, for a binary artifact.
    payload_sha256 : str or None, optional
        Digest of the bytes at ``payload_url``.
    """

    key: CacheKey
    url: str
    sha256: str
    payload_url: str | None = None
    payload_sha256: str | None = None


@dataclass(frozen=True)
class GlobalArtifactManifest:
    """Fetch manifest for Tier-1 artifacts that are too large to commit.

    Parameters
    ----------
    artifacts : list of GlobalArtifact
        Every artifact the manifest offers, in fetch order.
    """

    artifacts: list[GlobalArtifact]


_ENVELOPE_FIELDS = (
    "schema_version",
    "cache_key",
    "ri_id",
    "artifact_type",
    "producer_id",
    "config",
    "created_at",
    "value",
)
_KEY_FIELDS = ("ri_id", "artifact_type", "producer_id", "config")
_ARTIFACT_FIELDS = ("key", "url", "sha256", "payload_url", "payload_sha256")
_MANIFEST_FIELDS = ("artifacts",)


def _parse_object(content: str | bytes, what: str) -> dict[str, JsonValue]:
    """Decode JSON text and require a top-level object."""
    try:
        decoded = json.loads(content)
    except ValueError as exc:
        raise CacheIntegrityError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise CacheIntegrityError(f"{what} must be a JSON object")
    return cast("dict[str, JsonValue]", decoded)


def _reject_unknown_fields(
    source: dict[str, JsonValue], known: tuple[str, ...], what: str
) -> None:
    """Reject any field this reader does not know about.

    A reader that drops an unrecognised field loads a value a writer thought it
    had stored in full, so an envelope from a newer writer fails here instead.
    """
    unknown = sorted(set(source) - set(known))
    if unknown:
        raise CacheIntegrityError(f"{what} has unknown fields: {', '.join(unknown)}")


def _require_str(source: dict[str, JsonValue], field: str, what: str) -> str:
    """Return a required string field."""
    value = source.get(field)
    if not isinstance(value, str):
        raise CacheIntegrityError(f"{what} field {field} must be a string")
    return value


def _optional_str(source: dict[str, JsonValue], field: str, what: str) -> str | None:
    """Return an optional string field, absent or null reading as ``None``."""
    value = source.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CacheIntegrityError(f"{what} field {field} must be a string or null")
    return value


def _require_config(
    source: dict[str, JsonValue], field: str, what: str
) -> dict[str, JsonValue]:
    """Return a required configuration object."""
    value = source.get(field)
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"{what} field {field} must be an object")
    return value


def _require_schema_version(source: dict[str, JsonValue], what: str) -> Literal[1]:
    """Return the on-disk format version, rejecting one this reader cannot read."""
    value = source.get("schema_version", _SCHEMA_VERSION)
    if value != _SCHEMA_VERSION:
        raise CacheIntegrityError(
            f"{what} has schema_version {value!r}, expected {_SCHEMA_VERSION}"
        )
    return _SCHEMA_VERSION


def _parse_envelope(content: str | bytes, source: str | Path) -> CacheEnvelope:
    """Parse and validate one on-disk cache envelope."""
    what = f"Cache envelope at {source}"
    fields = _parse_object(content, what)
    _reject_unknown_fields(fields, _ENVELOPE_FIELDS, what)
    if "value" not in fields:
        raise CacheIntegrityError(f"{what} is missing its value")
    return CacheEnvelope(
        cache_key=_require_str(fields, "cache_key", what),
        ri_id=_require_str(fields, "ri_id", what),
        artifact_type=_require_str(fields, "artifact_type", what),
        producer_id=_require_str(fields, "producer_id", what),
        config=_require_config(fields, "config", what),
        created_at=_require_str(fields, "created_at", what),
        value=fields["value"],
        schema_version=_require_schema_version(fields, what),
    )


def _parse_key(fields: JsonValue, what: str) -> CacheKey:
    """Parse and validate one manifest entry's cache key."""
    if not isinstance(fields, dict):
        raise CacheIntegrityError(f"{what} field key must be an object")
    _reject_unknown_fields(fields, _KEY_FIELDS, f"{what} key")
    return CacheKey(
        ri_id=_require_str(fields, "ri_id", f"{what} key"),
        artifact_type=_require_str(fields, "artifact_type", f"{what} key"),
        producer_id=_require_str(fields, "producer_id", f"{what} key"),
        config=_require_config(fields, "config", f"{what} key"),
    )


def _parse_global_artifact(fields: JsonValue, what: str) -> GlobalArtifact:
    """Parse and validate one manifest entry."""
    if not isinstance(fields, dict):
        raise CacheIntegrityError(f"{what} entries must be objects")
    _reject_unknown_fields(fields, _ARTIFACT_FIELDS, what)
    return GlobalArtifact(
        key=_parse_key(fields.get("key"), what),
        url=_require_str(fields, "url", what),
        sha256=_require_str(fields, "sha256", what),
        payload_url=_optional_str(fields, "payload_url", what),
        payload_sha256=_optional_str(fields, "payload_sha256", what),
    )


def _parse_manifest(content: str | bytes, source: str | Path) -> GlobalArtifactManifest:
    """Parse and validate a fetch manifest."""
    what = f"Fetch manifest at {source}"
    fields = _parse_object(content, what)
    _reject_unknown_fields(fields, _MANIFEST_FIELDS, what)
    artifacts = fields.get("artifacts")
    if not isinstance(artifacts, list):
        raise CacheIntegrityError(f"{what} field artifacts must be a list")
    return GlobalArtifactManifest(
        artifacts=[_parse_global_artifact(entry, what) for entry in artifacts]
    )


def _serialize_envelope(envelope: CacheEnvelope) -> bytes:
    """Render one envelope in the field order the on-disk format fixed.

    The order is the one pydantic's ``model_dump_json`` produced, and a
    published manifest pins the SHA-256 of these exact bytes, so an envelope
    written before this serialiser still has to round-trip byte for byte.
    """
    payload: dict[str, JsonValue] = {
        "schema_version": envelope.schema_version,
        "cache_key": envelope.cache_key,
        "ri_id": envelope.ri_id,
        "artifact_type": envelope.artifact_type,
        "producer_id": envelope.producer_id,
        "config": envelope.config,
        "created_at": envelope.created_at,
        "value": envelope.value,
    }
    rendered = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
    return rendered.encode() + b"\n"


@dataclass(frozen=True)
class CacheResult:
    """Cached value plus provenance about how it was resolved."""

    value: JsonValue
    tier: CacheTier
    cache_hit: bool
    path: Path


@dataclass(frozen=True)
class ArtifactResult:
    """Cached binary artifact plus its JSON metadata and provenance."""

    metadata: JsonValue
    tier: CacheTier
    cache_hit: bool
    path: Path
    envelope_path: Path


class TieredResultCache:
    """Resolve global artifacts first-run-fast and store user changes locally.

    Lookup order is local, fetched-global, then shipped-global. A local result
    therefore overrides a committed artifact after an explicit refresh, while a
    new caller still receives the committed default immediately.
    """

    def __init__(
        self,
        *,
        global_dir: Path | None = None,
        local_dir: Path | None = None,
        fetched_global_dir: Path | None = None,
    ) -> None:
        """Configure physical stores without creating directories eagerly."""
        self.global_dir = (
            Path(global_dir) if global_dir is not None else _PACKAGE_GLOBAL_DIR
        )
        self.local_dir = Path(local_dir) if local_dir is not None else artifacts_dir()
        self.fetched_global_dir = (
            Path(fetched_global_dir)
            if fetched_global_dir is not None
            else self.local_dir / "fetched-global"
        )

    @staticmethod
    def _envelope(key: CacheKey, value: JsonValue) -> CacheEnvelope:
        return CacheEnvelope(
            cache_key=key.digest,
            ri_id=key.ri_id,
            artifact_type=key.artifact_type,
            producer_id=key.producer_id,
            config=key.config,
            created_at=datetime.now(timezone.utc).isoformat(),
            value=value,
        )

    @staticmethod
    def _decode(content: str | bytes, key: CacheKey, source: str | Path) -> JsonValue:
        envelope = _parse_envelope(content, source)
        if envelope.cache_key != key.digest:
            raise CacheIntegrityError(f"Cache key mismatch at {source}")
        if (
            envelope.ri_id != key.ri_id
            or envelope.artifact_type != key.artifact_type
            or envelope.producer_id != key.producer_id
            or envelope.config != key.config
        ):
            raise CacheIntegrityError(f"Cache metadata mismatch at {source}")
        return envelope.value

    @classmethod
    def _load(cls, path: Path, key: CacheKey) -> JsonValue:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CacheIntegrityError(
                f"Cannot read cache artifact at {path}: {exc}"
            ) from exc
        return cls._decode(content, key, path)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_name = handle.name
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _write(self, key: CacheKey, value: JsonValue, tier: CacheTier) -> Path:
        root = self.local_dir if tier == "local" else self.global_dir
        path = root / key.relative_path
        payload = _serialize_envelope(self._envelope(key, value))
        self._atomic_write(path, payload)
        return path

    @staticmethod
    def _payload_relative_path(key: CacheKey) -> Path:
        """Return the payload path paired with a key's JSON envelope."""
        return key.relative_path.with_suffix(".artifact")

    @staticmethod
    def _artifact_value(
        path: Path, metadata: JsonValue, media_type: str
    ) -> dict[str, JsonValue]:
        """Describe one opaque payload without placing it inside JSON."""
        content_sha256 = _sha256_of_file(path)
        return {
            "artifact": {
                "bytes": path.stat().st_size,
                "media_type": media_type,
                "sha256": content_sha256,
            },
            "metadata": metadata,
        }

    @staticmethod
    def _validate_artifact_value(value: JsonValue, path: Path) -> JsonValue:
        """Validate the payload checksum and return producer metadata."""
        if not isinstance(value, dict):
            raise CacheIntegrityError(
                "Binary artifact envelope value must be an object"
            )
        descriptor = value.get("artifact")
        if not isinstance(descriptor, dict):
            raise CacheIntegrityError(
                "Binary artifact envelope is missing its artifact descriptor"
            )
        expected_sha256 = descriptor.get("sha256")
        expected_bytes = descriptor.get("bytes")
        if not isinstance(expected_sha256, str) or not isinstance(expected_bytes, int):
            raise CacheIntegrityError(
                "Binary artifact descriptor has invalid checksum or size fields"
            )
        try:
            actual_bytes = path.stat().st_size
            actual_sha256 = _sha256_of_file(path)
        except OSError as exc:
            raise CacheIntegrityError(
                f"Cannot read binary cache artifact at {path}: {exc}"
            ) from exc
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise CacheIntegrityError(
                f"Binary cache artifact checksum or size mismatch at {path}"
            )
        return value.get("metadata")

    def get(self, key: CacheKey) -> CacheResult | None:
        """Load a local override or global default without computing."""
        candidates: tuple[tuple[CacheTier, Path], ...] = (
            ("local", self.local_dir / key.relative_path),
            ("global", self.fetched_global_dir / key.relative_path),
            ("global", self.global_dir / key.relative_path),
        )
        for tier, path in candidates:
            if path.is_file():
                value = self._load(path, key)
                metering.report_cache_hit()
                return CacheResult(value=value, tier=tier, cache_hit=True, path=path)
        return None

    def get_artifact(self, key: CacheKey) -> ArtifactResult | None:
        """Load and validate an opaque local or global payload."""
        candidates: tuple[tuple[CacheTier, Path], ...] = (
            ("local", self.local_dir),
            ("global", self.fetched_global_dir),
            ("global", self.global_dir),
        )
        for tier, root in candidates:
            envelope_path = root / key.relative_path
            payload_path = root / self._payload_relative_path(key)
            if envelope_path.is_file():
                value = self._load(envelope_path, key)
                metadata = self._validate_artifact_value(value, payload_path)
                metering.report_cache_hit()
                return ArtifactResult(
                    metadata=metadata,
                    tier=tier,
                    cache_hit=True,
                    path=payload_path,
                    envelope_path=envelope_path,
                )
        return None

    def load_or_compute(
        self,
        key: CacheKey,
        compute: Callable[[], JsonValue],
        *,
        force_refresh: bool = False,
        write_tier: CacheTier = "local",
    ) -> CacheResult:
        """Load one artifact or compute and atomically persist it."""
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                # `get` reported this hit to the meter already.
                return cached
        value = compute()
        path = self._write(key, value, write_tier)
        return CacheResult(value=value, tier=write_tier, cache_hit=False, path=path)

    async def aload_or_compute(
        self,
        key: CacheKey,
        compute: Callable[[], Awaitable[JsonValue]],
        *,
        force_refresh: bool = False,
        write_tier: CacheTier = "local",
    ) -> CacheResult:
        """Async load-or-compute variant for agent and LLM evaluation calls."""
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                # `get` reported this hit to the meter already.
                return cached
        value = await compute()
        path = self._write(key, value, write_tier)
        return CacheResult(value=value, tier=write_tier, cache_hit=False, path=path)

    def load_or_compute_artifact(
        self,
        key: CacheKey,
        compute: Callable[[Path], JsonValue],
        *,
        media_type: str = "application/octet-stream",
        force_refresh: bool = False,
        write_tier: CacheTier = "local",
    ) -> ArtifactResult:
        """Load an opaque payload or build and atomically persist it.

        ``compute`` receives an empty temporary path and must create one file at
        that path. The payload is installed first and its validated JSON
        envelope is installed last, so a crash cannot expose a partial payload
        as a cache hit.
        """
        if not force_refresh:
            cached = self.get_artifact(key)
            if cached is not None:
                # `get_artifact` reported this hit to the meter already.
                return cached
        root = self.local_dir if write_tier == "local" else self.global_dir
        payload_path = root / self._payload_relative_path(key)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=payload_path.parent,
                prefix=f".{payload_path.name}.",
                delete=False,
            ) as handle:
                temporary_name = handle.name
            temporary_path = Path(temporary_name)
            metadata = compute(temporary_path)
            if not temporary_path.is_file():
                raise ValueError(
                    "artifact producer did not create the requested payload file"
                )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            value = self._artifact_value(temporary_path, metadata, media_type)
            os.replace(temporary_path, payload_path)
            temporary_name = None
            envelope_path = self._write(key, value, write_tier)
            return ArtifactResult(
                metadata=metadata,
                tier=write_tier,
                cache_hit=False,
                path=payload_path,
                envelope_path=envelope_path,
            )
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def fetch_global(
        self,
        key: CacheKey,
        url: str,
        *,
        sha256: str,
        timeout: float = 60.0,
    ) -> Path:
        """Fetch, checksum, validate, and install one oversized global artifact."""
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            content = response.read()
        actual_sha256 = digest_mismatch(content, sha256)
        if actual_sha256 is not None:
            raise CacheIntegrityError(
                f"Downloaded artifact checksum mismatch: {actual_sha256} != {sha256}"
            )
        self._decode(content, key, url)
        destination = self.fetched_global_dir / key.relative_path
        self._atomic_write(destination, content)
        return destination

    def fetch_global_artifact(
        self,
        key: CacheKey,
        envelope_url: str,
        payload_url: str,
        *,
        envelope_sha256: str,
        payload_sha256: str,
        timeout: float = 60.0,
    ) -> Path:
        """Fetch and install a checksum-pinned envelope plus opaque payload.

        The payload streams to a temporary file in blocks and is hashed as it
        arrives, so memory use does not grow with the payload.
        """
        payload_destination = self.fetched_global_dir / self._payload_relative_path(key)
        payload_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=payload_destination.parent,
                prefix=f".{payload_destination.name}.",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                digest = hashlib.sha256()
                with urllib.request.urlopen(  # noqa: S310
                    payload_url, timeout=timeout
                ) as response:
                    for block in iter(lambda: response.read(_HASH_BLOCK_BYTES), b""):
                        digest.update(block)
                        handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if digest.hexdigest() != payload_sha256:
                mismatch = f"{digest.hexdigest()} != {payload_sha256}"
                raise CacheIntegrityError(
                    f"Downloaded payload checksum mismatch: {mismatch}"
                )
            with urllib.request.urlopen(  # noqa: S310
                envelope_url, timeout=timeout
            ) as response:
                envelope_content = response.read()
            actual_envelope_sha256 = digest_mismatch(envelope_content, envelope_sha256)
            if actual_envelope_sha256 is not None:
                mismatch = f"{actual_envelope_sha256} != {envelope_sha256}"
                raise CacheIntegrityError(
                    f"Downloaded envelope checksum mismatch: {mismatch}"
                )
            value = self._decode(envelope_content, key, envelope_url)
            os.replace(temporary_name, payload_destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        envelope_destination = self.fetched_global_dir / key.relative_path
        self._validate_artifact_value(value, payload_destination)
        self._atomic_write(envelope_destination, envelope_content)
        return envelope_destination

    def fetch_manifest(self, manifest_path: Path) -> list[Path]:
        """Fetch and install every checksum-pinned artifact in a manifest."""
        manifest = _parse_manifest(
            Path(manifest_path).read_text(encoding="utf-8"), manifest_path
        )
        installed: list[Path] = []
        for artifact in manifest.artifacts:
            if artifact.payload_url is None and artifact.payload_sha256 is None:
                installed.append(
                    self.fetch_global(
                        artifact.key, artifact.url, sha256=artifact.sha256
                    )
                )
            elif (
                artifact.payload_url is not None and artifact.payload_sha256 is not None
            ):
                installed.append(
                    self.fetch_global_artifact(
                        artifact.key,
                        artifact.url,
                        artifact.payload_url,
                        envelope_sha256=artifact.sha256,
                        payload_sha256=artifact.payload_sha256,
                    )
                )
            else:
                raise ValueError(
                    "manifest artifacts must provide payload_url with payload_sha256"
                )
        return installed

    def export_local(self, destination: Path) -> list[Path]:
        """Copy validated local artifacts into a user-owned instance or fork."""
        destination = Path(destination)
        written: list[Path] = []
        if not self.local_dir.is_dir():
            return written
        for source in sorted(self.local_dir.rglob("*.json")):
            if self.fetched_global_dir in source.parents:
                continue
            envelope = _parse_envelope(source.read_text(encoding="utf-8"), source)
            key = CacheKey(
                ri_id=envelope.ri_id,
                artifact_type=envelope.artifact_type,
                producer_id=envelope.producer_id,
                config=envelope.config,
            )
            self._load(source, key)
            if source.relative_to(self.local_dir) != key.relative_path:
                raise CacheIntegrityError(f"Cache path mismatch at {source}")
            relative = source.relative_to(self.local_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            written.append(target)
            value = self._load(source, key)
            if isinstance(value, dict) and isinstance(value.get("artifact"), dict):
                payload_source = self.local_dir / self._payload_relative_path(key)
                self._validate_artifact_value(value, payload_source)
                payload_target = destination / self._payload_relative_path(key)
                payload_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(payload_source, payload_target)
                written.append(payload_target)
        return written

    def push_local(self, destination: Path) -> list[Path]:
        """Validate and copy local results to another filesystem location."""
        return self.export_local(destination)
