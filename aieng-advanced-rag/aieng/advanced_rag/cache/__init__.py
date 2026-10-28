"""Reusable two-tier cache for Advanced RAG reference implementations.

:mod:`~aieng.advanced_rag.cache.paths` resolves the one root this package
caches under; :mod:`~aieng.advanced_rag.cache.artifacts` holds the cache
itself. Read ``artifacts``' module docstring for the tier order.
"""

from aieng.advanced_rag.cache import paths
from aieng.advanced_rag.cache.artifacts import (
    ArtifactResult,
    CacheIntegrityError,
    CacheKey,
    CacheResult,
    CacheTier,
    GlobalArtifact,
    GlobalArtifactManifest,
    JsonValue,
    TieredResultCache,
    digest_mismatch,
)
from aieng.advanced_rag.cache.directories import (
    fetch_directory,
    pack_directory,
    store_directory,
    tree_digest,
)
from aieng.advanced_rag.cache.paths import root


__all__ = [
    "ArtifactResult",
    "CacheIntegrityError",
    "CacheKey",
    "CacheResult",
    "CacheTier",
    "GlobalArtifact",
    "GlobalArtifactManifest",
    "JsonValue",
    "TieredResultCache",
    "digest_mismatch",
    "fetch_directory",
    "pack_directory",
    "paths",
    "root",
    "store_directory",
    "tree_digest",
]
