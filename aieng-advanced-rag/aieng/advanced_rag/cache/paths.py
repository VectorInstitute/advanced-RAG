"""The one root every cache in this package writes under.

Three conventions competed before this module existed: the corpus loader's
``~/.cache/aieng-advanced-rag/corpora``, the artifact cache's
``cwd/.cache/advanced-rag``, and the ``ADVANCED_RAG_CACHE_DIR`` variable that
only the artifact cache read. A root under the working directory is wrong for a
notebook, which runs from wherever the participant opened it, and wrong for CI,
which throws the working directory away between jobs. Everything now hangs off
:func:`root`, so one variable moves the whole cache and one directory holds it.
"""

from __future__ import annotations

import os
from pathlib import Path


#: Variable that moves the cache root, read on every call rather than at import.
ENV_VAR = "ADVANCED_RAG_CACHE_DIR"
_DEFAULT_ROOT = Path.home() / ".cache" / "aieng-advanced-rag"


def root() -> Path:
    """Return the directory every cache in this package hangs off.

    Returns
    -------
    Path
        ``ADVANCED_RAG_CACHE_DIR`` when it is set and not empty, otherwise
        ``~/.cache/aieng-advanced-rag``. The directory is not created here;
        each cache creates the subtree it writes to.
    """
    configured = os.environ.get(ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_ROOT


def artifacts_dir() -> Path:
    """Return the artifact cache's Tier-2 root.

    Returns
    -------
    Path
        ``root() / "artifacts"``.
    """
    return root() / "artifacts"


def corpora_dir() -> Path:
    """Return the root the pinned corpus downloads are cached under.

    Returns
    -------
    Path
        ``root() / "corpora"``, which is where
        :func:`aieng.advanced_rag.corpora.cache_dir` puts them unless
        ``ADVANCED_RAG_CORPUS_CACHE`` says otherwise.
    """
    return root() / "corpora"


__all__ = ["ENV_VAR", "artifacts_dir", "corpora_dir", "root"]
