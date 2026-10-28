"""Where the Vector proxy key and endpoint come from, in one place.

On a Coder workspace the platform's ``onboard`` command exports
``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` into the shell, and nothing is
written to disk. On a laptop the key usually lives in the repository's ``.env``
as ``VECTOR_PROXY_API_KEY``, the name this project used before the platform
injected keys. Both have to work, so every reader in the package and the
reference implementations asks this module instead of reading the environment
itself.

The order is the environment first, then ``.env``. An exported key always wins,
so a stale ``.env`` left in a workspace cannot shadow the key ``onboard``
loaded. Within each source ``OPENAI_API_KEY`` comes before
``VECTOR_PROXY_API_KEY``, but an exported ``OPENAI_API_KEY`` counts only when
the exported ``OPENAI_BASE_URL`` is on the proxy's host. Without that, it may be
a credential for OpenAI's own API, which must never be sent to the proxy.

The module imports nothing beyond the standard library and never prints a key.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


#: The bootcamp's OpenAI-compatible endpoint, used when nothing sets another.
PROXY_BASE_URL = "https://proxy.vectorinstitute.ai/v1"

#: The names a proxy key is read under, in order. ``OPENAI_API_KEY`` is what the
#: platform injects; ``VECTOR_PROXY_API_KEY`` is the alias the repository's
#: ``.env`` has always used.
KEY_NAMES: tuple[str, ...] = ("OPENAI_API_KEY", "VECTOR_PROXY_API_KEY")

#: The name the endpoint is read under. The platform injects it with the key.
BASE_URL_NAME = "OPENAI_BASE_URL"


class MissingProxyKeyError(RuntimeError):
    """No proxy key in the environment or in a ``.env`` file."""


def is_placeholder(value: str) -> bool:
    """Return whether ``value`` is empty or still a template placeholder.

    ``.env.example`` and the setup guides write ``<your_key>``, and other Vector
    templates write ``your_...`` or end a sample with ``...``. A key like that
    reaches the proxy and comes back 401, which is a worse error than saying
    the key is missing.

    Parameters
    ----------
    value : str
        The raw value.

    Returns
    -------
    bool
        True for an empty, angle-bracketed, ``your_``-prefixed, or
        ``...``-terminated value.
    """
    stripped = value.strip()
    return (
        not stripped
        or (stripped.startswith("<") and stripped.endswith(">"))
        or stripped.lower().startswith("your_")
        or stripped.endswith("...")
    )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a mapping, without touching the environment.

    The repository's ``.env`` has been written as ``NAME = value``, with spaces
    around ``=``, which shell sourcing rejects, so the file is parsed here.
    Comments, blank lines, an ``export`` prefix, and one layer of matching
    quotes are handled. Nothing is expanded.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    dict of str to str
        Every assignment in the file. A later line wins over an earlier one.
    """
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Return the nearest ``.env`` at or above ``start``, stopping at the repo root.

    The walk stops at the first directory that holds ``.git``, a directory in a
    clone and a file in a worktree, so a worktree inside another checkout does
    not pick up that checkout's ``.env``.

    Parameters
    ----------
    start : Path or None, optional
        Where to begin. Defaults to the working directory, which is the
        repository root for the scripts and the notebook's folder for a
        notebook.

    Returns
    -------
    Path or None
        The file, or None when no ``.env`` sits between ``start`` and the
        repository root.
    """
    here = (start or Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            return None
    return None


def points_at_proxy(url: str) -> bool:
    """Return whether ``url`` is on the Vector proxy's host.

    Parameters
    ----------
    url : str
        An endpoint, such as the value of ``OPENAI_BASE_URL``.

    Returns
    -------
    bool
        True when the host matches :data:`PROXY_BASE_URL`'s host.
    """
    return urlparse(url.strip()).hostname == urlparse(PROXY_BASE_URL).hostname


def _process_key_names(names: Sequence[str]) -> list[str]:
    # An exported OPENAI_API_KEY with no proxy endpoint beside it is most likely
    # a credential for OpenAI's own API, and must not be sent to the Vector
    # proxy. The platform always exports the two together.
    trusted = points_at_proxy(os.environ.get(BASE_URL_NAME, ""))
    return [name for name in names if name != "OPENAI_API_KEY" or trusted]


def _lookup(
    names: Sequence[str],
    env_path: Path | None,
    start: Path | None,
    process_names: Sequence[str] | None = None,
) -> str | None:
    for name in names if process_names is None else process_names:
        value = os.environ.get(name, "")
        if not is_placeholder(value):
            return value.strip()
    path = env_path if env_path is not None else find_env_file(start)
    if path is None or not path.is_file():
        return None
    values = read_env_file(path)
    for name in names:
        value = values.get(name, "")
        if not is_placeholder(value):
            return value.strip()
    return None


def resolve_api_key(
    env_path: Path | None = None,
    *,
    start: Path | None = None,
    names: Sequence[str] = KEY_NAMES,
) -> str:
    """Return the proxy key from the environment, falling back to ``.env``.

    Parameters
    ----------
    env_path : Path or None, optional
        A ``.env`` file to read when the environment holds no key. ``None``
        looks for one with :func:`find_env_file`.
    start : Path or None, optional
        Where :func:`find_env_file` begins when ``env_path`` is None.
    names : sequence of str, optional
        The variable names to try, in order. Defaults to :data:`KEY_NAMES`.
        A caller with its own override, such as ``LLM_API_KEY``, puts it first.

    Returns
    -------
    str
        The key, stripped.

    Raises
    ------
    MissingProxyKeyError
        When no name holds a non-placeholder value in either source.
    """
    value = _lookup(names, env_path, start, _process_key_names(names))
    if value is None:
        where = str(env_path) if env_path is not None else "a repository .env"
        raise MissingProxyKeyError(
            f"No proxy key. Set {' or '.join(names)} in the environment "
            f"(on a Coder workspace, onboard exports OPENAI_API_KEY), or put "
            f"VECTOR_PROXY_API_KEY in {where}."
        )
    return value


def resolve_base_url(env_path: Path | None = None, *, start: Path | None = None) -> str:
    """Return the proxy endpoint: ``OPENAI_BASE_URL``, then ``.env``, then the default.

    Parameters
    ----------
    env_path : Path or None, optional
        A ``.env`` file to read when the environment does not set the endpoint.
    start : Path or None, optional
        Where :func:`find_env_file` begins when ``env_path`` is None.

    Returns
    -------
    str
        The endpoint, without a trailing slash.
    """
    value = _lookup((BASE_URL_NAME,), env_path, start)
    return (value or PROXY_BASE_URL).rstrip("/")


__all__ = [
    "BASE_URL_NAME",
    "KEY_NAMES",
    "PROXY_BASE_URL",
    "MissingProxyKeyError",
    "find_env_file",
    "is_placeholder",
    "points_at_proxy",
    "read_env_file",
    "resolve_api_key",
    "resolve_base_url",
]
