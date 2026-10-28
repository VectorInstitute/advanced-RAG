"""Install the checksum-pinned Tier-1 artifacts a fetch manifest describes.

Run as ``python -m aieng.advanced_rag.cache <manifest>``. Each artifact is
installed into the fetched-global tier only after its envelope digest, its
payload digest, and its key metadata all agree, so a truncated or substituted
download cannot be read as a standard artifact. Installed paths are printed one
per line.

``global-artifacts.example.json`` beside this module is the manifest shape. Tier
1 ships empty, so the template's artifact list is empty too: a workstream
publishing a store fills it in with the URL and digest of every envelope, plus
``payload_url`` and ``payload_sha256`` for each opaque payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aieng.advanced_rag.cache.artifacts import TieredResultCache


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m aieng.advanced_rag.cache",
        description="Install checksum-pinned Tier-1 artifacts from a manifest.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="JSON manifest holding key, URL, and SHA-256 for each artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Install every artifact in the manifest named on the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status. ``0`` once every artifact is installed.
    """
    arguments = _parse_args(argv)
    for installed in TieredResultCache().fetch_manifest(arguments.manifest):
        print(installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
