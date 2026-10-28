"""Fetch the data each reference implementation needs, verified against its pins.

Run it from an environment synced with ``--all-extras``::

    $ python -m aieng.advanced_rag.corpora.download --core
    $ python -m aieng.advanced_rag.corpora.download 3
    $ python -m aieng.advanced_rag.corpora.download --list

``--core`` fetches what every RI starts from and stays small, about 35 MB of
corpus files plus the 130 MB ``BAAI/bge-small-en-v1.5`` embedding model. An RI
number, or the RI's folder name, adds what only that RI reads: the 84
FinanceBench filings for RI #3 (159 MB) and the ServerFault slice for RI #4,
which is built from an 860 MB Internet Archive dump and needs the ``7z``
program.

Every corpus file goes through the loader that pins it, so it is checked
against its SHA-256 whether it was downloaded now or found in the cache, and
the command can run any number of times. Each fetched path is printed with its
digest. One failure does not stop the rest, and the exit status is 1 when any
item failed.

The cache is ``~/.cache/aieng-advanced-rag/corpora`` unless
``ADVANCED_RAG_CORPUS_CACHE`` or ``ADVANCED_RAG_CACHE_DIR`` moves it. The model
goes to the Hugging Face cache, which ``HF_HOME`` moves.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import contract_nli, cuad, financebench, multihop_rag, musique, serverfault
from .sources import fetch


#: The local embedding model every dense arm and the environment check use.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: Files in the model repository that sentence-transformers never reads. They
#: are most of the repository's bytes.
EMBEDDING_IGNORED = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    # sentence-transformers reads model.safetensors when it exists.
    "pytorch_model.bin",
)


def fetch_embedding_model() -> Path:
    """Download the local embedding model into the Hugging Face cache.

    Needs ``huggingface_hub``, which the ``local`` extra brings through
    sentence-transformers.

    Returns
    -------
    Path
        The snapshot directory.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(EMBEDDING_MODEL, ignore_patterns=list(EMBEDDING_IGNORED))
    )


def _musique() -> list[Path]:
    paths = musique.fetch_musique()
    return [paths.questions, paths.corpus]


def _financebench_annotations() -> list[Path]:
    return [
        fetch(financebench.FINANCEBENCH_OPEN_SOURCE),
        fetch(financebench.FINANCEBENCH_DOCUMENT_INFORMATION),
    ]


def _financebench_pdfs() -> list[Path]:
    return sorted(financebench.fetch_financebench().pdfs.values())


def _serverfault_slice() -> list[Path]:
    from .sources import cache_dir

    questions = serverfault.SLICE_10K_QUESTIONS
    serverfault.fetch_slice(questions)
    return [cache_dir() / f"serverfault-slice-{questions}.json"]


@dataclass(frozen=True)
class Item:
    """One thing the command can fetch.

    Parameters
    ----------
    name : str
        The name printed and accepted on the command line.
    description : str
        What it is and roughly how large, for ``--list``.
    run : callable
        Fetches the item and returns the local paths it produced.
    """

    name: str
    description: str
    run: Callable[[], list[Path]]


#: Everything the command knows how to fetch, by name.
ITEMS: dict[str, Item] = {
    item.name: item
    for item in (
        Item(
            "cuad",
            "CUAD test split, contracts with span answers (7 MB)",
            lambda: [cuad.fetch_cuad_test()],
        ),
        Item(
            "contract-nli",
            "ContractNLI test split (2 MB, from a 65 MB archive)",
            lambda: [contract_nli.fetch_contract_nli()],
        ),
        Item(
            "multihop-rag",
            "MultiHop-RAG queries, for the null controls (5 MB)",
            lambda: [multihop_rag.fetch_multihop_queries()],
        ),
        Item(
            "musique",
            "MuSiQue questions and passages, HippoRAG 2 packaging (18 MB)",
            _musique,
        ),
        Item(
            "financebench",
            "FinanceBench questions and document metadata (1 MB)",
            _financebench_annotations,
        ),
        Item(
            "financebench-pdfs",
            "The 84 FinanceBench filings the questions cite (159 MB)",
            _financebench_pdfs,
        ),
        Item(
            "serverfault",
            "ServerFault 10,000-question slice (32 MB, built from an 860 MB dump; needs 7z)",
            _serverfault_slice,
        ),
        Item(
            "bge-small",
            f"{EMBEDDING_MODEL}, the local embedding model (130 MB)",
            lambda: [fetch_embedding_model()],
        ),
    )
}

#: What every reference implementation starts from. Small enough for every
#: workspace start.
CORE: tuple[str, ...] = (
    "cuad",
    "contract-nli",
    "multihop-rag",
    "musique",
    "financebench",
    "bge-small",
)

#: What each reference implementation reads, by number.
RIS: dict[str, tuple[str, ...]] = {
    "1": ("cuad", "contract-nli", "multihop-rag", "bge-small"),
    "2": ("musique", "multihop-rag", "bge-small"),
    "3": ("financebench", "financebench-pdfs", "bge-small"),
    "4": ("musique", "serverfault", "bge-small"),
    # RI #5 runs agent-native retrieval over the corpora of RIs #1 to #4.
    "5": CORE,
}

#: The RI folder names under ``implementations/``, accepted in place of a number.
RI_FOLDERS: dict[str, str] = {
    "regulatory_policy_qa": "1",
    "cross_document_synthesis": "2",
    "table_structured_reasoning": "3",
    "graphrag_technical_support": "4",
    "agent_native_retrieval": "5",
}


def plan(ris: Sequence[str], *, core: bool = False) -> list[str]:
    """Return the item names to fetch, in a stable order and without repeats.

    Parameters
    ----------
    ris : sequence of str
        RI numbers, RI folder names, or item names.
    core : bool, optional
        Include :data:`CORE`.

    Returns
    -------
    list of str
        Item names in :data:`ITEMS` order.

    Raises
    ------
    ValueError
        For a name that is none of those.
    """
    wanted: set[str] = set(CORE) if core else set()
    for raw in ris:
        key = RI_FOLDERS.get(raw.strip().rstrip("/").split("/")[-1], raw.strip())
        key = key.removeprefix("ri").removeprefix("#")
        if key in RIS:
            wanted.update(RIS[key])
        elif key in ITEMS:
            wanted.add(key)
        else:
            raise ValueError(
                f"unknown RI or item {raw!r}: use 1 to 5, an RI folder name, "
                f"or one of {', '.join(ITEMS)}"
            )
    return [name for name in ITEMS if name in wanted]


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> str:
    """Return one printable line for a fetched path."""
    if path.is_dir():
        return f"{path} (directory)"
    return f"{path} sha256={sha256_of(path)}"


def run(names: Sequence[str], *, out: Callable[[str], None] = print) -> int:
    """Fetch each named item, print what landed where, and return an exit status.

    Parameters
    ----------
    names : sequence of str
        Item names from :data:`ITEMS`.
    out : callable, optional
        Where lines go. Defaults to ``print``.

    Returns
    -------
    int
        0 when every item was fetched and verified, 1 otherwise.
    """
    failed: list[str] = []
    for name in names:
        try:
            paths = ITEMS[name].run()
        except Exception as error:  # noqa: BLE001 - one bad item must not stop the rest
            failed.append(name)
            out(f"FAIL {name}: {type(error).__name__}: {error}")
            continue
        if len(paths) > 3:
            out(
                f"ok   {name}: {len(paths)} files under {paths[0].parent}, each verified"
            )
            continue
        for path in paths:
            out(f"ok   {name}: {describe(path)}")
    if failed:
        out(f"{len(failed)} of {len(names)} failed: {', '.join(failed)}")
        return 1
    out(f"all {len(names)} fetched and verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and fetch. Return an exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m aieng.advanced_rag.corpora.download",
        description="Fetch and verify the data a reference implementation needs.",
    )
    parser.add_argument(
        "ri",
        nargs="*",
        help="RI number (1 to 5), RI folder name, or item name",
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="fetch the small set every RI starts from",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what each RI and item covers, and fetch nothing",
    )
    args = parser.parse_args(argv)

    if args.list:
        for item in ITEMS.values():
            print(f"{item.name:18} {item.description}")
        print(f"{'--core':18} {', '.join(CORE)}")
        for number, covered in RIS.items():
            print(f"{'RI ' + number:18} {', '.join(covered)}")
        return 0
    if not args.ri and not args.core:
        parser.error("name an RI, an item, or --core (or --list to see them)")
    try:
        names = plan(args.ri, core=args.core)
    except ValueError as error:
        parser.error(str(error))
    return run(names)


__all__ = [
    "CORE",
    "EMBEDDING_MODEL",
    "ITEMS",
    "RIS",
    "RI_FOLDERS",
    "Item",
    "fetch_embedding_model",
    "main",
    "plan",
    "run",
]


if __name__ == "__main__":
    sys.exit(main())
