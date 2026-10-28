"""Guard: no file in the package contains a name pinned by its SHA-256 digest.

The test reads every text file under the package directory, its tests
included, and fails on any occurrence of the name in any case, as a substring
of any length of text. The name is not stored in any form. Only its digest and
its length are: the scan lowercases each file, takes every window of that
length inside a run of ASCII letters, and compares the window's digest with the
pinned one. The name is all ASCII letters, so a window that crosses a
non-letter cannot match, and the scan is as strict as a case-insensitive
substring search.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[5]
DIGEST = "69451f744e174b9c18bba231d7d2d6e0785df2fc8cdf7929028e14f891367871"
LENGTH = 8
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".toml",
    ".txt",
    ".cfg",
    ".ipynb",
}
SKIP_PARTS = {"__pycache__", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def text_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and not SKIP_PARTS & set(path.parts)
    )


def contains_digest(text: str, digest: str, length: int) -> bool:
    """Return whether a letter window of ``length`` in ``text`` hashes to ``digest``."""
    for run in re.findall(rf"[a-z]{{{length},}}", text.lower()):
        for start in range(len(run) - length + 1):
            window = run[start : start + length]
            if hashlib.sha256(window.encode()).hexdigest() == digest:
                return True
    return False


def test_the_scan_finds_a_pinned_word_in_any_case_and_position():
    # A stand-in word of the same length checks the scan itself, so a broken
    # scan cannot pass the real check vacuously.
    word = "bootcamp"
    digest = hashlib.sha256(word.encode()).hexdigest()
    assert len(word) == LENGTH
    assert contains_digest("the BootCamp's notes", digest, LENGTH)
    assert contains_digest("prebootcamps", digest, LENGTH)
    assert contains_digest('{"k": "x_BOOTCAMP_y"}', digest, LENGTH)
    assert not contains_digest("boot camp", digest, LENGTH)


def test_the_scan_covers_the_package_and_its_tests():
    files = {path.relative_to(PACKAGE_ROOT).as_posix() for path in text_files()}
    assert "aieng/advanced_rag/evaluation/synthetic/taxonomy.py" in files
    assert "pyproject.toml" in files
    assert len(files) > 100


def test_no_file_contains_the_pinned_name():
    offenders = [
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in text_files()
        if contains_digest(
            path.read_text(encoding="utf-8", errors="ignore"), DIGEST, LENGTH
        )
    ]
    assert not offenders, f"the pinned name appears in {offenders}"
