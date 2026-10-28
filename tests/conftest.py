"""Shared helpers for the repository-root onboarding gate.

The platform's ``onboard`` command exports the bootcamp keys into the shell and
then runs ``tests/test_integration.py`` with this venv's interpreter. Locally,
the keys can sit in the repository ``.env`` instead, so that file is read here,
and a value already in the environment always wins over it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from aieng.advanced_rag.proxy import find_env_file, is_placeholder, read_env_file


ROOT = Path(__file__).resolve().parents[1]

_env_file = find_env_file(ROOT)
if _env_file is not None:
    for _name, _value in read_env_file(_env_file).items():
        os.environ.setdefault(_name, _value)


def env(name: str) -> str:
    """Return a stripped environment value, or ``""`` when missing or a placeholder."""
    value = os.environ.get(name, "")
    return "" if is_placeholder(value) else value.strip()


def require_env(*names: str) -> None:
    """Fail the calling test when any of ``names`` is missing or a placeholder.

    A failure, not a skip: ``onboard`` marks a participant as onboarded when
    pytest exits 0, and a skipped test exits 0. A test that may legitimately
    have no credential, such as Langfuse, decides to skip before calling this.
    """
    missing = [name for name in names if not env(name)]
    if missing:
        pytest.fail(
            f"Not configured: {', '.join(missing)}. On a Coder workspace, onboarding "
            "exports the bootcamp keys into your shell; reload them with "
            'eval "$(onboard --bootcamp-name advanced-rag-bootcamp --skip-test)". '
            "On your own machine, export them or put them in the repository .env. "
            "See docs/setup.md."
        )
