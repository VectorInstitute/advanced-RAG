"""Onboarding gate: one test per credential in the provisioned secret.

The platform runs this on a Coder workspace's first start, through::

    onboard --bootcamp-name advanced-rag-bootcamp \
        --test-script tests/test_integration.py --test-marker integration_test

It marks the participant as onboarded only when pytest exits 0. The proxy key,
which every participant's secret holds as ``OPENAI_API_KEY`` and
``OPENAI_BASE_URL``, fails the gate when it is missing or rejected. Langfuse
keys are provisioned per team only when a bootcamp uses them, so their test
skips when both are absent and fails when they are present and rejected.

Run it by hand from the repository root::

    $ pytest tests/test_integration.py -v

The file makes one chat completion of a few tokens and, with Langfuse keys, one
Langfuse authentication call. It downloads nothing, so it finishes well inside
the 120 seconds ``onboard`` allows. The model and corpus checks live in
``implementations/getting_started/00_environment_check.ipynb``, because a cold
download can take longer than that.
"""

from __future__ import annotations

import os

import pytest
from conftest import env, require_env


pytestmark = pytest.mark.integration_test

#: The model the check calls. The default is an on-prem model that every
#: bootcamp key reaches at no cost. ``ADVANCED_RAG_CHECK_MODEL`` names another.
CHECK_MODEL = os.environ.get("ADVANCED_RAG_CHECK_MODEL", "Qwen3_6-35B-A3B")


def test_vector_proxy_llm() -> None:
    """Resolve the proxy key and get one completion through the proxy."""
    try:
        from aieng.advanced_rag.proxy import (
            MissingProxyKeyError,
            resolve_api_key,
            resolve_base_url,
        )
        from aieng.advanced_rag.reasoning import THINKING_OFF
        from openai import OpenAI
    except ImportError as error:
        pytest.fail(f"Missing dependency: {error}. Run: uv sync --all-packages --all-extras --dev")

    try:
        key = resolve_api_key()
    except MissingProxyKeyError as error:
        pytest.fail(str(error))

    client = OpenAI(base_url=resolve_base_url(), api_key=key, max_retries=1, timeout=60.0)
    # Thinking off: with it on, a reasoning model can spend max_tokens before it
    # answers and return content=None, which would read as a broken proxy.
    response = client.chat.completions.create(
        model=CHECK_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
        temperature=0,
        extra_body=THINKING_OFF,
    )
    text = (response.choices[0].message.content or "").strip()
    assert text, f"The proxy returned an empty completion from {CHECK_MODEL}"


def test_langfuse_accepts_the_keys() -> None:
    """Check the team's Langfuse keys, or skip when the bootcamp provisions none."""
    if not env("LANGFUSE_PUBLIC_KEY") and not env("LANGFUSE_SECRET_KEY"):
        pytest.skip("No Langfuse keys provisioned; tracing stays off.")
    require_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

    try:
        from langfuse import get_client
    except ImportError as error:
        pytest.fail(f"Missing the tracing extra: {error}. Run: uv sync --all-packages --all-extras --dev")

    hint = (
        "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, and LANGFUSE_HOST or "
        "LANGFUSE_BASE_URL outside the default region."
    )
    try:
        accepted = get_client().auth_check()
    except Exception as error:  # noqa: BLE001 - the SDK raises its own error types
        pytest.fail(f"Langfuse rejected the keys ({type(error).__name__}). {hint}")
    assert accepted, f"Langfuse rejected the keys. {hint}"
