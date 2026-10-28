"""Load RI #2's reference worked examples in tests, or skip without them."""

from collections.abc import Callable
from typing import Any

import pytest


def reference_value(step: Callable[..., Any]) -> Any:
    """Return a worked example from the installed release, or skip without it."""
    from aieng.advanced_rag.models import REFERENCE_MODELS  # noqa: PLC0415
    from aieng.advanced_rag.runs import ReferenceMissingError  # noqa: PLC0415

    try:
        return step(models=REFERENCE_MODELS, show=False).value
    except ReferenceMissingError:
        pytest.skip("RI #2's reference artifacts are not installed; run scripts/fetch_reference_artifacts.py")
