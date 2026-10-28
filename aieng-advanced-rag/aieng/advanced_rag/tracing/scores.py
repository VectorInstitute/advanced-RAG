"""Attach numeric scores to a Langfuse trace.

Scores are how a run's evaluation becomes visible next to its trace: a trajectory
metric or a judge verdict is one named number on the trace it describes.

Like the rest of the subpackage, this does nothing without ``LANGFUSE_PUBLIC_KEY``
and ``LANGFUSE_SECRET_KEY`` and never raises for a missing SDK, so an evaluation
cell stays runnable for a participant who has no Langfuse project.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from .langfuse import _langfuse_credentials_present


logger = logging.getLogger(__name__)


def record_scores(
    trace_id: str | None,
    scores: Mapping[str, float | bool],
    *,
    comment: str | None = None,
) -> int:
    """Send each score to Langfuse against one trace.

    Parameters
    ----------
    trace_id : str or None
        The trace the scores describe, from ``LangfuseTraceResult.trace_id``.
        None means nothing was traced, so nothing is sent.
    scores : mapping
        Score name to value. A bool is sent as 1.0 or 0.0, because Langfuse
        charts and filters numeric scores most directly.
    comment : str, optional
        Free text stored with every score, such as the judge's reason.

    Returns
    -------
    int
        How many scores were sent. Zero when there is no trace, no credentials,
        or no SDK.
    """
    if not trace_id or not scores or not _langfuse_credentials_present():
        return 0
    try:
        from langfuse import get_client  # noqa: PLC0415
    except ImportError:
        return 0

    client = get_client()
    sent = 0
    for name, value in scores.items():
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment,
        )
        sent += 1
    client.flush()
    logger.debug("sent %d scores to trace %s", sent, trace_id)
    return sent


__all__ = ["record_scores"]
