"""Langfuse tracing for the bootcamp's notebook and script runs.

This subpackage is not re-exported from :mod:`aieng.advanced_rag`, because it
needs the ``tracing`` extra to send anything. Import it by path:

    from aieng.advanced_rag.tracing import init_langfuse_tracing, langfuse_trace

    init_langfuse_tracing()
    with langfuse_trace("ri1-threshold-sweep", tags=["advanced-rag"]) as trace:
        answers = sweep.run()
    print(trace.trace_url)

``project_traces`` turns fetched traces into the inputs ``score_retrieval`` and
``score_null_detection`` take, and ``DROPPED`` lists what that discards.
``record_scores`` attaches named numeric scores to a trace.

Both entry points do nothing without ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` in the environment, so a cell that calls them stays
runnable for a participant who has no Langfuse project.
"""

from .langfuse import LangfuseTraceResult, init_langfuse_tracing, langfuse_trace
from .projection import DROPPED, Projection, project_traces
from .scores import record_scores


__all__ = [
    "DROPPED",
    "LangfuseTraceResult",
    "Projection",
    "init_langfuse_tracing",
    "langfuse_trace",
    "project_traces",
    "record_scores",
]
