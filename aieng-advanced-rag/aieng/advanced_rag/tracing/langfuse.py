"""Langfuse tracing bootstrap for notebook and script runs.

Call :func:`init_langfuse_tracing` once at process startup. The bootstrap is
safe to call repeatedly and does nothing when credentials or an optional
integration are unavailable. Use :func:`langfuse_trace` to group one run under
an explicit root observation and flush it before the cell returns.

The only dependency this module asks for is ``langfuse``, in the ``tracing``
extra. It also instruments Google ADK, LangChain and LangGraph, LlamaIndex, and
Pydantic-AI when their OpenInference instrumentors happen to be importable, and
each of those paths soft-fails to a no-op otherwise. None of them is a
dependency of this package: LangChain and LlamaIndex are not used here, and the
ADK instrumentor has to be checked against the exact ``google-adk==2.10.0`` pin
in the ``agents`` extra before it can be offered. ``litellm`` ships in that same
extra, so the ``langfuse_otel`` callback path works with the extras this
package already declares.

Nothing exports without ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``,
which is what makes either entry point safe to call from a notebook whose
outputs are committed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


def _langfuse_credentials_present() -> bool:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


class _LangfuseTracingBootstrap:
    """Register framework tracing integrations once per process."""

    __slots__ = (
        "_google_adk_instrumented",
        "_langchain_instrumented",
        "_langfuse_client_initialized",
        "_litellm_instrumented",
        "_llama_index_instrumented",
        "_pydantic_ai_instrumented",
    )

    def __init__(self) -> None:
        self._litellm_instrumented = False
        self._google_adk_instrumented = False
        self._langchain_instrumented = False
        self._llama_index_instrumented = False
        self._pydantic_ai_instrumented = False
        self._langfuse_client_initialized = False

    def init(self) -> None:
        """Initialize tracing when credentials and integrations are available."""
        if not _langfuse_credentials_present():
            logger.debug(
                "Skipping Langfuse tracing: set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY.",
            )
            return

        # OpenInference's ADK instrumentor uses the global OTel tracer provider.
        # Creating the Langfuse client first attaches Langfuse's span processor
        # to that provider so the instrumented ADK spans have an exporter.
        self._ensure_langfuse_client()
        self._register_litellm_langfuse_otel()
        self._instrument_google_adk()
        self._instrument_langchain()
        self._instrument_llama_index()
        self._instrument_pydantic_ai()

    def instrumented(self) -> frozenset[str]:
        """Report which integrations this bootstrap has registered.

        Returns
        -------
        frozenset of str
            One name per integration that succeeded, drawn from
            ``"langfuse_client"``, ``"litellm"``, ``"google_adk"``,
            ``"langchain"``, ``"llama_index"``, and ``"pydantic_ai"``. The set
            is empty before :meth:`init` runs and when credentials are absent.
        """
        registered = {
            "langfuse_client": self._langfuse_client_initialized,
            "litellm": self._litellm_instrumented,
            "google_adk": self._google_adk_instrumented,
            "langchain": self._langchain_instrumented,
            "llama_index": self._llama_index_instrumented,
            "pydantic_ai": self._pydantic_ai_instrumented,
        }
        return frozenset(name for name, done in registered.items() if done)

    def _ensure_langfuse_client(self) -> None:
        if self._langfuse_client_initialized:
            return
        try:
            from langfuse import get_client
        except ImportError:
            logger.debug(
                "langfuse not installed; skipping Langfuse client initialization.",
            )
            return

        try:
            get_client()
        except Exception:  # pragma: no cover - SDK/environment-specific failure
            logger.exception("Langfuse get_client() failed; ADK spans may not export.")
            return
        self._langfuse_client_initialized = True

    def _register_litellm_langfuse_otel(self) -> None:
        """Register LiteLLM's Langfuse OpenTelemetry callback."""
        if self._litellm_instrumented:
            return
        try:
            import litellm
        except ImportError:
            logger.debug("litellm not installed; skipping LiteLLM Langfuse callback.")
            return

        existing = list(getattr(litellm, "callbacks", None) or [])
        if "langfuse_otel" not in existing:
            litellm.callbacks = [*existing, "langfuse_otel"]
        self._litellm_instrumented = True

    def _instrument_google_adk(self) -> None:
        """Instrument Google ADK against the global OpenTelemetry provider."""
        if self._google_adk_instrumented:
            return
        try:
            from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        except ImportError:
            logger.debug(
                "openinference-instrumentation-google-adk not installed; "
                "skipping ADK instrumentation.",
            )
            return

        try:
            GoogleADKInstrumentor().instrument()
        except Exception:  # pragma: no cover - instrumentor-specific failure
            logger.exception("GoogleADKInstrumentor().instrument() failed.")
            return
        self._google_adk_instrumented = True

    def _instrument_langchain(self) -> None:
        """Instrument LangChain and LangGraph against the global provider."""
        if self._langchain_instrumented:
            return
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
        except ImportError:
            logger.debug(
                "openinference-instrumentation-langchain not installed; "
                "skipping LangChain and LangGraph instrumentation.",
            )
            return

        try:
            LangChainInstrumentor().instrument()
        except Exception:  # pragma: no cover - instrumentor-specific failure
            logger.exception("LangChainInstrumentor().instrument() failed.")
            return
        self._langchain_instrumented = True

    def _instrument_llama_index(self) -> None:
        """Instrument LlamaIndex against the global OpenTelemetry provider."""
        if self._llama_index_instrumented:
            return
        try:
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        except ImportError:
            logger.debug(
                "openinference-instrumentation-llama-index not installed; "
                "skipping LlamaIndex instrumentation.",
            )
            return

        try:
            LlamaIndexInstrumentor().instrument()
        except Exception:  # pragma: no cover - instrumentor-specific failure
            logger.exception("LlamaIndexInstrumentor().instrument() failed.")
            return
        self._llama_index_instrumented = True

    def _instrument_pydantic_ai(self) -> None:
        """Enable Pydantic-AI's native spans on the global provider."""
        if self._pydantic_ai_instrumented:
            return
        try:
            from pydantic_ai import Agent
        except ImportError:
            logger.debug(
                "pydantic-ai not installed; skipping Pydantic-AI instrumentation.",
            )
            return

        try:
            Agent.instrument_all()
        except Exception:  # pragma: no cover - framework-specific failure
            logger.exception("Agent.instrument_all() failed.")
            return
        self._pydantic_ai_instrumented = True


_bootstrap = _LangfuseTracingBootstrap()


def init_langfuse_tracing() -> None:
    """Wire LiteLLM, ADK, LangGraph, Pydantic-AI, and LlamaIndex to Langfuse.

    The function does nothing unless ``LANGFUSE_PUBLIC_KEY`` and
    ``LANGFUSE_SECRET_KEY`` are present. When credentials are available it:

    1. Creates the Langfuse client, attaching its span processor to the global
       OpenTelemetry provider.
    2. Appends ``"langfuse_otel"`` to ``litellm.callbacks`` once.
    3. Instruments Google ADK, LangChain and LangGraph, and LlamaIndex once
       using their OpenInference instrumentors when installed.
    4. Enables Pydantic-AI's native OpenTelemetry instrumentation once.

    Set ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL`` for a non-default region. A
    short-lived notebook or script run can call
    ``langfuse.get_client().flush()`` before exiting, which
    :func:`langfuse_trace` already does for the run it wraps.
    """
    _bootstrap.init()


@dataclass
class LangfuseTraceResult:
    """Identifiers populated after a :func:`langfuse_trace` context exits.

    Attributes
    ----------
    trace_id : str or None
        The Langfuse trace identifier, or None when nothing was traced.
    trace_url : str or None
        A link to the trace in the Langfuse UI, or None when nothing was
        traced.
    """

    trace_id: str | None = None
    trace_url: str | None = None


@contextmanager
def langfuse_trace(
    name: str,
    *,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[LangfuseTraceResult]:
    """Group one run in a Langfuse root observation and flush it.

    Parameters
    ----------
    name : str
        The root observation and trace name.
    tags : sequence of str, optional
        Tags to attach to the trace.
    metadata : mapping, optional
        Metadata to attach to the trace.
    user_id : str, optional
        The Langfuse user identifier for the trace.
    session_id : str, optional
        The Langfuse session identifier for the trace.

    Yields
    ------
    LangfuseTraceResult
        Empty inside the block. Its fields are filled in once the block exits,
        and both stay None when credentials or the Langfuse SDK are missing.

    Notes
    -----
    Without credentials or the Langfuse SDK this is a no-op context manager
    that raises nothing, so a notebook keeps its cached and offline execution
    path.
    """
    result = LangfuseTraceResult()
    if not _langfuse_credentials_present():
        yield result
        return

    init_langfuse_tracing()
    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:
        yield result
        return

    client = get_client()
    try:
        with client.start_as_current_observation(name=name, as_type="agent"):
            with propagate_attributes(
                trace_name=name,
                tags=list(tags) if tags else None,
                metadata=dict(metadata) if metadata else None,
                user_id=user_id,
                session_id=session_id,
            ):
                yield result
            result.trace_id = client.get_current_trace_id()
    finally:
        client.flush()
        if result.trace_id:
            result.trace_url = client.get_trace_url(trace_id=result.trace_id)
