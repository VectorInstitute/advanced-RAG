"""Tests for the Langfuse tracing bootstrap."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from aieng.advanced_rag.tracing.langfuse import (
    _langfuse_credentials_present,
    _LangfuseTracingBootstrap,
    init_langfuse_tracing,
    langfuse_trace,
)


def _all_dependencies(*, callbacks: list[str] | None = None) -> dict[str, MagicMock]:
    """Return fake optional dependencies for a complete bootstrap."""
    litellm_module = MagicMock()
    litellm_module.callbacks = list(callbacks or [])

    adk_instrumentor_class = MagicMock(return_value=MagicMock())
    google_adk_module = MagicMock()
    google_adk_module.GoogleADKInstrumentor = adk_instrumentor_class

    langchain_instrumentor_class = MagicMock(return_value=MagicMock())
    langchain_module = MagicMock()
    langchain_module.LangChainInstrumentor = langchain_instrumentor_class

    llama_index_instrumentor_class = MagicMock(return_value=MagicMock())
    llama_index_module = MagicMock()
    llama_index_module.LlamaIndexInstrumentor = llama_index_instrumentor_class

    pydantic_ai_module = MagicMock()

    return {
        "langfuse": MagicMock(),
        "litellm": litellm_module,
        "openinference": MagicMock(),
        "openinference.instrumentation": MagicMock(),
        "openinference.instrumentation.google_adk": google_adk_module,
        "openinference.instrumentation.langchain": langchain_module,
        "openinference.instrumentation.llama_index": llama_index_module,
        "pydantic_ai": pydantic_ai_module,
    }


def test_credentials_must_both_be_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "  ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    assert _langfuse_credentials_present() is False

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY")
    assert _langfuse_credentials_present() is False


def test_init_is_a_no_op_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    dependencies = _all_dependencies()

    with patch.dict(sys.modules, dependencies):
        bootstrap = _LangfuseTracingBootstrap()
        bootstrap.init()

    assert bootstrap.instrumented() == frozenset()


def test_init_registers_each_integration_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    dependencies = _all_dependencies(callbacks=["existing_callback"])

    with patch.dict(sys.modules, dependencies):
        bootstrap = _LangfuseTracingBootstrap()
        bootstrap.init()
        bootstrap.init()

    assert bootstrap.instrumented() == frozenset(
        {
            "langfuse_client",
            "litellm",
            "google_adk",
            "langchain",
            "llama_index",
            "pydantic_ai",
        },
    )
    assert dependencies["langfuse"].get_client.call_count == 1
    assert dependencies["litellm"].callbacks == [
        "existing_callback",
        "langfuse_otel",
    ]
    adk_module = dependencies["openinference.instrumentation.google_adk"]
    instrumentor = adk_module.GoogleADKInstrumentor
    assert instrumentor.call_count == 1
    instrumentor.return_value.instrument.assert_called_once_with()
    langchain_module = dependencies["openinference.instrumentation.langchain"]
    langchain_instrumentor = langchain_module.LangChainInstrumentor
    assert langchain_instrumentor.call_count == 1
    langchain_instrumentor.return_value.instrument.assert_called_once_with()
    llama_index_module = dependencies["openinference.instrumentation.llama_index"]
    llama_index_instrumentor = llama_index_module.LlamaIndexInstrumentor
    assert llama_index_instrumentor.call_count == 1
    llama_index_instrumentor.return_value.instrument.assert_called_once_with()
    dependencies["pydantic_ai"].Agent.instrument_all.assert_called_once_with()


def test_optional_imports_fail_softly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")

    with patch.dict(
        sys.modules,
        {
            "langfuse": None,
            "litellm": None,
            "openinference": None,
            "openinference.instrumentation": None,
            "openinference.instrumentation.google_adk": None,
            "openinference.instrumentation.langchain": None,
            "openinference.instrumentation.llama_index": None,
            "pydantic_ai": None,
        },
    ):
        bootstrap = _LangfuseTracingBootstrap()
        bootstrap.init()

    assert bootstrap.instrumented() == frozenset()


def test_public_function_delegates_to_process_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = MagicMock()
    monkeypatch.setattr(
        "aieng.advanced_rag.tracing.langfuse._bootstrap",
        bootstrap,
    )

    init_langfuse_tracing()

    bootstrap.init.assert_called_once_with()


def test_trace_context_groups_attributes_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    dependencies = _all_dependencies()
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-123"
    client.get_trace_url.return_value = "https://langfuse.test/trace-123"
    dependencies["langfuse"].get_client.return_value = client

    with (
        patch.dict(sys.modules, dependencies),
        langfuse_trace(
            "notebook-run",
            tags=["advanced-rag"],
            metadata={"framework": "test"},
        ) as result,
    ):
        assert result.trace_id is None

    assert result.trace_id == "trace-123"
    assert result.trace_url == "https://langfuse.test/trace-123"
    client.start_as_current_observation.assert_called_once_with(
        name="notebook-run",
        as_type="agent",
    )
    dependencies["langfuse"].propagate_attributes.assert_called_once_with(
        trace_name="notebook-run",
        tags=["advanced-rag"],
        metadata={"framework": "test"},
        user_id=None,
        session_id=None,
    )
    client.flush.assert_called_once_with()


def test_trace_context_is_a_no_op_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A notebook that commits its outputs calls both entry points
    # unconditionally, so neither may raise or export without credentials.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    dependencies = _all_dependencies()

    with patch.dict(sys.modules, dependencies):
        init_langfuse_tracing()
        with langfuse_trace(
            "notebook-run",
            tags=["advanced-rag"],
            metadata={"framework": "test"},
            user_id="participant",
            session_id="session-1",
        ) as result:
            assert result.trace_id is None

    assert result.trace_id is None
    assert result.trace_url is None
    dependencies["langfuse"].get_client.assert_not_called()
    assert dependencies["litellm"].callbacks == []
