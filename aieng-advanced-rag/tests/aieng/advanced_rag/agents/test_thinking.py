"""The reasoning switch, pinned to the shape this deployment reads.

The value matters more than it looks. ADK's own thinking configuration never
reaches a LiteLLM request, and ``reasoning_effort`` raises before one is sent, so
this nesting is the only route that works and a change to it is a change in
behaviour rather than in style.
"""

from __future__ import annotations

from aieng.advanced_rag.agents import thinking_extra_body
from aieng.advanced_rag.reasoning import THINKING_OFF


def test_the_switch_is_nested_under_chat_template_kwargs():
    assert thinking_extra_body(False) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert thinking_extra_body(True) == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_off_is_the_same_body_the_client_path_injects():
    assert thinking_extra_body(False) == THINKING_OFF


def test_reasoning_effort_is_not_in_it():
    assert "reasoning_effort" not in thinking_extra_body(False)
