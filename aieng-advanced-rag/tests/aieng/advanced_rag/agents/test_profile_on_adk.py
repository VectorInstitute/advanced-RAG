"""Generated profile tools on the ADK agent loop, against a loopback endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("google.adk", reason="the agents extra is not installed")

from aieng.advanced_rag.agents.corpus_tools import (  # noqa: E402
    ArmConfig,
    CorpusToolServer,
)
from aieng.advanced_rag.agents.profile import ProfileTools, load_profile  # noqa: E402
from aieng.advanced_rag.agents.session_loop import run_session  # noqa: E402
from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: E402

from .test_plugins import FAKE_KEY, MODEL, StubProxy, message, tool_call  # noqa: E402
from .test_profile import session, write_shop  # noqa: E402


@pytest.fixture
def shop(tmp_path: Path) -> ProfileTools:
    folder = write_shop(tmp_path)
    profile = load_profile(folder / "profile.yaml", root=folder)
    return ProfileTools(profile, profile.corpus())


@pytest.fixture
def responses(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "responses.sqlite")


@pytest.fixture
def endpoint() -> Iterator[Any]:
    stubs: list[StubProxy] = []

    def make(reply: Any) -> StubProxy:
        stubs.append(StubProxy(reply))
        return stubs[-1]

    yield make
    for stub in stubs:
        stub.close()


def _run(stub: StubProxy, the_session: Any, store: ResponseCache) -> Any:
    return run_session(
        "What went wrong?",
        the_session,
        model=MODEL,
        base_url=stub.base_url,
        api_key=FAKE_KEY,
        cache=store,
    )


def _offered(body: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in body.get("tools") or []]


def test_the_loop_offers_and_runs_generated_tools(
    shop: ProfileTools, endpoint: Any, responses: ResponseCache
) -> None:
    def reply(body: dict[str, Any], index: int) -> dict[str, Any]:
        if index == 0:
            call = tool_call(
                "open_section", {"doc_id": "tickets/a.md", "section": "2"}, "a"
            )
            return message(calls=[call])
        return message("Answer: charged twice")

    stub = endpoint(reply)
    run = _run(stub, session(shop, generic=False), responses)
    assert run.answer == "charged twice"
    assert not run.nudged
    assert "open_section" in _offered(stub.requests[0])
    assert run.read_ids == ["tickets/a.md"]


def test_the_read_guard_counts_a_section_tool_as_reading(
    shop: ProfileTools, endpoint: Any, responses: ResponseCache
) -> None:
    arm = ArmConfig("only-sql", ("search",), extra_tools=("sql",))
    server = CorpusToolServer(shop.corpus, search=lambda q, k: [])
    shop.register(server)

    plain = endpoint(lambda body, index: message("Answer: guess"))
    assert not _run(plain, server.session(arm), responses).nudged

    answers = endpoint(
        lambda body, index: message("Answer: a" if index == 0 else "Answer: b")
    )
    reading = server.session(arm.with_(extra_tools=("open_section",)))
    assert _run(answers, reading, responses).nudged
