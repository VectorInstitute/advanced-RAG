"""IRCoT sequence, grounded query updates, provenance, and bounded stops."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.retrieval.index import DenseIndex
from synthesis_lab.ircot import IRCoTAnswerer, IRCoTTools


def fixture():
    passages = [
        Passage(f"p{i}", f"Title {i}", "Mira wrote Beacon." if i == 0 else "Mira was born in Oslo.") for i in range(20)
    ]
    vectors = np.asarray([[1, 0] if i < 10 else [0, 1] for i in range(20)], dtype=np.float32)

    def encode(queries):
        return [[0, 1] if "Mira" in q else [1, 0] for q in queries]

    return passages, vectors, encode


def test_tools_require_updates_and_use_discovered_entity():
    p, v, encode = fixture()
    tools = IRCoTTools(p, DenseIndex([x.passage_id for x in p], v), encode)
    first = tools.start("Where was Beacon's author born?")
    assert first["passages"][0]["passage_id"] == "p0"
    assert "error" in tools.retrieve_next()
    assert "error" in tools.update_evidence([], ["invented"], "Mira birthplace", "continue")
    assert (
        tools.update_evidence(
            [{"text": "Mira wrote Beacon.", "passage_ids": ["p0"]}], ["p0"], "Mira birthplace", "continue"
        )["status"]
        == "continue"
    )
    second = tools.retrieve_next()
    assert second["passages"][0]["passage_id"] == "p10"
    assert tools.trace[1]["facts"][0]["passage_ids"] == ["p0"]
    assert (
        tools.update_evidence(
            [{"text": "Mira was born in Oslo.", "passage_ids": ["p10"]}], ["p0", "p10"], "", "sufficient"
        )["status"]
        == "sufficient"
    )
    assert tools.retrieve_next() == {"stop": "sufficient"}


def test_repeat_and_round_limits():
    p, v, encode = fixture()
    tools = IRCoTTools(p, DenseIndex([x.passage_id for x in p], v), encode, max_rounds=1)
    tools.start("original")
    assert tools.update_evidence([], [], "more", "continue")["status"] == "retrieval_limit"
    tools = IRCoTTools(p, DenseIndex([x.passage_id for x in p], v), encode)
    tools.start("original")
    tools.update_evidence([], [], "  ORIGINAL ", "continue")
    assert tools.retrieve_next()["stop"] == "repeated_query"
    with pytest.raises(ValueError, match="reused"):
        tools.start("another")


def test_answerer_uses_source_text_and_fresh_sessions():
    p, v, encode = fixture()
    sessions = []
    prompts = []

    def factory(functions):
        update, retrieve = functions
        sessions.append(update.__self__)

        async def run(prompt):
            update([{"text": "Mira wrote Beacon.", "passage_ids": ["p0"]}], ["p0"], "Mira birthplace", "continue")
            retrieve()
            update([{"text": "Mira was born in Oslo.", "passage_ids": ["p10"]}], ["p0", "p10"], "", "sufficient")

        return SimpleNamespace(arun=run)

    def complete(prompt):
        prompts.append(prompt)
        return '{"answer":"Oslo","sources":[1,2],"not_stated":false}'

    answerer = IRCoTAnswerer(p, v, encode, complete, factory)
    for qid in ("q1", "q2"):
        result = answerer.answer(qid, "Where was Beacon's author born?")
        assert result.answer.text == "Oslo"
        assert list(result.answer.chunk_ids) == ["p0", "p10"]
        assert answerer.traces[qid]["generation_ids"][:2] == ["p0", "p10"]
    assert sessions[0] is not sessions[1]
    assert all("Mira was born in Oslo." in text for text in prompts)


def test_provider_failure_propagates():
    p, v, encode = fixture()

    async def fail(prompt):
        raise RuntimeError("provider unavailable")

    answerer = IRCoTAnswerer(p, v, encode, lambda _: "", lambda tools: SimpleNamespace(arun=fail))
    with pytest.raises(RuntimeError, match="provider"):
        asyncio.run(answerer.aanswer("q", "original"))
