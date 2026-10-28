"""The agent's tools, called as the plain Python functions they are.

Nothing here imports ADK, which is the point: a tool is callable, and testable,
without the framework that declares it.
"""

from __future__ import annotations

from aieng.advanced_rag.agents.tools import as_passage, retrieval_tools
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance


def make_evidence(index: int, doc: str = "C1") -> Evidence:
    return Evidence(
        text=f"passage {index}",
        provenance=Provenance(
            doc_id=doc, chunk_id=f"{doc}::{index:04d}", section="Termination"
        ),
        score=1.0 - index / 10,
        retriever="stub",
    )


class StubRetriever:
    """The required half of the retrieval interface, and nothing else."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: object = None,
    ) -> list[Evidence]:
        self.calls.append((question, k, scope))
        return [make_evidence(index) for index in range(k)]


class FetchingRetriever(StubRetriever):
    """A retriever that also declares the fetch capability."""

    def __init__(self) -> None:
        super().__init__()
        self.fetched: list[list[str]] = []

    def fetch(self, chunk_ids: list[str]) -> list[Evidence]:
        self.fetched.append(list(chunk_ids))
        return [make_evidence(index) for index, _ in enumerate(chunk_ids)]


def test_a_passage_carries_its_text_and_its_citation():
    passage = as_passage(make_evidence(0))
    assert passage["text"] == "passage 0"
    assert passage["citation"] == "C1 > Termination"
    assert passage["chunk_id"] == "C1::0000"
    assert passage["score"] == 1.0


def test_search_is_one_plain_function_with_no_framework_on_it():
    tools = retrieval_tools(StubRetriever())
    assert [tool.__name__ for tool in tools] == ["search_corpus"]
    assert not hasattr(tools[0], "declaration")


def test_the_fetch_tool_appears_only_when_the_retriever_can_fetch():
    assert [tool.__name__ for tool in retrieval_tools(FetchingRetriever())] == [
        "search_corpus",
        "fetch_passages",
    ]


def test_search_returns_passages_the_model_can_read():
    retriever = StubRetriever()
    search = retrieval_tools(retriever, k=2)[0]
    result = search("what notice does termination require")
    assert [passage["text"] for passage in result["passages"]] == [
        "passage 0",
        "passage 1",
    ]


def test_k_and_scope_are_bound_by_the_caller_and_never_offered_to_the_model():
    retriever = StubRetriever()
    search = retrieval_tools(retriever, k=3, scope="C7")[0]
    assert list(search.__annotations__) == ["question", "return"]
    search("who signed")
    assert retriever.calls == [("who signed", 3, "C7")]


def test_fetch_asks_the_retriever_for_the_identifiers_it_was_given():
    retriever = FetchingRetriever()
    fetch = retrieval_tools(retriever)[1]
    result = fetch(["C1::0000", "C1::0001"])
    assert retriever.fetched == [["C1::0000", "C1::0001"]]
    assert len(result["passages"]) == 2


def test_every_tool_annotates_its_parameters_so_a_schema_can_be_built():
    for tool in retrieval_tools(FetchingRetriever()):
        parameters = [name for name in tool.__annotations__ if name != "return"]
        assert parameters, tool.__name__
        assert tool.__doc__
