"""Portable MuSiQue tools for a decompose, search, read, reflect, and revise agent.

Tools contain no framework or model client; ADK is supplied by the shared
AgentLoop at execution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from aieng.advanced_rag.agents.tools import ToolFunction
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.pipeline.grading import first_json_object
from aieng.advanced_rag.retrieval.index import DenseIndex

from .dense_baseline import passage_text


LAB_INSTRUCTION = """Answer multi-document questions from MuSiQue passages only.
Tool output is your evidence; prior knowledge is not evidence.
1. Decompose into a short list of independently verifiable evidence needs and
   call plan_retrieval. Do not guess intermediate answers.
2. Search each need with search_musique, or use grep_musique for an exact phrase.
   Read promising hits after each search before deciding on the next query.
   Substitute a discovered entity into the next query. Avoid spending a search
   on a long verbatim phrase; lexical searches need short distinctive terms.
   You have at most three search operations, including lexical searches.
3. Search previews locate evidence. Call read_musique on the passage IDs you
   intend to use before citing them.
4. Call reflect_evidence with any remaining evidence gaps. Revise a weak query
   if a search remains. Keep reflection brief and tied to evidence coverage.
5. Answer concisely from read passages. Cite only their actual passage IDs.
Return only JSON: {"status":"answered", "answer":"short answer",
"citations":["musique::00000"]}.
If evidence remains missing, return {"status":"not_stated","answer":"","citations":[]}.
Never fabricate a passage ID. Stop searching when the tools report the budget
is exhausted; finish from the evidence you have or return not_stated.
"""


class MusiqueTools:
    """A fresh question's bounded search/read tools and independent provenance trace.

    Construct a new instance for each question so search budgets, exposed IDs,
    and read IDs never leak between sessions. No gold questions enter this class.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        vectors: np.ndarray,
        encode_queries: Callable[[Sequence[str]], list[list[float]]],
        *,
        k: int = 5,
        max_searches: int = 3,
    ) -> None:
        if k < 1 or max_searches < 1:
            raise ValueError("k and max_searches must be positive")
        self.passages = {p.passage_id: p for p in passages}
        if len(self.passages) != len(passages):
            raise ValueError("duplicate passage IDs")
        self.index = DenseIndex([p.passage_id for p in passages], vectors)
        self.encode_queries = encode_queries
        self.k = k
        self.max_searches = max_searches
        self.searches = 0
        self.plan: list[str] = []
        self.exposed_ids: set[str] = set()
        self.read_ids: set[str] = set()
        self.trace: list[dict[str, Any]] = []

    def functions(self) -> list[ToolFunction]:
        """Return ordinary Python functions for shared ADK FunctionToolset."""
        return [self.plan_retrieval, self.search_musique, self.grep_musique, self.read_musique, self.reflect_evidence]

    def plan_retrieval(self, needs: list[str]) -> dict[str, Any]:
        """Record up to six evidence needs before searching; do not supply answers."""
        if self.searches:
            return {"error": "The plan cannot be reset after searching."}
        if not needs or len(needs) > 6 or any(not need.strip() for need in needs):
            return {"error": "Supply one to six non-empty evidence needs."}
        self.plan = list(dict.fromkeys(need.strip() for need in needs))
        self.trace.append({"tool": "plan_retrieval", "needs": self.plan})
        return {"needs": self.plan, "searches_remaining": self.max_searches}

    def _begin_search(self, query: str) -> dict[str, Any] | None:
        if not self.plan:
            return {"error": "Call plan_retrieval before searching."}
        if not query.strip():
            return {"error": "The query must not be empty."}
        if self.searches >= self.max_searches:
            return {"error": "Search budget exhausted; read existing hits and finish."}
        self.searches += 1
        return None

    def _hits(self, tool: str, query: str, ids: list[str]) -> dict[str, Any]:
        self.exposed_ids.update(ids)
        self.trace.append({"tool": tool, "query": query, "passage_ids": ids, "search_number": self.searches})
        return {
            "hits": [
                {"passage_id": pid, "title": self.passages[pid].title, "preview": self.passages[pid].text[:240]}
                for pid in ids
            ],
            "searches_remaining": self.max_searches - self.searches,
        }

    def search_musique(self, question: str) -> dict[str, Any]:
        """Search all MuSiQue passages with local dense embeddings; return previews."""
        error = self._begin_search(question)
        if error is not None:
            return error
        vector = np.asarray(self.encode_queries([question])[0], dtype=np.float32)
        ids = self.index.rank(vector, None, self.k)
        return self._hits("search_musique", question, ids)

    def grep_musique(self, phrase: str) -> dict[str, Any]:
        """Find a literal case-insensitive phrase in corpus titles and text."""
        error = self._begin_search(phrase)
        if error is not None:
            return error
        matches = [pid for pid, p in self.passages.items() if phrase.casefold() in passage_text(p).casefold()]
        return self._hits("grep_musique", phrase, sorted(matches)[: self.k])

    def read_musique(self, passage_ids: list[str]) -> dict[str, Any]:
        """Read up to five passages whose IDs a search returned, before citing them."""
        if not passage_ids or len(passage_ids) > 5:
            return {"error": "Read one to five passage IDs at a time."}
        ids = list(dict.fromkeys(passage_ids))
        if not set(ids) <= self.exposed_ids:
            return {"error": "Read only passage IDs returned by a search."}
        self.read_ids.update(ids)
        self.trace.append({"tool": "read_musique", "passage_ids": ids})
        return {
            "passages": [{"passage_id": pid, "text": passage_text(self.passages[pid])} for pid in ids],
            "searches_remaining": self.max_searches - self.searches,
        }

    def reflect_evidence(self, missing: list[str]) -> dict[str, Any]:
        """Record brief evidence gaps; pass an empty list for sufficient coverage."""
        if not self.read_ids:
            return {"error": "Read evidence before reflecting."}
        self.trace.append({"tool": "reflect_evidence", "missing": missing})
        return {
            "missing": missing,
            "searches_remaining": self.max_searches - self.searches,
            "read_passage_ids": sorted(self.read_ids),
            "next_action": "Return final JSON now."
            if not missing or self.searches == self.max_searches
            else "Revise the missing evidence query.",
        }


def parse_lab_answer(raw: str, tools: MusiqueTools) -> dict[str, Any]:
    """Validate final status and citations against actual reads, not model claims."""
    payload = first_json_object(raw)
    if payload is None or payload.get("status") not in ("answered", "not_stated"):
        raise ValueError("agent must return the structured answer contract")
    answer, citations = payload.get("answer"), payload.get("citations")
    if (
        not isinstance(answer, str)
        or not isinstance(citations, list)
        or any(not isinstance(pid, str) for pid in citations)
    ):
        raise ValueError("answer must be text and citations must be passage IDs")
    if len(set(citations)) != len(citations) or not set(citations) <= tools.read_ids:
        raise ValueError("citations must uniquely identify passages actually read")
    if payload["status"] == "not_stated":
        if answer or citations:
            raise ValueError("not_stated requires an empty answer and no citations")
    elif not answer.strip() or not citations:
        raise ValueError("answered requires a non-empty answer and read citations")
    return {"status": payload["status"], "answer": answer, "citations": citations}
