"""A bounded starter agent using the shared ADK loop and scoped retrieval tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from aieng.advanced_rag import spend
from aieng.advanced_rag.agents.loop import AgentLoop
from aieng.advanced_rag.agents.tools import ToolFunction, retrieval_tools
from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import choose_models
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever
from google.adk.agents.invocation_context import LlmCallsLimitExceededError

from .answering import balanced_sample
from .common import EMBEDDING, OUTPUT, ROOT, provenance, require_compute, run_path, write_json


AgentFactory = Callable[[Any, str, Any, list[dict[str, Any]]], AgentLoop]


def audited_tools(retriever: Any, scope: str, trace: list[dict[str, Any]]) -> list[ToolFunction]:
    """Keep shared tool signatures and record the source IDs actually returned."""

    def audit(tool: ToolFunction) -> ToolFunction:
        @wraps(tool)
        def call(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = tool(*args, **kwargs)
            trace.append(
                {
                    "tool": tool.__name__,
                    "arguments": kwargs or {"positional": list(args)},
                    "returned_chunk_ids": [p["chunk_id"] for p in result.get("passages", [])],
                }
            )
            return cast("dict[str, Any]", result)

        return call

    return [audit(tool) for tool in retrieval_tools(retriever, k=5, scope=scope)]


def starter_agent(retriever: Any, scope: str, question_meter: Any, trace: list[dict[str, Any]]) -> AgentLoop:
    """Construct the shared pattern; accept an editable factory in notebooks."""
    models = choose_models()
    spend.session()
    return AgentLoop(
        audited_tools(retriever, scope, trace),
        model=models.generation.model,
        base_url=resolve_base_url(start=ROOT),
        api_key=resolve_api_key(start=ROOT),
        instruction="Search the scoped contract before answering. Cite the returned passage citations. If the contract does not state the answer, explicitly say so. Do not use outside knowledge.",
        name="regulatory_qa_starter",
        enable_thinking=False,
        max_model_calls=6,
        charge_to=question_meter,
    )


async def run_agent_examples(
    factory: AgentFactory = starter_agent, *, corpus: Any = None, labels_known: bool = True
) -> dict[str, Any]:
    """Run a fixed sample with contract scope bound outside the agent's control."""
    if corpus is None:
        corpus = contract_nli.build_slice(contract_nli.fetch_contract_nli(), max_documents=2)
    queries = balanced_sample(corpus.queries) if labels_known else list(corpus.queries)
    if not queries or len(queries) > 20:
        raise ValueError("starter examples require between one and twenty questions")
    documents = {chunk.doc_id for chunk in corpus.chunks}
    if any(query.doc_id not in documents for query in queries):
        raise ValueError("each starter question must name an indexed document")
    embedder = SentenceTransformerEmbedder(EMBEDDING)
    retrievers = {}
    for doc_id in {q.doc_id for q in queries}:
        retriever = DenseRetriever(embedder)
        # Scope fetch as well as search: a guessed foreign ID cannot widen the corpus.
        retriever.index([chunk for chunk in corpus.chunks if chunk.doc_id == doc_id])
        retrievers[doc_id] = retriever
    meter = UsageMeter()
    report = {
        **provenance(),
        "pattern": "shared AgentLoop plus retrieval_tools; fixed scope and bounded model calls",
        "max_model_calls": 6,
        "results": [],
        "limitations": [
            "A bounded starter demonstration, not an agent accuracy benchmark.",
            "Own-corpus examples have unknown answerability labels unless independently annotated.",
            "Uses the current shared agent interfaces; no unavailable future notebook template is claimed as a dependency.",
        ],
    }
    for query in queries:
        trace: list[dict[str, Any]] = []
        with meter.question(query.query_id) as question_meter:
            loop = factory(retrievers[query.doc_id], query.doc_id, question_meter, trace)
            if report["results"] and (
                loop.model != report["model"] or loop.max_model_calls != report["max_model_calls"]
            ):
                raise ValueError("the starter factory must keep model and call budget consistent across questions")
            report["model"] = loop.model
            report["model_roles"]["generation"] = loop.model
            report["max_model_calls"] = loop.max_model_calls
            try:
                answer = await loop.arun(query.question)
                row = {
                    "query_id": query.query_id,
                    "doc_id": query.doc_id,
                    "is_null": not query.is_answerable if labels_known else None,
                    "answer": answer.text,
                    "tool_calls": list(answer.tool_calls),
                    "model_calls": answer.model_calls,
                    "stop_reason": "completed",
                    "empty_final_answer": answer.abstained,
                }
            except LlmCallsLimitExceededError:
                row = {
                    "query_id": query.query_id,
                    "doc_id": query.doc_id,
                    "is_null": not query.is_answerable if labels_known else None,
                    "answer": None,
                    "stop_reason": "model_call_limit",
                    "model_calls": question_meter.totals().calls,
                }
        report["results"].append(row)
        row["tool_trace"] = trace
        row["scope_enforcement"] = "physically isolated per-contract index; both search and fetch are restricted"
        report["usage"] = meter.to_dict()
        write_json(run_path("agent-progress.json"), report)
    write_json(OUTPUT / "agent.json", report)
    return report


def measure_agent(
    *, factory: AgentFactory = starter_agent, corpus: Any = None, labels_known: bool = True
) -> dict[str, Any]:
    """Execute the starter pattern on a compute node."""
    require_compute()
    return asyncio.run(run_agent_examples(factory, corpus=corpus, labels_known=labels_known))
