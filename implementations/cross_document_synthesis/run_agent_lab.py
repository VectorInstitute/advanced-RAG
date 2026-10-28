"""Run the migrated ADK MuSiQue lab; model and embedding cells are gated by --live."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.corpora.musique import MUSIQUE_LICENCE, MUSIQUE_REVISION, fetch_musique, load_musique
from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1
from aieng.advanced_rag.metering import UsageMeter
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url
from synthesis_lab.agent_lab import LAB_INSTRUCTION, MusiqueTools, parse_lab_answer
from synthesis_lab.runtime import (
    DEFAULT_CACHE,
    MODEL,
    MODEL_REVISION,
    embeddings,
    local_embedder,
    write_json,
)


DEFAULT_OUTPUT = DEFAULT_CACHE.parent / "ri2-experiments/agent-lab.json"


async def run_live(
    cache_dir: Path = DEFAULT_CACHE,
    *,
    output: Path = DEFAULT_CACHE.parent / "ri2-experiments/agent-lab.json",
    models: ModelSet | None = None,
) -> dict[str, Any]:
    """Run one question per hop count, with fresh tools, session, and usage scope."""
    from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: PLC0415 - live only
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError  # noqa: PLC0415

    models = models or choose_models()
    spend.session()
    api_key = resolve_api_key()
    passages, questions = load_musique(fetch_musique(dest=cache_dir / "corpus"))
    embedder = local_embedder(cache_dir)
    docs, _, embedding_record = embeddings(cache_dir, passages, questions, embedder=embedder)
    selected = [next(q for q in questions if q.hops == hops) for hops in (2, 3, 4)]
    meter = UsageMeter()
    report: dict[str, Any] = {
        "corpus_revision": MUSIQUE_REVISION,
        "corpus_passages": len(passages),
        "framework": "ADK",
        "adk_version": importlib.metadata.version("google-adk"),
        "model": models.generation.model,
        "model_roles": models.as_dict(),
        "base_url": resolve_base_url(),
        "embedding_model": MODEL,
        "embedding_revision": MODEL_REVISION,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_searches": 3,
        "max_model_calls": 10,
        "workers": 1,
        "embeddings": embedding_record,
        "attribution": MUSIQUE_LICENCE.attribution,
        "purpose": "Three-question migration smoke test, not a quality benchmark",
        "results": [],
    }
    for q in selected:
        tools = MusiqueTools(passages, docs, embedder.encode_queries)
        answer: dict[str, Any]
        with meter.question(q.question_id) as question_meter:
            loop = AgentLoop(
                tools.functions(),
                model=models.generation.model,
                base_url=resolve_base_url(),
                api_key=api_key,
                instruction=LAB_INSTRUCTION,
                name="musique_agent_lab",
                enable_thinking=False,
                max_tokens=768,
                max_model_calls=10,
                charge_to=question_meter,
            )
            try:
                result = await loop.arun(q.question)
            except LlmCallsLimitExceededError:
                # Preserve spent usage and distinguish budget stops from refusals.
                answer = {"status": "not_stated", "answer": "", "citations": []}
                model_calls = question_meter.totals().calls
                tool_calls = [row["tool"] for row in tools.trace]
                stop_reason = "model_call_limit"
            else:
                answer = parse_lab_answer(result.text, tools)
                model_calls = result.model_calls
                tool_calls = list(result.tool_calls)
                stop_reason = "completed"
        prediction = answer["answer"] if answer["status"] == "answered" else ""
        report["results"].append(
            {
                "question_id": q.question_id,
                "hops": q.hops,
                **answer,
                "trace": tools.trace,
                "searches": tools.searches,
                "exposed_ids": sorted(tools.exposed_ids),
                "read_ids": sorted(tools.read_ids),
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                "stop_reason": stop_reason,
                "usage": question_meter.totals().to_dict(),
                "exact_match": exact_match(prediction, (q.answer, *q.answer_aliases)),
                "token_f1": token_f1(prediction, (q.answer, *q.answer_aliases)),
                "all_supports_read": set(q.supporting_ids) <= tools.read_ids,
            }
        )
        report["usage"] = meter.to_dict()
        # Preserve each completed question if a later proxy call fails.
        write_json(output, report)
        print(
            f"Completed {q.hops}-hop question: {answer['status']} ({model_calls} model calls; {stop_reason})",
            flush=True,
        )
    return report


def main() -> None:
    """Inspect saved outputs by default; explicitly enable fresh model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (DEFAULT_CACHE.parent / "ri2-experiments/agent-lab.json" if args.live else DEFAULT_OUTPUT)
    if args.live:
        models = choose_models(show=True)
        try:
            report = asyncio.run(run_live(args.cache_dir, output=output, models=models))
        finally:
            print(spend.report())
    else:
        report = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "framework": report["framework"],
                "base_url": report["base_url"],
                "embedding_model": report["embedding_model"],
                "results": [
                    {k: row[k] for k in ("question_id", "hops", "status", "searches", "model_calls")}
                    for row in report["results"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
