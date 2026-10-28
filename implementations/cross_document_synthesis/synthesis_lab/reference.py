"""RI #2's model-produced worked examples, each one step behind ``load_or_run``.

The notebooks show worked examples that hold MuSiQue question text, model
answers, decomposition plans, and agent traces. Those stay out of the
repository and ship in RI #2's reference release instead, the way RI #4 ships
its graph indexes: ``scripts/setup.sh`` installs every
``reference-artifacts.json`` into the artifact cache, and each function here
returns, through :func:`aieng.advanced_rag.runs.load_or_run`:

1. your own earlier result for your model, when the cache holds one;
2. with ``live`` off, the reference built with ``Qwen3_6-35B-A3B`` on Vector's
   cluster, labelled with what a rebuild on your model would cost;
3. with ``live`` on, a run on your model, stored in your local cache.

Each step's key holds every input but the model that changes its output, so a
notebook edit that changes the selection or the prompt finds no reference and
needs ``RUN_EXPENSIVE = True``. ``build_reference.py`` stores the recorded
results under these keys for ``scripts/publish_artifacts.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import tempfile
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypeVar

from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_CORPUS,
    MUSIQUE_LICENCE,
    MUSIQUE_QUESTIONS,
    MUSIQUE_REVISION,
    MusiqueQuestion,
)
from aieng.advanced_rag.models import REFERENCE_MODELS, ModelChoice, ModelSet
from aieng.advanced_rag.runs import Provenance, Step, StepResult, load_or_run

from .decomposition import DECOMPOSITION_TEMPLATE
from .dense_baseline import ANSWER_TEMPLATE
from .ircot import IRCOT_INSTRUCTION
from .runtime import DEFAULT_CACHE, MODEL, MODEL_REVISION


RI_ID = "ri2"
REFERENCE_MODEL = REFERENCE_MODELS.generation.model

#: Questions per hop count in the decomposition worked example.
DECOMPOSITION_PER_HOP = 10
#: The question whose scorer-only hop graph notebook 01 prints.
EXAMPLE_QUESTION_ID = "2hop__13548_13529"
#: Bounds of the starter agent's one-question ADK run.
STARTER_CONFIG: Mapping[str, int] = {"rounds": 4, "exposed": 5, "model_calls": 10}
#: Questions per hop count in the graph worked example.
GRAPH_PER_HOP = 1

#: Fields a runner records about the machine it ran on, dropped from stored results.
HOST_FIELDS = ("execution",)

T = TypeVar("T")


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a prompt or instruction."""
    return hashlib.sha256(text.encode()).hexdigest()


def _encoder() -> list[str]:
    return [MODEL, MODEL_REVISION]


def scrub(value: dict[str, Any]) -> dict[str, Any]:
    """Return a result without host and scheduler fields."""
    return {k: v for k, v in value.items() if k not in HOST_FIELDS}


def run_async(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine to completion from synchronous code, inside a notebook too.

    A notebook already runs an event loop, so the coroutine runs on a worker
    thread with a copy of the caller's context, which keeps the usage meter
    that ``load_or_run`` binds.
    """
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, asyncio.run, factory()).result()


def recorded_usage(rows: list[Mapping[str, Any]], model: str = REFERENCE_MODEL) -> dict[str, dict[str, int]]:
    """Return a build's usage in :class:`Provenance` form from per-question usage rows.

    Calls count cache hits too, so a rebuild estimate covers the whole build.
    """
    return {
        model: {
            "calls": sum(int(r.get("calls", 0)) + int(r.get("cache_hits", 0)) for r in rows),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
            "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in rows),
        }
    }


def reference_provenance(producer: str, usage: dict[str, dict[str, int]], note: str, code: str = "") -> Provenance:
    """Return the ``produced_by`` block of a recorded reference result."""
    return Provenance.for_model(
        REFERENCE_MODEL,
        response_models=(REFERENCE_MODEL,),
        producer=f"{RI_ID}/{producer}",
        usage=usage,
        note=note,
        code_hash=code,
    )


# ---------------------------------------------------------------------------
# Steps


def decomposition_step(per_hop: int = DECOMPOSITION_PER_HOP) -> Step:
    """Return the step for notebook 01's decomposition worked example."""
    return Step(
        RI_ID,
        "decomposition-example",
        "run_decomposition",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "selection": f"first {per_hop} questions per hop count in pinned file order",
            "per_hop": per_hop,
            "planning_prompt_sha256": sha256_text(DECOMPOSITION_TEMPLATE),
            "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
            "encoder": _encoder(),
            "retrieval_depth": 20,
            "evidence_budget": 5,
            "max_subquestions": 4,
        },
        code=(_decomposition_code(),),
    )


def _decomposition_code() -> object:
    from . import decomposition  # noqa: PLC0415

    return decomposition


def agent_lab_step() -> Step:
    """Return the step for notebook 02's three-question agent run."""
    from .agent_lab import LAB_INSTRUCTION  # noqa: PLC0415

    return Step(
        RI_ID,
        "agent-lab",
        "run_agent_lab",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "selection": "first question per hop count in pinned file order",
            "instruction_sha256": sha256_text(LAB_INSTRUCTION),
            "encoder": _encoder(),
            "max_searches": 3,
            "max_model_calls": 10,
        },
        code=(_agent_lab_code(),),
    )


def _agent_lab_code() -> object:
    from . import agent_lab  # noqa: PLC0415

    return agent_lab


def ircot_step() -> Step:
    """Return the step for notebook 03's IRCoT traces, one question per hop count."""
    return Step(
        RI_ID,
        "ircot-examples",
        "run_benchmark/ircot",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "selection": "first question per hop category in pinned file order",
            "instruction_sha256": sha256_text(IRCOT_INSTRUCTION),
            "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
            "encoder": _encoder(),
            "rounds": 4,
            "exposed_per_round": 5,
            "model_calls": 10,
        },
        role="extraction",
        code=(_ircot_code(),),
    )


def _ircot_code() -> object:
    from . import ircot  # noqa: PLC0415

    return ircot


def graph_step(backend: str = "hipporag2", per_hop: int = GRAPH_PER_HOP) -> Step:
    """Return the step for notebook 04's graph queries over RI #4's released index."""
    from .graph_release import release_entry  # noqa: PLC0415

    entry = release_entry(backend)
    return Step(
        RI_ID,
        "graph-examples",
        f"run_graph_comparison/{backend}",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "selection": f"first {per_hop} questions per hop category in pinned file order",
            "per_hop": per_hop,
            "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
            "encoder": _encoder(),
            "retrieval_depth": 20,
            "generation_passages": 5,
        },
        role="extraction",
        upstream={"index": entry["sha256"]},
        code=(_graph_code(),),
    )


def _graph_code() -> object:
    from . import graph  # noqa: PLC0415

    return graph


def starter_step(
    instruction: str = IRCOT_INSTRUCTION,
    config: Mapping[str, int] = STARTER_CONFIG,
    question_id: str | None = None,
) -> Step:
    """Return the step for notebook 99's one-question ADK development run."""
    return Step(
        RI_ID,
        "starter-agent",
        "99_starter_agent",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "question_id": question_id or EXAMPLE_QUESTION_ID,
            "instruction_sha256": sha256_text(instruction),
            "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
            "encoder": _encoder(),
            "config": dict(config),
        },
        role="extraction",
        code=(_ircot_code(),),
    )


def dense_step() -> Step:
    """Return the step for the dense baseline's 1,000 cited answers."""
    return Step(
        RI_ID,
        "dense-predictions",
        "run_dense_baseline",
        {
            "corpus_revision": MUSIQUE_REVISION,
            "questions": 1000,
            "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
            "encoder": _encoder(),
            "retrieval_depth": 20,
            "evidence_budget": 5,
        },
        code=(_dense_code(),),
    )


def _dense_code() -> object:
    from . import dense_baseline  # noqa: PLC0415

    return dense_baseline


# ---------------------------------------------------------------------------
# The worked example's scorer-only annotation


def worked_example(question: MusiqueQuestion, passages: int, questions: list[MusiqueQuestion]) -> dict[str, Any]:
    """Return one question's gold answer, supports, and hop graph, for scoring only."""
    hop_counts: dict[str, int] = {}
    for q in questions:
        hop_counts[str(q.hops)] = hop_counts.get(str(q.hops), 0) + 1
    return {
        "purpose": "Scorer-only gold annotation for the worked example; it never enters planning or answering.",
        "corpus_revision": MUSIQUE_REVISION,
        "questions_sha256": MUSIQUE_QUESTIONS.sha256,
        "corpus_sha256": MUSIQUE_CORPUS.sha256,
        "corpus_passages": passages,
        "corpus_questions": len(questions),
        "hop_counts": hop_counts,
        "question_id": question.question_id,
        "question": question.question,
        "answer": question.answer,
        "answer_aliases": list(question.answer_aliases),
        "supporting_ids": list(question.supporting_ids),
        "hop_graph": json.loads(json.dumps(asdict(question.hop_graph))),
        "attribution": MUSIQUE_LICENCE.attribution,
    }


# ---------------------------------------------------------------------------
# Steps behind load_or_run


def decomposition_example(
    *, models: ModelSet | None = None, live: bool = False, per_hop: int = DECOMPOSITION_PER_HOP, show: bool = True
) -> StepResult:
    """Return the decomposition worked example: ``{"report": ..., "example": ...}``."""
    step = decomposition_step(per_hop)

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique  # noqa: PLC0415
        from run_decomposition import run_live  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            report = run_live(DEFAULT_CACHE, Path(directory) / "live.json", per_hop, models=models)
        passages, questions = load_musique(fetch_musique(dest=DEFAULT_CACHE / "corpus"))
        chosen = next(q for q in questions if q.question_id == EXAMPLE_QUESTION_ID)
        return {"report": scrub(report), "example": worked_example(chosen, len(passages), questions)}

    return load_or_run(step, compute, models=models, live=live, show=show)


def agent_lab(*, models: ModelSet | None = None, live: bool = False, show: bool = True) -> StepResult:
    """Return the three-question ADK agent run of notebook 02."""
    step = agent_lab_step()

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from run_agent_lab import run_live  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            report = run_async(lambda: run_live(output=Path(directory) / "live.json", models=models))
        return scrub(report)

    return load_or_run(step, compute, models=models, live=live, show=show)


def first_per_category(questions: tuple[Any, ...], per_category: int = 1) -> tuple[Any, ...]:
    """Return the first questions of each hop category, in pinned file order."""
    return tuple(
        q
        for category in ("2-hop", "3-hop", "4-hop")
        for q in [q for q in questions if q.category == category][:per_category]
    )


def ircot_examples(*, models: ModelSet | None = None, live: bool = False, show: bool = True) -> StepResult:
    """Return IRCoT traces for the first question of each hop category."""
    step = ircot_step()

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from run_benchmark import run  # noqa: PLC0415

        from .corpora import load_corpus  # noqa: PLC0415

        corpus = load_corpus("musique", DEFAULT_CACHE)
        corpus = replace(corpus, questions=first_per_category(corpus.questions))
        with tempfile.TemporaryDirectory() as directory:
            report = run_async(
                lambda: run(
                    corpus,
                    "ircot",
                    Path(directory) / "ircot.json",
                    DEFAULT_CACHE.parent / "ri2-complete",
                    models=models,
                )
            )
        by_id = {q.question_id: q for q in corpus.questions}
        return {
            "selection": "First question per hop category in pinned file order.",
            "corpus_revision": MUSIQUE_REVISION,
            "examples": [
                {
                    "question_id": r["question_id"],
                    "question": by_id[r["question_id"]].question,
                    "category": by_id[r["question_id"]].category,
                    "answer": r["answer"],
                    "abstained": r["abstained"],
                    "trace": r.get("trace", {}),
                    "usage": r["usage"],
                }
                for r in report["records"]
            ],
        }

    return load_or_run(step, compute, models=models, live=live, show=show)


def graph_examples(
    *, backend: str = "hipporag2", models: ModelSet | None = None, live: bool = False, show: bool = True
) -> StepResult:
    """Return graph-retrieval answers for the first question of each hop category."""
    step = graph_step(backend)

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from run_graph_comparison import run  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            report = run_async(
                lambda: run(None, Path(directory) / "graph.json", per_hop=GRAPH_PER_HOP, backend=backend, models=models)
            )
        return graph_summary(report)

    return load_or_run(step, compute, models=models, live=live, show=show)


def graph_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parts of a graph report the notebook shows."""
    spec = report["spec"]
    texts = {q["id"]: q["text"] for q in spec["questions"]}
    return {
        "backend": spec["method"],
        "implementation": spec["implementation"],
        "artifact_key": spec["artifact_key"],
        "index_coverage": spec["index_coverage"],
        "model_roles": spec["model_roles"],
        "records": [
            {
                "question": texts[r["question_id"]],
                **{
                    k: r.get(k)
                    for k in (
                        "question_id",
                        "answer",
                        "abstained",
                        "retrieved_ids",
                        "generation_ids",
                        "citation_ids",
                        "stop_reason",
                        "failure",
                        "usage",
                    )
                },
            }
            for r in report["records"]
        ],
        "scores": report["scores"],
    }


def starter_development(
    *,
    instruction: str = IRCOT_INSTRUCTION,
    config: Mapping[str, int] = STARTER_CONFIG,
    question_id: str | None = None,
    models: ModelSet | None = None,
    live: bool = False,
    show: bool = True,
) -> StepResult:
    """Return notebook 99's one-question ADK run, with an editable instruction."""
    step = starter_step(instruction, config, question_id)

    def compute(choice: ModelChoice) -> dict[str, Any]:
        return run_async(lambda: _starter_run(instruction, config, question_id, models))

    return load_or_run(step, compute, models=models, live=live, show=show)


async def _starter_run(
    instruction: str, config: Mapping[str, int], question_id: str | None, models: ModelSet | None
) -> dict[str, Any]:
    from aieng.advanced_rag.agents.loop import AgentLoop  # noqa: PLC0415 - live only
    from aieng.advanced_rag.agents.tools import ToolFunction  # noqa: PLC0415
    from aieng.advanced_rag.metering import UsageMeter  # noqa: PLC0415
    from aieng.advanced_rag.models import choose_models  # noqa: PLC0415
    from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415

    from .corpora import load_corpus  # noqa: PLC0415
    from .evaluation import question_scores  # noqa: PLC0415
    from .ircot import IRCoTAnswerer  # noqa: PLC0415
    from .runtime import CachedProxyCompletion  # noqa: PLC0415
    from .vectors import corpus_vectors  # noqa: PLC0415

    models = models or choose_models()
    corpus = load_corpus("musique", DEFAULT_CACHE)
    docs, encode, embedding_manifest = corpus_vectors(corpus, DEFAULT_CACHE.parent / "ri2-complete")
    complete = CachedProxyCompletion(DEFAULT_CACHE, models.generation.model)

    def loop(functions: list[ToolFunction]) -> AgentLoop:
        return AgentLoop(
            functions,
            model=models.extraction.model,
            base_url=resolve_base_url(),
            api_key=resolve_api_key(),
            instruction=instruction,
            max_model_calls=config["model_calls"],
            max_tokens=768,
            enable_thinking=False,
        )

    agent = IRCoTAnswerer(
        corpus.passages, docs, encode, complete, loop, max_rounds=config["rounds"], exposed=config["exposed"]
    )
    wanted = question_id or EXAMPLE_QUESTION_ID
    question = next(q for q in corpus.questions if q.question_id == wanted)
    meter = UsageMeter()
    with meter.question(question.question_id):
        answer = await agent.aanswer(question.question_id, question.question)
    trace = agent.traces[question.question_id]
    row = {
        "question_id": question.question_id,
        "answer": answer.answer.text,
        "abstained": answer.abstained,
        "retrieved_ids": trace["retrieved_ids"],
        "generation_ids": trace["generation_ids"],
        "citation_ids": list(answer.answer.chunk_ids),
        "dropped_citations": list(answer.answer.dropped_citations),
        "stop_reason": trace["stop_reason"],
        "failure": None,
        "usage": meter.per_question()[question.question_id].to_dict(),
    }
    return {
        "purpose": "One-question ADK development check; not a method benchmark",
        "config": dict(config),
        "model": models.generation.model,
        "model_roles": models.as_dict(),
        "corpus": corpus.provenance,
        "embedding": {"model": MODEL, "revision": MODEL_REVISION, "vectors_sha256": embedding_manifest.get("sha256")},
        "instruction_sha256": sha256_text(instruction),
        "answer_prompt_sha256": sha256_text(ANSWER_TEMPLATE),
        "record": row,
        "steps": trace["steps"],
        "scores": question_scores(question, row),
    }


def dense_predictions(*, models: ModelSet | None = None, live: bool = False, show: bool = True) -> StepResult:
    """Return the dense baseline's 1,000 cited answers: ``{"predictions": [...]}``."""
    step = dense_step()

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from run_dense_baseline import run_baseline  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            run_baseline(DEFAULT_CACHE, Path(directory), answer=True, models=models)
            predictions = json.loads((Path(directory) / "predictions.json").read_text())
        return {"predictions": predictions}

    return load_or_run(step, compute, models=models, live=live, show=show)


def reference_dense_predictions() -> list[dict[str, Any]]:
    """Return the reference dense predictions from the installed release.

    Raises
    ------
    aieng.advanced_rag.runs.ReferenceMissingError
        When the reference artifacts are not installed. Install them with
        ``python scripts/fetch_reference_artifacts.py``.
    """
    rows: list[dict[str, Any]] = dense_predictions(models=REFERENCE_MODELS, show=False).value["predictions"]
    return rows
