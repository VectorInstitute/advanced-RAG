"""Small synthetic evidence chain for offline, editable notebook experiments."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.pipeline.spine import SpineResult
from aieng.advanced_rag.retrieval.index import DenseIndex

from .corpora import Corpus, Question
from .graph import GraphAnswerer
from .ircot import IRCoTAnswerer, IRCoTTools


LAB = Path(__file__).resolve().parents[1]
QUESTION = "Where was the author of Beacon born?"


def synthetic_corpus() -> tuple[Corpus, np.ndarray, Callable[[Sequence[str]], list[list[float]]]]:
    """Return an invented two-document chain and an explicit test encoder."""
    passages = [Passage(f"demo::{i:02d}", f"Example {i}", f"Unrelated example {i}.") for i in range(20)]
    passages[0] = Passage("demo::00", "Beacon", "Mira wrote Beacon.")
    passages[10] = Passage("demo::10", "Mira", "Mira was born in Oslo.")
    vectors = np.asarray(
        [[1, 0, 0] if i == 0 else [0, 1, 0] if i == 10 else [0, 0, 1] for i in range(20)], dtype=np.float32
    )

    def encode(queries: Sequence[str]) -> list[list[float]]:
        return [[0, 1, 0] if "Mira" in q else [1, 0, 0] for q in queries]

    question = Question("demo-q", QUESTION, ("Oslo",), ("demo::00", "demo::10"), "2-hop")
    return Corpus("synthetic", tuple(passages), (question,), {"synthetic": True}), vectors, encode


def source_answer(prompt: str) -> str:
    """Simulate generation from source text; this is not a model quality estimate."""
    if "Mira was born in Oslo." in prompt and "Mira wrote Beacon." in prompt:
        return '{"answer":"Oslo","sources":[1,2],"not_stated":false}'
    return '{"answer":"","sources":[],"not_stated":true}'


def scripted_loop(functions: Sequence[Callable[..., Any]]) -> SimpleNamespace:
    """Exercise the real tools using a clearly scripted evidence-dependent policy."""
    update, retrieve = functions

    async def arun(prompt: str) -> None:
        first = update(
            [{"text": "Mira wrote Beacon.", "passage_ids": ["demo::00"]}],
            ["demo::00"],
            "Where was Mira born?",
            "continue",
        )
        if first["status"] != "continue":
            return
        retrieve()
        update(
            [{"text": "Mira was born in Oslo.", "passage_ids": ["demo::10"]}],
            ["demo::00", "demo::10"],
            "",
            "sufficient",
        )

    return SimpleNamespace(arun=arun)


async def ircot_example(*, rounds: int = 4) -> tuple[SpineResult, dict[str, Any]]:
    """Execute the actual answerer using scripted tools and synthetic vectors."""
    corpus, vectors, encode = synthetic_corpus()
    answerer = IRCoTAnswerer(corpus.passages, vectors, encode, source_answer, scripted_loop, max_rounds=rounds)
    result = await answerer.aanswer("demo-q", QUESTION)
    return result, answerer.traces["demo-q"]


async def graph_example(*, include_bridge: bool = True) -> tuple[SpineResult, dict[str, Any]]:
    """Execute the common answerer with an editable synthetic graph ranking."""
    corpus, _, _ = synthetic_corpus()

    async def rank(question: str, k: int) -> list[str]:
        preferred = ["demo::00", "demo::10"] if include_bridge else ["demo::00"]
        return (
            preferred
            + [
                p.passage_id
                for p in corpus.passages
                if p.passage_id not in preferred and (include_bridge or p.passage_id != "demo::10")
            ]
        )[:k]

    answerer = GraphAnswerer(corpus.passages, rank, source_answer)
    result = await answerer.aanswer("demo-q", QUESTION)
    return result, answerer.traces["demo-q"]


def starter_tools(*, rounds: int = 4, exposed: int = 5) -> IRCoTTools:
    """Return configurable framework-neutral tools for an RI #2 starter agent."""
    corpus, vectors, encode = synthetic_corpus()
    return IRCoTTools(
        corpus.passages,
        DenseIndex([p.passage_id for p in corpus.passages], vectors),
        encode,
        max_rounds=rounds,
        exposed=exposed,
    )


def require_compute() -> None:
    """Keep notebook execution off a Slurm login node while allowing local use."""
    if shutil.which("srun") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Request Slurm compute and start the notebook kernel inside the allocation.")


def read_result(relative_path: str) -> dict[str, Any]:
    """Read a saved participant result without fetching corpora or calling a model."""
    value: dict[str, Any] = json.loads((LAB / "outputs" / relative_path).read_text())
    return value


def show_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Render comparison tables with the notebook environment's display support."""
    from IPython.display import Markdown, display  # noqa: PLC0415 - notebook-only dependency

    def cell(value: Any) -> str:
        text = f"{value:.4f}" if isinstance(value, float) else str(value)
        return text.replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    display(Markdown("\n".join(lines)))  # type: ignore[no-untyped-call]
