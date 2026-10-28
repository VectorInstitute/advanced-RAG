"""Shared fixtures: a small contract-like corpus and a scripted stand-in for each model.

The stub reads which prompt it was sent from the system prompt, so one stub can
play the generator, the checker, and the judge, and each test overrides only the
behaviour it is about.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import pytest

from aieng.advanced_rag.evaluation.synthetic import filters, generate, nulls, rewrite
from aieng.advanced_rag.evaluation.synthetic import statements as statements_module
from aieng.advanced_rag.evaluation.synthetic.llm import Model
from aieng.advanced_rag.evaluation.synthetic.sampling import Document
from aieng.advanced_rag.retrieval.embedders import HashEmbedder


PARTIES = ("Acme", "Borealis", "Cobalt", "Dunmore", "Elmwood", "Fernhill", "Granite")
THINGS = ("pricing", "source code", "customer lists", "roadmaps", "test data")


def contract(index: int) -> str:
    """Return a contract-like document with facts unique to ``index``."""
    party = PARTIES[index]
    lines = [f"MUTUAL NON-DISCLOSURE AGREEMENT NUMBER {index}"]
    for clause, thing in enumerate(THINGS):
        years = index + clause + 2
        lines.append(
            f"Clause {clause}. {party} shall keep the {thing} of the other party "
            f"confidential for {years} years after the agreement ends, and shall "
            f"return every copy of the {thing} within {years * 10} days of a "
            f"written request from the disclosing party."
        )
        lines.append(
            f"The obligations on {thing} in clause {clause} do not apply to "
            f"material that {party} already held before signing, or that "
            f"becomes public through no fault of {party}."
        )
    return "\n".join(lines)


@pytest.fixture
def corpus() -> list[Document]:
    return [Document(f"nda-{i}", contract(i)) for i in range(len(PARTIES))]


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


def first_block(prompt: str) -> str:
    """Return the text between the first ``<<<`` and ``>>>`` in a prompt."""
    match = re.search(r"<<<\n(.*?)\n>>>", prompt, flags=re.S)
    return match.group(1) if match else ""


def field(prompt: str, label: str) -> str:
    match = re.search(rf"^{label}: (.*)$", prompt, flags=re.M)
    return match.group(1).strip() if match else ""


class Script:
    """Default replies for every prompt the build sends, each overridable.

    Questions are written as ``What does the agreement say: <statement>?`` so
    the stub checker can answer from the statement, and the judge compares the
    two strings exactly.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.handlers: dict[str, Callable[[str], dict]] = {
            statements_module.STATEMENT_SYSTEM: self.statements,
            generate.QUESTION_SYSTEM: self.question,
            filters.ANSWER_SYSTEM: self.answer,
            filters.JUDGE_SYSTEM: self.judge,
            filters.UNIQUE_SYSTEM: lambda prompt: {"answering": []},
            filters.NULL_SYSTEM: self.null,
            nulls.NEAR_MISS_SYSTEM: self.near_miss,
            rewrite.PARAPHRASE_SYSTEM: self.paraphrase,
        }

    def statements(self, prompt: str) -> dict:
        passage = first_block(prompt)
        sentences = [s for s in passage.split("\n") if s.startswith("Clause")]
        return {
            "statements": sentences[:1] + ["This sentence is in no passage at all."]
        }

    def question(self, prompt: str) -> dict:
        statement = first_block(prompt)
        return {
            "question": f"What does the agreement say: {statement}?",
            "answer": statement,
        }

    def answer(self, prompt: str) -> dict:
        excerpt = re.search(r"\[1\] (.*)", prompt)
        return {"answerable": True, "answer": excerpt.group(1) if excerpt else ""}

    def judge(self, prompt: str) -> dict:
        reference = field(prompt, "Reference answer")
        candidate = field(prompt, "Candidate answer")
        return {"verdict": "match" if reference == candidate else "mismatch"}

    def null(self, prompt: str) -> dict:
        return {"answered": "no", "passage": None}

    def near_miss(self, prompt: str) -> dict:
        source = field(prompt, "Question")
        return {
            "question": f"Who audits the vault in: {source[-60:]}",
            "changed": "topic",
        }

    def paraphrase(self, prompt: str) -> dict:
        source = field(prompt, "Question")
        digest = hashlib.sha256(source.encode()).hexdigest()[:6]
        return {"question": f"How is item {digest} handled?"}

    def __call__(self, system: str, prompt: str) -> str:
        self.calls.append(system)
        handler = self.handlers.get(system)
        if handler is None:
            raise AssertionError(f"unexpected prompt: {system[:60]}")
        return json.dumps(handler(prompt))


@pytest.fixture
def script() -> Script:
    return Script()


def model(name: str, complete: Callable[[str, str], str]) -> Model:
    return Model(name=name, complete=complete)
