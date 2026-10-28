"""Pinned HippoRAG 2 NaturalQuestions control; gold contexts are scorer inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .musique import MUSIQUE_REVISION, Passage
from .sources import Licence, PinnedSource, fetch, verify


NQ_REVISION = MUSIQUE_REVISION
NQ_LICENCE = Licence(
    name="CC BY-SA 3.0",
    url="https://creativecommons.org/licenses/by-sa/3.0/",
    permits_commercial_use=True,
    attribution="Natural Questions, Kwiatkowski et al., TACL 2019. https://ai.google.com/research/NaturalQuestions",
    notes=(
        "The Natural Questions source card records CC BY-SA 3.0. The osunlp/HippoRAG_2 "
        "dataset card, which repackages the questions and passages, states no licence. "
        "Share-alike applies to outputs derived from Natural Questions text."
    ),
)
_BASE = f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{NQ_REVISION}"
NQ_QUESTIONS = PinnedSource(
    "nq-rear-questions",
    f"{_BASE}/nq_rear.json",
    "8acadaeff625790e8e10764d660d9ec46de6ac24eb241ac584c0171deca1d9cd",
    NQ_LICENCE,
)
NQ_CORPUS = PinnedSource(
    "nq-rear-corpus",
    f"{_BASE}/nq_rear_corpus.json",
    "c672a7c1813f1a3e1e7a0b1a29784bf7c5cf2d7a2934cb3ffc3b347843b42864",
    NQ_LICENCE,
)


@dataclass(frozen=True)
class NQPaths:
    """The two immutable source files, also accepted from a local cache."""

    questions: Path
    corpus: Path


@dataclass(frozen=True)
class NQQuestion:
    """One single-hop control question with all accepted reference strings."""

    question_id: str
    question: str
    answers: tuple[str, ...]
    supporting_ids: tuple[str, ...]


def fetch_nq(*, dest: Path | None = None) -> NQPaths:
    """Fetch digest-checked files without registering unresolved packaging terms."""
    return NQPaths(fetch(NQ_QUESTIONS, dest=dest), fetch(NQ_CORPUS, dest=dest))


def parse_nq(
    question_payload: bytes, corpus_payload: bytes
) -> tuple[list[Passage], list[NQQuestion]]:
    """Parse validated structure; separate from I/O for small synthetic fixtures."""
    raw_passages = json.loads(corpus_payload)
    raw_questions = json.loads(question_payload)
    if not isinstance(raw_passages, list) or not isinstance(raw_questions, list):
        raise ValueError("NQ files must contain lists")
    passages = []
    by_text = {}
    for i, row in enumerate(raw_passages):
        title, text = row["title"], row["text"]
        if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("NQ passages require title and nonempty text")
        if (title, text) in by_text:
            raise ValueError("duplicate NQ passage")
        passage = Passage(f"nq::{i:05d}", title, text)
        passages.append(passage)
        by_text[title, text] = passage.passage_id
    questions = []
    for i, row in enumerate(raw_questions):
        question, answers = row["question"], row["reference"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("empty NQ question")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(a, str) or not a.strip() for a in answers)
        ):
            raise ValueError("NQ references must be nonempty strings")
        supports = []
        for context in row["contexts"]:
            if type(context.get("is_supporting")) is not bool:
                raise ValueError("support flags must be boolean")
            if context["is_supporting"]:
                key = (context["title"], context["text"])
                if key not in by_text:
                    raise ValueError("support context missing from NQ corpus")
                supports.append(by_text[key])
        if not supports:
            raise ValueError("NQ supports must be nonempty")
        # The packaging repeats identical supporting contexts in ten questions.
        supports = list(dict.fromkeys(supports))
        questions.append(
            NQQuestion(
                f"nq::{i:04d}", question, tuple(dict.fromkeys(answers)), tuple(supports)
            )
        )
    return passages, questions


def load_nq(paths: NQPaths | None = None) -> tuple[list[Passage], list[NQQuestion]]:
    """Verify both files before parsing, including explicit local paths."""
    paths = paths or fetch_nq()
    questions, corpus = paths.questions.read_bytes(), paths.corpus.read_bytes()
    verify(questions, NQ_QUESTIONS.sha256, what="NQ questions")
    verify(corpus, NQ_CORPUS.sha256, what="NQ corpus")
    passages, records = parse_nq(questions, corpus)
    if len(passages) != 9633 or len(records) != 1000:
        raise ValueError("pinned NQ counts differ")
    return passages, records
