"""Corpus-neutral benchmark annotations; methods receive only passages and text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_CORPUS,
    MUSIQUE_QUESTIONS,
    MUSIQUE_REVISION,
    Passage,
    fetch_musique,
    load_musique,
)
from aieng.advanced_rag.corpora.natural_questions import (
    NQ_CORPUS,
    NQ_QUESTIONS,
    NQ_REVISION,
    fetch_nq,
    load_nq,
)


@dataclass(frozen=True)
class Question:
    """Evaluation annotation excluded from the Answerer constructor."""

    question_id: str
    question: str
    answers: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    category: str
    original_id: str = ""
    wording: str = "original"


@dataclass(frozen=True)
class Corpus:
    """Verified corpus identity, passages and scorer-only annotations."""

    name: str
    passages: tuple[Passage, ...]
    questions: tuple[Question, ...]
    provenance: dict[str, Any]


def load_corpus(name: str, cache: Path, *, ruling: Path | None = None) -> Corpus:
    """Load the available corpora; MultiHop-RAG articles need a rights record."""
    if name == "musique":
        passages, questions = load_musique(fetch_musique(dest=cache / "corpus"))
        return Corpus(
            name,
            tuple(passages),
            tuple(
                Question(q.question_id, q.question, (q.answer, *q.answer_aliases), q.supporting_ids, f"{q.hops}-hop")
                for q in questions
            ),
            {
                "revision": MUSIQUE_REVISION,
                "questions_sha256": MUSIQUE_QUESTIONS.sha256,
                "corpus_sha256": MUSIQUE_CORPUS.sha256,
            },
        )
    if name == "nq":
        passages, nq = load_nq(fetch_nq(dest=cache / "corpus"))
        return Corpus(
            name,
            tuple(passages),
            tuple(Question(q.question_id, q.question, q.answers, q.supporting_ids, "single-hop") for q in nq),
            {
                "revision": NQ_REVISION,
                "questions_sha256": NQ_QUESTIONS.sha256,
                "corpus_sha256": NQ_CORPUS.sha256,
            },
        )
    if name == "multihop-rag" and ruling is not None:
        from aieng.advanced_rag.corpora.multihop_articles import load_articles, read_approval  # noqa: PLC0415
        from aieng.advanced_rag.corpora.multihop_rag import MULTIHOP_QUERIES  # noqa: PLC0415

        approval = read_approval(ruling)
        passages, records = load_articles(ruling=ruling, dest=cache / "corpus")
        return Corpus(
            name,
            tuple(passages),
            tuple(Question(q.question_id, q.question, (q.answer,), q.supporting_ids, q.category) for q in records),
            {
                "revision": approval.source.url,
                "questions_sha256": MULTIHOP_QUERIES.sha256,
                "corpus_sha256": approval.source.sha256,
                "article_ruling": approval.reference,
                "redistribution": approval.redistribution,
            },
        )
    raise ValueError(
        f"corpus unavailable: {name}; the article text is outside ODC-BY, so pass --article-ruling with a rights record"
    )


def load_rewrites(path: Path, corpus: Corpus) -> Corpus:
    """Load a version and paired question_id/plain/obfuscated rewrite records."""
    payload = json.loads(path.read_bytes())
    if (
        set(payload) != {"version", "pairs"}
        or not isinstance(payload["version"], str)
        or not payload["version"].strip()
    ):
        raise ValueError("rewrites require a version and pairs")
    if corpus.name != "musique" or not isinstance(payload["pairs"], list) or not payload["pairs"]:
        raise ValueError("rewrites require nonempty MuSiQue pairs")
    known = {q.question_id: q for q in corpus.questions}
    seen: set[str] = set()
    records = []
    for pair in payload["pairs"]:
        if not isinstance(pair, dict) or set(pair) != {"question_id", "plain", "obfuscated"}:
            raise ValueError("each rewrite requires exactly question_id, plain, obfuscated")
        qid = pair["question_id"]
        if qid not in known or qid in seen:
            raise ValueError("unknown or duplicate rewrite ID")
        seen.add(qid)
        for wording in ("plain", "obfuscated"):
            text = pair[wording]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("rewrite pair is incomplete")
            records.append(
                replace(
                    known[qid], question_id=f"{qid}::{wording}", question=text.strip(), original_id=qid, wording=wording
                )
            )
    return replace(
        corpus,
        questions=tuple(records),
        provenance=corpus.provenance
        | {
            "rewrite_version": payload["version"],
            "rewrite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
