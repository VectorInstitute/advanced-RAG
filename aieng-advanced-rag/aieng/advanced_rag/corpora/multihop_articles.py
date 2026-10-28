"""Opt-in MultiHop-RAG articles, separate from RI #1's unchanged null loader.

ODC-BY covers the MultiHop-RAG database, not the publishers' copyright in the
article text, so no article source is enabled by default. The caller passes a
rights record that names its permission and the exact source pin before this
module fetches content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .multihop_rag import MULTIHOP_LICENCE, MULTIHOP_QUERIES
from .musique import Passage
from .sources import PinnedSource, fetch, verify


ANSWERABLE_TYPES = frozenset({"inference_query", "comparison_query", "temporal_query"})


@dataclass(frozen=True)
class ArticleApproval:
    """A caller's rights record and the immutable article file it covers."""

    reference: str
    source: PinnedSource
    redistribution: str


@dataclass(frozen=True)
class ArticleQuestion:
    """Answerable question with article-level relevance judgments."""

    question_id: str
    question: str
    answer: str
    category: str
    supporting_ids: tuple[str, ...]


def read_approval(path: Path) -> ArticleApproval:
    """Require a rights record scoped to article use, not the database licence."""
    row = json.loads(path.read_text())
    if row.get("status") != "approved" or row.get("scope") != "multihop-rag-articles":
        raise ValueError(
            "MultiHop-RAG article ruling lacks status: approved or does not cover articles"
        )
    for key in ("reference", "url", "sha256", "redistribution"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"article ruling requires {key}")
    if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) or not re.search(
        r"/resolve/[a-f0-9]{40}/", row["url"]
    ):
        raise ValueError("article source needs an immutable revision URL and SHA-256")
    return ArticleApproval(
        row["reference"],
        PinnedSource(
            "multihop-rag-approved-articles",
            row["url"],
            row["sha256"],
            MULTIHOP_LICENCE,
        ),
        row["redistribution"],
    )


def parse_articles(
    question_bytes: bytes, article_bytes: bytes
) -> tuple[list[Passage], list[ArticleQuestion]]:
    """Map title/author evidence to full articles; reject ambiguous associations."""
    articles, queries = json.loads(article_bytes), json.loads(question_bytes)
    if not isinstance(articles, list) or not isinstance(queries, list):
        raise ValueError("MultiHop-RAG files must be lists")
    passages = []
    by_title: dict[str, list[tuple[str, str]]] = {}
    for i, article in enumerate(articles):
        title, body = article["title"], article["body"]
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(body, str)
            or not body.strip()
        ):
            raise ValueError("articles need nonempty title/body strings")
        pid = f"multihop-rag::{i:05d}"
        passages.append(Passage(pid, title, body))
        by_title.setdefault(title, []).append((str(article.get("author", "")), pid))
    records = []
    for i, row in enumerate(queries):
        kind = row.get("question_type")
        if kind == "null_query":
            continue
        if kind not in ANSWERABLE_TYPES:
            raise ValueError("unrecognized MultiHop-RAG answerable query type")
        if (
            not isinstance(row.get("query"), str)
            or not row["query"].strip()
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
        ):
            raise ValueError("answerable queries need question and answer text")
        supports = []
        for evidence in row["evidence_list"]:
            candidates = by_title.get(evidence["title"], [])
            if len(candidates) > 1:
                candidates = [c for c in candidates if c[0] == evidence.get("author")]
            if len(candidates) != 1:
                raise ValueError("article evidence is missing or ambiguous")
            supports.append(candidates[0][1])
        if not supports:
            raise ValueError("answerable question has no supporting articles")
        records.append(
            ArticleQuestion(
                f"multihop-rag::{i:04d}",
                row["query"],
                row["answer"],
                kind,
                tuple(dict.fromkeys(supports)),
            )
        )
    return passages, records


def load_articles(
    *, ruling: Path, dest: Path | None = None
) -> tuple[list[Passage], list[ArticleQuestion]]:
    """Fetch only with a valid rights record; verify local and downloaded bytes."""
    approval = read_approval(ruling)
    query_path = fetch(MULTIHOP_QUERIES, dest=dest)
    article_path = fetch(approval.source, dest=dest)
    query_bytes, article_bytes = query_path.read_bytes(), article_path.read_bytes()
    verify(query_bytes, MULTIHOP_QUERIES.sha256, what="MultiHop-RAG questions")
    verify(article_bytes, approval.source.sha256, what="MultiHop-RAG articles")
    return parse_articles(query_bytes, article_bytes)
