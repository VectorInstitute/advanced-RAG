"""MultiHop-RAG null control, and the guard that keeps article text out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip(
    "aieng.advanced_rag.retrieval.corpus",
    reason="corpus types have not landed in aieng.advanced_rag.retrieval yet",
)

from aieng.advanced_rag.corpora import multihop_rag  # noqa: E402


def test_only_null_queries_are_loaded(multihop_file: Path) -> None:
    queries = multihop_rag.load_null_queries(multihop_file)
    assert len(queries) == 2
    assert all(query.category == "null_query" for query in queries)


def test_every_loaded_query_is_unanswerable(multihop_file: Path) -> None:
    queries = multihop_rag.load_null_queries(multihop_file)
    assert all(not query.is_answerable for query in queries)
    assert all(query.answers == () for query in queries)


def test_no_article_text_reaches_the_slice(multihop_file: Path) -> None:
    # ODC-BY does not cover the publishers' copyright in the article bodies, so
    # a quoted fact leaking into a derived slice is a licence problem, not a
    # cosmetic one.
    queries = multihop_rag.load_null_queries(multihop_file)
    rendered = json.dumps([query.__dict__ for query in queries], default=str)
    assert "must not leak" not in rendered


def test_queries_are_not_scoped_to_a_document(multihop_file: Path) -> None:
    queries = multihop_rag.load_null_queries(multihop_file)
    assert all(query.contract_id == multihop_rag.NO_DOCUMENT for query in queries)


def test_query_ids_are_stable_positions(multihop_file: Path) -> None:
    queries = multihop_rag.load_null_queries(multihop_file)
    assert [query.query_id for query in queries] == [
        "multihop-null-0001",
        "multihop-null-0002",
    ]


def test_a_null_query_carrying_evidence_is_rejected(tmp_path: Path) -> None:
    payload = [
        {
            "query": "Anything?",
            "answer": multihop_rag.NULL_ANSWER,
            "question_type": "null_query",
            "evidence_list": [{"fact": "Article text that should not be here."}],
        }
    ]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="carries evidence"):
        multihop_rag.load_null_queries(path)


def test_no_loader_exists_for_the_article_corpus() -> None:
    # corpus.json holds 609 full article bodies from named publishers. There is
    # deliberately no way to pull it through this package.
    exported = dir(multihop_rag)
    assert not any("corpus" in name.lower() for name in exported)
    assert "corpus.json" not in multihop_rag.MULTIHOP_QUERIES.url


def test_pinned_source_is_a_revision_not_a_branch() -> None:
    assert "/resolve/main/" not in multihop_rag.MULTIHOP_QUERIES.url
    assert len(multihop_rag.MULTIHOP_QUERIES.sha256) == 64
