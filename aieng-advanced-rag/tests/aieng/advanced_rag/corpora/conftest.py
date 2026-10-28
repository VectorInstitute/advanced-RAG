"""Fixtures for the corpus loader tests.

Every test here is offline. The pinned sources are real downloads and a network
fetch in the unit suite would make it flaky and slow, so the fixtures are
hand-built files in each corpus's on-disk format.

Each test module skips itself when ``aieng.advanced_rag.retrieval.corpus`` is
absent. The corpus types are moving into that path in parallel with this
package; until that lands, skipping is the correct outcome rather than a
failure. The guard lives in the test modules and not here, because a conftest
cannot be skipped: raising ``Skipped`` during conftest import is a collection
error, not a skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CUAD_QUESTION = (
    'Highlight the parts (if any) of this contract related to "Governing Law" '
    "that should be reviewed by a lawyer. Details: Which state's law governs "
    "the interpretation of the contract?"
)


@pytest.fixture
def cuad_file(tmp_path: Path) -> Path:
    """Build a two-contract CUAD file with one answerable and one null question."""
    text = "A" * 100 + "This Agreement is governed by Ontario law." + "B" * 100
    start = text.index("This Agreement")
    payload = {
        "version": "test_v1.0",
        "data": [
            {
                "title": "contract-one",
                "paragraphs": [
                    {
                        "context": text,
                        "qas": [
                            {
                                "id": "q-answerable",
                                "question": CUAD_QUESTION,
                                "answers": [
                                    {
                                        "text": "This Agreement is governed by "
                                        "Ontario law.",
                                        "answer_start": start,
                                    }
                                ],
                                "is_impossible": False,
                            },
                            {
                                "id": "q-null",
                                "question": CUAD_QUESTION.replace(
                                    "Governing Law", "Audit Rights"
                                ),
                                "answers": [],
                                "is_impossible": True,
                            },
                        ],
                    }
                ],
            },
            {
                "title": "contract-two",
                "paragraphs": [
                    {
                        "context": "C" * 50,
                        "qas": [
                            {
                                "id": "q-other",
                                "question": CUAD_QUESTION,
                                "answers": [],
                                "is_impossible": True,
                            }
                        ],
                    }
                ],
            },
        ],
    }
    path = tmp_path / "cuad_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def contract_nli_file(tmp_path: Path) -> Path:
    """Build a one-document ContractNLI file with one evidenced and one null label."""
    first = "NON-DISCLOSURE AGREEMENT."
    second = " The Recipient shall return all materials."
    third = " Nothing further is agreed."
    text = first + second + third
    payload = {
        "documents": [
            {
                "id": 1,
                "file_name": "example.pdf",
                "text": text,
                "document_type": "search-pdf",
                "url": "https://example.invalid/example.pdf",
                "spans": [
                    [0, len(first)],
                    [len(first), len(first) + len(second)],
                    [len(first) + len(second), len(text)],
                ],
                "annotation_sets": [
                    {
                        "annotations": {
                            "nda-16": {"choice": "Entailment", "spans": [1]},
                            "nda-11": {"choice": "NotMentioned", "spans": []},
                        }
                    }
                ],
            }
        ],
        "labels": {
            "nda-16": {
                "short_description": "Return of confidential information",
                "hypothesis": "Receiving Party shall destroy or return some "
                "Confidential Information upon the termination of Agreement.",
            },
            "nda-11": {
                "short_description": "No reverse engineering",
                "hypothesis": "Receiving Party shall not reverse engineer any "
                "objects which embody Disclosing Party's Confidential "
                "Information.",
            },
        },
    }
    path = tmp_path / "contract_nli_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def multihop_file(tmp_path: Path) -> Path:
    """Build a MultiHop-RAG query file with two null queries and one answerable."""
    payload = [
        {
            "query": "Which company did both outlets report on?",
            "answer": "Example Corp",
            "question_type": "inference_query",
            "evidence_list": [
                {
                    "title": "An article",
                    "author": "A Reporter",
                    "url": "https://example.invalid/a",
                    "source": "Example News",
                    "category": "technology",
                    "published_at": "2023-10-01T00:00:00+00:00",
                    "fact": "Quoted article text that must not leak.",
                }
            ],
        },
        {
            "query": "What did the report say about an unrelated topic?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
        {
            "query": "Which award was given for a film not covered here?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    path = tmp_path / "MultiHopRAG.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
