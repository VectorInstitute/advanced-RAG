"""Pinned control integrity and single-hop support annotation semantics."""

import json

import pytest

from aieng.advanced_rag.corpora.natural_questions import (
    NQ_CORPUS,
    NQ_QUESTIONS,
    NQPaths,
    load_nq,
    parse_nq,
)
from aieng.advanced_rag.corpora.sources import ChecksumMismatchError


def payload():
    """Return a source-shaped fixture with repeated supporting context."""
    context = {"title": "A", "text": "Evidence", "is_supporting": True}
    return [
        {
            "question": "Who?",
            "reference": ["Ada", "A. Ada"],
            "contexts": [context, dict(context)],
        }
    ], [{"title": "A", "text": "Evidence"}]


def test_duplicate_support_contexts_are_one_judgment_and_aliases_are_preserved():
    questions, corpus = payload()
    passages, records = parse_nq(
        json.dumps(questions).encode(), json.dumps(corpus).encode()
    )
    assert records[0].supporting_ids == (passages[0].passage_id,)
    assert records[0].answers == ("Ada", "A. Ada")
    assert records[0].question_id == "nq::0000"


@pytest.mark.parametrize(
    "change",
    (
        "missing_support",
        "nonboolean",
        "empty_answers",
        "empty_question",
        "duplicate_passage",
    ),
)
def test_invalid_annotations_fail(change):
    questions, corpus = payload()
    if change == "missing_support":
        questions[0]["contexts"][0]["text"] = "not in corpus"
    elif change == "nonboolean":
        questions[0]["contexts"][0]["is_supporting"] = "true"
    elif change == "empty_answers":
        questions[0]["reference"] = []
    elif change == "empty_question":
        questions[0]["question"] = ""
    else:
        corpus.append(corpus[0])
    with pytest.raises(ValueError):
        parse_nq(json.dumps(questions).encode(), json.dumps(corpus).encode())


def test_local_paths_cannot_bypass_pins(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("[]")
    with pytest.raises(ChecksumMismatchError):
        load_nq(NQPaths(path, path))
    assert "/resolve/5ec05b38deecc3318bb432c69865959c56058990/" in NQ_QUESTIONS.url
    assert NQ_CORPUS.sha256 != NQ_QUESTIONS.sha256
