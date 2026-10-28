"""Unit tests for the diagram-claim helpers that need no model, dataset, or OCR."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
from PIL import Image
from vdocs import diagrams


def _qrels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [302, 302, 303, 304, 5],
            "corpus_id": [10, 11, 12, 13, 14],
            "score": [1, 2, 2, 2, 2],
            "content_type": [["Image", "Text"], ["Infographic"], ["Text"], ["Text", "Table"], ["Image"]],
        }
    )


def test_visual_pairs_keep_english_picture_regions_only() -> None:
    assert diagrams.visual_pairs(_qrels(), [302, 303, 304]) == [(302, 10, 1), (302, 11, 2)]


def test_text_pairs_need_every_region_to_be_text() -> None:
    assert diagrams.text_pairs(_qrels(), [302, 303, 304]) == [(303, 12, 2)]


def test_best_page_prefers_score_then_lower_id() -> None:
    assert diagrams.best_page_per_query([(1, 9, 1), (1, 7, 2), (1, 5, 2), (2, 3, 1)]) == {1: 5, 2: 3}


def test_ocr_lines_are_shares_of_the_page_in_reading_order() -> None:
    result = types.SimpleNamespace(
        boxes=np.array(
            [
                [[100, 200], [200, 200], [200, 220], [100, 220]],
                [[10, 10], [50, 10], [50, 30], [10, 30]],
                [[5, 201], [60, 201], [60, 221], [5, 221]],
            ]
        ),
        txts=("right", "title", "left"),
    )
    lines = diagrams.ocr_lines(Image.new("RGB", (400, 400)), lambda _: result)
    assert [line["text"] for line in lines] == ["title", "left", "right"]
    assert lines[0]["box"] == [0.025, 0.025, 0.125, 0.075]
    assert diagrams.ocr_text(lines) == "title\nleft\nright"
    assert diagrams.ocr_with_boxes(lines).splitlines()[1] == "[2 2 12 8] title"


def test_ocr_lines_handle_a_page_with_no_text() -> None:
    empty = types.SimpleNamespace(boxes=None, txts=None)
    assert diagrams.ocr_lines(Image.new("RGB", (10, 10)), lambda _: empty) == []


def test_text_only_image_keeps_boxes_and_whitens_the_rest() -> None:
    page = Image.new("RGB", (100, 100), "black")
    out = diagrams.text_only_image(page, [{"box": [0.1, 0.1, 0.3, 0.2]}], pad=0.0)
    assert out.getpixel((20, 15)) == (0, 0, 0)
    assert out.getpixel((60, 60)) == (255, 255, 255)


def test_facts_and_coverage() -> None:
    facts = diagrams.parse_facts(
        {"facts": [{"fact": "A leads to B", "kind": "Relational"}, {"fact": "C is 3", "kind": "plain"}, {"fact": ""}]}
    )
    assert facts == [{"fact": "A leads to B", "kind": "relational"}, {"fact": "C is 3", "kind": "plain"}]
    assert diagrams.numbered(facts) == "1. A leads to B\n2. C is 3"
    assert diagrams.stated_indices({"stated": [1, "2", 9, "x"]}, 2) == {0, 1}
    assert diagrams.stated_indices({}, 2) == set()
    by_pair = {"q1": facts, "q2": [{"fact": "D", "kind": "plain"}]}
    stated = {"q1": {1}, "q2": {0}}
    assert diagrams.coverage(by_pair, stated) == {"pairs": 2, "facts": 3, "recall": 0.6667, "all_stated": 0.5}
    assert diagrams.coverage(by_pair, stated, "relational") == {
        "pairs": 1,
        "facts": 1,
        "recall": 0.0,
        "all_stated": 0.0,
    }


def test_parse_flag() -> None:
    assert diagrams.parse_flag('```json\n{"correct": true}\n```', "correct") is True
    assert diagrams.parse_flag('{"correct": "yes"}', "correct") is None
    assert diagrams.parse_flag("no json", "correct") is None
