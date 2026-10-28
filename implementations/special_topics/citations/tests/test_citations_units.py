"""Unit tests for the citation topic's data, control, and report helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aieng.advanced_rag.evaluation.citations import CitedAnswer, parse_inline
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance
from cites import data, report
from cites.pipeline import attribute_posthoc, cite_rank1, score_in_batches, shuffle_citations


TEXT = "Alpha clause. Beta clause says return. Gamma clause."


def split(tmp_path: Path) -> Path:
    spans = [[0, 13], [14, 38], [39, 52]]
    payload = {
        "labels": {
            "nda-1": {"hypothesis": "Receiving Party shall return information.", "short_description": "r"},
            "nda-2": {"hypothesis": "Receiving Party may share information.", "short_description": "s"},
            "nda-3": {"hypothesis": "Agreement lasts forever.", "short_description": "f"},
            "nda-4": {"hypothesis": "Agreement names a court.", "short_description": "c"},
        },
        "documents": [
            {
                "id": 7,
                "text": TEXT,
                "spans": spans,
                "annotation_sets": [
                    {
                        "annotations": {
                            "nda-1": {"choice": "Entailment", "spans": [1]},
                            "nda-2": {"choice": "Contradiction", "spans": [0, 2]},
                            "nda-3": {"choice": "NotMentioned", "spans": []},
                            "nda-4": {"choice": "NotMentioned", "spans": []},
                        },
                    },
                ],
            },
        ],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_labelled_items_keep_direction_and_spans_and_nulls_are_sampled(tmp_path: Path) -> None:
    items, texts = data.load(split(tmp_path), documents=1, nulls_per_document=1)
    by_key = {i.key: i for i in items}
    assert by_key["nda-1"].expected == "yes"
    assert by_key["nda-1"].gold == ((14, 38),)
    assert by_key["nda-2"].expected == "no"
    assert sum(i.is_null for i in items) == 1
    assert texts == {"7": TEXT}


def test_the_question_names_all_three_outcomes(tmp_path: Path) -> None:
    items, _ = data.load(split(tmp_path), documents=1)
    question = items[0].question
    assert question.startswith("Does the agreement say that receiving Party shall return information?")
    assert "not stated" in question


@pytest.mark.parametrize(
    ("text", "verdict"),
    [("Yes, it does [1].", "yes"), ("**No.** It says the opposite.", "no"), ("Not stated.", None), ("Yesterday", None)],
)
def test_the_verdict_is_the_first_word(text: str, verdict: str | None) -> None:
    assert data.verdict_of(text) == verdict


def test_a_small_subset_still_holds_every_label(tmp_path: Path) -> None:
    items, _ = data.load(split(tmp_path), documents=1, nulls_per_document=2)
    assert {i.label for i in data.subset(items, 2)} == {"Entailment", "Contradiction", "NotMentioned"}


def evidence(query_id: str) -> list[Evidence]:
    return [
        Evidence(text=TEXT[a:b], provenance=Provenance(doc_id="7", chunk_id=f"7::{n}", start=a, end=b))
        for n, (a, b) in enumerate([(0, 13), (14, 38), (39, 52)])
    ]


def test_shuffling_moves_every_citation_and_keeps_the_statements() -> None:
    original = parse_inline("7::nda-1", "It says return [2]. It is short [1][3].", evidence(""))
    moved = shuffle_citations([original], evidence, seed=0)[0]
    assert [s.text for s in moved.statements] == [s.text for s in original.statements]
    for before, after in zip(original.statements, moved.statements):
        assert len(after.citations) == len(before.citations)
        assert all(b.source != a.source for b, a in zip(before.citations, after.citations))
    assert moved.style == "shuffled"


def test_an_abstention_survives_the_shuffle() -> None:
    abstained = CitedAnswer("q", (), "inline", abstained=True)
    assert shuffle_citations([abstained], evidence, seed=0) == [abstained]


class FakeChecker:
    def __init__(self) -> None:
        self.batches: list[int] = []
        self._memo: dict[tuple[str, str], float] = {}

    def claim_for(self, claim: str, question: str = "") -> str:
        return claim

    def probabilities(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        self.batches.append(len(pairs))
        return [1.0 for _ in pairs]


def test_batching_asks_for_joint_and_single_pairs_in_one_call() -> None:
    answer = parse_inline("q", "It says return [2][3].", evidence(""))
    checker = FakeChecker()
    score_in_batches(checker, [answer])  # type: ignore[arg-type]
    assert checker.batches == [3]


def test_outcomes_read_verdicts_refusals_and_abstentions(tmp_path: Path) -> None:
    items, _ = data.load(split(tmp_path), documents=1)
    item = items[0]
    assert report.outcome(item, parse_inline(item.query_id, "Yes, it does [2].", evidence(""))) == "yes"
    assert report.outcome(item, CitedAnswer(item.query_id, (), "inline", abstained=True)) == "not stated"
    assert report.outcome(item, parse_inline(item.query_id, "It is short [1].", evidence(""))) == "unclear"


def test_rank1_points_every_statement_at_the_top_passage() -> None:
    original = parse_inline("q", "It says return [2]. It is short [3].", evidence(""))
    ranked = cite_rank1([original], evidence)[0]
    assert [[c.source for c in s.citations] for s in ranked.statements] == [[1], [1]]


class WordChecker(FakeChecker):
    """Scores a pair 1.0 when the claim's last word appears in the premise."""

    def probabilities(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        return [1.0 if claim.split()[-1].strip(".").lower() in premise.lower() else 0.0 for premise, claim in pairs]


def test_posthoc_cites_the_best_sentence_and_leaves_unsupported_statements_bare() -> None:
    original = parse_inline("q", "The text says return [1]. Nothing mentions zebras [1].", evidence(""))
    moved = attribute_posthoc([original], evidence, WordChecker())[0]  # type: ignore[arg-type]
    first, second = moved.statements
    assert first.citations[0].source == 2
    assert TEXT[first.citations[0].start : first.citations[0].end].strip() == "Beta clause says return."
    assert second.citations == ()
