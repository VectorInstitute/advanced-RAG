"""The committed RI #3 results are consistent with each other and carry no text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
STRATEGIES = ("plain", "table_aware")


@pytest.fixture(scope="module")
def scores() -> dict:
    return json.loads((OUTPUTS / "table_retrieval_scores.json").read_text())


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((OUTPUTS / "ingestion_summary.json").read_text())


def test_the_summary_totals_are_the_sum_of_its_documents(summary):
    documents = summary["per_document"]
    assert summary["documents"] == len(documents) == 84
    for field in ("pages", "tables", "failures"):
        assert summary[field] == sum(d[field] for d in documents.values())
    assert summary["failures"] == 0


def test_both_chunkings_score_every_question_over_every_filing(scores, summary):
    for strategy in STRATEGIES:
        row = scores[strategy]
        assert row["filings"] == summary["documents"]
        assert row["queries_scored"] == 150
        assert row["uncovered"] == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_rate_is_a_rate(scores, strategy):
    row = scores[strategy]
    for arm in row["retrieval"].values():
        assert arm["queries"] == 150
        for name in ("retrieval_failure_rate", "hit_rate", "mrr"):
            assert 0.0 <= arm[name] <= 1.0
        assert arm["retrieval_failure_rate"] + arm["hit_rate"] == pytest.approx(1.0)
    attached = row["attachment"]
    assert attached["attached_pairs"] <= attached["chunk_cell_pairs"]
    assert attached["cells_ever_complete"] <= attached["scored_cells"]


def test_table_aware_chunking_attaches_more_cells_than_plain_windows(scores):
    plain, aware = (scores[s]["attachment"] for s in STRATEGIES)
    assert aware["attachment_rate"] > plain["attachment_rate"]
    assert aware["cell_recoverable_rate"] > plain["cell_recoverable_rate"]


def test_no_result_file_carries_question_or_filing_text():
    for path in OUTPUTS.glob("*.json"):
        text = path.read_text()
        assert "evidence_text" not in text and "question" not in text, path.name
