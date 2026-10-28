"""Unit tests for the replication measures that need no model, corpus, or parser."""

from __future__ import annotations

from collections import Counter

import pytest
from vdocs import replication


def test_remap_pages_points_every_page_number_at_the_source() -> None:
    exported = {
        "pages": {"1": {"page_no": 1, "size": {}}, "2": {"page_no": 2, "size": {}}},
        "texts": [{"prov": [{"page_no": 2, "bbox": {}}]}],
        "tables": [{"prov": [{"page_no": 1}], "data": {}}],
    }
    out = replication.remap_pages(exported, [4, 27])
    assert sorted(out["pages"]) == ["28", "5"]
    assert out["pages"]["28"]["page_no"] == 28
    assert out["texts"][0]["prov"][0]["page_no"] == 28
    assert out["tables"][0]["prov"][0]["page_no"] == 5
    assert exported["texts"][0]["prov"][0]["page_no"] == 2


def test_numbers_drops_formatting_and_keeps_multiplicity() -> None:
    assert replication.numbers("Revenue $1,234.5 and (12) and 12% and 2021") == Counter(
        {"1234.5": 1, "12": 2, "2021": 1}
    )


def test_page_fidelity_counts_lost_and_invented_numbers() -> None:
    row = replication.page_fidelity("Net sales 1,200 in 2021", "Net sa1es 1,200 in 2O21 7")
    assert row["numbers_ref"] == 2
    assert row["numbers_kept"] == 1
    assert row["numbers_extra"] == 1
    assert not row["identical"]
    assert 0 < row["word_accuracy"] < 1


def test_identical_text_scores_one() -> None:
    row = replication.page_fidelity("| a | 1 |\n|---|---|", "| a | 1 |\n|---|---|")
    assert row["identical"] and row["word_accuracy"] == 1.0


def test_table_cells_reads_single_number_cells_on_one_page() -> None:
    exported = {
        "tables": [
            {
                "prov": [{"page_no": 3}],
                "data": {"table_cells": [{"text": "$1,000"}, {"text": "Total"}, {"text": "1 2"}]},
            },
            {"prov": [{"page_no": 4}], "data": {"table_cells": [{"text": "5"}]}},
        ]
    }
    assert replication.table_cells(exported, 3) == Counter({"1000": 1})
    assert replication.table_count(exported, 4) == 1


@pytest.mark.parametrize(("area", "name"), [(0.001, "<0.5%"), (0.005, "0.5-1%"), (0.04, "1-5%"), (0.3, ">=10%")])
def test_area_bins(area: float, name: str) -> None:
    assert replication.area_bin(area) == name


def test_unsupported_summary() -> None:
    judged = [
        replication.count_verdicts({"claims": [{"verdict": "supported"}, {"verdict": "unsupported"}]}),
        replication.count_verdicts({"claims": [{"verdict": "supported"}, {"verdict": "generic"}]}),
        replication.count_verdicts({}),
    ]
    summary = replication.summarise_unsupported(judged)
    assert summary == {"n": 2, "claims_per_description": 1.5, "any_unsupported": 0.5, "claim_share": 0.3333}
