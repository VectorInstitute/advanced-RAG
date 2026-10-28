"""The special topic's helpers and its committed reference artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqsets
from aieng.advanced_rag.evaluation.synthetic import (
    EvidenceSpan,
    Passage,
    SyntheticQuestion,
    SyntheticSet,
)
from aieng.advanced_rag.evaluation.synthetic.statements import Statement
from aieng.advanced_rag.models import family_of
from explore_adjacent_statements import pairs_of, widened


HERE = Path(__file__).resolve().parents[1]
RI1_VALIDATION = HERE.parents[1] / "regulatory_policy_qa" / "outputs" / "synthetic_qa" / "contract_nli_validation.json"


def statement(passage: Passage, start: int, end: int) -> Statement:
    quote = "x" * (end - start)
    return Statement(passage, start, end, quote, quote)


def test_pairs_take_the_first_two_statements_that_do_not_overlap():
    one = Passage("d", 0, 100, "x" * 100)
    two = Passage("d", 100, 200, "x" * 100)
    pairs = pairs_of(
        [
            statement(one, 40, 60),
            statement(one, 10, 30),
            statement(one, 20, 50),
            statement(two, 120, 140),
        ]
    )
    assert [(a.start, b.start) for a, b in pairs] == [(10, 40)]


def test_widening_grows_the_span_around_itself_and_keeps_the_quote_honest():
    text = "".join(chr(97 + i % 26) for i in range(200))
    question = SyntheticQuestion(
        query_id="q",
        question="?",
        reference="r",
        evidence=(EvidenceSpan("d", 100, 110, text[100:110]),),
        scope="d",
    )
    wide = widened(SyntheticSet([question]), {"d": text}, 50).questions[0].evidence[0]
    assert (wide.start, wide.end) == (80, 130)
    assert wide.quote == text[wide.start : wide.end]
    edge = SyntheticQuestion("e", "?", "r", (EvidenceSpan("d", 0, 10, text[:10]),), scope="d")
    clipped = widened(SyntheticSet([edge]), {"d": text}, 50).questions[0].evidence[0]
    assert clipped.start == 0 and clipped.end - clipped.start == 50


def test_ranks_put_the_highest_mean_first():
    assert sqsets.ranks({"a": 0.2, "b": 0.9, "c": 0.5}) == {"b": 1, "c": 2, "a": 3}


def envelopes(artifact_type: str) -> list[dict]:
    paths = sorted(sqsets.BASELINE.glob(f"{sqsets.RI_ID}/{artifact_type}/**/*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_the_reference_validation_matches_the_committed_ri1_result():
    found = envelopes("validation")
    assert len(found) == 1, "expected one reference validation artifact"
    value = found[0]["value"]
    produced = value["produced_by"]
    assert family_of(produced["model"]) == "onprem"
    record = value["result"]["validation"]
    committed = json.loads(RI1_VALIDATION.read_text(encoding="utf-8"))
    for variant in ("plain", "paraphrase"):
        for metric in ("ndcg@10", "recall@3", "mrr"):
            ours = record["agreement"][variant]["all_arms"][metric]
            theirs = committed["agreement"][variant]["all_arms"][metric]
            assert ours["kendall_tau"] == theirs["kendall_tau"], (variant, metric)
            assert ours["mean_gap"] == theirs["mean_gap"], (variant, metric)
    plain = record["agreement"]["plain"]["all_arms"]["ndcg@10"]
    assert plain["kendall_tau"] == pytest.approx(0.8571, abs=1e-4)
    assert record["agreement"]["paraphrase"]["all_arms"]["ndcg@10"]["kendall_tau"] == pytest.approx(0.5714, abs=1e-4)


def test_the_explore_result_records_its_models_and_all_three_label_sets():
    found = envelopes("explore-adjacent-statements")
    assert len(found) == 1
    value = found[0]["value"]
    assert family_of(value["produced_by"]["model"]) == "onprem"
    result = value["result"]
    assert set(result["label_sets"]) == {"plain", "pairs", "widened"}
    assert result["mode"].startswith("explore")
    for row in result["label_sets"].values():
        assert -1.0 <= row["all_arms"]["ndcg@10"]["kendall_tau"] <= 1.0
