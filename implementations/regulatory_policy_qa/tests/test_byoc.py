"""The bring-your-own-corpus notebook's reference set and scores, and its settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aieng.advanced_rag.models import family_of


RI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RI))

import byoc  # noqa: E402


def envelopes(artifact_type: str) -> list[dict]:
    paths = sorted(byoc.BASELINE.glob(f"{byoc.RI_ID}/{artifact_type}/**/*.json"))
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return [e for e in loaded if e["producer_id"] == byoc.PRODUCER]


def test_the_stand_in_reference_set_is_shipped_with_its_models():
    found = envelopes("question-set")
    assert len(found) == 1
    value = found[0]["value"]
    assert family_of(value["produced_by"]["model"]) == "onprem"
    result = value["result"]
    assert result["meta"]["config"] == json.loads(json.dumps(byoc.DEMO_CONFIG.to_dict()))
    assert result["meta"]["models"]["generator"] == "Qwen3_8-27B"
    answerable = [q for q in result["questions"] if q["evidence"]]
    assert answerable and len(answerable) < len(result["questions"])


def test_the_stand_in_scores_cover_every_arm():
    found = envelopes("synthetic-arm-scores")
    assert len(found) == 1
    scores = found[0]["value"]
    assert len(scores["arms"]) == 8
    assert set(scores["mean"]["ndcg@10"]) == set(scores["arms"])
    assert scores["versus_best_ndcg@10"]["best"] in scores["arms"]


def test_the_small_settings_keep_a_first_run_cheap():
    assert byoc.DEMO_CONFIG.passages_per_document == 2
    assert byoc.DEMO_CONFIG.paraphrase is False
    generic = byoc.config_for(None, demo=False)
    assert generic.passages_per_document == 2 and generic.scope == "corpus"
