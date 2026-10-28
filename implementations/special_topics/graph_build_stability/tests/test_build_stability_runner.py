"""The runner's parsing and naming need no model or corpus."""

from __future__ import annotations

import run_stability as rs


def test_build_names_carry_every_varied_setting() -> None:
    assert rs.build_name("atlas", "Qwen3_6-35B-A3B", 0.7, 2) == "atlas-Qwen3_6-35B-A3B-t0.7-r2"
    assert rs.build_name("atlas", "m", 0.0, 1) == "atlas-m-t0-r1"
    assert rs.build_name("atlas", "m", None, 1) == "atlas-m-tdefault-r1"


def test_atlas_parse_reads_triples_and_rejects_garbage() -> None:
    raw = '```json\n[{"Head": "A", "Relation": "r", "Tail": "B"}, {"Head": "A", "Relation": "r", "Tail": "B"}]\n```'
    assert rs.atlas_parse(raw) == [{"Head": "A", "Relation": "r", "Tail": "B"}]
    assert rs.atlas_parse("<think>x</think>[]") == []


def test_schema_messages_add_the_closed_relation_list() -> None:
    free = rs.musique_messages("atlas", "Title\nText")
    closed = rs.musique_messages("schema", "Title\nText")
    assert free[0] == closed[0]
    assert "happened_on" in closed[1]["content"] and "happened_on" not in free[1]["content"]
    assert closed[1]["content"].endswith("\nTitle\nText")
