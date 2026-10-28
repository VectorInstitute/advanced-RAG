"""The cached-or-live path for question sets, the folder loader, and the BEIR export."""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.evaluation.synthetic import GenerationConfig
from aieng.advanced_rag.evaluation.synthetic.replay import (
    REFERENCE_ROLES,
    Reference,
    Roles,
    corpus_digest,
    default_checker,
    load_folder,
    load_or_build,
    rebuild_estimate,
    set_from_value,
    set_step,
    set_value,
    write_beir,
)
from aieng.advanced_rag.evaluation.synthetic.sampling import Document
from aieng.advanced_rag.evaluation.synthetic.validation import CORPUS_SCOPE, queries_for
from aieng.advanced_rag.models import choose_models
from aieng.advanced_rag.runs import Provenance, ReferenceMissingError
from aieng.advanced_rag.spend import ModelPrice

from .test_build import build


DOCS = [Document("a", "alpha " * 40), Document("b", "beta " * 40)]
CONFIG = GenerationConfig(domain="test documents")


def test_roles_check_with_another_family_than_the_generator():
    google = Roles.from_models(choose_models("google"))
    assert google == Roles("gemini-3.5-flash-lite", "gpt-6-luna", "gemini-3.8-flash")
    assert not google.same_family()
    for family in ("anthropic", "openai"):
        roles = Roles.from_models(choose_models(family))
        assert roles.checker == "gemini-3.5-flash-lite"
        assert not roles.same_family()


def test_an_onprem_generator_keeps_a_free_checker_that_differs():
    roles = Roles.from_models(choose_models("onprem"))
    assert roles.generator == "Qwen3_6-35B-A3B" and roles.checker == "Qwen3_8-27B"
    assert default_checker("Qwen3_8-27B") == "Qwen3_6-35B-A3B"
    assert default_checker("gpt-oss-120b") == "Qwen3_6-35B-A3B"


def test_default_checker_reads_the_family_not_the_name_prefix():
    assert default_checker("gemini-4-argon") == "gpt-6-luna"
    assert default_checker("claude-haiku-5-5") == "gemini-3.5-flash-lite"


def test_an_explicit_checker_overrides_the_default():
    judging = Roles.from_models(choose_models("google"), checker="gemini-3.8-flash")
    assert judging.checker == "gemini-3.8-flash" and judging.same_family()
    crossed = Roles.from_models(choose_models("google"), checker="claude-haiku-5-5")
    assert crossed.checker == "claude-haiku-5-5" and not crossed.same_family()


def test_the_key_names_the_checker_and_the_judge_and_the_corpus():
    roles = Roles("gen", "check", "judge")
    step = set_step("ri", "producer", DOCS, CONFIG, roles)
    assert step.config["checker"] == "check" and step.config["judge"] == "judge"
    other = set_step("ri", "producer", DOCS, CONFIG, Roles("gen", "other", "judge"))
    assert step.config != other.config
    changed = [Document("a", "alpha " * 41), DOCS[1]]
    assert corpus_digest(changed) != corpus_digest(DOCS)


def test_the_estimate_prices_each_role_at_its_own_rate():
    table = {
        "cheap": ModelPrice("cheap", 1.0, 2.0),
        "judge": ModelPrice("judge", 10.0, 20.0),
    }
    provenance = Provenance(
        model=REFERENCE_ROLES.generator,
        usage={
            REFERENCE_ROLES.generator: {
                "calls": 10,
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
            },
            REFERENCE_ROLES.checker: {
                "calls": 10,
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
            },
            REFERENCE_ROLES.judge: {
                "calls": 4,
                "prompt_tokens": 0,
                "completion_tokens": 1_000_000,
            },
        },
    )
    roles = Roles("cheap", "judge", "judge")
    calls, dollars = rebuild_estimate(provenance, REFERENCE_ROLES, roles, table=table)
    assert calls == 24
    assert dollars == pytest.approx(1.0 + 10.0 + 20.0)
    half = rebuild_estimate(provenance, REFERENCE_ROLES, roles, scale=0.5, table=table)
    assert half == (12, pytest.approx(15.5))
    unpriced = rebuild_estimate(
        provenance, REFERENCE_ROLES, Roles("cheap", "nobody", "judge"), table=table
    )
    assert unpriced == (24, None)


@pytest.fixture
def cache(tmp_path) -> TieredResultCache:
    return TieredResultCache(
        local_dir=tmp_path / "local", global_dir=tmp_path / "shipped"
    )


def test_without_a_result_of_yours_the_reference_is_shown_with_your_estimate(
    cache, capsys
):
    reference_step = set_step("ri", "producer", DOCS, CONFIG, REFERENCE_ROLES)
    load_or_build(
        reference_step,
        lambda choice: {"built_by": choice.model},
        roles=REFERENCE_ROLES,
        live=True,
        cache=cache,
    )
    roles = Roles("gen-x", "check-y", "judge-y")
    mine = set_step("ri", "producer", DOCS, CONFIG, roles)
    calls = []
    capsys.readouterr()
    result = load_or_build(
        mine,
        lambda choice: calls.append(choice.model) or {"built_by": choice.model},
        roles=roles,
        live=False,
        cache=cache,
        reference=Reference(reference_step, description="(the reference)"),
    )
    printed = capsys.readouterr().out
    assert calls == []
    assert result.origin == "reference"
    assert result.value == {"built_by": REFERENCE_ROLES.generator}
    assert "reference artifact" in printed and "(the reference)" in printed
    assert "your build: gen-x generates, check-y checks, judge-y judges" in printed


def test_a_live_build_runs_once_and_is_then_yours(cache, capsys):
    reference_step = set_step("ri", "producer", DOCS, CONFIG, REFERENCE_ROLES)
    load_or_build(
        reference_step,
        lambda choice: {"built_by": choice.model},
        roles=REFERENCE_ROLES,
        live=True,
        cache=cache,
    )
    roles = Roles("gen-x", "check-y", "judge-y")
    mine = set_step("ri", "producer", DOCS, CONFIG, roles)
    calls = []
    capsys.readouterr()

    def compute(choice):
        calls.append(choice.model)
        return {"built_by": choice.model}

    first = load_or_build(
        mine,
        compute,
        roles=roles,
        live=True,
        cache=cache,
        reference=Reference(reference_step),
    )
    again = load_or_build(
        mine,
        compute,
        roles=roles,
        live=False,
        cache=cache,
        reference=Reference(reference_step),
    )
    assert calls == ["gen-x"]
    assert first.value == again.value == {"built_by": "gen-x"}
    assert again.origin == "yours"
    assert "running on" not in capsys.readouterr().out


def test_no_reference_and_no_flag_says_how_to_build(cache):
    mine = set_step(
        "ri", "producer", DOCS, CONFIG, Roles("gen-x", "check-y", "judge-y")
    )
    with pytest.raises(ReferenceMissingError, match="RUN_EXPENSIVE"):
        load_or_build(
            mine,
            lambda choice: {},
            roles=Roles("gen-x", "check-y", "judge-y"),
            live=False,
            cache=cache,
        )


def test_a_stored_set_round_trips(corpus, script, embedder):
    generated = build(corpus, script, embedder)
    value = json.loads(json.dumps(set_value(generated)))
    restored = set_from_value(value)
    assert [q.to_dict() for q in restored] == [q.to_dict() for q in generated]
    assert restored.meta["funnel"]["accepted"] == generated.funnel.to_dict()["accepted"]


def test_the_beir_export_names_relevance_for_one_chunking(
    corpus, script, embedder, tmp_path
):
    generated = build(corpus, script, embedder)
    chunks = [
        c
        for d in corpus
        for c in chunk_document(d.doc_id, d.text, chunk_chars=300, overlap_chars=50)
    ]
    counts = write_beir(generated, chunks, tmp_path / "beir")
    lines = (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()
    assert lines[0] == "query-id\tcorpus-id\tscore"
    assert counts["qrels"] == len(lines) - 1 > 0
    assert counts["queries"] == len(generated)
    judged = {line.split("\t")[0] for line in lines[1:]}
    assert judged == {q.query_id for q in generated.answerable}
    first = json.loads((tmp_path / "beir" / "corpus.jsonl").read_text().splitlines()[0])
    assert set(first) == {"_id", "title", "text"}


def test_the_folder_loader_reads_text_markdown_and_skips_the_rest(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "one.txt").write_text("first document")
    (tmp_path / "sub" / "two.md").write_text("# second")
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "table.csv").write_text("a,b")
    documents = load_folder(tmp_path)
    assert [d.doc_id for d in documents] == ["one.txt", "sub/two.md"]
    assert [d.doc_id for d in load_folder(tmp_path, limit=1)] == ["one.txt"]
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path / "sub" / "missing")


def test_corpus_wide_queries_share_one_scope(corpus, script, embedder):
    generated = build(corpus, script, embedder)
    scoped = queries_for(generated)
    wide = queries_for(generated, corpus_wide=True)
    assert {q.doc_id for q in wide} == {CORPUS_SCOPE}
    assert {q.doc_id for q in scoped} <= {d.doc_id for d in corpus}
    assert [q.query_id for q in wide] == [q.query_id for q in scoped]


def test_a_live_build_does_not_need_the_reference_artifact(cache):
    reference_step = set_step("ri", "producer", DOCS, CONFIG, REFERENCE_ROLES)
    roles = Roles("gen-x", "check-y", "judge-y")
    mine = set_step("ri", "producer", DOCS, CONFIG, roles)
    result = load_or_build(
        mine,
        lambda choice: {"built_by": choice.model},
        roles=roles,
        live=True,
        cache=cache,
        reference=Reference(reference_step),
    )
    assert result.value == {"built_by": "gen-x"}
    with pytest.raises(ReferenceMissingError):
        other = set_step(
            "ri", "producer", DOCS, CONFIG, Roles("gen-z", "check-y", "judge-y")
        )
        load_or_build(
            other,
            lambda choice: {},
            roles=Roles("gen-z", "check-y", "judge-y"),
            cache=cache,
            reference=Reference(reference_step),
        )
