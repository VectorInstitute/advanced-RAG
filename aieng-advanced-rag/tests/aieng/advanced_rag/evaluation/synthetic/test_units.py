"""Quote location, the filters, the funnel, the config, and rank agreement."""

from __future__ import annotations

import json
import math

import pytest

from aieng.advanced_rag.evaluation.synthetic import (
    EXAMPLE_CONFIG,
    GenerationConfig,
    config_from_dict,
    kendall_tau,
    load_config,
    locate_quote,
    pairwise_agreement,
)
from aieng.advanced_rag.evaluation.synthetic.agreement import compare_rankings
from aieng.advanced_rag.evaluation.synthetic.filters import (
    corpus_answers,
    deduplicate,
    round_trip,
    uniqueness,
)
from aieng.advanced_rag.evaluation.synthetic.llm import family, parse_json
from aieng.advanced_rag.evaluation.synthetic.probe import Hit, ProbeIndex
from aieng.advanced_rag.evaluation.synthetic.sampling import (
    Document,
    held_out_split,
    passages_of,
    sample_passages,
)
from aieng.advanced_rag.evaluation.synthetic.statements import extract_statements
from aieng.advanced_rag.evaluation.synthetic.testset import Funnel
from aieng.advanced_rag.retrieval.corpus import Chunk

from .conftest import model


TEXT = (
    "1. Term.\nThe Recipient shall hold the Information in confidence for five (5) "
    "years.\n2. Return.  The Recipient\nshall return all copies on “written” "
    "request."
)


class TestLocateQuote:
    def test_an_exact_quote_maps_to_its_offsets(self):
        quote = "hold the Information in confidence"
        start, end = locate_quote(TEXT, quote)
        assert TEXT[start:end] == quote

    def test_whitespace_case_and_curly_quotes_are_forgiven(self):
        span = locate_quote(
            TEXT, 'the recipient shall return all copies on "written" request.'
        )
        assert span is not None
        start, end = span
        assert TEXT[start:end].startswith("The Recipient\nshall return")
        assert TEXT[start:end].endswith("request.")

    def test_an_absent_quote_is_none(self):
        assert locate_quote(TEXT, "The Recipient may sell the Information.") is None

    def test_the_search_stays_inside_the_window(self):
        start = TEXT.index("2. Return")
        assert locate_quote(TEXT, "hold the Information", start=start) is None

    def test_a_very_short_folded_quote_is_refused(self):
        assert locate_quote(TEXT, "  ") is None
        assert locate_quote(TEXT, "RECIPIENT") is None

    def test_extract_statements_drops_what_it_cannot_find(self):
        document = Document("d", TEXT)
        passage = passages_of(document, target_chars=50, min_chars=10)[0]
        reply = json.dumps(
            {
                "statements": [
                    "1. Term.\nThe Recipient",
                    "not in the text at all, really",
                ]
            }
        )
        found, missing = extract_statements(
            model("g", lambda s, p: reply), passage, TEXT, count=2
        )
        assert [s.quote for s in found] == ["1. Term.\nThe Recipient"]
        assert missing == ["not in the text at all, really"]


class TestSampling:
    def test_held_out_documents_are_disjoint_and_at_least_the_minimum(self):
        documents = [Document(f"d{i}", "x " * 100) for i in range(40)]
        indexed, held = held_out_split(documents, share=0.1, minimum=3)
        assert len(held) == 4
        assert not {d.doc_id for d in indexed} & {d.doc_id for d in held}

    def test_passages_carry_their_offsets(self, corpus):
        for passage in passages_of(corpus[0], target_chars=300):
            assert corpus[0].text[passage.start : passage.end] == passage.text

    def test_sampling_spreads_across_each_document(self, corpus):
        passages, report = sample_passages(corpus, per_document=3, target_chars=300)
        per_doc = {}
        for passage in passages:
            per_doc.setdefault(passage.doc_id, []).append(passage.start)
        assert all(len(starts) == 3 for starts in per_doc.values())
        assert report["passages_drawn"] == len(passages)

    def test_boilerplate_is_drawn_once(self):
        same = "The parties agree to keep everything confidential forever. " * 8
        documents = [Document("a", same), Document("b", same)]
        passages, report = sample_passages(
            documents,
            per_document=1,
            encode=lambda texts: [[1.0, 0.0] for _ in texts],
            target_chars=200,
        )
        assert len(passages) == 1
        assert report["skipped_as_boilerplate"] == 1


def hits(texts: list[str], doc_id: str = "d") -> list[Hit]:
    offset = 0
    out = []
    for rank, text in enumerate(texts, start=1):
        out.append(
            Hit(
                Chunk(
                    f"{doc_id}::{rank:04d}", doc_id, text, offset, offset + len(text)
                ),
                1.0,
                rank,
            )
        )
        offset += len(text)
    return out


class TestFilters:
    def test_the_round_trip_rejects_a_mismatched_answer(self):
        checker = model("c", lambda s, p: '{"answerable": true, "answer": "ten years"}')
        judge = model("j", lambda s, p: '{"verdict": "mismatch", "reason": "differs"}')
        result = round_trip(
            checker, judge, "How long?", "five years", ["five (5) years"]
        )
        assert not result["passed"]
        assert result["verdict"] == "mismatch"

    def test_the_round_trip_skips_the_judge_when_the_quote_has_no_answer(self):
        calls = []
        judge = model("j", lambda s, p: calls.append(p) or '{"verdict": "match"}')
        checker = model("c", lambda s, p: '{"answerable": false, "answer": ""}')
        assert not round_trip(checker, judge, "q", "a", ["x"])["passed"]
        assert calls == []

    def test_uniqueness_adds_an_equivalent_passage(self):
        passages = hits(
            ["Clause nine repeats: confidential for five years.", "Unrelated."]
        )
        texts = {"d": "".join(h.chunk.text for h in passages)}
        reply = json.dumps(
            {
                "answering": [
                    {
                        "passage": 1,
                        "quote": "confidential for five years.",
                        "agrees": True,
                    }
                ]
            }
        )
        found = uniqueness(
            model("c", lambda s, p: reply), "How long?", "five years", passages, texts
        )
        assert found.status == "equivalent"
        (span,) = found.equivalent
        assert texts["d"][span.start : span.end] == "confidential for five years."
        assert span.role == "equivalent"

    def test_uniqueness_drops_an_ambiguous_question(self):
        passages = hits(["Confidential for ten years."])
        reply = '{"answering": [{"passage": 1, "quote": "ten years", "agrees": false}]}'
        found = uniqueness(
            model("c", lambda s, p: reply),
            "How long?",
            "five years",
            passages,
            {"d": passages[0].chunk.text},
        )
        assert found.status == "ambiguous"

    def test_uniqueness_with_no_answering_passage_is_unique(self):
        passages = hits(["Unrelated."])
        found = uniqueness(
            model("c", lambda s, p: '{"answering": []}'),
            "q",
            "a",
            passages,
            {"d": "Unrelated."},
        )
        assert found.status == "unique"

    def test_dedup_keeps_one_of_two_near_identical_questions(self):
        vectors = {"a": [1.0, 0.0], "b": [0.999, 0.0447], "c": [0.0, 1.0]}
        kept, dropped = deduplicate(
            [("a", "d1", "x"), ("b", "d1", "x'"), ("c", "d2", "y")], vectors, 0.92
        )
        assert kept == ["a", "c"]
        assert dropped[0].dropped == "b"

    def test_dedup_prefers_the_less_covered_document(self):
        vectors = {"a": [1.0, 0.0], "a2": [0.0, 1.0], "b": [1.0, 0.0]}
        # d1 already has two kept questions when b arrives from uncovered d2,
        # but b's duplicate is a, from d1, so b replaces a.
        kept, _ = deduplicate(
            [("a", "d1", ""), ("a2", "d1", ""), ("b", "d2", "")], vectors, 0.92
        )
        assert set(kept) == {"a2", "b"}

    def test_an_unparseable_null_check_counts_as_answered(self):
        verdict = corpus_answers(model("c", lambda s, p: "no json"), "q", hits(["x"]))
        assert verdict["answered"] == "yes"


class TestProbe:
    def test_gold_rank_and_exclusion(self, corpus, embedder):
        probe = ProbeIndex(corpus[:2], embedder, chunk_chars=300, overlap_chars=50)
        text = corpus[0].text
        start = text.index("Clause 3.")
        span = ("nda-0", start, start + 40)
        rank = probe.gold_rank(
            "Acme roadmaps confidential years", [span], method="bm25", scope="nda-0"
        )
        assert rank is not None
        excluded = probe.search(
            "Acme roadmaps", method="bm25", k=50, scope="nda-0", exclude=[span]
        )
        assert all(
            not (h.chunk.start < span[2] and span[1] < h.chunk.end) for h in excluded
        )
        with pytest.raises(ValueError, match="method"):
            probe.search("x", method="splade")


class TestFunnel:
    def test_a_candidate_ends_once(self):
        funnel = Funnel()
        funnel.finish("c1", "single_hop", "accepted", "accepted")
        with pytest.raises(ValueError, match="already ended"):
            funnel.finish("c1", "single_hop", "filtered", "round_trip")
        with pytest.raises(ValueError, match="outcome"):
            funnel.finish("c2", "single_hop", "kept", "x")

    def test_stage_rates_are_per_kind(self):
        funnel = Funnel()
        funnel.check("null_check", True, "held_out_null")
        funnel.check("null_check", False, "near_miss_null")
        stages = {
            (s["kind"], s["stage"]): s["acceptance_rate"] for s in funnel.stages()
        }
        assert stages == {
            ("held_out_null", "null_check"): 1.0,
            ("near_miss_null", "null_check"): 0.0,
        }


class TestConfig:
    def test_the_example_config_loads(self):
        config = load_config(EXAMPLE_CONFIG)
        assert config.domain.startswith("loan agreements")
        assert len(config.few_shot) == 3
        assert {c.name for c in config.users} == {"credit_analyst", "borrower"}

    def test_json_configs_load_and_unknown_keys_fail(self, tmp_path):
        path = tmp_path / "team.json"
        path.write_text(
            json.dumps({"domain": "patents", "few_shot": ["prior art for X?"]})
        )
        assert load_config(path).few_shot == ("prior art for X?",)
        with pytest.raises(ValueError, match="unknown config key"):
            config_from_dict({"domian": "typo"})

    def test_the_category_draw_is_stable_per_key(self):
        config = GenerationConfig()
        assert config.sample("k1") == config.sample("k1")
        names = {config.sample(f"k{i}")["question"].name for i in range(60)}
        assert len(names) > 1

    def test_an_invalid_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            GenerationConfig(scope="page")


class TestAgreement:
    def test_identical_rankings_give_one(self):
        assert kendall_tau([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_reversed_rankings_give_minus_one(self):
        assert kendall_tau([0.1, 0.2, 0.3], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_one_adjacent_swap_of_four(self):
        # Six pairs, one discordant: (5 - 1) / 6.
        assert kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)

    def test_a_constant_list_is_nan(self):
        assert math.isnan(kendall_tau([1, 1, 1], [1, 2, 3]))

    def test_pairwise_agreement_flags_a_significant_reversal(self):
        questions = [f"q{i}" for i in range(40)]
        real = {"a": dict.fromkeys(questions, 1.0), "b": dict.fromkeys(questions, 0.0)}
        generated = {
            "a": dict.fromkeys(questions, 0.0),
            "b": dict.fromkeys(questions, 1.0),
        }
        (pair,) = pairwise_agreement(real, generated, samples=200, seed=3)
        assert pair["real"]["significant"] and pair["generated"]["significant"]
        assert not pair["same_sign"]
        assert pair["significant_reversal"]

    def test_compare_rankings_reports_tau_and_gap(self):
        questions = [f"q{i}" for i in range(20)]
        real = {
            arm: dict.fromkeys(questions, score)
            for arm, score in (("x", 0.2), ("y", 0.5), ("z", 0.8))
        }
        generated = {
            arm: dict.fromkeys(questions, score + 0.1)
            for arm, score in (("x", 0.2), ("y", 0.5), ("z", 0.8))
        }
        summary = compare_rankings(real, generated, samples=200)
        assert summary["kendall_tau"] == 1.0
        assert summary["real_ranking"] == ["z", "y", "x"]
        assert summary["mean_gap"] == pytest.approx(0.1)
        assert summary["sign_agreement"] == 1.0


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure: {"a": 2} done') == {"a": 2}
    assert parse_json("[1, 2]") == {}
    assert parse_json("") == {}


def test_family_groups_model_names():
    assert family("Qwen3_8-27B") == family("Qwen3_6-35B-A3B") == "qwen"
    assert family("claude-haiku-5-5") == "claude"
    assert family("gpt-5.4-mini") != family("gemini-3.5-flash-lite")
