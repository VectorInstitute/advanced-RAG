"""Tests for the abstention variants and the question-aware MiniCheck claim."""

from __future__ import annotations

import json
from pathlib import Path

from aieng.advanced_rag.evaluation.citations import QUESTION_CLAIM, CitedAnswer, MiniCheckChecker, Statement
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance
from cites import explore


TEXT = "Alpha clause. Beta clause says return. Gamma clause."
RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "results.json"
LIVE = Path(__file__).resolve().parents[1] / "outputs" / "live_check.json"


def document() -> Evidence:
    provenance = Provenance(doc_id="7", chunk_id="7::full", start=0, end=len(TEXT))
    return Evidence(text=TEXT, provenance=provenance, retriever="none")


def payload(*statements: dict) -> str:
    return json.dumps({"answer": "No", "statements": list(statements), "not_stated": False})


def test_a_statement_resting_on_the_sentinel_is_dropped_and_an_empty_answer_abstains() -> None:
    silent = payload({"text": "The agreement does not say.", "quotes": [{"source": 1, "quote": explore.SENTINEL}]})
    answer = explore.sentinel("7::a", "Q?", document(), lambda prompt: silent)
    assert answer.abstained
    assert answer.statements == ()


def test_real_quotes_survive_next_to_a_sentinel_statement() -> None:
    mixed = payload(
        {"text": "Beta clause says return.", "quotes": [{"source": 1, "quote": "Beta clause says return."}]},
        {"text": "Nothing else.", "quotes": [{"source": 1, "quote": explore.SENTINEL.lower()}]},
    )
    answer = explore.sentinel("7::a", "Q?", document(), lambda prompt: mixed)
    assert not answer.abstained
    assert [s.text for s in answer.statements] == ["Beta clause says return."]
    assert answer.citations[0].match == "exact"


def test_extract_abstains_on_an_empty_list_without_a_second_call() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return '{"clauses": []}'

    answer = explore.extract_then_answer("7::a", "Q?", document(), complete)
    assert answer.abstained
    assert len(prompts) == 1


def test_extract_locates_second_call_quotes_in_the_whole_agreement() -> None:
    replies = iter(
        [
            '{"clauses": ["Beta clause says return."]}',
            payload({"text": "It must be returned.", "quotes": [{"source": 1, "quote": "Beta clause says return."}]}),
        ],
    )
    answer = explore.extract_then_answer("7::a", "Q?", document(), lambda prompt: next(replies))
    assert answer.citations[0].span == (14, 38)


def test_the_self_check_abstains_only_when_the_quotes_are_not_addressed() -> None:
    base = explore.sentinel(
        "7::a",
        "Q?",
        document(),
        lambda p: payload({"text": "S.", "quotes": [{"source": 1, "quote": "Gamma clause."}]}),
    )
    assert explore.self_check(base, "Q?", lambda p: '{"addressed": false}').abstained
    assert explore.self_check(base, "Q?", lambda p: '{"addressed": true}') is base


def test_standalone_rewrites_statements_and_keeps_citations() -> None:
    base = explore.sentinel(
        "7::a",
        "Q?",
        document(),
        lambda p: payload({"text": "Yes", "quotes": [{"source": 1, "quote": "Gamma clause."}]}),
    )
    rewritten = explore.standalone(base, "Q?", lambda p: "Yes, the agreement has a gamma clause.")
    assert rewritten.statements[0].text.startswith("Yes, the agreement")
    assert rewritten.statements[0].citations == base.statements[0].citations
    assert isinstance(rewritten, CitedAnswer)
    assert Statement("x").citations == ()


def test_minicheck_writes_the_question_into_the_claim_only_when_asked() -> None:
    plain = MiniCheckChecker()
    asked = MiniCheckChecker(question_in_claim=True)
    assert plain.claim_for("Yes", "Does it?") == "Yes"
    assert asked.claim_for("Yes", "Does it?") == QUESTION_CLAIM.format(question="Does it?", claim="Yes")
    assert asked.claim_for("Yes", "") == "Yes"
    assert asked.name.endswith("+question")


def test_the_committed_explore_results_name_their_model_and_cover_every_variant() -> None:
    explored = json.loads(RESULTS.read_text(encoding="utf-8"))["explore"]
    assert explored["produced_by"]["model"] == "Qwen3_6-35B-A3B"
    assert set(explored["abstention"]) == {"fulldoc", "fulldoc-sentinel", "fulldoc-extract", "fulldoc-check"}
    reference = explored["abstention"]["fulldoc"]["correctness"]["null_abstention"]
    extract = explored["abstention"]["fulldoc-extract"]["correctness"]["null_abstention"]
    assert extract - reference > 0.1


def test_the_live_check_names_both_models_and_stays_under_the_cap() -> None:
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    assert live["produced_by"]["model"] == "gpt-6-luna"
    assert live["produced_by"]["judging_model"] == "claude-haiku-5-5"
    assert live["cost"]["total_usd"] < 3.0
    assert live["arms"]["shuffled"]["llm"]["gold_precision"] < live["arms"]["inline"]["llm"]["gold_precision"]
