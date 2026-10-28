"""Tests for citation parsing, gold-span scoring, and ALCE-style judging."""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.evaluation.citations import (
    CITATION_TABLE_HEADER,
    Agreement,
    CitationReport,
    CitedAnswer,
    JudgedAnswer,
    MiniCheckChecker,
    Statement,
    SupportVerdict,
    agreement,
    citation_pairs,
    from_anthropic,
    from_grounded,
    generate_cited,
    judge_citations,
    locate_quote,
    overlaps,
    parse_inline,
    parse_structured,
    parse_support,
    score_against_gold,
    split_statements,
)
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline.generate import GroundedAnswer
from aieng.advanced_rag.provenance import Provenance


DOC = (
    "1. Definitions. Confidential Information means all non-public information. "
    "2. Return. The Recipient shall return or destroy all Confidential Information "
    "within thirty days of termination. 3. Term. This Agreement lasts two years."
)


def passage(start: int, end: int, index: int) -> Evidence:
    return Evidence(
        text=DOC[start:end],
        provenance=Provenance(
            doc_id="nda-1", chunk_id=f"nda-1::{index:04d}", start=start, end=end
        ),
    )


RETURN_AT = DOC.index("2. Return")
TERM_AT = DOC.index("3. Term")
EVIDENCE = [
    passage(0, RETURN_AT, 0),
    passage(RETURN_AT, TERM_AT, 1),
    passage(TERM_AT, len(DOC), 2),
]
RETURN_SENTENCE = "The Recipient shall return or destroy all Confidential Information within thirty days of termination."
GOLD = [(DOC.index(RETURN_SENTENCE), DOC.index(RETURN_SENTENCE) + len(RETURN_SENTENCE))]


class KeywordChecker:
    """Supports a claim when every keyword it names appears in the premise."""

    name = "keywords"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, premise: str, claim: str, *, question: str = ""
    ) -> SupportVerdict:
        self.calls.append((premise, claim))
        needed = [w for w in ("return", "thirty", "two years") if w in claim.lower()]
        found = [w for w in needed if w in premise.lower()]
        if needed and len(found) == len(needed):
            return SupportVerdict("full")
        return SupportVerdict("partial" if found else "none")


# --------------------------------------------------------------------- quotes


def test_an_exact_quote_is_found_at_its_offsets() -> None:
    text = EVIDENCE[1].text
    start, end, how = locate_quote("return or destroy", text)  # type: ignore[misc]
    assert text[start:end] == "return or destroy"
    assert how == "exact"


def test_curly_quotes_case_and_spacing_still_match_after_folding() -> None:
    text = "The “Recipient”   shall return it."
    found = locate_quote('the "recipient" shall RETURN it', text)
    assert found is not None
    assert found[2] == "normalised"
    assert text[found[0] : found[1]] == "The “Recipient”   shall return it"


def test_a_one_character_slip_matches_fuzzily() -> None:
    found = locate_quote(
        "shall retrun or destroy all Confidential Information", EVIDENCE[1].text
    )
    assert found is not None
    assert found[2] == "fuzzy"


def test_an_ellipsis_quote_spans_from_its_first_part_to_its_last() -> None:
    text = EVIDENCE[1].text
    found = locate_quote("shall return ... within thirty days", text)
    assert found is not None
    assert text[found[0] : found[1]].startswith("shall return")
    assert text[found[0] : found[1]].endswith("within thirty days")


def test_a_quote_the_source_does_not_contain_is_not_found() -> None:
    assert (
        locate_quote("The Recipient may keep one archival copy.", EVIDENCE[1].text)
        is None
    )


# --------------------------------------------------------------------- parsing


def test_a_marker_after_the_full_stop_belongs_to_the_sentence_before_it() -> None:
    assert split_statements("It must be returned. [2] It lasts two years [3].") == [
        "It must be returned. [2]",
        "It lasts two years [3].",
    ]


def test_inline_markers_cite_their_own_sentence() -> None:
    answer = parse_inline(
        "q",
        "Information must be returned within thirty days [2]. The term is two years [1, 3].",
        EVIDENCE,
    )
    assert [s.text for s in answer.statements] == [
        "Information must be returned within thirty days.",
        "The term is two years.",
    ]
    assert [c.source for c in answer.statements[1].citations] == [1, 3]
    assert answer.statements[0].citations[0].span == (RETURN_AT, TERM_AT)


def test_a_chunk_identifier_marker_resolves_to_its_passage() -> None:
    answer = parse_inline("q", "It lasts two years [nda-1::0002].", EVIDENCE)
    assert answer.statements[0].citations[0].source == 3


def test_an_out_of_range_number_and_an_invented_identifier_are_dropped() -> None:
    answer = parse_inline(
        "q", "It lasts two years [7]. It is returned [nda-9::0001].", EVIDENCE
    )
    assert answer.dropped == ("7", "nda-9::0001")
    assert answer.citations == ()


def test_a_bracketed_word_is_not_a_citation() -> None:
    answer = parse_inline("q", "The clause [sic] lasts two years [3].", EVIDENCE)
    assert answer.statements[0].text == "The clause [sic] lasts two years."
    assert answer.dropped == ()


def test_the_abstention_sentence_abstains() -> None:
    assert parse_inline("q", "Not stated in the provided sources.", EVIDENCE).abstained


def test_quotes_resolve_to_document_offsets() -> None:
    raw = json.dumps(
        {
            "statements": [
                {
                    "text": "Information is returned.",
                    "quotes": [{"source": 2, "quote": "return or destroy"}],
                },
            ],
            "not_stated": False,
        },
    )
    citation = parse_structured("q", raw, EVIDENCE).citations[0]
    assert DOC[citation.start : citation.end] == "return or destroy"  # type: ignore[misc]
    assert citation.text == "return or destroy"
    assert citation.match == "exact"


def test_an_unfound_quote_falls_back_to_its_passage_and_says_so() -> None:
    raw = json.dumps(
        {
            "statements": [
                {"text": "x", "quotes": [{"source": 2, "quote": "keep a copy forever"}]}
            ]
        }
    )
    citation = parse_structured("q", raw, EVIDENCE).citations[0]
    assert citation.match == "unmatched"
    assert citation.span == (RETURN_AT, TERM_AT)


def test_not_stated_abstains_and_prose_falls_back_to_inline_markers() -> None:
    assert parse_structured(
        "q", '{"statements": [], "not_stated": true}', EVIDENCE
    ).abstained
    fallback = parse_structured("q", "It lasts two years [3].", EVIDENCE)
    assert fallback.style == "quotes"
    assert fallback.citations[0].source == 3


def test_a_grounded_answer_becomes_one_statement_citing_everything() -> None:
    grounded = GroundedAnswer(
        query_id="q",
        text="Returned within thirty days.",
        citations=(EVIDENCE[1].provenance, EVIDENCE[2].provenance),
    )
    answer = from_grounded(grounded, EVIDENCE)
    assert len(answer.statements) == 1
    assert [c.source for c in answer.citations] == [2, 3]


def test_anthropic_char_locations_shift_by_the_passage_offset() -> None:
    blocks = [
        {"type": "text", "text": "The Recipient returns it "},
        {
            "type": "text",
            "text": "within thirty days",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "within thirty days",
                    "document_index": 1,
                    "start_char_index": EVIDENCE[1].text.index("within"),
                    "end_char_index": EVIDENCE[1].text.index("within")
                    + len("within thirty days"),
                },
            ],
        },
    ]
    answer = from_anthropic("q", blocks, EVIDENCE)
    citation = answer.citations[0]
    assert DOC[citation.start : citation.end] == "within thirty days"  # type: ignore[misc]
    assert len(answer.statements) == 2


def test_a_round_trip_through_a_dict_keeps_everything() -> None:
    answer = parse_inline("q", "It lasts two years [3].", EVIDENCE)
    assert CitedAnswer.from_dict(json.loads(json.dumps(answer.to_dict()))) == answer


# --------------------------------------------------------------------- generation


@pytest.mark.parametrize(
    ("style", "reply"),
    [
        ("answer", '{"answer": "Two years.", "sources": [3], "not_stated": false}'),
        ("inline", "The agreement lasts two years [3]."),
        (
            "quotes",
            '{"statements": [{"text": "Two years.", "quotes": [{"source": 3, "quote": "lasts two years"}]}]}',
        ),
    ],
)
def test_every_style_reads_its_own_format(style: str, reply: str) -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return reply

    answer = generate_cited("q", "How long?", EVIDENCE, complete, style=style)  # type: ignore[arg-type]
    assert answer.citations[0].source == 3
    assert "[3]" in prompts[0]


def test_no_evidence_abstains_without_a_call() -> None:
    def complete(prompt: str) -> str:
        raise AssertionError("no call expected")

    assert generate_cited("q", "How long?", [], complete).abstained


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown citation style"):
        generate_cited("q", "?", EVIDENCE, lambda p: "", style="footnotes")  # type: ignore[arg-type]


# --------------------------------------------------------------------- gold spans


def test_overlap_is_measured_against_the_shorter_range() -> None:
    assert overlaps((0, 1200), (100, 200))
    assert not overlaps((0, 1200), (1150, 1300))
    assert overlaps((100, 160), (90, 200))
    assert not overlaps((0, 10), (10, 20))


def test_gold_precision_and_recall_count_citations_and_spans() -> None:
    answer = parse_inline("q", "It is returned within thirty days [2][3].", EVIDENCE)
    score = score_against_gold(answer, GOLD, doc_id="nda-1", shown=EVIDENCE)
    assert score.hits == (True, False)
    assert score.precision == 0.5
    assert score.recall == 1.0
    assert score.shown_recall == 1.0


def test_a_citation_of_another_document_never_hits() -> None:
    answer = parse_inline("q", "It is returned [2].", EVIDENCE)
    assert score_against_gold(answer, GOLD, doc_id="nda-2").hits == (False,)


def test_gold_not_shown_to_the_model_lowers_only_plain_recall() -> None:
    answer = parse_inline("q", "It lasts two years [1].", EVIDENCE[:1])
    score = score_against_gold(answer, GOLD, doc_id="nda-1", shown=EVIDENCE[:1])
    assert score.recall == 0.0
    assert score.shown_recall is None


# --------------------------------------------------------------------- judging


def test_alce_marks_an_unneeded_citation_imprecise() -> None:
    answer = parse_inline(
        "q", "Information is returned within thirty days [2][3].", EVIDENCE
    )
    judged = judge_citations(answer, KeywordChecker())
    assert judged.recall == 1.0
    assert judged.statements[0].alone == ("full", "none")
    assert judged.statements[0].precise == (True, False)
    assert judged.precision == 0.5


def test_an_uncited_statement_has_no_recall() -> None:
    answer = parse_inline("q", "Information is returned within thirty days.", EVIDENCE)
    judged = judge_citations(answer, KeywordChecker())
    assert judged.recall == 0.0
    assert judged.precision is None


def test_two_citations_that_only_support_together_are_both_precise() -> None:
    answer = parse_inline("q", "It is returned and lasts two years [2][3].", EVIDENCE)
    judged = judge_citations(answer, KeywordChecker())
    assert judged.statements[0].alone == ("partial", "partial")
    assert judged.statements[0].precise == (True, True)


def test_a_single_citation_is_checked_once() -> None:
    checker = KeywordChecker()
    judge_citations(
        parse_inline("q", "It is returned within thirty days [2].", EVIDENCE), checker
    )
    assert len(checker.calls) == 1


def test_judged_answers_round_trip() -> None:
    judged = judge_citations(
        parse_inline("q", "It lasts two years [3].", EVIDENCE), KeywordChecker()
    )
    assert JudgedAnswer.from_dict(json.loads(json.dumps(judged.to_dict()))) == judged


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ('{"support": "full", "reason": "stated"}', "full"),
        ('```json\n{"support": "Partial"}\n```', "partial"),
        ("The source gives no support.", "none"),
        ('{"support": "yes"}', "none"),
    ],
)
def test_support_replies_parse_to_three_levels(raw: str, label: str) -> None:
    assert parse_support(raw).label == label


def test_minicheck_thresholds_its_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = MiniCheckChecker(threshold=0.6)
    monkeypatch.setattr(checker, "probabilities", lambda pairs: [0.55])
    assert checker("p", "c").label == "none"
    monkeypatch.setattr(checker, "probabilities", lambda pairs: [0.75])
    assert checker("p", "c") == SupportVerdict("full", score=0.75)


def test_minicheck_reads_the_question_in_the_claim_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    def record(checker: MiniCheckChecker) -> None:
        monkeypatch.setattr(
            checker, "probabilities", lambda pairs: seen.extend(pairs) or [0.9]
        )

    plain, asked = MiniCheckChecker(), MiniCheckChecker(question_in_claim=True)
    record(plain)
    record(asked)
    plain("premise", "Yes", question="Does it?")
    asked("premise", "Yes", question="Does it?")
    assert seen == [("premise", "Yes"), ("premise", "Question: Does it?\nAnswer: Yes")]


# --------------------------------------------------------------------- agreement


def test_kappa_matches_a_worked_example() -> None:
    table = agreement(
        [(True, True)] * 20
        + [(False, True)] * 5
        + [(True, False)] * 10
        + [(False, False)] * 15
    )
    assert table == Agreement(both=20, judge_only=5, gold_only=10, neither=15)
    assert table.accuracy == 0.7
    assert table.kappa == pytest.approx(0.4)


def test_citation_pairs_line_up_gold_hits_with_single_citation_verdicts() -> None:
    answer = parse_inline(
        "q", "Information is returned within thirty days [2][3].", EVIDENCE
    )
    gold = score_against_gold(answer, GOLD, doc_id="nda-1")
    judged = judge_citations(answer, KeywordChecker())
    assert citation_pairs([gold], [judged]) == [(True, True), (False, False)]


def test_a_report_summarises_one_arm() -> None:
    answers = [
        parse_inline(
            "a", "Information is returned within thirty days [2][3].", EVIDENCE
        ),
        parse_inline("b", "Not stated in the provided sources.", EVIDENCE),
        CitedAnswer("c", (Statement("x"),), "inline"),
    ]
    gold = [score_against_gold(answers[0], GOLD, doc_id="nda-1")]
    judged = [judge_citations(a, KeywordChecker()) for a in answers if not a.abstained]
    report = CitationReport("inline", answers, gold, judged)
    row = report.to_dict()
    assert row["answered"] == 2
    assert row["gold_precision"] == 0.5
    assert row["judged_recall"] == 0.5
    assert row["uncited_statement_rate"] == 0.5
    assert report.table_row().count("|") == CITATION_TABLE_HEADER.splitlines()[0].count(
        "|"
    )


def test_an_uncited_refusal_in_prose_abstains_and_a_cited_one_does_not() -> None:
    assert parse_inline(
        "q", "The agreement does not mention reverse engineering.", EVIDENCE
    ).abstained
    cited = parse_inline("q", "The agreement does not mention a fee [1].", EVIDENCE)
    assert not cited.abstained


def test_the_quote_format_keeps_its_direct_answer() -> None:
    raw = json.dumps(
        {
            "answer": "Yes.",
            "statements": [{"text": "It lasts two years.", "sources": [3]}],
        }
    )
    answer = parse_structured("q", raw, EVIDENCE)
    assert answer.direct == "Yes."
    assert CitedAnswer.from_dict(answer.to_dict()) == answer


def test_a_stock_not_stated_sentence_beside_an_answer_is_not_a_statement() -> None:
    answer = parse_inline(
        "q", "It lasts two years [3]. Not stated in the provided sources.", EVIDENCE
    )
    assert [s.text for s in answer.statements] == ["It lasts two years."]
