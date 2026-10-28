"""Tables and the committed results for the citation comparison.

Every function here reads a :class:`~cites.pipeline.Pipeline` and calls no
model, so the notebook prints the same numbers from the cache that
``build_reference.py report`` writes to ``outputs/results.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.citations import (
    CITATION_TABLE_HEADER,
    Agreement,
    CitationReport,
    CitedAnswer,
    GoldScore,
    JudgedAnswer,
    agreement,
    citation_pairs,
)
from aieng.advanced_rag.pipeline.generate import looks_like_abstention

from . import data, explore
from .pipeline import ARMS, EMBEDDER, GENERATED, MINICHECK, K, NotCachedError, Pipeline, Which


CHECKERS = ("llm", "minicheck")


def available(pipe: Pipeline, which: Which = "all") -> tuple[str, ...]:
    """Return the arms that can be scored now.

    ``posthoc`` needs MiniCheck's citations, which a live run builds only with
    the local checker on, so it is left out, with a note, when none are cached.
    """
    arms = []
    for arm in ARMS:
        try:
            pipe.answers(arm, which)
        except NotCachedError as error:
            print(f"{arm}: skipped ({error})")
            continue
        arms.append(arm)
    return tuple(arms)


def judged_by(pipe: Pipeline, arm: str, which: Which, checker: str) -> list[JudgedAnswer]:
    """Return one checker's verdicts on one arm, empty without cached MiniCheck."""
    if checker == "llm":
        return pipe.judged(arm, which)
    try:
        return pipe.minichecked(arm, which)
    except NotCachedError:
        return []


def citation_report(pipe: Pipeline, arm: str, which: Which, checker: str = "llm") -> CitationReport:
    """Return one arm's citation report with one checker's verdicts."""
    return CitationReport(arm, pipe.answers(arm, which), pipe.gold(arm, which), judged_by(pipe, arm, which, checker))


def arm_table(pipe: Pipeline, which: Which, checker: str = "llm", arms: Sequence[str] | None = None) -> str:
    """Render the per-arm Markdown table for one checker."""
    arms = available(pipe, which) if arms is None else arms
    rows = [CITATION_TABLE_HEADER]
    rows.extend(citation_report(pipe, arm, which, checker).table_row() for arm in arms)
    return "\n".join(rows)


def outcome(item: data.Item, answer: CitedAnswer) -> str:
    """Return ``yes``, ``no``, ``not stated``, or ``unclear`` for one answer."""
    if answer.abstained:
        return "not stated"
    verdict = data.verdict_of(answer.direct)
    if verdict is not None:
        return verdict
    return "not stated" if looks_like_abstention(answer.direct) else "unclear"


def correctness(pipe: Pipeline, arm: str, which: Which = "all") -> dict[str, Any]:
    """Score verdicts against ContractNLI's labels, without a judge.

    A labelled question is correct when the answer begins with the expected
    Yes or No. A null is correct when the answer abstains or says the
    agreement does not address it.
    """
    labelled = nulls = right = null_right = unclear = cited_null = 0
    for answer in pipe.answers(arm, which):
        item = pipe.item(answer.query_id)
        said = outcome(item, answer)
        unclear += said == "unclear"
        if item.is_null:
            nulls += 1
            null_right += said == "not stated"
            cited_null += bool(answer.citations) and said != "not stated"
        else:
            labelled += 1
            right += said == item.expected
    return {
        "labelled": labelled,
        "verdict_accuracy": _ratio(right, labelled),
        "nulls": nulls,
        "null_abstention": _ratio(null_right, nulls),
        "cited_answer_on_null": cited_null,
        "unclear": unclear,
    }


def agreement_for(pipe: Pipeline, arm: str, which: Which, checker: str) -> Agreement:
    """Pair each citation's gold hit with the checker's single-citation verdict."""
    return agreement(citation_pairs(pipe.gold(arm, which), judged_by(pipe, arm, which, checker)))


def statement_agreement(pipe: Pipeline, arm: str, which: Which, checker: str) -> Agreement:
    """Pair, per statement, "cites a gold span" with "judged fully supported".

    This is the reading a participant would act on: does a statement the judge
    calls supported actually rest on the annotated evidence? Only statements
    with at least one citation are counted.
    """
    golds = {g.query_id: g for g in pipe.gold(arm, which)}
    answers = {a.query_id: a for a in pipe.answers(arm, which)}
    pairs: list[tuple[bool, bool]] = []
    for judged in judged_by(pipe, arm, which, checker):
        gold = golds.get(judged.query_id)
        if gold is None:
            continue
        hits = iter(gold.hits)
        for statement, verdict in zip(answers[judged.query_id].statements, judged.statements):
            cited = [next(hits) for _ in statement.citations]
            # An uncited statement fails both scorers by definition, so it
            # would add agreement that says nothing about the judge.
            if cited:
                pairs.append((any(cited), verdict.recall))
    return agreement(pairs)


def quote_matches(pipe: Pipeline, which: Which = "all") -> dict[str, int]:
    """Count how the quote arm's quotes were found in their sources."""
    counts: Counter[str] = Counter()
    for answer in pipe.answers("quotes", which):
        counts.update(c.match for c in answer.citations)
    return dict(sorted(counts.items()))


def citation_widths(pipe: Pipeline, arm: str, which: Which = "all") -> float | None:
    """Return the median width in characters of what an arm's citations point at."""
    widths = sorted(len(c.text) for a in pipe.answers(arm, which) for c in a.citations)
    return float(widths[len(widths) // 2]) if widths else None


def disagreements(
    pipe: Pipeline, arm: str, checker: str = "llm", which: Which = "all", limit: int = 5
) -> list[dict[str, Any]]:
    """Return citations the checker accepted that overlap no gold span.

    Each row carries the statement, the cited text, and the gold evidence, so
    you can read why the two scorers differ.
    """
    golds = {g.query_id: g for g in pipe.gold(arm, which)}
    answers = {a.query_id: a for a in pipe.answers(arm, which)}
    rows: list[dict[str, Any]] = []
    for judged in judged_by(pipe, arm, which, checker):
        gold = golds.get(judged.query_id)
        if gold is None:
            continue
        item = pipe.item(judged.query_id)
        text = pipe.texts()[item.doc_id]
        citations = answers[judged.query_id].citations
        statements = [s.text for s in answers[judged.query_id].statements for _ in s.citations]
        for citation, hit, label, statement in zip(citations, gold.hits, judged.alone, statements):
            if label == "full" and not hit:
                rows.append(
                    {
                        "query_id": item.query_id,
                        "label": item.label,
                        "statement": statement,
                        "cited": citation.text,
                        "gold": [text[a:b] for a, b in item.gold],
                    },
                )
        if len(rows) >= limit:
            break
    return rows[:limit]


def usage(pipe: Pipeline, which: Which = "all") -> dict[str, Any]:
    """Return calls and tokens of each model step, from the stored provenance."""
    rows: dict[str, Any] = {}
    for arm in available(pipe, which):
        if arm in GENERATED:
            provenance = pipe.answer_result(arm, which).provenance
            rows[f"answers/{arm}"] = provenance.to_dict()["usage"] if provenance else None
        provenance = pipe.judge_result(arm, which).provenance
        rows[f"judge/{arm}"] = provenance.to_dict()["usage"] if provenance else None
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summary(pipe: Pipeline, which: Which = "all") -> dict[str, Any]:
    """Return every number the notebook and the README quote."""
    arms: dict[str, Any] = {}
    for arm in available(pipe, which):
        arms[arm] = {
            "llm": citation_report(pipe, arm, which, "llm").to_dict(),
            "minicheck": citation_report(pipe, arm, which, "minicheck").to_dict(),
            "correctness": correctness(pipe, arm, which),
            "median_cited_chars": citation_widths(pipe, arm, which),
            "agreement": {
                checker: {
                    "citation": agreement_for(pipe, arm, which, checker).to_dict(),
                    "statement": statement_agreement(pipe, arm, which, checker).to_dict(),
                }
                for checker in CHECKERS
            },
        }
    items = pipe.items(which)
    return {
        "which": which,
        "questions": len(items),
        "labelled": sum(not i.is_null for i in items),
        "contradictions": sum(i.label == "Contradiction" for i in items),
        "nulls": sum(i.is_null for i in items),
        "documents": len({i.doc_id for i in items}),
        "arms": arms,
        "quote_matches": quote_matches(pipe, which),
    }


def per_question(pipe: Pipeline, which: Which = "all") -> list[dict[str, Any]]:
    """Return per-question identifiers and scores, with no contract text."""
    names = available(pipe, which)
    golds: dict[str, dict[str, GoldScore]] = {arm: {g.query_id: g for g in pipe.gold(arm, which)} for arm in names}
    judged = {arm: {j.query_id: j for j in pipe.judged(arm, which)} for arm in names}
    rows = []
    for item in pipe.items(which):
        row: dict[str, Any] = {"query_id": item.query_id, "label": item.label, "gold_spans": len(item.gold)}
        for arm in names:
            answer = next(a for a in pipe.answers(arm, which) if a.query_id == item.query_id)
            gold = golds[arm].get(item.query_id)
            verdicts = judged[arm].get(item.query_id)
            row[arm] = {
                "outcome": outcome(item, answer),
                "citations": len(answer.citations),
                "gold_precision": None if gold is None else gold.precision,
                "gold_recall": None if gold is None else gold.recall,
                "judged_recall": None if verdicts is None else verdicts.recall,
                "judged_precision": None if verdicts is None else verdicts.precision,
            }
        rows.append(row)
    return rows


#: Arms MiniCheck re-reads with the question in the claim.
WITH_QUESTION = ("answer", "inline", "quotes", "fulldoc")


def _judged_row(pipe: Pipeline, arm: str, judged: list[JudgedAnswer]) -> dict[str, Any]:
    """Return gold and judged scores plus gold agreement for one set of verdicts."""
    row = CitationReport(arm, pipe.answers(arm, "all"), pipe.gold(arm, "all"), judged).to_dict()
    pairs = citation_pairs(pipe.gold(arm, "all"), judged)
    return {"scores": row, "agreement": agreement(pairs).to_dict()}


def explore_summary(pipe: Pipeline) -> dict[str, Any]:
    """Return the numbers of the abstention variants and the question-aware MiniCheck.

    ``abstention`` holds, per arm, how many of the nulls the arm declined, the
    verdict accuracy on labelled questions, and the LLM judge's scores. The
    ``fulldoc`` row repeats the reference run for comparison. ``minicheck``
    holds the judged scores and gold agreement of MiniCheck with and without
    the question in the claim, and with the shipped generator's statements
    rewritten to stand alone. ``usage`` holds calls and tokens of the model
    steps these variants ran.
    """
    abstention: dict[str, Any] = {}
    for arm in ("fulldoc", *(a for a in explore.VARIANTS if a.startswith("fulldoc"))):
        answers = pipe.answers(arm, "all")
        nulls = [a for a in answers if pipe.item(a.query_id).is_null]
        labelled = [a for a in answers if not pipe.item(a.query_id).is_null]
        said_none = [a.text for a in nulls if not a.abstained]
        abstention[arm] = {
            "correctness": correctness(pipe, arm, "all"),
            "labelled_declined": sum(a.abstained for a in labelled),
            "null_answered_text_says_silent": explore.lexical_negation(said_none),
            "llm": citation_report(pipe, arm, "all").to_dict(),
        }
    checks: dict[str, Any] = {}
    for arm in WITH_QUESTION:
        plain = pipe.minichecked(arm, "all")
        asked = pipe.minichecked(arm, "all", with_question=True)
        checks[arm] = {
            "claim_only": _judged_row(pipe, arm, plain),
            "question_and_claim": _judged_row(pipe, arm, asked),
            "llm_judge": _judged_row(pipe, arm, pipe.judged(arm, "all")),
        }
    standalone = pipe.minichecked("answer-standalone", "all")
    checks["answer-standalone"] = {
        "claim_only": _judged_row(pipe, "answer-standalone", standalone),
        "llm_judge_on_original_answer_arm": checks["answer"]["llm_judge"],
    }
    usage: dict[str, Any] = {}
    for arm in explore.VARIANTS:
        provenance = pipe.answer_result(arm, "all").provenance
        usage[f"answers/{arm}"] = provenance.to_dict()["usage"] if provenance else None
    for arm in (a for a in explore.VARIANTS if a.startswith("fulldoc")):
        provenance = pipe.judge_result(arm, "all").provenance
        usage[f"judge/{arm}"] = provenance.to_dict()["usage"] if provenance else None
    generation = pipe.answer_result("fulldoc-sentinel", "all").provenance
    judging = pipe.judge_result("fulldoc-sentinel", "all").provenance
    produced_by = {
        "model": generation.model if generation else pipe.models.generation.model,
        "judging_model": judging.model if judging else pipe.models.judging.model,
        "local_checker": MINICHECK,
        "note": "one run, explore mode, the same 257 questions as the reference run",
    }
    return {"produced_by": produced_by, "abstention": abstention, "minicheck": checks, "usage": usage}


def write_results(pipe: Pipeline, path: Path) -> Path:
    """Write ``results.json``: the summary on both question sets, with provenance."""
    generation = pipe.answer_result("inline", "all").provenance
    judging = pipe.judge_result("inline", "all").provenance
    payload = {
        "produced_by": {
            "model": generation.model if generation else pipe.models.generation.model,
            "generation": generation.to_dict() if generation else None,
            "judging": judging.to_dict() if judging else None,
            "local_checker": MINICHECK,
            "embeddings": EMBEDDER,
            "note": "one run, explore mode, ContractNLI test split, first 24 agreements",
        },
        "config": {**pipe.dataset_config(), "evidence_budget": K},
        "all": summary(pipe, "all"),
        "subset": summary(pipe, "subset"),
        "usage": usage(pipe, "all"),
        "questions": per_question(pipe, "all"),
        "explore": explore_summary(pipe),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CHECKERS",
    "WITH_QUESTION",
    "explore_summary",
    "available",
    "agreement_for",
    "arm_table",
    "citation_report",
    "citation_widths",
    "correctness",
    "disagreements",
    "judged_by",
    "outcome",
    "per_question",
    "quote_matches",
    "statement_agreement",
    "summary",
    "usage",
    "write_results",
]
