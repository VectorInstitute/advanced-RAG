"""Evaluate saved ContractNLI answers independently of replay storage and clients.

Correctness uses gold references; grounding uses only the evidence shown to
generation. Canonical pipeline abstentions determine report denominators.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.answers import AnswerJudgement, AnswerReport, judge_answer
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline.generate import ABSTENTION_TEXT
from aieng.advanced_rag.retrieval.corpus import Query


MAX_CHARS = 1200
VALIDATION_ATTEMPTS = 3


@dataclass(frozen=True)
class JudgeVerdicts:
    """Keep the independent correctness and grounding passes together."""

    correctness: AnswerJudgement
    grounding: AnswerJudgement


CORRECTNESS_PROMPT = """Grade the semantic correctness of one contract answer.
Use only the question, reference, and candidate answer below. Retrieval and
citations do not determine correctness. The reference states the expected truth
direction; supporting passages justify it, rather than adding required wording.
A candidate may answer with an equivalent contract clause or natural language.
Do not require the words Entailment, Contradiction, supported, or contradicted.
Do not require reproducing all supporting passages or their ancillary details.
A bare quotation counts if it clearly conveys the reference's substantive answer.
An answer that reverses the reference's truth direction is incorrect.
An abstention on these answerable questions is incorrect.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{answer}

Return JSON only with these keys:
"correct": boolean, whether the substantive answer agrees with the reference;
"grounded": false (not evaluated in this pass);
"abstained": boolean, whether the candidate declines to answer;
"reason": one sentence, at most 30 words, explaining correctness.
"""


GROUNDING_PROMPT = """Grade whether one contract answer is supported by its sources.
Use only the retrieved evidence below. No gold reference is supplied. Being
supported by retrieved evidence does not establish correctness against the full
contract. Judge factual claims and truth-direction claims in the answer.
An abstention makes no affirmative factual claims and is vacuously grounded.

Question:
{question}

Evidence the system was given:
{evidence}

Candidate answer:
{answer}

Return JSON only with these keys:
"correct": false (not evaluated in this pass);
"grounded": boolean, whether every factual claim is supported by these sources;
"abstained": boolean, whether the candidate declines to answer;
"reason": one sentence, at most 30 words, explaining grounding.
"""


def references(path: Path, documents: int) -> dict[str, str]:
    """Restore label polarity alongside the annotated gold spans."""
    payload = json.loads(path.read_text())
    result = {}
    for document in payload["documents"][:documents]:
        for key, annotation in document["annotation_sets"][0]["annotations"].items():
            label = annotation["choice"]
            if label == "NotMentioned":
                continue
            if label not in ("Entailment", "Contradiction"):
                raise ValueError(f"unknown ContractNLI label: {label}")
            hypothesis = payload["labels"][key]["hypothesis"]
            polarity = "supported" if label == "Entailment" else "contradicted"
            spans = []
            for index in annotation["spans"]:
                start, end = document["spans"][index]
                spans.append(document["text"][start:end])
            result[f"{document['id']}::{key}"] = (
                f"Expected substantive answer: The hypothesis is {polarity}: {hypothesis}\n"
                "Supporting passages (justification, not required answer wording):\n" + "\n".join(spans)
            )
    return result


def checked_complete(complete: Callable[[str], str]) -> Callable[[str], str]:
    """Reject malformed verdicts before the shared parser's permissive casts."""

    def checked(prompt: str) -> str:
        error = ""
        for attempt in range(VALIDATION_ATTEMPTS):
            retry = (
                ""
                if attempt == 0
                else (
                    f"\nOutput validation retry {attempt}: return valid JSON only. "
                    "Use literal true or false for all three boolean fields, and a quoted reason. "
                    "No comments, missing values, or schema placeholders. "
                    'Valid syntax example (values are illustrative): {"correct": false, '
                    '"grounded": false, "abstained": false, "reason": "A quoted sentence."}. '
                    "Reassess the original inputs and choose the actual verdicts."
                )
            )
            raw = complete(prompt + retry)
            start, end = raw.find("{"), raw.rfind("}")
            try:
                if start < 0 or end <= start:
                    raise ValueError("judge returned no JSON object")
                payload = json.loads(raw[start : end + 1])
                if (
                    not isinstance(payload, dict)
                    or any(type(payload.get(key)) is not bool for key in ("correct", "grounded", "abstained"))
                    or not isinstance(payload.get("reason"), str)
                ):
                    raise ValueError("judge verdict must contain three booleans and a reason")
                return json.dumps(payload)
            except ValueError as invalid:
                error = str(invalid)
        raise ValueError(f"judge failed output validation after three attempts: {error}")

    return checked


def validate_outcomes(
    rows: list[dict[str, Any]],
    queries: Sequence[Query],
    table: Mapping[str, list[Evidence]],
) -> dict[str, dict[str, Any]]:
    """Require exactly the saved slice and evidence for every answerable row."""
    indexed = {row["query_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {q.query_id for q in queries}:
        raise ValueError("outcome IDs do not match the corpus slice exactly")
    for query in queries:
        row = indexed[query.query_id]
        if row["is_null"] != (not bool(query.answers)) or row["doc_id"] != query.doc_id:
            raise ValueError(f"outcome corpus mismatch: {query.query_id}")
        if bool(query.answers) and query.query_id not in table:
            raise ValueError(f"missing evidence: {query.query_id}")
        if not isinstance(row["answer"], str) or type(row["abstained"]) is not bool:
            raise ValueError(f"invalid saved answer: {query.query_id}")
        if row["abstained"] != (row["answer"] == ABSTENTION_TEXT):
            raise ValueError(f"saved abstention is not the canonical pipeline response: {query.query_id}")
    return indexed


def score_answers(  # noqa: PLR0917 - explicit fixed inputs for an auditable replay
    arm: str,
    queries: Sequence[Query],
    outcomes: Mapping[str, dict[str, Any]],
    table: Mapping[str, list[Evidence]],
    gold: Mapping[str, str],
    complete: Callable[[str], str],
    *,
    budget: int,
    workers: int,
) -> dict[str, Any]:
    """Judge all answerable questions, retaining abstentions in the denominator."""
    answerable = [query for query in queries if bool(query.answers)]
    checked = checked_complete(complete)

    def score(query: Query) -> JudgeVerdicts:
        return judge_query(
            query, outcomes[query.query_id]["answer"], gold[query.query_id], table[query.query_id][:budget], checked
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(score, answerable))
    return build_answer_report(arm, budget, outcomes, verdicts)


def judge_query(
    query: Query,
    answer: str,
    reference: str,
    evidence: Sequence[Evidence],
    complete: Callable[[str], str],
) -> JudgeVerdicts:
    """Evaluate correctness without retrieved text, and grounding without gold."""
    correctness = judge_answer(
        query.query_id,
        query.question,
        answer,
        reference,
        [],
        complete,
        max_chars=MAX_CHARS,
        template=CORRECTNESS_PROMPT,
    )
    grounding = judge_answer(
        query.query_id,
        query.question,
        answer,
        "",
        evidence,
        complete,
        max_chars=MAX_CHARS,
        template=GROUNDING_PROMPT,
    )
    return JudgeVerdicts(correctness, grounding)


def normalize_verdict(verdicts: JudgeVerdicts, outcome: Mapping[str, Any]) -> AnswerJudgement:
    """Apply the canonical abstention policy without changing either raw pass."""
    correctness, grounding = verdicts.correctness, verdicts.grounding
    abstained = outcome["abstained"]
    if abstained and outcome["answer"] != ABSTENTION_TEXT:
        raise ValueError("cannot normalize a noncanonical abstention")
    # A prose refusal the pipeline did not flag can still be judged correct
    # against a reference, so the judge's own abstention verdict also rules it out.
    return AnswerJudgement(
        correctness.query_id,
        correctness.correct and not abstained and not correctness.abstained,
        grounding.grounded or abstained,
        abstained,
        correctness.reason,
    )


def build_answer_report(
    arm: str,
    budget: int,
    outcomes: Mapping[str, dict[str, Any]],
    verdicts: Sequence[JudgeVerdicts],
) -> dict[str, Any]:
    """Aggregate normalized rates and retain the raw diagnostic verdicts."""
    judgements = [normalize_verdict(v, outcomes[v.correctness.query_id]) for v in verdicts]
    attempted = sum(not j.abstained for j in judgements)
    report: dict[str, Any] = AnswerReport(arm, budget, attempted, judgements).to_dict()
    report["pipeline_abstentions"] = len(judgements) - attempted
    report["judge_pipeline_abstention_disagreements"] = sum(
        v.correctness.abstained != outcomes[v.correctness.query_id]["abstained"] for v in verdicts
    )
    report["correct_count"] = sum(j.correct for j in judgements)
    report["correct_among_pipeline_attempts"] = sum(
        j.correct and not outcomes[j.query_id]["abstained"] for j in judgements
    )
    report["judge_pass_abstention_disagreements"] = sum(
        v.correctness.abstained != v.grounding.abstained for v in verdicts
    )
    for row, pair in zip(report["judgements"], verdicts, strict=True):
        correctness, grounding = pair.correctness, pair.grounding
        row["grounding_reason"] = grounding.reason
        row["raw_correct"] = correctness.correct
        row["raw_grounded"] = grounding.grounded
        row["raw_correctness_abstained"] = correctness.abstained
        row["raw_grounding_abstained"] = grounding.abstained
    report["abstention_rule"] = (
        "canonical pipeline abstentions are incorrect on answerable queries and vacuously grounded; raw judge verdicts are retained"
    )
    return report


def correctness_consistency(
    reports: Sequence[dict[str, Any]],
    outcomes: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, int]:
    """Require identical answers to a query to have identical correctness."""
    dense = {j["query_id"]: j for j in reports[0]["judgements"]}
    graph = {j["query_id"]: j for j in reports[1]["judgements"]}
    identical = 0
    for query_id, verdict in dense.items():
        if outcomes["dense"][query_id]["answer"] == outcomes["lightrag-mix"][query_id]["answer"]:
            identical += 1
            if verdict["correct"] != graph[query_id]["correct"]:
                raise ValueError(f"inconsistent correctness for identical answers: {query_id}")
    return {"identical_answer_pairs": identical, "correctness_disagreements": 0}
