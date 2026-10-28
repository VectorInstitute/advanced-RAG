"""Scoring for both halves of the harness: retrieval and answers.

``metrics`` scores ranked chunk identifiers against judgments derived from a
dataset, so it needs no model and no corpus. ``answers`` scores generated text,
which cannot be done that way, so a model judges it against a reference.

The two table headers are named apart because a report usually prints both:
``RETRIEVAL_TABLE_HEADER`` for retrievers and ``TABLE_HEADER`` for answers.
"""

from .answers import (
    COVERAGE_TABLE_HEADER,
    TABLE_HEADER,
    AnswerCase,
    AnswerJudgement,
    AnswerReport,
    CoverageReport,
    answer_with_budget,
    judge_answer,
    judge_answers,
    judge_coverage,
    judge_prompt,
    parse_coverage_verdicts,
    parse_judgement,
)
from .metrics import TABLE_HEADER as RETRIEVAL_TABLE_HEADER
from .metrics import (
    NullReport,
    RetrievalReport,
    budget_k,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_failure,
    retrieved_chars,
    score_null_detection,
    score_retrieval,
)


ANSWER_TABLE_HEADER = TABLE_HEADER

__all__ = [
    "ANSWER_TABLE_HEADER",
    "COVERAGE_TABLE_HEADER",
    "RETRIEVAL_TABLE_HEADER",
    "TABLE_HEADER",
    "AnswerCase",
    "AnswerJudgement",
    "AnswerReport",
    "CoverageReport",
    "NullReport",
    "RetrievalReport",
    "answer_with_budget",
    "budget_k",
    "hit_at_k",
    "judge_answer",
    "judge_answers",
    "judge_coverage",
    "judge_prompt",
    "ndcg_at_k",
    "parse_coverage_verdicts",
    "parse_judgement",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_failure",
    "retrieved_chars",
    "score_null_detection",
    "score_retrieval",
]
