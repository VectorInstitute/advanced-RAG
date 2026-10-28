"""Write one question about one located statement.

The statement's offsets are already fixed, so the question inherits a gold span
that does not depend on how the question is worded. What varies per question is
who asks, what kind of lookup it is, and its surface form, drawn from the team's
:class:`~aieng.advanced_rag.evaluation.synthetic.config.GenerationConfig`, plus
the team's few-shot examples of real questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Category, GenerationConfig
from .llm import Model, parse_json
from .statements import Statement


QUESTION_SYSTEM = (
    "You write realistic user questions for a search evaluation. Reply with one "
    "JSON object and nothing else."
)

QUESTION_PROMPT = """The documents are {domain}.

Write ONE question that the statement below answers.

Who is asking: {user}
Kind of question: {question_kind}
Form: {phrasing}
{scope_note}
{few_shot}
Rules:
- The statement must fully answer the question. The answer must not need
  anything outside the statement.
- Ask what this user would ask. Do not mention "the statement", "the passage",
  "the text", or "the document" unless a user would naturally say it.
- Do not copy a long run of words from the statement into the question.
- If no realistic question of this kind can be answered by the statement,
  set "reject" to true.

Statement:
<<<
{statement}
>>>

Surrounding text, for context only:
<<<
{context}
>>>

Return JSON: {{"question": "...", "answer": "a short answer, taken from the statement", "reject": false, "reason": "one sentence"}}"""

SCOPE_NOTES = {
    "document": (
        "The user asks this about one document they already have open, so the "
        "question does not need to say which document it is about."
    ),
    "corpus": (
        "The user asks this of the whole collection, so the question has to "
        "carry the detail that picks out where the answer is."
    ),
}


def few_shot_block(examples: tuple[str, ...], limit: int = 8) -> str:
    """Render the team's real questions as style examples, or nothing."""
    if not examples:
        return ""
    lines = "\n".join(f"- {example}" for example in examples[:limit])
    return (
        "\nReal questions from these users, as examples of style and length. "
        "Do not copy their content:\n" + lines + "\n"
    )


@dataclass(frozen=True)
class Draft:
    """A written question before any filter has seen it.

    Parameters
    ----------
    statement : Statement
        What it was written from.
    question : str
        The question text. Empty when the generator rejected.
    answer : str
        The generator's reference answer.
    categories : dict
        The sampled user, question category, and phrasing names.
    rejected : bool
        Whether the generator declined.
    reason : str
        The generator's one-line reason.
    """

    statement: Statement
    question: str
    answer: str
    categories: dict[str, str]
    rejected: bool = False
    reason: str = ""


def write_question(
    model: Model,
    statement: Statement,
    config: GenerationConfig,
    *,
    key: str,
    context_chars: int = 1500,
) -> Draft:
    """Ask the generator for one question about one statement.

    Parameters
    ----------
    model : Model
        The generator.
    statement : Statement
        The located statement.
    config : GenerationConfig
        Categories, few-shot examples, domain, and scope.
    key : str
        Seeds the category draw, so it is stable under concurrency.
    context_chars : int, optional
        Passage text shown around the statement.

    Returns
    -------
    Draft
        The question, or a rejection.
    """
    drawn: dict[str, Category] = config.sample(key)
    payload: dict[str, Any] = parse_json(
        model(
            QUESTION_SYSTEM,
            QUESTION_PROMPT.format(
                domain=config.domain,
                user=drawn["user"].description,
                question_kind=drawn["question"].description,
                phrasing=drawn["phrasing"].description,
                scope_note=SCOPE_NOTES[config.scope],
                few_shot=few_shot_block(config.few_shot),
                statement=statement.quote,
                context=statement.passage.text[:context_chars],
            ),
        )
    )
    names = {role: category.name for role, category in drawn.items()}
    question = str(payload.get("question") or "").strip()
    rejected = bool(payload.get("reject")) or not question
    return Draft(
        statement=statement,
        question="" if rejected else question,
        answer=str(payload.get("answer") or "").strip(),
        categories=names,
        rejected=rejected,
        reason=str(payload.get("reason") or "")[:300],
    )


__all__ = [
    "QUESTION_PROMPT",
    "SCOPE_NOTES",
    "Draft",
    "few_shot_block",
    "write_question",
]
