"""Statement-first extraction: the model quotes, the code finds the quote.

Know Your RAG (de Lima et al., COLING 2025 Industry) generates a question per
statement condensed from the context. Here the statement is a verbatim quote,
and :func:`locate_quote` finds it in the document text and records its
character offsets. A quote the model paraphrased or invented is not found and
is dropped, at no cost. The offsets are what make the gold label survive any
chunker: they are mapped onto chunks with the harness's overlap rule at scoring
time, not fixed to one chunking when the question is written.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import Model, parse_json
from .sampling import Passage


# Characters a model routinely changes when it copies text: curly and straight
# quotes, dashes, and non-breaking spaces.
_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def _folded_with_map(text: str) -> tuple[str, list[int]]:
    """Return text folded to lowercase single-spaced form, and an offset map.

    ``mapping[i]`` is the offset in ``text`` of the folded string's character
    ``i``, so a match in the folded string maps back to original offsets.
    """
    folded: list[str] = []
    mapping: list[int] = []
    previous_space = True
    for index, char in enumerate(text.translate(_FOLD)):
        if char.isspace():
            if previous_space:
                continue
            folded.append(" ")
            mapping.append(index)
            previous_space = True
        else:
            folded.append(char.lower())
            mapping.append(index)
            previous_space = False
    return "".join(folded), mapping


def locate_quote(
    text: str,
    quote: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[int, int] | None:
    """Find a quote in a document and return its character offsets.

    An exact match is tried first. Failing that, both strings are compared
    after folding case, collapsing whitespace, and normalising quote marks and
    dashes, and the match is mapped back to the original offsets.

    Parameters
    ----------
    text : str
        The document text.
    quote : str
        What the model said it copied.
    start : int, optional
        Search from this offset, usually the passage start.
    end : int or None, optional
        Search up to this offset, usually the passage end. ``None`` searches to
        the end of the document.

    Returns
    -------
    tuple of (int, int) or None
        Half-open offsets into ``text``, trimmed of surrounding whitespace, or
        ``None`` when the quote is not in the searched range or is shorter
        than ten characters after folding.
    """
    end = len(text) if end is None else end
    needle = quote.strip().strip("\"'").strip()
    if not needle:
        return None
    exact = text.find(needle, start, end)
    if exact != -1:
        return exact, exact + len(needle)

    window = text[start:end]
    folded, mapping = _folded_with_map(window)
    target, _ = _folded_with_map(needle)
    target = target.strip()
    if len(target) < 10:
        return None
    position = folded.find(target)
    if position == -1:
        return None
    first = start + mapping[position]
    last = start + mapping[position + len(target) - 1] + 1
    return first, last


@dataclass(frozen=True)
class Statement:
    """A located quote, ready for a question to be written about it.

    Parameters
    ----------
    passage : Passage
        The passage the quote was drawn from.
    start : int
        Offset of the quote's first character, into the document.
    end : int
        Offset one past its last character.
    quote : str
        ``document.text[start:end]``, which may differ from what the model typed
        in whitespace or punctuation.
    claimed : str
        The text the model returned.
    """

    passage: Passage
    start: int
    end: int
    quote: str
    claimed: str

    @property
    def doc_id(self) -> str:
        """Return the source document."""
        return self.passage.doc_id


STATEMENT_SYSTEM = (
    "You select facts from documents for an evaluation set. Reply with one JSON "
    "object and nothing else."
)

STATEMENT_PROMPT = """Read the passage and copy out up to {count} statements a
reader of this kind of document might want to look up.

Rules:
- Copy each statement EXACTLY as it appears, character for character. Do not
  fix spelling, do not shorten, do not join sentences from different places.
- Each statement is one or two consecutive sentences, or one clause of a list,
  between 40 and 400 characters.
- Choose statements that state something specific: an obligation, a
  condition, a number, a period, a party, an exception, a definition.
- Skip headings, blank form fields, signature blocks, and boilerplate that
  says nothing specific.
- If nothing in the passage is worth a question, return an empty list.

Passage:
<<<
{passage}
>>>

Return JSON: {{"statements": ["<exact copy>", ...]}}"""


def extract_statements(
    model: Model,
    passage: Passage,
    document_text: str,
    *,
    count: int = 2,
) -> tuple[list[Statement], list[str]]:
    """Ask for verbatim statements from one passage and locate each one.

    Parameters
    ----------
    model : Model
        The generator.
    passage : Passage
        The passage to quote from.
    document_text : str
        The whole document, which the offsets index into.
    count : int, optional
        Statements to ask for, at most.

    Returns
    -------
    tuple of (list of Statement, list of str)
        The located statements, and the claimed quotes that could not be found.
    """
    payload = parse_json(
        model(
            STATEMENT_SYSTEM,
            STATEMENT_PROMPT.format(count=count, passage=passage.text),
        )
    )
    claimed = payload.get("statements") or []
    located: list[Statement] = []
    missing: list[str] = []
    seen: set[tuple[int, int]] = set()
    for item in claimed[:count] if isinstance(claimed, list) else []:
        quote = str(item).strip()
        span = locate_quote(document_text, quote, start=passage.start, end=passage.end)
        if span is None or span in seen:
            missing.append(quote)
            continue
        seen.add(span)
        located.append(
            Statement(
                passage, span[0], span[1], document_text[span[0] : span[1]], quote
            )
        )
    return located, missing


__all__ = ["STATEMENT_PROMPT", "Statement", "extract_statements", "locate_quote"]
