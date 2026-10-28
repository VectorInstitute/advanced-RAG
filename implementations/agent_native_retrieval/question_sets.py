"""Load RI #5's three small question sets, and the lexical checks they share.

The sets sit in ``outputs/``:

* ``musique_paraphrase_control.jsonl``: MuSiQue questions reworded so that no
  capitalized span of their gold passages appears in them. A literal ``grep``
  on the question's words cannot jump to the evidence, so this is the set where
  ``grep`` is built to lose.
* ``musique_near_miss_nulls.jsonl``: questions that look like MuSiQue questions
  and are answered nowhere in the 11,656 passages, each with the ``grep`` hit
  counts of the terms that changed.
* ``serverfault_error_strings.jsonl``: questions keyed on an exact error string
  or identifier that occurs in one document of the 10,000-question slice. The
  committed rows hold no post text. :func:`load_serverfault_questions` reads the
  key out of the local slice by its offsets and checks it against its digest.

    from aieng.advanced_rag.corpora import musique, serverfault
    import question_sets

    control = question_sets.load_jsonl(question_sets.PARAPHRASE_CONTROL)
    nulls = question_sets.load_jsonl(question_sets.NEAR_MISS_NULLS)
    errors = question_sets.load_serverfault_questions(
        serverfault.documents(serverfault.fetch_slice())
    )
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
PARAPHRASE_CONTROL = OUTPUTS / "musique_paraphrase_control.jsonl"
NEAR_MISS_NULLS = OUTPUTS / "musique_near_miss_nulls.jsonl"
SERVERFAULT_ERRORS = OUTPUTS / "serverfault_error_strings.jsonl"

#: A run of capitalized words, allowing short lowercase joiners inside a name
#: ("Bank of America", "Rio de Janeiro"). The rule the model-free grep probe used.
_SPAN = re.compile(r"[A-Z][\w'’.-]*(?:\s+(?:of|the|de|and|for|in)?\s*[A-Z][\w'’.-]*)*")

#: Capitalized words that open a sentence or a question and name nothing.
_FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "after",
        "before",
        "but",
        "by",
        "during",
        "for",
        "from",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    ]
)

_WORD = re.compile(r"[\w'’.-]+")


def capitalized_spans(text: str) -> list[str]:
    """Return the capitalized spans of ``text``, with leading function words dropped.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    list of str
        Spans in order of appearance, duplicates kept. A span that is only a
        function word, such as a sentence-initial "The", is dropped.
    """
    found: list[str] = []
    for match in _SPAN.finditer(text):
        words = match.group(0).strip(" .-'’").split()
        while words and words[0].lower() in _FUNCTION_WORDS:
            words = words[1:]
        if words:
            found.append(" ".join(words))
    return found


def contains_words(haystack: str, needle: str) -> bool:
    """Return whether ``needle`` is in ``haystack`` as whole words, ignoring case."""
    pattern = r"(?<![\w])" + re.escape(needle.lower()) + r"(?![\w])"
    return re.search(pattern, haystack.lower()) is not None


@dataclass(frozen=True)
class Lexicon:
    """Word statistics of a corpus, for telling names from ordinary words.

    Parameters
    ----------
    document_frequency : mapping of str to int
        Lower-cased word to the number of passages containing it.
    common : frozenset of str
        Lower-cased words the corpus writes in lower case more often than
        capitalized. A sentence-initial "Located" is one; "Paris" is not.
    passages : int
        Passages counted.
    """

    document_frequency: Mapping[str, int]
    common: frozenset[str]
    passages: int

    @classmethod
    def build(cls, texts: Iterable[str]) -> Lexicon:
        """Count words over ``texts``, one text per passage."""
        lower: dict[str, int] = {}
        upper: dict[str, int] = {}
        frequency: dict[str, int] = {}
        passages = 0
        for text in texts:
            passages += 1
            seen: set[str] = set()
            for word in _WORD.findall(text):
                key = word.lower()
                counts = upper if word[:1].isupper() else lower
                counts[key] = counts.get(key, 0) + 1
                seen.add(key)
            for key in seen:
                frequency[key] = frequency.get(key, 0) + 1
        common = frozenset(w for w, n in lower.items() if n > upper.get(w, 0))
        return cls(frequency, common, passages)


def name_spans(
    passages: Iterable[str],
    lexicon: Lexicon | None = None,
    *,
    max_word_share: float = 0.01,
) -> set[str]:
    """Return the capitalized spans of ``passages`` that a question must not carry.

    Every capitalized span of two or more words counts. A single capitalized
    word counts, whether it stands alone or sits inside a longer span, when it
    is not a function word and, given a lexicon, is neither ordinary vocabulary
    nor common in the corpus. ``grep`` for "Gaytan" finds "César Gaytan" among
    a handful of passages, so "Gaytan" counts; "County" or "Italian" matches
    hundreds of passages and selects nothing, so it does not.

    Parameters
    ----------
    passages : iterable of str
        Gold passage titles and texts.
    lexicon : Lexicon or None, optional
        Corpus statistics. Without one, every word of every span counts.
    max_word_share : float, optional
        A single word counts when at most this share of passages contains it.

    Returns
    -------
    set of str
        The spans.
    """

    def ordinary(word: str) -> bool:
        key = word.lower()
        return len(word) < 2 or key in _FUNCTION_WORDS or (lexicon is not None and key in lexicon.common)

    def rare(word: str) -> bool:
        if lexicon is None:
            return True
        ceiling = max(1.0, lexicon.passages * max_word_share)
        return lexicon.document_frequency.get(word.lower(), 0) <= ceiling

    spans: set[str] = set()
    for text in passages:
        for span in capitalized_spans(text):
            words = span.split()
            if len(words) > 1:
                spans.add(span)
            spans.update(w for w in words if not ordinary(w) and rare(w))
    return spans


def leaked_spans(
    question: str,
    passages: Iterable[str],
    lexicon: Lexicon | None = None,
    *,
    max_word_share: float = 0.01,
) -> list[str]:
    """Return the :func:`name_spans` of ``passages`` that appear in ``question``.

    A span appears when the question contains it as whole words, ignoring case.

    Returns
    -------
    list of str
        Distinct leaked spans, sorted. Empty is the goal.
    """
    spans = name_spans(passages, lexicon, max_word_share=max_word_share)
    return sorted(s for s in spans if contains_words(question, s))


def grep_count(corpus_lower: Sequence[str], term: str) -> int:
    """Count passages containing ``term`` as a case-insensitive substring.

    This is the probe's literal match: the same rule a ``grep -i -F`` over
    title and text applies.
    """
    needle = term.lower().strip()
    if not needle:
        return 0
    return sum(1 for text in corpus_lower if needle in text)


def grep_hits(corpus_lower: Sequence[str], term: str) -> list[int]:
    """Return the positions of passages containing ``term``, ignoring case."""
    needle = term.lower().strip()
    if not needle:
        return []
    return [i for i, text in enumerate(corpus_lower) if needle in text]


def new_terms(changed: str, original: str) -> list[str]:
    """Return capitalized spans and numbers in ``changed`` that ``original`` lacks.

    These are the details a near-miss edit introduced, which is what a ``grep``
    for the edited question would search for.
    """
    terms = capitalized_spans(changed) + re.findall(r"\b\d{3,4}\b", changed)
    seen: list[str] = []
    for term in terms:
        if not contains_words(original, term) and term not in seen:
            seen.append(term)
    return seen


def token_f1(prediction: str, references: Iterable[str]) -> float:
    """Return the best SQuAD-style token F1 of ``prediction`` against any reference."""

    def tokens(text: str) -> list[str]:
        text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
        return re.findall(r"\w+", text)

    best = 0.0
    predicted = tokens(prediction)
    for reference in references:
        gold = tokens(reference)
        common = sum(min(predicted.count(t), gold.count(t)) for t in set(gold))
        if not common:
            continue
        precision, recall = common / len(predicted), common / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return the rows of a JSONL file, skipping blank lines."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as JSONL, one sorted-key object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ServerFault: question text is built locally from the slice.
# ---------------------------------------------------------------------------

#: Question wording per key kind. The wording is ours; only ``{key}`` comes from
#: a post, and it is read from the local slice, never from the committed file.
TEMPLATES: Mapping[str, str] = {
    "message": 'I get the error "{key}". What causes it, and how do I fix it?',
    "code": "My server reports {key}. What does it mean, and how do I resolve it?",
    "identifier": "What is {key}, and how do I fix the problem it is involved in?",
}


@dataclass(frozen=True)
class ErrorStringQuestion:
    """One ServerFault question, materialised from the slice.

    Parameters
    ----------
    query_id : str
        Stable identifier, ``sf-err-`` and a number.
    question : str
        The template with the key filled in.
    key : str
        The exact error string or identifier.
    kind : str
        ``message``, ``code``, or ``identifier``.
    gold_doc_ids : tuple of str
        The documents of the slice that contain the key, which is the one
        document the row names.
    """

    query_id: str
    question: str
    key: str
    kind: str
    gold_doc_ids: tuple[str, ...]


def _field_text(document: Any, field: str) -> str:
    if field == "title":
        return str(document.title)
    if field == "question":
        return str(document.question_text)
    raise ValueError(f"unknown field {field!r}")


def load_serverfault_questions(documents: Sequence[Any], path: Path = SERVERFAULT_ERRORS) -> list[ErrorStringQuestion]:
    """Materialise the ServerFault set from the local slice.

    Parameters
    ----------
    documents : sequence
        ``aieng.advanced_rag.corpora.serverfault.documents`` of the
        10,000-question slice.
    path : Path, optional
        The committed rows.

    Returns
    -------
    list of ErrorStringQuestion
        In file order.

    Raises
    ------
    ValueError
        If a gold document is missing, a key's digest differs, or the key no
        longer occurs in exactly the gold documents.
    """
    by_id = {d.doc_id: d for d in documents}
    texts = {d.doc_id: d.text for d in documents}
    questions: list[ErrorStringQuestion] = []
    for row in load_jsonl(path):
        doc_id = row["key_location"]["doc_id"]
        if doc_id not in by_id:
            raise ValueError(f"{row['query_id']}: {doc_id} is not in the slice")
        location = row["key_location"]
        source = _field_text(by_id[doc_id], location["field"])
        key = source[location["start"] : location["end"]]
        if sha256_text(key) != row["key_sha256"]:
            raise ValueError(f"{row['query_id']}: the key at its offsets has changed")
        holders = tuple(sorted(i for i, text in texts.items() if key in text))
        if holders != tuple(sorted(row["gold_doc_ids"])):
            raise ValueError(f"{row['query_id']}: the key occurs in {list(holders)}, expected {row['gold_doc_ids']}")
        questions.append(
            ErrorStringQuestion(
                query_id=row["query_id"],
                question=TEMPLATES[row["kind"]].format(key=key),
                key=key,
                kind=row["kind"],
                gold_doc_ids=tuple(row["gold_doc_ids"]),
            )
        )
    return questions


__all__ = [
    "NEAR_MISS_NULLS",
    "OUTPUTS",
    "PARAPHRASE_CONTROL",
    "SERVERFAULT_ERRORS",
    "TEMPLATES",
    "ErrorStringQuestion",
    "Lexicon",
    "capitalized_spans",
    "contains_words",
    "grep_count",
    "grep_hits",
    "leaked_spans",
    "load_jsonl",
    "load_serverfault_questions",
    "name_spans",
    "new_terms",
    "sha256_text",
    "token_f1",
    "write_jsonl",
]
