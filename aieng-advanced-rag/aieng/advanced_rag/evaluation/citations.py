"""Citation precision and recall for cited answers.

A cited answer makes two promises: each statement is true of the sources it
cites, and each cited source is there for a reason. This module turns both into
numbers in three steps.

1. Read the citations out of an answer. :func:`parse_inline` reads ``[1]`` or
   ``[<chunk_id>]`` markers sentence by sentence, :func:`parse_structured`
   reads JSON statements with source numbers or verbatim quotes,
   :func:`from_grounded` adapts the shipped generator's ``GroundedAnswer``,
   and :func:`from_anthropic` adapts the content blocks Anthropic's citations
   feature returns. Every path produces the same :class:`CitedAnswer`, whose
   citations carry document character offsets.
2. Score against gold evidence where a dataset has it.
   :func:`score_against_gold` counts a citation as correct when its character
   range overlaps an annotated evidence span (ContractNLI and CUAD both
   annotate spans).
3. Score with a support check where it does not. :func:`judge_citations`
   applies the ALCE definitions with any :class:`SupportChecker`: an LLM judge
   (:class:`LLMSupportChecker`) or a local NLI-style model
   (:class:`MiniCheckChecker`).

:func:`agreement` compares the two scorers citation by citation, which is how
you find out whether a judge can stand in for missing gold labels on your own
corpus.

:func:`generate_cited` is the generator side. ``style="quotes"`` asks the model
to quote the exact words it relies on, the prompt-level equivalent of the
quoted spans Anthropic's citations feature returns, and
:func:`locate_quote` finds each quote in its source so a quote the model made
up is caught rather than trusted.

Prior art and why this design
-----------------------------
ALCE (Gao et al., EMNLP 2023, MIT licence) defines citation recall per
statement (the concatenated cited passages entail it) and citation precision
per citation (a citation is irrelevant when it fails alone and the rest still
entail the statement), judged by the TRUE T5-11B NLI model. Those definitions
are used here unchanged. The ALCE code is not imported: it is an evaluation
script bound to its own data format and to an 11-billion-parameter checker that
does not fit a CPU workspace, and the definitions are a dozen lines.

TREC RAG 2024 (Thakur et al., 2025) judges each answer sentence against its
first cited passage on three levels (full, partial, no support) and found
GPT-4o matched NIST assessors exactly on 56% of sentences judged from scratch.
:class:`SupportVerdict` keeps those three levels, and ALCE's binary entailment
reads ``full`` as supported.

RAGAS faithfulness splits an answer into statements and has an LLM verdict
each against the whole retrieved context. It ignores which source a statement
cites, so it measures grounding, not citation quality.
:func:`~aieng.advanced_rag.evaluation.answers.judge_answer` already reports
grounding that way. FActScore (Min et al., 2023) splits text into atomic facts
before checking; answers in this harness run one to three sentences, so each
sentence is the statement, as in ALCE, and no extra decomposition call is paid.

TRUE, AlignScore, and MiniCheck are trained support checkers. MiniCheck-Flan-T5-
Large (Tang et al., EMNLP 2024; MIT weights, Apache-2.0 code) is reported to
match GPT-4 on the LLM-AggreFact benchmark at 770M parameters, so
:class:`MiniCheckChecker` runs it locally. The ``minicheck`` package is not on
PyPI, so its 30-line inference path is reproduced with ``transformers`` instead
of adding a git dependency. AttributionBench (Li et al., ACL Findings 2024)
reports that even a fine-tuned GPT-3.5 reaches about 80% macro-F1 on
attribution and that most errors miss fine-grained detail, which is why
:func:`agreement` reports a judge against gold spans before you trust it.

Provider citation features return different shapes: Anthropic returns the cited
text with character offsets into each document, Cohere and Gemini return answer
spans linked to source indices, and OpenAI's file search returns file
annotations. :func:`from_anthropic` maps the first onto :class:`CitedAnswer`.
The others carry no source span, so they reduce to source-number citations.

The completion function is a parameter throughout, so a cached, metered client
drives every model call.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

from ..graphrag.evidence import Evidence
from ..graphrag.synthesis import build_prompt


Style = Literal["answer", "inline", "quotes"]
Support = Literal["full", "partial", "none"]
Match = Literal["passage", "exact", "normalised", "fuzzy", "unmatched"]

#: How :func:`generate_cited` asks for citations, and what each style returns.
STYLES: tuple[Style, ...] = ("answer", "inline", "quotes")

INLINE_TEMPLATE = """Answer the question using only the sources below.

Question: {query}

Sources:
{sources}

Rules:
- Use only what the sources state. Add nothing from outside them.
- Write one to three sentences. Each sentence must make sense without the
  question, so name what it is about. When the question asks for yes or no,
  begin the first sentence with Yes or No.
- End every sentence with the numbers of the sources that support it, like
  [1] or [2][4], before the full stop.
- If the sources do not answer the question, reply exactly:
  Not stated in the provided sources.
"""

QUOTE_TEMPLATE = """Answer the question using only the sources below.

Question: {query}

Sources:
{sources}

Rules:
- Use only what the sources state. Add nothing from outside them.
- Write one to three statements. Each statement must make sense without the
  question, so name what it is about.
- For every statement, copy the exact words from the sources that support it.
  A quote is a contiguous piece of one source, copied character for character,
  usually one sentence. Do not paraphrase inside a quote.
- If the sources do not answer the question, set "not_stated" to true and
  leave "answer" and "statements" empty.

Return one JSON object and nothing else, in this form:
{{"answer": "<the direct answer in one sentence, beginning with Yes or No when the question asks for it>", "statements": [{{"text": "<statement>", "quotes": [{{"source": <number>, "quote": "<exact words>"}}]}}], "not_stated": false}}
"""

SUPPORT_PROMPT = """You are checking whether a source supports a statement.

Read only the source text. Decide whether the statement is true according to
that text alone, without outside knowledge. The question is context for what
the statement refers to; do not judge whether the statement answers it.

Question (context only):
{question}

Source text:
{premise}

Statement:
{claim}

Return one JSON object and nothing else, with exactly these keys:
  "support": "full" if the source text states or directly implies everything
    the statement says; "partial" if it supports some of it and is silent on
    or contradicts the rest; "none" if it does not support the statement.
  "reason": one sentence, at most 25 words.
"""

#: How a claim is written when :class:`MiniCheckChecker` reads the question too.
QUESTION_CLAIM = "Question: {question}\nAnswer: {claim}"

_MARKER_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")
_ABSTAIN_RE = re.compile(r"^\s*not stated in the provided sources\.?\s*$", re.I)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FOLD = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Citation:
    """One citation, resolved to the text and document range it points at.

    Parameters
    ----------
    source : int or None
        The 1-based source number in the prompt, or None when the citation was
        given by chunk identifier or by a provider's document index alone.
    doc_id : str
        The document cited.
    chunk_id : str or None
        The retrieved passage cited.
    text : str
        What a support check reads: the located quote, or the passage when
        there is no quote or the quote was not found.
    start, end : int or None
        Character range of ``text`` in the document, half-open. None when the
        passage carries no offsets.
    quote : str or None
        The quote as the model wrote it, when it gave one.
    match : str
        How the quote was found: ``exact``, ``normalised`` (after folding case,
        whitespace, and typographic quotes), ``fuzzy``, or ``unmatched``. A
        citation with no quote is ``passage``.
    """

    source: int | None
    doc_id: str
    chunk_id: str | None
    text: str
    start: int | None = None
    end: int | None = None
    quote: str | None = None
    match: Match = "passage"

    @property
    def span(self) -> tuple[int, int] | None:
        """Return the cited character range, or None without offsets."""
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)


@dataclass(frozen=True)
class Statement:
    """One sentence or statement of an answer and the citations attached to it."""

    text: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class CitedAnswer:
    """An answer split into statements, each with resolved citations.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    statements : tuple of Statement
        The answer's statements in order. Empty on an abstention.
    style : str
        How the citations were read: ``answer``, ``inline``, ``quotes``, or
        ``anthropic``.
    abstained : bool, optional
        Whether the answer declined.
    dropped : tuple of str, optional
        Citation markers that named no passage the model was shown, such as an
        out-of-range number or an invented chunk identifier.
    raw : str, optional
        The model's unedited response.
    answer : str, optional
        A direct answer given apart from the statements, as the quote format
        asks for. Empty when the statements are the answer.
    """

    query_id: str
    statements: tuple[Statement, ...]
    style: str
    abstained: bool = False
    dropped: tuple[str, ...] = ()
    raw: str = ""
    answer: str = ""

    @property
    def direct(self) -> str:
        """Return the direct answer when one was given, else the answer text."""
        return self.answer or self.text

    @property
    def text(self) -> str:
        """Return the answer text, statements joined by spaces."""
        return " ".join(statement.text for statement in self.statements)

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Return every citation in statement order, repeats included."""
        return tuple(c for statement in self.statements for c in statement.citations)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "query_id": self.query_id,
            "style": self.style,
            "abstained": self.abstained,
            "dropped": list(self.dropped),
            "raw": self.raw,
            "answer": self.answer,
            "statements": [
                {"text": s.text, "citations": [asdict(c) for c in s.citations]}
                for s in self.statements
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CitedAnswer:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            query_id=str(payload["query_id"]),
            style=str(payload["style"]),
            abstained=bool(payload.get("abstained", False)),
            dropped=tuple(payload.get("dropped", ())),
            raw=str(payload.get("raw", "")),
            answer=str(payload.get("answer", "")),
            statements=tuple(
                Statement(
                    text=str(s["text"]),
                    citations=tuple(Citation(**c) for c in s.get("citations", ())),
                )
                for s in payload.get("statements", ())
            ),
        )


# --------------------------------------------------------------------- quotes


def _fold(text: str) -> tuple[str, list[int]]:
    """Fold text for matching and map each folded character to its original index.

    Case, runs of whitespace, and typographic quotes and dashes are folded, so
    a quote that differs from its source only in those still locates exactly.
    """
    folded: list[str] = []
    origin: list[int] = []
    previous_space = True
    for index, original in enumerate(unicodedata.normalize("NFC", text)):
        character = _FOLD.get(original, original).casefold()
        if character.isspace():
            if previous_space:
                continue
            character, previous_space = " ", True
        else:
            previous_space = False
        folded.append(character)
        origin.append(index)
    while folded and folded[-1] == " ":
        folded.pop()
        origin.pop()
    return "".join(folded), origin


def _locate_part(
    part: str, text: str, min_ratio: float
) -> tuple[int, int, Match] | None:
    index = text.find(part)
    if index >= 0:
        return index, index + len(part), "exact"
    folded_text, origin = _fold(text)
    folded_part, _ = _fold(part)
    if not folded_part:
        return None
    index = folded_text.find(folded_part)
    if index >= 0:
        last = index + len(folded_part) - 1
        return origin[index], origin[last] + 1, "normalised"
    matcher = difflib.SequenceMatcher(None, folded_text, folded_part, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    first, final = blocks[0], blocks[-1]
    width = final.a + final.size - first.a
    # A fuzzy match has to cover most of the quote and stay about its length,
    # so scattered common words across a passage do not add up to a match.
    if matched / len(folded_part) < min_ratio or width > 1.5 * len(folded_part):
        return None
    return origin[first.a], origin[final.a + final.size - 1] + 1, "fuzzy"


def locate_quote(
    quote: str, text: str, *, min_ratio: float = 0.9
) -> tuple[int, int, Match] | None:
    """Find a quote in a passage, exactly, after folding, or approximately.

    An ellipsis in the quote splits it into parts that are located in order,
    and the range runs from the first part's start to the last part's end.

    Parameters
    ----------
    quote : str
        The quote as the model wrote it.
    text : str
        The passage it claims to come from.
    min_ratio : float, optional
        Share of the folded quote's characters a fuzzy match must cover.

    Returns
    -------
    tuple of (int, int, str) or None
        Start and end offsets in ``text`` and how it matched, or None when the
        quote is not in the passage.
    """
    parts = [
        p.strip(" \"'") for p in _ELLIPSIS_RE.split(quote.strip()) if p.strip(" \"'")
    ]
    if not parts:
        return None
    start: int | None = None
    end = 0
    worst: Match = "exact"
    order: dict[Match, int] = {"exact": 0, "normalised": 1, "fuzzy": 2}
    for part in parts:
        found = _locate_part(part, text[end:], min_ratio)
        if found is None:
            return None
        part_start, part_end, how = found
        start = end + part_start if start is None else start
        end += part_end
        worst = how if order[how] > order[worst] else worst
    assert start is not None
    return start, end, worst


# --------------------------------------------------------------------- parsing


def _offset(item: Evidence) -> int | None:
    return item.provenance.start


def _passage_citation(number: int | None, item: Evidence) -> Citation:
    start = _offset(item)
    return Citation(
        source=number,
        doc_id=item.doc_id,
        chunk_id=item.chunk_id,
        text=item.text,
        start=start,
        end=start + len(item.text) if start is not None else item.provenance.end,
    )


def _quote_citation(number: int, item: Evidence, quote: str) -> Citation:
    found = locate_quote(quote, item.text)
    if found is None:
        return replace(_passage_citation(number, item), quote=quote, match="unmatched")
    start, end, how = found
    offset = _offset(item)
    return Citation(
        source=number,
        doc_id=item.doc_id,
        chunk_id=item.chunk_id,
        text=item.text[start:end],
        start=offset + start if offset is not None else None,
        end=offset + end if offset is not None else None,
        quote=quote,
        match=how,
    )


def _resolve_token(
    token: str, evidence: Sequence[Evidence], by_id: Mapping[str, int]
) -> tuple[int | None, str | None]:
    """Return the 1-based source number a marker token names, or what to drop."""
    token = token.strip().strip("'\"")
    token = re.sub(r"^(?:source|doc|chunk|passage)\s*[:#]?\s*", "", token, flags=re.I)
    if token.isdigit():
        number = int(token)
        return (number, None) if 1 <= number <= len(evidence) else (None, token)
    if token in by_id:
        return by_id[token], None
    # Only an identifier-shaped token is a failed citation; "[sic]" is not one.
    if "::" in token or re.fullmatch(r"[\w.-]*\d[\w.-]*", token):
        return None, token
    return None, None


def _markers(
    text: str, evidence: Sequence[Evidence], by_id: Mapping[str, int]
) -> tuple[list[int], list[str], bool]:
    """Return cited numbers, dropped tokens, and whether any marker was a citation."""
    numbers: list[int] = []
    dropped: list[str] = []
    any_marker = False
    for match in _MARKER_RE.finditer(text):
        tokens = re.split(r"[,;]|\s+and\s+", match.group(1))
        resolved = [_resolve_token(t, evidence, by_id) for t in tokens if t.strip()]
        if not any(n is not None or d is not None for n, d in resolved):
            continue
        any_marker = True
        for number, bad in resolved:
            if number is not None and number not in numbers:
                numbers.append(number)
            if bad is not None:
                dropped.append(bad)
    return numbers, dropped, any_marker


def _strip_markers(
    text: str, evidence: Sequence[Evidence], by_id: Mapping[str, int]
) -> str:
    def replace(match: re.Match[str]) -> str:
        tokens = re.split(r"[,;]|\s+and\s+", match.group(1))
        resolved = [_resolve_token(t, evidence, by_id) for t in tokens if t.strip()]
        is_citation = any(n is not None or d is not None for n, d in resolved)
        return "" if is_citation else match.group(0)

    cleaned = _MARKER_RE.sub(replace, text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def split_statements(text: str) -> list[str]:
    """Split an answer into sentences, keeping each one's citation markers.

    A marker placed after the full stop (``... text. [2] Next``) belongs to the
    sentence before it, as a reader would take it, so it is moved back.

    Parameters
    ----------
    text : str
        The answer.

    Returns
    -------
    list of str
        Non-empty sentences in order.
    """
    pieces: list[str] = []
    for raw_line in re.split(r"\n+", text or ""):
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", raw_line).strip()
        if line:
            pieces.extend(p.strip() for p in _SENTENCE_END_RE.split(line) if p.strip())
    merged: list[str] = []
    for piece in pieces:
        rest = piece
        leading = re.match(r"^((?:\[[^\[\]\n]{1,120}\]\s*)+)(.*)$", piece, re.S)
        if leading and merged:
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}"
            rest = leading.group(2).strip()
        if rest:
            merged.append(rest)
    return merged


def _id_index(evidence: Sequence[Evidence]) -> dict[str, int]:
    index: dict[str, int] = {}
    for number, item in enumerate(evidence, start=1):
        if item.chunk_id:
            index.setdefault(item.chunk_id, number)
    return index


def parse_inline(query_id: str, text: str, evidence: Sequence[Evidence]) -> CitedAnswer:
    """Read an answer whose sentences carry ``[n]`` or ``[<chunk_id>]`` markers.

    Each marker cites the sentence it sits in. A number outside the sources or
    an identifier no passage carries goes to ``dropped``. An answer abstains
    when it is the abstention sentence, or when it cites nothing and reads as a
    refusal.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    text : str
        The answer text.
    evidence : sequence of Evidence
        The passages the model was shown, numbered from 1 in this order.

    Returns
    -------
    CitedAnswer
        One statement per sentence.
    """
    raw = text or ""
    if not raw.strip() or _ABSTAIN_RE.match(raw):
        return CitedAnswer(query_id, (), "inline", abstained=True, raw=raw)
    by_id = _id_index(evidence)
    statements: list[Statement] = []
    dropped: list[str] = []
    for sentence in split_statements(raw):
        numbers, bad, _ = _markers(sentence, evidence, by_id)
        dropped.extend(bad)
        clean = _strip_markers(sentence, evidence, by_id)
        # A stock "not stated" sentence beside an answer states nothing about
        # the sources, so it is not a statement to score.
        if not clean or _ABSTAIN_RE.match(clean):
            continue
        statements.append(
            Statement(
                clean,
                tuple(_passage_citation(n, evidence[n - 1]) for n in numbers),
            ),
        )
    if not any(s.citations for s in statements):
        from ..pipeline.generate import looks_like_abstention

        # An uncited refusal in prose ("The agreement does not address this.")
        # is an abstention, as the shipped generator's phrase fallback reads it.
        if looks_like_abstention(raw):
            return CitedAnswer(query_id, (), "inline", abstained=True, raw=raw)
    return CitedAnswer(
        query_id, tuple(statements), "inline", dropped=tuple(dropped), raw=raw
    )


def _first_json(raw: str) -> dict[str, Any] | None:
    match = _JSON_RE.search(raw or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_number(value: Any) -> int | None:
    try:
        return int(str(value).strip().strip("[]"))
    except (TypeError, ValueError):
        return None


def parse_structured(
    query_id: str, raw: str, evidence: Sequence[Evidence]
) -> CitedAnswer:
    """Read a JSON answer whose statements cite source numbers or quotes.

    The accepted shape is :data:`QUOTE_TEMPLATE`'s: a ``statements`` list
    whose items hold ``text`` and either ``quotes`` (``source`` and ``quote``)
    or ``sources`` (numbers), with ``not_stated`` for an abstention. A reply
    with no parsable JSON is read with :func:`parse_inline`, so a model that
    ignored the format is still scored on what it wrote.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    raw : str
        The model's response.
    evidence : sequence of Evidence
        The passages the model was shown, numbered from 1 in this order.

    Returns
    -------
    CitedAnswer
        One statement per item, with quotes located in their sources.
    """
    payload = _first_json(raw)
    if payload is None:
        return replace(parse_inline(query_id, raw, evidence), style="quotes")
    items = payload.get("statements") or []
    if bool(payload.get("not_stated")) or not items:
        return CitedAnswer(query_id, (), "quotes", abstained=True, raw=raw)
    statements: list[Statement] = []
    dropped: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        citations: list[Citation] = []
        for quote in item.get("quotes") or []:
            number = (
                _as_number(quote.get("source")) if isinstance(quote, dict) else None
            )
            words = (
                str(quote.get("quote", "")).strip() if isinstance(quote, dict) else ""
            )
            if number is None or not 1 <= number <= len(evidence):
                dropped.append(
                    str(quote.get("source") if isinstance(quote, dict) else quote)
                )
                continue
            if words:
                citations.append(_quote_citation(number, evidence[number - 1], words))
            else:
                citations.append(_passage_citation(number, evidence[number - 1]))
        for value in item.get("sources") or []:
            number = _as_number(value)
            if number is None or not 1 <= number <= len(evidence):
                dropped.append(str(value))
            elif all(c.source != number for c in citations):
                citations.append(_passage_citation(number, evidence[number - 1]))
        statements.append(Statement(str(item["text"]).strip(), tuple(citations)))
    return CitedAnswer(
        query_id,
        tuple(statements),
        "quotes",
        dropped=tuple(dropped),
        raw=raw,
        answer=str(payload.get("answer") or "").strip(),
    )


def from_grounded(answer: Any, evidence: Sequence[Evidence]) -> CitedAnswer:
    """Adapt a ``GroundedAnswer``, whose citations cover the whole answer.

    ``pipeline.generate`` cites at answer level, so the answer becomes one
    statement carrying every citation. Citation precision then asks whether
    each source was needed for the answer as a whole.

    Parameters
    ----------
    answer : GroundedAnswer
        From :func:`aieng.advanced_rag.pipeline.generate.generate`.
    evidence : sequence of Evidence
        The passages the model was shown.

    Returns
    -------
    CitedAnswer
        One statement, or none on an abstention.
    """
    if answer.abstained:
        return CitedAnswer(
            answer.query_id, (), "answer", abstained=True, raw=answer.raw
        )
    # Provenance holds a dict, so it cannot key a mapping; its identifiers can.
    numbers = {
        (item.doc_id, item.chunk_id, item.provenance.start): n
        for n, item in enumerate(evidence, start=1)
    }
    keys = [(p.doc_id, p.chunk_id, p.start) for p in answer.citations]
    citations = tuple(
        _passage_citation(numbers[k], evidence[numbers[k] - 1])
        for k in keys
        if k in numbers
    )
    return CitedAnswer(
        answer.query_id,
        (Statement(answer.text.strip(), citations),),
        "answer",
        dropped=tuple(str(n) for n in answer.dropped_citations),
        raw=answer.raw,
    )


def from_anthropic(
    query_id: str,
    content: Sequence[Mapping[str, Any]],
    documents: Sequence[Evidence],
) -> CitedAnswer:
    """Adapt the content blocks of a Messages response with citations enabled.

    Each ``text`` block becomes a statement and its ``citations`` become
    :class:`Citation` records. A ``char_location`` citation's offsets are
    relative to the document as sent and are shifted by the passage's own
    offset into the corpus document. A ``content_block_location`` citation
    names a range of a custom-content document's blocks and is read as a cite
    of the passage. Pass the response's ``content`` as dicts, for example
    ``[block.model_dump() for block in response.content]``.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    content : sequence of mapping
        The response's content blocks.
    documents : sequence of Evidence
        The passages sent as document blocks, in ``document_index`` order.

    Returns
    -------
    CitedAnswer
        One statement per non-empty text block.
    """
    statements: list[Statement] = []
    dropped: list[str] = []
    for block in content:
        if block.get("type") != "text" or not str(block.get("text", "")).strip():
            continue
        citations: list[Citation] = []
        for cite in block.get("citations") or []:
            index = cite.get("document_index")
            if not isinstance(index, int) or not 0 <= index < len(documents):
                dropped.append(str(index))
                continue
            item = documents[index]
            if cite.get("type") == "char_location":
                start, end = int(cite["start_char_index"]), int(cite["end_char_index"])
                offset = _offset(item)
                citations.append(
                    Citation(
                        source=index + 1,
                        doc_id=item.doc_id,
                        chunk_id=item.chunk_id,
                        text=item.text[start:end],
                        start=offset + start if offset is not None else None,
                        end=offset + end if offset is not None else None,
                        quote=str(cite.get("cited_text", "")),
                        match="exact",
                    ),
                )
            else:
                citations.append(_passage_citation(index + 1, item))
        statements.append(Statement(str(block["text"]).strip(), tuple(citations)))
    return CitedAnswer(
        query_id,
        tuple(statements),
        "anthropic",
        abstained=not statements,
        dropped=tuple(dropped),
    )


# --------------------------------------------------------------------- generation


def generate_cited(
    query_id: str,
    question: str,
    evidence: Sequence[Evidence],
    complete: Callable[[str], str],
    *,
    style: Style = "inline",
    budget: int | None = None,
    max_chars: int | None = 1200,
    template: str | None = None,
) -> CitedAnswer:
    """Answer from evidence with citations in one of three styles.

    ``answer`` uses the shipped generator, whose citations cover the whole
    answer. ``inline`` asks for ``[n]`` markers on every sentence, as ALCE
    prompts do. ``quotes`` asks for the exact supporting words of every
    statement, which :func:`locate_quote` then finds in the sources.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    question : str
        The question.
    evidence : sequence of Evidence
        Retrieved passages in rank order. An empty sequence abstains without a
        model call.
    complete : callable
        Takes a prompt, returns text.
    style : str, optional
        ``answer``, ``inline``, or ``quotes``.
    budget : int or None, optional
        Passages to place in the prompt.
    max_chars : int or None, optional
        Per-passage truncation in the prompt.
    template : str or None, optional
        A replacement prompt with ``{query}`` and ``{sources}``, returning the
        same format as the style's own.

    Returns
    -------
    CitedAnswer
        The answer, split into statements with resolved citations.
    """
    if style not in STYLES:
        raise ValueError(f"unknown citation style {style!r}; expected one of {STYLES}")
    used = list(evidence)[:budget] if budget is not None else list(evidence)
    if not used:
        return CitedAnswer(query_id, (), style, abstained=True)
    if style == "answer":
        from ..pipeline.generate import GENERATION_TEMPLATE, parse_answer

        raw = complete(
            build_prompt(question, used, template or GENERATION_TEMPLATE, max_chars)
        )
        grounded = parse_answer(query_id, raw, used, require_citations=False)
        return from_grounded(grounded, used)
    chosen = template or (INLINE_TEMPLATE if style == "inline" else QUOTE_TEMPLATE)
    raw = complete(build_prompt(question, used, chosen, max_chars))
    if style == "inline":
        return parse_inline(query_id, raw, used)
    return parse_structured(query_id, raw, used)


# --------------------------------------------------------------------- gold spans


def overlaps(
    first: tuple[int, int], second: tuple[int, int], min_overlap: float = 0.5
) -> bool:
    """Return whether two ranges share at least ``min_overlap`` of the shorter one.

    Measuring against the shorter range treats both granularities fairly: a
    1,200-character passage hits a 100-character gold span when it contains
    half of it, and a 60-character quote hits when half of the quote lies
    inside the span.
    """
    shared = min(first[1], second[1]) - max(first[0], second[0])
    shorter = min(first[1] - first[0], second[1] - second[0])
    if shared <= 0 or shorter <= 0:
        return False
    return shared / shorter >= min_overlap


@dataclass(frozen=True)
class GoldScore:
    """Citation precision and recall for one answer against annotated spans.

    Parameters
    ----------
    query_id : str
        Identifier of the question.
    hits : tuple of bool
        Per citation, in :attr:`CitedAnswer.citations` order, whether it
        overlaps a gold span.
    gold : int
        Gold spans for the question.
    covered : int
        Gold spans overlapped by at least one citation.
    shown : int
        Gold spans that overlapped a passage the model was shown, the most any
        citation could cover.
    """

    query_id: str
    hits: tuple[bool, ...]
    gold: int
    covered: int
    shown: int

    @property
    def precision(self) -> float | None:
        """Return the share of citations that overlap gold, or None with none."""
        return sum(self.hits) / len(self.hits) if self.hits else None

    @property
    def recall(self) -> float | None:
        """Return the share of gold spans cited, or None with no gold spans."""
        return self.covered / self.gold if self.gold else None

    @property
    def shown_recall(self) -> float | None:
        """Return recall over the gold spans the model was shown."""
        return self.covered / self.shown if self.shown else None


def score_against_gold(
    answer: CitedAnswer,
    gold_spans: Sequence[tuple[int, int]],
    *,
    doc_id: str | None = None,
    shown: Sequence[Evidence] = (),
    min_overlap: float = 0.5,
) -> GoldScore:
    """Score one answer's citations against annotated evidence spans.

    Parameters
    ----------
    answer : CitedAnswer
        The parsed answer.
    gold_spans : sequence of (int, int)
        Character ranges of the evidence in the document.
    doc_id : str or None, optional
        The document the spans belong to. A citation of another document never
        hits. None accepts any document.
    shown : sequence of Evidence, optional
        The passages the model was shown, for :attr:`GoldScore.shown_recall`.
    min_overlap : float, optional
        See :func:`overlaps`.

    Returns
    -------
    GoldScore
        Per-citation hits and span coverage.
    """
    gold: list[tuple[int, int]] = [(int(a), int(b)) for a, b in gold_spans]

    def hits(citation: Citation, span: tuple[int, int]) -> bool:
        if doc_id is not None and citation.doc_id != doc_id:
            return False
        return citation.span is not None and overlaps(citation.span, span, min_overlap)

    flags = tuple(any(hits(c, g) for g in gold) for c in answer.citations)
    covered = sum(1 for g in gold if any(hits(c, g) for c in answer.citations))
    shown_ranges: list[tuple[int, int]] = [
        (e.provenance.start, e.provenance.start + len(e.text))
        for e in shown
        if e.provenance.start is not None and (doc_id is None or e.doc_id == doc_id)
    ]
    reachable = sum(
        1 for g in gold if any(overlaps(r, g, min_overlap) for r in shown_ranges)
    )
    return GoldScore(answer.query_id, flags, len(gold), covered, reachable)


# --------------------------------------------------------------------- support checks


@dataclass(frozen=True)
class SupportVerdict:
    """Whether a premise supports a claim, on the TREC RAG three-level scale.

    Parameters
    ----------
    label : str
        ``full``, ``partial``, or ``none``.
    score : float or None, optional
        A probability of support, from a checker that gives one.
    reason : str, optional
        The judge's one-line reason, kept for auditing.
    """

    label: Support
    score: float | None = None
    reason: str = ""

    @property
    def supported(self) -> bool:
        """Return whether the support is full, ALCE's entailment."""
        return self.label == "full"


class SupportChecker(Protocol):
    """Anything that says whether a premise supports a claim."""

    name: str

    def __call__(
        self, premise: str, claim: str, *, question: str = ""
    ) -> SupportVerdict:
        """Return the verdict for one premise and claim."""
        ...


def parse_support(raw: str) -> SupportVerdict:
    """Read a support judge's JSON, treating anything unreadable as ``none``.

    A judge that ignored the format has not said the claim is supported, so an
    unreadable reply is no support rather than an error that loses the run.
    """
    payload = _first_json(raw)
    label = str((payload or {}).get("support", "")).strip().lower()
    if label not in ("full", "partial", "none"):
        lowered = (raw or "").lower()
        label = next(
            (w for w in ("partial", "full", "none") if f'"{w}"' in lowered), "none"
        )
    return SupportVerdict(label, reason=str((payload or {}).get("reason", ""))[:300])  # type: ignore[arg-type]


class LLMSupportChecker:
    """A support check made by a chat model.

    Parameters
    ----------
    complete : callable
        Takes a prompt, returns text. Supply a cached, metered client running
        the judging model.
    name : str, optional
        A label for reports, normally the model name.
    template : str, optional
        Prompt with ``{question}``, ``{premise}``, and ``{claim}``, returning
        :func:`parse_support`'s JSON.
    max_premise_chars : int or None, optional
        Truncate the premise to this many characters.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        name: str = "llm",
        *,
        template: str = SUPPORT_PROMPT,
        max_premise_chars: int | None = 6000,
    ) -> None:
        self.complete = complete
        self.name = name
        self.template = template
        self.max_premise_chars = max_premise_chars

    def __call__(
        self, premise: str, claim: str, *, question: str = ""
    ) -> SupportVerdict:
        """Return the judge's verdict for one premise and claim."""
        if self.max_premise_chars is not None:
            premise = premise[: self.max_premise_chars]
        prompt = self.template.format(
            question=question or "(none)", premise=premise, claim=claim
        )
        return parse_support(self.complete(prompt))


class MiniCheckChecker:
    """MiniCheck-Flan-T5-Large run locally, a 770M-parameter support checker.

    This reproduces the ``minicheck`` package's inference for the Flan-T5
    checkpoint: the input is ``"predict: " + premise + </s> + claim``, and the
    support probability is the softmax over the first decoded token's logits
    for ``0`` and ``1`` (token ids 3 and 209). The model returns two levels, so
    a verdict is ``full`` or ``none``. It reads the claim alone; ``question``
    is ignored, which is why statements must stand on their own.

    Needs ``torch`` and ``transformers``, imported on first use. The weights
    (about 3 GB) download from Hugging Face once.

    Parameters
    ----------
    model_id : str, optional
        The checkpoint.
    threshold : float, optional
        Support probability at or above which the verdict is ``full``.
    max_length : int, optional
        Token limit for premise and claim together.
    device : str, optional
        ``cpu`` or a CUDA device.
    question_in_claim : bool, optional
        Prefix the claim with the question (see :data:`QUESTION_CLAIM`) when
        the caller passes one, so a bare "Yes" or "No" has something to check.
    """

    name = "MiniCheck-Flan-T5-Large"

    def __init__(
        self,
        model_id: str = "lytang/MiniCheck-Flan-T5-Large",
        *,
        threshold: float = 0.5,
        max_length: int = 2048,
        device: str = "cpu",
        question_in_claim: bool = False,
    ) -> None:
        self.question_in_claim = question_in_claim
        if question_in_claim:
            self.name = f"{self.name}+question"
        self.model_id = model_id
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._memo: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model = model.to(self.device).eval()

    def claim_for(self, claim: str, question: str = "") -> str:
        """Return the text the checker reads as the claim.

        This is ``claim`` itself unless ``question_in_claim`` is on and a
        question is given.
        """
        if self.question_in_claim and question.strip():
            return QUESTION_CLAIM.format(question=question.strip(), claim=claim)
        return claim

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], batch_size: int = 8
    ) -> list[float]:
        """Return the support probability of each (premise, claim) pair.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Premise and claim.
        batch_size : int, optional
            Pairs per forward pass.

        Returns
        -------
        list of float
            Probability that each premise supports its claim.
        """
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        if todo:
            import torch

            self._load()
            eos = self._tokenizer.eos_token
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                inputs = self._tokenizer(
                    [f"predict: {premise}{eos}{claim}" for premise, claim in batch],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                decoder = torch.zeros(
                    (len(batch), 1), dtype=torch.long, device=self.device
                )
                with torch.no_grad():
                    logits = self._model(**inputs, decoder_input_ids=decoder).logits
                probs = torch.softmax(logits[:, 0, [3, 209]], dim=-1)[:, 1].tolist()
                self._memo.update(zip(batch, probs))
        return [self._memo[p] for p in pairs]

    def __call__(
        self, premise: str, claim: str, *, question: str = ""
    ) -> SupportVerdict:
        """Return the checker's verdict for one premise and claim."""
        probability = self.probabilities([(premise, self.claim_for(claim, question))])[
            0
        ]
        label: Support = "full" if probability >= self.threshold else "none"
        return SupportVerdict(label, score=round(probability, 4))


# --------------------------------------------------------------------- ALCE scoring


@dataclass(frozen=True)
class StatementJudgement:
    """ALCE verdicts for one statement.

    Parameters
    ----------
    text : str
        The statement.
    recall : bool
        It has a citation and its cited texts together fully support it.
    joint : str
        The support label for the cited texts together, ``none`` with no
        citation.
    alone : tuple of str
        Per citation, the support label for that citation's text alone.
    precise : tuple of bool
        Per citation, ALCE precision: the statement has recall and the citation
        is not irrelevant.
    """

    text: str
    recall: bool
    joint: Support
    alone: tuple[Support, ...]
    precise: tuple[bool, ...]


@dataclass(frozen=True)
class JudgedAnswer:
    """ALCE citation recall and precision for one answer."""

    query_id: str
    checker: str
    statements: tuple[StatementJudgement, ...]

    @property
    def recall(self) -> float | None:
        """Return the share of statements fully supported by their citations."""
        if not self.statements:
            return None
        return sum(s.recall for s in self.statements) / len(self.statements)

    @property
    def precision(self) -> float | None:
        """Return the share of citations that are not irrelevant."""
        flags = [p for s in self.statements for p in s.precise]
        return sum(flags) / len(flags) if flags else None

    @property
    def alone(self) -> tuple[Support, ...]:
        """Return each citation's single-source support label, in answer order."""
        return tuple(label for s in self.statements for label in s.alone)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "query_id": self.query_id,
            "checker": self.checker,
            "recall": self.recall,
            "precision": self.precision,
            "statements": [asdict(s) for s in self.statements],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JudgedAnswer:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            str(payload["query_id"]),
            str(payload["checker"]),
            tuple(
                StatementJudgement(
                    s["text"],
                    bool(s["recall"]),
                    s["joint"],
                    tuple(s["alone"]),
                    tuple(s["precise"]),
                )
                for s in payload["statements"]
            ),
        )


def judge_citations(
    answer: CitedAnswer,
    check: SupportChecker,
    *,
    question: str = "",
) -> JudgedAnswer:
    """Score one answer's citations with ALCE's definitions and a support check.

    For each statement, recall is 1 when it has a citation and the cited texts
    joined together fully support it. A citation is irrelevant when it does not
    support the statement alone and the remaining citations still do; its
    precision is 1 when the statement has recall and it is not irrelevant.
    Every citation is also checked alone, which ALCE skips when a statement has
    one citation, because :func:`agreement` compares that verdict with gold.

    Cited texts are deduplicated before joining, so two quotes from one
    sentence are not read twice.

    Parameters
    ----------
    answer : CitedAnswer
        The parsed answer. An abstention returns no statements.
    check : SupportChecker
        The support check.
    question : str, optional
        Passed to the check as context.

    Returns
    -------
    JudgedAnswer
        Per-statement verdicts.
    """
    memo: dict[tuple[str, str], SupportVerdict] = {}

    def verdict(texts: Sequence[str], claim: str) -> SupportVerdict:
        premise = "\n\n".join(dict.fromkeys(t.strip() for t in texts if t.strip()))
        if not premise:
            return SupportVerdict("none")
        key = (premise, claim)
        if key not in memo:
            memo[key] = check(premise, claim, question=question)
        return memo[key]

    judged: list[StatementJudgement] = []
    for statement in answer.statements:
        cited = statement.citations
        if not cited:
            judged.append(StatementJudgement(statement.text, False, "none", (), ()))
            continue
        texts = [c.text for c in cited]
        joint = verdict(texts, statement.text)
        alone = [verdict([c.text], statement.text) for c in cited]
        precise: list[bool] = []
        for index, single in enumerate(alone):
            if not joint.supported:
                precise.append(False)
                continue
            rest = texts[:index] + texts[index + 1 :]
            irrelevant = (
                not single.supported
                and bool(rest)
                and verdict(rest, statement.text).supported
            )
            precise.append(not irrelevant)
        judged.append(
            StatementJudgement(
                statement.text,
                joint.supported,
                joint.label,
                tuple(v.label for v in alone),
                tuple(precise),
            ),
        )
    return JudgedAnswer(
        answer.query_id, getattr(check, "name", "checker"), tuple(judged)
    )


# --------------------------------------------------------------------- agreement


@dataclass(frozen=True)
class Agreement:
    """A 2x2 table of gold against a judge, with accuracy and Cohen's kappa.

    Parameters
    ----------
    both : int
        Gold and judge both say yes.
    judge_only : int
        The judge says yes and gold says no.
    gold_only : int
        Gold says yes and the judge says no.
    neither : int
        Both say no.
    """

    both: int
    judge_only: int
    gold_only: int
    neither: int

    @property
    def total(self) -> int:
        """Return the number of items compared."""
        return self.both + self.judge_only + self.gold_only + self.neither

    @property
    def accuracy(self) -> float | None:
        """Return the share of items where gold and judge agree."""
        return (self.both + self.neither) / self.total if self.total else None

    @property
    def kappa(self) -> float | None:
        """Return Cohen's kappa, agreement corrected for chance."""
        n = self.total
        if not n:
            return None
        observed = (self.both + self.neither) / n
        gold_yes = (self.both + self.gold_only) / n
        judge_yes = (self.both + self.judge_only) / n
        expected = gold_yes * judge_yes + (1 - gold_yes) * (1 - judge_yes)
        return None if expected == 1 else (observed - expected) / (1 - expected)

    @property
    def judge_precision(self) -> float | None:
        """Return the share of the judge's yes verdicts that gold confirms."""
        yes = self.both + self.judge_only
        return self.both / yes if yes else None

    @property
    def judge_recall(self) -> float | None:
        """Return the share of gold's yes items the judge also accepts."""
        yes = self.both + self.gold_only
        return self.both / yes if yes else None

    def to_dict(self) -> dict[str, Any]:
        """Return the table and its rates."""
        return {
            "both": self.both,
            "judge_only": self.judge_only,
            "gold_only": self.gold_only,
            "neither": self.neither,
            "total": self.total,
            "accuracy": _round(self.accuracy),
            "kappa": _round(self.kappa),
            "judge_precision": _round(self.judge_precision),
            "judge_recall": _round(self.judge_recall),
        }


def agreement(pairs: Iterable[tuple[bool, bool]]) -> Agreement:
    """Count how often a judge agrees with gold labels.

    Parameters
    ----------
    pairs : iterable of (bool, bool)
        ``(gold, judge)`` per item, such as per citation: whether it overlaps a
        gold span, and whether the judge said its text supports the statement.

    Returns
    -------
    Agreement
        The 2x2 table.
    """
    counts = {(True, True): 0, (False, True): 0, (True, False): 0, (False, False): 0}
    for gold, judge in pairs:
        counts[(bool(gold), bool(judge))] += 1
    return Agreement(
        counts[(True, True)],
        counts[(False, True)],
        counts[(True, False)],
        counts[(False, False)],
    )


def citation_pairs(
    golds: Iterable[GoldScore], judged: Iterable[JudgedAnswer]
) -> list[tuple[bool, bool]]:
    """Pair each citation's gold hit with the judge's single-citation verdict.

    Parameters
    ----------
    golds : iterable of GoldScore
        Gold scores, one per answer.
    judged : iterable of JudgedAnswer
        Judged answers for the same questions, in any order.

    Returns
    -------
    list of (bool, bool)
        ``(overlaps gold, judged fully supported alone)`` per citation.
    """
    by_id = {j.query_id: j for j in judged}
    pairs: list[tuple[bool, bool]] = []
    for gold in golds:
        found = by_id.get(gold.query_id)
        if found is None:
            continue
        pairs.extend(
            (hit, label == "full") for hit, label in zip(gold.hits, found.alone)
        )
    return pairs


# --------------------------------------------------------------------- reports


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


CITATION_TABLE_HEADER = (
    "| Arm | Answered | Citations per answer | Quotes found | Gold precision | Gold recall "
    "| Judged recall | Judged precision |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


@dataclass(frozen=True)
class CitationReport:
    """Citation quality over one arm's answers.

    Means are macro averages over answered questions, as ALCE reports them.

    Parameters
    ----------
    method : str
        The arm's name.
    answers : list of CitedAnswer
        Every answer, abstentions included.
    gold : list of GoldScore, optional
        Gold scores for the answers that have gold spans.
    judged : list of JudgedAnswer, optional
        ALCE verdicts from one checker.
    """

    method: str
    answers: list[CitedAnswer]
    gold: list[GoldScore] = field(default_factory=list)
    judged: list[JudgedAnswer] = field(default_factory=list)

    @property
    def answered(self) -> int:
        """Return how many answers did not abstain."""
        return sum(1 for a in self.answers if not a.abstained)

    @property
    def citations_per_answer(self) -> float | None:
        """Return the mean number of citations per answered question."""
        return _mean(len(a.citations) for a in self.answers if not a.abstained)

    @property
    def quote_found_rate(self) -> float | None:
        """Return the share of quotes found in their source, None without quotes."""
        quoted = [c for a in self.answers for c in a.citations if c.quote is not None]
        if not quoted:
            return None
        return sum(1 for c in quoted if c.match != "unmatched") / len(quoted)

    @property
    def uncited_statement_rate(self) -> float | None:
        """Return the share of answered statements that carry no citation."""
        statements = [s for a in self.answers for s in a.statements]
        if not statements:
            return None
        return sum(1 for s in statements if not s.citations) / len(statements)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "method": self.method,
            "questions": len(self.answers),
            "answered": self.answered,
            "citations_per_answer": _round(self.citations_per_answer),
            "quote_found_rate": _round(self.quote_found_rate),
            "uncited_statement_rate": _round(self.uncited_statement_rate),
            "gold_precision": _round(_mean(g.precision for g in self.gold)),
            "gold_recall": _round(_mean(g.recall for g in self.gold)),
            "gold_shown_recall": _round(_mean(g.shown_recall for g in self.gold)),
            "judged_recall": _round(_mean(j.recall for j in self.judged)),
            "judged_precision": _round(_mean(j.precision for j in self.judged)),
        }

    def table_row(self, label: str | None = None) -> str:
        """Render one row under :data:`CITATION_TABLE_HEADER`."""
        row = self.to_dict()

        def cell(key: str) -> str:
            value = row[key]
            return "n/a" if value is None else f"{value:.3f}"

        return (
            f"| {label or self.method} | {row['answered']} | {cell('citations_per_answer')} "
            f"| {cell('quote_found_rate')} | {cell('gold_precision')} | {cell('gold_recall')} "
            f"| {cell('judged_recall')} | {cell('judged_precision')} |"
        )


__all__ = [
    "CITATION_TABLE_HEADER",
    "INLINE_TEMPLATE",
    "QUESTION_CLAIM",
    "QUOTE_TEMPLATE",
    "STYLES",
    "SUPPORT_PROMPT",
    "Agreement",
    "Citation",
    "CitationReport",
    "CitedAnswer",
    "GoldScore",
    "JudgedAnswer",
    "LLMSupportChecker",
    "MiniCheckChecker",
    "Statement",
    "StatementJudgement",
    "SupportChecker",
    "SupportVerdict",
    "agreement",
    "citation_pairs",
    "from_anthropic",
    "from_grounded",
    "generate_cited",
    "judge_citations",
    "locate_quote",
    "overlaps",
    "parse_inline",
    "parse_structured",
    "parse_support",
    "score_against_gold",
    "split_statements",
]
