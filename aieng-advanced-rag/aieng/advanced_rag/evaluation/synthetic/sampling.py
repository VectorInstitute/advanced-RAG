"""Which text the generator sees: the held-out split, passages, pairs, and tags.

Four selection steps come before any model call, and none of them uses a graph
or a retriever's success:

* :func:`held_out_split` keeps some documents out of the index, so questions
  written from them are unanswerable by construction (the held-out nulls).
* :func:`sample_passages` picks passages spread across documents and across
  each document, and skips boilerplate a near-duplicate passage in another
  document already covers.
* :func:`select_pairs` proposes document pairs inside a cosine band for the
  bridge agent. A question set seeded from one method's extracted graph grades
  that graph and favours methods that build a similar one, so pairs come from
  embedding similarity, which no method under comparison authored.
* :func:`tag_conditions` turns corpus tags into fan-out gold sets.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


Encode = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Document:
    """One corpus document as the generator reads it.

    Parameters
    ----------
    doc_id : str
        Stable identifier, the same one the index uses.
    text : str
        The exact text the chunker sees. Evidence offsets index into it.
    title : str, optional
        A display title.
    tags : tuple of str, optional
        Corpus metadata for fan-out conditions.
    """

    doc_id: str
    text: str
    title: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Passage:
    """A window of a document handed to the statement extractor.

    Parameters
    ----------
    doc_id : str
        The source document.
    start : int
        Offset of the first character.
    end : int
        Offset one past the last character.
    text : str
        ``document.text[start:end]``.
    """

    doc_id: str
    start: int
    end: int
    text: str

    @property
    def passage_id(self) -> str:
        """Return ``doc_id@start-end``."""
        return f"{self.doc_id}@{self.start}-{self.end}"


def held_out_split(
    documents: Sequence[Document],
    *,
    share: float = 0.1,
    minimum: int = 3,
    seed: int = 0,
) -> tuple[list[Document], list[Document]]:
    """Split documents into the indexed part and the held-out part.

    Parameters
    ----------
    documents : sequence of Document
        The corpus.
    share : float, optional
        Fraction to hold out.
    minimum : int, optional
        Hold out at least this many, when the corpus has more than twice as many.
    seed : int, optional
        Shuffle seed.

    Returns
    -------
    tuple of (list of Document, list of Document)
        ``(indexed, held_out)``, each in the corpus's own order.
    """
    count = max(round(len(documents) * share), minimum)
    if len(documents) <= 2 * count:
        count = len(documents) // 3
    ids = [d.doc_id for d in documents]
    random.Random(seed).shuffle(ids)
    held = set(ids[:count])
    return (
        [d for d in documents if d.doc_id not in held],
        [d for d in documents if d.doc_id in held],
    )


_BREAK = re.compile(r"\n\s*\n|\n")


def passages_of(
    document: Document, *, target_chars: int = 1200, min_chars: int = 200
) -> list[Passage]:
    """Cut a document into passages on line breaks, near ``target_chars`` each.

    Lines are merged greedily until a passage reaches ``target_chars``. A line
    longer than twice the target is cut at the last sentence end before that
    width, so one unbroken page does not become one passage.

    Parameters
    ----------
    document : Document
        The document.
    target_chars : int, optional
        Rough passage width.
    min_chars : int, optional
        Passages shorter than this are dropped, as headings and signature
        blocks rarely carry a statement worth a question.

    Returns
    -------
    list of Passage
        In document order, with offsets into ``document.text``.
    """
    text = document.text
    cuts = [0] + [m.end() for m in _BREAK.finditer(text)] + [len(text)]
    pieces: list[tuple[int, int]] = []
    for line_start, end in zip(cuts, cuts[1:]):
        start = line_start
        while end - start > 2 * target_chars:
            window = text[start : start + 2 * target_chars]
            stop = max(window.rfind(". "), window.rfind("; "))
            cut = start + (stop + 2 if stop > target_chars // 2 else 2 * target_chars)
            pieces.append((start, cut))
            start = cut
        if end > start:
            pieces.append((start, end))

    passages: list[Passage] = []
    begin = None
    for start, end in pieces:
        if begin is None:
            begin = start
        if end - begin >= target_chars:
            passages.append(Passage(document.doc_id, begin, end, text[begin:end]))
            begin = None
    if begin is not None and len(text) > begin:
        passages.append(Passage(document.doc_id, begin, len(text), text[begin:]))
    return [p for p in passages if len(p.text.strip()) >= min_chars]


def sample_passages(
    documents: Sequence[Document],
    *,
    per_document: int = 4,
    seed: int = 0,
    target_chars: int = 1200,
    encode: Encode | None = None,
    boilerplate_cosine: float = 0.95,
) -> tuple[list[Passage], dict[str, Any]]:
    """Pick passages spread across documents and across each document.

    Each document's passages are split into ``per_document`` equal runs by
    position and one passage is drawn from each run, so a question set does not
    pile up on the opening recitals. Changing ``per_document`` changes the
    runs and therefore the draw, so a cached run stays free only while the
    setting stays fixed.

    Parameters
    ----------
    documents : sequence of Document
        The documents to draw from.
    per_document : int, optional
        Passages per document, at most.
    seed : int, optional
        Draw seed.
    target_chars : int, optional
        Passed to :func:`passages_of`.
    encode : callable or None, optional
        A local encoder. When given, a passage whose nearest neighbour in
        another document is above ``boilerplate_cosine`` is drawn at most once
        across the corpus.
    boilerplate_cosine : float, optional
        The near-duplicate threshold.

    Returns
    -------
    tuple of (list of Passage, dict)
        The passages, and a report of how many were seen, drawn, and skipped as
        boilerplate.
    """
    rng = random.Random(seed)
    by_doc = {d.doc_id: passages_of(d, target_chars=target_chars) for d in documents}
    drawn: list[Passage] = []
    for passages in by_doc.values():
        if not passages:
            continue
        runs = min(per_document, len(passages))
        size = len(passages) / runs
        for run in range(runs):
            low, high = int(run * size), max(int((run + 1) * size), int(run * size) + 1)
            drawn.append(passages[rng.randrange(low, high)])

    skipped = 0
    if encode is not None and drawn:
        drawn, skipped = _skip_boilerplate(drawn, encode, boilerplate_cosine)
    report = {
        "documents": len(documents),
        "passages_seen": sum(len(p) for p in by_doc.values()),
        "passages_drawn": len(drawn),
        "skipped_as_boilerplate": skipped,
    }
    return drawn, report


def _skip_boilerplate(
    passages: list[Passage], encode: Encode, threshold: float
) -> tuple[list[Passage], int]:
    vectors = [normalize(v) for v in encode([p.text for p in passages])]
    kept: list[int] = []
    skipped = 0
    for index, passage in enumerate(passages):
        duplicate = any(
            passages[other].doc_id != passage.doc_id
            and cosine(vectors[index], vectors[other]) > threshold
            for other in kept
        )
        if duplicate:
            skipped += 1
        else:
            kept.append(index)
    return [passages[i] for i in kept], skipped


# ---------------------------------------------------------------------------
# Document pairs in a cosine band, for the bridge agent.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePair:
    """Two documents proposed as bridgeable, with the evidence for proposing."""

    doc_a: str
    doc_b: str
    similarity: float
    shared_terms: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """Return an order-independent identity, so a pair is proposed once."""
        return (
            (self.doc_a, self.doc_b)
            if self.doc_a <= self.doc_b
            else (self.doc_b, self.doc_a)
        )


def normalize(vector: Sequence[float]) -> list[float]:
    """Return the vector scaled to unit length, or zeros for a zero vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * len(vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two unit vectors."""
    return sum(a * b for a, b in zip(left, right))


def embed_documents(
    documents: Sequence[Document], encode: Encode, batch_size: int = 32
) -> dict[str, list[float]]:
    """Embed each document once, with a local encoder.

    Parameters
    ----------
    documents : sequence of Document
        The documents.
    encode : callable
        A local encoder. Embeddings never go through the proxy.
    batch_size : int, optional
        Encoder batch size.

    Returns
    -------
    dict
        Document id to unit vector.
    """
    vectors: dict[str, list[float]] = {}
    for start in range(0, len(documents), batch_size):
        batch = list(documents[start : start + batch_size])
        for document, vector in zip(batch, encode([d.text for d in batch])):
            vectors[document.doc_id] = normalize(vector)
    return vectors


def _words(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 3 and w.isalnum()}


def document_frequencies(documents: Sequence[Any]) -> dict[str, int]:
    """Count how many documents each term appears in."""
    counts: dict[str, int] = {}
    for document in documents:
        for word in _words(document.text):
            counts[word] = counts.get(word, 0) + 1
    return counts


def shared_rare_terms(
    text_a: str,
    text_b: str,
    document_frequency: dict[str, int],
    total_documents: int,
    *,
    limit: int = 5,
    max_document_share: float = 0.1,
) -> tuple[str, ...]:
    """Return terms both documents use that are rare across the corpus.

    A graph-free hint at what the bridge might be. Terms in more than
    ``max_document_share`` of documents say nothing and are dropped. The agent
    still has to confirm the connection is real.
    """
    ceiling = max(total_documents * max_document_share, 1)
    shared = [
        word
        for word in _words(text_a) & _words(text_b)
        if 0 < document_frequency.get(word, 0) <= ceiling
    ]
    shared.sort(key=lambda word: (document_frequency.get(word, 0), word))
    return tuple(shared[:limit])


def select_pairs(
    documents: Sequence[Document],
    vectors: dict[str, list[float]],
    *,
    band: tuple[float, float] = (0.76, 0.88),
    limit: int = 100,
    max_per_document: int = 3,
    seed: int = 0,
) -> list[CandidatePair]:
    """Choose document pairs whose similarity falls inside the band.

    Too similar, and the two documents are near-duplicates, so a multi-hop
    question is answerable from either. Too dissimilar, and the agent invents a
    bridge. The band only proposes: the agent decides.

    Parameters
    ----------
    documents : sequence of Document
        The documents.
    vectors : dict
        Output of :func:`embed_documents`.
    band : tuple of float, optional
        Inclusive cosine window. The default was tuned on a ServerFault slice
        with BGE-small, where 0.45 to 0.75 admits nearly every pair and 0.76 to
        0.88 admits about 106 of 33,153. Survey another corpus with
        :func:`band_report` first.
    limit : int, optional
        Maximum pairs to return.
    max_per_document : int, optional
        Cap on pairs sharing a document, so one hub does not dominate.
    seed : int, optional
        Shuffle seed, so the selection is not the single tightest cluster.

    Returns
    -------
    list of CandidatePair
        Proposed pairs, most similar first within the band.

    Raises
    ------
    ValueError
        If the band's low end is above its high end.
    """
    low, high = band
    if not low <= high:
        raise ValueError(f"band must be (low, high) with low <= high, got {band}")
    by_id = {d.doc_id: d for d in documents}
    frequencies = document_frequencies(documents)
    ids = [d.doc_id for d in documents if d.doc_id in vectors]
    scored = _pairs_in_band(ids, vectors, low, high)
    random.Random(seed).shuffle(scored)
    scored.sort(key=lambda pair: -pair.similarity)

    used: dict[str, int] = {}
    chosen: list[CandidatePair] = []
    seen: set[tuple[str, str]] = set()
    for pair in scored:
        if pair.key in seen:
            continue
        if used.get(pair.doc_a, 0) >= max_per_document:
            continue
        if used.get(pair.doc_b, 0) >= max_per_document:
            continue
        seen.add(pair.key)
        used[pair.doc_a] = used.get(pair.doc_a, 0) + 1
        used[pair.doc_b] = used.get(pair.doc_b, 0) + 1
        chosen.append(
            CandidatePair(
                pair.doc_a,
                pair.doc_b,
                pair.similarity,
                shared_rare_terms(
                    by_id[pair.doc_a].text,
                    by_id[pair.doc_b].text,
                    frequencies,
                    len(ids),
                ),
            )
        )
        if len(chosen) >= limit:
            break
    return chosen


def _pairs_in_band(
    ids: Sequence[str],
    vectors: dict[str, list[float]],
    low: float,
    high: float,
    block: int = 512,
) -> list[CandidatePair]:
    """Return every pair whose cosine lies in the band.

    With numpy this is a blocked matrix product, which keeps memory to one block
    of rows at 10,000 documents. Without it, a pure-Python double loop.
    """
    try:
        import numpy as np
    except ImportError:
        return [
            CandidatePair(a, b, round(cosine(vectors[a], vectors[b]), 4))
            for index, a in enumerate(ids)
            for b in ids[index + 1 :]
            if low <= cosine(vectors[a], vectors[b]) <= high
        ]
    matrix = np.asarray([vectors[doc_id] for doc_id in ids], dtype=np.float32)
    scored: list[CandidatePair] = []
    for start in range(0, len(ids), block):
        stop = min(start + block, len(ids))
        similarities = matrix[start:stop] @ matrix.T
        for row in range(stop - start):
            absolute = start + row
            window = similarities[row, absolute + 1 :]
            for offset in np.nonzero((window >= low) & (window <= high))[0]:
                other = absolute + 1 + int(offset)
                scored.append(
                    CandidatePair(
                        ids[absolute], ids[other], round(float(window[offset]), 4)
                    )
                )
    return scored


def band_report(
    documents: Sequence[Document],
    vectors: dict[str, list[float]],
    bands: Sequence[tuple[float, float]],
) -> dict[str, int]:
    """Count the pairs in each candidate band, to tune it before any model call."""
    ids = [d.doc_id for d in documents if d.doc_id in vectors]
    counts = {f"{low}-{high}": 0 for low, high in bands}
    for index, a in enumerate(ids):
        for b in ids[index + 1 :]:
            similarity = cosine(vectors[a], vectors[b])
            for low, high in bands:
                if low <= similarity <= high:
                    counts[f"{low}-{high}"] += 1
    return counts


# ---------------------------------------------------------------------------
# Tag conditions, for fan-out questions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A predicate over the corpus and every document satisfying it.

    Parameters
    ----------
    kind : str
        What sort of predicate this is. Only ``tag`` is built here.
    value : str
        The predicate, such as a tag name.
    documents : tuple of str
        Every document satisfying it. This is the gold set.
    """

    kind: str
    value: str
    documents: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        """Return the number of documents in the gold set."""
        return len(self.documents)

    def describe(self) -> str:
        """Render the condition as the generator sees it."""
        return f"documents tagged '{self.value}'"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return asdict(self)


def tag_conditions(
    documents: Sequence[Document], min_documents: int = 3, max_documents: int = 8
) -> list[Condition]:
    """Build fan-out conditions from corpus tags.

    Parameters
    ----------
    documents : sequence of Document
        Documents with ``tags``.
    min_documents : int, optional
        Below this the question is a comparison, not a fan-out.
    max_documents : int, optional
        Above this the answer cannot be enumerated in one response.

    Returns
    -------
    list of Condition
        Sorted by size then value, so selection is deterministic.

    Raises
    ------
    ValueError
        If ``min_documents`` is below 3 or above ``max_documents``.
    """
    if min_documents < 3:
        raise ValueError("a fan-out question needs at least three documents")
    if max_documents < min_documents:
        raise ValueError("max_documents must not be below min_documents")
    by_tag: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        for tag in document.tags:
            by_tag[tag].append(document.doc_id)
    conditions = [
        Condition("tag", tag, tuple(sorted(doc_ids)))
        for tag, doc_ids in by_tag.items()
        if min_documents <= len(doc_ids) <= max_documents
    ]
    conditions.sort(key=lambda c: (c.size, c.value))
    return conditions


__all__ = [
    "CandidatePair",
    "Condition",
    "Document",
    "Encode",
    "Passage",
    "band_report",
    "cosine",
    "document_frequencies",
    "embed_documents",
    "normalize",
    "held_out_split",
    "passages_of",
    "sample_passages",
    "select_pairs",
    "shared_rare_terms",
    "tag_conditions",
]
