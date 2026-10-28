"""The ContractNLI question set, with labels and human evidence spans kept.

The package loader turns each ContractNLI annotation into a retrieval query and
drops the label's direction. Citation scoring needs both: the label tells
whether a "yes" or a "no" is correct, and the evidence spans are the gold that
citations are scored against. So this module reads the split itself, through
the same pinned file.

Every annotated hypothesis of the first :data:`DOCUMENTS` documents is kept
(``Entailment`` and ``Contradiction``), with :data:`NULLS_PER_DOCUMENT`
``NotMentioned`` hypotheses per document as nulls. A null has no evidence to
cite, so it tests only that the generator abstains rather than citing
something.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.corpora.chunking import chunk_document
from aieng.advanced_rag.retrieval.corpus import Chunk


#: Documents taken from the test split, in file order.
DOCUMENTS = 24
#: NotMentioned hypotheses kept per document as nulls.
NULLS_PER_DOCUMENT = 2
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200
SEED = 0

#: ContractNLI is three-way, so the question names all three outcomes. Asked
#: only for Yes or No, the pilot model answered "No" when the agreement was
#: silent, which reads as a contradiction.
QUESTION = (
    "Does the agreement say that {hypothesis}? Begin the answer with Yes if it does, "
    "or No if it says the opposite. If the agreement does not address this, say it is not stated."
)

_VERDICT = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Item:
    """One question with its label and gold evidence.

    Parameters
    ----------
    query_id : str
        ``<doc_id>::<hypothesis key>``, as the package loader names it.
    doc_id : str
        The agreement.
    key : str
        The ContractNLI hypothesis key, such as ``nda-11``.
    hypothesis : str
        The hypothesis as ContractNLI states it.
    label : str
        ``Entailment``, ``Contradiction``, or ``NotMentioned``.
    gold : tuple of (int, int)
        Character ranges of the annotated evidence. Empty for a null.
    """

    query_id: str
    doc_id: str
    key: str
    hypothesis: str
    label: str
    gold: tuple[tuple[int, int], ...]

    @property
    def question(self) -> str:
        """Return the question the generator is asked."""
        hypothesis = self.hypothesis.rstrip(".")
        return QUESTION.format(hypothesis=hypothesis[0].lower() + hypothesis[1:])

    @property
    def expected(self) -> str | None:
        """Return the correct verdict: ``yes``, ``no``, or None for a null."""
        verdicts = {"Entailment": "yes", "Contradiction": "no"}
        return verdicts.get(self.label)

    @property
    def is_null(self) -> bool:
        """Return whether the agreement does not mention the hypothesis."""
        return bool(self.label == contract_nli.NOT_MENTIONED)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return asdict(self)


def verdict_of(text: str) -> str | None:
    """Return ``yes`` or ``no`` from the answer's first word, or None.

    Pass :attr:`CitedAnswer.direct`, which is the quote format's separate
    answer when it has one.
    """
    match = _VERDICT.match(text or "")
    return match.group(1).lower() if match else None


def split_path() -> Path:
    """Return the pinned ContractNLI test split, downloading it once."""
    return Path(contract_nli.fetch_contract_nli())


def load(
    path: Path | None = None,
    *,
    documents: int = DOCUMENTS,
    nulls_per_document: int = NULLS_PER_DOCUMENT,
    seed: int = SEED,
) -> tuple[list[Item], dict[str, str]]:
    """Return the question set and the text of each document used.

    Parameters
    ----------
    path : Path or None, optional
        The split. :func:`split_path` when omitted.
    documents : int, optional
        Documents to take, in file order.
    nulls_per_document : int, optional
        NotMentioned hypotheses to keep per document, chosen with ``seed``.
    seed : int, optional
        Seed for the null choice.

    Returns
    -------
    tuple of (list of Item, dict)
        The items in document order, and document identifier to text.
    """
    payload = json.loads((path or split_path()).read_text(encoding="utf-8"))
    hypotheses = payload["labels"]
    rng = random.Random(seed)
    items: list[Item] = []
    texts: dict[str, str] = {}
    for document in payload["documents"][:documents]:
        doc_id = str(document["id"])
        texts[doc_id] = str(document["text"])
        offsets = document["spans"]
        annotations = document["annotation_sets"][0]["annotations"]
        nulls: list[Item] = []
        for key in sorted(annotations):
            annotation = annotations[key]
            item = Item(
                query_id=f"{doc_id}::{key}",
                doc_id=doc_id,
                key=key,
                hypothesis=str(hypotheses[key]["hypothesis"]),
                label=str(annotation["choice"]),
                gold=tuple((int(offsets[i][0]), int(offsets[i][1])) for i in annotation["spans"]),
            )
            (nulls if item.is_null else items).append(item)
        items.extend(rng.sample(nulls, min(nulls_per_document, len(nulls))))
    order = {doc_id: n for n, doc_id in enumerate(texts)}
    items.sort(key=lambda i: (order[i.doc_id], i.key))
    return items, texts


def chunks(texts: dict[str, str]) -> list[Chunk]:
    """Chunk every document at RI #1's default width and overlap."""
    return [
        chunk
        for doc_id, text in texts.items()
        for chunk in chunk_document(doc_id, text, chunk_chars=CHUNK_CHARS, overlap_chars=OVERLAP_CHARS)
    ]


def subset(items: Sequence[Item], size: int, *, seed: int = SEED) -> list[Item]:
    """Return a stratified subset for a live run, keeping each label's share.

    Each label keeps at least one item when the set has one, so a small
    subset still holds a contradiction and a null.

    Parameters
    ----------
    items : sequence of Item
        The full question set.
    size : int
        Items to keep.
    seed : int, optional
        Sampling seed.

    Returns
    -------
    list of Item
        The subset, in the order of ``items``.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Item]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    chosen: set[str] = set()
    for _label, group in sorted(by_label.items()):
        share = max(1, round(size * len(group) / len(items)))
        chosen.update(i.query_id for i in rng.sample(group, min(share, len(group))))
    return [i for i in items if i.query_id in chosen]


__all__ = [
    "CHUNK_CHARS",
    "DOCUMENTS",
    "NULLS_PER_DOCUMENT",
    "OVERLAP_CHARS",
    "QUESTION",
    "SEED",
    "Item",
    "chunks",
    "load",
    "split_path",
    "subset",
    "verdict_of",
]
