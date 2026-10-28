"""CUAD: contract-clause QA with span-derived relevance judgments.

The Contract Understanding Atticus Dataset ships as SQuAD-style extractive QA
over commercial contracts. Each contract is one long ``context`` string and each
answer carries a character offset into it, so a chunk is relevant exactly when
it overlaps a gold span.

Two properties shape every result read off this corpus:

1. The 41 questions are templated per clause category and repeat verbatim for
   every contract. A query carries no signal about *which* contract it belongs
   to, so retrieval is scoped to one contract and the task is locating the
   clause inside it. See :func:`queries_for_document`.
2. 2,938 of the 4,182 test questions are ``is_impossible``: the category does
   not appear in that contract. Those are the in-domain null cases RI #1 needs
   for "not stated" detection, and they carry no judgments.

Licence: CC BY 4.0, asserted by The Atticus Project, which permits commercial
use. See :data:`CUAD_LICENCE` and ``implementations/regulatory_policy_qa/
CORPUS.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query, Slice

from .chunking import check_reachable, chunk_document, relevant_chunk_ids
from .sources import Licence, PinnedSource, fetch


CUAD_LICENCE = Licence(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    permits_commercial_use=True,
    attribution=(
        "Contract Understanding Atticus Dataset (CUAD) v1, The Atticus Project, "
        "licensed CC BY 4.0. https://doi.org/10.5281/zenodo.4595826"
    ),
    notes=(
        "The Atticus Project's card states CUAD is 'free to the public for "
        "commercial and non-commercial use'. Contract text originates in SEC "
        "EDGAR filings; the CC BY 4.0 grant over the compiled dataset is the "
        "Atticus Project's own assertion."
    ),
)

#: The standard held-out split, pinned to the only commit that ever wrote
#: ``data.zip``. The Hugging Face mirror ``theatticusproject/cuad-qa`` is a
#: loading script whose ``_URL`` points at the moving ``main`` reference, so
#: pinning a Hugging Face revision does not pin CUAD's data. The zip does.
CUAD_TEST = PinnedSource(
    name="cuad-test",
    url=(
        "https://github.com/The-Atticus-Project/cuad/raw/"
        "7aad649183846688381a147eeb3601aff590fe4a/data.zip"
    ),
    sha256="f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a",
    licence=CUAD_LICENCE,
    member="test.json",
    member_sha256="007b6a40b0c65247f881627375c3d2e9b6aeeb5dfa89957494feed0765a1a073",
)

# CUAD question text is: Highlight the parts (if any) of this contract related
# to "<Category>" that should be reviewed by a lawyer. Details: <gloss>
_CATEGORY_RE = re.compile(r'related to "([^"]+)"')


def fetch_cuad_test(*, dest: Path | None = None, force: bool = False) -> Path:
    """Download and verify the CUAD test split, returning the local path."""
    return fetch(CUAD_TEST, dest=dest, force=force)


def parse_category(question: str) -> str:
    """Pull the clause category out of a templated CUAD question."""
    match = _CATEGORY_RE.search(question)
    return match.group(1) if match else "unknown"


def load_cuad(path: Path) -> Iterator[tuple[str, str, list[Query]]]:
    """Yield ``(doc_id, contract_text, queries)`` from a CUAD SQuAD file.

    Parameters
    ----------
    path : Path
        A CUAD ``test.json`` or ``train_separate_questions.json``.

    Yields
    ------
    tuple of (str, str, list of Query)
        The contract title, its full text, and every question asked of it.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for document in payload["data"]:
        doc_id = document["title"]
        for paragraph in document["paragraphs"]:
            yield doc_id, paragraph["context"], _queries(doc_id, paragraph["qas"])


def _queries(doc_id: str, qas: list[dict[str, Any]]) -> list[Query]:
    queries: list[Query] = []
    for qa in qas:
        question = str(qa["question"])
        answers = tuple(
            AnswerSpan(
                text=str(answer["text"]),
                start=int(answer["answer_start"]),
                end=int(answer["answer_start"]) + len(str(answer["text"])),
            )
            for answer in qa["answers"]
        )
        queries.append(
            Query(
                query_id=str(qa["id"]),
                doc_id=doc_id,
                category=parse_category(question),
                question=question,
                answers=answers,
            )
        )
    return queries


def queries_for_document(slice_: Slice, doc_id: str) -> list[Query]:
    """Return the queries scoped to one contract.

    CUAD's questions are identical across contracts, so a retriever asked to
    pick the right contract from the question alone is being measured on a task
    the data cannot support. Scope to a document first.
    """
    return [query for query in slice_.queries if query.contract_id == doc_id]


def build_slice(
    path: Path,
    *,
    max_documents: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> Slice:
    """Load CUAD, chunk it, and derive judgments from the answer offsets.

    Parameters
    ----------
    path : Path
        A CUAD SQuAD file, from :func:`fetch_cuad_test`.
    max_documents : int or None, optional
        Take the first N contracts in file order, which is stable across runs,
        so a smaller slice stays a subset of a larger one.
    chunk_chars : int, optional
        Chunk width in characters.
    overlap_chars : int, optional
        Chunk overlap in characters.

    Returns
    -------
    Slice
        Chunks, queries, and the derived judgments.
    """
    chunks: list[Chunk] = []
    queries: list[Query] = []
    qrels: dict[str, set[str]] = {}

    for count, (doc_id, text, doc_queries) in enumerate(load_cuad(path)):
        if max_documents is not None and count >= max_documents:
            break
        doc_chunks = chunk_document(
            doc_id, text, chunk_chars=chunk_chars, overlap_chars=overlap_chars
        )
        chunks.extend(doc_chunks)
        for query in doc_queries:
            queries.append(query)
            qrels[query.query_id] = relevant_chunk_ids(doc_chunks, query.answers)

    check_reachable(queries, qrels)
    return Slice(chunks=chunks, queries=queries, qrels=qrels)
