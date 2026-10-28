"""ContractNLI: NDA clause QA with human-annotated evidence spans.

Recommended replacement for PolicyQA, which cannot ship (see
``implementations/regulatory_policy_qa/CORPUS.md``). ContractNLI is 607
non-disclosure agreements annotated against 17 fixed hypotheses. Each
annotation is ``Entailment``, ``Contradiction`` or ``NotMentioned``, and the
first two carry evidence spans identified by a human annotator.

Two things make it fit the RI #1 harness without adaptation:

1. Evidence is given as indices into a per-document list of character offset
   pairs, so gold spans map onto chunks by the same overlap rule CUAD uses.
2. ``NotMentioned`` is a labelled in-domain null. The test split is 903 of
   2,091 annotations, a 43% null rate against CUAD's 70%, which gives the null
   detector a second operating point on a different document genre.

The hypotheses are fixed across documents, so, as with CUAD, a query carries no
signal about which document it belongs to. Scope retrieval to one document.

Licence: CC BY 4.0, released by Hitachi America, Ltd. See
:data:`CONTRACT_NLI_LICENCE`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query, Slice

from .chunking import check_reachable, chunk_document, relevant_chunk_ids
from .sources import Licence, PinnedSource, fetch


CONTRACT_NLI_LICENCE = Licence(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    permits_commercial_use=True,
    attribution=(
        "ContractNLI: A Dataset for Document-level Natural Language Inference "
        "for Contracts, Koreeda and Manning, Findings of EMNLP 2021, "
        "(c) Hitachi America, Ltd., licensed CC BY 4.0. "
        "https://stanfordnlp.github.io/contract-nli/"
    ),
    notes=(
        "The distribution ships a TERMS file alongside the licence: use is "
        "granted 'in accordance with the terms and conditions of the Creative "
        "Commons Attribution 4.0 International Public License', with a DMCA "
        "notice-and-takedown address for the source documents."
    ),
)

#: Pinned to the gh-pages commit that serves the release zip. The published
#: download link at ``stanfordnlp.github.io`` resolves to the same bytes.
CONTRACT_NLI_TEST = PinnedSource(
    name="contract-nli-test",
    url=(
        "https://raw.githubusercontent.com/stanfordnlp/contract-nli/"
        "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip"
    ),
    sha256="e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
    licence=CONTRACT_NLI_LICENCE,
    member="contract-nli/test.json",
    member_sha256="460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387",
)

CONTRACT_NLI_DEV = PinnedSource(
    name="contract-nli-dev",
    url=CONTRACT_NLI_TEST.url,
    sha256=CONTRACT_NLI_TEST.sha256,
    licence=CONTRACT_NLI_LICENCE,
    member="contract-nli/dev.json",
    member_sha256="310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e",
)

CONTRACT_NLI_TRAIN = PinnedSource(
    name="contract-nli-train",
    url=CONTRACT_NLI_TEST.url,
    sha256=CONTRACT_NLI_TEST.sha256,
    licence=CONTRACT_NLI_LICENCE,
    member="contract-nli/train.json",
    member_sha256="dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5",
)

#: Labels with no evidence spans. Treated as null queries by :func:`build_slice`.
NOT_MENTIONED = "NotMentioned"


def fetch_contract_nli(
    source: PinnedSource = CONTRACT_NLI_TEST,
    *,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and verify one ContractNLI split, returning the local path.

    The archive is 65 MB because it carries the source PDFs; only the requested
    JSON split is written to the cache.
    """
    return fetch(source, dest=dest, force=force)


def load_contract_nli(path: Path) -> Iterator[tuple[str, str, list[Query]]]:
    """Yield ``(doc_id, document_text, queries)`` from a ContractNLI split.

    Parameters
    ----------
    path : Path
        A ContractNLI ``train.json``, ``dev.json`` or ``test.json``.

    Yields
    ------
    tuple of (str, str, list of Query)
        The document id, its full text, and one query per hypothesis. Queries
        labelled ``NotMentioned`` carry no answers and are null cases.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    hypotheses = payload["labels"]
    for document in payload["documents"]:
        doc_id = str(document["id"])
        text = str(document["text"])
        yield doc_id, text, _queries(doc_id, text, document, hypotheses)


def _queries(
    doc_id: str,
    text: str,
    document: dict[str, Any],
    hypotheses: dict[str, dict[str, str]],
) -> list[Query]:
    offsets = document["spans"]
    annotations = document["annotation_sets"][0]["annotations"]
    queries: list[Query] = []
    for key, hypothesis in hypotheses.items():
        annotation = annotations.get(key)
        if annotation is None:
            continue
        answers = _answers(text, offsets, annotation)
        queries.append(
            Query(
                query_id=f"{doc_id}::{key}",
                doc_id=doc_id,
                category=hypothesis["short_description"],
                question=hypothesis["hypothesis"],
                answers=answers,
            )
        )
    return queries


def _answers(
    text: str,
    offsets: list[list[int]],
    annotation: dict[str, Any],
) -> tuple[AnswerSpan, ...]:
    """Resolve evidence span indices into character-offset answer spans.

    ``NotMentioned`` annotations carry an empty span list by construction, which
    is what makes them null queries.
    """
    if annotation["choice"] == NOT_MENTIONED:
        return ()
    spans: list[AnswerSpan] = []
    for index in annotation["spans"]:
        start, end = offsets[index]
        spans.append(AnswerSpan(text=text[start:end], start=start, end=end))
    return tuple(spans)


def build_slice(
    path: Path,
    *,
    max_documents: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> Slice:
    """Load a ContractNLI split, chunk it, and derive judgments from evidence.

    Parameters
    ----------
    path : Path
        A ContractNLI split, from :func:`fetch_contract_nli`.
    max_documents : int or None, optional
        Take the first N documents in file order, which is stable across runs.
    chunk_chars : int, optional
        Chunk width in characters.
    overlap_chars : int, optional
        Chunk overlap in characters.

    Returns
    -------
    Slice
        Chunks, queries, and the derived judgments. ``slice.null_queries`` holds
        the ``NotMentioned`` cases.
    """
    chunks: list[Chunk] = []
    queries: list[Query] = []
    qrels: dict[str, set[str]] = {}

    for count, (doc_id, text, doc_queries) in enumerate(load_contract_nli(path)):
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
