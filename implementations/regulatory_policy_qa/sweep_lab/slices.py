"""Which slice of which corpus the sweep measures, and what it costs to run.

ContractNLI is the headline null set. Its 43.2% ``NotMentioned`` rate is an
operating point a threshold can be tuned against; CUAD's 70% ``is_impossible``
rate is not, because always abstaining scores 0.70 there, so CUAD is a secondary
in-domain check and is reported as one. MultiHop-RAG supplies the out-of-domain
control.

Corpus licensing, pinned versions, and the attribution that has to travel with
any published slice are in ``CORPUS.md``. Nothing here redistributes corpus
text: the loaders read from the local checksum-verified cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aieng.advanced_rag.corpora import contract_nli, cuad, multihop_rag
from aieng.advanced_rag.retrieval.corpus import Query, Slice


def contract_nli_slice(
    path: Path,
    *,
    documents: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> Slice:
    """Return a chunked ContractNLI slice with judgments derived from spans.

    Parameters
    ----------
    path : Path
        A ContractNLI split from ``contract_nli.fetch_contract_nli``.
    documents : int or None, optional
        Take the first N documents in file order, which is stable across runs.
        ``None`` takes the whole split.
    chunk_chars : int, optional
        Chunk width in characters, at the value every published number for this
        project was measured at.
    overlap_chars : int, optional
        Chunk overlap in characters.

    Returns
    -------
    Slice
        Chunks, queries, and judgments.
    """
    return contract_nli.build_slice(
        path,
        max_documents=documents,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )


def cuad_slice(
    path: Path,
    *,
    contracts: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> Slice:
    """Return a chunked CUAD slice, for the secondary in-domain check.

    Parameters
    ----------
    path : Path
        CUAD ``test.json`` from ``cuad.fetch_cuad_test``.
    contracts : int or None, optional
        Take the first N contracts in file order.
    chunk_chars : int, optional
        Chunk width in characters.
    overlap_chars : int, optional
        Chunk overlap in characters.

    Returns
    -------
    Slice
        Chunks, queries, and judgments.
    """
    return cuad.build_slice(
        path,
        max_documents=contracts,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )


def scoped_null_queries(
    path: Path,
    doc_ids: Sequence[str],
    *,
    limit: int | None = None,
) -> list[Query]:
    """Return MultiHop-RAG null queries, each scoped onto a contract.

    The loader leaves ``doc_id`` empty, because MultiHop-RAG's own corpus is not
    redistributable and is not used here. Retrieval in this harness is scoped to
    one document, so an unscoped query retrieves nothing, grades zero, and
    abstains at every threshold. That would make the control measure the empty
    string rather than the detector.

    Assigning each query a real contract fixes it. The question is about a news
    event and the passages are contract clauses, so the evidence is genuinely
    unrelated and the query is a far null: a detector that fails here has failed
    on the easy case. Assignment is round-robin over ``doc_ids``, so it is
    deterministic and spreads across documents.

    Parameters
    ----------
    path : Path
        ``MultiHopRAG.json`` from ``multihop_rag.fetch_multihop_queries``.
    doc_ids : sequence of str
        Documents to scope onto, normally the ones the headline slice indexed.
    limit : int or None, optional
        Take the first N null queries in file order.

    Returns
    -------
    list of Query
        Null queries carrying a retrieval scope and no answers.

    Raises
    ------
    ValueError
        If ``doc_ids`` is empty, since there would be nothing to scope onto.
    """
    if not doc_ids:
        raise ValueError("scoped_null_queries needs at least one document to scope onto")
    queries = multihop_rag.load_null_queries(path)
    if limit is not None:
        queries = queries[:limit]
    return [
        Query(
            query_id=query.query_id,
            doc_id=doc_ids[index % len(doc_ids)],
            category=query.category,
            question=query.question,
            answers=(),
        )
        for index, query in enumerate(queries)
    ]


def estimate_calls(
    queries: int,
    *,
    attempts: int = 2,
    max_chunks: int = 5,
    llm_grader: bool = True,
    rewriter: bool = True,
) -> dict[str, int]:
    """Return the model calls one harvest arm makes, before it is launched.

    A harvest takes every attempt whatever the grades say and generates on every
    attempt's evidence, because that is what makes the whole threshold grid free
    afterwards. It therefore costs more than any single pipeline run and far
    less than running the grid. The rewriter is not called on the final attempt,
    since there is nothing left to retry with.

    Parameters
    ----------
    queries : int
        Queries in the arm.
    attempts : int, optional
        Retrievals recorded per query.
    max_chunks : int, optional
        Passages the LLM grader scores per attempt, one call each.
    llm_grader : bool, optional
        False for the model-free ``HeuristicGrader``, which grades at no cost.
    rewriter : bool, optional
        Whether a model rewriter runs between attempts.

    Returns
    -------
    dict
        Calls by stage, and the total. This is an upper bound: an attempt that
        retrieves nothing grades and generates without a call.
    """
    grading = queries * attempts * max_chunks if llm_grader else 0
    rewriting = queries * (attempts - 1) if rewriter else 0
    generation = queries * attempts
    return {
        "grading": grading,
        "rewriting": rewriting,
        "generation": generation,
        "total": grading + rewriting + generation,
    }
