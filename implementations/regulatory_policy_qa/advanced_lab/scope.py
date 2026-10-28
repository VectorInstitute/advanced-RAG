"""A matched retrieval-scope audit for the answering-policy study."""

from __future__ import annotations

from typing import Any

from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.evaluation.metrics import score_retrieval
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever

from .answering import balanced_sample
from .common import EMBEDDING, OUTPUT, digest, provenance, require_compute, write_json


def retrieval_scores(rankings: dict[str, list[str]], qrels: dict[str, set[str]], *, k: int = 10) -> dict[str, float]:
    """Apply the same explicit cutoff to MRR and retrieval failure."""
    truncated = {query_id: ranking[:k] for query_id, ranking in rankings.items()}
    report = score_retrieval(truncated, qrels, k=k)
    return {"retrieval_failure": report.retrieval_failure_rate, "mrr": report.mrr}


def measure_scope() -> dict[str, Any]:
    """Isolate contract scoping on a fixed balanced sample without chat calls."""
    require_compute()
    path = contract_nli.fetch_contract_nli()
    corpus = contract_nli.build_slice(path, max_documents=20)
    queries = balanced_sample(corpus.queries)
    dense = DenseRetriever(SentenceTransformerEmbedder(EMBEDDING))
    dense.index(corpus.chunks)
    by_id = {c.chunk_id: c for c in corpus.chunks}
    arms = {}
    qrels = {q.query_id: corpus.qrels[q.query_id] for q in queries if q.is_answerable}
    for name, unscoped in (("scoped", False), ("unscoped", True)):
        rankings = {}
        outcomes = []
        for query in queries:
            evidence = dense.retrieve(query.question, 10, scope=None if unscoped else query.doc_id)
            ids = [item.provenance.chunk_id or "" for item in evidence]
            if not ids or any(not key for key in ids):
                raise ValueError("scope audit requires nonempty chunk-based evidence in both arms")
            outcomes.append(
                {
                    "query_id": query.query_id,
                    "expected_doc_id": query.doc_id,
                    "retrieved_doc_ids": [by_id[key].doc_id for key in ids],
                    "cross_contract_hits": sum(by_id[key].doc_id != query.doc_id for key in ids),
                }
            )
            if query.is_answerable:
                rankings[query.query_id] = ids
        arms[name] = {
            "retrieval": retrieval_scores(rankings, qrels),
            "cross_contract_hits": sum(row["cross_contract_hits"] for row in outcomes),
            "queries_with_cross_contract_hits": sum(row["cross_contract_hits"] > 0 for row in outcomes),
            "outcomes": outcomes,
        }
    report = {
        **provenance(),
        "corpus_sha256": digest(path),
        "questions": len(queries),
        "answerable_questions": len(qrels),
        "k": 10,
        "arms": arms,
        "hosted_calls": 0,
        "selection": "first answerable and first null query per each of 20 documents; same sample as the policy study",
    }
    write_json(OUTPUT / "scope.json", report)
    return report
