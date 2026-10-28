"""Score RI #1's retrieval arms under two label sets and compare the rankings.

The validation question is whether generated labels order retrieval
configurations the way human labels do. :func:`build_arms` returns the eight
arms the ContractNLI run compared, :func:`score_arms` scores any of them
against any qrels, and :func:`agreement` compares two score tables with
Kendall's tau and per-pair bootstrap signs.

:func:`validate` runs the whole comparison on one generated set and returns the
JSON record the RI #1 script commits and the special-topics notebook reads.
Embeddings and the reranker run locally; nothing here calls a model on the
proxy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from aieng.advanced_rag.corpora.chunking import chunk_document, relevant_chunk_ids
from aieng.advanced_rag.retrieval.corpus import Chunk, Query

from .agreement import compare_rankings, metric, pairwise_agreement, per_question
from .config import Category, GenerationConfig
from .sampling import Document
from .testset import SyntheticSet


DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#: Metric name and cut-off. A cut-off of zero means the metric has none (MRR).
METRICS: tuple[tuple[str, int], ...] = (("recall", 3), ("ndcg", 10), ("mrr", 0))

#: The configuration of the committed ContractNLI run. It has no few-shot
#: examples: ContractNLI's 17 hypotheses repeat verbatim across train, dev, and
#: test, so "real questions from the train side" would hand the generator the
#: test queries and make the agreement check circular.
CONTRACT_NLI_CONFIG = GenerationConfig(
    domain="non-disclosure agreements between companies",
    users=(
        Category(
            "in_house_counsel",
            "A lawyer reviewing the agreement, who uses contract vocabulary and asks precise questions.",
            2.0,
        ),
        Category(
            "business_manager",
            "A manager about to share information under the agreement, who asks "
            "in plain words what they may or may not do.",
            1.0,
        ),
    ),
    passages_per_document=3,
    statements_per_passage=2,
    passage_chars=1200,
    null_share=0.2,
    near_miss_share=0.5,
    held_out_share=0.1,
    held_out_minimum=3,
    held_out_passages=6,
    scope="document",
    seed=0,
    workers=12,
)

Search = Callable[[Query, int], list[Any]]
Arms = dict[str, tuple[Search, list[Chunk]]]
Scores = dict[str, dict[str, dict[str, float]]]


def load_contract_nli(count: int) -> tuple[list[Document], dict[str, list[Query]]]:
    """Return the first ``count`` ContractNLI test documents and their real queries.

    The pinned archive is fetched and verified on first use.
    """
    from aieng.advanced_rag.corpora import contract_nli  # noqa: PLC0415

    path = contract_nli.fetch_contract_nli()
    documents: list[Document] = []
    queries: dict[str, list[Query]] = {}
    for index, (doc_id, text, doc_queries) in enumerate(
        contract_nli.load_contract_nli(path)
    ):
        if index >= count:
            break
        documents.append(Document(doc_id, text))
        queries[doc_id] = doc_queries
    return documents, queries


class Reranked:
    """Dense first stage to ``depth``, then a cross-encoder to ``k``."""

    def __init__(
        self, first: Any, reranker: Any, texts: dict[str, str], depth: int = 20
    ) -> None:
        self.first, self.reranker, self.texts, self.depth = (
            first,
            reranker,
            texts,
            depth,
        )

    def search(self, query: Query, k: int = 10) -> list[Any]:
        """Return the reranked top ``k``."""
        from aieng.advanced_rag.retrieval.retrievers import Retrieved  # noqa: PLC0415

        candidates = [
            (h.chunk_id, self.texts[h.chunk_id])
            for h in self.first.search(query, k=self.depth)
        ]
        return [
            Retrieved(cid, 0.0)
            for cid in self.reranker.rerank(query.question, candidates, k)
        ]


def chunks_for(documents: Sequence[Document], width: int, overlap: int) -> list[Chunk]:
    """Chunk every document with fixed-width character windows."""
    return [
        c
        for d in documents
        for c in chunk_document(
            d.doc_id, d.text, chunk_chars=width, overlap_chars=overlap
        )
    ]


#: The document id every chunk is indexed under for corpus-wide search.
CORPUS_SCOPE = "*"


def build_arms(
    documents: Sequence[Document], embedder: Any, *, corpus_wide: bool = False
) -> Arms:
    """Return arm name to (search function, the chunks it ranks).

    The eight arms are BM25, dense, two hybrid weights, dense with a MiniLM
    reranker, and a shuffled control at 1,200-character chunks, and BM25 and
    dense at 600-character chunks.

    RI #1's retrievers rank one document's chunks per query. With
    ``corpus_wide`` every chunk is indexed under :data:`CORPUS_SCOPE`, so a
    query scoped to it ranks the whole corpus; the returned chunks keep their
    real documents, so qrels map onto them unchanged.
    """
    from aieng.advanced_rag.retrieval.rerankers import (
        CrossEncoderReranker,  # noqa: PLC0415
    )
    from aieng.advanced_rag.retrieval.retrievers import (  # noqa: PLC0415
        DenseRetriever,
        ShuffledRetriever,
    )
    from aieng.advanced_rag.retrieval.sparse import (  # noqa: PLC0415
        BM25Retriever,
        HybridRetriever,
    )

    arms: Arms = {}
    reranker = CrossEncoderReranker(RERANKER)
    for width, overlap in ((1200, 200), (600, 100)):
        chunks = chunks_for(documents, width, overlap)
        indexed = (
            [replace(c, doc_id=CORPUS_SCOPE) for c in chunks] if corpus_wide else chunks
        )
        bm25 = BM25Retriever()
        bm25.index(indexed)
        dense = DenseRetriever(embedder)
        dense.index(indexed)
        tag = f"@{width}"
        arms[f"bm25{tag}"] = (bm25.search, chunks)
        arms[f"dense{tag}"] = (dense.search, chunks)
        if width == 1200:
            for weight in (0.5, 0.2):
                hybrid = HybridRetriever(dense, bm25, dense_weight=weight)
                arms[f"hybrid-rrf-{weight:.1f}{tag}"] = (hybrid.search, chunks)
            texts = {c.chunk_id: c.text for c in chunks}
            arms[f"dense+rerank{tag}"] = (
                Reranked(dense, reranker, texts).search,
                chunks,
            )
            control = ShuffledRetriever(seed=0)
            control.index(indexed)
            arms[f"shuffled-control{tag}"] = (control.search, chunks)
    return arms


def real_qrels(
    queries: Sequence[Query], chunks: Sequence[Chunk]
) -> dict[str, set[str]]:
    """Map human-annotated evidence spans onto one arm's chunks."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    return {
        q.query_id: relevant_chunk_ids(by_doc[q.doc_id], q.answers) for q in queries
    }


def queries_for(generated: SyntheticSet, *, corpus_wide: bool = False) -> list[Query]:
    """Return a set's answerable questions as queries the arms can run.

    Parameters
    ----------
    generated : SyntheticSet
        One wording variant.
    corpus_wide : bool, optional
        Scope every query to :data:`CORPUS_SCOPE`, to match arms built with
        ``corpus_wide``. Otherwise each query keeps its own document.
    """
    if corpus_wide:
        # A corpus-scoped question carries a gold span in every document that
        # holds its quote, and regulations and contracts repeat boilerplate, so
        # it may name several documents. to_query refuses that, and the corpus-
        # wide arms need no single document: the qrels map every span.
        return [
            Query(
                query_id=q.query_id,
                doc_id=CORPUS_SCOPE,
                category=q.category,
                question=q.question,
                answers=q.answer_spans(),
            )
            for q in generated.answerable
        ]
    return [q.to_query() for q in generated.answerable]


def metric_name(name: str, k: int) -> str:
    """Return the label a metric is reported under, such as ``ndcg@10``."""
    return f"{name}@{k}" if k else name


def score_arms(
    arms: Arms,
    queries: Sequence[Query],
    qrels_for: Callable[[list[Chunk]], dict[str, set[str]]],
) -> Scores:
    """Return metric name to arm to question to score."""
    scores: Scores = {metric_name(m, k): {} for m, k in METRICS}
    for arm, (search, chunks) in arms.items():
        qrels = qrels_for(chunks)
        runs = {q.query_id: [h.chunk_id for h in search(q, 10)] for q in queries}
        for name, k in METRICS:
            scores[metric_name(name, k)][arm] = per_question(
                runs, qrels, metric(name, k or 10)
            )
    return scores


def mean_scores(scores: Scores) -> dict[str, dict[str, float]]:
    """Return metric name to arm to mean score, rounded to four places."""
    return {
        name: {
            arm: round(sum(values.values()) / len(values), 4) if values else 0.0
            for arm, values in by_arm.items()
        }
        for name, by_arm in scores.items()
    }


def agreement(
    real: Scores, generated: Scores, groups: dict[str, Sequence[str]]
) -> dict[str, dict[str, Any]]:
    """Return group to metric to :func:`compare_rankings`, bootstrapping pairs once."""
    out: dict[str, dict[str, Any]] = {group: {} for group in groups}
    for name, real_scores in real.items():
        pairs = pairwise_agreement(real_scores, generated[name], samples=1000)
        for group, arms in groups.items():
            out[group][name] = compare_rankings(
                {a: real_scores[a] for a in arms},
                {a: generated[name][a] for a in arms},
                pairs=pairs,
            )
    return out


def arm_groups(names: Sequence[str]) -> dict[str, Sequence[str]]:
    """Return the arm subsets the agreement is reported over."""
    return {
        "all_arms": list(names),
        "fixed_chunking_arms": [a for a in names if a.endswith("@1200")],
        "without_control": [a for a in names if not a.startswith("shuffled")],
    }


def validate(
    generated: SyntheticSet,
    documents: Sequence[Document],
    real_by_doc: dict[str, list[Query]],
    embedder: Any,
    *,
    variants: Sequence[str] = ("plain", "paraphrase"),
) -> dict[str, Any]:
    """Score every arm under real and generated labels and compare the rankings.

    Parameters
    ----------
    generated : SyntheticSet
        A set built over ``documents``; its ``meta`` names the indexed ones.
    documents : sequence of Document
        The corpus the set was built from.
    real_by_doc : dict
        Document id to its human-labelled queries.
    embedder : Any
        A local query and document encoder, such as
        :class:`~aieng.advanced_rag.retrieval.embedders.CachedQueryEmbedder`.
    variants : sequence of str, optional
        Wording variants to score.

    Returns
    -------
    dict
        ``arms``, ``real_mean``, the scored ``documents`` and query count, and
        per variant the question count and the agreement per arm group and
        metric.
    """
    indexed = set(generated.meta["documents"]["indexed"])
    scored = [d for d in documents if d.doc_id in indexed]
    real_queries = [q for d in scored for q in real_by_doc[d.doc_id] if q.is_answerable]
    encode = getattr(embedder, "encode_queries", None)
    if callable(encode):
        # Every question is encoded once, in batches, before the arms run, so
        # the per-query searches read cached vectors.
        encode(
            sorted({q.question for q in real_queries} | {q.question for q in generated})
        )
    arms = build_arms(scored, embedder)
    names = list(arms)
    real = score_arms(
        arms, real_queries, lambda chunks: real_qrels(real_queries, chunks)
    )
    results: dict[str, Any] = {}
    for variant in variants:
        chosen = generated.variant(variant)
        queries = queries_for(chosen)
        scores = score_arms(arms, queries, chosen.qrels)
        results[variant] = {
            "questions": len(queries),
            "paraphrased": sum(
                1 for q in chosen.answerable if q.variant == "paraphrase"
            ),
            **agreement(real, scores, arm_groups(names)),
        }
    return {
        "arms": names,
        "documents_scored": len(scored),
        "real_answerable_queries": len(real_queries),
        "real_mean": mean_scores(real),
        "agreement": results,
    }


__all__ = [
    "CONTRACT_NLI_CONFIG",
    "CORPUS_SCOPE",
    "DENSE_MODEL",
    "METRICS",
    "RERANKER",
    "Reranked",
    "agreement",
    "arm_groups",
    "build_arms",
    "chunks_for",
    "load_contract_nli",
    "mean_scores",
    "metric_name",
    "queries_for",
    "real_qrels",
    "score_arms",
    "validate",
]
