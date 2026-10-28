"""Guards for the four published CUAD retrieval figures.

Each figure is a named constant with the configuration it was measured under, and
each test re-measures it two ways: through ``search``, which is how it was
published, and through ``retrieve``, which is the surface callers program
against. The two agreeing is what says moving onto the interface did not move a
baseline.

The 20-contract CUAD slice is not committed, so these tests skip unless the file
is there. Point ``ADVANCED_RAG_CUAD_TEST_JSON`` at a CUAD ``test.json``, or run
``scripts/fetch_test_data.py`` to put it in the corpus cache, and run:

    pytest tests/aieng/advanced_rag/retrieval/test_published_baselines.py

Nothing here reaches the network or a hosted embedding API. The dense arms run
``BAAI/bge-small-en-v1.5`` from the local Hugging Face cache and skip when it is
absent; indexing 954 chunks takes around 35 seconds per arm on a CPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NoReturn

import pytest

from aieng.advanced_rag.corpora.sources import cache_dir
from aieng.advanced_rag.evaluation.metrics import RetrievalReport, score_retrieval
from aieng.advanced_rag.graphrag.evidence import to_runs
from aieng.advanced_rag.retrieval.corpus import Chunk, Query, Slice, build_slice
from aieng.advanced_rag.retrieval.retrievers import (
    DenseRetriever,
    ShuffledRetriever,
    run_retriever,
)
from aieng.advanced_rag.retrieval.sparse import BM25Retriever


# Every figure below was measured on the first 20 contracts of the CUAD test
# split, chunked at 1200 characters with 200 of overlap, scored at k=10 over the
# 250 answerable queries, with retrieval scoped to the query's own contract.

# DenseRetriever over SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5",
# query_prompt=""). The committed baseline predates the embedder applying the
# model's published query prompt, so the prompt has to be disabled to reproduce
# it.
DENSE_PROMPTLESS_FAILURE_RATE = 0.2400
DENSE_PROMPTLESS_MRR = 0.4348

# The same run with the shipped SentenceTransformerEmbedder default, which
# applies the model's published query prompt to queries. The failure rate does
# not move, so only the mean reciprocal rank guards against a changed default.
DENSE_DEFAULT_PROMPT_FAILURE_RATE = 0.2400
DENSE_DEFAULT_PROMPT_MRR = 0.4632

# BM25Retriever at its shipped k1=1.2 and b=0.75, fitted over the whole slice.
BM25_MRR = 0.4528

# ShuffledRetriever(seed=0), the negative control: the same candidate pool in a
# fixed random order, every score 0.0.
CONTROL_FAILURE_RATE = 0.4800
CONTROL_MRR = 0.1895

# What the control becomes if anything sorts its results by descending score and
# breaks ties on chunk identifier: all-zero scores then rank each contract's
# first ten chunks in document order. This pair is the regression the interface's
# refusal to promise an order exists to prevent, and it is why the conformance
# kit checks shape and provenance rather than order.
CONTROL_RESORTED_FAILURE_RATE = 0.5400
CONTROL_RESORTED_MRR = 0.3083

# ColbertRetriever over answerdotai/answerai-colbert-small-v1 through fastembed.
# Both figures match the retrieval-frontier memo to four decimal places.
COLBERT_FAILURE_RATE = 0.2240
COLBERT_MRR = 0.5348

# SpladeRetriever over prithivida/Splade_PP_en_v1 through fastembed, which serves
# it from the Qdrant/Splade_PP_en_v1 mirror. These are what the code measures
# today. The retrieval-frontier memo published 0.2040 and 0.5377 for the same
# model and chunks, and that run recorded 149.5 expanded terms per chunk where
# this one produces 197.9. The mirror has not changed since March 2024 and
# fastembed 0.8.0 and 0.8.1 encode identically, so the cause is unresolved and
# the memo's SPLADE row does not reproduce. Until it is explained, this guard
# protects the retriever against a change and does not confirm the memo.
SPLADE_FAILURE_RATE = 0.2000
SPLADE_MRR = 0.6139

K = 10
MODEL = "BAAI/bge-small-en-v1.5"
# The repositories fastembed downloads, which are not always the model's name:
# it serves SPLADE from a mirror.
REPRESENTATION_REPOS = (
    "Qdrant/Splade_PP_en_v1",
    "answerdotai/answerai-colbert-small-v1",
)


def unavailable(reason: str) -> NoReturn:
    """Skip on a developer machine, fail in CI.

    These guards protect the published figures, and a skipped guard checks
    nothing: four of them went unchecked for weeks because a missing dataset
    read as a pass. GitHub sets ``CI``, so a run there that cannot reach the data
    fails and says what to fetch. Run ``scripts/fetch_test_data.py`` to have both
    datasets on any machine.
    """
    message = f"{reason}; run scripts/fetch_test_data.py to fetch it"
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def cuad_path() -> Path:
    """Return the CUAD test split to measure against, or the path that is missing.

    An explicit ``ADVANCED_RAG_CUAD_TEST_JSON`` wins over the corpus cache.
    """
    override = os.environ.get("ADVANCED_RAG_CUAD_TEST_JSON")
    if override:
        return Path(override)
    return cache_dir() / "cuad-test.json"


@pytest.fixture(scope="module")
def sliced() -> Slice:
    path = cuad_path()
    if not path.is_file():
        unavailable(f"the CUAD test split is not at {path}")
    prepared = build_slice(path, max_contracts=20)
    assert prepared.summary()["chunks"] == 954, "the slice is not the measured one"
    return prepared


@pytest.fixture(scope="module")
def local_model() -> str:
    try:
        import huggingface_hub as hub
        import sentence_transformers  # noqa: F401
    except ImportError:
        unavailable("the local extra is not installed")
    if hub.try_to_load_from_cache(MODEL, "config.json") is None:
        unavailable(f"{MODEL} is not in the local Hugging Face cache")
    return MODEL


@pytest.fixture(scope="module")
def representation_models() -> None:
    try:
        import fastembed  # noqa: F401
        import huggingface_hub as hub
    except ImportError:
        unavailable("the representations extra is not installed")
    for repo in REPRESENTATION_REPOS:
        if hub.try_to_load_from_cache(repo, "config.json") is None:
            unavailable(f"{repo} is not in the local Hugging Face cache")


def score_both_paths(
    retriever: Any, prepared: Slice
) -> tuple[RetrievalReport, RetrievalReport]:
    """Index once, then score the ``search`` path and the ``retrieve`` path."""
    retriever.index(prepared.chunks)
    searched = run_retriever(retriever, prepared.answerable, k=K)
    retrieved = {
        query.query_id: to_runs(
            retriever.retrieve(query.question, K, scope=query.doc_id)
        )
        for query in prepared.answerable
    }
    return (
        score_retrieval(searched, prepared.qrels, k=K),
        score_retrieval(retrieved, prepared.qrels, k=K),
    )


def embedder(model: str, **kwargs: object):
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model, **kwargs)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the model load off the network, so a guard cannot silently download."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_bm25_reproduces_its_published_mrr(sliced):
    searched, retrieved = score_both_paths(BM25Retriever(), sliced)

    assert searched.mrr == pytest.approx(BM25_MRR, abs=1e-4)
    assert retrieved.mrr == pytest.approx(BM25_MRR, abs=1e-4)
    assert searched.to_dict() == retrieved.to_dict()


def test_the_negative_control_reproduces_its_published_numbers(sliced):
    searched, retrieved = score_both_paths(ShuffledRetriever(seed=0), sliced)

    assert searched.retrieval_failure_rate == pytest.approx(
        CONTROL_FAILURE_RATE, abs=1e-4
    )
    assert searched.mrr == pytest.approx(CONTROL_MRR, abs=1e-4)
    assert retrieved.to_dict() == searched.to_dict()
    # Not the re-sorted pair: the interface returns the control's shuffle order.
    assert searched.mrr != pytest.approx(CONTROL_RESORTED_MRR, abs=1e-3)


def test_dense_reproduces_its_published_numbers_without_the_query_prompt(
    sliced, local_model
):
    retriever = DenseRetriever(embedder(local_model, query_prompt=""))

    searched, retrieved = score_both_paths(retriever, sliced)

    assert searched.retrieval_failure_rate == pytest.approx(
        DENSE_PROMPTLESS_FAILURE_RATE, abs=1e-4
    )
    assert searched.mrr == pytest.approx(DENSE_PROMPTLESS_MRR, abs=1e-4)
    assert retrieved.to_dict() == searched.to_dict()


def test_dense_with_the_shipped_embedder_default_reproduces_its_own_mrr(
    sliced, local_model
):
    retriever = DenseRetriever(embedder(local_model))

    searched, retrieved = score_both_paths(retriever, sliced)

    assert searched.retrieval_failure_rate == pytest.approx(
        DENSE_DEFAULT_PROMPT_FAILURE_RATE, abs=1e-4
    )
    assert searched.mrr == pytest.approx(DENSE_DEFAULT_PROMPT_MRR, abs=1e-4)
    assert retrieved.to_dict() == searched.to_dict()
    # The whole point of keeping both dense figures: the failure rate cannot tell
    # these two configurations apart, and the mean reciprocal rank can.
    assert DENSE_DEFAULT_PROMPT_FAILURE_RATE == DENSE_PROMPTLESS_FAILURE_RATE
    assert DENSE_DEFAULT_PROMPT_MRR != DENSE_PROMPTLESS_MRR


def test_the_control_keeps_its_shuffle_order_through_the_interface():
    # This one needs no corpus, so it runs everywhere: the interface must hand
    # back the control's shuffle order, not a ranking sorted on identifier.
    chunks = [
        Chunk(
            chunk_id=f"D1::{index:04d}",
            doc_id="D1",
            text=f"clause {index} about delivery schedules",
            start=index * 40,
            end=index * 40 + 39,
        )
        for index in range(20)
    ]
    control = ShuffledRetriever(seed=0)
    control.index(chunks)

    query = Query("q", "D1", "cat", "anything")
    through_search = [hit.chunk_id for hit in control.search(query, k=10)]
    through_interface = [
        item.chunk_id for item in control.retrieve("anything", 10, scope="D1")
    ]

    assert through_interface == through_search
    assert through_interface != sorted(through_interface)


def test_colbert_reproduces_its_published_figures(sliced, representation_models):
    from aieng.advanced_rag.retrieval.representations import ColbertRetriever

    searched, retrieved = score_both_paths(ColbertRetriever(), sliced)

    for report in (searched, retrieved):
        assert report.retrieval_failure_rate == pytest.approx(
            COLBERT_FAILURE_RATE, abs=1e-4
        )
        assert report.mrr == pytest.approx(COLBERT_MRR, abs=1e-4)
    assert searched.to_dict() == retrieved.to_dict()


def test_splade_holds_its_measured_figures(sliced, representation_models):
    from aieng.advanced_rag.retrieval.representations import SpladeRetriever

    searched, retrieved = score_both_paths(SpladeRetriever(), sliced)

    for report in (searched, retrieved):
        assert report.retrieval_failure_rate == pytest.approx(
            SPLADE_FAILURE_RATE, abs=1e-4
        )
        assert report.mrr == pytest.approx(SPLADE_MRR, abs=1e-4)
    assert searched.to_dict() == retrieved.to_dict()
