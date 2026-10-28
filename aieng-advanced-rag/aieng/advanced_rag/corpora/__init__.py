"""Corpus loaders for RI #1, with the licence terms attached to each source.

Every corpus here is pinned to an immutable URL and verified against a SHA-256
digest on each fetch, and every one carries a :class:`Licence` record stating
what it permits. ``implementations/regulatory_policy_qa/CORPUS.md`` is the
provenance document those records summarise; read it before adding a corpus or
redistributing a derived slice.

    from aieng.advanced_rag.corpora import cuad, multihop_rag

    eval_slice = cuad.build_slice(cuad.fetch_cuad_test(), max_documents=20)
    far_nulls = multihop_rag.load_null_queries(
        multihop_rag.fetch_multihop_queries()
    )

Three corpora, three roles. CUAD supplies span QA over contracts and 2,938
in-domain nulls. ContractNLI supplies a second document genre with
human-annotated evidence spans and a 43% null rate; it is the recommended
replacement for PolicyQA, which is licensed for non-commercial use only and
cannot ship with this program. MultiHop-RAG supplies the out-of-domain null
control, restricted to its 301 null queries.
"""

from . import contract_nli, cuad, financebench, multihop_rag
from .chunking import chunk_document, relevant_chunk_ids
from .sources import (
    ChecksumMismatchError,
    Licence,
    PinnedSource,
    cache_dir,
    cached_path,
    fetch,
    is_cached,
)


#: Every source this package will fetch, keyed by the name used on disk.
SOURCES: dict[str, PinnedSource] = {
    source.name: source
    for source in (
        cuad.CUAD_TEST,
        contract_nli.CONTRACT_NLI_TRAIN,
        contract_nli.CONTRACT_NLI_DEV,
        contract_nli.CONTRACT_NLI_TEST,
        multihop_rag.MULTIHOP_QUERIES,
    )
}

__all__ = [
    "SOURCES",
    "ChecksumMismatchError",
    "Licence",
    "PinnedSource",
    "cache_dir",
    "cached_path",
    "chunk_document",
    "contract_nli",
    "cuad",
    "fetch",
    "financebench",
    "is_cached",
    "multihop_rag",
    "relevant_chunk_ids",
]
