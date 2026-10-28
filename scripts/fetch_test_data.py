#!/usr/bin/env python3
"""Fetch the data the published-figure guards measure against.

The guards in ``aieng-advanced-rag/tests/aieng/advanced_rag/retrieval/
test_published_baselines.py`` re-measure retrieval figures on the first 20
contracts of the CUAD test split, with a local embedding model for the dense
arms and two fastembed models for the SPLADE and ColBERT arms. They skip on a
machine that lacks any of them, and a skipped guard checks nothing, so CI and a
fresh checkout both run this first.

The CUAD split comes through the package's loader, which verifies the SHA-256 of
the archive and of the member it extracts, so a changed upstream file fails
here rather than moving a figure. The embedding model comes from the Hugging
Face Hub without the ONNX, OpenVINO, and framework-duplicate weights that
sentence-transformers does not read, which is most of the repository's bytes.

``huggingface_hub`` comes with the package's ``local`` extra and ``fastembed`` with
``representations``, so run this from an environment synced with ``--all-extras``.

Examples
--------
Fetch both, then run the guards::

    $ python scripts/fetch_test_data.py
    $ cd aieng-advanced-rag
    $ pytest tests/aieng/advanced_rag/retrieval/test_published_baselines.py
"""

from __future__ import annotations

import sys

from aieng.advanced_rag.corpora.cuad import fetch_cuad_test
from aieng.advanced_rag.corpora.download import EMBEDDING_MODEL, fetch_embedding_model
from aieng.advanced_rag.retrieval.representations import (
    FastembedColbertEncoder,
    FastembedSparseEncoder,
)


# EMBEDDING_MODEL must match MODEL in test_published_baselines.py, which looks the
# model up in the Hugging Face cache by this name and skips when it is not there.


def main() -> int:
    """Fetch both datasets and print where each landed. Return an exit status."""
    print(f"cuad test split: {fetch_cuad_test()}")
    # The revision is printed so a figure that drifts can be traced to a changed
    # snapshot of the model rather than to the code under test.
    print(f"{EMBEDDING_MODEL}: {fetch_embedding_model()}")
    # fastembed picks its own files and serves SPLADE from a mirror, so building
    # the encoders is what fetches exactly what the guards later read offline.
    FastembedSparseEncoder()
    FastembedColbertEncoder()
    print("splade and colbert models fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
