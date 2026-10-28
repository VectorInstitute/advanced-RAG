"""MultiHop-RAG null queries: the out-of-domain control for "not stated".

RI #1 uses this dataset for one thing, the Null control, and this module loads
only that: the 301 queries whose ``question_type`` is ``null_query``. Those
queries have an empty ``evidence_list`` and the fixed answer
``Insufficient information.``, so the slice they produce contains no news
article text of any kind.

That restriction is deliberate and it is a licensing boundary, not a
convenience. MultiHop-RAG is published under ODC-BY 1.0, which governs rights in
the *database* and says so: "this License does not cover any rights (other than
Database Rights or in contract) in individual Contents contained in the
Database" (section 2.4). The database's contents are 609 full article bodies
from The New York Times, the Financial Times, The Guardian, Wired, TechCrunch
and 44 other publishers, each copyright its publisher. Redistributing those is
not something ODC-BY grants, so this module never fetches ``corpus.json``.
The separate ``multihop_articles`` module requires a rights record for the
articles and an immutable source pin before fetching article content; no article
source is enabled by default. See ``implementations/regulatory_policy_qa/
CORPUS.md`` and RI #2's provenance record.

Know what this control measures before reading a number off it. The null
queries ask about news events, so against a contract corpus they are far nulls,
separable on topic alone. CUAD's 2,938 ``is_impossible`` questions and
ContractNLI's ``NotMentioned`` labels are the near nulls, where the category is
plausible for the document and absent. A null detector that handles the
far case and not the near one has not been shown to work.
"""

from __future__ import annotations

import json
from pathlib import Path

from aieng.advanced_rag.retrieval.corpus import Query

from .sources import Licence, PinnedSource, fetch


MULTIHOP_LICENCE = Licence(
    name="ODC-BY 1.0",
    url="https://opendatacommons.org/licenses/by/1-0/",
    permits_commercial_use=True,
    attribution=(
        "Contains information from MultiHop-RAG (Tang and Yang, COLM 2024, "
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG) which is made "
        "available under the ODC Attribution License."
    ),
    notes=(
        "ODC-BY 1.0 section 2.4 excludes rights in the individual contents of "
        "the database. The article bodies in corpus.json are third-party "
        "copyright and are not covered; only the null-query slice, which "
        "contains no article text, is redistributable on these terms."
    ),
)

#: Query file only. ``corpus.json`` is deliberately absent from this module.
MULTIHOP_QUERIES = PinnedSource(
    name="multihop-rag-queries",
    url=(
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/"
        "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/MultiHopRAG.json"
    ),
    sha256="03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    licence=MULTIHOP_LICENCE,
)

#: The fixed answer every null query carries upstream.
NULL_ANSWER = "Insufficient information."

#: Null queries belong to no document. ``Query`` requires a document id, so use
#: this sentinel rather than inventing an association the data does not have.
NO_DOCUMENT = ""

_NULL_QUESTION_TYPE = "null_query"


def fetch_multihop_queries(*, dest: Path | None = None, force: bool = False) -> Path:
    """Download and verify the MultiHop-RAG query file, returning its path."""
    return fetch(MULTIHOP_QUERIES, dest=dest, force=force)


def load_null_queries(path: Path) -> list[Query]:
    """Return the null-control queries from a MultiHop-RAG query file.

    Queries of any other ``question_type`` are dropped, along with every
    ``evidence_list`` entry, so nothing quoted from a news article survives into
    the returned objects.

    Parameters
    ----------
    path : Path
        A MultiHop-RAG ``MultiHopRAG.json``, from
        :func:`fetch_multihop_queries`.

    Returns
    -------
    list of Query
        Queries with no answers, in file order. Every one is a null case, so
        ``query.is_answerable`` is False throughout.

    Raises
    ------
    ValueError
        If a query typed as null carries evidence, which would mean the upstream
        file changed shape and the no-article-text guarantee no longer holds.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries: list[Query] = []
    for index, record in enumerate(payload):
        if record.get("question_type") != _NULL_QUESTION_TYPE:
            continue
        if record.get("evidence_list"):
            raise ValueError(
                f"null query {index} carries evidence; the upstream file "
                "changed and may now embed article text. Recheck CORPUS.md "
                "before using this slice."
            )
        queries.append(
            Query(
                query_id=f"multihop-null-{index:04d}",
                doc_id=NO_DOCUMENT,
                category=_NULL_QUESTION_TYPE,
                question=str(record["query"]),
                answers=(),
            )
        )
    return queries
