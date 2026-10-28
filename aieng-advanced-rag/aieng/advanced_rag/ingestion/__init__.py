"""Document ingestion: convert once, write the section folder every RI reads.

:mod:`~aieng.advanced_rag.ingestion.docling` holds the harness. It takes the
converter as an injected callable, so ``docling`` is not a dependency of this
package and only a caller who converts PDFs installs it. Read that module's
docstring for the on-disk shape, which is the contract two reference
implementations read.

:mod:`~aieng.advanced_rag.ingestion.tables` holds :class:`TableStorage`, the
four-valued setting that decides what is kept for each table, and the
ingest-time check that gates the one value whose output makes claims about a
table.
"""

from aieng.advanced_rag.ingestion.docling import (
    ARTIFACT_TYPE,
    PARSER_VERSION,
    DoclingConverter,
    IngestionResult,
    TableImageRenderer,
    ingest_corpus,
    ingest_document,
    source_sha256,
)
from aieng.advanced_rag.ingestion.tables import (
    TableStorage,
    TableValueCheck,
    check_values_on_page,
)


__all__ = [
    "ARTIFACT_TYPE",
    "PARSER_VERSION",
    "DoclingConverter",
    "IngestionResult",
    "TableImageRenderer",
    "TableStorage",
    "TableValueCheck",
    "check_values_on_page",
    "ingest_corpus",
    "ingest_document",
    "source_sha256",
]
