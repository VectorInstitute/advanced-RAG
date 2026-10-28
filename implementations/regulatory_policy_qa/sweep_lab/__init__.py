"""Machinery for the RI #1 threshold sweep.

The sweep itself is in ``aieng.advanced_rag.evaluation.sweep``: harvesting a
trace, replaying it at any threshold pair, and scoring both sides of the result.
What lives here is everything that is specific to this run and does not belong
in the shared library: which slice of which corpus is measured, how the
out-of-domain null control is scoped onto a contract corpus, the cached and
metered proxy client, and the report writer.
"""

from aieng.advanced_rag.graphrag.proxy import ResponseCache

from .proxy import CachedCompleter, ProxyClient, load_api_key, sweep_key
from .report import markdown_table, write_json
from .slices import (
    contract_nli_slice,
    cuad_slice,
    estimate_calls,
    scoped_null_queries,
)


__all__ = [
    "CachedCompleter",
    "ProxyClient",
    "ResponseCache",
    "contract_nli_slice",
    "cuad_slice",
    "estimate_calls",
    "load_api_key",
    "markdown_table",
    "scoped_null_queries",
    "sweep_key",
    "write_json",
]
