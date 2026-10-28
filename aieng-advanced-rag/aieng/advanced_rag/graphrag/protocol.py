"""The install-time error a graph adapter raises, and the retired graph protocol.

:class:`~aieng.advanced_rag.retrieval.interface.Retriever` is the interface the
reference implementations program against. A graph method is a retrieval method,
so it has no interface of its own, and ``GraphRetriever`` is kept here as an
alias of the shared one.

Answer synthesis is not part of it. See :mod:`aieng.advanced_rag.graphrag.synthesis`
for an optional helper that practitioners are expected to replace.
"""

from __future__ import annotations

from ..retrieval.interface import Retriever


class MissingExtraError(ImportError):
    """Raised when an adapter's optional dependency is not installed.

    Carries the exact install command, because the failure otherwise surfaces as
    a bare ``ModuleNotFoundError`` from deep inside a third-party import.
    """

    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f"The '{extra}' adapter needs the {package} package. "
            f"Install it with: pip install 'aieng-advanced-rag[{extra}]'"
        )
        self.extra = extra
        self.package = package


#: Deprecated alias for :class:`MissingExtraError`, kept for callers of the old
#: name. Deprecated in 0.1.0; use
#: ``MissingExtraError``.
MissingExtra = MissingExtraError


#: Deprecated alias for
#: :class:`~aieng.advanced_rag.retrieval.interface.Retriever`, kept because the
#: reference implementations type against the graph name. Deprecated in 0.1.1;
#: use ``Retriever``. Its first parameter is ``question`` rather than ``query``.
GraphRetriever = Retriever
