"""A reusable conformance kit for the retrieval interface.

Point it at a retriever and it reports whether that retriever honours
:class:`~aieng.advanced_rag.retrieval.interface.Retriever`. It is library code
rather than a test, so a reference implementation runs the same checks over its
own method with its own corpus and its own question:

>>> from aieng.advanced_rag.retrieval.conformance import check_retrieval
>>> report = check_retrieval(  # doctest: +SKIP
...     my_retriever, "which law governs?", chunks=my_chunks
... )
>>> report.ok  # doctest: +SKIP
True

What it checks is shape and provenance. What it deliberately does not check is
order, listed in :data:`NOT_CHECKED`. A rule that results arrive in descending
score, with ties broken on identifier, would rewrite the negative control: on the
committed 20-contract CUAD slice that moves its retrieval-failure rate from
0.4800 to 0.5400 and its mean reciprocal rank from 0.1895 to 0.3083. It would
also fail two graph adapters outright, because neither reports a per-passage
score.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..graphrag.evidence import Evidence
from .interface import (
    CAPABILITIES,
    Candidate,
    Hints,
    Retriever,
    ScopeUnsupported,
    declares,
)


#: What a conforming implementation is checked for, in the order checked.
CHECKS = (
    "declares the interface",
    "returns a list of evidence",
    "k bounds the count",
    "every passage carries text",
    "every passage carries provenance",
    "every passage names the method that produced it",
    "two identical calls agree",
    "an empty hint changes nothing",
    "a scope either filters or raises ScopeUnsupported",
    "the callable shape agrees with retrieve",
    "each declared capability returns its own type",
)

#: What this kit does not check, and will not.
NOT_CHECKED = (
    "the order passages arrive in, beyond two calls agreeing",
    "that scores descend, which two adapters cannot report at all",
    "that k counts the same unit everywhere: it bounds passages on every "
    "adapter, but on the LightRAG one it is the entity and relationship budget "
    "as well, so that arm can return fewer than k",
)


@dataclass(frozen=True)
class ConformanceReport:
    """What one conformance run found.

    Parameters
    ----------
    retriever : str
        The retriever's own name, or its class name when it has none.
    failures : tuple of str
        One entry per check that failed, naming what was expected. Empty for a
        conforming retriever. :data:`CHECKS` lists what was looked for.
    """

    retriever: str
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether every check passed."""
        return not self.failures

    def summary(self) -> str:
        """Render the run as one line, with every failure listed after it.

        Returns
        -------
        str
            A line naming the retriever, then one line per failure.
        """
        if not self.failures:
            return f"{self.retriever}: conforms"
        head = f"{self.retriever}: {len(self.failures)} conformance failures"
        return "\n".join([head, *(f"  - {failure}" for failure in self.failures)])

    def raise_for_failures(self) -> None:
        """Raise when any check failed, so a test can call this directly.

        Raises
        ------
        AssertionError
            If this report holds any failure.
        """
        if self.failures:
            raise AssertionError(self.summary())


def check_retrieval(
    retriever: Any,
    question: str,
    *,
    chunks: Sequence[Any] | None = None,
    k: int = 5,
    scope: str | None = None,
    capabilities: Sequence[str] | None = None,
) -> ConformanceReport:
    """Index one retriever, then check it against the interface synchronously.

    Indexing is part of the check: a retriever that declares the ``index``
    capability is handed ``chunks`` before anything is retrieved, so the corpus
    the checks see is the one they scored.

    Parameters
    ----------
    retriever : Any
        The retriever to check. It is indexed first when ``chunks`` is given.
    question : str
        A question the corpus can answer, so the checks see real passages.
    chunks : sequence or None, optional
        The corpus to index, for a retriever that declares the ``index``
        capability. ``None`` checks a retriever that is already indexed.
    k : int, optional
        Depth the count checks use.
    scope : str or None, optional
        A scope to check the filter with. ``None`` skips the scope check.
    capabilities : sequence of str or None, optional
        Which optional capabilities to check, from ``index``, ``find``,
        ``fetch``, ``expand``, ``read``, and ``grep``. ``None`` checks every one
        the retriever declares.

    Returns
    -------
    ConformanceReport
        The failures, empty when the retriever conforms.
    """
    _index(retriever, chunks)
    results = {
        "retrieve": _call(retriever.retrieve, question, k),
        "repeat": _call(retriever.retrieve, question, k),
        "hinted": _call(retriever.retrieve, question, k, hints=Hints()),
        "at k=1": _call(retriever.retrieve, question, 1),
    }
    if scope is not None:
        results["scoped"] = _call(retriever.retrieve, question, k, scope=scope)
    failures = _judge(retriever, results, question, k, scope, capabilities)
    return ConformanceReport(_label(retriever), tuple(failures))


async def acheck_retrieval(
    retriever: Any,
    question: str,
    *,
    chunks: Sequence[Any] | None = None,
    k: int = 5,
    scope: str | None = None,
    capabilities: Sequence[str] | None = None,
) -> ConformanceReport:
    """Index one retriever, then check it through ``aretrieve``.

    Indexing is part of the check, as in :func:`check_retrieval`. This is the
    entry point for a method whose asynchronous twin is the real one,
    such as the LightRAG adapter, whose synchronous ``retrieve`` refuses to run
    inside an event loop.

    Parameters
    ----------
    retriever : Any
        The retriever to check. It is indexed first when ``chunks`` is given.
    question : str
        A question the corpus can answer, so the checks see real passages.
    chunks : sequence or None, optional
        The corpus to index, for a retriever that declares ``index``.
    k : int, optional
        Depth the count checks use.
    scope : str or None, optional
        A scope to check the filter with. ``None`` skips the scope check.
    capabilities : sequence of str or None, optional
        Which optional capabilities to check. ``None`` checks every one the
        retriever declares.

    Returns
    -------
    ConformanceReport
        The failures, empty when the retriever conforms.
    """
    _index(retriever, chunks)
    results = {
        "aretrieve": await _acall(retriever.aretrieve, question, k),
        "repeat": await _acall(retriever.aretrieve, question, k),
        "hinted": await _acall(retriever.aretrieve, question, k, hints=Hints()),
        "at k=1": await _acall(retriever.aretrieve, question, 1),
    }
    if scope is not None:
        results["scoped"] = await _acall(retriever.aretrieve, question, k, scope=scope)
    failures = _judge(
        retriever, results, question, k, scope, capabilities, callable_shape=False
    )
    return ConformanceReport(_label(retriever), tuple(failures))


def _label(retriever: Any) -> str:
    return str(getattr(retriever, "name", "") or type(retriever).__name__)


def _index(retriever: Any, chunks: Sequence[Any] | None) -> None:
    if chunks is not None and declares(retriever, "index"):
        retriever.index(chunks)


def _call(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Return what the call returned, or the exception it raised."""
    try:
        return method(*args, **kwargs)
    except Exception as error:  # the kit reports a raise, it does not propagate
        return error


async def _acall(method: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return await method(*args, **kwargs)
    except Exception as error:
        return error


def _judge(
    retriever: Any,
    results: dict[str, Any],
    question: str,
    k: int,
    scope: str | None,
    capabilities: Sequence[str] | None,
    callable_shape: bool = True,
) -> list[str]:
    """Collect every failure across the gathered results."""
    failures: list[str] = []
    if not isinstance(retriever, Retriever):
        failures.append(
            "does not declare the interface: it needs a name, retrieve, and aretrieve"
        )

    # The first gathered result is the plain call every other check compares to.
    primary = results[next(iter(results))]
    for label, outcome in results.items():
        if label == "scoped":
            continue
        failures.extend(_judge_passages(label, outcome, 1 if label == "at k=1" else k))

    failures.extend(_judge_agreement(results))
    if scope is not None:
        failures.extend(_judge_scope(results["scoped"], scope))
    if callable_shape:
        failures.extend(_judge_callable(retriever, question, scope, primary))
    failures.extend(_judge_capabilities(retriever, question, primary, capabilities))
    return failures


def _judge_passages(label: str, outcome: Any, k: int | None) -> list[str]:
    if isinstance(outcome, Exception):
        return [f"{label} raised {type(outcome).__name__}: {outcome}"]
    if not isinstance(outcome, list):
        return [f"{label} returned {type(outcome).__name__}, not a list"]

    failures: list[str] = []
    if k is not None and len(outcome) > k:
        failures.append(f"{label} returned {len(outcome)} passages for k={k}")
    for position, item in enumerate(outcome):
        if not isinstance(item, Evidence):
            failures.append(
                f"{label} returned {type(item).__name__} at {position}, not Evidence"
            )
            continue
        if not item.text:
            failures.append(f"{label} returned an empty passage at {position}")
        if not item.provenance.doc_id:
            failures.append(f"{label} returned no document id at {position}")
        if not item.retriever:
            failures.append(f"{label} named no retriever at {position}")
    return failures


def _judge_agreement(results: dict[str, Any]) -> list[str]:
    primary = results[next(iter(results))]
    if isinstance(primary, Exception) or not isinstance(primary, list):
        return []

    failures: list[str] = []
    expected = _ids(primary)
    if _comparable(results["repeat"]) and _ids(results["repeat"]) != expected:
        failures.append("two identical calls returned different passages")
    if _comparable(results["hinted"]) and _ids(results["hinted"]) != expected:
        failures.append(
            "an empty Hints() changed the ranking: a hint an implementation "
            "ignores has to rank exactly as hints=None does"
        )
    return failures


def _judge_scope(outcome: Any, scope: str) -> list[str]:
    if isinstance(outcome, ScopeUnsupported):
        return []
    if isinstance(outcome, Exception):
        return [
            f"a scope raised {type(outcome).__name__} rather than "
            f"ScopeUnsupported: {outcome}"
        ]
    stray = {item.doc_id for item in outcome} - {scope}
    if stray:
        return [f"a scope of {scope!r} returned passages from {sorted(stray)}"]
    return []


def _judge_callable(
    retriever: Any, question: str, scope: str | None, primary: Any
) -> list[str]:
    if not callable(retriever):
        return ["is not callable, so it cannot stand in for pipeline.spine.Retrieve"]
    called = _call(retriever, question, scope)
    if isinstance(called, ScopeUnsupported) and scope is not None:
        return []
    if isinstance(called, Exception):
        return [f"the callable shape raised {type(called).__name__}: {called}"]
    deep = _call(retriever.retrieve, question, 10, scope=scope)
    if _comparable(deep) and _ids(list(called)) != _ids(deep):
        return ["the callable shape disagrees with retrieve at its own default k"]
    return []


def _judge_capabilities(
    retriever: Any,
    question: str,
    primary: Any,
    capabilities: Sequence[str] | None,
) -> list[str]:
    wanted = CAPABILITIES if capabilities is None else tuple(capabilities)
    unknown = sorted(set(wanted) - set(CAPABILITIES))
    if unknown:
        return [f"asked for capabilities this kit does not know: {unknown}"]
    if not _comparable(primary) or not primary:
        return []

    first = primary[0]
    chunk_id = first.chunk_id or first.doc_id
    failures: list[str] = []
    if "find" in wanted and declares(retriever, "find"):
        failures.extend(_judge_candidates("find", _call(retriever.find, question, 2)))
    if "fetch" in wanted and declares(retriever, "fetch"):
        failures.extend(_judge_passages("fetch", _call(retriever.fetch, [chunk_id]), 1))
    if "expand" in wanted and declares(retriever, "expand"):
        failures.extend(
            _judge_passages("expand", _call(retriever.expand, [chunk_id]), None)
        )
    if "read" in wanted and declares(retriever, "read"):
        failures.extend(_judge_read(retriever, first.doc_id))
    if "grep" in wanted and declares(retriever, "grep"):
        pattern = re.escape(first.text.strip()[:20])
        failures.extend(_judge_candidates("grep", _call(retriever.grep, pattern, 2)))
    return failures


def _judge_read(retriever: Any, doc_id: str) -> list[str]:
    failures: list[str] = []
    for label, outcome in (
        ("read", _call(retriever.read, doc_id, start=0, end=40)),
        ("read_lines", _call(retriever.read_lines, doc_id, first=1, last=1)),
    ):
        if isinstance(outcome, Exception):
            failures.append(f"{label} raised {type(outcome).__name__}: {outcome}")
        elif not isinstance(outcome, Evidence):
            failures.append(
                f"{label} returned {type(outcome).__name__}, not one Evidence"
            )
        elif outcome.provenance.doc_id != doc_id:
            failures.append(f"{label} reported the wrong document id")
    return failures


def _judge_candidates(label: str, outcome: Any) -> list[str]:
    if isinstance(outcome, Exception):
        return [f"{label} raised {type(outcome).__name__}: {outcome}"]
    if not isinstance(outcome, list):
        return [f"{label} returned {type(outcome).__name__}, not a list"]
    return [
        f"{label} returned {type(item).__name__} at {position}, not Candidate"
        for position, item in enumerate(outcome)
        if not isinstance(item, Candidate)
    ]


def _comparable(outcome: Any) -> bool:
    """Return whether a gathered result is a list rather than a raised error."""
    return isinstance(outcome, list)


def _ids(evidence: list[Evidence]) -> list[str]:
    return [item.chunk_id or item.doc_id for item in evidence]


def assert_conforms(
    retriever: Any,
    question: str,
    *,
    chunks: Sequence[Any] | None = None,
    k: int = 5,
    scope: str | None = None,
    capabilities: Sequence[str] | None = None,
) -> None:
    """Check a retriever through both entry points and raise on any failure.

    The one-call form for a test, named for the raise: a conforming retriever
    returns nothing and a failing one raises. Indexes the retriever when
    ``chunks`` is given. Call it from outside an event loop, because it drives
    ``aretrieve`` with :func:`asyncio.run`.

    Parameters
    ----------
    retriever : Any
        The retriever to check.
    question : str
        A question the corpus can answer.
    chunks : sequence or None, optional
        The corpus to index.
    k : int, optional
        Depth the count checks use.
    scope : str or None, optional
        A scope to check the filter with.
    capabilities : sequence of str or None, optional
        Which optional capabilities to check.

    Raises
    ------
    AssertionError
        If any check failed, with every failure named.
    """
    report = check_retrieval(
        retriever,
        question,
        chunks=chunks,
        k=k,
        scope=scope,
        capabilities=capabilities,
    )
    report.raise_for_failures()
    asyncio.run(
        acheck_retrieval(
            retriever,
            question,
            k=k,
            scope=scope,
            capabilities=capabilities,
        )
    ).raise_for_failures()


__all__ = [
    "CHECKS",
    "NOT_CHECKED",
    "ConformanceReport",
    "acheck_retrieval",
    "assert_conforms",
    "check_retrieval",
]

#: Deprecated alias for :func:`assert_conforms`, kept because the reference
#: implementations call it under the old name. Deprecated in 0.1.1; use
#: ``assert_conforms``, which says that it raises.
run_checks = assert_conforms
