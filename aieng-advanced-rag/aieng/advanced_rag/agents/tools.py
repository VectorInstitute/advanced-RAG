"""Plain Python tools an agent can call, with no framework in the signature.

A tool here is an ordinary function: annotated parameters, a docstring, and a
JSON-friendly return value. Nothing in this module imports an agent framework
and no function carries a decorator, because the framework choice has to stay
reversible: the same functions go to a different runtime, or to a direct call in
a test, without being rewritten.

ADK reads a bare callable's signature and docstring itself and wraps it in its
own ``FunctionTool``, so the wrapping belongs at the boundary in
:mod:`aieng.advanced_rag.agents.toolset` rather than on the function.

Two things a tool function has to keep. Every parameter carries an annotation,
because that annotation is what becomes the declared JSON schema, and the
docstring is what the model reads to decide when to call it. Deferred
annotations are fine: both ADK and ``google-genai`` resolve them with
``typing.get_type_hints``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

from ..retrieval.interface import Fetchable, Retriever, declares


#: A tool function: annotated arguments in, a JSON-friendly mapping out. The
#: mapping is what reaches the model, so it holds text rather than objects.
ToolFunction: TypeAlias = Callable[..., dict[str, Any]]


def as_passage(evidence: Any) -> dict[str, Any]:
    """Render one piece of evidence as the mapping a model reads.

    Parameters
    ----------
    evidence : Evidence
        A retrieved passage.

    Returns
    -------
    dict
        ``text``, ``citation``, ``chunk_id``, and ``score``. The citation
        travels with the text so an answer can cite without a second lookup,
        and the chunk identifier is what a follow-up fetch asks for.
    """
    return {
        "text": evidence.text,
        "citation": evidence.cite(),
        "chunk_id": evidence.chunk_id,
        "score": evidence.score,
    }


def retrieval_tools(
    retriever: Retriever,
    k: int = 10,
    *,
    scope: str | None = None,
) -> list[ToolFunction]:
    """Return search tools over one retriever, as plain functions.

    ``k`` and ``scope`` are bound here rather than declared as tool arguments.
    The scope is the corpus's own filter, and a model that could widen it would
    be able to answer a per-contract question from another contract, so the
    caller sets it once and the model never sees it.

    The retriever is called synchronously inside an asynchronous tool, which
    holds the invocation's task until it returns. That is the right trade for a
    loop that issues its model calls one at a time; concurrency comes from
    running several invocations, which is also where the proxy's ceiling of 8
    concurrent requests applies.

    Parameters
    ----------
    retriever : Retriever
        Anything satisfying the shared retrieval interface.
    k : int, optional
        Passages per search.
    scope : str or None, optional
        The corpus's own filter, passed through on every search. ``None`` means
        the whole corpus.

    Returns
    -------
    list of ToolFunction
        ``search_corpus`` always, and ``fetch_passages`` when the retriever
        declares the fetch capability.

    Examples
    --------
    >>> tools = retrieval_tools(retriever, k=5)  # doctest: +SKIP
    >>> [tool.__name__ for tool in tools]  # doctest: +SKIP
    ['search_corpus', 'fetch_passages']
    """

    def search_corpus(question: str) -> dict[str, Any]:
        """Search the corpus and return the passages that answer a question.

        Parameters
        ----------
        question : str
            What to look for, in the words the user used.

        Returns
        -------
        dict
            ``passages``, each carrying its text and its citation.
        """
        found = retriever.retrieve(question, k, scope=scope)
        return {"passages": [as_passage(item) for item in found]}

    tools: list[ToolFunction] = [search_corpus]
    if declares(retriever, "fetch"):
        # ``declares`` checks at runtime what the cast states to the checker: the
        # capability protocols type an argument and no longer answer isinstance,
        # so they narrow nothing.
        tools.append(_fetch_tool(cast("Fetchable", retriever)))
    return tools


def _fetch_tool(retriever: Fetchable) -> ToolFunction:
    """Return the fetch tool bound to one retriever.

    Parameters
    ----------
    retriever : Fetchable
        A retriever that can return named passages in full.

    Returns
    -------
    ToolFunction
        ``fetch_passages``.
    """

    def fetch_passages(chunk_ids: list[str]) -> dict[str, Any]:
        """Return named passages in full, for identifiers a search returned.

        Parameters
        ----------
        chunk_ids : list of str
            Identifiers taken from an earlier search result.

        Returns
        -------
        dict
            ``passages``, in the order asked, with one entry per identifier the
            corpus holds.
        """
        found: Sequence[Any] = retriever.fetch(chunk_ids)
        return {"passages": [as_passage(item) for item in found]}

    return fetch_passages


__all__ = ["ToolFunction", "as_passage", "retrieval_tools"]
