"""One toolset, wrapping plain Python functions for an ADK agent.

This is the whole framework boundary on the tool side, and it is deliberately
the only file in the package that knows a tool has to be declared to anything.
``FunctionTool`` reads a callable's signature and docstring and builds the
declaration itself, so the functions in :mod:`aieng.advanced_rag.agents.tools`
carry no decorator and no import of ADK. Swapping the framework means replacing
this file.

A toolset rather than a bare list of callables, because ``LlmAgent`` accepts
either and a toolset is what lets one object own a set of tools that belong
together, along with the ``tool_filter`` and ``tool_name_prefix`` the base class
already implements.

Reach this module by path. It needs the ``agents`` extra, so the package's
``__init__`` does not import it, the way ``retrieval`` leaves out the modules
that need a model.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

from ..graphrag.protocol import MissingExtraError
from .tools import ToolFunction


try:
    from google.adk.tools.base_toolset import BaseToolset
    from google.adk.tools.function_tool import FunctionTool
except ImportError as error:  # pragma: no cover - import guard
    raise MissingExtraError("agents", "google-adk") from error


class FunctionToolset(BaseToolset):
    """Plain Python functions, offered to an agent as tools.

    Named for the family it sits in: ADK ships ``FunctionTool``, ``MCPToolset``,
    and ``LangchainTool``, and "Python" says nothing inside a Python library.

    Parameters
    ----------
    functions : iterable of ToolFunction
        The functions to expose. Each needs annotated parameters and a
        docstring, because those become the declared schema and the description
        the model reads.
    tool_filter : ToolPredicate or list of str, optional
        Narrows what an invocation sees, handled by the base class.
    tool_name_prefix : str or None, optional
        Prepended to every tool name, handled by the base class. Use it when two
        toolsets would otherwise offer the same name.

    Examples
    --------
    >>> toolset = FunctionToolset(retrieval_tools(retriever))  # doctest: +SKIP
    >>> agent = LlmAgent(name="rag", model=model, tools=[toolset])  # doctest: +SKIP
    """

    def __init__(
        self,
        functions: Iterable[ToolFunction],
        *,
        tool_filter: Any = None,
        tool_name_prefix: str | None = None,
    ) -> None:
        super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)
        self._functions = tuple(functions)

    async def get_tools(self, readonly_context: Any = None) -> list[Any]:
        """Return one tool per function, wrapped for this framework.

        Parameters
        ----------
        readonly_context : ReadonlyContext or None, optional
            The invocation's context, which this toolset does not read: the set
            of tools is fixed when the toolset is built.

        Returns
        -------
        list of BaseTool
            One ``FunctionTool`` per function, in the order given.
        """
        return [FunctionTool(func=function) for function in self._functions]


def tool_declarations(functions: Iterable[ToolFunction]) -> list[dict[str, Any]]:
    """Return the tool declarations an agent sends for these functions.

    The declarations are what ``FunctionTool`` builds from each function's
    signature and docstring, so a digest of them changes exactly when the
    request a model sees changes. ADK exposes them only through
    ``FunctionTool._get_declaration``, which also warns that the JSON-schema
    declaration feature is experimental; this function is the one place that
    reaches in, and it keeps the warning out of the caller's output.

    Parameters
    ----------
    functions : iterable of ToolFunction
        The functions, in the order the agent offers them.

    Returns
    -------
    list of dict
        One ``FunctionDeclaration`` per function, dumped with
        ``mode="json"`` and without empty fields.
    """
    declarations: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for function in functions:
            declaration = FunctionTool(func=function)._get_declaration()
            if declaration is not None:
                declarations.append(
                    declaration.model_dump(mode="json", exclude_none=True)
                )
    return declarations


__all__ = ["FunctionToolset", "tool_declarations"]

#: Deprecated alias for :class:`FunctionToolset`. Deprecated in 0.1.1; use
#: ``FunctionToolset``, which matches ADK's own ``FunctionTool``.
PythonToolset = FunctionToolset
