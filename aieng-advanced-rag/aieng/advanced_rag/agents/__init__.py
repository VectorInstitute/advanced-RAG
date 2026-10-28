"""An agent loop for the bootcamp, with the framework kept at one boundary.

The tools are plain Python. A function with annotated parameters, a docstring,
and a JSON-friendly return value is all a tool is here, and the framework's
wrapper goes on at the edge, because the register records that the agent
framework has to stay replaceable. These modules touch ADK:
``agents.toolset`` declares the tools, ``agents.loop`` runs the turns,
``agents.model`` is the governed model client, ``agents.plugins`` caches
responses and holds a question to its budget, and ``agents.session_loop`` runs
a corpus tool session and returns an ``AgentRun``.

Those need the ``agents`` extra, so this ``__init__`` does not import them,
the way ``retrieval`` leaves out the modules that need a model. Reach them by
path once ``pip install 'aieng-advanced-rag[agents]'`` has run:

    from aieng.advanced_rag.agents.loop import AgentLoop
    from aieng.advanced_rag.agents.tools import retrieval_tools

    loop = AgentLoop(retrieval_tools(retriever))
    with meter.question("q1"):
        answer = await loop.arun("What notice does termination require?")

Seven modules are importable with nothing installed: ``agents.records``
holds the per-question ``AgentRun`` record, ``agents.tools`` builds
the functions, ``agents.corpus_tools`` is the tool server (``search``, ``grep``,
and ``open`` over a corpus, with every call logged), ``agents.profile`` turns a
short description of a corpus (identifiers, tables, sections, code) into more
tools for that server, ``agents.thinking`` holds
the reasoning switch and why it travels where it does, and ``agents.usage``
turns one ADK event into the figures :mod:`aieng.advanced_rag.metering` records.
A YAML profile needs PyYAML, which the bootcamp environment has; a JSON profile
needs nothing.
"""

from .corpus_tools import (
    STANDARD_ARMS,
    ArmConfig,
    Corpus,
    CorpusDocument,
    CorpusToolServer,
    DenseSearch,
    RegexGrep,
    RipgrepGrep,
    ToolKind,
    ToolSession,
    ToolTrace,
)
from .profile import DataProfile, ProfileError, ProfileTools, build_server, load_profile
from .thinking import thinking_extra_body
from .tools import ToolFunction, as_passage, retrieval_tools
from .usage import ParsedEvent, read_event, report_event, sum_usage, usage_of_event


__all__ = [
    "STANDARD_ARMS",
    "ArmConfig",
    "Corpus",
    "CorpusDocument",
    "CorpusToolServer",
    "DataProfile",
    "DenseSearch",
    "ParsedEvent",
    "ProfileError",
    "ProfileTools",
    "RegexGrep",
    "RipgrepGrep",
    "ToolKind",
    "ToolSession",
    "ToolTrace",
    "ToolFunction",
    "as_passage",
    "build_server",
    "load_profile",
    "read_event",
    "report_event",
    "retrieval_tools",
    "sum_usage",
    "thinking_extra_body",
    "usage_of_event",
]
