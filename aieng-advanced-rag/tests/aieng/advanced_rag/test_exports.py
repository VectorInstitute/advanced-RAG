"""The package's export surface.

What a module promises in ``__all__`` has to resolve, the documented way to
configure the spine has to be reachable from the package the spine lives in, and
a deprecated alias has to stay out of both, so ``import *`` never binds a retired
name.
"""

from __future__ import annotations

import importlib

import pytest


#: The modules that carry their own ``__all__``, by import path.
DECLARED = (
    "aieng.advanced_rag",
    "aieng.advanced_rag.agents",
    "aieng.advanced_rag.agents.loop",
    "aieng.advanced_rag.agents.model",
    "aieng.advanced_rag.agents.plugins",
    "aieng.advanced_rag.agents.records",
    "aieng.advanced_rag.agents.session_loop",
    "aieng.advanced_rag.agents.thinking",
    "aieng.advanced_rag.agents.tools",
    "aieng.advanced_rag.agents.usage",
    "aieng.advanced_rag.batch",
    "aieng.advanced_rag.cache",
    "aieng.advanced_rag.cache.paths",
    "aieng.advanced_rag.corpora",
    "aieng.advanced_rag.evaluation",
    "aieng.advanced_rag.graphrag",
    "aieng.advanced_rag.graphrag.lightrag_index",
    "aieng.advanced_rag.ingestion",
    "aieng.advanced_rag.ingestion.docling",
    "aieng.advanced_rag.ingestion.tables",
    "aieng.advanced_rag.metering",
    "aieng.advanced_rag.models",
    "aieng.advanced_rag.pipeline",
    "aieng.advanced_rag.pipeline.answering",
    "aieng.advanced_rag.proxy",
    "aieng.advanced_rag.retrieval",
    "aieng.advanced_rag.retrieval.conformance",
    "aieng.advanced_rag.retrieval.interface",
    "aieng.advanced_rag.retrieval.stores",
    "aieng.advanced_rag.runs",
    "aieng.advanced_rag.spend",
    # Declared but not re-exported from the package root: unlike the meter and
    # the cache, tracing needs a third-party dependency, so it stays a named
    # subpackage you reach for deliberately.
    "aieng.advanced_rag.tracing",
)

#: Every retired name, and the module it stays importable from.
ALIASES = (
    ("aieng.advanced_rag.retrieval.interface", "Retrieval", "Retriever"),
    ("aieng.advanced_rag.retrieval.interface", "RetrievalMixin", "RetrieverBase"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsIndex", "Indexable"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsFind", "Findable"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsFetch", "Fetchable"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsExpand", "Expandable"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsRead", "Readable"),
    ("aieng.advanced_rag.retrieval.interface", "SupportsGrep", "Greppable"),
    ("aieng.advanced_rag.retrieval.conformance", "run_checks", "assert_conforms"),
    ("aieng.advanced_rag.metering", "CostMeter", "UsageMeter"),
    ("aieng.advanced_rag.metering", "QuestionCost", "QuestionUsage"),
    ("aieng.advanced_rag.metering", "QuestionScope", "QuestionMeter"),
    ("aieng.advanced_rag.metering", "RunCost", "RunUsage"),
    ("aieng.advanced_rag.metering", "CompletionFacts", "ParsedCompletion"),
    ("aieng.advanced_rag.agents.usage", "EventFacts", "ParsedEvent"),
    ("aieng.advanced_rag.pipeline.answering", "Move", "LoopDecision"),
    ("aieng.advanced_rag.pipeline.answering", "Step", "RetrievalStep"),
    ("aieng.advanced_rag.pipeline.answering", "PromptSource", "PromptSelector"),
    ("aieng.advanced_rag.pipeline.answering", "FixedPrompt", "FixedPromptSelector"),
    (
        "aieng.advanced_rag.pipeline.answering",
        "MethodPrompt",
        "RetrieverPromptSelector",
    ),
    ("aieng.advanced_rag.pipeline.answering", "NoAnalysis", "PassthroughAnalysis"),
    ("aieng.advanced_rag.graphrag.protocol", "GraphRetriever", "Retriever"),
)

#: What configures `AdvancedRagSpine`, which has to be reachable beside it.
SPINE_CONTROLS = (
    "CorrectivePolicy",
    "DecompositionAnalysis",
    "FixedPromptSelector",
    "LoopPolicy",
    "PassthroughAnalysis",
    "PromptSelector",
    "QueryAnalysis",
    "RetrieverPromptSelector",
    "RewriterAnalysis",
    "SingleShotPolicy",
    "step_bound",
)


@pytest.mark.parametrize("name", DECLARED)
def test_every_declared_export_resolves(name):
    module = importlib.import_module(name)
    assert [entry for entry in module.__all__ if not hasattr(module, entry)] == []
    assert len(set(module.__all__)) == len(module.__all__)


@pytest.mark.parametrize(("module_name", "retired", "current"), ALIASES)
def test_a_retired_name_still_imports_and_stays_out_of_all(
    module_name, retired, current
):
    module = importlib.import_module(module_name)
    assert getattr(module, retired) is not None
    assert retired not in getattr(module, "__all__", ())


@pytest.mark.parametrize("name", SPINE_CONTROLS)
def test_the_spine_is_configurable_from_the_package_it_lives_in(name):
    pipeline = importlib.import_module("aieng.advanced_rag.pipeline")
    assert name in pipeline.__all__


def test_the_meter_is_reachable_from_the_package_root():
    root = importlib.import_module("aieng.advanced_rag")
    assert "metering" in root.__all__
    assert root.UsageMeter is root.metering.UsageMeter


def test_the_cache_is_reachable_from_the_package_root():
    root = importlib.import_module("aieng.advanced_rag")
    assert "cache" in root.__all__
    assert root.TieredResultCache is root.cache.TieredResultCache
    assert root.CacheKey is root.cache.CacheKey
