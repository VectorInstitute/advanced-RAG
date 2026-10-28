"""Shared library for the Advanced RAG Bootcamp.

``retrieval`` and ``evaluation`` are the shared evaluation harness: chunkers,
embedders, retrievers, and the metrics every reference implementation reports.
``graphrag`` is the boundary a graph method implements. Import the subpackages
by name; nothing heavy is imported here.

``metering`` is the exception, re-exported here because it is not an add-on:
``pipeline.generate`` imports it, so every pipeline run reports through it, and
the module itself imports nothing beyond the standard library. One question opens
a :class:`~aieng.advanced_rag.metering.QuestionMeter`, and every model call made
inside it is charged to that question.

``cache`` is here on the same terms. It is not an add-on either: every hit it
resolves is reported to the meter, so a warm re-run stays comparable with the
cold run it repeats, and it imports nothing beyond the standard library.
:class:`~aieng.advanced_rag.cache.TieredResultCache` and
:class:`~aieng.advanced_rag.cache.CacheKey` are the two names a caller needs.

``models``, ``spend``, and ``runs`` are the model-access layer, standard library
only as well: :func:`~aieng.advanced_rag.models.choose_models` picks one model
per role for a provider family, :mod:`~aieng.advanced_rag.spend` prices what the
meter counted and reads the key's budget, and
:func:`~aieng.advanced_rag.runs.load_or_run` returns your cached result, the
labelled reference artifact, or a live run on your model.
"""

import os


# litellm calls ``load_dotenv()`` at import unless LITELLM_MODE is set, and that
# walks up from its install directory past the repository root, so a ``.env`` in
# a parent or home directory would load its secrets into the kernel. Keys come
# from the environment or the repository's ``.env`` through ``proxy`` instead.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from aieng.advanced_rag import cache, metering, models, runs, spend  # noqa: E402
from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.example_impl import example_impl
from aieng.advanced_rag.metering import (
    CallUsage,
    MeteredClient,
    MeteredCompleter,
    QuestionMeter,
    QuestionUsage,
    RunUsage,
    UsageMeter,
    default_meter,
    meter_client,
    using,
)
from aieng.advanced_rag.models import ModelSet, choose_models
from aieng.advanced_rag.runs import Step, load_or_run


__all__ = [
    "CacheKey",
    "CallUsage",
    "MeteredClient",
    "MeteredCompleter",
    "ModelSet",
    "QuestionMeter",
    "QuestionUsage",
    "RunUsage",
    "Step",
    "TieredResultCache",
    "UsageMeter",
    "cache",
    "choose_models",
    "default_meter",
    "example_impl",
    "load_or_run",
    "meter_client",
    "metering",
    "models",
    "runs",
    "spend",
    "using",
]
