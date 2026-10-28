"""Steps for the synthetic question sets special topic, cached or live.

Two model-produced steps run through
:func:`aieng.advanced_rag.evaluation.synthetic.replay.load_or_build`:

- ``validation`` builds a single-hop set with nulls over the first ContractNLI
  test documents and scores RI #1's eight retrieval arms under ContractNLI's
  human labels and under the generated ones. The reference covers 28
  documents on Vector's on-prem models; a live run on your models covers
  :data:`LIVE_DOCUMENTS`, so it fits in hands-on notebook time.
- ``explore-adjacent-statements`` writes one question from two statements of
  the same passage, so each question has two gold spans, and scores the arms
  again. It is a single run on the on-prem models only.

Both reference artifacts sit in ``outputs/baseline_cache``, the shipped tier of
the cache, so a fresh workspace reads them with no model call.
``build_reference.py`` writes them.

Prior art and why this design: RAGAS's test-set generator, DeepEval's
Synthesizer, and LlamaIndex's ``generate_question_context_pairs`` all store
gold as context text or a chunk id, so their labels belong to one chunker, and
none writes in-domain null questions. ARES keeps a question only when one
retriever finds its passage, which tilts a comparison towards that retriever.
DataMorgana's weighted user and question categories are reused as
:class:`~aieng.advanced_rag.evaluation.synthetic.GenerationConfig`'s
categories; it is a hosted service, so it is not wrapped. The package writes
character spans (``Know Your RAG``'s statement-first design), so one set
scores every chunker in RI #1, and exports BEIR qrels per chunking for tools
that read them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.evaluation.synthetic import (
    Document,
    SyntheticSet,
    build_single_hop_set,
    proxy_model,
)
from aieng.advanced_rag.evaluation.synthetic.replay import (
    REFERENCE_ROLES,
    Reference,
    Roles,
    load_or_build,
    set_from_value,
    set_step,
    set_value,
)
from aieng.advanced_rag.evaluation.synthetic.validation import (
    CONTRACT_NLI_CONFIG,
    DENSE_MODEL,
    RERANKER,
    load_contract_nli,
    validate,
)
from aieng.advanced_rag.models import ModelChoice, ModelSet
from aieng.advanced_rag.retrieval.corpus import Query
from aieng.advanced_rag.runs import Step, StepResult


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
BASELINE = OUTPUTS / "baseline_cache"
RI_ID = "synthetic-qa"

#: Documents the reference validation run covers.
REFERENCE_DOCUMENTS = 28
#: Documents a live validation run covers by default: about a quarter of the
#: reference's calls.
LIVE_DOCUMENTS = 8
#: Requests per minute per model. Other jobs share the proxy's per-model caps.
PER_MINUTE = 15


def responses_path() -> Path:
    """Return the shared response cache, so a re-run of any build is free."""
    return Path(cache_root()) / "synthetic" / "responses.sqlite"


def shipped_cache() -> TieredResultCache:
    """Return the cache whose shipped tier holds this topic's reference artifacts."""
    return TieredResultCache(global_dir=BASELINE)


def contract_nli(count: int) -> tuple[list[Document], dict[str, list[Query]]]:
    """Return the first ``count`` ContractNLI test documents and their real queries."""
    loaded: tuple[list[Document], dict[str, list[Query]]] = load_contract_nli(count)
    return loaded


def models_for(roles: Roles, *, per_minute: int = PER_MINUTE, env_path: Path | None = None) -> dict[str, Any]:
    """Return the three cached, throttled proxy models a build takes."""
    return {
        role: proxy_model(name, cache_path=responses_path(), per_minute=per_minute, env_path=env_path)
        for role, name in roles.as_dict().items()
    }


def _arms_extra() -> dict[str, Any]:
    return {"embedder": DENSE_MODEL, "reranker": RERANKER, "arms": "ri1-eight"}


def validation_step(documents: Sequence[Document], roles: Roles) -> Step:
    """Return the validation step for these documents and roles."""
    return set_step(
        RI_ID,
        "evaluation.synthetic.validation",
        documents,
        CONTRACT_NLI_CONFIG,
        roles,
        artifact_type="validation",
        extra=_arms_extra(),
    )


def validation_compute(
    documents: Sequence[Document],
    real_by_doc: dict[str, list[Query]],
    roles: Roles,
    *,
    per_minute: int = PER_MINUTE,
    env_path: Path | None = None,
) -> Callable[[ModelChoice], dict[str, Any]]:
    """Return the compute ``load_or_run`` calls for a validation build."""

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
            CachedQueryEmbedder,
            SentenceTransformerEmbedder,
        )

        embedder = SentenceTransformerEmbedder(DENSE_MODEL)
        generated = build_single_hop_set(
            documents,
            CONTRACT_NLI_CONFIG,
            embedder=embedder,
            **models_for(roles, per_minute=per_minute, env_path=env_path),
        )
        scored = validate(generated, documents, real_by_doc, CachedQueryEmbedder(embedder))
        return {**set_value(generated), "validation": scored}

    return compute


def load_validation(
    *,
    models: ModelSet | None = None,
    roles: Roles = REFERENCE_ROLES,
    documents: int = REFERENCE_DOCUMENTS,
    live: bool = False,
    cache: TieredResultCache | None = None,
    env_path: Path | None = None,
) -> tuple[StepResult, SyntheticSet, dict[str, Any]]:
    """Return the validation run for ``roles`` on ``documents`` documents.

    With ``live`` off and no run of yours cached, this returns the 28-document
    reference and prints what your build would cost.

    Returns
    -------
    tuple
        The step result, the generated set, and the validation record.
    """
    docs, real = contract_nli(documents)
    step = validation_step(docs, roles)
    reference: Reference | None = None
    if roles != REFERENCE_ROLES or documents != REFERENCE_DOCUMENTS:
        ref_docs, _ = contract_nli(REFERENCE_DOCUMENTS)
        reference = Reference(
            validation_step(ref_docs, REFERENCE_ROLES),
            scale=documents / REFERENCE_DOCUMENTS,
            description=(f"(the reference covers {REFERENCE_DOCUMENTS} documents; your build covers {documents})"),
        )
    result = load_or_build(
        step,
        validation_compute(docs, real, roles, env_path=env_path),
        roles=roles,
        models=models,
        live=live,
        cache=cache or shipped_cache(),
        reference=reference,
    )
    return result, set_from_value(result.value), result.value["validation"]


def explore_step(documents: Sequence[Document], roles: Roles) -> Step:
    """Return the explore step: two statements per question, scored on the arms."""
    import hashlib  # noqa: PLC0415

    from explore_adjacent_statements import PAIR_PROMPT  # noqa: PLC0415

    return set_step(
        RI_ID,
        "explore_adjacent_statements.py",
        documents,
        CONTRACT_NLI_CONFIG,
        roles,
        artifact_type="explore-adjacent-statements",
        extra={
            **_arms_extra(),
            "pair_prompt": hashlib.sha256(PAIR_PROMPT.encode()).hexdigest()[:16],
        },
    )


def load_explore(*, cache: TieredResultCache | None = None, live: bool = False) -> StepResult:
    """Return the single run on the reference models, cached or rebuilt."""
    from explore_adjacent_statements import explore_compute  # noqa: PLC0415

    docs, real = contract_nli(REFERENCE_DOCUMENTS)
    return load_or_build(
        explore_step(docs, REFERENCE_ROLES),
        explore_compute(docs, real, REFERENCE_ROLES),
        roles=REFERENCE_ROLES,
        live=live,
        cache=cache or shipped_cache(),
    )


def ranks(means: dict[str, float]) -> dict[str, int]:
    """Return arm to rank, 1 for the highest mean."""
    order = sorted(means, key=lambda arm: (-means[arm], arm))
    return {arm: index + 1 for index, arm in enumerate(order)}


def arm_table(record: dict[str, Any], metric: str = "ndcg@10") -> list[dict[str, Any]]:
    """Return one row per arm: real and generated mean and rank, for one metric."""
    rows = []
    for variant in ("plain", "paraphrase"):
        if variant not in record["agreement"]:
            continue
        compared = record["agreement"][variant]["all_arms"][metric]
        real, generated = compared["real_mean"], compared["generated_mean"]
        real_rank, generated_rank = ranks(real), ranks(generated)
        for arm in sorted(real, key=lambda a: real_rank[a]):
            rows.append(
                {
                    "variant": variant,
                    "arm": arm,
                    "real": real[arm],
                    "real_rank": real_rank[arm],
                    "generated": generated[arm],
                    "generated_rank": generated_rank[arm],
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> None:
    """Write a result file with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


__all__ = [
    "BASELINE",
    "LIVE_DOCUMENTS",
    "PER_MINUTE",
    "REFERENCE_DOCUMENTS",
    "arm_table",
    "contract_nli",
    "explore_step",
    "load_explore",
    "load_validation",
    "models_for",
    "ranks",
    "responses_path",
    "shipped_cache",
    "validation_compute",
    "validation_step",
]
