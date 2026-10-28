"""Bring your own corpus: generate a small question set, then score RI #1's arms on it.

``04_bring_your_own_corpus.ipynb`` calls three functions. :func:`read_corpus`
reads a folder, or writes ten ContractNLI test contracts to a local folder as
a stand-in. :func:`generate` builds a single-hop set with nulls through
``load_or_build``, so with ``RUN_EXPENSIVE`` off the stand-in folder reads the
reference set built on Vector's on-prem models. :func:`score` ranks the eight
arms of the ContractNLI validation on the set, from the cache or locally.

The stand-in's reference set and scores sit in ``outputs/baseline_cache``, the
shipped tier RI #1's first notebook also reads. Rebuild them from this folder:

    python byoc.py --env-file <env_file>

where ``<env_file>`` is a proxy ``.env`` fallback, needed only when onboarding
has not exported a key and a response is not cached.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache import CacheKey, TieredResultCache
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.evaluation.multihop import paired_bootstrap
from aieng.advanced_rag.evaluation.synthetic import (
    Document,
    GenerationConfig,
    SyntheticSet,
    build_single_hop_set,
    load_config,
    proxy_model,
)
from aieng.advanced_rag.evaluation.synthetic.replay import (
    REFERENCE_ROLES,
    Reference,
    Roles,
    load_folder,
    load_or_build,
    set_from_value,
    set_step,
    set_value,
)
from aieng.advanced_rag.evaluation.synthetic.validation import (
    CONTRACT_NLI_CONFIG,
    DENSE_MODEL,
    RERANKER,
    build_arms,
    load_contract_nli,
    mean_scores,
    queries_for,
    score_arms,
)
from aieng.advanced_rag.models import ModelChoice, ModelSet
from aieng.advanced_rag.runs import StepResult


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "outputs" / "baseline_cache"
RI_ID = "ri1"
PRODUCER = "regulatory_policy_qa/04_bring_your_own_corpus"

#: ContractNLI test contracts written to the stand-in folder.
DEMO_DOCUMENTS = 10
#: Requests per minute per model on the proxy.
PER_MINUTE = 15

#: Sizes for a set built in hands-on notebook time: two passages per document,
#: no paraphrase variant (the ContractNLI validation kept plain wording), and
#: two held-out documents for nulls.
SMALL: dict[str, Any] = {
    "passages_per_document": 2,
    "statements_per_passage": 2,
    "held_out_minimum": 2,
    "held_out_passages": 4,
    "paraphrase": False,
}

#: The stand-in corpus's configuration: the validation's, at the small sizes.
DEMO_CONFIG: GenerationConfig = replace(CONTRACT_NLI_CONFIG, **SMALL)

ARMS_KEY: dict[str, Any] = {
    "embedder": DENSE_MODEL,
    "reranker": RERANKER,
    "arms": "ri1-eight",
}


def demo_folder() -> Path:
    """Write ten ContractNLI test contracts as text files and return the folder."""
    folder = Path(cache_root()) / "byoc-demo" / "contract_nli"
    documents, _ = load_contract_nli(DEMO_DOCUMENTS)
    folder.mkdir(parents=True, exist_ok=True)
    for document in documents:
        path = folder / f"{document.doc_id}.txt"
        if not path.exists() or path.read_text(encoding="utf-8") != document.text:
            path.write_text(document.text, encoding="utf-8")
    return folder


def read_corpus(folder: Path | None, *, limit: int | None = None) -> tuple[list[Document], bool]:
    """Return the documents in ``folder``, or the stand-in contracts when it is None.

    Returns
    -------
    tuple
        The documents, and whether they are the stand-in.
    """
    if folder is None:
        return load_folder(demo_folder()), True
    return load_folder(Path(folder).expanduser(), limit=limit), False


def config_for(path: Path | None, demo: bool) -> GenerationConfig:
    """Return the build settings: your YAML file, or the stand-in's at the small sizes.

    A YAML file is used as written, so keep its sizes small for a first run.
    """
    if path is not None:
        return load_config(Path(path).expanduser())
    return DEMO_CONFIG if demo else replace(GenerationConfig(), **SMALL)


def _models(roles: Roles, env_path: Path | None) -> dict[str, Any]:
    cache = cache_root() / "synthetic" / "responses.sqlite"
    return {
        role: proxy_model(name, cache_path=cache, per_minute=PER_MINUTE, env_path=env_path)
        for role, name in roles.as_dict().items()
    }


def generate(
    documents: list[Document],
    config: GenerationConfig,
    models: ModelSet | None,
    *,
    live: bool,
    demo: bool,
    roles: Roles | None = None,
    cache: TieredResultCache | None = None,
    env_path: Path | None = None,
) -> tuple[StepResult, SyntheticSet]:
    """Return the set: yours, the stand-in's reference, or a live build.

    Parameters
    ----------
    documents : list of Document
        From :func:`read_corpus`.
    config : GenerationConfig
        From :func:`config_for`.
    models : ModelSet or None
        From ``choose_models``.
    live : bool
        The notebook's ``RUN_EXPENSIVE``.
    demo : bool
        Whether the documents are the stand-in, which has a reference set.
    roles : Roles or None, optional
        Overrides :meth:`Roles.from_models`.
    cache : TieredResultCache or None, optional
        Defaults to the shipped tier in ``outputs/baseline_cache``.
    env_path : Path or None, optional
        A ``.env`` for the key resolver.
    """
    if roles is None:
        roles = Roles.from_models(models) if models is not None else REFERENCE_ROLES
    step = set_step(RI_ID, PRODUCER, documents, config, roles)
    reference = None
    if demo and config == DEMO_CONFIG:
        reference = Reference(set_step(RI_ID, PRODUCER, documents, config, REFERENCE_ROLES))

    def compute(choice: ModelChoice) -> dict[str, Any]:
        from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
            SentenceTransformerEmbedder,
        )

        built = build_single_hop_set(
            documents,
            config,
            embedder=SentenceTransformerEmbedder(DENSE_MODEL),
            **_models(roles, env_path),
        )
        value: dict[str, Any] = set_value(built)
        return value

    result = load_or_build(
        step,
        compute,
        roles=roles,
        models=models,
        live=live,
        cache=cache or TieredResultCache(global_dir=BASELINE),
        reference=reference,
    )
    return result, set_from_value(result.value)


def _versus_best(scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    means = {arm: sum(v.values()) / len(v) for arm, v in scores.items() if v}
    best = max(means, key=lambda arm: (means[arm], arm))
    order = sorted(scores[best])
    out = {}
    for arm in scores:
        if arm == best:
            continue
        result = paired_bootstrap([scores[best][q] for q in order], [scores[arm][q] for q in order], samples=2000)
        out[arm] = {
            "mean_difference": round(result.mean_difference, 4),
            "low": round(result.low, 4),
            "high": round(result.high, 4),
        }
    return {"best": best, "arms": out}


def score(
    result: StepResult,
    generated: SyntheticSet,
    documents: list[Document],
    *,
    live: bool,
    cache: TieredResultCache | None = None,
) -> dict[str, Any]:
    """Return RI #1's arm scores on the set's plain questions, cached or computed.

    The arms index the documents the set was built over, minus the held-out
    ones. A set built with ``scope: corpus`` is scored corpus-wide.

    Raises
    ------
    FileNotFoundError
        With ``live`` off, when no scores for this set are cached.
    """
    cache = cache or TieredResultCache(global_dir=BASELINE)
    corpus_wide = generated.meta.get("config", {}).get("scope") == "corpus"
    key = CacheKey(
        RI_ID,
        "synthetic-arm-scores",
        PRODUCER,
        {"set": result.digest, "corpus_wide": corpus_wide, **ARMS_KEY},
    )
    cached = cache.get(key)
    if cached is not None:
        print(f"arm scores: cached, for the set {result.digest[:12]}")
        return dict(cached.value)
    if not live:
        raise FileNotFoundError("no cached arm scores for this set; set RUN_EXPENSIVE = True to score it")

    def compute() -> dict[str, Any]:
        from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
            CachedQueryEmbedder,
            SentenceTransformerEmbedder,
        )

        plain = generated.variant("plain")
        indexed = set(generated.meta["documents"]["indexed"])
        scored = [d for d in documents if d.doc_id in indexed]
        embedder = CachedQueryEmbedder(SentenceTransformerEmbedder(DENSE_MODEL))
        embedder.encode_queries(sorted({q.question for q in plain}))
        arms = build_arms(scored, embedder, corpus_wide=corpus_wide)
        scores = score_arms(arms, queries_for(plain, corpus_wide=corpus_wide), plain.qrels)
        return {
            "arms": list(arms),
            "questions": len(plain.answerable),
            "documents_indexed": len(scored),
            "corpus_wide": corpus_wide,
            "mean": mean_scores(scores),
            "versus_best_ndcg@10": _versus_best(scores["ndcg@10"]),
        }

    print(f"arm scores: scoring the set {result.digest[:12]} locally")
    return dict(cache.load_or_compute(key, compute).value)


def main() -> int:
    """Build the stand-in's reference set and scores into ``outputs/baseline_cache``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    writer = TieredResultCache(local_dir=BASELINE)
    documents, demo = read_corpus(None)
    result, generated = generate(
        documents,
        DEMO_CONFIG,
        None,
        live=True,
        demo=demo,
        roles=REFERENCE_ROLES,
        cache=writer,
        env_path=args.env_file,
    )
    logging.info("set %s", generated.summary())
    scores = score(result, generated, documents, live=True, cache=writer)
    logging.info("nDCG@10 %s", scores["mean"]["ndcg@10"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
