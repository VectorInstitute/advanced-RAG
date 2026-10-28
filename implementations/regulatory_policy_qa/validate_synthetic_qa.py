"""Check whether generated labels rank RI #1's retrieval arms the way ContractNLI's do.

The run generates a single-hop question set with nulls over ContractNLI test
documents, then scores the same retrieval arms twice over the same documents:
once against ContractNLI's human-annotated evidence spans, and once against the
generated set's spans. Agreement is Kendall's tau between the two arm rankings
and the per-pair sign agreement of paired bootstrap differences.

This is a single run: one generation seed, one configuration, looking for
large disagreements or a surprising ordering, not a validated threshold.

    python validate_synthetic_qa.py --documents 28

Model calls go through the proxy with thinking off, throttled per model, and
cached, so a re-run costs nothing. Embeddings and the reranker run locally.

ContractNLI is CC BY 4.0, so the committed set keeps its quotes, with the
attribution and the statement of changes in ``outputs/synthetic_qa/ATTRIBUTION.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.evaluation.synthetic import build_single_hop_set, proxy_model
from aieng.advanced_rag.evaluation.synthetic.validation import (
    CONTRACT_NLI_CONFIG,
    DENSE_MODEL,
    RERANKER,
    load_contract_nli,
    validate,
)
from aieng.advanced_rag.retrieval.embedders import (
    CachedQueryEmbedder,
    SentenceTransformerEmbedder,
)
from aieng.advanced_rag.runs import Provenance


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "synthetic_qa"

GENERATOR = "Qwen3_8-27B"
CHECKER = "Qwen3_6-35B-A3B"
JUDGE = "Qwen3_8-Flash-Next"
CONFIG = CONTRACT_NLI_CONFIG


def env_file() -> Path | None:
    """Return the ``.env`` holding the key, looking past a worktree to the main repo.

    ``None`` lets the resolver read the environment and the nearest ``.env``.
    """
    local = HERE.parents[1] / ".env"
    if local.exists():
        return None
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        cwd=HERE,
    ).stdout.strip()
    candidate = Path(common).parent / ".env" if common else None
    return candidate if candidate is not None and candidate.exists() else None


def produced_by(usage: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the result's ``produced_by`` block, named for the generator.

    The block has one ``model`` slot, so it names the generator, whose
    questions the result measures, and lists the checker and the judge in the
    note and in ``usage``. ``usage`` counts the cold build, cached calls
    included, so a rebuild estimate does not shrink on a warm re-run.
    """
    block: dict[str, Any] = Provenance.for_model(
        GENERATOR,
        producer="regulatory_policy_qa/validate_synthetic_qa.py",
        usage={
            row["model"]: {
                "calls": row["cold_requests"],
                "prompt_tokens": row["cold_prompt_tokens"],
                "completion_tokens": row["cold_completion_tokens"],
            }
            for row in usage
        },
        note=(
            f"Generator {GENERATOR}, checker {CHECKER}, judge {JUDGE}, all with "
            "thinking off. Embeddings are local BAAI/bge-small-en-v1.5 and the "
            "reranker is a local cross-encoder."
        ),
    ).to_dict()
    return block


def main() -> int:
    """Generate, score, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--documents", type=int, default=28)
    parser.add_argument("--per-minute", type=int, default=18)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # A shared machine with dozens of runnable threads makes a single-query
    # encode take a second at torch's default thread count; four is ten times
    # faster here.
    torch.set_num_threads(4)
    started = time.perf_counter()

    documents, real_by_doc = load_contract_nli(args.documents)
    cache = cache_root() / "synthetic" / "responses.sqlite"
    models = {
        role: proxy_model(name, cache_path=cache, per_minute=args.per_minute, env_path=env_file())
        for role, name in (("generator", GENERATOR), ("checker", CHECKER), ("judge", JUDGE))
    }
    embedder = SentenceTransformerEmbedder(DENSE_MODEL)
    generated = build_single_hop_set(documents, CONFIG, embedder=embedder, **models)
    logging.info("generated %s", generated.summary())
    scored = validate(generated, documents, real_by_doc, CachedQueryEmbedder(embedder))
    results = scored["agreement"]
    names = scored["arms"]

    funnel = generated.funnel.to_dict() if generated.funnel else {}
    payload = {
        "mode": "explore: one generation seed, one configuration",
        "produced_by": produced_by(generated.meta["usage"]),
        "corpus": {
            "name": "ContractNLI test split",
            "documents_loaded": len(documents),
            "documents_scored": scored["documents_scored"],
            "held_out": generated.meta["documents"]["held_out"],
            "real_answerable_queries": scored["real_answerable_queries"],
            "licence": contract_nli.CONTRACT_NLI_LICENCE.name,
        },
        "models": {**generated.meta["models"], "embedder": DENSE_MODEL, "reranker": RERANKER},
        "same_family": generated.meta["same_family"],
        "config": generated.meta["config"],
        "set": generated.summary(),
        "funnel": funnel,
        "sampling": generated.meta["sampling"],
        "usage": generated.meta["usage"],
        "arms": names,
        "agreement": results,
        "generation_seconds": generated.meta["seconds"],
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    generated.to_jsonl(args.out / "contract_nli_generated.jsonl")
    (args.out / "contract_nli_validation.json").write_text(json.dumps(payload, indent=1) + "\n")
    for variant, result in results.items():
        for group, by_metric in result.items():
            if isinstance(by_metric, dict):
                taus = {m: r["kendall_tau"] for m, r in by_metric.items()}
                logging.info("%s %s tau %s", variant, group, taus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
