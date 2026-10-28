"""Write this topic's reference artifacts into ``outputs/baseline_cache``.

Two stages, each through the same ``load_or_run`` path a live notebook run
takes, on Vector's on-prem models (``Qwen3_8-27B`` generates,
``Qwen3_6-35B-A3B`` checks, ``Qwen3_8-Flash-Next`` judges):

- ``validation`` builds the 28-document ContractNLI set and scores RI #1's
  arms under real and generated labels. Its model calls are the ones
  ``regulatory_policy_qa/validate_synthetic_qa.py`` made, so with that run's
  response cache present it makes no new call.
- ``explore`` runs ``explore_adjacent_statements.py``.

Run from this folder:

    python build_reference.py validation explore --env-file <env_file>

where ``<env_file>`` is a ``.env`` holding ``VECTOR_PROXY_API_KEY``, needed
only when the key is not in the environment and a response is not cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqsets  # noqa: E402
from aieng.advanced_rag.cache import TieredResultCache  # noqa: E402
from aieng.advanced_rag.evaluation.synthetic.replay import (  # noqa: E402
    REFERENCE_ROLES,
    load_or_build,
)


def build_validation(env_file: Path | None, per_minute: int) -> None:
    """Build and store the 28-document validation run."""
    docs, real = sqsets.contract_nli(sqsets.REFERENCE_DOCUMENTS)
    result = load_or_build(
        sqsets.validation_step(docs, REFERENCE_ROLES),
        sqsets.validation_compute(docs, real, REFERENCE_ROLES, per_minute=per_minute, env_path=env_file),
        roles=REFERENCE_ROLES,
        live=True,
        cache=TieredResultCache(local_dir=sqsets.BASELINE),
    )
    tau = result.value["validation"]["agreement"]["plain"]["all_arms"]["ndcg@10"]
    logging.info("validation plain tau(nDCG@10) %s", tau["kendall_tau"])


def main() -> int:
    """Run the named stages in order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stages", nargs="+", choices=("validation", "explore"))
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--per-minute", type=int, default=sqsets.PER_MINUTE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.per_minute > 15:
        raise SystemExit("keep the on-prem models at 15 requests per minute or fewer")

    import torch  # noqa: PLC0415

    # A shared machine with dozens of runnable threads makes a single-query
    # encode slow at torch's default thread count.
    torch.set_num_threads(4)
    if "validation" in args.stages:
        build_validation(args.env_file, args.per_minute)
    if "explore" in args.stages:
        import explore_adjacent_statements  # noqa: PLC0415

        sys.argv = [sys.argv[0], "--per-minute", str(args.per_minute)]
        if args.env_file is not None:
            sys.argv += ["--env-file", str(args.env_file)]
        explore_adjacent_statements.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
