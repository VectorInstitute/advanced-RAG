"""Build the cached reference run on the on-prem model, stage by stage.

Every stage is resumable: model responses sit in the response cache and each
finished step sits in the artifact cache, so a stage that stopped part way
reruns at the cost of the calls it had not made. The stages run in order:

``rankings``   retrieve the top 5 chunks of each question's agreement (CPU)
``answers``    answer every question in each generated arm, on the full set
               and the live subset
``posthoc``    attach citations to the inline statements with MiniCheck (CPU)
``judge``      judge every arm's citations with the judging model
``minicheck``  judge every arm's citations with MiniCheck-Flan-T5-Large (CPU)
``explore``    build the quote-abstention variants and the question-aware
               MiniCheck checks (see ``cites/explore.py``)
``report``     write ``outputs/results.json``

Examples
--------
Build everything::

    python build_reference.py rankings answers posthoc judge minicheck report
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cites import pipeline, report  # noqa: E402
from cites.explore import VARIANTS  # noqa: E402
from cites.pipeline import ARMS, GENERATED, Pipeline, reference_models  # noqa: E402


WHICH = ("all", "subset")


def _rankings(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(len(pipe.rankings()), "questions ranked")


def _answers(pipe: Pipeline, args: argparse.Namespace) -> None:
    for arm in args.arms:
        if arm in GENERATED:
            for which in WHICH:
                answers = pipe.answers(arm, which)
                print(arm, which, len(answers), "answers", sum(a.abstained for a in answers), "abstained", flush=True)


def _posthoc(pipe: Pipeline, args: argparse.Namespace) -> None:
    for which in WHICH:
        answers = pipe.posthoc(which)
        cited = sum(1 for a in answers for s in a.statements if s.citations)
        print("posthoc", which, cited, "statements cited", flush=True)


def _judge(pipe: Pipeline, args: argparse.Namespace) -> None:
    for arm in args.arms:
        for which in WHICH:
            print(arm, which, len(pipe.judged(arm, which)), "judged", flush=True)


def _minicheck(pipe: Pipeline, args: argparse.Namespace) -> None:
    for arm in args.arms:
        print(arm, len(pipe.minichecked(arm, "all")), "checked", flush=True)


def _explore(pipe: Pipeline, args: argparse.Namespace) -> None:
    for arm in VARIANTS:
        answers = pipe.answers(arm, "all")
        print(arm, len(answers), "answers", sum(a.abstained for a in answers), "abstained", flush=True)
    for arm in (a for a in VARIANTS if a.startswith("fulldoc")):
        print(arm, len(pipe.judged(arm, "all")), "judged", flush=True)
    for arm in report.WITH_QUESTION:
        print(arm, len(pipe.minichecked(arm, "all", with_question=True)), "checked with the question", flush=True)
    print("answer-standalone", len(pipe.minichecked("answer-standalone", "all")), "checked", flush=True)


def _report(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(report.write_results(pipe, HERE / "outputs" / "results.json"))


STAGES: dict[str, Callable[[Pipeline, argparse.Namespace], None]] = {
    "rankings": _rankings,
    "answers": _answers,
    "posthoc": _posthoc,
    "judge": _judge,
    "minicheck": _minicheck,
    "explore": _explore,
    "report": _report,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named stages of the reference build."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=pipeline.WORKERS, help="concurrent requests per step")
    parser.add_argument("--rpm", type=int, default=pipeline.RPM, help="requests per minute for each model")
    parser.add_argument("--env-file", type=Path, default=None, help="a .env holding the proxy key")
    args = parser.parse_args(argv)

    pipeline.WORKERS = args.workers
    pipeline.RPM = args.rpm
    pipe = Pipeline(reference_models(), live=True, local_checker=True, confirm=True, env_path=args.env_file)
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        STAGES[stage](pipe, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
