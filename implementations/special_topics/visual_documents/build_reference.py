"""Build the cached reference run on the on-prem models, stage by stage.

Every stage is resumable: model responses sit in the response cache and each
finished step sits in the artifact cache, so a stage that stopped part way
reruns at the cost of the calls it had not made. The stages run in order:

``universe``    choose the page universe (no model)
``enrich``      describe pictures (E1, E2) and summarise complex tables (E2)
``questions``   build and check the question set
``text``        rank chunks for arms T, E1, E2, and L (CPU)
``index``       embed the universe's pages with ColModernVBERT, or store an
                index built elsewhere with ``--import-index``
``queries``     embed the questions with ColModernVBERT, or ``--import-queries``
``visual``      rank pages for VV and H
``probe``       time ColModernVBERT on the CPU and check it against the stored vectors
``answers``     answer every question for every row, on the full set and the subset
``verdicts``    score every answer
``audit``       check 30 descriptions per prompt against their crops
``demo``        run notebook 02's demonstration document on the reference models
``report``      write the committed results, which hold identifiers and metrics only

Examples
--------
Build everything up to the question set::

    python build_reference.py universe enrich questions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aieng.advanced_rag.cache import store_directory  # noqa: E402
from vdocs import arms, enrich, report  # noqa: E402
from vdocs.own_pages import DEMO_NAME, DEMO_QUESTIONS, ROWS, OwnPages  # noqa: E402
from vdocs.pipeline import Pipeline, reference_models  # noqa: E402


def _universe(pipe: Pipeline, args: argparse.Namespace) -> None:
    universe = pipe.universe()
    print({name: len(pages) for name, pages in universe.to_dict().items()}, len(universe.pages), "pages")


def _enrich(pipe: Pipeline, args: argparse.Namespace) -> None:
    for name in ("e2-pictures", "e1-pictures", "e2-tables"):
        print(name, len(pipe.enrichment(name).value), "elements")
    pipe.page_texts()


def _questions(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(json.dumps(pipe.question_set().value["funnel"], indent=1))
    if args.export_questions:
        # Question text derives from FinanceBench; this file is for a local GPU
        # embedding run and stays outside the repository.
        rows = [{"query_id": q["query_id"], "question": q["question"]} for q in pipe.questions()]
        args.export_questions.write_text(json.dumps(rows), encoding="utf-8")


def _text(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(pipe.text_rankings()["seconds"])


def _index(pipe: Pipeline, args: argparse.Namespace) -> None:
    if args.import_index:
        store_directory(pipe.cache, pipe.page_index_key(), args.import_index, force_refresh=True)
    print(pipe.page_index().meta, pipe.page_index().megabytes, "MB")


def _queries(pipe: Pipeline, args: argparse.Namespace) -> None:
    if args.import_queries:
        store_directory(pipe.cache, pipe.query_embeddings_key(), args.import_queries, force_refresh=True)
    print(pipe.query_embeddings()[1])


def _visual(pipe: Pipeline, args: argparse.Namespace) -> None:
    pipe.rankings()


def _probe(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(pipe.cpu_probe())


def _answers(pipe: Pipeline, args: argparse.Namespace) -> None:
    for row in args.rows:
        pipe.answers(row, "all")
        pipe.answers(row, "subset")


def _verdicts(pipe: Pipeline, args: argparse.Namespace) -> None:
    for row in args.rows:
        pipe.verdicts(row, "all")
        pipe.verdicts(row, "subset")


def _audit(pipe: Pipeline, args: argparse.Namespace) -> None:
    print(report.audit_descriptions(pipe).value)


def _demo(pipe: Pipeline, args: argparse.Namespace) -> None:
    own = OwnPages([HERE / "demo" / f"{DEMO_NAME}.pdf"], DEMO_QUESTIONS, pipe.models, live=True, env_path=args.env_file)
    own.figure_only_test()
    for row in ROWS:
        own.verdicts(row)
    print(own.visual_costs())


def _report(pipe: Pipeline, args: argparse.Namespace) -> None:
    for path in report.write_outputs(pipe, HERE / "outputs", pipe.cpu_probe()):
        print(path)


STAGES: dict[str, Callable[[Pipeline, argparse.Namespace], None]] = {
    "universe": _universe,
    "enrich": _enrich,
    "questions": _questions,
    "text": _text,
    "index": _index,
    "queries": _queries,
    "visual": _visual,
    "probe": _probe,
    "answers": _answers,
    "verdicts": _verdicts,
    "audit": _audit,
    "demo": _demo,
    "report": _report,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named stages of the reference build."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--import-index", type=Path, default=None, help="store this PageIndex directory")
    parser.add_argument("--import-queries", type=Path, default=None, help="store this query-embedding directory")
    parser.add_argument("--export-questions", type=Path, default=None, help="write question text here for a GPU run")
    parser.add_argument("--rows", nargs="*", default=list(arms.ROWS))
    parser.add_argument("--workers", type=int, default=16, help="concurrent requests per step")
    parser.add_argument("--env-file", type=Path, default=None, help="a .env holding the proxy key")
    args = parser.parse_args(argv)

    enrich.WORKERS = args.workers
    pipe = Pipeline(reference_models(), live=True, expensive=True, confirm=True, env_path=args.env_file)
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        STAGES[stage](pipe, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
