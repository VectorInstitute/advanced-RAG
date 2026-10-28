#!/usr/bin/env python3
"""Build RI #5's reference runs on MuSiQue and write the committed results.

Each arm runs over a question set on one model through ``load_or_run``, so the
runs land in the artifact cache under the key the notebooks read, and every
model response lands in the response cache. A re-run reads both back and makes
no call.

Examples
--------
The existence proof first, ``C5`` and ``C0`` on the 75 four-hop questions::

    $ python run_agent_arms.py --arms C5 C0 --set 4hop

The shared 300 for the three arms, then the committed results::

    $ python run_agent_arms.py --set shared --report outputs/musique_agent_arms.json

The scale ladder on the 139,416-passage pool::

    $ python run_agent_arms.py --arms C0 C5 C7 --corpus 139416 --device cuda

Both rungs of the ladder in one results file, on a paid model and judged::

    $ python run_agent_arms.py --family google --judge --set ladder \
        --sizes 11656 139416 --arms C0 C5 C7 --cache paid-responses.sqlite \
        --report outputs/musique_scale_ladder_gemini.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_arms import (
    MUSIQUE_DESCRIPTION,
    REFERENCE_MODEL,
    REFERENCE_RPM,
    SERVERFAULT_DESCRIPTION,
    WORKERS,
    abstained,
    dense_search,
    governed_client,
    judge_runs,
    ladder_questions,
    live_questions,
    musique_corpus,
    musique_set,
    run_arm,
    serverfault_set,
)
from agent_report import compare_f1, format_table, joined_table, public_row, ri4_rows, summarise
from aieng.advanced_rag import spend
from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_REVISION,
    TEST_PER_HOP,
    TEST_SEED,
    MusiqueQuestion,
    load_musique,
    stratified_test_set,
)
from aieng.advanced_rag.models import choose_models, endpoint_of


HERE = Path(__file__).resolve().parent
RI4_OUTPUTS = HERE.parent / "graphrag_technical_support" / "outputs"
RI4_RESULTS = RI4_OUTPUTS / "musique_graph_test.json"
#: RI #4's results files, one per answering model; the join picks the one whose
#: model answered the RI #5 arms.
RI4_RESULT_FILES = (RI4_RESULTS, RI4_OUTPUTS / "musique_graph_test_gemini.json")


def ri4_for(model: str) -> dict[str, Any]:
    """Return RI #4's rows answered by ``model``, or no rows when none exist.

    RI #4's rows join an RI #5 table only when the same model answered both.
    """
    for path in RI4_RESULT_FILES:
        if path.is_file() and json.loads(path.read_text(encoding="utf-8"))["test"]["model"] == model:
            rows: dict[str, Any] = ri4_rows(path)
            return rows
    return {"arms": {}}


def question_set(name: str, questions: Sequence[MusiqueQuestion]) -> list[MusiqueQuestion]:
    """Return a named slice of the shared 300.

    ``shared`` is all 300, ``4hop`` the 75 four-hop questions, ``live`` the
    hands-on subset for the reference model (10 per hop count), and ``ladder``
    the scale-ladder subset (the first 40 of each hop count).
    """
    if name == "shared":
        return list(questions)
    if name == "4hop":
        return [q for q in questions if q.hops == 4]
    if name == "live":
        return list(live_questions(questions, REFERENCE_MODEL))
    if name == "ladder":
        return list(ladder_questions(questions))
    raise ValueError(f"unknown question set {name!r}")


def progress(done: int, total: int) -> None:
    """Print a line every 25 questions."""
    if done % 25 == 0 or done == total:
        print(f"  {done}/{total}", flush=True)


def report(
    results: dict[str, Any],
    questions: Sequence[MusiqueQuestion],
    corpus_size: int,
    label: str,
    *,
    verdicts: dict[str, dict[str, bool]] | None = None,
    judge: str | None = None,
) -> dict[str, Any]:
    """Return the committed results: summaries, the joined table, and text-free rows.

    With ``verdicts`` (arm to question identifier to judged-correct), each arm
    also carries the judged-correct share by hop group. A run on a paid model
    joins the RI #4 rows answered by the same model, when they exist.
    """
    gold = {q.question_id: q.supporting_ids for q in questions}
    arms = {}
    rows = {}
    for name, value in results.items():
        public = [public_row(r) for r in value["runs"]]
        rows[name] = public
        arms[name] = {"model": value["model"], "summary": summarise(public, gold)}
        if verdicts and name in verdicts:
            arms[name]["judged"] = judged_shares(public, verdicts[name])
    ri5 = {"arms": arms}
    block = produced_by(results, judge)
    ri4 = ri4_for(block["model"]) if corpus_size == 11_656 else {"arms": {}}
    tables = {group: joined_table(ri5, ri4, group) for group in ("all", "2hop", "3hop", "4hop")}
    comparisons = {}
    if "C5" in rows and "C0" in rows:
        if any(r["hops"] == 4 for r in rows["C0"]):
            comparisons["C5_minus_C0_f1_4hop"] = compare_f1(rows["C5"], rows["C0"], hops=4)
        comparisons["C5_minus_C0_f1_all"] = compare_f1(rows["C5"], rows["C0"])
    return {
        "test": {
            "questions": len(questions),
            "question_set": label,
            "per_hop": {str(k): v for k, v in TEST_PER_HOP.items()} if label == "shared" else None,
            "seed": TEST_SEED,
            "passages": corpus_size,
            "dataset_revision": MUSIQUE_REVISION,
            "model": produced_by(results)["model"],
            "judge": judge,
            "encoder": "BAAI/bge-small-en-v1.5",
            "max_model_calls": 30,
            "grep": "Python regular expressions over the passages in memory, case-insensitive by default",
            "label": block["label"],
        },
        "produced_by": block,
        "arms": arms,
        "joined": tables,
        "comparisons": comparisons,
        "rows": rows,
    }


def judged_shares(rows: Sequence[dict[str, Any]], verdicts: dict[str, bool]) -> dict[str, Any]:
    """Return the judged-correct share over all questions and by hop count."""
    out: dict[str, Any] = {}
    for group, keep in (("all", None), ("2hop", 2), ("3hop", 3), ("4hop", 4)):
        picked = [r for r in rows if keep is None or r["hops"] == keep]
        if picked:
            out[group] = round(sum(bool(verdicts.get(r["question_id"])) for r in picked) / len(picked), 4)
    return out


def produced_by(results: dict[str, Any], judge: str | None = None) -> dict[str, Any]:
    """Return the ``produced_by`` block: the model the arms ran on, and its calls.

    ``label`` is ``reference trajectories`` for the on-prem model and
    ``paid-model rerun`` for any other, so a table never reads as the wrong
    model's. ``endpoint`` comes from the model, and ``judge`` names the model
    that graded the answers, when one did.
    """
    models = sorted({value["model"] for value in results.values()})
    onprem = all(endpoint_of(m) == "vector-cluster" for m in models)
    block = {
        "model": models[0] if len(models) == 1 else ", ".join(models),
        "endpoint": "vector-cluster" if onprem else "vector-proxy-paid",
        "producer": "run_agent_arms.py",
        "model_calls": sum(r.get("model_calls", 0) for value in results.values() for r in value["runs"]),
        "label": "reference trajectories" if onprem else "paid-model rerun",
    }
    if judge:
        block["judge"] = judge
    return block


def set_report(results: dict[str, Any], questions: Sequence[Any], name: str) -> dict[str, Any]:
    """Return the paraphrase or null results: scores, abstention, and cost."""
    gold = {q.question_id: q.supporting_ids for q in questions}
    arms = {}
    rows = {}
    for arm, value in results.items():
        public = []
        for run in value["runs"]:
            row = public_row(run)
            row["abstained"] = abstained(run.get("answer", ""))
            public.append(row)
        rows[arm] = public
        n = len(public)
        entry: dict[str, Any] = {
            "model": value["model"],
            "questions": n,
            "abstention_rate": round(sum(r["abstained"] for r in public) / n, 4),
            "calls_per_question": round(sum(r["model_calls"] for r in public) / n, 2),
            "tokens_per_question": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in public) / n, 1),
            "exhaustion_rate": round(sum(bool(r["exhausted"]) for r in public) / n, 4),
            "zero_hit_grep_share": round(
                sum(r.get("zero_hit_greps", 0) for r in public)
                / max(sum(r.get("tools", {}).get("grep", 0) for r in public), 1),
                4,
            ),
        }
        if name == "paraphrase":
            s = summarise(public, gold)["all"]
            entry.update(
                {k: s[k] for k in ("exact_match", "f1", "all_supporting_recall_read", "all_supporting_recall_seen")}
            )
        arms[arm] = entry
    return {
        "test": {"questions": len(questions), "question_set": name, "passages": 11_656, "model": REFERENCE_MODEL},
        "produced_by": produced_by(results),
        "arms": arms,
        "rows": rows,
    }


def serverfault_report(results: dict[str, Any], questions: Sequence[Any]) -> dict[str, Any]:
    """Return the ServerFault results: gold posts read and seen, and cost."""
    gold = {q.question_id: set(q.supporting_ids) for q in questions}
    arms = {}
    rows = {}
    for name, value in results.items():
        public = [public_row(r) for r in value["runs"]]
        rows[name] = public
        n = len(public)
        arms[name] = {
            "model": value["model"],
            "questions": n,
            "gold_read": round(sum(bool(gold[r["question_id"]] & set(r["read_ids"])) for r in public) / n, 4),
            "gold_seen": round(
                sum(bool(gold[r["question_id"]] & (set(r["read_ids"]) | set(r["seen_ids"]))) for r in public) / n, 4
            ),
            "calls_per_question": round(sum(r["model_calls"] for r in public) / n, 2),
            "tokens_per_question": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in public) / n, 1),
            "seconds_per_question": round(sum(r["seconds"] for r in public) / n, 2),
            "exhaustion_rate": round(sum(bool(r["exhausted"]) for r in public) / n, 4),
        }
    return {
        "test": {"questions": len(questions), "corpus": "serverfault-10k", "model": REFERENCE_MODEL},
        "produced_by": produced_by(results),
        "arms": arms,
        "rows": rows,
    }


def ladder_report(
    by_size: dict[int, dict[str, Any]],
    questions: Sequence[MusiqueQuestion],
    verdicts: dict[int, dict[str, dict[str, bool]]] | None = None,
    judge: str | None = None,
) -> dict[str, Any]:
    """Return the scale-ladder results: per corpus size, each arm's summary and rows.

    The shape is the committed ``musique_scale_ladder.json``'s, with the judged
    share per arm when ``verdicts`` (size to arm to question to judged-correct)
    is given, and the paired ``C5`` minus ``C0`` F1 difference per size.
    """
    sizes = {}
    every: dict[str, Any] = {}
    for size, results in by_size.items():
        out = report(results, questions, size, "ladder", verdicts=(verdicts or {}).get(size), judge=judge)
        sizes[str(size)] = {"arms": out["arms"], "comparisons": out["comparisons"], "rows": out["rows"]}
        every.update({f"{name}@{size}": value for name, value in results.items()})
    block = produced_by(every, judge)
    return {
        "test": {
            "questions": len(questions),
            "question_set": "first 40 per hop count of the shared 300",
            "model": block["model"],
            "judge": judge,
            "label": block["label"],
        },
        "produced_by": block,
        "sizes": sizes,
    }


def write_report(
    args: argparse.Namespace,
    results: dict[str, Any],
    questions: Sequence[Any],
    verdicts: dict[str, dict[str, bool]] | None = None,
    judge: str | None = None,
) -> None:
    """Print the results and write them to ``--report`` when given."""
    if args.set in ("paraphrase", "nulls"):
        out = set_report(results, questions, args.set)
        print(json.dumps(out["arms"], indent=1))
    elif args.corpus == "serverfault":
        out = serverfault_report(results, questions)
        print(json.dumps(out["arms"], indent=1))
    else:
        out = report(results, questions, int(args.corpus), args.set, verdicts=verdicts, judge=judge)
        print(format_table(out["joined"]["all"]))
        print()
        print(format_table(out["joined"].get("4hop", [])))
        for name, value in out["comparisons"].items():
            print(name, value)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def judge_arms(
    args: argparse.Namespace,
    models: Any,
    results: dict[str, Any],
    *,
    questions: Sequence[MusiqueQuestion],
    corpus: Any,
    label: str,
    client: Any,
) -> tuple[dict[str, dict[str, bool]], str]:
    """Judge every arm's answers; return the verdicts by arm and the judge's name."""
    judge = models.judging.model
    used = client if client is not None and args.cache else governed_client(rpm=args.rpm, cache_path=args.cache)
    verdicts: dict[str, dict[str, bool]] = {}
    for name, value in results.items():
        print(f"judging {name} with {judge}", flush=True)
        step = judge_runs(
            value["runs"],
            questions,
            corpus=corpus,
            answered_by=value["model"],
            label=f"{name}/{label}",
            models=models,
            live=True,
            client=used,
        )
        verdicts[name] = step.value["verdicts"]
    return verdicts, judge


def run_ladder(args: argparse.Namespace, questions: Sequence[MusiqueQuestion], models: Any, client: Any) -> int:
    """Run the arms at each ``--sizes`` corpus, judge them, and write one ladder report.

    Each size's runs carry the label the notebooks read (``ladder-<size>``), so
    the per-question cache a shared-300 run filled is reused where they overlap.
    """
    by_size: dict[int, dict[str, Any]] = {}
    verdicts: dict[int, dict[str, dict[str, bool]]] = {}
    judge = None
    for size in args.sizes:
        corpus = musique_corpus(size)
        label = f"ladder-{size}" + (f"-first{args.limit}" if args.limit else "")
        search_cache: list[Any] = []

        def search(corpus: Any = corpus, cache: list[Any] = search_cache) -> Any:
            if not cache:
                cache.append(dense_search(corpus, device=args.device))
            return cache[0]

        results: dict[str, Any] = {}
        for name in args.arms:
            print(f"arm {name}: {len(questions)} questions on {corpus.name}", flush=True)
            results[name] = run_arm(
                name,
                questions,
                corpus=corpus,
                label=label,
                models=models,
                live=not args.cached_only,
                confirm=args.confirm,
                search=search,
                client=client,
                description=MUSIQUE_DESCRIPTION,
                rpm=args.rpm,
                workers=args.workers,
                progress=progress,
            ).value
        by_size[size] = results
        if args.judge:
            verdicts[size], judge = judge_arms(
                args, models, results, questions=questions, corpus=corpus, label=label, client=client
            )
    out = ladder_report(by_size, questions, verdicts or None, judge)
    for size, entry in out["sizes"].items():
        for name, arm in entry["arms"].items():
            s = arm["summary"]["all"]
            judged = arm.get("judged", {}).get("all")
            print(
                f"{size:>7} {name:<4} F1 {s['f1']:.3f} EM {s['exact_match']:.3f} judged {judged} "
                f"calls/q {s['calls_per_question']:.2f} tokens/q {s['tokens_per_question']:,.0f} "
                f"exhausted {s['exhaustion_rate']:.2f}"
            )
        for name, value in entry["comparisons"].items():
            print(size, name, value)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line, refusing a paid model without its own cache."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", default=["C0", "C5", "grep-iterative"])
    parser.add_argument("--set", default="shared", choices=("shared", "4hop", "live", "ladder", "paraphrase", "nulls"))
    parser.add_argument("--corpus", default="11656", choices=("11656", "139416", "serverfault"))
    parser.add_argument("--rpm", type=int, default=REFERENCE_RPM)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default="cpu", help="torch device for embedding the corpus")
    parser.add_argument("--report", type=Path, default=None, help="write the committed results here")
    parser.add_argument("--limit", type=int, default=None, help="first N questions, for a smoke run")
    parser.add_argument("--cached-only", action="store_true", help="read the runs, never call a model")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="run a step whose pre-run estimate is over the confirmation threshold",
    )
    parser.add_argument(
        "--family",
        default="onprem",
        choices=("onprem", "google", "anthropic", "openai"),
        help="model family for answering; google also judges with the shared judge",
    )
    parser.add_argument("--cache", type=Path, default=None, help="response cache file; paid runs use their own file")
    parser.add_argument("--judge-model", default=None, help="override the judging model, e.g. gemini-3.8-flash")
    parser.add_argument("--judge", action="store_true", help="judge every arm's answers with the family's judge")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        choices=(11_656, 139_416),
        help="with --set ladder, run every listed corpus size and write one ladder report",
    )
    args = parser.parse_args(argv)
    if args.sizes and args.set != "ladder":
        parser.error("--sizes runs the scale ladder; pass --set ladder with it")
    args.paid = args.family != "onprem" or bool(args.judge_model)
    if args.paid and args.cache is None:
        parser.error("a paid model needs --cache, so its responses stay out of the on-prem cache")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the arms and, with ``--report``, write the results file."""
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    description = MUSIQUE_DESCRIPTION
    if args.corpus == "serverfault":
        corpus, questions = serverfault_set()
        questions = questions[: args.limit]
        description = SERVERFAULT_DESCRIPTION
        args.set = "errors"
    elif args.set in ("paraphrase", "nulls"):
        questions = musique_set(args.set)[: args.limit]
        corpus = musique_corpus(int(args.corpus))
    else:
        _, all_questions = load_musique()
        questions = question_set(args.set, stratified_test_set(all_questions))[: args.limit]
        corpus = musique_corpus(int(args.corpus))
    models = choose_models(args.family)
    if args.judge_model:
        models = choose_models(args.family, judging=args.judge_model)
    paid = args.paid
    if paid:
        # load_or_run never opens the spend session, so the
        # ADVANCED_RAG_SPEND_LIMIT_USD ceiling is installed here.
        spend.session()
    client = None if args.cached_only else governed_client(rpm=args.rpm, cache_path=args.cache)
    search_cache: list[Any] = []

    def search() -> Any:
        if not search_cache:
            search_cache.append(dense_search(corpus, device=args.device))
        return search_cache[0]

    if args.sizes:
        status = run_ladder(args, questions, models, client)
        if paid:
            print(f"spend guard estimate ${spend.session().estimated_usd:,.2f}")
        return status

    label = f"{args.set}-{args.corpus}" + (f"-first{args.limit}" if args.limit else "")
    results: dict[str, Any] = {}
    for name in args.arms:
        print(f"arm {name}: {len(questions)} questions on {corpus.name}", flush=True)
        result = run_arm(
            name,
            questions,
            corpus=corpus,
            label=label,
            models=models,
            live=not args.cached_only,
            confirm=args.confirm,
            search=search,
            client=client,
            description=description,
            rpm=args.rpm,
            workers=args.workers,
            progress=progress,
        )
        results[name] = result.value
    verdicts, judge = (
        judge_arms(args, models, results, questions=questions, corpus=corpus, label=label, client=client)
        if args.judge
        else (None, None)
    )
    write_report(args, results, questions, verdicts, judge)
    if paid:
        print(f"spend guard estimate ${spend.session().estimated_usd:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
