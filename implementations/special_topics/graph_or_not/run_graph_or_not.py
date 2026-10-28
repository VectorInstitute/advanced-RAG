#!/usr/bin/env python3
"""Run the graph-or-not experiments: X1 (MuSiQue), X3 (ServerFault), and X2 (readiness).

X1 ranks RI #4's 300 MuSiQue questions with a stack of local models and
answers each from the top five passages with RI #4's reader and prompt, so its
rows compare with RI #4's committed ``hipporag2`` rows question by question.
X3 asks 60 relational questions of the ServerFault graph and of retrieval over
the posts. X2 computes the graph-readiness statistics on the corpora at hand.

Model calls go through the shared proxy client: throttled, cached, and counted.
On-prem answers use their own response cache; a paid judge needs ``--paid-cache``
and honours ``ADVANCED_RAG_SPEND_LIMIT_USD``. Results hold identifiers and
scores only, never question or passage text from the corpora.

Examples
--------
Rank and answer, then judge with the shared judge::

    $ python run_graph_or_not.py x1
    $ ADVANCED_RAG_SPEND_LIMIT_USD=5 python run_graph_or_not.py x1 --judge \
        --paid-cache <paid_responses.sqlite>

Relational questions and the readiness table::

    $ python run_graph_or_not.py x3
    $ python run_graph_or_not.py x2

where ``<paid_responses.sqlite>`` is a response cache file kept apart from the
on-prem one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RI4 = HERE.parents[1] / "graphrag_technical_support"
sys.path[:0] = [str(HERE), str(RI4)]

from aieng.advanced_rag import spend  # noqa: E402
from aieng.advanced_rag.cache.paths import root as cache_root  # noqa: E402
from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, govern  # noqa: E402
from aieng.advanced_rag.models import endpoint_of, request_params  # noqa: E402
from openai import OpenAI  # noqa: E402


OUT = HERE / "outputs"
READER = "Qwen3_6-35B-A3B"
JUDGE = "gemini-3.8-flash"
GRAPH_FILE = Path(cache_root()) / "graphrag" / "serverfault" / "graph" / "graph.json"


def work() -> Path:
    """Return this topic's cache folder, outside the reference index folders."""
    path = Path(cache_root()) / "special_topics" / "graph-or-not"
    path.mkdir(parents=True, exist_ok=True)
    return path


def produced_by(model: str, producer: str, judge: str | None = None) -> dict[str, Any]:
    """Return the ``produced_by`` block, labelled by where the answering model runs."""
    onprem = endpoint_of(model) == "vector-cluster"
    block: dict[str, Any] = {
        "model": model,
        "endpoint": "vector-cluster" if onprem else "vector-proxy-paid",
        "producer": f"run_graph_or_not.py {producer}",
        "label": "reference trajectories" if onprem else "paid-model rerun",
    }
    if judge:
        block["judge"] = judge
    return block


def client(cache: Path, rpm: int, ledger: CallLedger) -> Any:
    """Return the governed proxy client."""
    from build_indexes import BASE_URL, api_key  # noqa: PLC0415

    return govern(
        OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        cache=ResponseCache(cache),
        limiter=RateLimiter(per_minute=rpm),
        ledger=ledger,
        max_retries=16,
    )


def completer(governed: Any, model: str, max_tokens: int) -> Callable[[str], str]:
    """Return ``complete(prompt) -> text`` at temperature 0, as RI #4 calls it."""
    params = request_params(model, max_tokens=max_tokens, temperature=0.0)
    params.pop("extra_body", None)

    def complete(prompt: str) -> str:
        response = governed.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **params
        )
        return str(response.choices[0].message.content or "")

    return complete


def _cuda() -> bool:
    import torch  # noqa: PLC0415

    return bool(torch.cuda.is_available())


def write(name: str, payload: dict[str, Any]) -> Path:
    """Write one results file under ``outputs/``."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- X1


def x1_rankings(
    questions: Sequence[Any],
    passages: Sequence[Any],
    hippo: dict[str, list[str]],
    device: str,
    *,
    inputs_only: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Return every X1 arm's top 10 per question, ranking when none are cached.

    The inputs file holds corpus text, so it stays in the cache folder.
    """
    from build_indexes import work_root  # noqa: PLC0415

    cached = work() / f"x1-rankings-{len(questions)}.json"
    if not cached.is_file():
        inputs = work() / f"x1-inputs-{len(questions)}.json"
        inputs.write_text(
            json.dumps(
                {
                    "questions": [{"id": q.question_id, "text": q.question} for q in questions],
                    "passages": [{"id": p.passage_id, "text": f"{p.title}\n{p.text}"} for p in passages],
                    "hippo": hippo,
                    "dense_matrix": str(work_root() / f"dense-{len(passages)}.npy"),
                }
            ),
            encoding="utf-8",
        )
        if inputs_only:
            raise SystemExit(
                f"wrote {inputs}; rank with: python rank_x1.py --inputs {inputs} --out {cached} --device cuda"
            )
        import rank_x1  # noqa: PLC0415

        rank_x1.main(["--inputs", str(inputs), "--out", str(cached), "--device", device])
    rankings: dict[str, dict[str, list[str]]] = json.loads(cached.read_text(encoding="utf-8"))
    return rankings


def ri4_rows(keep: set[str], name: str = "musique_graph_test.json") -> dict[str, list[Any]]:
    """Return RI #4's committed ``dense`` and ``hipporag2`` rows for the questions.

    ``name`` picks the RI #4 results file, so a run with another reader or prompt
    compares against RI #4's rows for that same reader and prompt.
    """
    from musique_report import Row  # noqa: PLC0415

    committed = json.loads((RI4 / "outputs" / name).read_text(encoding="utf-8"))
    return {
        arm: [
            Row(r["question_id"], r["hops"], tuple(r["ranked"]), r["em"], r["f1"])
            for r in committed["rows"][arm]
            if r["question_id"] in keep
        ]
        for arm in ("dense", "hipporag2")
    }


def answer_arms(
    arms: Sequence[str],
    questions: Sequence[Any],
    rankings: dict[str, dict[str, list[str]]],
    *,
    complete: Callable[[str], str],
    text_of: Callable[[str], str],
    ledger: CallLedger,
    workers: int,
    prompt: str = "standard",
    passages: int = 5,
) -> tuple[dict[str, list[Any]], dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Answer every question from each arm's top passages with an RI #4 prompt."""
    from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1  # noqa: PLC0415
    from musique_arms import ANSWER_PROMPTS, answer  # noqa: PLC0415

    template = ANSWER_PROMPTS[prompt]
    from musique_report import Row  # noqa: PLC0415

    async def answer_arm(arm: str, replies: dict[str, str]) -> list[Any]:
        gate = asyncio.Semaphore(workers)

        async def one(question: Any) -> Any:
            ranked = rankings[question.question_id][arm]
            async with gate:
                reply = await answer(complete, question.question, ranked[:passages], text_of, template)
            replies[question.question_id] = reply
            golds = (question.answer, *question.answer_aliases)
            return Row(
                question.question_id, question.hops, tuple(ranked), exact_match(reply, golds), token_f1(reply, golds)
            )

        return list(await asyncio.gather(*(one(q) for q in questions)))

    rows: dict[str, list[Any]] = {}
    replies: dict[str, dict[str, str]] = {}
    costs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        before = ledger.to_dict()
        started = time.perf_counter()
        replies[arm] = {}
        rows[arm] = asyncio.run(answer_arm(arm, replies[arm]))
        after = ledger.to_dict()
        costs[arm] = {k: after[k] - before[k] for k in ("calls", "total_tokens", "cache_hits")}
        costs[arm]["wall_seconds"] = round(time.perf_counter() - started, 1)
        mean_f1 = sum(r.f1 for r in rows[arm]) / len(rows[arm])
        print(f"arm {arm}: F1 {mean_f1:.3f}, {costs[arm]['calls']} calls", flush=True)
    return rows, replies, costs


def run_x1(args: argparse.Namespace) -> int:
    """Rank, answer, optionally judge, and write ``outputs/x1_musique.json``."""
    from aieng.advanced_rag.corpora.musique import load_musique, stratified_test_set  # noqa: PLC0415
    from gon.x1 import ANSWERED_ARMS, RETRIEVAL_ONLY_ARMS, recall_table  # noqa: PLC0415
    from musique_report import compare, summarise  # noqa: PLC0415
    from run_musique_test import abstention, judge_rows, judged_shares  # noqa: PLC0415

    passages, all_questions = load_musique()
    questions = stratified_test_set(all_questions)[: args.questions or None]
    gold = {q.question_id: q.supporting_ids for q in questions}
    baseline = ri4_rows(set(gold), args.baseline)
    hippo = {r.question_id: list(r.ranked) for r in baseline["hipporag2"]}
    rankings = x1_rankings(questions, passages, hippo, args.device, inputs_only=args.write_inputs)
    # RI #4's own ranking, so a run can answer it under the same settings as the arms.
    for qid, ranked in hippo.items():
        rankings[qid]["hipporag2"] = ranked
    # The reproduced dense ranking must match RI #4's committed one, or the reader
    # comparison is off.
    agree = sum(rankings[r.question_id]["dense"] == list(r.ranked) for r in baseline["dense"])
    print(f"dense rankings identical to RI #4 for {agree}/{len(questions)} questions", flush=True)

    by_id = {p.passage_id: f"{p.title}\n{p.text}" for p in passages}
    reader = args.reader or READER
    arms = tuple(args.arms) if args.arms else ANSWERED_ARMS
    paid_reader = endpoint_of(reader) != "vector-cluster"
    if args.judge and args.answer_passages != 5:
        raise SystemExit("the judge reads an answer's top five passages, so --judge needs --answer-passages 5")
    if (paid_reader or args.judge) and args.paid_cache is None:
        raise SystemExit("a paid reader or judge needs --paid-cache, so paid responses stay out of the on-prem cache")
    if paid_reader or args.judge:
        spend.session()
    ledger = CallLedger("graph-or-not-x1")
    cache = args.paid_cache if paid_reader else (args.cache or work() / "responses-onprem.sqlite")
    complete = completer(client(cache, args.rpm, ledger), reader, 256)
    rows, replies, costs = answer_arms(
        arms,
        questions,
        rankings,
        complete=complete,
        text_of=by_id.__getitem__,
        ledger=ledger,
        workers=args.workers,
        prompt=args.answer_prompt,
        passages=args.answer_passages,
    )

    judged: dict[str, dict[str, float]] = {}
    if args.judge:
        judge = completer(client(args.paid_cache, args.rpm, CallLedger("graph-or-not-x1-judge")), JUDGE, 300)
        for arm in arms:
            verdicts = judge_rows(rows[arm], replies[arm], questions, complete=judge, text_of=by_id.__getitem__)
            judged[arm] = judged_shares(rows[arm], verdicts)
            print(f"judged {arm}: {judged[arm]['all']:.3f}", flush=True)
    if paid_reader or args.judge:
        print(f"spend guard estimate ${spend.session().estimated_usd:,.2f}")

    payload = {
        "test": {
            "questions": len(questions),
            "passages": len(passages),
            "reader": reader,
            "judge": JUDGE if args.judge else None,
            "answer_passages": args.answer_passages,
            "answer_prompt": args.answer_prompt,
            "models": {
                "dense": "BAAI/bge-small-en-v1.5",
                "sparse": "prithivida/Splade_PP_en_v1",
                "late_interaction": "answerdotai/answerai-colbert-small-v1",
                "reranker": "BAAI/bge-reranker-v2-m3",
                "light_reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "baseline_rows": f"graphrag_technical_support/outputs/{args.baseline}",
            "dense_rankings_identical_to_ri4": agree,
        },
        "produced_by": produced_by(reader, "x1", JUDGE if args.judge else None),
        "arms": {
            arm: {
                "summary": summarise(rows[arm], gold),
                "cost": costs[arm],
                "abstention": abstention(replies[arm]),
                **({"judged": judged[arm]} if arm in judged else {}),
            }
            for arm in arms
        },
        "retrieval_only": {
            arm: recall_table({q: rankings[q][arm] for q in gold}, gold) for arm in (*RETRIEVAL_ONLY_ARMS, "dense")
        },
        "comparisons": {
            **{f"{arm}_vs_hipporag2": compare(rows[arm], baseline["hipporag2"], gold) for arm in arms},
            **{f"{arm}_vs_dense": compare(rows[arm], baseline["dense"], gold) for arm in arms},
            # A paid reader at temperature 0 still varies between runs, so when this
            # run answers hipporag2 too, the arms are also compared with that answer.
            **{
                f"{arm}_vs_hipporag2_this_run": compare(rows[arm], rows["hipporag2"], gold)
                for arm in arms
                if "hipporag2" in rows and arm != "hipporag2"
            },
        },
        "rows": {arm: [r.to_dict() for r in rows[arm]] for arm in arms},
    }
    print(f"wrote {write(args.out or 'x1_musique.json', payload)}")
    return 0


# ---------------------------------------------------------------- X3


def serverfault_texts() -> dict[str, str]:
    """Return post text by document identifier for the 10,000-question slice."""
    from aieng.advanced_rag.corpora import serverfault  # noqa: PLC0415

    return {d.doc_id: d.text for d in serverfault.documents(serverfault.fetch_slice())}


def serverfault_dense(texts: dict[str, str]) -> Callable[[str, int], list[str]]:
    """Return ``search(query, k)`` over whole posts with ``bge-small-en-v1.5``."""
    import numpy as np  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    ids = sorted(texts)
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", device="cuda" if _cuda() else "cpu")
    cache = work() / f"serverfault-posts-bge-small-{len(ids)}.npy"
    if cache.is_file():
        matrix = np.load(cache)
    else:
        matrix = np.asarray(embedder.encode([texts[i] for i in ids]), dtype=np.float32)
        np.save(cache, matrix)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)

    def search(query: str, k: int) -> list[str]:
        q = np.asarray(embedder.encode_queries([query])[0], dtype=np.float32)
        scores = matrix @ (q / max(float(np.linalg.norm(q)), 1e-9))
        order = np.argsort(-scores, kind="stable")[:k]
        return [ids[i] for i in order]

    return search


def run_x3(args: argparse.Namespace) -> int:
    """Generate the relational questions, run every arm, and write the X3 results."""
    from gon import relational as rel  # noqa: PLC0415

    texts = serverfault_texts()
    graph = rel.Graph.load(GRAPH_FILE)
    questions = rel.generate(graph, texts)
    if args.questions:
        questions = [q for f in rel.FAMILIES for q in [x for x in questions if x.family == f][: args.questions]]
    ledger = CallLedger("graph-or-not-x3")
    complete = completer(client(args.cache or work() / "responses-onprem.sqlite", args.rpm, ledger), READER, 1024)
    samples = sorted(
        n for n, t in graph.types.items() if t in {"Software", "Command"} and len(graph.entity_docs[n]) > 40
    )
    arms = rel.RelationalArms(
        texts, rel.sqlite_graph(graph), serverfault_dense(texts), complete, samples=samples[:25], workers=args.workers
    )

    results: dict[str, dict[str, Any]] = {arm: {} for arm in arms.names}
    traces: dict[str, dict[str, Any]] = {"text2sql": {}, "text2sql-repair": {}}
    costs: dict[str, dict[str, int]] = {}
    for arm in arms.names:
        before = ledger.to_dict()
        started = time.perf_counter()
        for q in questions:
            answer, trace = arms.answer(arm, q)
            if trace is not None:
                traces[arm][q.qid] = trace
            results[arm][q.qid] = {
                "answer": answer[:200] if isinstance(answer, list) else answer,
                "score": rel.score(q, answer),
            }
        after = ledger.to_dict()
        costs[arm] = {k: after[k] - before[k] for k in ("calls", "total_tokens", "cache_hits")}
        print(f"arm {arm}: {costs[arm]['calls']} calls in {time.perf_counter() - started:.0f}s", flush=True)

    summary = x3_summary(questions, results, traces)
    payload = {
        "test": {
            "questions": len(questions),
            "per_family": {f: sum(q.family == f for q in questions) for f in rel.FAMILIES},
            "seed": rel.SEED,
            "posts": len(texts),
            "graph": "RI #4 ServerFault graph, Qwen3_8-27B direct extraction, 7 entity and 7 relation types",
            "reader": READER,
            "dense": "BAAI/bge-small-en-v1.5 over whole posts, first 512 tokens; top 10 read, top 50 mapped",
            "query_language": "SQLite over entities, mentions, relations; no graph database in the locked environment",
        },
        "produced_by": produced_by(READER, "x3"),
        "summary": summary,
        "costs": costs,
        "text2sql_traces": traces,
        "questions": [q.to_dict() for q in questions],
        "answers": results,
    }
    full = work() / "x3-full.json"
    full.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {full} (with question and answer text, kept out of the repository)")
    print(f"wrote {finalize_x3(payload, args.out or 'x3_serverfault.json')}")
    return 0


def finalize_x3(payload: dict[str, Any] | None = None, name: str = "x3_serverfault.json") -> Path:
    """Write the public X3 file with coverage and the hand-check summary.

    Reads the full payload from the cache folder when none is given, so the hand
    labels in ``outputs/x3_hand_check.json`` can be merged after a run.
    """
    from gon import relational as rel  # noqa: PLC0415

    if payload is None:
        payload = json.loads((work() / "x3-full.json").read_text(encoding="utf-8"))
    questions = [rel.Question(**q) for q in payload["questions"]]
    graph = rel.Graph.load(GRAPH_FILE)
    out = public_x3(payload)
    out["graph_coverage"] = rel.mention_coverage(graph, serverfault_texts(), questions)
    labels = OUT / "x3_hand_check.json"
    if labels.is_file():
        hand = json.loads(labels.read_text(encoding="utf-8"))
        facts, mentions = hand["relation_facts"], hand["lexical_only_mentions"]
        out["hand_check"] = {
            "relation_facts": {"checked": len(facts), "supported": sum(f["supported"] for f in facts)},
            "lexical_only_mentions": {
                "checked": len(mentions),
                "real_mentions": sum(m["real_mention"] for m in mentions),
            },
            "path_connectors": {
                "checked": len(hand.get("path_connectors", [])),
                "meaningful": sum(p["meaningful_link"] for p in hand.get("path_connectors", [])),
            },
            "labels": "x3_hand_check.json",
        }
    return write(name, out)


def public_x3(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the X3 results without question wording, model answers, or generated SQL.

    Questions keep their family, relation, entity labels (graph names, as RI #4's
    committed graph sample holds them), and golds; each answer keeps its size and
    scores.
    """
    out = {k: v for k, v in payload.items() if k not in {"questions", "answers", "text2sql_traces"}}
    out["questions"] = [{k: v for k, v in q.items() if k != "question"} for q in payload["questions"]]
    out["scores"] = {
        arm: {
            qid: {
                "answer_size": (len(a["answer"]) if isinstance(a["answer"], list) else a["answer"]),
                "score": a["score"],
            }
            for qid, a in answers.items()
        }
        for arm, answers in payload["answers"].items()
    }
    out["text2sql_outcomes"] = payload["text2sql_traces"]
    return out


def x3_summary(
    questions: Sequence[Any], results: dict[str, dict[str, Any]], traces: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Return the mean score per arm, family, and gold."""
    out: dict[str, Any] = {}
    for arm, answers in results.items():
        out[arm] = {}
        for family in ("count", "co_mention", "relation", "path"):
            picked = [q for q in questions if q.family == family]
            if not picked:
                continue
            metric = "within_20pct" if family == "count" else "f1"
            cell: dict[str, float] = {}
            for gold in ("graph", "lexical"):
                values = [answers[q.qid]["score"][gold][metric] for q in picked if gold in answers[q.qid]["score"]]
                if values:
                    cell[f"{gold}_{metric}"] = round(sum(values) / len(values), 4)
            out[arm][family] = cell
        if arm in traces and traces[arm]:
            t = traces[arm].values()
            out[arm]["sql_errors"] = sum(x["error"] for x in t)
            out[arm]["sql_empty"] = sum(x["empty"] for x in t)
            out[arm]["sql_failed_after_repair"] = sum(
                bool(x.get("after_repair_error") or x.get("after_repair_empty")) for x in t if x.get("repaired")
            )
    return out


# ---------------------------------------------------------------- X2


def run_x2(args: argparse.Namespace) -> int:
    """Compute the readiness statistics and write ``outputs/x2_readiness.json``."""
    from gon import readiness  # noqa: PLC0415

    payload = readiness.reference_table(args, work=work(), graph_file=GRAPH_FILE, texts=serverfault_texts)
    payload["produced_by"] = produced_by(READER, "x2")
    print(f"wrote {write(args.out or 'x2_readiness.json', payload)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run one experiment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiment", choices=("x1", "x2", "x3", "x3-finalize"))
    parser.add_argument("--questions", type=int, default=None, help="first N questions (X3: per family)")
    parser.add_argument("--cache", type=Path, default=None, help="on-prem response cache")
    parser.add_argument("--paid-cache", type=Path, default=None, help="paid response cache, for the judge")
    parser.add_argument("--judge", action="store_true", help="X1: judge every answer with gemini-3.8-flash")
    parser.add_argument("--device", default="cpu", help="X1: torch device for ranking when none is cached")
    parser.add_argument(
        "--write-inputs", action="store_true", help="X1: write the ranking inputs and stop, to rank on another machine"
    )
    parser.add_argument("--reader", default=None, help=f"X1: answering model (default {READER})")
    parser.add_argument(
        "--arms", nargs="*", default=None, help="X1: answered arms to run, hipporag2 among them (default all but it)"
    )
    parser.add_argument(
        "--answer-prompt",
        choices=("standard", "best-guess"),
        default="standard",
        help="X1: RI #4's answering instruction; standard offers 'unknown', best-guess does not",
    )
    parser.add_argument("--answer-passages", type=int, default=5, help="X1: passages the reader sees (default 5)")
    parser.add_argument(
        "--baseline",
        default="musique_graph_test.json",
        help="X1: RI #4 results file under its outputs/ whose dense and hipporag2 rows the arms are compared with",
    )
    parser.add_argument("--rpm", type=int, default=80)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=None, help="results file name under outputs/")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.experiment == "x3-finalize":
        print(f"wrote {finalize_x3(name=args.out or 'x3_serverfault.json')}")
        return 0
    return {"x1": run_x1, "x2": run_x2, "x3": run_x3}[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())
