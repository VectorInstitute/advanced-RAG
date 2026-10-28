#!/usr/bin/env python3
"""Run the MuSiQue graph test: five retrieval arms over 300 questions by hop.

Every arm ranks the same 11,656 passages for the same 300 questions (150 two-hop,
75 three-hop, 75 four-hop, a fixed-seed draw), and the same model answers each
question from the arm's top five passages. The report gives, per arm and hop count,
the share of questions with every supporting passage in the top 5 and top 10, exact
match and F1, and what each arm spent in model calls and tokens. Each graph arm is
also compared with the dense baseline on the same questions, with a paired bootstrap
on F1 and exact McNemar tests, so the write-up can say whether a difference is more
than noise.

The two graph indexes load from the artifact cache, where ``build_indexes.py`` stored
them, by the key of their build model (``--build-model``, the reference model by
default). Questions are answered on a separate model (``--answer-model``, or the
generation model of ``--family`` from ``choose_models``, the on-prem set by
default), so answering on a paid model still loads the reference indexes. The
results record both models and the digest of each index they read. Nothing in
the results holds question or passage text.

Examples
--------
Run everything and write the results::

    $ python run_musique_test.py --out outputs/musique_graph_test.json

Answer over the reference indexes on another model::

    $ python run_musique_test.py --answer-model gpt-6-luna --out /tmp/answers.json

Answer on the participants' default model, with a response cache of its own,
and judge every answer with the shared judge::

    $ ADVANCED_RAG_SPEND_LIMIT_USD=10 python run_musique_test.py --family google \
        --arms dense dense-iterative hipporag2 --judge --judge-model gemini-3.8-flash \
        --cache paid-responses.sqlite --out outputs/musique_graph_test_gemini.json

Only the selected arms' indexes are restored, so a run without ``lightrag``
never reads the LightRAG index.

Answer another run's stored rankings on a second model, so both read the same
five passages and only the answering step differs::

    $ ADVANCED_RAG_SPEND_LIMIT_USD=5 python run_musique_test.py --family google \
        --arms dense dense-iterative hipporag2 \
        --rankings-from outputs/musique_graph_test.json \
        --judge --judge-model gemini-3.8-flash --cache paid-responses.sqlite \
        --out outputs/musique_reader_swap_gemini.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import TieredResultCache, fetch_directory
from aieng.advanced_rag.corpora.musique import (
    MUSIQUE_REVISION,
    TEST_PER_HOP,
    TEST_SEED,
    MusiqueQuestion,
    Passage,
    load_musique,
    stratified_test_set,
)
from aieng.advanced_rag.evaluation.answers import judge_answer
from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1
from aieng.advanced_rag.graphrag.adapters.hipporag2 import HippoRAG2Retriever
from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.graphrag.proxy import (
    CallLedger,
    RateLimiter,
    ResponseCache,
    async_llm,
    govern,
)
from aieng.advanced_rag.models import ROLE_ENV, choose_models, endpoint_of, request_params
from aieng.advanced_rag.provenance import Provenance
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from build_indexes import (
    BASE_URL,
    DEPRECATED_MODEL_ENV,
    ENCODER,
    HIPPORAG_KEYWORD,
    REFERENCE_EXTRACTION_MODEL,
    api_key,
    cache_key,
    embedder_function,
    work_root,
)
from musique_arms import (
    ANSWER_PROMPTS,
    Arm,
    DenseArm,
    HybridArm,
    IterativeDenseArm,
    LightRagArm,
    ReplayArm,
    SyncEvidenceArm,
    WithholdArm,
    answer,
    passage_id_map,
    split_flag,
)
from musique_report import Row, compare, summarise
from openai import AsyncOpenAI, OpenAI


ANSWER_PASSAGES = 5
RANK_DEPTH = 10
ARM_NAMES = ("dense", "dense-iterative", "hipporag2", "lightrag", "hybrid")
#: Which passages ``--withhold`` drops from every ranking: none, every supporting
#: passage, only the one that states the final answer (the last decomposition
#: step), or every passage, so the reader answers from no passages at all.
WITHHOLD = ("none", "supporting", "final", "everything")
#: Arms that read each graph index; an index is restored only when one runs.
NEEDS_INDEX = {"hipporag2": ("hipporag2", "hybrid"), "lightrag": ("lightrag",)}


def answer_model(cli: str | None, family: str) -> str:
    """Return the answering model: the flag, then the family's generation model.

    ``choose_models`` reads ``ADVANCED_RAG_GENERATION_MODEL``. The deprecated
    ``LLM_MODEL`` is honoured for answering only when neither is set.
    """
    if cli:
        return cli
    if not os.environ.get(ROLE_ENV["generation"]) and os.environ.get(DEPRECATED_MODEL_ENV):
        warnings.warn(
            f"{DEPRECATED_MODEL_ENV} is deprecated: set {ROLE_ENV['generation']} or "
            "--answer-model for the answering model",
            DeprecationWarning,
            stacklevel=2,
        )
        return os.environ[DEPRECATED_MODEL_ENV].strip()
    model: str = choose_models(family).generation.model
    return model


def judging_model(cli: str | None, family: str) -> str:
    """Return the judging model: the flag, then the family's judging model."""
    if cli:
        return cli
    model: str = choose_models(family).judging.model
    return model


def produced_by(model: str, judge: str | None = None) -> dict[str, Any]:
    """Return the ``produced_by`` block for a results file.

    ``label`` is ``reference trajectories`` for an on-prem answering model and
    ``paid-model rerun`` for any other, so a table never reads as the wrong
    model's.
    """
    onprem = endpoint_of(model) == "vector-cluster"
    block: dict[str, Any] = {
        "model": model,
        "endpoint": "vector-cluster" if onprem else "vector-proxy-paid",
        "producer": "run_musique_test.py",
        "label": "reference trajectories" if onprem else "paid-model rerun",
    }
    if judge:
        block["judge"] = judge
    return block


def restore(
    method: str,
    count: int,
    total: int,
    override: Path | None,
    dest: Path,
    *,
    build_model: str = REFERENCE_EXTRACTION_MODEL,
) -> Path:
    """Return a directory holding an index, restored from the artifact cache.

    The key names the build model, which stays the reference model unless the
    caller names another build, so changing the answering model never changes
    which index is restored.

    Raises
    ------
    SystemExit
        If the cache holds no index for this build.
    """
    if override is not None:
        return override
    key = cache_key(method, count, total, build_model)
    restored = fetch_directory(TieredResultCache(), key, dest / method, replace=True)
    if restored is None:
        raise SystemExit(
            f"no {method} index of {count} passages built with {build_model} in the "
            "artifact cache; fetch the reference indexes or build one with build_indexes.py"
        )
    return Path(restored)


async def run_arm(
    arm: Arm,
    questions: Sequence[MusiqueQuestion],
    *,
    complete: Callable[[str], str],
    text_of: Callable[[str], str],
    ledger: CallLedger,
    workers: int,
    answers: dict[str, str] | None = None,
    template: str = ANSWER_PROMPTS["standard"],
) -> tuple[list[Row], dict[str, Any]]:
    """Rank and answer every question with one arm, and say what it cost.

    ``answers``, when given, receives each question's answer text by question
    identifier, for the judge. The rows never hold it. An answer marked unsure
    is scored without the mark, and ``answers`` keeps the mark.
    """
    before = ledger.to_dict()
    gate = asyncio.Semaphore(workers)

    async def one(question: MusiqueQuestion) -> Row:
        async with gate:
            ranked = await arm.rank(question.question, RANK_DEPTH)
            reply = await answer(complete, question.question, ranked[:ANSWER_PASSAGES], text_of, template)
        if answers is not None:
            answers[question.question_id] = reply
        golds = (question.answer, *question.answer_aliases)
        scored, _ = split_flag(reply)
        return Row(
            question.question_id,
            question.hops,
            tuple(ranked),
            exact_match(scored, golds),
            token_f1(scored, golds),
        )

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(q) for q in questions))
    after = ledger.to_dict()
    spent = {
        key: after[key] - before[key]
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "cache_hits")
    }
    cost = {
        **spent,
        "calls_per_question": round(spent["calls"] / len(questions), 2),
        "tokens_per_question": round(spent["total_tokens"] / len(questions), 1),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    return list(rows), cost


def judge_rows(
    rows: Sequence[Row],
    answers: dict[str, str],
    questions: Sequence[MusiqueQuestion],
    *,
    complete: Callable[[str], str],
    text_of: Callable[[str], str],
) -> dict[str, bool]:
    """Judge each answer against the reference, given the passages it was answered from.

    The evidence is the arm's top five passages, which is what the answering
    model read. A judge reply that does not parse counts as incorrect.
    """
    by_id = {q.question_id: q for q in questions}
    verdicts: dict[str, bool] = {}
    for row in rows:
        question = by_id[row.question_id]
        evidence = [
            Evidence(text=text_of(pid), provenance=Provenance(doc_id=pid)) for pid in row.ranked[:ANSWER_PASSAGES]
        ]
        try:
            verdict = judge_answer(
                question.question_id,
                question.question,
                answers.get(row.question_id) or "(no answer)",
                question.answer,
                evidence,
                complete,
            )
            verdicts[row.question_id] = verdict.correct
        except ValueError:
            verdicts[row.question_id] = False
    return verdicts


def declines(reply: str) -> bool:
    """Return True when an answer declines: "unknown" or nothing."""
    return reply.strip().strip(".\"'").lower() in ("", "unknown")


def abstention(answers: dict[str, str]) -> float:
    """Return the share of answers that decline: "unknown" or nothing."""
    declined = [declines(a) for a in answers.values()]
    return round(sum(declined) / len(declined), 4) if declined else 0.0


def outcomes(answers: dict[str, str]) -> dict[str, str]:
    """Label each answer ``declined``, ``flagged`` (marked unsure), or ``answered``.

    The labels hold no answer text, so they can be committed with the rows.
    """
    labels: dict[str, str] = {}
    for qid, reply in sorted(answers.items()):
        text, flagged = split_flag(reply)
        labels[qid] = "declined" if declines(text) else "flagged" if flagged else "answered"
    return labels


def withheld_passages(questions: Sequence[MusiqueQuestion], mode: str) -> dict[str, tuple[str, ...]]:
    """Return, by question text, the supporting passages ``--withhold`` drops."""
    if mode in ("none", "everything"):
        return {}
    if mode == "supporting":
        return {q.question: q.supporting_ids for q in questions}
    if mode == "final":
        return {q.question: q.supporting_ids[-1:] for q in questions}
    raise ValueError(f"unknown withhold mode {mode!r}")


def judged_shares(rows: Sequence[Row], verdicts: dict[str, bool]) -> dict[str, float]:
    """Return the judged-correct share over all questions and by hop count."""
    out: dict[str, float] = {}
    for group, keep in (("all", None), ("2hop", 2), ("3hop", 3), ("4hop", 4)):
        picked = [r for r in rows if keep is None or r.hops == keep]
        if picked:
            out[group] = round(sum(bool(verdicts.get(r.question_id)) for r in picked) / len(picked), 4)
    return out


def withhold_arms(arms: dict[str, Arm], questions: Sequence[MusiqueQuestion], mode: str) -> dict[str, Arm]:
    """Wrap every arm so the reader never sees the passages ``mode`` withholds.

    The reader reads the next-ranked passages in place of the dropped ones, so
    a question becomes unanswerable from its passages by construction.
    """
    if mode == "none":
        return arms
    if mode == "everything":
        return {name: WithholdArm(arm, {}, everything=True) for name, arm in arms.items()}
    withheld = withheld_passages(questions, mode)
    return {name: WithholdArm(arm, withheld) for name, arm in arms.items()}


async def graph_arms(
    args: argparse.Namespace,
    restored: Sequence[str],
    *,
    passages: Sequence[Passage],
    dense: DenseArm,
    embedder: SentenceTransformerEmbedder,
    governed: tuple[Any, ResponseCache, RateLimiter, CallLedger],
    model: str,
) -> dict[str, Arm]:
    """Restore the graph indexes named in ``restored`` and return their arms.

    An index that no selected arm reads is never restored or opened.
    """
    sync_client, cache, limiter, ledger = governed
    total = len(passages)
    index_count = args.index_passages or total
    run_dir = args.run_dir or work_root() / "run"
    arms: dict[str, Arm] = {}
    if "hipporag2" in restored:
        hippo_dir = restore("hipporag2", index_count, total, args.hipporag_dir, run_dir, build_model=args.build_model)
        hippo = SyncEvidenceArm(
            "hipporag2",
            HippoRAG2Retriever.build(
                kg_dir=str(hippo_dir),
                llm_client=sync_client,
                llm_model=model,
                encoder_model=ENCODER,
                keyword=HIPPORAG_KEYWORD,
                include_concept=False,
                include_events=False,
            ),
            id_map=passage_id_map(hippo_dir / "kg_graphml" / f"{HIPPORAG_KEYWORD}_graph.graphml", passages),
        )
        arms["hipporag2"] = hippo
        arms["hybrid"] = HybridArm(dense, hippo)

    if "lightrag" in restored:
        light_dir = restore("lightrag", index_count, total, args.lightrag_dir, run_dir, build_model=args.build_model)
        llm = async_llm(
            AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
            model,
            cache=cache,
            limiter=limiter,
            ledger=ledger,
            max_retries=16,
        )
        light = await LightRAGRetriever.build(str(light_dir), llm, embedder_function(embedder), mode="mix")
        arms["lightrag"] = LightRagArm(light)
    return arms


def replay_arms(path: Path, names: Sequence[str], questions: Sequence[MusiqueQuestion]) -> dict[str, Arm]:
    """Return arms that replay the rankings a results file stored, by question text."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    text_by_id = {q.question_id: q.question for q in questions}
    return {
        name: ReplayArm(
            name,
            {text_by_id[r["question_id"]]: r["ranked"] for r in stored["rows"][name] if r["question_id"] in text_by_id},
        )
        for name in names
    }


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    """Build the arms, run them, and return the full results."""
    passages, all_questions = load_musique()
    passages = passages[: args.passages] if args.passages else passages
    known = {p.passage_id for p in passages}
    if args.passages:
        # A smoke run over a few passages uses any question they can answer.
        questions = [q for q in all_questions if set(q.supporting_ids) <= known]
    else:
        questions = stratified_test_set(all_questions)
    if args.questions:
        questions = questions[: args.questions]
    gold = {q.question_id: q.supporting_ids for q in questions}
    by_id: dict[str, Passage] = {p.passage_id: p for p in passages}

    def text_of(passage_id: str) -> str:
        found = by_id[passage_id]
        return f"{found.title}\n{found.text}"

    ledger = CallLedger("musique-test")
    cache = ResponseCache(args.cache or work_root() / "responses-test.sqlite")
    limiter = RateLimiter(per_minute=args.rpm)
    sync_client = govern(
        OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        cache=cache,
        limiter=limiter,
        ledger=ledger,
        max_retries=16,
    )

    model = args.answer_model
    # govern injects each family's extra_body per call, and keeping it out of the
    # request keeps the cache keys of the reference answers unchanged.
    params = request_params(model, max_tokens=256, temperature=0.0)
    params.pop("extra_body", None)

    def complete(prompt: str, *, using: str = model, extra: dict[str, Any] = params) -> str:
        response = sync_client.chat.completions.create(
            model=using,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        return str(response.choices[0].message.content or "")

    embedder = SentenceTransformerEmbedder(ENCODER)
    dense = DenseArm(passages, embedder, cache=work_root() / f"dense-{len(passages)}.npy")
    total = len(passages)
    arms: dict[str, Arm] = {"dense": dense, "dense-iterative": IterativeDenseArm(dense, complete, text_of)}

    selected = [name for name in ARM_NAMES if not args.arms or name in args.arms]
    restored = [method for method, users in NEEDS_INDEX.items() if set(users) & set(selected)]
    if args.rankings_from is not None:
        # Replay: every selected arm returns the stored run's ranking, so no
        # index is restored and no retrieval call is made.
        arms.update(replay_arms(args.rankings_from, selected, questions))
        restored = []
    arms.update(
        await graph_arms(
            args,
            restored,
            passages=passages,
            dense=dense,
            embedder=embedder,
            governed=(sync_client, cache, limiter, ledger),
            model=model,
        )
    )

    arms = withhold_arms(arms, questions, args.withhold)

    results: dict[str, list[Row]] = {}
    costs: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, str]] = {}
    for name in selected:
        print(f"arm {name}: {len(questions)} questions", flush=True)
        answers[name] = {}
        results[name], costs[name] = await run_arm(
            arms[name],
            questions,
            complete=complete,
            text_of=text_of,
            ledger=ledger,
            workers=args.workers,
            answers=answers[name],
            template=ANSWER_PROMPTS[args.answer_prompt],
        )
        print(f"  done in {costs[name]['wall_seconds']}s, {costs[name]['calls']} calls", flush=True)

    judged: dict[str, dict[str, float]] = {}
    if args.judge:
        judge_params = request_params(args.judge_model, max_tokens=300, temperature=0.0)
        judge_params.pop("extra_body", None)

        def judge_complete(prompt: str) -> str:
            return complete(prompt, using=args.judge_model, extra=judge_params)

        for name in selected:
            print(f"judging {name} with {args.judge_model}", flush=True)
            verdicts = await asyncio.to_thread(
                judge_rows, results[name], answers[name], questions, complete=judge_complete, text_of=text_of
            )
            judged[name] = judged_shares(results[name], verdicts)

    comparisons = {
        f"{name}_vs_dense": compare(results[name], results["dense"], gold)
        for name in selected
        if name != "dense" and "dense" in results
    }
    return {
        "test": {
            "questions": len(questions),
            "per_hop": {str(k): v for k, v in TEST_PER_HOP.items()},
            "seed": TEST_SEED,
            "passages": total,
            "dataset_revision": MUSIQUE_REVISION,
            "model": model,
            "build_model": args.build_model,
            "index_keys": {
                method: cache_key(method, args.index_passages or total, total, args.build_model).digest
                for method in restored
            },
            "judge": args.judge_model if args.judge else None,
            "encoder": ENCODER,
            "answer_passages": ANSWER_PASSAGES,
            "rank_depth": RANK_DEPTH,
            "rankings_from": args.rankings_from.name if args.rankings_from is not None else None,
            "answer_prompt": args.answer_prompt,
            "withhold": args.withhold,
            "hipporag2": "atlas-rag variant, entity-relation stage only",
            "lightrag": "mix mode, default top_k of 40",
        },
        "produced_by": produced_by(model, args.judge_model if args.judge else None),
        "arms": {
            name: {
                "summary": summarise(results[name], gold),
                "cost": costs[name],
                "abstained": abstention(answers[name]),
                "flagged": round(
                    sum(v == "flagged" for v in outcomes(answers[name]).values()) / max(len(answers[name]), 1), 4
                ),
                **({"judged": judged[name]} if name in judged else {}),
            }
            for name in selected
        },
        "comparisons_with_dense": comparisons,
        "rows": {name: [r.to_dict() for r in results[name]] for name in selected},
        "outcomes": {name: outcomes(answers[name]) for name in selected},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the test and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arms", nargs="*", default=None, choices=ARM_NAMES)
    parser.add_argument("--questions", type=int, default=None, help="first N test questions")
    parser.add_argument("--passages", type=int, default=None, help="restrict to the first N")
    parser.add_argument("--index-passages", type=int, default=None)
    parser.add_argument("--hipporag-dir", type=Path, default=None)
    parser.add_argument("--lightrag-dir", type=Path, default=None)
    parser.add_argument(
        "--answer-model",
        default=None,
        help="answering model (default: the generation model of --family)",
    )
    parser.add_argument(
        "--family",
        default="onprem",
        help="model family whose generation model answers (default: onprem)",
    )
    parser.add_argument(
        "--build-model",
        default=REFERENCE_EXTRACTION_MODEL,
        help=f"model the restored indexes were built with (default: {REFERENCE_EXTRACTION_MODEL})",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rpm", type=int, default=80)
    parser.add_argument("--cache", type=Path, default=None, help="response cache file; a paid model needs its own")
    parser.add_argument("--run-dir", type=Path, default=None, help="where restored indexes are unpacked")
    parser.add_argument("--judge", action="store_true", help="judge every answer with the judging model")
    parser.add_argument(
        "--rankings-from",
        type=Path,
        default=None,
        help="answer the rankings stored in this results file instead of retrieving",
    )
    parser.add_argument(
        "--answer-prompt",
        default="standard",
        choices=sorted(ANSWER_PROMPTS),
        help="answering instruction: standard offers 'unknown', best-guess does not, "
        "flagged always answers and marks a guess '(unsure)'",
    )
    parser.add_argument(
        "--withhold",
        default="none",
        choices=WITHHOLD,
        help="drop supporting passages from every ranking, to make the questions unanswerable: "
        "all of them (supporting), the one stating the final answer (final), or every passage, "
        "for a closed-book answer (everything)",
    )
    parser.add_argument("--judge-model", default=None, help="judging model (default: the judging model of --family)")
    args = parser.parse_args(argv)
    args.answer_model = answer_model(args.answer_model, args.family)
    args.judge_model = judging_model(args.judge_model, args.family)
    paid = endpoint_of(args.answer_model) != "vector-cluster" or (
        args.judge and endpoint_of(args.judge_model) != "vector-cluster"
    )
    if paid and args.cache is None:
        parser.error("a paid model needs --cache, so its responses stay out of the on-prem cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if paid:
        # Opening the session installs the ADVANCED_RAG_SPEND_LIMIT_USD ceiling.
        spend.session()

    results = asyncio.run(amain(args))
    text = json.dumps(results, indent=1) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for name, arm in results["arms"].items():
        overall = arm["summary"]["all"]
        print(
            f"{name:16s} all-supporting@5 {overall['all_supporting_recall_at_5']:.3f} "
            f"@10 {overall['all_supporting_recall_at_10']:.3f} "
            f"EM {overall['exact_match']:.3f} F1 {overall['f1']:.3f} "
            f"| {arm['cost']['calls_per_question']} calls/q"
            + (f" | judged {arm['judged']['all']:.3f}" if "judged" in arm else "")
        )
    if paid:
        print(f"spend guard estimate ${spend.session().estimated_usd:,.2f}")
        print(spend.session().report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
