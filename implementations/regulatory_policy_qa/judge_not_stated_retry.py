#!/usr/bin/env python3
"""Judge the answers a retry on the generator's "not stated" signal recovers.

The harvest in ``outputs/traces_contract-nli_llm.json`` records, for every
ContractNLI question, two attempts: the original retrieval and one after an
``LLMRewriter`` rewrite, each with a generation. A policy that retries only when
the first generation abstains reads its decision straight off those traces. The
traces keep the abstain-or-answer decision and not the answer text, so the
decision can be replayed for free but its correctness cannot.

This script regenerates the answers on the recorded evidence (same chunk
identifiers, same order, same prompt, the original question), judges each
answered answerable question against the ContractNLI gold label with the
correctness and grounding passes in ``graphrag_null/judge_lab.py``, and scores
both arms by balanced accuracy where an answer counts only when it is correct.

The harvest's own response cache was not kept, so the regeneration is a second
run at temperature zero. Where it disagrees with the trace on whether to
abstain, the trace decision is kept for the decision metrics and the question
is counted as unreproduced for the judged ones.

Two optional arms separate the rewrite from a second draw. ``--resample-arm``
retries each first-attempt refusal with the identical prompt on a fresh call.
``--reversed-arm`` retries it on attempt 1's top five in reverse order. Neither
rewrites or retrieves again, so each costs one call per refusal.

Every model is a parameter and every response is cached, so a re-run with
``--cache-only`` costs nothing and needs no key. Embeddings are not used.

Examples
--------
Judge with the shared generation/judging roles, 30 requests per minute for
each model, and both control arms. Cache-only replay keeps the reference models::

    python3 judge_not_stated_retry.py --resample-arm --reversed-arm

Verify the committed report from the cache alone::

    python3 judge_not_stated_retry.py --resample-arm --reversed-arm --cache-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "aieng-advanced-rag"))
sys.path.insert(0, str(HERE / "graphrag_null"))
sys.path.insert(0, str(HERE))

from aieng.advanced_rag import spend  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.corpora import contract_nli  # noqa: E402
from aieng.advanced_rag.evaluation.multihop import mcnemar_p  # noqa: E402
from aieng.advanced_rag.evaluation.sweep import QueryTrace  # noqa: E402
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.models import choose_models  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.pipeline.generate import generate  # noqa: E402
from aieng.advanced_rag.pipeline.spine import evidence_from_hits  # noqa: E402
from aieng.advanced_rag.proxy import (  # noqa: E402 - use the source checkout after bootstrap
    resolve_api_key,
    resolve_base_url,
)
from aieng.advanced_rag.retrieval.corpus import Query  # noqa: E402
from aieng.advanced_rag.retrieval.retrievers import Retrieved  # noqa: E402
from aieng.advanced_rag.runs import Provenance  # noqa: E402
from judge_lab import MAX_CHARS, checked_complete, judge_query, references  # noqa: E402
from null_lab import Meter, MeteredCompleter, ResponseCache, proxy_batch  # noqa: E402
from sweep_lab import contract_nli_slice  # noqa: E402


OUTPUTS = HERE / "outputs"
DEFAULT_TRACES = OUTPUTS / "traces_contract-nli_llm.json"
DEFAULT_OUT = OUTPUTS / "not_stated_retry.json"
DEFAULT_STORE = Path.home() / ".cache" / "aieng-advanced-rag" / "not_stated_retry"
GENERATOR_MODEL = "Qwen3_6-35B-A3B"
JUDGE_MODEL = "Qwen3_8-Flash-Next"
DOCUMENTS = 40
EVIDENCE_BUDGET = 5
REVERSED = 2  # job index for the reversed-order explore arm

FINDING = (
    "Scored by judged correctness, the retry on the generator's not-stated answer beats "
    "generator-only, and a retry that re-sends the identical first-attempt prompt on a fresh "
    "call matches it question for question at fewer calls. The rewrite and the second "
    "retrieval add nothing measurable here: the gain is a second draw from a generator whose "
    "abstain-or-answer decision changes on about one answerable question in ten when the "
    "same prompt is sent again at temperature zero. The harvest cached responses on the exact "
    "request, so inside the traces an identical retry prompt always returned the first "
    "refusal, which hid this. Generation on every attempt uses the original question, so the "
    "rewrite changes only which chunks reach the generator and in what order."
)
WHAT_WOULD_MAKE_IT_WIN = (
    "The retry action would have to add information the first draw lacked, which needs "
    "first-pass retrieval that misses recoverable evidence. MuSiQue 2-hop (dense "
    "all-supporting recall@5 of 0.28) meets that. The cheapest test is a two-attempt "
    "harvest with no grader on RI #4's 150 2-hop questions, about 450 calls, with this "
    "script's resample arm as the control that the rewrite retry must beat."
)


@dataclass(frozen=True)
class Generated:
    """One regenerated answer on one recorded attempt's evidence."""

    query_id: str
    attempt: int
    abstained: bool
    text: str


def decisions(traces: Sequence[QueryTrace]) -> tuple[dict[str, bool], dict[str, bool], dict[str, int]]:
    """Return both arms' abstentions and the attempt each retry answers from."""
    control = {t.query_id: t.attempts[0].generated_abstained for t in traces}
    retry = {t.query_id: t.attempts[0].generated_abstained and t.attempts[-1].generated_abstained for t in traces}
    used = {t.query_id: (1 if t.attempts[0].generated_abstained and len(t.attempts) > 1 else 0) for t in traces}
    return control, retry, used


def balanced(traces: Sequence[QueryTrace], right: Callable[[QueryTrace], bool]) -> dict[str, float]:
    """Average the right-rate over nulls and over answerable questions."""
    nulls = [t for t in traces if t.is_null]
    answerable = [t for t in traces if not t.is_null]
    null_rate = sum(map(right, nulls)) / len(nulls)
    answer_rate = sum(map(right, answerable)) / len(answerable)
    return {
        "balanced_accuracy": round((null_rate + answer_rate) / 2, 4),
        "null_right_rate": round(null_rate, 4),
        "answerable_right_rate": round(answer_rate, 4),
    }


def paired(
    traces: Sequence[QueryTrace], arm: Callable[[QueryTrace], bool], control: Callable[[QueryTrace], bool]
) -> dict[str, Any]:
    """Return discordant counts and the exact McNemar p-value."""
    b = sum(arm(t) and not control(t) for t in traces)
    c = sum(control(t) and not arm(t) for t in traces)
    return {"arm_right_only": b, "control_right_only": c, "mcnemar_p": round(mcnemar_p(b, c), 5)}


def completer(
    model: str, store: Path, env: Path, rpm: int, *, cache_only: bool, max_tokens: int = 512
) -> MeteredCompleter:
    """Return a cached, rate-limited, thinking-off completer for one model."""
    connection: dict[str, Any] = {}
    if not cache_only:
        spend.session()
        connection = {"rpm": rpm, "timeout": 300, "api_key": resolve_api_key(env), "base_url": resolve_base_url(env)}
    batch = proxy_batch(model, ResponseCache(store / f"{model}.sqlite"), cache_only=cache_only, **connection)
    return MeteredCompleter(batch, meter=Meter(), max_tokens=max_tokens)


def attempt_evidence(trace: QueryTrace, attempt: int, chunks: Mapping[str, Any]) -> list[Evidence]:
    """Rebuild the recorded evidence for one attempt, in recorded order."""
    hits = [Retrieved(chunk_id=chunk_id, score=0.0) for chunk_id in trace.attempts[attempt].evidence_ids]
    evidence: list[Evidence] = evidence_from_hits(hits, chunks)
    return evidence


def digest(text: str) -> str:
    """Hash answer text, so the report identifies answers without carrying them."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def produced_by(generator: str, judge: str, cold_calls: Mapping[str, int]) -> dict[str, Any]:
    """Return the result's ``produced_by`` block, named for the generator.

    The block has one ``model`` slot. The retry policy changes the generator's
    answers, so the generator takes the slot and the judge sits in the note and
    in ``usage``. ``cold_calls`` maps each role to its calls for a cold run.
    """
    usage: dict[str, dict[str, int]] = {}
    for model, role in ((generator, "generator"), (judge, "judge")):
        usage.setdefault(model, {"calls": 0})["calls"] += cold_calls[role]
    block: dict[str, Any] = Provenance.for_model(
        generator,
        producer="regulatory_policy_qa/judge_not_stated_retry.py",
        usage=usage,
        note=(
            f"Generator {generator} regenerates the answers and judge {judge} grades "
            "them, both with thinking off. Calls are the main comparison's cold run; "
            "the explore arms add their own, counted under each arm's calls."
        ),
    ).to_dict()
    return block


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0914, PLR0915 - one auditable pass
    """Regenerate, judge, and score both arms."""
    traces = [QueryTrace.from_dict(row) for row in json.loads(args.traces.read_text())]
    path = contract_nli.fetch_contract_nli()
    eval_slice = contract_nli_slice(path, documents=args.documents)
    chunks = {chunk.chunk_id: chunk for chunk in eval_slice.chunks}
    queries: dict[str, Query] = {q.query_id: q for q in eval_slice.queries}
    if {t.query_id for t in traces} != set(queries):
        raise ValueError("trace query IDs do not match the ContractNLI slice")
    gold = references(path, args.documents)

    control, retry, used = decisions(traces)
    by_id = {t.query_id: t for t in traces}

    # Regenerate attempt 1 for every answerable question, attempt 2 for every
    # answerable first-attempt refusal, and attempt 2 for the nulls the retry
    # turns into answers. Nulls answered at attempt 1 are wrong whatever they say.
    jobs = [(t.query_id, 0) for t in traces if not t.is_null]
    jobs += [(t.query_id, 1) for t in traces if not t.is_null and t.attempts[0].generated_abstained]
    jobs += [
        (t.query_id, 1) for t in traces if t.is_null and t.attempts[0].generated_abstained and not retry[t.query_id]
    ]
    if args.resample_arm:
        # Control arm: retry every first-attempt refusal with the identical
        # prompt on a fresh call. Answerable attempt-1 prompts are already in
        # the job list; this adds the null ones.
        jobs += [(t.query_id, 0) for t in traces if t.is_null and t.attempts[0].generated_abstained]
    if args.reversed_arm:
        # Explore arm: retry every first-attempt refusal on attempt 1's own top
        # five in reverse order, with no rewrite and no second retrieval.
        jobs += [(t.query_id, REVERSED) for t in traces if t.attempts[0].generated_abstained]

    gen = completer(args.generator_model, args.store, args.env, args.rpm, cache_only=args.cache_only)
    judge = completer(args.judge_model, args.store, args.env, args.rpm, cache_only=args.cache_only)
    checked = checked_complete(judge)

    def one(job: tuple[str, int]) -> tuple[Generated, Any]:
        query_id, attempt = job
        trace = by_id[query_id]
        if attempt == REVERSED:
            evidence = attempt_evidence(trace, 0, chunks)[:EVIDENCE_BUDGET][::-1]
        else:
            evidence = attempt_evidence(trace, attempt, chunks)
        answer = generate(query_id, trace.question, evidence, gen, budget=EVIDENCE_BUDGET, max_chars=MAX_CHARS)
        record = Generated(query_id, attempt, answer.abstained, answer.text)
        verdict = None
        if not answer.abstained and not trace.is_null:
            verdict = judge_query(queries[query_id], answer.text, gold[query_id], evidence[:EVIDENCE_BUDGET], checked)
        return record, verdict

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, jobs))
    generated = {(r.query_id, r.attempt): (r, v) for r, v in results}

    def judged_right(
        arm: Mapping[str, bool], attempt_of: Callable[[str], int], *, regen_decision: bool
    ) -> Callable[[QueryTrace], bool]:
        """Return a scorer that credits an answerable question only when correct."""

        def right(t: QueryTrace) -> bool:
            if t.is_null:
                return arm[t.query_id]
            record, verdict = generated[(t.query_id, attempt_of(t.query_id))]
            if not regen_decision and arm[t.query_id]:
                return False
            if regen_decision and record.abstained:
                return False
            # A regenerated refusal under a trace decision to answer is not reproduced.
            # It counts as wrong, the conservative reading for both arms alike.
            if record.abstained or verdict is None:
                return False
            return bool(verdict.correctness.correct and not verdict.correctness.abstained)

        return right

    def control_attempt(_: str) -> int:
        return 0

    def retry_attempt(query_id: str) -> int:
        return used[query_id]

    # The retry arm under regenerated decisions: answer at attempt 1 unless the
    # regenerated attempt 1 refused, then attempt 2. Attempt 2 was regenerated
    # only for trace refusals, so a regenerated-only refusal falls back to it
    # when available and otherwise stays a refusal.
    def regen_retry_attempt(query_id: str) -> int:
        first, _ = generated[(query_id, 0)]
        return 1 if first.abstained and (query_id, 1) in generated else 0

    decision_right = {
        "generator_only": lambda t: control[t.query_id] == t.is_null,
        "retry_on_not_stated": lambda t: retry[t.query_id] == t.is_null,
    }
    judged = {
        "generator_only": judged_right(control, control_attempt, regen_decision=False),
        "retry_on_not_stated": judged_right(retry, retry_attempt, regen_decision=False),
    }
    regen = {
        "generator_only": judged_right(control, control_attempt, regen_decision=True),
        "retry_on_not_stated": judged_right(retry, regen_retry_attempt, regen_decision=True),
    }

    answerable = [t for t in traces if not t.is_null]
    recovered = [t for t in answerable if t.attempts[0].generated_abstained and not retry[t.query_id]]
    lost_nulls = [t for t in traces if t.is_null and t.attempts[0].generated_abstained and not retry[t.query_id]]

    def tally(group: Sequence[QueryTrace], attempt: int) -> dict[str, int]:
        rows = [generated[(t.query_id, attempt)] for t in group]
        answered = [(r, v) for r, v in rows if not r.abstained]
        correct = sum(bool(v and v.correctness.correct and not v.correctness.abstained) for _, v in answered)
        grounded_wrong = sum(bool(v and not v.correctness.correct and v.grounding.grounded) for _, v in answered)
        return {
            "questions": len(group),
            "regenerated_answered": len(answered),
            "regenerated_abstained": len(rows) - len(answered),
            "correct": correct,
            "confidently_wrong": len(answered) - correct,
            "confidently_wrong_but_grounded": grounded_wrong,
        }

    attempt1_answered = [t for t in answerable if not t.attempts[0].generated_abstained]
    reproduction = {
        "answerable_attempt1_agree": sum(
            generated[(t.query_id, 0)][0].abstained == t.attempts[0].generated_abstained for t in answerable
        ),
        "answerable_attempt1_total": len(answerable),
        "attempt2_agree": sum(
            generated[(q, 1)][0].abstained == by_id[q].attempts[1].generated_abstained for q, a in generated if a == 1
        ),
        "attempt2_total": sum(1 for _, a in generated if a == 1),
    }

    per_question = []
    for t in recovered + lost_nulls:
        record, verdict = generated[(t.query_id, 1)]
        per_question.append(
            {
                "query_id": t.query_id,
                "is_null": t.is_null,
                "regenerated_abstained": record.abstained,
                "answer_sha256_16": digest(record.text),
                "correct": None if verdict is None else verdict.correctness.correct,
                "grounded": None if verdict is None else verdict.grounding.grounded,
                "reason": None if verdict is None else verdict.correctness.reason,
                "attempt1_top5": list(t.attempts[0].evidence_ids[:EVIDENCE_BUDGET]),
                "attempt2_top5": list(t.attempts[1].evidence_ids[:EVIDENCE_BUDGET]),
                "gold_in_attempt1_top5": bool(set(t.attempts[0].evidence_ids[:EVIDENCE_BUDGET]) & t.gold_chunk_ids),
                "gold_in_attempt2_top5": bool(set(t.attempts[1].evidence_ids[:EVIDENCE_BUDGET]) & t.gold_chunk_ids),
            }
        )

    report_extra: dict[str, Any] = {}
    if args.reversed_arm:
        flipped = {t.query_id: generated[(t.query_id, REVERSED)][0].abstained for t in traces if control[t.query_id]}
        reversed_abstain = {q: control[q] and flipped.get(q, False) for q in control}

        def reversed_decision(t: QueryTrace) -> bool:
            return bool(reversed_abstain[t.query_id] == t.is_null)

        def reversed_judged(t: QueryTrace) -> bool:
            if t.is_null:
                return reversed_abstain[t.query_id]
            if not control[t.query_id]:
                return judged["generator_only"](t)
            record, verdict = generated[(t.query_id, REVERSED)]
            return bool(
                not record.abstained and verdict and verdict.correctness.correct and not verdict.correctness.abstained
            )

        rev_recovered = [t for t in answerable if control[t.query_id] and not reversed_abstain[t.query_id]]
        report_extra["explore_reversed_order_retry"] = {
            "description": "retry each first-attempt refusal once on attempt 1's top five in reverse order; no rewrite, no second retrieval",
            "calls": len(traces) + sum(control.values()),
            "decision_scored": balanced(traces, reversed_decision)
            | {"paired_vs_generator_only": paired(traces, reversed_decision, decision_right["generator_only"])},
            "correctness_scored": balanced(traces, reversed_judged)
            | {
                "paired_vs_generator_only": paired(traces, reversed_judged, judged["generator_only"]),
                "paired_vs_rewrite_retry": paired(traces, reversed_judged, judged["retry_on_not_stated"]),
            },
            "recovered_answerable": tally(rev_recovered, REVERSED),
            "nulls_turned_into_answers": sum(
                1 for t in traces if t.is_null and control[t.query_id] and not reversed_abstain[t.query_id]
            ),
        }

    if args.resample_arm:
        resampled = {q: control[q] and generated[(q, 0)][0].abstained for q in control if control[q]}
        resample_abstain = {q: resampled.get(q, False) for q in control}

        def resample_decision(t: QueryTrace) -> bool:
            return bool(resample_abstain[t.query_id] == t.is_null)

        def resample_judged(t: QueryTrace) -> bool:
            if t.is_null:
                return resample_abstain[t.query_id]
            if not control[t.query_id]:
                return judged["generator_only"](t)
            record, verdict = generated[(t.query_id, 0)]
            return bool(
                not record.abstained and verdict and verdict.correctness.correct and not verdict.correctness.abstained
            )

        res_recovered = [t for t in answerable if control[t.query_id] and not resample_abstain[t.query_id]]
        report_extra["explore_resample_retry"] = {
            "description": (
                "retry each first-attempt refusal once with the identical prompt on a fresh call "
                "at temperature zero; no rewrite, no second retrieval. Measures serving nondeterminism."
            ),
            "calls": len(traces) + sum(control.values()),
            "decision_scored": balanced(traces, resample_decision)
            | {"paired_vs_generator_only": paired(traces, resample_decision, decision_right["generator_only"])},
            "correctness_scored": balanced(traces, resample_judged)
            | {
                "paired_vs_generator_only": paired(traces, resample_judged, judged["generator_only"]),
                "paired_vs_rewrite_retry": paired(traces, resample_judged, judged["retry_on_not_stated"]),
            },
            "recovered_answerable": tally(res_recovered, 0),
            "nulls_turned_into_answers": sum(
                1 for t in traces if t.is_null and control[t.query_id] and not resample_abstain[t.query_id]
            ),
        }

    calls_control = len(traces)
    calls_retry = len(traces) + 2 * sum(control.values())
    cold_calls = {"generator": len(jobs), "judge": 2 * sum(v is not None for _, v in results)}
    return {
        "mode": "explore",
        "produced_by": produced_by(args.generator_model, args.judge_model, cold_calls),
        "corpus": f"ContractNLI test, first {args.documents} documents, {len(traces)} queries",
        "traces": str(args.traces.relative_to(REPO)),
        "traces_sha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "evidence_budget": EVIDENCE_BUDGET,
        "note": (
            "Generation always uses the original question; the rewrite only changes "
            "which chunks are retrieved and their order. Answers are regenerated at "
            "temperature zero on the recorded chunk IDs because the harvest cache was not kept."
        ),
        "calls": {
            "generator_only": calls_control,
            "retry_on_not_stated": calls_retry,
            "ratio": round(calls_retry / calls_control, 2),
        },
        "decision_scored": {name: balanced(traces, fn) for name, fn in decision_right.items()}
        | {"paired": paired(traces, decision_right["retry_on_not_stated"], decision_right["generator_only"])},
        "correctness_scored_trace_decisions": {name: balanced(traces, fn) for name, fn in judged.items()}
        | {"paired": paired(traces, judged["retry_on_not_stated"], judged["generator_only"])},
        "correctness_scored_regenerated_decisions": {name: balanced(traces, fn) for name, fn in regen.items()}
        | {"paired": paired(traces, regen["retry_on_not_stated"], regen["generator_only"])},
        "recovered_answerable": tally(recovered, 1),
        "attempt1_answers_answerable": tally(attempt1_answered, 0),
        "nulls_turned_into_answers": {
            "questions": len(lost_nulls),
            "regenerated_answered": sum(not generated[(t.query_id, 1)][0].abstained for t in lost_nulls),
        },
        "reproduction": reproduction,
        "logical_calls_for_a_cold_run": cold_calls,
        "finding": FINDING,
        "what_would_make_it_win": WHAT_WOULD_MAKE_IT_WIN,
        "discordant_questions": per_question,
    } | report_extra


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="response caches, one SQLite file per model")
    parser.add_argument("--env", type=Path, default=REPO / ".env")
    parser.add_argument("--generator-model", help="Override the shared generation role")
    parser.add_argument("--judge-model", help="Override the shared judging role")
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--rpm", type=int, default=30, help="requests per minute, per model")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resample-arm", action="store_true", help="also run the identical-prompt resample control")
    parser.add_argument("--reversed-arm", action="store_true", help="also run the reversed-evidence explore arm")
    parser.add_argument("--cache-only", action="store_true", help="forbid model calls; replay from the caches")
    args = parser.parse_args(argv)
    models = choose_models(generation=args.generator_model, judging=args.judge_model)
    args.generator_model = args.generator_model or (GENERATOR_MODEL if args.cache_only else models.generation.model)
    args.judge_model = args.judge_model or (JUDGE_MODEL if args.cache_only else models.judging.model)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run and write the report."""
    args = parse_args(argv)
    args.traces = args.traces.resolve()
    report = run(args)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        k: report[k]
        for k in (
            "calls",
            "decision_scored",
            "correctness_scored_trace_decisions",
            "correctness_scored_regenerated_decisions",
            "recovered_answerable",
            "attempt1_answers_answerable",
            "nulls_turned_into_answers",
            "reproduction",
            "logical_calls_for_a_cold_run",
        )
    } | {k: v for k, v in report.items() if k.startswith("explore_")}
    print(json.dumps(summary, indent=2))
    if not args.cache_only:
        print(spend.report(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
