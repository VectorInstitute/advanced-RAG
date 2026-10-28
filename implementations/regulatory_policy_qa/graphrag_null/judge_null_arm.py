#!/usr/bin/env python3
"""Judge saved null-arm answers without generating answers or retrieving again.

The private store holds evidence and a resumable judge-response cache. Public
reports contain verdicts and input hashes, never gold passages or judge prompts.
Evaluation rules and report construction live in ``judge_lab.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "aieng-advanced-rag"))
sys.path.insert(0, str(HERE))

from aieng.advanced_rag import spend  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.corpora import cuad  # noqa: E402
from aieng.advanced_rag.corpora.contract_nli import (  # noqa: E402
    build_slice,
    fetch_contract_nli,
)
from aieng.advanced_rag.graphrag.evidence import Evidence  # noqa: E402
from aieng.advanced_rag.models import choose_models  # noqa: E402 - use the source checkout after bootstrap
from aieng.advanced_rag.proxy import (  # noqa: E402 - use the source checkout after bootstrap
    resolve_api_key,
    resolve_base_url,
)
from aieng.advanced_rag.retrieval.corpus import Query  # noqa: E402
from judge_lab import (  # noqa: E402
    CORRECTNESS_PROMPT,
    GROUNDING_PROMPT,
    MAX_CHARS,
    correctness_consistency,
    references,
    score_answers,
    validate_outcomes,
)
from null_lab import (  # noqa: E402
    Meter,
    MeteredCompleter,
    ResponseCache,
    load_evidence,
    proxy_batch,
)


ARMS = ("dense", "lightrag-mix")
DEFAULT_MODEL = "Qwen3_6-35B-A3B"


@dataclass(frozen=True)
class JudgeOptions:
    """Paths and execution settings, independent of loaded data and credentials."""

    store: Path
    results: Path
    env: Path
    model: str = DEFAULT_MODEL
    workers: int = 8
    rpm: int = 90
    out: Path | None = None
    cache_only: bool = False
    arms: tuple[str, ...] = ARMS


@dataclass(frozen=True)
class ReplayInputs:
    """Validated corpus, saved answers, evidence, and portable input digests."""

    summary: dict[str, Any]
    queries: Sequence[Query]
    gold: dict[str, str]
    tables: dict[str, dict[str, list[Evidence]]]
    outcomes: dict[str, dict[str, dict[str, Any]]]
    hashes: dict[str, str]
    budget: int


def digest(path: Path) -> str:
    """Hash an input file for the public provenance record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> JudgeOptions:
    """Parse replay options independently of credentials and corpus access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=HERE.parent / "outputs/graphrag_null/measured_thresholds")
    parser.add_argument("--env", type=Path, default=REPO / ".env")
    parser.add_argument(
        "--model", help="Override the shared judging role; offline replay defaults to the reference judge"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=int, default=90)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--out", type=Path, help="write reports here; cache-only defaults to verifying saved reports")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="forbid API calls and require cached judge responses; no credentials needed",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.rpm < 1:
        parser.error("workers and rpm must be positive")
    if len(args.arms) != len(set(args.arms)):
        parser.error("arms must be unique")
    args.model = args.model or (DEFAULT_MODEL if args.cache_only else choose_models().judging.model)
    args.arms = tuple(arm for arm in ARMS if arm in args.arms)
    return JudgeOptions(**vars(args))


def publish_reports(out: Path, reports: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Atomically replace each report and publish their hashes in the summary last."""
    out.mkdir(parents=True, exist_ok=True)
    payloads = {f"answer_judgements_{r['method']}.json": json.dumps(r, indent=2) + "\n" for r in reports}
    summary["report_sha256"] = {name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in payloads.items()}
    payloads["answer_judging_summary.json"] = json.dumps(summary, indent=2) + "\n"
    for name, raw in payloads.items():
        temporary = out / f".{name}.{os.getpid()}.tmp"
        temporary.write_text(raw)
        temporary.replace(out / name)


def verify_reports(results: Path, reports: Sequence[dict[str, Any]], inputs: dict[str, str]) -> None:
    """Verify input hashes and exact verdicts without rewriting the saved summary."""
    saved = json.loads((results / "answer_judging_summary.json").read_text())
    if saved["input_sha256"] != inputs:
        raise ValueError("saved judging inputs have changed")
    for report in reports:
        path = results / f"answer_judgements_{report['method']}.json"
        if report != json.loads(path.read_text()) or digest(path) != saved["report_sha256"][path.name]:
            raise ValueError(f"cached judging does not reproduce {path.name}")
    print(f"{len(reports)} saved reports reproduced exactly; zero API calls and no files changed.", flush=True)


def load_inputs(args: JudgeOptions) -> ReplayInputs:
    """Load and validate every saved input before constructing a model client."""
    summary_path = args.results / "null_arm_summary.json"
    summary = json.loads(summary_path.read_text())
    documents = summary["corpus"]["contracts"]
    budget = summary["pipeline"]["evidence_budget"]
    if summary["pipeline"]["rewriter"] is not None or budget != 5:
        raise ValueError("this replay requires the original question and five-passage budget")
    if summary["corpus"]["name"].startswith("CUAD"):
        corpus_path = cuad.fetch_cuad_test(dest=args.store / "corpus")
        queries = cuad.build_slice(corpus_path, max_documents=documents).queries
        gold = {
            query.query_id: "Annotated answer: " + "\n".join(span.text for span in query.answers)
            for query in queries
            if query.is_answerable
        }
    else:
        corpus_path = fetch_contract_nli(dest=args.store / "corpus")
        queries = build_slice(corpus_path, max_documents=documents).queries
        gold = references(corpus_path, documents)
    if set(gold) != {q.query_id for q in queries if bool(q.answers)}:
        raise ValueError("gold references do not match the answerable slice")
    # Stable roles make the cache verification portable to a relocated checkout/store.
    inputs = {summary_path.name: digest(summary_path), f"corpus/{corpus_path.name}": digest(corpus_path)}
    tables, outcomes = {}, {}
    for arm in args.arms:
        evidence_path = args.store / "evidence" / f"{arm}.json"
        outcome_path = args.results / f"outcomes_{arm}.json"
        tables[arm] = load_evidence(evidence_path)
        outcomes[arm] = validate_outcomes(json.loads(outcome_path.read_text()), queries, tables[arm])
        for query in queries:
            if query.answers:
                expected_chars = sum(len(item.text) for item in tables[arm][query.query_id][:budget])
                if outcomes[arm][query.query_id]["evidence_chars"] != expected_chars:
                    raise ValueError(f"saved evidence budget mismatch: {arm}/{query.query_id}")
        inputs[f"evidence/{evidence_path.name}"] = digest(evidence_path)
        inputs[outcome_path.name] = digest(outcome_path)
    return ReplayInputs(summary, queries, gold, tables, outcomes, inputs, budget)


def create_completer(args: JudgeOptions, meter: Meter) -> MeteredCompleter:
    """Construct a cached client; offline verification never reads credentials."""
    # Shared resolution supports onboarding credentials and the .env fallback.
    connection: dict[str, Any] = {}
    if not args.cache_only:
        spend.session()
        connection = {
            "rpm": args.rpm,
            "timeout": 600,
            "api_key": resolve_api_key(args.env),
            "base_url": resolve_base_url(args.env),
        }
    batch = proxy_batch(
        args.model,
        ResponseCache(args.store / "answer_judge_cache.sqlite"),
        cache_only=args.cache_only,
        **connection,
    )
    return MeteredCompleter(batch, meter=meter)


def score_replay(
    inputs: ReplayInputs, complete: Callable[[str], str], meter: Meter, *, workers: int
) -> list[dict[str, Any]]:
    """Score selected arms in fixed order, sharing the response cache and meter."""
    reports = []
    for arm in inputs.tables:
        meter.phase = f"answer-judge:{arm}"
        print(f"Judging {arm}: {len(inputs.gold)} answerable questions (including abstentions)", flush=True)
        report = score_answers(
            arm,
            inputs.queries,
            inputs.outcomes[arm],
            inputs.tables[arm],
            inputs.gold,
            complete,
            budget=inputs.budget,
            workers=workers,
        )
        reports.append(report)
        print(f"{arm}: {report['correct_count']}/{report['questions']} correct", flush=True)
    return reports


def build_summary(
    args: JudgeOptions,
    inputs: ReplayInputs,
    reports: Sequence[dict[str, Any]],
    meter: Meter,
    consistency: dict[str, int],
) -> dict[str, Any]:
    """Record execution provenance separately from answer-level measurements."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": {
            "model": args.model,
            "temperature": 0,
            "max_tokens": 512,
            "reasoning": "disabled",
            "protocol": "separate semantic correctness and evidence grounding passes",
            "correctness_prompt_sha256": hashlib.sha256(CORRECTNESS_PROMPT.encode()).hexdigest(),
            "grounding_prompt_sha256": hashlib.sha256(GROUNDING_PROMPT.encode()).hexdigest(),
        },
        "generation_model": inputs.summary["model"],
        "evidence_budget": inputs.budget,
        "max_chars_per_passage": MAX_CHARS,
        "reference": "CUAD annotated answer spans"
        if inputs.summary["corpus"]["name"].startswith("CUAD")
        else "ContractNLI annotation polarity expressed in natural language, with gold spans as supporting justification; classification labels are not required wording",
        "denominator": "all answerable queries, including pipeline abstentions",
        "abstention_rule": "canonical saved pipeline flag; incorrect and vacuously grounded on answerable queries; both raw judge passes are preserved",
        "cache_only": args.cache_only,
        "limitations": [
            "Automated judgments; generator and judge use the same model, with possible shared errors.",
            "Separate correctness and grounding passes; no human adjudication or independence assumption across questions in a contract.",
            "Grounding includes vacuously grounded canonical abstentions; hallucination excludes pipeline abstentions.",
            "Correctness uses CUAD's annotated answer spans; acceptable paraphrases and incomplete annotations require human review."
            if inputs.summary["corpus"]["name"].startswith("CUAD")
            else "Correctness measures agreement with ContractNLI's binary annotations; qualified answers can conflict with those labels.",
        ],
        "input_sha256": inputs.hashes,
        "runner_sha256": digest(Path(__file__)),
        "judging_sha256": digest(HERE / "judge_lab.py"),
        "cost": meter.to_dict(),
        "consistency_checks": consistency,
        "arms": [{k: v for k, v in report.items() if k != "judgements"} for report in reports],
    }


def finish_replay(args: JudgeOptions, reports: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Verify saved reports or explicitly publish them, keeping the cold run's cost."""
    if args.cache_only and args.out is None:
        verify_reports(args.results, reports, summary["input_sha256"])
        return
    prior = args.results / "answer_judging_summary.json"
    if args.cache_only and prior.exists():
        previous = json.loads(prior.read_text())
        # A cache-only publication costs nothing, so it carries the cold run's cost.
        summary["cold_run_cost"] = previous.get("cold_run_cost", previous["cost"])
    publish_reports(args.out or args.results, reports, summary)


def main(argv: Sequence[str] | None = None) -> None:
    """Load inputs, score selected arms, check consistency, and verify or publish."""
    options = parse_args(argv)
    inputs = load_inputs(options)
    meter = Meter(phase="answer-judge")
    complete = create_completer(options, meter)
    reports = score_replay(inputs, complete, meter, workers=options.workers)
    consistency = (
        correctness_consistency(reports, inputs.outcomes)
        if len(reports) > 1
        else {
            "identical_answer_pairs": 0,
            "correctness_disagreements": 0,
            "cross_arm_check_applicable": 0,
        }
    )
    summary = build_summary(options, inputs, reports, meter, consistency)
    finish_replay(options, reports, summary)
    if not options.cache_only:
        print(spend.report(), file=sys.stderr)


if __name__ == "__main__":
    main()
