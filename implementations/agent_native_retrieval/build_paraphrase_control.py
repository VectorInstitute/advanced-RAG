"""Build the paraphrase control: MuSiQue questions with no gold-passage name in them.

A literal ``grep`` agent reaches a MuSiQue question's first passage through the
names the question carries. This set rewords questions so that no capitalized
span of any gold passage appears in them, and keeps a rewrite only if a model
reading the gold passages alone answers it correctly. It is the set on which
``grep`` is built to lose.

Candidates are tried in a fixed order: the 4-hop questions of the shared
300-question test, then the other 4-hop questions, then 3-hop questions in the
same order, until the target is reached. The first attempt is
:func:`~aieng.advanced_rag.evaluation.synthetic.rewrite.paraphrase_question`,
which sees only the question and its answer. Later attempts are
:func:`~aieng.advanced_rag.evaluation.synthetic.rewrite.obfuscate_question`,
which also sees the first-hop passage and the decomposition, so a name the
generator does not know can be replaced with a description of it; they are
told every name in the gold passages, so a description does not bring in a
new one. Every span still leaking is added to the banned list before the next
attempt. The check is
:func:`~aieng.advanced_rag.evaluation.synthetic.filters.round_trip`.

    python build_paraphrase_control.py --target 50
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import build_common
import question_sets as qs
from aieng.advanced_rag.corpora import musique
from aieng.advanced_rag.evaluation.synthetic import Model
from aieng.advanced_rag.evaluation.synthetic.filters import round_trip
from aieng.advanced_rag.evaluation.synthetic.rewrite import (
    obfuscate_question,
    paraphrase_question,
)


OUT = qs.PARAPHRASE_CONTROL

#: A rewrite that points at "the first document" cannot be asked of a corpus.
_SOURCE_WORDS = re.compile(r"\b(document|passage|excerpt)s?\b", re.IGNORECASE)


def candidates(questions: Sequence[musique.MusiqueQuestion]) -> list[musique.MusiqueQuestion]:
    """Return 4-hop then 3-hop questions, the shared test's first within each."""
    shared = {q.question_id for q in musique.stratified_test_set(questions)}
    ordered: list[musique.MusiqueQuestion] = []
    for hops in (4, 3):
        pool = [q for q in questions if q.hops == hops]
        ordered += [q for q in pool if q.question_id in shared]
        ordered += [q for q in pool if q.question_id not in shared]
    return ordered


def first_hop_reachable(question: str, gold_first: int, corpus_lower: Sequence[str]) -> bool:
    """Return whether grep for a span of the question finds the first gold passage."""
    return any(
        gold_first in qs.grep_hits(corpus_lower, span) for span in qs.capitalized_spans(question) if len(span) >= 3
    )


def rewrite_one(
    question: musique.MusiqueQuestion,
    *,
    by_id: dict[str, musique.Passage],
    lexicon: qs.Lexicon,
    corpus_lower: Sequence[str],
    generator: Model,
    checker: Model,
    attempts: int,
) -> dict[str, Any]:
    """Rewrite and check one question; return its full record."""
    gold = [by_id[pid] for pid in question.supporting_ids]
    gold_fields = [t for p in gold for t in (p.title, p.text)]
    gold_texts = [f"{p.title}\n{p.text}" for p in gold]
    first = int(question.supporting_ids[0].split("::")[1])
    answers = (question.answer, *question.answer_aliases)

    before = qs.leaked_spans(question.question, gold_fields, lexicon)
    original = round_trip(checker, checker, question.question, question.answer, gold_texts)

    bridge = "; ".join(f"{step.question} -> {step.answer}" for step in question.decomposition)
    all_names = sorted(qs.name_spans(gold_fields, lexicon), key=lambda n: (-len(n.split()), n))
    banned = list(before)
    rewritten, leaked, tries = "", list(before), 0
    for tries in range(1, attempts + 1):
        if tries == 1:
            rewritten = paraphrase_question(generator, question.question, question.answer, banned)
        else:
            reply = obfuscate_question(
                generator,
                question.question,
                answer=question.answer,
                bridge=bridge,
                first_document=gold_texts[0],
                # Every name in the gold passages, so the description of a
                # removed name does not bring in another one.
                leaked=banned + sorted(set(all_names) - set(banned)),
            )
            rewritten = str(reply.get("question") or "").strip() or rewritten
        if not rewritten:
            continue
        leaked = qs.leaked_spans(rewritten, gold_fields, lexicon)
        if not leaked:
            break
        banned = sorted(set(banned) | set(leaked))

    answer_given = any(qs.contains_words(rewritten, a) for a in answers if a.strip())
    names_source = bool(_SOURCE_WORDS.search(rewritten))
    record: dict[str, Any] = {
        "source_question_id": question.question_id,
        "hops": question.hops,
        "original_question": question.question,
        "question": rewritten,
        "answer": question.answer,
        "answer_aliases": list(question.answer_aliases),
        "supporting_ids": list(question.supporting_ids),
        "leaked_before": before,
        "leaked_after": leaked if rewritten else before,
        "attempts": tries,
        "answer_in_question": answer_given,
        "names_a_source": names_source,
        "original_round_trip": original["verdict"],
        "first_hop_grep_reaches_gold_before": first_hop_reachable(question.question, first, corpus_lower),
    }
    if not rewritten or leaked or answer_given or names_source:
        record.update(stage="rewrite", accepted=False)
        return record
    check = round_trip(checker, checker, rewritten, question.answer, gold_texts)
    record.update(
        # Spans the strict rule (every capitalized word) would also flag. They
        # match more than 1% of passages, so they select nothing for grep.
        common_spans_kept=qs.leaked_spans(rewritten, gold_fields, None),
    )
    record.update(
        checker_answer=check["answer"],
        verdict=check["verdict"],
        verdict_reason=check["reason"],
        token_f1=round(qs.token_f1(check["answer"], answers), 4),
        first_hop_grep_reaches_gold_after=first_hop_reachable(rewritten, first, corpus_lower),
        stage="round_trip",
        accepted=bool(check["passed"]),
    )
    return record


def main() -> None:
    """Build the set and its summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--generator", default=build_common.DEFAULT_GENERATOR)
    parser.add_argument("--checker", default=build_common.DEFAULT_CHECKER)
    parser.add_argument("--generator-per-minute", type=int, default=15)
    parser.add_argument("--checker-per-minute", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--trace", type=Path, default=None, help="also write every tried record")
    args = parser.parse_args()

    clock = build_common.Clock()
    passages, questions = musique.load_musique()
    by_id = {p.passage_id: p for p in passages}
    corpus_lower = [f"{p.title}\n{p.text}".lower() for p in passages]
    lexicon = qs.Lexicon.build(f"{p.title}\n{p.text}" for p in passages)
    generator = build_common.model(args.generator, args.generator_per_minute)
    checker = build_common.model(args.checker, args.checker_per_minute)

    ordered = candidates(questions)
    shared = {q.question_id for q in musique.stratified_test_set(questions)}
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(ordered), args.batch):
            batch = ordered[start : start + args.batch]
            records += list(
                pool.map(
                    lambda q: rewrite_one(
                        q,
                        by_id=by_id,
                        lexicon=lexicon,
                        corpus_lower=corpus_lower,
                        generator=generator,
                        checker=checker,
                        attempts=args.attempts,
                    ),
                    batch,
                )
            )
            kept = sum(r["accepted"] for r in records)
            print(f"tried {len(records)}, accepted {kept}", flush=True)
            if kept >= args.target:
                break

    accepted = [r for r in records if r["accepted"]][: args.target]
    tried = records[: records.index(accepted[-1]) + 1] if accepted else records
    rows = []
    for n, record in enumerate(accepted, start=1):
        row = {
            k: v
            for k, v in record.items()
            if k not in ("accepted", "stage", "leaked_after", "answer_in_question", "names_a_source")
        }
        row.update(
            query_id=f"musique-paraphrase-{n:03d}",
            in_shared_test=record["source_question_id"] in shared,
            is_answerable=True,
            generator=generator.name,
            checker=checker.name,
        )
        rows.append(row)
    qs.write_jsonl(args.out, rows)
    if args.trace is not None:
        qs.write_jsonl(args.trace, tried)

    stages = Counter("accepted" if r["accepted"] else f"rejected_at_{r['stage']}" for r in tried)
    build_common.write_summary(
        args.out.with_suffix(".summary.json"),
        {
            "set": args.out.name,
            "generator": generator.name,
            "checker": checker.name,
            "judge": checker.name,
            "attempts_per_question": args.attempts,
            "tried": len(tried),
            "tried_by_hops": dict(Counter(r["hops"] for r in tried)),
            "accepted": len(rows),
            "accepted_by_hops": dict(Counter(r["hops"] for r in rows)),
            "acceptance_rate": round(len(rows) / len(tried), 4) if tried else 0.0,
            "funnel": dict(stages),
            "rejected_rewrite_still_leaks": sum(
                1 for r in tried if r["stage"] == "rewrite" and r["question"] and r["leaked_after"]
            ),
            "rejected_rewrite_names_a_source": sum(1 for r in tried if r["stage"] == "rewrite" and r["names_a_source"]),
            "rejected_rewrite_gives_answer": sum(
                1 for r in tried if r["stage"] == "rewrite" and r["answer_in_question"]
            ),
            "rejected_round_trip_verdicts": dict(
                Counter(r["verdict"] for r in tried if r["stage"] == "round_trip" and not r["accepted"])
            ),
            "original_round_trip_match_rate_tried": round(
                sum(r["original_round_trip"] == "match" for r in tried) / max(len(tried), 1), 4
            ),
            "first_hop_grep_reaches_gold_before": sum(r["first_hop_grep_reaches_gold_before"] for r in rows),
            "first_hop_grep_reaches_gold_after": sum(r["first_hop_grep_reaches_gold_after"] for r in rows),
            "seconds": clock.seconds(),
        },
        generator,
        checker,
    )
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
