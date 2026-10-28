"""Build the near-miss nulls: MuSiQue-like questions the 11,656 passages do not answer.

Each null starts from a question of the shared 300-question test. The generator
changes the one detail that makes the question's final passage answer it
(:func:`~aieng.advanced_rag.evaluation.synthetic.nulls.near_miss_question`).
The checker then reads what three searches return over the whole 11,656-passage
corpus: the BM25 top 10, the dense top 10, and up to 10 passages that a literal
``grep`` for the changed terms finds. The null is kept only when the checker
judges that none of them answers it, fully or in part
(:func:`~aieng.advanced_rag.evaluation.synthetic.filters.corpus_answers`, the
CRUMQs check that :func:`~aieng.advanced_rag.evaluation.synthetic.nulls.check_null`
runs).

Every row records the terms the edit introduced and how many passages ``grep``
returns for each, in the 11,656 passages and in the 139,416-passage pool, so a
notebook can test whether zero ``grep`` hits signal that a question has no
answer.

    python build_near_miss_nulls.py --per-hop 2=25 3=12 4=13
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import build_common
import question_sets as qs
from aieng.advanced_rag.corpora import musique
from aieng.advanced_rag.evaluation.synthetic import Document, Model, ProbeIndex
from aieng.advanced_rag.evaluation.synthetic.filters import corpus_answers
from aieng.advanced_rag.evaluation.synthetic.nulls import near_miss_question, null_evidence
from aieng.advanced_rag.evaluation.synthetic.probe import Hit
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder


OUT = qs.NEAR_MISS_NULLS

DOMAIN = (
    "short Wikipedia passages about people, places, organisations, creative "
    "works, and events, and the questions chain facts across several of them"
)

_SOURCE_WORDS = re.compile(r"\b(document|passage|excerpt|statement)s?\b", re.IGNORECASE)


def candidates(questions: Sequence[musique.MusiqueQuestion], seed: int) -> list[musique.MusiqueQuestion]:
    """Return the shared test's questions in a seeded order."""
    shared = musique.stratified_test_set(questions)
    order = list(shared)
    random.Random(seed).shuffle(order)
    return order


class Corpus:
    """The two corpus sizes as ``grep`` sees them, and the probe over the smaller."""

    def __init__(
        self,
        passages: Sequence[musique.Passage],
        pool: Sequence[musique.Passage],
        probe: ProbeIndex,
    ) -> None:
        self.passages = list(passages)
        self.lower = [f"{p.title}\n{p.text}".lower() for p in passages]
        self.pool_lower = [f"{p.title}\n{p.text}".lower() for p in pool]
        self.probe = probe

    def grep_hits(self, term: str, limit: int) -> list[Hit]:
        """Return up to ``limit`` passages containing ``term``, as probe hits."""
        hits = []
        for rank, index in enumerate(qs.grep_hits(self.lower, term)[:limit], start=1):
            passage = self.passages[index]
            text = f"{passage.title}\n{passage.text}"
            chunk = Chunk(f"{passage.passage_id}::grep", passage.passage_id, text, 0, len(text))
            hits.append(Hit(chunk, 0.0, rank))
        return hits


def build_one(
    question: musique.MusiqueQuestion,
    *,
    by_id: dict[str, musique.Passage],
    corpus: Corpus,
    generator: Model,
    checker: Model,
) -> dict[str, Any]:
    """Write and check one near-miss null; return its full record."""
    final = by_id[question.supporting_ids[-1]]
    written = near_miss_question(generator, question.question, f"{final.title}: {final.text}", DOMAIN)
    text = written["question"]
    record: dict[str, Any] = {
        "source_question_id": question.question_id,
        "hops": question.hops,
        "original_question": question.question,
        "question": text,
        "changed": written["changed"],
    }
    if not text or text.strip().lower() == question.question.strip().lower():
        record.update(stage="write", accepted=False)
        return record
    if _SOURCE_WORDS.search(text):
        record.update(stage="names_a_source", accepted=False)
        return record

    terms = qs.new_terms(text, question.question)
    counts = [
        {
            "term": term,
            "hits_11656": qs.grep_count(corpus.lower, term),
            "hits_139416": qs.grep_count(corpus.pool_lower, term),
        }
        for term in terms
    ]
    record["new_terms"] = counts
    record["grep_zero_hits"] = any(c["hits_11656"] == 0 for c in counts) if counts else None
    record["grep_zero_hits_pool"] = any(c["hits_139416"] == 0 for c in counts) if counts else None

    hits = null_evidence(corpus.probe, text, scope=None, k=10)
    seen = {h.chunk.doc_id for h in hits}
    for count in sorted(counts, key=lambda c: c["hits_11656"]):
        for hit in corpus.grep_hits(count["term"], 10):
            if hit.chunk.doc_id not in seen and len(hits) < 30:
                seen.add(hit.chunk.doc_id)
                hits.append(hit)
    verdict = corpus_answers(checker, text, hits)
    record.update(
        null_check={
            "answered": verdict["answered"],
            "reason": verdict["reason"],
            "passages_read": len(hits),
        },
        stage="null_check",
        accepted=verdict["answered"] == "no",
    )
    return record


def parse_quota(values: Sequence[str]) -> dict[int, int]:
    """Parse ``hops=count`` pairs."""
    quota = {}
    for value in values:
        hops, count = value.split("=")
        quota[int(hops)] = int(count)
    return quota


def main() -> None:
    """Build the set and its summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-hop", nargs="+", default=["2=25", "3=12", "4=13"])
    parser.add_argument("--generator", default=build_common.DEFAULT_GENERATOR)
    parser.add_argument("--checker", default=build_common.DEFAULT_CHECKER)
    parser.add_argument("--generator-per-minute", type=int, default=15)
    parser.add_argument("--checker-per-minute", type=int, default=30)
    parser.add_argument("--seed", type=int, default=musique.TEST_SEED)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--trace", type=Path, default=None, help="also write every tried record")
    args = parser.parse_args()
    quota = parse_quota(args.per_hop)

    clock = build_common.Clock()
    passages, questions = musique.load_musique()
    by_id = {p.passage_id: p for p in passages}
    documents = [Document(p.passage_id, f"{p.title}\n{p.text}", p.title) for p in passages]
    probe = ProbeIndex(documents, SentenceTransformerEmbedder())
    corpus = Corpus(passages, musique.load_pool(), probe)
    generator = build_common.model(args.generator, args.generator_per_minute)
    checker = build_common.model(args.checker, args.checker_per_minute)

    ordered = [q for q in candidates(questions, args.seed) if q.hops in quota]
    records: list[dict[str, Any]] = []
    kept: Counter[int] = Counter()

    def full(hops: int) -> bool:
        return kept[hops] >= quota[hops]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        remaining = list(ordered)
        while remaining and not all(full(h) for h in quota):
            batch, remaining = remaining[: args.batch], remaining[args.batch :]
            batch = [q for q in batch if not full(q.hops)]
            for record in pool.map(
                lambda q: build_one(q, by_id=by_id, corpus=corpus, generator=generator, checker=checker),
                batch,
            ):
                if not full(record["hops"]):
                    records.append(record)
                    kept[record["hops"]] += record["accepted"]
            print(f"tried {len(records)}, kept {dict(kept)}", flush=True)

    rows = []
    for n, record in enumerate((r for r in records if r["accepted"]), start=1):
        row = {k: v for k, v in record.items() if k not in ("accepted", "stage")}
        row.update(
            query_id=f"musique-null-{n:03d}",
            is_answerable=False,
            answer="",
            generator=generator.name,
            checker=checker.name,
        )
        rows.append(row)
    qs.write_jsonl(args.out, rows)
    if args.trace is not None:
        qs.write_jsonl(args.trace, records)

    stages = Counter("accepted" if r["accepted"] else f"rejected_at_{r['stage']}" for r in records)
    with_terms = [r for r in rows if r["grep_zero_hits"] is not None]
    build_common.write_summary(
        args.out.with_suffix(".summary.json"),
        {
            "set": args.out.name,
            "generator": generator.name,
            "checker": checker.name,
            "probe": "BM25 and bge-small-en-v1.5 top 10 each, plus up to 10 grep hits per new term",
            "quota_by_hops": quota,
            "tried": len(records),
            "tried_by_hops": dict(Counter(r["hops"] for r in records)),
            "accepted": len(rows),
            "accepted_by_hops": dict(Counter(r["hops"] for r in rows)),
            "acceptance_rate": round(len(rows) / len(records), 4) if records else 0.0,
            "funnel": dict(stages),
            "null_check_verdicts_rejected": dict(
                Counter(
                    r["null_check"]["answered"] for r in records if r["stage"] == "null_check" and not r["accepted"]
                )
            ),
            "accepted_with_new_terms": len(with_terms),
            "accepted_grep_zero_hits_11656": sum(bool(r["grep_zero_hits"]) for r in with_terms),
            "accepted_grep_zero_hits_139416": sum(bool(r["grep_zero_hits_pool"]) for r in with_terms),
            "seconds": clock.seconds(),
        },
        generator,
        checker,
    )
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
