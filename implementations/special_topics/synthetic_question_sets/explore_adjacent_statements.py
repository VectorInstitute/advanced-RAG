"""Explore: write each question from two statements, to test the small-chunk bias.

The ContractNLI validation found that generated labels overrate 600-character
chunks. A generated question has one short gold span, while a real ContractNLI
query averages about two gold chunks, so a small chunk loses less under
generated labels. This run tests two ways of giving generated questions more
gold:

- Pairs (model calls): the generator writes one question that needs two
  statements from the same passage, so each question has two gold spans.
- Widened spans (no model calls): each plain question's gold span is widened
  around itself to the mean total gold length of the real queries.

Both are scored on the same eight RI #1 arms as the validation, against the
same real labels, and compared by Kendall's tau and by where the two
600-character arms rank. One run: one seed, one configuration, on the
on-prem models (``Qwen3_8-27B`` generates, ``Qwen3_6-35B-A3B`` checks,
``Qwen3_8-Flash-Next`` judges) at 15 requests per minute or fewer per model.

    python explore_adjacent_statements.py --env-file <env_file>

The statement-extraction calls are the validation build's, so they come from
the response cache; only the pair questions and their checks are new calls.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag.evaluation.synthetic import (
    Document,
    EvidenceSpan,
    Funnel,
    SyntheticQuestion,
    SyntheticSet,
    extract_statements,
    held_out_split,
    parse_json,
    sample_passages,
)
from aieng.advanced_rag.evaluation.synthetic.build import (
    _deduplicate,
    map_concurrent,
)
from aieng.advanced_rag.evaluation.synthetic.filters import round_trip, uniqueness
from aieng.advanced_rag.evaluation.synthetic.generate import SCOPE_NOTES
from aieng.advanced_rag.evaluation.synthetic.llm import Model, usage_of
from aieng.advanced_rag.evaluation.synthetic.probe import ProbeIndex
from aieng.advanced_rag.evaluation.synthetic.replay import Roles
from aieng.advanced_rag.evaluation.synthetic.statements import Statement
from aieng.advanced_rag.evaluation.synthetic.testset import question_id
from aieng.advanced_rag.evaluation.synthetic.validation import (
    CONTRACT_NLI_CONFIG,
    DENSE_MODEL,
    agreement,
    arm_groups,
    build_arms,
    mean_scores,
    queries_for,
    real_qrels,
    score_arms,
)
from aieng.advanced_rag.models import ModelChoice
from aieng.advanced_rag.retrieval.corpus import Query


PAIR_SYSTEM = "You write realistic user questions for a search evaluation. Reply with one JSON object and nothing else."

PAIR_PROMPT = """The documents are {domain}.

Write ONE question whose full answer needs BOTH statements below.

Who is asking: {user}
Kind of question: {question_kind}
Form: {phrasing}
{scope_note}
Rules:
- The answer must use something from statement A and something from
  statement B. If either statement alone answers the question, it is wrong.
- The two statements together must fully answer the question. The answer must
  not need anything outside them.
- Ask what this user would ask. Do not mention "the statement", "the passage",
  "the text", or "the document" unless a user would naturally say it.
- Do not copy a long run of words from the statements into the question.
- If no realistic question of this kind needs both statements, set "reject" to
  true.

Statement A:
<<<
{first}
>>>

Statement B:
<<<
{second}
>>>

Return JSON: {{"question": "...", "answer": "a short answer that uses both statements", "reject": false, "reason": "one sentence"}}"""

KIND = "pair"


def pairs_of(statements: Sequence[Statement]) -> list[tuple[Statement, Statement]]:
    """Return the first two non-overlapping statements of each passage, in order."""
    by_passage: dict[str, list[Statement]] = {}
    for statement in statements:
        by_passage.setdefault(statement.passage.passage_id, []).append(statement)
    pairs = []
    for group in by_passage.values():
        ordered = sorted(group, key=lambda s: s.start)
        for second in ordered[1:]:
            if second.start >= ordered[0].end:
                pairs.append((ordered[0], second))
                break
    return pairs


def _pair_chain(
    pair: tuple[Statement, Statement],
    *,
    generator: Model,
    checker: Model,
    judge: Model,
    probe: ProbeIndex,
    texts: dict[str, str],
    funnel: Funnel,
) -> SyntheticQuestion | None:
    first, second = pair
    config = CONTRACT_NLI_CONFIG
    candidate = question_id(KIND, first.doc_id, first.start, second.end)
    drawn = config.sample(candidate)
    payload = parse_json(
        generator(
            PAIR_SYSTEM,
            PAIR_PROMPT.format(
                domain=config.domain,
                user=drawn["user"].description,
                question_kind=drawn["question"].description,
                phrasing=drawn["phrasing"].description,
                scope_note=SCOPE_NOTES[config.scope],
                first=first.quote,
                second=second.quote,
            ),
        )
    )
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not funnel.check("question_written", bool(question) and not payload.get("reject"), KIND):
        funnel.finish(candidate, KIND, "rejected", "question_written", "generator declined")
        return None
    trip = round_trip(checker, judge, question, answer, [first.quote, second.quote])
    if not funnel.check("round_trip", trip["passed"], KIND):
        funnel.finish(candidate, KIND, "filtered", "round_trip", str(trip["verdict"]))
        return None
    # Recorded, not filtered: whether statement A alone already answers it.
    alone = round_trip(checker, judge, question, answer, [first.quote])
    gold = (
        EvidenceSpan(first.doc_id, first.start, first.end, first.quote),
        EvidenceSpan(second.doc_id, second.start, second.end, second.quote),
    )
    hits = probe.search(
        question,
        method="dense",
        k=config.uniqueness_k,
        scope=first.doc_id,
        exclude=[(g.doc_id, g.start, g.end) for g in gold],
    )
    unique = uniqueness(checker, question, answer, hits, texts)
    if not funnel.check("uniqueness", unique.status != "ambiguous", KIND):
        funnel.finish(candidate, KIND, "filtered", "uniqueness", "another passage gives a different answer")
        return None
    return SyntheticQuestion(
        query_id=candidate,
        question=question,
        reference=answer,
        evidence=(*gold, *unique.equivalent),
        kind=KIND,
        scope=first.doc_id,
        group_id=candidate,
        provenance={
            "generator": generator.name,
            "checker": checker.name,
            "judge": judge.name,
            "first_alone_answers": bool(alone["passed"]),
            "uniqueness": unique.status,
        },
    )


def build_pairs(
    documents: Sequence[Document], *, generator: Model, checker: Model, judge: Model, embedder: Any
) -> SyntheticSet:
    """Build the pair set over the validation's indexed documents and passages."""
    config = CONTRACT_NLI_CONFIG
    funnel = Funnel()
    texts = {d.doc_id: d.text for d in documents}
    indexed, held_out = held_out_split(
        documents, share=config.held_out_share, minimum=config.held_out_minimum, seed=config.seed
    )
    probe = ProbeIndex(
        indexed, embedder, chunk_chars=config.probe_chunk_chars, overlap_chars=config.probe_overlap_chars
    )
    passages, _ = sample_passages(
        indexed,
        per_document=config.passages_per_document,
        seed=config.seed,
        target_chars=config.passage_chars,
        encode=probe.encode_texts,
        boilerplate_cosine=config.boilerplate_cosine,
    )
    found = map_concurrent(
        lambda p: extract_statements(generator, p, texts[p.doc_id], count=config.statements_per_passage),
        passages,
        config.workers,
    )
    statements = [s for located, _ in found for s in located]
    pairs = pairs_of(statements)
    chain = lambda pair: _pair_chain(  # noqa: E731
        pair, generator=generator, checker=checker, judge=judge, probe=probe, texts=texts, funnel=funnel
    )
    passed = [q for q in map_concurrent(chain, pairs, config.workers) if q is not None]
    accepted = _deduplicate(passed, probe, config, funnel)
    for question in accepted:
        funnel.finish(question.query_id, KIND, "accepted", "accepted")
    meta = {
        "documents": {"indexed": [d.doc_id for d in indexed], "held_out": [d.doc_id for d in held_out]},
        "statements_located": len(statements),
        "pairs": len(pairs),
        "usage": [usage_of(m) for m in {m.name: m for m in (generator, checker, judge)}.values()],
    }
    return SyntheticSet(accepted, funnel, meta)


def widened(plain: SyntheticSet, texts: dict[str, str], target: int) -> SyntheticSet:
    """Return the plain set with each gold span widened to ``target`` characters."""
    questions = []
    for q in plain.answerable:
        gold = q.evidence[0]
        total = sum(span.end - span.start for span in q.evidence)
        grow = max(0, target - total)
        text = texts[gold.doc_id]
        start = max(0, gold.start - grow // 2)
        end = min(len(text), gold.end + grow - (gold.start - start))
        span = EvidenceSpan(gold.doc_id, start, end, text[start:end])
        questions.append(replace(q, evidence=(span, *q.evidence[1:])))
    return SyntheticSet(questions, None, plain.meta)


def gold_stats(spans_per_query: Sequence[Sequence[tuple[int, int]]]) -> dict[str, float]:
    """Return mean spans per query and mean total gold characters per query."""
    return {
        "queries": len(spans_per_query),
        "spans_per_query": round(statistics.fmean(len(s) for s in spans_per_query), 2),
        "gold_chars_per_query": round(statistics.fmean(sum(e - b for b, e in s) for s in spans_per_query), 1),
    }


def explore_compute(
    documents: Sequence[Document],
    real_by_doc: dict[str, list[Query]],
    roles: Roles,
    *,
    plain: SyntheticSet | None = None,
    per_minute: int = 15,
    env_path: Path | None = None,
) -> Callable[[ModelChoice], dict[str, Any]]:
    """Return the compute ``load_or_run`` calls for the follow-up run.

    ``plain`` is the validation's set; it is read from the cache when omitted.
    """

    def compute(choice: ModelChoice) -> dict[str, Any]:
        import sqsets  # noqa: PLC0415
        from aieng.advanced_rag.retrieval.embedders import (  # noqa: PLC0415
            CachedQueryEmbedder,
            SentenceTransformerEmbedder,
        )

        started = time.perf_counter()
        base = plain if plain is not None else sqsets.load_validation()[1]
        base = base.variant("plain")
        embedder = SentenceTransformerEmbedder(DENSE_MODEL)
        models = sqsets.models_for(roles, per_minute=per_minute, env_path=env_path)
        pairs = build_pairs(documents, embedder=embedder, **models)
        logging.info("pair set %s", pairs.summary())

        indexed = set(base.meta["documents"]["indexed"])
        scored = [d for d in documents if d.doc_id in indexed]
        texts = {d.doc_id: d.text for d in documents}
        real_queries = [q for d in scored for q in real_by_doc[d.doc_id] if q.is_answerable]
        real_stats = gold_stats([[(a.start, a.end) for a in q.answers] for q in real_queries])
        target = int(real_stats["gold_chars_per_query"])
        wide = widened(base, texts, target)

        cached = CachedQueryEmbedder(embedder)
        cached.encode_queries(
            sorted({q.question for q in real_queries} | {q.question for q in base} | {q.question for q in pairs})
        )
        arms = build_arms(scored, cached)
        names = list(arms)
        real = score_arms(arms, real_queries, lambda chunks: real_qrels(real_queries, chunks))
        label_sets = {"plain": base, "pairs": pairs, "widened": wide}
        results: dict[str, Any] = {}
        for name, labels in label_sets.items():
            scores = score_arms(arms, queries_for(labels), labels.qrels)
            results[name] = {
                "gold": gold_stats([[(s.start, s.end) for s in q.evidence] for q in labels.answerable]),
                "generated_mean": mean_scores(scores),
                **agreement(real, scores, arm_groups(names)),
            }
        alone = [bool(q.provenance.get("first_alone_answers")) for q in pairs]
        return {
            "mode": "explore: one generation seed, one configuration",
            "arms": names,
            "real": {"gold": real_stats, "mean": mean_scores(real)},
            "label_sets": results,
            "pair_set": {
                "summary": pairs.summary(),
                "funnel": pairs.funnel.to_dict() if pairs.funnel else {},
                "pairs_tried": pairs.meta["pairs"],
                "first_statement_alone_answers": round(sum(alone) / len(alone), 4) if alone else None,
                "usage": pairs.meta["usage"],
                "questions": [q.to_dict() for q in pairs],
            },
            "widened_target_chars": target,
            "seconds": round(time.perf_counter() - started, 1),
        }

    return compute


def main() -> int:
    """Run the explore step on the on-prem models and store it in the shipped tier."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--per-minute", type=int, default=15)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import sqsets  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from aieng.advanced_rag.cache import TieredResultCache  # noqa: PLC0415
    from aieng.advanced_rag.evaluation.synthetic.replay import (  # noqa: PLC0415
        REFERENCE_ROLES,
        load_or_build,
    )

    torch.set_num_threads(4)
    if args.per_minute > 15:
        raise SystemExit("keep the on-prem models at 15 requests per minute or fewer")
    docs, real = sqsets.contract_nli(sqsets.REFERENCE_DOCUMENTS)
    _, plain, _ = sqsets.load_validation()
    writer = TieredResultCache(local_dir=sqsets.BASELINE)
    result = load_or_build(
        sqsets.explore_step(docs, REFERENCE_ROLES),
        explore_compute(docs, real, REFERENCE_ROLES, plain=plain, per_minute=args.per_minute, env_path=args.env_file),
        roles=REFERENCE_ROLES,
        live=True,
        cache=writer,
    )
    for name, row in result.value["label_sets"].items():
        tau = row["all_arms"]["ndcg@10"]["kendall_tau"]
        logging.info("%s gold %s tau(nDCG@10) %s", name, row["gold"], tau)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
