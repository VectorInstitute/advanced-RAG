"""Run the stages end to end and return a :class:`SyntheticSet` with its funnel.

:func:`build_single_hop_set` is the path most teams need: statement-first
single-hop questions, the filter chain, in-domain nulls, a paraphrase variant
of each answerable question, and the recorded difficulty labels.
:func:`build_multi_hop_questions` and :func:`build_fanout_questions` run the
ported pair agent and tag fan-out generator, and return questions in the same
record, so the sets concatenate.

Every candidate ends in exactly one outcome in the funnel. Candidates run
concurrently, which overlaps calls to the generator, checker, and judge; each
model's own rate limiter bounds its requests, and the category draw is seeded
per candidate, so the result does not depend on scheduling.
"""

from __future__ import annotations

import contextvars
import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, TypeVar

from .agent import Proposal, assigned_type, locate_evidence, run_agent
from .config import GenerationConfig
from .fanout import coverage, propose
from .filters import (
    all_but_one_suffices,
    deduplicate,
    multi_hop_filters,
    round_trip,
    single_document_suffices,
    uniqueness,
)
from .generate import write_question
from .labels import difficulty
from .llm import Model, family, usage_of
from .nulls import check_null, near_miss_question
from .probe import ProbeIndex, TextEncoder
from .rewrite import leaked_terms, paraphrase_question, term_frequencies
from .sampling import (
    CandidatePair,
    Condition,
    Document,
    held_out_split,
    sample_passages,
)
from .statements import Statement, extract_statements
from .taxonomy import BY_NAME, HELD_OUT_NULL, NEAR_MISS_NULL, PARTIAL_NULL, SINGLE_HOP
from .testset import EvidenceSpan, Funnel, SyntheticQuestion, SyntheticSet, question_id


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: The reference answer recorded on a null question.
NULL_REFERENCE = "Not stated in the documents."


def map_concurrent(fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """Apply ``fn`` to every item on a thread pool, keeping input order.

    Each item runs in a copy of the caller's context. A pool worker otherwise
    starts with an empty one, and the usage meter that ``load_or_run`` binds
    would never see the calls made in it.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    contexts = [contextvars.copy_context() for _ in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda pair: pair[0].run(fn, pair[1]), zip(contexts, items))
        )


def _scope_for(config: GenerationConfig, doc_id: str) -> str | None:
    return doc_id if config.scope == "document" else None


def _answerable_chain(
    statement: Statement,
    *,
    config: GenerationConfig,
    generator: Model,
    checker: Model,
    judge: Model,
    probe: ProbeIndex,
    texts: dict[str, str],
    funnel: Funnel,
) -> SyntheticQuestion | None:
    """Write, round-trip, and uniqueness-check one statement's question."""
    candidate = question_id(
        "single_hop", statement.doc_id, statement.start, statement.end
    )
    kind = SINGLE_HOP.name
    draft = write_question(generator, statement, config, key=candidate)
    if not funnel.check("question_written", not draft.rejected, kind):
        funnel.finish(
            candidate,
            kind,
            "rejected",
            "question_written",
            draft.reason or "generator declined",
        )
        return None

    trip = round_trip(checker, judge, draft.question, draft.answer, [statement.quote])
    if not funnel.check("round_trip", trip["passed"], kind):
        funnel.finish(
            candidate,
            kind,
            "filtered",
            "round_trip",
            f"{trip['verdict']}: {trip['reason']}"[:200],
        )
        return None

    scope = _scope_for(config, statement.doc_id)
    gold = EvidenceSpan(
        statement.doc_id, statement.start, statement.end, statement.quote
    )
    hits = probe.search(
        draft.question,
        method="dense",
        k=config.uniqueness_k,
        scope=scope,
        exclude=[(gold.doc_id, gold.start, gold.end)],
    )
    unique = uniqueness(checker, draft.question, draft.answer, hits, texts)
    if not funnel.check("uniqueness", unique.status != "ambiguous", kind):
        funnel.finish(
            candidate,
            kind,
            "filtered",
            "uniqueness",
            "another passage gives a different answer",
        )
        return None

    return SyntheticQuestion(
        query_id=candidate,
        question=draft.question,
        reference=draft.answer,
        evidence=(gold, *unique.equivalent),
        kind=kind,
        scope=scope,
        group_id=candidate,
        provenance={
            "user_category": draft.categories["user"],
            "question_category": draft.categories["question"],
            "phrasing": draft.categories["phrasing"],
            "generator": generator.name,
            "checker": checker.name,
            "judge": judge.name,
            "round_trip_answer": trip["answer"],
            "uniqueness": unique.status,
            "claimed_quote": statement.claimed,
        },
    )


def _statements(
    passages: Sequence[Any],
    *,
    generator: Model,
    texts: dict[str, str],
    config: GenerationConfig,
    funnel: Funnel,
    kind: str,
) -> list[Statement]:
    """Extract and locate statements, recording unlocated quotes in the funnel."""
    results = map_concurrent(
        lambda p: extract_statements(
            generator, p, texts[p.doc_id], count=config.statements_per_passage
        ),
        passages,
        config.workers,
    )
    located: list[Statement] = []
    for passage, (found, missing) in zip(passages, results):
        for statement in found:
            funnel.check("quote_located", True, kind)
            located.append(statement)
        for quote in missing:
            funnel.check("quote_located", False, kind)
            funnel.finish(
                question_id(kind, "unlocated", passage.passage_id, quote),
                kind,
                "filtered",
                "quote_located",
                "quote not found in the document",
            )
    return located


def _null_batches(
    sources: Sequence[T],
    quota: int,
    attempt: Callable[[T], SyntheticQuestion | None],
    *,
    workers: int,
    funnel: Funnel,
) -> list[SyntheticQuestion]:
    """Try sources in batches until ``quota`` nulls are accepted or sources run out.

    Each batch is twice the remaining quota, so a run does not spend calls on
    candidates it will not keep at the acceptance rates seen so far. A batch
    that overshoots keeps the first ``quota`` in source order and records the
    rest as filtered, so the null share stays what the config asked for.
    """
    accepted: list[SyntheticQuestion] = []
    position = 0
    while len(accepted) < quota and position < len(sources):
        size = max(2 * (quota - len(accepted)), 2)
        batch = sources[position : position + size]
        position += len(batch)
        for result in map_concurrent(attempt, batch, workers):
            if result is None:
                continue
            if len(accepted) < quota:
                accepted.append(result)
            else:
                funnel.finish(
                    result.query_id,
                    result.kind,
                    "filtered",
                    "null_quota",
                    "over the null share",
                )
    return accepted


def build_single_hop_set(
    documents: Sequence[Document],
    config: GenerationConfig,
    *,
    generator: Model,
    checker: Model,
    judge: Model,
    embedder: TextEncoder,
) -> SyntheticSet:
    """Generate a single-hop set with nulls, a paraphrase variant, and labels.

    Parameters
    ----------
    documents : sequence of Document
        The corpus, read through the same loader as the index, so offsets match.
    config : GenerationConfig
        Categories, few-shot examples, sizes, and scope.
    generator : Model
        Writes statements, questions, near-miss nulls, and paraphrases.
    checker : Model
        Answers from gold quotes, checks uniqueness, and verifies nulls. Use a
        different model, ideally a different family, from the generator.
    judge : Model
        Compares the checker's round-trip answer with the reference.
    embedder : TextEncoder
        A local encoder for the probe index and deduplication.

    Returns
    -------
    SyntheticSet
        Accepted questions, the funnel, and a ``meta`` record of models,
        configuration, document split, sampling, and usage.
    """
    started = time.perf_counter()
    funnel = Funnel()
    texts = {d.doc_id: d.text for d in documents}
    indexed, held_out = held_out_split(
        documents,
        share=config.held_out_share,
        minimum=config.held_out_minimum,
        seed=config.seed,
    )
    probe = ProbeIndex(
        indexed,
        embedder,
        chunk_chars=config.probe_chunk_chars,
        overlap_chars=config.probe_overlap_chars,
    )
    frequencies = term_frequencies([d.text for d in indexed])

    passages, sample_report = sample_passages(
        indexed,
        per_document=config.passages_per_document,
        seed=config.seed,
        target_chars=config.passage_chars,
        encode=probe.encode_texts,
        boilerplate_cosine=config.boilerplate_cosine,
    )
    logger.info("sampled %d passages from %d documents", len(passages), len(indexed))
    statements = _statements(
        passages,
        generator=generator,
        texts=texts,
        config=config,
        funnel=funnel,
        kind=SINGLE_HOP.name,
    )
    logger.info("located %d statements", len(statements))

    chain = lambda s: _answerable_chain(  # noqa: E731
        s,
        config=config,
        generator=generator,
        checker=checker,
        judge=judge,
        probe=probe,
        texts=texts,
        funnel=funnel,
    )
    passed = [
        q for q in map_concurrent(chain, statements, config.workers) if q is not None
    ]
    answerable = _deduplicate(passed, probe, config, funnel)
    logger.info("accepted %d answerable questions", len(answerable))

    quota = round(config.null_share / (1 - config.null_share) * len(answerable))
    near_quota = round(quota * config.near_miss_share)
    held_quota = quota - near_quota
    nulls = _near_miss_nulls(
        answerable,
        near_quota,
        config=config,
        generator=generator,
        checker=checker,
        probe=probe,
        funnel=funnel,
    )
    nulls += _held_out_nulls(
        held_out,
        indexed,
        quota=held_quota,
        config=config,
        generator=generator,
        checker=checker,
        probe=probe,
        texts=texts,
        funnel=funnel,
    )
    nulls = _deduplicate(nulls, probe, config, funnel)
    logger.info("accepted %d null questions", len(nulls))

    variants = (
        _paraphrases(
            answerable,
            config,
            generator=generator,
            checker=checker,
            judge=judge,
            frequencies=frequencies,
            total=len(indexed),
            funnel=funnel,
        )
        if config.paraphrase
        else []
    )
    questions = [
        replace(q, labels=difficulty(q, probe, frequencies, len(indexed)))
        for q in [*answerable, *variants, *nulls]
    ]
    for question in [*answerable, *nulls]:
        funnel.finish(question.query_id, question.kind, "accepted", "accepted")

    meta = {
        "models": {
            "generator": generator.name,
            "checker": checker.name,
            "judge": judge.name,
        },
        "same_family": {
            "generator_checker": family(generator.name) == family(checker.name),
            "generator_judge": family(generator.name) == family(judge.name),
        },
        "config": config.to_dict(),
        "documents": {
            "indexed": [d.doc_id for d in indexed],
            "held_out": [d.doc_id for d in held_out],
        },
        "sampling": sample_report,
        "statements_located": len(statements),
        "usage": [usage_of(m) for m in _distinct((generator, checker, judge))],
        "seconds": round(time.perf_counter() - started, 1),
    }
    return SyntheticSet(questions, funnel, meta)


def _distinct(models: Iterable[Model]) -> list[Model]:
    seen: dict[str, Model] = {}
    for model in models:
        seen.setdefault(model.name, model)
    return list(seen.values())


def _deduplicate(
    questions: list[SyntheticQuestion],
    probe: ProbeIndex,
    config: GenerationConfig,
    funnel: Funnel,
) -> list[SyntheticQuestion]:
    if not questions:
        return []
    vectors = dict(
        zip(
            [q.query_id for q in questions],
            probe.encode_texts([q.question for q in questions]),
        )
    )
    items = [
        (q.query_id, (q.documents or (q.scope or "",))[0], q.question)
        for q in questions
    ]
    kept, dropped = deduplicate(items, vectors, config.dedup_cosine)
    kept_set = set(kept)
    by_id = {q.query_id: q for q in questions}
    for question in questions:
        funnel.check("deduplicated", question.query_id in kept_set, question.kind)
    for duplicate in dropped:
        question = by_id[duplicate.dropped]
        funnel.finish(
            question.query_id,
            question.kind,
            "filtered",
            "deduplicated",
            f"near-duplicate of {duplicate.kept} at cosine {duplicate.similarity}",
        )
    return [by_id[query_id] for query_id in kept]


def _near_miss_nulls(
    answerable: Sequence[SyntheticQuestion],
    quota: int,
    *,
    config: GenerationConfig,
    generator: Model,
    checker: Model,
    probe: ProbeIndex,
    funnel: Funnel,
) -> list[SyntheticQuestion]:
    sources = list(answerable)
    random.Random(f"{config.seed}:near-miss").shuffle(sources)
    kind = NEAR_MISS_NULL.name

    def attempt(source: SyntheticQuestion) -> SyntheticQuestion | None:
        candidate = question_id(kind, source.query_id)
        gold = source.evidence[0]
        written = near_miss_question(
            generator, source.question, gold.quote, config.domain
        )
        if not funnel.check("null_written", bool(written["question"]), kind):
            funnel.finish(
                candidate,
                kind,
                "rejected",
                "null_written",
                "generator returned no question",
            )
            return None
        scope = source.scope
        verdict = check_null(
            checker, probe, written["question"], scope=scope, k=config.null_check_k
        )
        if not funnel.check("null_check", verdict["passed"], kind):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "null_check",
                f"corpus answers it ({verdict['answered']})",
            )
            return None
        return SyntheticQuestion(
            query_id=candidate,
            question=written["question"],
            reference=NULL_REFERENCE,
            kind=kind,
            scope=scope,
            group_id=candidate,
            provenance={
                "source_question": source.query_id,
                "changed": written["changed"],
                "generator": generator.name,
                "checker": checker.name,
                "null_check_passages": verdict["passages"],
            },
        )

    return _null_batches(sources, quota, attempt, workers=config.workers, funnel=funnel)


def _held_out_nulls(
    held_out: Sequence[Document],
    indexed: Sequence[Document],
    *,
    quota: int,
    config: GenerationConfig,
    generator: Model,
    checker: Model,
    probe: ProbeIndex,
    texts: dict[str, str],
    funnel: Funnel,
) -> list[SyntheticQuestion]:
    if not held_out or quota <= 0:
        return []
    kind = HELD_OUT_NULL.name
    passages, _ = sample_passages(
        held_out,
        per_document=config.held_out_passages,
        seed=config.seed,
        target_chars=config.passage_chars,
    )
    statements = _statements(
        passages,
        generator=generator,
        texts=texts,
        config=config,
        funnel=funnel,
        kind=kind,
    )
    random.Random(f"{config.seed}:held-out").shuffle(statements)
    targets = [d.doc_id for d in indexed]

    def attempt(statement: Statement) -> SyntheticQuestion | None:
        candidate = question_id(kind, statement.doc_id, statement.start, statement.end)
        draft = write_question(generator, statement, config, key=candidate)
        if not funnel.check("null_written", not draft.rejected, kind):
            funnel.finish(
                candidate,
                kind,
                "rejected",
                "null_written",
                draft.reason or "generator declined",
            )
            return None
        scope = (
            random.Random(f"{config.seed}:{candidate}").choice(targets)
            if config.scope == "document"
            else None
        )
        verdict = check_null(
            checker, probe, draft.question, scope=scope, k=config.null_check_k
        )
        if not funnel.check("null_check", verdict["passed"], kind):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "null_check",
                f"corpus answers it ({verdict['answered']})",
            )
            return None
        return SyntheticQuestion(
            query_id=candidate,
            question=draft.question,
            reference=NULL_REFERENCE,
            kind=kind,
            scope=scope,
            group_id=candidate,
            provenance={
                "held_out_document": statement.doc_id,
                "user_category": draft.categories["user"],
                "question_category": draft.categories["question"],
                "phrasing": draft.categories["phrasing"],
                "generator": generator.name,
                "checker": checker.name,
                "null_check_passages": verdict["passages"],
            },
        )

    return _null_batches(
        statements, quota, attempt, workers=config.workers, funnel=funnel
    )


def _paraphrases(
    answerable: Sequence[SyntheticQuestion],
    config: GenerationConfig,
    *,
    generator: Model,
    checker: Model,
    judge: Model,
    frequencies: dict[str, int],
    total: int,
    funnel: Funnel,
) -> list[SyntheticQuestion]:
    def attempt(source: SyntheticQuestion) -> SyntheticQuestion | None:
        quote = " ".join(s.quote for s in source.evidence if s.role == "gold")
        before = leaked_terms(
            source.question, "", quote, frequencies, total, max_document_share=0.5
        )
        if not funnel.check("has_leak", bool(before), "paraphrase"):
            return None
        rewritten = paraphrase_question(
            generator, source.question, source.reference, before
        )
        after = (
            leaked_terms(
                rewritten, "", quote, frequencies, total, max_document_share=0.5
            )
            if rewritten
            else before
        )
        if not funnel.check(
            "leak_reduced", bool(rewritten) and len(after) < len(before), "paraphrase"
        ):
            return None
        trip = round_trip(
            checker,
            judge,
            rewritten,
            source.reference,
            [s.quote for s in source.evidence],
        )
        if not funnel.check("round_trip", trip["passed"], "paraphrase"):
            return None
        return replace(
            source,
            query_id=f"{source.query_id}-p",
            question=rewritten,
            variant="paraphrase",
            provenance={
                **dict(source.provenance),
                "paraphrase_of": source.query_id,
                "leaked_before": list(before),
                "leaked_after": list(after),
                "paraphrase_round_trip_answer": trip["answer"],
            },
        )

    return [
        q for q in map_concurrent(attempt, answerable, config.workers) if q is not None
    ]


# ---------------------------------------------------------------------------
# Multi-hop and fan-out, from the ported generators.
# ---------------------------------------------------------------------------


def build_multi_hop_questions(
    pairs: Sequence[CandidatePair],
    documents: Sequence[Document],
    *,
    generator: Model,
    checker: Model,
    search_corpus: Callable[[str, int], list[tuple[str, str]]],
    held_out: Iterable[str] = (),
    probe: ProbeIndex | None = None,
    funnel: Funnel | None = None,
    workers: int = 4,
    max_steps: int = 6,
) -> tuple[list[SyntheticQuestion], Funnel]:
    """Turn candidate pairs into checked multi-hop questions, with spans.

    A pair whose second document is held out becomes a partial null when the
    agent proposes a question: half its evidence is indexed and half is not.
    It is kept only if ``probe`` (built over the indexed documents) finds
    nothing that answers it.

    Parameters
    ----------
    pairs : sequence of CandidatePair
        From :func:`~aieng.advanced_rag.evaluation.synthetic.sampling.select_pairs`.
    documents : sequence of Document
        Every document the pairs name, held-out ones included.
    generator : Model
        Runs the agent.
    checker : Model
        Runs the verifier, the defeat test, and the null check.
    search_corpus : callable
        The agent's search tool, ``(query, k) -> [(doc_id, snippet)]``.
    held_out : iterable of str, optional
        Held-out document identifiers.
    probe : ProbeIndex or None, optional
        Required for partial nulls.
    funnel : Funnel or None, optional
        Shared with another build, or a new one.
    workers : int, optional
        Pairs processed at once.
    max_steps : int, optional
        Agent tool budget.

    Returns
    -------
    tuple of (list of SyntheticQuestion, Funnel)
        Accepted questions, scoped to the corpus, and the funnel.
    """
    funnel = funnel if funnel is not None else Funnel()
    texts = {d.doc_id: d.text for d in documents}
    held = set(held_out)
    requested = BY_NAME["bridge"]

    def attempt(pair: CandidatePair) -> SyntheticQuestion | None:
        partial = pair.doc_b in held or pair.doc_a in held
        kind = PARTIAL_NULL.name if partial else requested.name
        candidate = question_id(kind, *pair.key)
        result = run_agent(
            generator,
            pair,
            requested,
            read_document=lambda d: texts.get(d, ""),
            search_corpus=search_corpus,
            max_steps=max_steps,
        )
        if not funnel.check("agent_proposed", isinstance(result, Proposal), kind):
            funnel.finish(
                candidate,
                kind,
                "rejected",
                "agent_proposed",
                getattr(result, "reason", ""),
            )
            return None
        assert isinstance(result, Proposal)
        spans, missing = locate_evidence(result, texts)
        if not funnel.check("quote_located", not missing, kind):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "quote_located",
                f"no located quote from {missing[:2]}",
            )
            return None
        if partial:
            return _partial_null(
                candidate,
                result,
                checker=checker,
                probe=probe,
                funnel=funnel,
                generator=generator,
            )
        kind = assigned_type(result, requested).name
        verdict = multi_hop_filters(
            checker,
            result.question,
            answer=result.answer,
            bridge=result.bridge,
            gold=result.needs_documents,
            documents=texts,
        )
        if not funnel.check("multi_hop_filters", verdict.accepted, kind):
            funnel.finish(
                candidate, kind, "filtered", "multi_hop_filters", verdict.reason
            )
            return None
        funnel.finish(candidate, kind, "accepted", "accepted")
        return SyntheticQuestion(
            query_id=candidate,
            question=result.question,
            reference=result.answer,
            evidence=tuple(spans),
            kind=kind,
            group_id=candidate,
            provenance={
                "bridge": result.bridge,
                "pair": list(result.pair),
                "agent_steps": result.steps,
                "generator": generator.name,
                "checker": checker.name,
            },
        )

    accepted = [q for q in map_concurrent(attempt, pairs, workers) if q is not None]
    return accepted, funnel


def _partial_null(
    candidate: str,
    proposal: Proposal,
    *,
    checker: Model,
    probe: ProbeIndex | None,
    funnel: Funnel,
    generator: Model,
) -> SyntheticQuestion | None:
    kind = PARTIAL_NULL.name
    if probe is None:
        funnel.finish(
            candidate,
            kind,
            "filtered",
            "null_check",
            "no probe index to verify against",
        )
        return None
    verdict = check_null(checker, probe, proposal.question, scope=None)
    if not funnel.check("null_check", verdict["passed"], kind):
        funnel.finish(
            candidate,
            kind,
            "filtered",
            "null_check",
            f"corpus answers it ({verdict['answered']})",
        )
        return None
    funnel.finish(candidate, kind, "accepted", "accepted")
    return SyntheticQuestion(
        query_id=candidate,
        question=proposal.question,
        reference=NULL_REFERENCE,
        kind=kind,
        group_id=candidate,
        provenance={
            "pair": list(proposal.pair),
            "bridge": proposal.bridge,
            "generator": generator.name,
            "checker": checker.name,
        },
    )


def build_fanout_questions(
    conditions: Sequence[Condition],
    documents: Sequence[Document],
    *,
    generator: Model,
    checker: Model,
    min_coverage: float = 0.8,
    seed: int = 0,
    funnel: Funnel | None = None,
    workers: int = 4,
) -> tuple[list[SyntheticQuestion], Funnel]:
    """Turn tag conditions into checked fan-out questions, with spans.

    Parameters
    ----------
    conditions : sequence of Condition
        From :func:`~aieng.advanced_rag.evaluation.synthetic.sampling.tag_conditions`.
    documents : sequence of Document
        The documents the conditions name.
    generator : Model
        Writes the question.
    checker : Model
        Runs the defeat and all-but-one tests.
    min_coverage : float, optional
        Share of gold documents that must contribute a located quote.
    seed : int, optional
        Seeds which document the all-but-one test removes.
    funnel : Funnel or None, optional
        Shared with another build, or a new one.
    workers : int, optional
        Conditions processed at once.

    Returns
    -------
    tuple of (list of SyntheticQuestion, Funnel)
        Accepted questions and the funnel.
    """
    funnel = funnel if funnel is not None else Funnel()
    by_id = {d.doc_id: d for d in documents}
    kind = "fan_out"

    def attempt(condition: Condition) -> SyntheticQuestion | None:
        candidate = question_id(kind, condition.value, *condition.documents)
        docs = [by_id[d] for d in condition.documents]
        proposal = propose(generator, condition, docs)
        if not funnel.check("fanout_proposed", proposal is not None, kind):
            funnel.finish(
                candidate,
                kind,
                "rejected",
                "fanout_proposed",
                "generator rejected the condition",
            )
            return None
        assert proposal is not None
        if not funnel.check(
            "fanout_coverage", coverage(proposal) >= min_coverage, kind
        ):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "fanout_coverage",
                "too few documents contribute",
            )
            return None
        located = Proposal(
            question=proposal.question,
            answer=proposal.answer,
            needs_documents=proposal.needs_documents,
            bridge="",
            rationale="",
            question_type=kind,
            pair=(docs[0].doc_id, docs[-1].doc_id),
            steps=1,
            quotes=list(proposal.per_document.items()),
        )
        spans, missing = locate_evidence(located, {d.doc_id: d.text for d in docs})
        if not funnel.check(
            "quote_located", len(missing) <= len(docs) * (1 - min_coverage), kind
        ):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "quote_located",
                f"no located quote from {missing[:2]}",
            )
            return None
        first = docs[0]
        if not funnel.check(
            "single_document_defeat",
            not single_document_suffices(
                checker, proposal.question, first.doc_id, first.text
            ),
            kind,
        ):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "single_document_defeat",
                "one document answers it",
            )
            return None
        dropped = random.Random(f"{seed}:{candidate}").choice(docs)
        remaining = [(d.doc_id, d.text) for d in docs if d is not dropped]
        if not funnel.check(
            "all_but_one",
            not all_but_one_suffices(checker, proposal.question, remaining),
            kind,
        ):
            funnel.finish(
                candidate,
                kind,
                "filtered",
                "all_but_one",
                "answerable with a gold document removed",
            )
            return None
        funnel.finish(candidate, kind, "accepted", "accepted")
        return SyntheticQuestion(
            query_id=candidate,
            question=proposal.question,
            reference=proposal.answer,
            evidence=tuple(spans),
            kind=kind,
            group_id=candidate,
            provenance={
                "condition": condition.to_dict(),
                "dropped_document": dropped.doc_id,
                "generator": generator.name,
                "checker": checker.name,
            },
        )

    accepted = [
        q for q in map_concurrent(attempt, conditions, workers) if q is not None
    ]
    return accepted, funnel


__all__ = [
    "NULL_REFERENCE",
    "build_fanout_questions",
    "build_multi_hop_questions",
    "build_single_hop_set",
    "map_concurrent",
]
