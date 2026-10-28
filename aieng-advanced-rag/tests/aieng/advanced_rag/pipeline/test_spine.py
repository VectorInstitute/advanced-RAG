"""Tests for the composed pipeline: retrieve, grade, correct, answer or abstain."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aieng.advanced_rag.pipeline.answering import (
    CorrectivePolicy,
    FixedPromptSelector,
    PassthroughAnalysis,
    QueryPlan,
    RewriterAnalysis,
    SingleShotPolicy,
)
from aieng.advanced_rag.pipeline.grading import (
    Action,
    ChunkGrade,
    EvidenceGrade,
    GradingThresholds,
    Verdict,
    pooled_score,
)
from aieng.advanced_rag.pipeline.hierarchical import split_document, to_evidence
from aieng.advanced_rag.pipeline.rewrite import Rewrite, RewritePlan
from aieng.advanced_rag.pipeline.spine import (
    AdvancedRagSpine,
    SpineConfig,
    abstentions,
    evidence_from_hits,
    retrieval_runs,
    run_spine,
    with_small_to_big,
)


# The loop policy owns the verdict, so the boundary a test drives the decision
# across is stated on ``SpineConfig`` and mirrored on the grader, which keeps
# reporting its own advisory verdict. The shipped ``insufficient`` is 0.0, under
# which no score is insufficient at all, so a test of the abstain path has to
# state a boundary of its own.
LOOP_BOUNDARY = GradingThresholds(sufficient=0.6, insufficient=0.3)


class FakeGrader:
    """Grades from a queue of scores, so a test states the loop it wants.

    The thresholds are pinned here rather than defaulted, so the advisory
    verdict on each ``Attempt`` reads against the same boundary the loop policy
    applies. A test that drives the decision across that boundary also passes
    :data:`LOOP_BOUNDARY` to ``SpineConfig``, which is what the policy reads.
    """

    name = "fake"

    def __init__(
        self, *scores: float, thresholds: GradingThresholds | None = None
    ) -> None:
        self.queue = list(scores)
        self.last = 0.0
        self.questions: list[str] = []
        self.thresholds = thresholds or LOOP_BOUNDARY

    def grade(self, question, evidence, query_id=""):
        self.questions.append(question)
        if self.queue:
            self.last = self.queue.pop(0)
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=self.thresholds.verdict(self.last),
            score=self.last,
            reason=f"scored {self.last}",
            thresholds=self.thresholds,
        )


class FakeRewriter:
    """Returns one fixed rewrite, recording the feedback it was given."""

    name = "fake-rewriter"

    def __init__(self, rewritten: str = "rewritten question") -> None:
        self.rewritten = rewritten
        self.feedback: list[str] = []

    def rewrite(self, question, feedback=""):
        self.feedback.append(feedback)
        if not self.rewritten:
            return RewritePlan(original=question, outcome="no-change")
        return RewritePlan(
            original=question,
            rewrites=(Rewrite(self.rewritten, self.name),),
        )


class ThreeWayRewriter:
    """Returns three candidates, so a plan above the default bound is visible."""

    name = "three-way"

    def rewrite(self, question, feedback=""):
        return RewritePlan(
            original=question,
            rewrites=tuple(
                Rewrite(f"{question} v{index}", self.name) for index in (1, 2, 3)
            ),
        )


class PerPassageGrader:
    """Scores each passage by its text and pools the way the package pools.

    ``FakeGrader`` states a pooled score outright, which cannot show where that
    number came from. This one scores passages and calls ``pooled_score``, so a
    test can place the only scoring passage in one candidate's ranking and watch
    it carry the whole attempt.
    """

    name = "per-passage"

    def __init__(self, scores, thresholds=None) -> None:
        self.scores = scores
        self.thresholds = thresholds or LOOP_BOUNDARY

    def grade(self, question, evidence, query_id=""):
        grades = tuple(
            ChunkGrade(
                chunk_id=item.chunk_id or item.doc_id,
                score=self.scores.get(item.text, 0.0),
                reason="s",
            )
            for item in evidence
        )
        score = pooled_score(grades)
        return EvidenceGrade(
            query_id=query_id,
            question=question,
            verdict=self.thresholds.verdict(score),
            score=score,
            reason=f"scored {score}",
            chunk_grades=grades,
            thresholds=self.thresholds,
        )


class Recorder:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


ANSWER = json.dumps({"answer": "Ontario law governs.", "sources": [1]})


def retriever(make_evidence, texts=("first passage", "second passage")):
    """Return a retrieval callable that records the questions it was asked."""
    asked: list[tuple[str, str | None]] = []

    def retrieve(question, scope=None, k=10):
        asked.append((question, scope))
        return make_evidence(list(texts))

    retrieve.asked = asked
    return retrieve


def tagged_retriever(make_evidence):
    """Return a retriever whose passages name the candidate that found them.

    The last word of the question becomes the document, so each step of a
    multi-candidate plan contributes its own chunk identifiers and ``to_runs``
    does not fold them together.
    """
    asked: list[str] = []

    def retrieve(question, scope=None, k=10):
        asked.append(question)
        tag = question.rsplit(" ", 1)[-1]
        return make_evidence([f"{tag} first", f"{tag} second"], doc_id=tag)

    retrieve.asked = asked
    return retrieve


def test_sufficient_evidence_answers_on_the_first_attempt(make_evidence):
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(retriever(make_evidence), FakeGrader(0.9), model)
    result = spine.answer("q1", "who signs")
    assert len(result.attempts) == 1
    assert result.attempts[0].action is Action.ANSWER
    assert result.attempts[0].strategy == "original"
    assert not result.abstained
    assert len(model.prompts) == 1


def test_a_weak_grade_retrieves_again_with_the_rewritten_question(make_evidence):
    retrieve = retriever(make_evidence)
    grader = FakeGrader(0.0, 0.9)
    rewriter = FakeRewriter()
    spine = AdvancedRagSpine(retrieve, grader, Recorder(ANSWER), rewriter=rewriter)
    result = spine.answer("q1", "who signs")
    assert [attempt.question for attempt in result.attempts] == [
        "who signs",
        "rewritten question",
    ]
    assert result.attempts[0].action is Action.REWRITE
    assert result.attempts[1].strategy == "fake-rewriter"
    assert not result.abstained


def test_the_rewriter_is_told_why_the_previous_attempt_failed(make_evidence):
    rewriter = FakeRewriter()
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.0, 0.9),
        Recorder(ANSWER),
        rewriter=rewriter,
    )
    spine.answer("q1", "who signs")
    assert rewriter.feedback == ["scored 0.0"]


def test_weak_evidence_and_no_rewriter_abstains_without_a_generation_call(
    make_evidence,
):
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.0),
        model,
        config=SpineConfig(thresholds=LOOP_BOUNDARY),
    )
    result = spine.answer("q1", "who signs")
    assert result.abstained
    assert len(result.attempts) == 1
    assert result.attempts[0].action is Action.ABSTAIN
    assert model.prompts == []
    assert "insufficient" in result.answer.reason


def test_weak_evidence_on_every_attempt_abstains_at_the_end(make_evidence):
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.0, 0.0),
        model,
        rewriter=FakeRewriter(),
        config=SpineConfig(thresholds=LOOP_BOUNDARY),
    )
    result = spine.answer("q1", "who signs")
    assert len(result.attempts) == 2
    assert result.abstained
    assert model.prompts == []


def test_ambiguous_evidence_on_the_last_attempt_is_answered(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.4),
        Recorder(ANSWER),
        config=SpineConfig(max_attempts=1),
    )
    result = spine.answer("q1", "who signs")
    assert result.attempts[0].verdict is Verdict.AMBIGUOUS
    assert result.attempts[0].action is Action.ANSWER
    assert not result.abstained


def test_one_attempt_means_no_corrective_loop(make_evidence):
    retrieve = retriever(make_evidence)
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.0),
        Recorder(ANSWER),
        rewriter=FakeRewriter(),
        config=SpineConfig(max_attempts=1, thresholds=LOOP_BOUNDARY),
    )
    result = spine.answer("q1", "who signs")
    assert len(retrieve.asked) == 1
    assert result.abstained


def test_recommended_single_shot_path_lets_generation_decide(make_evidence):
    """A weak advisory grade cannot pre-empt the generator in RI #1's chosen path."""
    retrieve = retriever(make_evidence)
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.0),
        model,
        config=SpineConfig(max_attempts=1),
        policy=SingleShotPolicy(),
    )

    result = spine.answer("q1", "who signs")

    assert len(retrieve.asked) == 1
    assert len(model.prompts) == 1
    assert result.attempts[0].action is Action.ANSWER
    assert not result.abstained


def test_a_rewriter_that_finds_nothing_settles_rather_than_looping(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.4),
        Recorder(ANSWER),
        rewriter=FakeRewriter(rewritten=""),
    )
    result = spine.answer("q1", "who signs")
    assert len(result.attempts) == 1
    assert result.attempts[0].action is Action.ANSWER


def test_the_pre_retrieval_rewriter_changes_the_first_query(make_evidence):
    retrieve = retriever(make_evidence)
    spine = AdvancedRagSpine(
        retrieve, FakeGrader(0.9), Recorder(ANSWER), prepare=FakeRewriter("stripped")
    )
    result = spine.answer("q1", "who signs")
    assert retrieve.asked[0][0] == "stripped"
    assert result.attempts[0].strategy == "fake-rewriter"


def test_a_prepare_step_that_finds_nothing_keeps_the_original(make_evidence):
    retrieve = retriever(make_evidence)
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.9),
        Recorder(ANSWER),
        prepare=FakeRewriter(rewritten=""),
    )
    result = spine.answer("q1", "who signs")
    assert retrieve.asked[0][0] == "who signs"
    assert result.attempts[0].strategy == "original"


def test_a_three_candidate_plan_retrieves_once_per_candidate(make_evidence):
    retrieve = tagged_retriever(make_evidence)
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.9),
        Recorder(ANSWER),
        config=SpineConfig(max_attempts=1),
        analysis=RewriterAnalysis(ThreeWayRewriter(), max_steps=3),
    )
    result = spine.answer("q1", "who signs")

    assert retrieve.asked == ["who signs v1", "who signs v2", "who signs v3"]
    assert len(result.evidence) == 6
    # One attempt is one trace row, so `merge_steps` names every candidate
    # rather than passing the first one off as the attempt.
    assert result.attempts[0].question == ("who signs v1 | who signs v2 | who signs v3")
    assert result.attempts[0].strategy == "three-way"


def test_the_pooled_grade_is_the_maximum_across_every_candidate(make_evidence):
    # Only the third candidate retrieves a scoring passage, and pooling by
    # maximum is what lets it carry the attempt. This is why `max_steps`
    # defaults to 1: a wider plan can only move a verdict toward SUFFICIENT,
    # which moves the abstention rate every committed figure was measured at.
    scores = {"v3 first": 0.9}

    def spine_at(max_steps):
        return AdvancedRagSpine(
            tagged_retriever(make_evidence),
            PerPassageGrader(scores),
            Recorder(ANSWER),
            config=SpineConfig(max_attempts=1, thresholds=LOOP_BOUNDARY),
            analysis=RewriterAnalysis(ThreeWayRewriter(), max_steps=max_steps),
        )

    wide = spine_at(3).answer("q1", "who signs")
    assert wide.attempts[0].score == pytest.approx(0.9)
    assert wide.attempts[0].verdict is Verdict.SUFFICIENT
    assert wide.abstained is False

    narrow = spine_at(1).answer("q1", "who signs")
    assert narrow.attempts[0].score == pytest.approx(0.0)
    assert narrow.attempts[0].verdict is Verdict.INSUFFICIENT
    assert narrow.abstained is True


def test_a_wider_plan_widens_the_grader_and_leaves_the_reader_alone(make_evidence):
    # Deliberate, and documented on `RewriterAnalysis`. The spine concatenates
    # each candidate's ranking in step-then-rank order and the generation budget
    # takes the head of that list, so the reader sees the first candidate and
    # nothing else while the grade rests on the third. Interleaving the rankings
    # or fusing them by score is a retrieval-side decision with its own
    # measurement and is not a seam here, so the order is pinned rather than
    # fixed. A later change that fuses has to move this test on purpose.
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(
        tagged_retriever(make_evidence),
        PerPassageGrader({"v3 first": 0.9}),
        model,
        config=SpineConfig(max_attempts=1, evidence_budget=2, thresholds=LOOP_BOUNDARY),
        analysis=RewriterAnalysis(ThreeWayRewriter(), max_steps=3),
    )
    result = spine.answer("q1", "who signs")

    assert [item.text for item in result.evidence] == [
        "v1 first",
        "v1 second",
        "v2 first",
        "v2 second",
        "v3 first",
        "v3 second",
    ]
    prompt = model.prompts[0]
    assert "v1 first" in prompt
    assert "v1 second" in prompt
    assert "v2 first" not in prompt
    assert "v3 first" not in prompt


def test_the_corrective_retry_is_no_narrower_than_the_first_attempt(make_evidence):
    # The retry used to keep one candidate whatever the first attempt planned,
    # so the pass that needed the most evidence retrieved the least.
    retrieve = tagged_retriever(make_evidence)
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.1, 0.9),
        Recorder(ANSWER),
        rewriter=ThreeWayRewriter(),
        config=SpineConfig(max_attempts=2, thresholds=LOOP_BOUNDARY),
        analysis=RewriterAnalysis(ThreeWayRewriter(), max_steps=3),
    )
    result = spine.answer("q1", "who signs")

    assert len(result.attempts) == 2
    assert len(retrieve.asked) == 6
    assert result.attempts[1].question == ("who signs v1 | who signs v2 | who signs v3")


def test_an_attempt_records_what_the_rewriter_did(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.9),
        Recorder(ANSWER),
        prepare=FakeRewriter(rewritten=""),
    )
    attempt = spine.answer("q1", "who signs").attempts[0]

    # The question as asked was retrieved either way, so the outcome is the only
    # record that a rewriter ran at all and found nothing to change.
    assert attempt.question == "who signs"
    assert attempt.rewrite_outcome == "no-change"
    assert attempt.to_dict()["rewrite_outcome"] == "no-change"


def test_an_attempt_under_no_rewriter_names_no_outcome(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence), FakeGrader(0.9), Recorder(ANSWER)
    )
    assert spine.answer("q1", "who signs").attempts[0].rewrite_outcome == ""


def test_the_generator_is_asked_the_original_question_not_the_rewrite(
    make_evidence,
):
    model = Recorder(ANSWER)
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.0, 0.9),
        model,
        rewriter=FakeRewriter(),
    )
    spine.answer("q1", "who signs the agreement")
    assert "who signs the agreement" in model.prompts[0]
    assert "rewritten question" not in model.prompts[0]


def test_top_k_bounds_the_grade_and_the_budget_bounds_the_prompt(make_evidence):
    model = Recorder(ANSWER)
    retrieve = retriever(make_evidence, texts=["one", "two", "three", "four"])
    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.9),
        model,
        config=SpineConfig(top_k=3, evidence_budget=1),
    )
    result = spine.answer("q1", "who signs")
    assert len(result.evidence) == 3
    assert "two" not in model.prompts[0]


def test_a_configured_depth_above_the_default_reaches_the_retriever(make_evidence):
    # A depth above the retriever's own default of 10 used to be capped at 10:
    # the spine retrieved at the default and truncated afterwards, and a slice
    # can only shrink a ranking.
    depths: list[int] = []

    def retrieve(question, scope=None, k=10):
        depths.append(k)
        return make_evidence([f"passage {index}" for index in range(k)])

    spine = AdvancedRagSpine(
        retrieve,
        FakeGrader(0.9),
        Recorder(ANSWER),
        config=SpineConfig(top_k=20, evidence_budget=2),
    )
    result = spine.answer("q1", "who signs")
    assert depths == [20]
    assert len(result.evidence) == 20


def test_small_to_big_forwards_the_depth_it_was_called_with():
    text = "clause one. " * 200
    hierarchy = split_document(
        "doc1", text, parent_chars=800, child_chars=200, overlap_chars=0
    )
    depths: list[int] = []

    def retrieve(question, scope=None, k=10):
        depths.append(k)
        return to_evidence(list(hierarchy.children[:2]))

    with_small_to_big(retrieve, hierarchy)("who signs", None, 4)
    assert depths == [4]


def test_the_scope_reaches_the_retriever(make_evidence):
    retrieve = retriever(make_evidence)
    spine = AdvancedRagSpine(retrieve, FakeGrader(0.9), Recorder(ANSWER))
    spine.answer("q1", "who signs", scope="contract-7")
    assert retrieve.asked[0][1] == "contract-7"


def test_run_spine_reads_the_corpus_query_shape(make_evidence):
    retrieve = retriever(make_evidence)
    spine = AdvancedRagSpine(retrieve, FakeGrader(0.9), Recorder(ANSWER))
    queries = [
        SimpleNamespace(query_id="q1", question="who signs", contract_id="c1"),
        SimpleNamespace(query_id="q2", question="what governs"),
    ]
    results = run_spine(spine, queries)
    assert [asked[1] for asked in retrieve.asked] == ["c1", None]
    assert [result.query_id for result in results] == ["q1", "q2"]


def test_the_run_feeds_both_scorers(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.9, 0.0),
        Recorder(ANSWER),
        config=SpineConfig(thresholds=LOOP_BOUNDARY),
    )
    results = [
        spine.answer("q1", "who signs"),
        spine.answer("q2", "unanswerable"),
    ]
    assert abstentions(results) == {"q1": False, "q2": True}
    assert retrieval_runs(results)["q1"] == ["doc1::000", "doc1::001"]


def test_a_result_serialises_with_its_trace(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence), FakeGrader(0.9), Recorder(ANSWER)
    )
    payload = json.loads(json.dumps(spine.answer("q1", "who signs").to_dict()))
    assert payload["attempts"][0]["action"] == "answer"
    assert payload["attempts"][0]["verdict"] == "sufficient"
    assert payload["retrieved_ids"] == ["doc1::000", "doc1::001"]


def test_small_to_big_composes_at_the_retrieval_boundary():
    text = "clause one. " * 200
    hierarchy = split_document(
        "doc1", text, parent_chars=800, child_chars=200, overlap_chars=0
    )

    def retrieve(question, scope=None, k=10):
        return to_evidence(list(hierarchy.children[:2]))

    expanded = with_small_to_big(retrieve, hierarchy)("who signs")
    assert expanded[0].text == hierarchy.parents[0].text
    assert expanded[0].chunk_id == hierarchy.children[0].node_id


def test_hits_are_joined_to_their_chunks_and_unknown_hits_are_dropped():
    chunks = {
        "c::000": SimpleNamespace(
            chunk_id="c::000", doc_id="c", text="passage", start=0, end=7
        )
    }
    hits = [
        SimpleNamespace(chunk_id="c::000", score=0.9),
        SimpleNamespace(chunk_id="c::999", score=0.1),
    ]
    evidence = evidence_from_hits(hits, chunks, retriever="dense")
    assert len(evidence) == 1
    assert evidence[0].text == "passage"
    assert evidence[0].provenance.doc_id == "c"
    assert evidence[0].retriever == "dense"


def test_a_config_that_would_answer_from_nothing_is_rejected():
    for bad in ({"top_k": 0}, {"evidence_budget": 0}, {"max_attempts": 0}):
        with pytest.raises(ValueError):
            SpineConfig(**bad)


def test_a_pair_that_says_the_same_thing_twice_is_rejected(make_evidence):
    retrieve = retriever(make_evidence)
    with pytest.raises(ValueError, match="prepare or analysis"):
        AdvancedRagSpine(
            retrieve,
            FakeGrader(0.9),
            Recorder(ANSWER),
            prepare=FakeRewriter("stripped"),
            analysis=PassthroughAnalysis(),
        )
    with pytest.raises(ValueError, match="template or prompt"):
        AdvancedRagSpine(
            retrieve,
            FakeGrader(0.9),
            Recorder(ANSWER),
            template="TEMPLATE-A {query} {sources}",
            prompt=FixedPromptSelector("TEMPLATE-B {query} {sources}"),
        )


def test_the_prompt_is_the_only_place_the_template_is_reported(make_evidence):
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.9),
        Recorder(ANSWER),
        prompt=FixedPromptSelector("TEMPLATE-B {query} {sources}"),
    )
    # `spine.template` used to report the losing half of the pair.
    assert not hasattr(spine, "template")
    assert spine.prompt.get_template(QueryPlan(question="q"), ()) == (
        "TEMPLATE-B {query} {sources}"
    )


def test_thresholds_on_the_config_and_a_policy_of_your_own_is_rejected(make_evidence):
    with pytest.raises(ValueError, match="config.thresholds or policy"):
        AdvancedRagSpine(
            retriever(make_evidence),
            FakeGrader(0.9),
            Recorder(ANSWER),
            config=SpineConfig(thresholds=LOOP_BOUNDARY),
            policy=CorrectivePolicy(GradingThresholds(0.1, 0.05)),
        )
    # A policy beside a configuration that states no thresholds is the
    # documented way to bring your own, and it still works.
    spine = AdvancedRagSpine(
        retriever(make_evidence),
        FakeGrader(0.9),
        Recorder(ANSWER),
        config=SpineConfig(max_attempts=1),
        policy=CorrectivePolicy(GradingThresholds(0.1, 0.05)),
    )
    assert spine.answer("q1", "who signs").abstained is False
