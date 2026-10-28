"""The multi-hop and fan-out generators.

The filter tests matter most: they stop a generator from certifying its own
output, so a bug there would produce a set that is not multi-hop.
"""

from __future__ import annotations

import json

import pytest

from aieng.advanced_rag.evaluation.synthetic import (
    build_fanout_questions,
    build_multi_hop_questions,
)
from aieng.advanced_rag.evaluation.synthetic.agent import (
    Proposal,
    Rejection,
    assigned_type,
    locate_evidence,
    run_agent,
)
from aieng.advanced_rag.evaluation.synthetic.fanout import FanoutProposal, coverage
from aieng.advanced_rag.evaluation.synthetic.filters import (
    ALL_BUT_ONE_SYSTEM,
    NULL_SYSTEM,
    SUFFICIENCY_SYSTEM,
    VERIFY_SYSTEM,
    multi_hop_filters,
)
from aieng.advanced_rag.evaluation.synthetic.probe import ProbeIndex
from aieng.advanced_rag.evaluation.synthetic.rewrite import (
    accept,
    check_rewrite,
    leaked_terms,
)
from aieng.advanced_rag.evaluation.synthetic.sampling import (
    CandidatePair,
    Condition,
    Document,
    band_report,
    cosine,
    document_frequencies,
    embed_documents,
    select_pairs,
    shared_rare_terms,
    tag_conditions,
)
from aieng.advanced_rag.evaluation.synthetic.taxonomy import (
    ALL_TYPES,
    BY_NAME,
    MULTI_HOP_MIX,
    MULTI_HOP_TYPES,
    NULL_TYPES,
    REPORT_LABELS,
    SINGLE_HOP,
    capability_coverage,
    composability_holds,
)

from .conftest import model


VOCAB = ("nginx", "postgres", "timeout", "tuning", "replication", "cron")


def encode(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] for t in texts]


@pytest.fixture
def documents():
    return [
        Document("d1", "nginx timeout upstream gateway timeout"),
        Document("d2", "nginx tuning worker connections timeout"),
        Document("d3", "postgres replication streaming replication"),
        Document("d4", "postgres replication slots tuning"),
        Document("d5", "cron scheduling"),
    ]


class TestTaxonomy:
    def test_the_mix_is_a_distribution_of_known_types(self):
        assert sum(MULTI_HOP_MIX.values()) == pytest.approx(1.0)
        assert set(MULTI_HOP_MIX) <= set(BY_NAME)

    def test_multi_hop_types_need_at_least_two_documents(self):
        assert all(t.min_documents >= 2 for t in MULTI_HOP_TYPES)
        assert SINGLE_HOP.min_documents == 1

    def test_null_types_are_unanswerable_and_labelled_so(self):
        assert all(
            not t.answerable and t.report_label == "unanswerable" for t in NULL_TYPES
        )

    def test_every_type_has_a_capability_and_a_report_label(self):
        assert all(t.capability and t.report_label in REPORT_LABELS for t in ALL_TYPES)

    def test_coverage_counts_by_capability_and_ignores_unknown_names(self):
        assert (
            capability_coverage(["bridge", "temporal"])["path-finding multi-hop"] == 2
        )
        assert capability_coverage(["not_a_type"]) == {}

    def test_composability(self):
        assert not composability_holds(BY_NAME["bridge"], ["d1", "d1"])
        assert composability_holds(BY_NAME["bridge"], ["d1", "d2"])
        assert composability_holds(BY_NAME["unanswerable"], [])


class TestPairSelection:
    def test_the_band_excludes_pairs_outside_it(self, documents):
        vectors = embed_documents(documents, encode)
        pairs = select_pairs(documents, vectors, band=(0.9, 1.0), limit=10)
        assert all(0.9 <= pair.similarity <= 1.0 for pair in pairs)

    def test_related_documents_are_paired_over_unrelated_ones(self, documents):
        vectors = embed_documents(documents, encode)
        keys = {
            p.key for p in select_pairs(documents, vectors, band=(0.5, 0.999), limit=10)
        }
        assert ("d3", "d4") in keys
        assert ("d3", "d5") not in keys

    def test_near_duplicates_sit_above_the_band(self, documents):
        documents = [
            *documents,
            Document("d6", "postgres replication streaming replication"),
        ]
        vectors = embed_documents(documents, encode)
        keys = {
            p.key for p in select_pairs(documents, vectors, band=(0.5, 0.95), limit=20)
        }
        assert ("d3", "d6") not in keys

    def test_max_per_document_caps_a_hub(self, documents):
        vectors = embed_documents(documents, encode)
        pairs = select_pairs(
            documents, vectors, band=(0.0, 1.0), limit=50, max_per_document=1
        )
        appearances: dict[str, int] = {}
        for pair in pairs:
            for doc in (pair.doc_a, pair.doc_b):
                appearances[doc] = appearances.get(doc, 0) + 1
        assert max(appearances.values()) <= 1

    def test_selection_is_reproducible_and_limited(self, documents):
        vectors = embed_documents(documents, encode)
        first = select_pairs(documents, vectors, band=(0.0, 1.0), limit=2, seed=7)
        second = select_pairs(documents, vectors, band=(0.0, 1.0), limit=2, seed=7)
        assert [p.key for p in first] == [p.key for p in second]
        assert len(first) == 2

    def test_an_invalid_band_is_rejected(self, documents):
        with pytest.raises(ValueError, match="low <= high"):
            select_pairs(documents, embed_documents(documents, encode), band=(0.9, 0.1))

    def test_band_report_counts_every_pair_once(self, documents):
        vectors = embed_documents(documents, encode)
        report = band_report(documents, vectors, [(0.0, 0.5), (0.5, 1.0)])
        assert sum(report.values()) == len(documents) * (len(documents) - 1) // 2

    def test_pair_key_and_cosine(self):
        assert CandidatePair("b", "a", 0.5).key == CandidatePair("a", "b", 0.5).key
        assert cosine([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)

    def test_shared_rare_terms_drop_common_ones(self):
        shared = shared_rare_terms(
            "server vacuum issue",
            "server vacuum tuning",
            {"server": 100, "vacuum": 2},
            100,
        )
        assert "vacuum" in shared and "server" not in shared
        docs = [Document("a", "nginx nginx nginx"), Document("b", "nginx postgres")]
        assert document_frequencies(docs)["nginx"] == 2


class TestAgentLoop:
    pair = CandidatePair("d1", "d2", 0.8, ("timeout",))

    def run(self, replies, **kwargs):
        stream = iter(replies) if isinstance(replies, list) else None
        reply = (lambda s, p: next(stream)) if stream else (lambda s, p: replies)
        return run_agent(
            model("g", reply),
            self.pair,
            BY_NAME["bridge"],
            read_document=kwargs.pop("read", lambda d: "text"),
            search_corpus=kwargs.pop("search", lambda q, k: []),
            **kwargs,
        )

    def test_the_agent_can_reject(self):
        outcome = self.run('{"action":"reject","reason":"topical overlap only"}')
        assert isinstance(outcome, Rejection)
        assert "topical" in outcome.reason

    def test_the_agent_reads_then_proposes_with_quotes(self):
        reads: list[str] = []
        outcome = self.run(
            [
                '{"action":"read","doc_id":"d1"}',
                json.dumps(
                    {
                        "action": "propose",
                        "question": "q?",
                        "answer": "a",
                        "needs_documents": ["d1", "d2"],
                        "bridge": "shared timeout",
                        "evidence": [
                            {"doc_id": "d1", "quote": "upstream gateway"},
                            {"doc_id": "d2", "quote": "worker connections"},
                        ],
                    }
                ),
            ],
            read=lambda d: reads.append(d) or "body",
        )
        assert isinstance(outcome, Proposal)
        assert reads == ["d1"] and outcome.tool_calls == ["read:d1"]
        texts = {
            "d1": "nginx timeout upstream gateway timeout",
            "d2": "nginx tuning worker connections timeout",
        }
        spans, missing = locate_evidence(outcome, texts)
        assert not missing
        assert [texts[s.doc_id][s.start : s.end] for s in spans] == [
            "upstream gateway",
            "worker connections",
        ]

    def test_running_out_of_steps_and_garbage_are_rejections(self):
        assert isinstance(
            self.run('{"action":"search","query":"x"}', max_steps=3), Rejection
        )
        assert isinstance(self.run("I refuse to answer", max_steps=2), Rejection)

    def test_a_proposal_without_a_question_is_not_accepted(self):
        outcome = self.run(
            [
                '{"action":"propose","answer":"a"}',
                '{"action":"reject","reason":"gave up"}',
            ]
        )
        assert isinstance(outcome, Rejection)

    def test_the_type_comes_from_the_documents_needed(self):
        one = Proposal("q", "a", ["d1"], "", "", "bridge", ("d1", "d2"), 1)
        assert assigned_type(one, BY_NAME["bridge"]).name == "single_hop"
        three = Proposal(
            "q", "a", ["d1", "d2"], "", "", "bridge_comparison", ("d1", "d2"), 1
        )
        assert assigned_type(three, BY_NAME["bridge_comparison"]).name == "bridge"


def checker(*, multi_hop=True, natural=True, answerable=True, sufficient=False):
    def complete(system, prompt):
        if system == SUFFICIENCY_SYSTEM:
            return json.dumps({"sufficient": sufficient})
        if system == ALL_BUT_ONE_SYSTEM:
            return json.dumps({"sufficient": sufficient})
        if system == NULL_SYSTEM:
            return json.dumps({"answered": "no"})
        assert system == VERIFY_SYSTEM
        return json.dumps(
            {
                "is_truly_multi_hop": multi_hop,
                "is_natural": natural,
                "is_answerable": answerable,
            }
        )

    return model("checker", complete)


DOCS = {"d1": "text one", "d2": "text two"}


class TestMultiHopFilters:
    def call(self, stub, gold=("d1", "d2"), **kwargs):
        return multi_hop_filters(
            stub,
            "q?",
            answer="a",
            bridge="fact",
            gold=list(gold),
            documents=DOCS,
            **kwargs,
        )

    def test_a_good_proposal_is_accepted(self):
        assert self.call(checker()).accepted

    def test_the_defeat_test_overrides_a_positive_verifier(self):
        result = self.call(checker(sufficient=True))
        assert (
            not result.accepted
            and result.sufficient_documents
            and "alone" in result.reason
        )

    def test_verifier_rejections_are_honoured(self):
        assert "not multi-hop" in self.call(checker(multi_hop=False)).reason
        assert "artefact" in self.call(checker(natural=False)).reason
        assert self.call(checker(natural=False), require_natural=False).accepted

    def test_unknown_or_too_few_documents_are_rejected(self):
        assert "requires 2" in self.call(checker(), gold=("d1",)).reason
        assert not self.call(checker(), gold=("d1", "ghost")).accepted


class TestRewrite:
    df = {"pg_hba.conf": 1, "postgres": 40, "replication": 3}

    def test_leaked_terms_are_rare_and_from_the_second_document_only(self):
        leaked = leaked_terms(
            "Why does replication fail after editing pg_hba.conf in postgres?",
            "postgres replication",
            "edit pg_hba.conf and reload postgres",
            self.df,
            100,
        )
        assert leaked == ("pg_hba.conf",)

    def test_a_rewrite_that_keeps_the_leak_is_refused(self):
        checks = check_rewrite(
            "What about pg_hba.conf?",
            "a",
            "",
            "edit pg_hba.conf",
            document_frequency=self.df,
            total_documents=100,
        )
        assert not accept(checks)

    def test_a_rewrite_that_collapses_to_one_document_is_refused(self):
        checks = check_rewrite(
            "What file controls access?",
            "a",
            "",
            "edit pg_hba.conf",
            document_frequency=self.df,
            total_documents=100,
            still_needs_both=lambda q: True,
        )
        assert checks["leaks_removed"] and not accept(checks)


def tagged_corpus() -> list[Document]:
    return [
        Document(
            f"t{i}",
            f"Policy {i}. Remote staff in region {i} get {i + 2} days of leave.",
            tags=("leave",),
        )
        for i in range(4)
    ] + [Document("u", "Unrelated text about parking.", tags=("parking",))]


class TestFanout:
    def test_conditions_come_from_tags_within_the_size_range(self):
        conditions = tag_conditions(tagged_corpus(), min_documents=3, max_documents=8)
        assert [c.value for c in conditions] == ["leave"]
        with pytest.raises(ValueError, match="three"):
            tag_conditions(tagged_corpus(), min_documents=2)

    def test_coverage_counts_distinct_contributions(self):
        condition = Condition("tag", "leave", ("t0", "t1", "t2", "t3"))
        proposal = FanoutProposal("q", "a", condition, {"t0": "x", "t1": "y", "t2": ""})
        assert coverage(proposal) == 0.5

    def test_a_fanout_question_is_built_with_spans(self):
        documents = tagged_corpus()
        condition = tag_conditions(documents)[0]
        reply = json.dumps(
            {
                "question": "How many days of leave do remote staff get in each region?",
                "answer": "2, 3, 4, and 5 days",
                "per_document": {
                    f"t{i}": f"get {i + 2} days of leave" for i in range(4)
                },
            }
        )
        accepted, funnel = build_fanout_questions(
            [condition],
            documents,
            generator=model("g", lambda s, p: reply),
            checker=checker(),
        )
        (question,) = accepted
        assert question.kind == "fan_out" and len(question.documents) == 4
        assert funnel.to_dict()["accepted"] == 1

    def test_a_fanout_question_answerable_without_a_document_is_filtered(self):
        documents = tagged_corpus()
        reply = json.dumps(
            {
                "question": "q?",
                "answer": "a",
                "per_document": {f"t{i}": f"get {i + 2} days" for i in range(4)},
            }
        )
        stub = model(
            "c", lambda s, p: json.dumps({"sufficient": s == ALL_BUT_ONE_SYSTEM})
        )
        accepted, funnel = build_fanout_questions(
            tag_conditions(documents),
            documents,
            generator=model("g", lambda s, p: reply),
            checker=stub,
        )
        assert accepted == []
        assert any("all_but_one" in r for r in funnel.to_dict()["drop_reasons"])


class TestMultiHopBuild:
    def proposal(self, a, b):
        return json.dumps(
            {
                "action": "propose",
                "question": "Which region's leave policy covers staff hired under policy 0?",
                "answer": "region 1",
                "needs_documents": [a, b],
                "bridge": "remote staff",
                "evidence": [
                    {"doc_id": a, "quote": "Remote staff"},
                    {"doc_id": b, "quote": "days of leave"},
                ],
            }
        )

    def test_a_pair_becomes_a_multi_hop_question_with_spans(self):
        documents = tagged_corpus()
        accepted, funnel = build_multi_hop_questions(
            [CandidatePair("t0", "t1", 0.8)],
            documents,
            generator=model("g", lambda s, p: self.proposal("t0", "t1")),
            checker=checker(),
            search_corpus=lambda q, k: [],
        )
        (question,) = accepted
        assert question.kind == "bridge" and question.documents == ("t0", "t1")
        assert funnel.to_dict()["accepted"] == 1

    def test_a_pair_with_a_held_out_document_becomes_a_partial_null(self, embedder):
        documents = tagged_corpus()
        probe = ProbeIndex([d for d in documents if d.doc_id != "t1"], embedder)
        accepted, _ = build_multi_hop_questions(
            [CandidatePair("t0", "t1", 0.8)],
            documents,
            generator=model("g", lambda s, p: self.proposal("t0", "t1")),
            checker=checker(),
            search_corpus=lambda q, k: [],
            held_out=["t1"],
            probe=probe,
        )
        (question,) = accepted
        assert question.kind == "partial_null" and not question.is_answerable
