"""What kind of question this is, and what it is meant to discriminate.

A question set that does not label which capability a question exercises can
say a method is worse without saying at what. Each type below maps onto one
capability, so a result reads as "weaker at entity aggregation" rather than
"scored lower".

Two coarse splits sit above the fine types. ``coarse`` is the three-bucket
split earlier graph-seeded generators used (``multi_hop_specific``,
``multi_hop_abstract``, ``single_hop``). ``report_label`` is the four-way label
set from Know Your RAG (de Lima et al., COLING 2025 Industry): ``fact_single``,
``summary``, ``reasoning``, and ``unanswerable``, which is the axis to report
team corpora on. It covers multi-hop, single-hop, and null types.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Know Your RAG's coarse labels, the report axis for team corpora.
REPORT_LABELS: tuple[str, ...] = ("fact_single", "summary", "reasoning", "unanswerable")


@dataclass(frozen=True)
class QuestionType:
    """One question shape and the retrieval capability it probes.

    Parameters
    ----------
    name : str
        Stable identifier used in the generated set.
    coarse : str
        ``multi_hop_specific``, ``multi_hop_abstract``, or ``single_hop``.
    capability : str
        The capability gap this question is meant to discriminate.
    description : str
        What the question must require of the reader.
    prompt_guidance : str
        Instruction handed to the generator.
    min_documents : int, optional
        Documents that must be needed to answer. Used by the filters.
    answerable : bool, optional
        ``False`` marks the null types, scored on abstention.
    source : str, optional
        The benchmark that established the shape.
    report_label : str, optional
        One of :data:`REPORT_LABELS`.
    """

    name: str
    coarse: str
    capability: str
    description: str
    prompt_guidance: str
    min_documents: int = 2
    answerable: bool = True
    source: str = ""
    report_label: str = "reasoning"


BRIDGE = QuestionType(
    name="bridge",
    coarse="multi_hop_specific",
    capability="path-finding multi-hop",
    description=(
        "A shared entity links two documents. The answer lives in the second, "
        "but the first is what identifies which second document to read."
    ),
    prompt_guidance=(
        "Write a question whose answer is in document B, but which cannot be "
        "asked or resolved without a fact from document A. Do not name the "
        "bridging entity in the question; make the reader find it in A first."
    ),
    source="HotpotQA bridge; MuSiQue 2-hop composition",
)

COMPARISON = QuestionType(
    name="comparison",
    coarse="multi_hop_abstract",
    capability="cross-document synthesis",
    description=(
        "Both documents describe the same kind of thing, and the answer is a "
        "difference, a preference, or a trade-off between them."
    ),
    prompt_guidance=(
        "Write a question that requires reading both documents and comparing "
        "them. The answer must be a judgement no single document states."
    ),
    source="HotpotQA comparison; MultiHop-RAG comparison query",
)

CAUSAL_CHAIN = QuestionType(
    name="causal_chain",
    coarse="multi_hop_specific",
    capability="explicit reasoning paths",
    description=(
        "One document establishes a cause or dependency, the other establishes "
        "its consequence or fix. The answer is the chain."
    ),
    prompt_guidance=(
        "Write a question whose answer is a causal or procedural chain running "
        "through both documents. The reader must connect a cause in one to an "
        "effect or remedy in the other."
    ),
    source="MultiHop-RAG inference query",
)

AGGREGATION = QuestionType(
    name="aggregation",
    coarse="multi_hop_abstract",
    capability="entity-centric aggregation",
    description=(
        "One entity appears in several documents and the answer collects what "
        "is known about it across all of them."
    ),
    prompt_guidance=(
        "Write a question about one specific entity that requires gathering "
        "facts stated in different documents. A single document must give an "
        "incomplete answer."
    ),
    source="FanOutQA set collection at small arity",
    report_label="summary",
)

TEMPORAL = QuestionType(
    name="temporal",
    coarse="multi_hop_specific",
    capability="path-finding multi-hop",
    description=(
        "The answer depends on ordering or on a change between documents, such "
        "as a superseded approach and its replacement."
    ),
    prompt_guidance=(
        "Write a question whose answer depends on sequence or change over time "
        "across the two documents, such as which approach supersedes which."
    ),
    source="MultiHop-RAG temporal query; 2WikiMultiHopQA inference over dates",
)

INFERENCE = QuestionType(
    name="inference",
    coarse="multi_hop_specific",
    capability="explicit reasoning paths",
    description=(
        "Two documents each state one relation, and the answer is a third "
        "relation that follows from composing them and is written down nowhere."
    ),
    prompt_guidance=(
        "Find a relation stated in document A and a relation stated in document "
        "B that share a term. Write a question whose answer is the composition "
        "of the two. The answer must not appear as a sentence in either "
        "document; it must have to be derived."
    ),
    source="2WikiMultiHopQA compositional and inference types",
)

BRIDGE_COMPARISON = QuestionType(
    name="bridge_comparison",
    coarse="multi_hop_abstract",
    capability="cross-document synthesis",
    description=(
        "Two bridges then a comparison: each of two entities is resolved "
        "through its own document, and the answer compares the two results."
    ),
    prompt_guidance=(
        "Identify two entities that each require a lookup in a different "
        "document, then write a question comparing a property of the two. The "
        "reader must do both lookups before the comparison is possible."
    ),
    min_documents=3,
    source="2WikiMultiHopQA bridge-comparison",
)

FAN_OUT = QuestionType(
    name="fan_out",
    coarse="multi_hop_abstract",
    capability="corpus-scale summarisation",
    description=(
        "The answer is a set whose size is not known in advance, so the reader "
        "must find every document satisfying a condition. This is the shape a "
        "chunk index handles worst."
    ),
    prompt_guidance=(
        "Write a question whose answer enumerates or counts every item in the "
        "corpus meeting a stated condition. Do not state how many there are."
    ),
    min_documents=3,
    source="FanOutQA; Microsoft GraphRAG global search",
    report_label="summary",
)

SINGLE_HOP = QuestionType(
    name="single_hop",
    coarse="single_hop",
    capability="calibration baseline",
    description=(
        "Answerable from one passage. If a system cannot answer these, "
        "retrieval is broken and no multi-hop result from it means anything."
    ),
    prompt_guidance="Write a direct question fully answerable from the statement.",
    min_documents=1,
    source="Know Your RAG statement-first generation",
    report_label="fact_single",
)

UNANSWERABLE = QuestionType(
    name="unanswerable",
    coarse="multi_hop_specific",
    capability="calibration baseline",
    description=(
        "Shaped like a bridge question, but the second hop is absent from the "
        "corpus, so the correct behaviour is to say so."
    ),
    prompt_guidance=(
        "Write a question that looks answerable from document A and requires a "
        "second fact that the corpus does not contain."
    ),
    min_documents=1,
    answerable=False,
    source="MuSiQue unanswerable variant; MultiHop-RAG null query",
    report_label="unanswerable",
)

HELD_OUT_NULL = QuestionType(
    name="held_out_null",
    coarse="single_hop",
    capability="abstention",
    description=(
        "Written from a document kept out of the index, in the corpus's own "
        "vocabulary, and checked to be unanswerable from what is indexed."
    ),
    prompt_guidance="Write a direct question fully answerable from the statement.",
    min_documents=1,
    answerable=False,
    source="CRUMQs corpus-level null check, with held-out documents in place of crawled ones",
    report_label="unanswerable",
)

NEAR_MISS_NULL = QuestionType(
    name="near_miss_null",
    coarse="single_hop",
    capability="abstention",
    description=(
        "An answerable question with the one detail that selects its evidence "
        "changed, so the document no longer answers it."
    ),
    prompt_guidance=(
        "Change the one detail that makes the statement answer the question, so "
        "that the document does not answer it."
    ),
    min_documents=1,
    answerable=False,
    source="ContractNLI NotMentioned labels",
    report_label="unanswerable",
)

PARTIAL_NULL = QuestionType(
    name="partial_null",
    coarse="multi_hop_specific",
    capability="abstention",
    description=(
        "A multi-hop question whose second hop is in a held-out document, so a "
        "system finds half its evidence and must still abstain."
    ),
    prompt_guidance=BRIDGE.prompt_guidance,
    min_documents=1,
    answerable=False,
    source="CRUMQs partially unanswerable questions",
    report_label="unanswerable",
)

MULTI_HOP_TYPES: tuple[QuestionType, ...] = (
    BRIDGE,
    COMPARISON,
    CAUSAL_CHAIN,
    AGGREGATION,
    TEMPORAL,
    INFERENCE,
    BRIDGE_COMPARISON,
    FAN_OUT,
)

NULL_TYPES: tuple[QuestionType, ...] = (
    UNANSWERABLE,
    HELD_OUT_NULL,
    NEAR_MISS_NULL,
    PARTIAL_NULL,
)

ALL_TYPES: tuple[QuestionType, ...] = (*MULTI_HOP_TYPES, SINGLE_HOP, *NULL_TYPES)

BY_NAME: dict[str, QuestionType] = {t.name: t for t in ALL_TYPES}

#: The experiment's multi-hop mix: roughly 60% specific, 30% abstract, and 10%
#: calibration, spread across the fine types.
MULTI_HOP_MIX: dict[str, float] = {
    "bridge": 0.22,
    "causal_chain": 0.18,
    "comparison": 0.12,
    "inference": 0.12,
    "aggregation": 0.08,
    "bridge_comparison": 0.06,
    "fan_out": 0.06,
    "temporal": 0.04,
    "single_hop": 0.06,
    "unanswerable": 0.06,
}

#: Rewrite levels, orthogonal to type. A question written while looking at its
#: evidence carries the evidence's surface terms, which hands the retriever the
#: answer; MuSiQue obfuscates for this reason.
OBFUSCATION_LEVELS: tuple[str, ...] = ("none", "paraphrase", "indirect")

OBFUSCATION_DESCRIPTIONS: dict[str, str] = {
    "none": "The question may use the wording of the evidence.",
    "paraphrase": (
        "Terms taken from the evidence are reworded, keeping the same meaning. "
        "Defeats exact term matching, not embeddings."
    ),
    "indirect": (
        "The entity that identifies the second document is not named at all. It "
        "is referred to only by a property established in the first document."
    ),
}


def composability_holds(question_type: QuestionType, gold_documents: list[str]) -> bool:
    """Check MuSiQue's composability constraint.

    A multi-hop question is multi-hop only if it needs at least as many
    documents as its type requires. Null types always pass.
    """
    if not question_type.answerable:
        return True
    return len(set(gold_documents)) >= question_type.min_documents


def type_for_documents(requested: QuestionType, needed: int) -> QuestionType:
    """Return the type a proposal actually is, given how many documents it needs.

    The experiment memo found the bridge agent sometimes proposes a question
    that one document answers while the type it was asked for needs two. The
    type is assigned from what the proposal needs, not from the request.
    """
    if needed <= 1 and requested.answerable:
        return SINGLE_HOP
    if needed < requested.min_documents:
        return BRIDGE
    return requested


def capability_coverage(types: list[str]) -> dict[str, int]:
    """Count generated questions per capability, to report beside any result."""
    counts: dict[str, int] = {}
    for name in types:
        question_type = BY_NAME.get(name)
        if question_type is not None:
            counts[question_type.capability] = (
                counts.get(question_type.capability, 0) + 1
            )
    return counts


__all__ = [
    "ALL_TYPES",
    "BY_NAME",
    "HELD_OUT_NULL",
    "MULTI_HOP_MIX",
    "MULTI_HOP_TYPES",
    "NEAR_MISS_NULL",
    "NULL_TYPES",
    "OBFUSCATION_DESCRIPTIONS",
    "OBFUSCATION_LEVELS",
    "PARTIAL_NULL",
    "REPORT_LABELS",
    "SINGLE_HOP",
    "UNANSWERABLE",
    "QuestionType",
    "capability_coverage",
    "composability_holds",
    "type_for_documents",
]
