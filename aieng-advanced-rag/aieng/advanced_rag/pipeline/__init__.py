"""The Advanced RAG pipeline stages between a question and a grounded answer.

The retrieval half of RI #1 is measured elsewhere: dense and lexical arms,
weighted rank fusion, cross-encoder reranking, and the chunking sweep all live
under ``aieng.advanced_rag.retrieval`` with the metrics that scored them. This
package is the other half, the stages a retrieval benchmark never exercises.

* :mod:`~aieng.advanced_rag.pipeline.rewrite` changes the question before it
  reaches the retriever.
* :mod:`~aieng.advanced_rag.pipeline.hierarchical` splits documents so the unit
  that gets ranked is smaller than the unit that gets read.
* :mod:`~aieng.advanced_rag.pipeline.grading` scores what came back and turns
  that score into an action.
* :mod:`~aieng.advanced_rag.pipeline.generate` answers from evidence with
  citations, or abstains and says why.
* :mod:`~aieng.advanced_rag.pipeline.spine` composes them into one loop.
* :mod:`~aieng.advanced_rag.pipeline.answering` holds the three decisions that
  loop delegates: what a question turns into before retrieval, what an attempt's
  outcome means, and which template the answer is rendered from. They are
  exported here, because they are how ``AdvancedRagSpine`` is configured.

Every stage takes its model access as a ``complete`` callable and its retrieval
as a ``retrieve`` callable, so each one runs and is tested without a client, an
index, or a network. Passages move between stages as
:class:`~aieng.advanced_rag.graphrag.evidence.Evidence`, the shared boundary
type, and citations are
:class:`~aieng.advanced_rag.provenance.Provenance` records.
"""

from .answering import (
    Answerer,
    CorrectivePolicy,
    DecompositionAnalysis,
    FixedPromptSelector,
    LoopDecision,
    LoopPolicy,
    LoopState,
    PassthroughAnalysis,
    PromptSelector,
    QueryAnalysis,
    QueryPlan,
    RetrievalStep,
    RetrieverPromptSelector,
    RewriterAnalysis,
    SingleShotPolicy,
    merge_steps,
    step_bound,
)
from .generate import (
    ABSTENTION_TEXT,
    GENERATION_TEMPLATE,
    ChatCompleter,
    GroundedAnswer,
    abstention_flags,
    generate,
    looks_like_abstention,
    parse_answer,
)
from .grading import (
    Action,
    ChunkGrade,
    EvidenceGrade,
    Grader,
    GradingThresholds,
    HeuristicGrader,
    LLMGrader,
    Verdict,
    decide,
    first_json_object,
)
from .hierarchical import (
    Hierarchy,
    Node,
    contextualize,
    detect_headings,
    small_to_big,
    split_document,
    split_documents,
    to_chunks,
    to_evidence,
)
from .rewrite import (
    CUAD_BOILERPLATE,
    LLMRewriter,
    Rewrite,
    RewritePlan,
    Rewriter,
    TemplateRewriter,
    parse_variants,
)
from .spine import (
    AdvancedRagSpine,
    Attempt,
    Retrieve,
    SpineConfig,
    SpineResult,
    abstentions,
    evidence_from_hits,
    retrieval_runs,
    run_spine,
    with_small_to_big,
)


__all__ = [
    "ABSTENTION_TEXT",
    "Action",
    "AdvancedRagSpine",
    "Answerer",
    "Attempt",
    "CUAD_BOILERPLATE",
    "ChatCompleter",
    "ChunkGrade",
    "CorrectivePolicy",
    "DecompositionAnalysis",
    "EvidenceGrade",
    "FixedPromptSelector",
    "GENERATION_TEMPLATE",
    "Grader",
    "GradingThresholds",
    "GroundedAnswer",
    "HeuristicGrader",
    "Hierarchy",
    "LLMGrader",
    "LLMRewriter",
    "LoopDecision",
    "LoopPolicy",
    "LoopState",
    "Node",
    "PassthroughAnalysis",
    "PromptSelector",
    "QueryAnalysis",
    "QueryPlan",
    "RetrievalStep",
    "Retrieve",
    "RetrieverPromptSelector",
    "Rewrite",
    "RewritePlan",
    "Rewriter",
    "RewriterAnalysis",
    "SingleShotPolicy",
    "SpineConfig",
    "SpineResult",
    "TemplateRewriter",
    "Verdict",
    "abstention_flags",
    "abstentions",
    "contextualize",
    "decide",
    "detect_headings",
    "evidence_from_hits",
    "first_json_object",
    "generate",
    "looks_like_abstention",
    "merge_steps",
    "parse_answer",
    "parse_variants",
    "retrieval_runs",
    "run_spine",
    "small_to_big",
    "split_document",
    "split_documents",
    "step_bound",
    "to_chunks",
    "to_evidence",
    "with_small_to_big",
]
