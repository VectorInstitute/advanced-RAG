"""Synthetic question sets for a corpus with no gold labels.

Most bootcamp teams bring a corpus with no gold passages and no reference
answers. This subpackage writes questions from the corpus, checks them, and
emits gold evidence as character spans, so the harness scores any chunker
against them with its own overlap rule.

    from aieng.advanced_rag.evaluation.synthetic import (
        Document, build_single_hop_set, load_config, proxy_model,
    )

    config = load_config(Path("team-config.yaml"))
    generated = build_single_hop_set(
        documents, config,
        generator=proxy_model("gpt-5.4-mini", cache_path=cache),
        checker=proxy_model("claude-haiku-5-5", cache_path=cache),
        judge=proxy_model("gemini-3.5-flash-lite", cache_path=cache),
        embedder=SentenceTransformerEmbedder(),
    )
    qrels = generated.qrels(chunks)

The output plugs into the existing harness without changing it.
:class:`SyntheticQuestion` has ``query_id``, ``question``, ``is_answerable``,
and ``doc_id``, so :class:`~aieng.advanced_rag.evaluation.runner.Experiment`
runs it; a null question has no evidence, so ``score_null_detection`` reads it;
and :meth:`SyntheticQuestion.to_query` returns the corpus ``Query`` the
per-document retrievers take.

Two rules carry over from the experiment this was ported from: no graph seeds
the questions, and no retriever's success or failure decides whether a question
is kept. Generated labels support comparing retrieval configurations by rank
(:mod:`.agreement`); they do not support comparing answer models, and their
absolute scores run high.
"""

from .agreement import (
    compare_rankings,
    kendall_tau,
    metric,
    pairwise_agreement,
    per_question,
    ranking,
)
from .build import (
    NULL_REFERENCE,
    build_fanout_questions,
    build_multi_hop_questions,
    build_single_hop_set,
)
from .config import (
    EXAMPLE_CONFIG,
    Category,
    GenerationConfig,
    config_from_dict,
    load_config,
)
from .labels import difficulty
from .llm import Model, family, parse_json, proxy_model, usage_of
from .probe import ProbeIndex
from .sampling import (
    CandidatePair,
    Condition,
    Document,
    Passage,
    band_report,
    embed_documents,
    held_out_split,
    passages_of,
    sample_passages,
    select_pairs,
    tag_conditions,
)
from .statements import Statement, extract_statements, locate_quote
from .taxonomy import ALL_TYPES, BY_NAME, REPORT_LABELS, QuestionType
from .testset import EvidenceSpan, Funnel, Outcome, SyntheticQuestion, SyntheticSet


__all__ = [
    "ALL_TYPES",
    "BY_NAME",
    "EXAMPLE_CONFIG",
    "NULL_REFERENCE",
    "REPORT_LABELS",
    "CandidatePair",
    "Category",
    "Condition",
    "Document",
    "EvidenceSpan",
    "Funnel",
    "GenerationConfig",
    "Model",
    "Outcome",
    "Passage",
    "ProbeIndex",
    "QuestionType",
    "Statement",
    "SyntheticQuestion",
    "SyntheticSet",
    "band_report",
    "build_fanout_questions",
    "build_multi_hop_questions",
    "build_single_hop_set",
    "compare_rankings",
    "config_from_dict",
    "difficulty",
    "embed_documents",
    "extract_statements",
    "family",
    "held_out_split",
    "kendall_tau",
    "load_config",
    "locate_quote",
    "metric",
    "pairwise_agreement",
    "parse_json",
    "passages_of",
    "per_question",
    "proxy_model",
    "ranking",
    "sample_passages",
    "select_pairs",
    "tag_conditions",
    "usage_of",
]
