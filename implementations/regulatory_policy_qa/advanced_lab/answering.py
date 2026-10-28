"""Controlled citation, scope, rewriting, and free-form QA experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, replace
from html.parser import HTMLParser
from typing import Any

from aieng.advanced_rag.corpora import contract_nli
from aieng.advanced_rag.evaluation.answers import judge_answer
from aieng.advanced_rag.evaluation.metrics import score_null_detection
from aieng.advanced_rag.models import REFERENCE_MODELS, choose_models
from aieng.advanced_rag.pipeline.generate import generate
from aieng.advanced_rag.pipeline.grading import LLMGrader
from aieng.advanced_rag.pipeline.rewrite import LLMRewriter
from aieng.advanced_rag.pipeline.spine import AdvancedRagSpine, SpineConfig, evidence_from_hits
from aieng.advanced_rag.retrieval.corpus import Chunk, Query, chunk_contract
from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder
from aieng.advanced_rag.retrieval.retrievers import DenseRetriever
from graphrag_null.judge_lab import CORRECTNESS_PROMPT, GROUNDING_PROMPT, checked_complete, references

from .common import EMBEDDING, OUTPUT, completer, digest, provenance, require_compute, run_path, write_json


REGULATORY_URL = (
    "https://www.govinfo.gov/content/pkg/USCODE-2024-title5/html/USCODE-2024-title5-partI-chap5-subchapII-sec552.htm"
)


def balanced_sample(queries: list[Query]) -> list[Query]:
    """Choose the first answerable and null query per document, before scoring."""
    selected: dict[tuple[str, bool], Query] = {}
    for query in queries:
        selected.setdefault((query.doc_id, query.is_answerable), query)
    return list(selected.values())


def judge_saved(query: Query, answer: Any, evidence: list[Any], reference: str, complete: Any) -> dict[str, Any]:
    """Separate correctness from grounding and preserve the canonical abstention."""
    checked = checked_complete(complete)
    correctness = judge_answer(
        query.query_id, query.question, answer.text, reference, [], checked, template=CORRECTNESS_PROMPT
    )
    grounding = judge_answer(
        query.query_id, query.question, answer.text, "", evidence, checked, template=GROUNDING_PROMPT
    )
    normalized = replace(
        correctness,
        correct=correctness.correct and not answer.abstained,
        grounded=grounding.grounded or answer.abstained,
        abstained=answer.abstained,
    )
    return {
        **asdict(normalized),
        "raw_correct": correctness.correct,
        "raw_grounded": grounding.grounded,
        "raw_correctness_abstained": correctness.abstained,
        "raw_grounding_abstained": grounding.abstained,
        "grounding_reason": grounding.reason,
    }


def summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the full answerability population and answerables separately."""
    nulls = {row["query_id"] for row in rows if row["is_null"]}
    answerable = [row for row in rows if not row["is_null"]]
    answered = [row for row in answerable if not row["abstained"]]
    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "nulls": len(nulls),
        "null_detection": score_null_detection({r["query_id"]: r["abstained"] for r in rows}, nulls).to_dict(),
        "answerable_correctness": sum(r["judgement"]["correct"] for r in answerable) / len(answerable)
        if answerable
        else None,
        "answerable_abstentions": sum(r["abstained"] for r in answerable),
        "answerable_grounding": sum(r["judgement"]["grounded"] for r in answerable) / len(answerable)
        if answerable
        else None,
        "hallucination_among_answered": sum(not r["judgement"]["grounded"] for r in answered) / len(answered)
        if answered
        else None,
    }


def previous_policy_execution() -> dict[str, Any] | None:
    """Retain metered provenance across repeated zero-call trace replays."""
    path = OUTPUT / "policies.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    if old["usage"]["calls"] == 0 and old.get("previous_execution"):
        return dict(old["previous_execution"])
    return {
        "generated_at": old["generated_at"],
        "report_sha256": digest(path),
        "usage": old["usage"],
        "note": "previous valid execution; current replay is strictly cache-only",
    }


def measure_policies(*, documents: int = 20, cache_only: bool = False) -> dict[str, Any]:  # noqa: PLR0915 - matched arms with shared role metering
    """Match citation policies on raw responses and isolate retrieval scope."""
    require_compute()
    path = contract_nli.fetch_contract_nli()
    corpus = contract_nli.build_slice(path, max_documents=documents)
    queries = balanced_sample(corpus.queries)
    gold = references(path, documents)
    dense = DenseRetriever(SentenceTransformerEmbedder(EMBEDDING))
    dense.index(corpus.chunks)
    by_id = {c.chunk_id: c for c in corpus.chunks}
    models = REFERENCE_MODELS if cache_only else choose_models()
    complete = completer(cache_only=cache_only, models=models)
    judge = completer(cache_only=cache_only, role="judging", models=models)
    rewrite = completer(cache_only=cache_only, role="extraction", models=models)
    judge.meter = rewrite.meter = complete.meter
    previous = previous_policy_execution() if cache_only else None
    rows = defaultdict(list)
    private_rows = []
    for query in queries:

        def retrieve(question: str, scope: str | None = None, k: int = 10, query: Query = query) -> list[Any]:
            if scope is None:
                # Empty doc IDs are filters, not an all-contracts search.
                return list(dense.retrieve(question, k, scope=None))
            return list(
                evidence_from_hits(
                    dense.search(replace(query, question=question, doc_id=scope or ""), k=k), by_id, "dense"
                )
            )

        scoped = retrieve(query.question, query.doc_id)[:5]
        raw_responses: list[str] = []

        def record(prompt: str, responses: list[str] = raw_responses) -> str:
            raw = str(complete(prompt))
            responses.append(raw)
            return raw

        complete.meter.phase = "generation:citation-pair"
        cited = generate(query.query_id, query.question, scoped, record, require_citations=True)
        if len(raw_responses) != 1:
            raise ValueError("paired citation experiment requires exactly one raw generation response")
        uncited = generate(
            query.query_id, query.question, scoped, lambda _, raw=raw_responses[0]: raw, require_citations=False
        )
        unscoped_evidence = retrieve(query.question, None)[:5]
        if not unscoped_evidence:
            raise ValueError("the unscoped control must retrieve from the indexed corpus")
        complete.meter.phase = "generation:dense-unscoped"
        unscoped = generate(query.query_id, query.question, unscoped_evidence, complete)
        loop = AdvancedRagSpine(
            retrieve, LLMGrader(judge), complete, rewriter=LLMRewriter(rewrite), config=SpineConfig(max_attempts=2)
        )
        complete.meter.phase = "pipeline:corrective-loop"
        loop_result = loop.answer(query.query_id, query.question, query.doc_id)
        # Judge grounding against the final evidence, including any rewrite.
        loop_evidence = list(loop_result.evidence)[:5]
        for name, answer, evidence in (
            ("citations-required", cited, scoped),
            ("citations-optional", uncited, scoped),
            ("dense-unscoped", unscoped, unscoped_evidence),
            ("corrective-loop", loop_result.answer, loop_evidence),
        ):
            if name != "dense-unscoped" and any(e.provenance.doc_id != query.doc_id for e in evidence):
                raise ValueError(f"{name}: evidence escaped the requested contract scope")
            reference = gold.get(query.query_id, "The contract does not state an answer to this question.")
            complete.meter.phase = f"judge:{name}"
            verdict = judge_saved(query, answer, evidence, reference, judge)
            rows[name].append(
                {
                    "query_id": query.query_id,
                    "doc_id": query.doc_id,
                    "is_null": not query.is_answerable,
                    "abstained": answer.abstained,
                    "citations": len(answer.citations),
                    "retrieved_doc_ids": [e.provenance.doc_id for e in evidence],
                    "reader_evidence_chars": sum(min(1200, len(e.text)) for e in evidence),
                    "judgement": verdict,
                    "attempt_trace": [
                        {key: value for key, value in attempt.to_dict().items() if key != "reason"}
                        for attempt in loop_result.attempts
                    ]
                    if name == "corrective-loop"
                    else [],
                    "raw_response_sha256": hashlib.sha256(raw_responses[0].encode()).hexdigest()
                    if name.startswith("citations-")
                    else None,
                }
            )
            private_rows.append(
                {
                    "arm": name,
                    "query_id": query.query_id,
                    "question": query.question,
                    "answer": answer.text,
                    "reference": reference,
                    "evidence": [e.text[:1200] for e in evidence],
                }
            )
        write_json(run_path("policy-answers.json"), private_rows)
        write_json(run_path("policy-progress.json"), dict(rows))
        print(f"Policy experiment: {sum(len(v) for v in rows.values())} arm outcomes", flush=True)
    report = {
        **provenance(models=models),
        "corpus_sha256": digest(path),
        "corpus": corpus.summary(),
        "selection": "first answerable and first null per document, selected before generation",
        "selected_ids": [q.query_id for q in queries],
        "citation_pairing": "both policies parse the identical saved raw generation response; no second generation call",
        "execution": "cached model responses only" if cache_only else "metered model calls with resumable cache",
        "arms": {name: {**summarize_answers(data), "outcomes": data} for name, data in rows.items()},
        "usage": complete.meter.to_dict(),
        "generation_and_loop_requests": {
            phase: {
                **complete.meter.phase_totals(phase),
                "responses_requested": complete.meter.phase_totals(phase)["calls"]
                + complete.meter.phase_totals(phase)["cache_hits"],
            }
            for phase in ("generation:citation-pair", "generation:dense-unscoped", "pipeline:corrective-loop")
        },
        "limitations": [
            "Small document-balanced demonstration, not the full-split threshold result.",
            "Generator and automated judge share a model; correctness is annotation agreement.",
        ],
    }
    if previous is not None:
        report["previous_execution"] = previous
    write_json(OUTPUT / "policies.json", report)
    return report


class TextExtractor(HTMLParser):
    """Extract government publication text without scripts or styles."""

    def __init__(self) -> None:
        """Initialize the text buffer and ignored-element depth."""
        super().__init__()
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Ignore executable and presentation elements."""
        if tag in ("script", "style"):
            self.ignored += 1

    def handle_endtag(self, tag: str) -> None:
        """Close an ignored element."""
        if tag in ("script", "style"):
            self.ignored -= 1

    def handle_data(self, data: str) -> None:
        """Keep visible nonempty text."""
        if not self.ignored and data.strip():
            self.parts.append(data.strip())


def load_regulatory() -> tuple[list[Chunk], dict[str, Any]]:
    """Load the shipped, digest-checked government snapshot for this experiment."""
    path = OUTPUT / "regulatory_source.json"
    source = json.loads(path.read_text())
    if (
        source["source_url"] != REGULATORY_URL
        or hashlib.sha256(source["text"].encode()).hexdigest() != source["text_sha256"]
    ):
        raise ValueError("regulatory source digest or origin mismatch")
    return chunk_contract(source["document_id"], source["text"]), source


def source_quote(text: str, quote: str) -> str:
    """Resolve whitespace-only differences to an exact original source span."""
    pattern = r"\s+".join(re.escape(token) for token in quote.split())
    found = re.search(pattern, text) if pattern else None
    if found is None:
        raise ValueError("supporting quote changes source wording")
    return text[found.start() : found.end()]


def generate_questions(chunks: list[Chunk], complete: Any) -> list[dict[str, Any]]:
    """Accept generated free-form questions only with exact source quotations."""
    selected = chunks[1:13:2]
    rows = []
    for index, chunk in enumerate(selected):
        prompt = (
            "Create one specific free-form question answerable entirely from this legal excerpt. "
            "Avoid yes/no and entailment questions. Return JSON with question, reference (a concise explanatory answer), "
            "and quote (an exact contiguous supporting quotation copied from the excerpt, at least 30 characters).\nExcerpt:\n"
            + chunk.text
        )
        for attempt in range(3):
            raw = complete(
                prompt
                + ("" if not attempt else f"\nRetry {attempt}: valid JSON and an exact verbatim quote are required.")
            )
            try:
                row = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
                if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "reference", "quote")):
                    raise ValueError("missing question fields")
                row["quote"] = source_quote(chunk.text, row["quote"])
                if len(row["quote"]) < 30:
                    raise ValueError("supporting quote is not in source")
                rows.append(
                    {
                        "query_id": f"foia-{index}",
                        "doc_id": chunk.doc_id,
                        **row,
                        "support_chunk_id": chunk.chunk_id,
                        "is_null": False,
                    }
                )
                break
            except (ValueError, KeyError):
                if attempt == 2:
                    raise ValueError("question generation failed source validation") from None
    rows.append(
        {
            "query_id": "foia-null",
            "doc_id": chunks[0].doc_id,
            "question": "What is the published telephone number of the lunar mining permit office?",
            "reference": "The provided statutory text does not state this information.",
            "quote": "",
            "support_chunk_id": None,
            "is_null": True,
        }
    )
    return rows


def measure_regulatory() -> dict[str, Any]:
    """Evaluate generated questions with reference-free retrieval and generation."""
    require_compute()
    chunks, source = load_regulatory()
    models = choose_models()
    complete = completer(max_tokens=1536, models=models)
    extractor = completer(max_tokens=1536, role="extraction", models=models)
    judge = completer(max_tokens=1536, role="judging", models=models)
    extractor.meter = judge.meter = complete.meter
    questions = generate_questions(chunks, extractor)
    dense = DenseRetriever(SentenceTransformerEmbedder(EMBEDDING))
    dense.index(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    rows, private = [], []
    for row in questions:
        query = Query(query_id=row["query_id"], doc_id=row["doc_id"], category="free-form", question=row["question"])
        evidence = evidence_from_hits(dense.search(query, k=5), by_id, "dense")
        answer = generate(query.query_id, query.question, evidence, complete)
        verdict = judge_saved(query, answer, evidence, row["reference"], judge)
        rows.append(
            {
                **row,
                "abstained": answer.abstained,
                "answer": answer.text,
                "judgement": verdict,
                "retrieved_ids": [e.provenance.chunk_id for e in evidence],
            }
        )
        private.append(
            {
                "user_input": query.question,
                "response": answer.text,
                "reference": row["reference"],
                "retrieved_contexts": [e.text[:1200] for e in evidence],
                "is_null": row["is_null"],
            }
        )
    write_json(run_path("regulatory-ragas-inputs.json"), private)
    write_json(run_path("regulatory_source.json"), source)
    report = {
        **provenance(models=models),
        "source": {k: v for k, v in source.items() if k != "text"},
        "question_generation": "model-generated free-form questions; exact supporting quotation required; one explicit off-topic null control",
        "summary": summarize_answers(rows),
        "outcomes": rows,
        "usage": complete.meter.to_dict(),
        "limitations": [
            "Synthetic question and reference quality is not human-adjudicated.",
            "Synthetic references are model-generated; this is an executable evaluation example, not an independently adjudicated benchmark.",
            "The 2024 source snapshot is for reproducible experiments, not a statement of current law.",
        ],
    }
    write_json(OUTPUT / "regulatory.json", report)
    return report
