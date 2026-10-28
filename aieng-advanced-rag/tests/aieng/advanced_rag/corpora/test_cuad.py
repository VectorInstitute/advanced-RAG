"""CUAD loading, category parsing, and judgment derivation."""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip(
    "aieng.advanced_rag.retrieval.corpus",
    reason="corpus types have not landed in aieng.advanced_rag.retrieval yet",
)

from aieng.advanced_rag.corpora import cuad  # noqa: E402


def test_category_comes_out_of_the_templated_question() -> None:
    question = (
        'Highlight the parts (if any) of this contract related to "Governing Law" '
        "that should be reviewed by a lawyer."
    )
    assert cuad.parse_category(question) == "Governing Law"


def test_unparseable_question_does_not_raise() -> None:
    assert cuad.parse_category("no category here") == "unknown"


def test_load_yields_one_entry_per_contract(cuad_file: Path) -> None:
    loaded = list(cuad.load_cuad(cuad_file))
    assert [doc_id for doc_id, _, _ in loaded] == ["contract-one", "contract-two"]


def test_answer_offsets_locate_the_gold_text(cuad_file: Path) -> None:
    _, text, queries = next(iter(cuad.load_cuad(cuad_file)))
    answer = queries[0].answers[0]
    assert text[answer.start : answer.end] == answer.text


def test_impossible_questions_become_null_queries(cuad_file: Path) -> None:
    eval_slice = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10)
    assert [q.query_id for q in eval_slice.answerable] == ["q-answerable"]
    assert {q.query_id for q in eval_slice.null_queries} == {"q-null", "q-other"}


def test_judgments_point_at_chunks_holding_the_span(cuad_file: Path) -> None:
    eval_slice = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10)
    gold = eval_slice.qrels["q-answerable"]
    assert gold
    answer = eval_slice.answerable[0].answers[0]
    for chunk in eval_slice.chunks:
        overlaps = chunk.start < answer.end and answer.start < chunk.end
        assert (chunk.chunk_id in gold) is overlaps


def test_null_queries_get_an_empty_judgment_set(cuad_file: Path) -> None:
    eval_slice = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10)
    assert eval_slice.qrels["q-null"] == set()


def test_max_documents_takes_a_stable_prefix(cuad_file: Path) -> None:
    first = cuad.build_slice(
        cuad_file, max_documents=1, chunk_chars=60, overlap_chars=10
    )
    both = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10)
    assert {c.contract_id for c in first.chunks} == {"contract-one"}
    assert {c.chunk_id for c in first.chunks} <= {c.chunk_id for c in both.chunks}


def test_queries_scope_to_one_document(cuad_file: Path) -> None:
    eval_slice = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10)
    scoped = cuad.queries_for_document(eval_slice, "contract-two")
    assert [q.query_id for q in scoped] == ["q-other"]


def test_summary_reports_the_null_rate(cuad_file: Path) -> None:
    summary = cuad.build_slice(cuad_file, chunk_chars=60, overlap_chars=10).summary()
    assert summary["queries"] == 3
    assert summary["null_queries"] == 2
    assert summary["null_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_pinned_source_is_immutable_and_permissive() -> None:
    source = cuad.CUAD_TEST
    # A branch reference would let the bytes change under a pinned digest.
    assert "/main/" not in source.url
    assert len(source.sha256) == 64
    assert source.licence.permits_commercial_use
    assert source.licence.name == "CC BY 4.0"
