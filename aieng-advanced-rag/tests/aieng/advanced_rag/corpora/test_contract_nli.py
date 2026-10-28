"""ContractNLI loading and the NotMentioned-to-null mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip(
    "aieng.advanced_rag.retrieval.corpus",
    reason="corpus types have not landed in aieng.advanced_rag.retrieval yet",
)

from aieng.advanced_rag.corpora import contract_nli  # noqa: E402


def test_one_query_per_hypothesis(contract_nli_file: Path) -> None:
    _, _, queries = next(iter(contract_nli.load_contract_nli(contract_nli_file)))
    assert {q.query_id for q in queries} == {"1::nda-16", "1::nda-11"}


def test_question_is_the_hypothesis_text(contract_nli_file: Path) -> None:
    _, _, queries = next(iter(contract_nli.load_contract_nli(contract_nli_file)))
    evidenced = next(q for q in queries if q.query_id == "1::nda-16")
    assert evidenced.question.startswith("Receiving Party shall destroy or return")
    assert evidenced.category == "Return of confidential information"


def test_evidence_indices_resolve_to_document_offsets(
    contract_nli_file: Path,
) -> None:
    _, text, queries = next(iter(contract_nli.load_contract_nli(contract_nli_file)))
    answer = next(q for q in queries if q.query_id == "1::nda-16").answers[0]
    assert text[answer.start : answer.end] == answer.text
    assert "return all materials" in answer.text


def test_not_mentioned_is_a_null_query(contract_nli_file: Path) -> None:
    eval_slice = contract_nli.build_slice(
        contract_nli_file, chunk_chars=40, overlap_chars=10
    )
    assert [q.query_id for q in eval_slice.null_queries] == ["1::nda-11"]
    assert eval_slice.qrels["1::nda-11"] == set()


def test_judgments_cover_the_evidence(contract_nli_file: Path) -> None:
    eval_slice = contract_nli.build_slice(
        contract_nli_file, chunk_chars=40, overlap_chars=10
    )
    gold = eval_slice.qrels["1::nda-16"]
    assert gold
    covered = "".join(
        chunk.text for chunk in eval_slice.chunks if chunk.chunk_id in sorted(gold)
    )
    assert "return" in covered


def test_max_documents_truncates(tmp_path: Path, contract_nli_file: Path) -> None:
    payload = json.loads(contract_nli_file.read_text(encoding="utf-8"))
    second = dict(payload["documents"][0])
    second["id"] = 2
    payload["documents"].append(second)
    path = tmp_path / "two_docs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    eval_slice = contract_nli.build_slice(
        path, max_documents=1, chunk_chars=40, overlap_chars=10
    )
    assert {chunk.contract_id for chunk in eval_slice.chunks} == {"1"}


def test_pinned_splits_share_one_verified_archive() -> None:
    splits = (
        contract_nli.CONTRACT_NLI_TRAIN,
        contract_nli.CONTRACT_NLI_DEV,
        contract_nli.CONTRACT_NLI_TEST,
    )
    assert len({source.url for source in splits}) == 1
    assert len({source.member_sha256 for source in splits}) == 3
    for source in splits:
        assert source.licence.permits_commercial_use
