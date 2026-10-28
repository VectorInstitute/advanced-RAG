"""FinanceBench loader: the pins, the parsing, and the digest checks, all offline."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from aieng.advanced_rag.corpora import financebench, sources
from aieng.advanced_rag.corpora.financebench import (
    FinanceQuery,
    fetch_financebench,
    load_queries,
    parse_queries,
    pdf_source,
)
from aieng.advanced_rag.corpora.sources import ChecksumMismatchError, PinnedSource


def record(qid="financebench_id_1", doc="ACME_2022_10K", pages=(3,), **extra):
    return {
        "financebench_id": qid,
        "doc_name": doc,
        "question": f"What is {qid}?",
        "answer": "$5",
        "question_type": "metrics-generated",
        "question_reasoning": None,
        "evidence": [
            {
                "doc_name": doc,
                "evidence_page_num": page,
                "evidence_text": f"text on page {page}",
            }
            for page in pages
        ],
        **extra,
    }


def lines(*records):
    return [json.dumps(r) for r in records]


def test_the_pins_name_a_commit_and_never_main():
    assert re.fullmatch(r"[0-9a-f]{40}", financebench.FINANCEBENCH_COMMIT)
    for source in (
        financebench.FINANCEBENCH_OPEN_SOURCE,
        financebench.FINANCEBENCH_DOCUMENT_INFORMATION,
        pdf_source("3M_2018_10K"),
    ):
        assert financebench.FINANCEBENCH_COMMIT in source.url
        assert "/main/" not in source.url
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256)


def test_every_cited_filing_has_a_distinct_digest():
    assert len(financebench.PDF_SHA256) == 84
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", h) for h in financebench.PDF_SHA256.values()
    )
    assert len(set(financebench.PDF_SHA256.values())) == 84


def test_the_licence_records_the_terms_and_leaves_the_ruling_alone():
    licence = financebench.FINANCEBENCH_LICENCE
    assert licence.name == "CC BY-NC 4.0"
    assert licence.permits_commercial_use is False
    assert "TAT-QA" in licence.notes


def test_a_question_carries_its_sorted_unique_pages_and_texts():
    [query] = parse_queries(lines(record(pages=(7, 3, 7))))
    assert isinstance(query, FinanceQuery)
    assert query.query_id == "financebench_id_1"
    assert query.doc_id == "ACME_2022_10K"
    assert query.gold_pages == (3, 7)
    assert len(query.evidence_texts) == 3
    assert query.reasoning == ""
    assert query.is_answerable


def test_page_zero_is_a_real_page():
    assert financebench.PAGE_BASE == 0
    [query] = parse_queries(lines(record(pages=(0,))))
    assert query.gold_pages == (0,)


def test_blank_lines_are_skipped():
    assert len(parse_queries(["", *lines(record()), "  "])) == 1


def test_a_question_without_evidence_is_rejected():
    with pytest.raises(ValueError, match="no evidence"):
        parse_queries(lines(record(evidence=[])))


def test_evidence_in_another_filing_is_rejected():
    bad = record()
    bad["evidence"][0]["doc_name"] = "OTHER_2021_10K"
    with pytest.raises(ValueError, match="OTHER_2021_10K"):
        parse_queries(lines(bad))


def test_an_unknown_filing_has_no_source():
    with pytest.raises(KeyError, match="NOPE"):
        pdf_source("NOPE")


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Serve fake bytes under fake pins, so the real fetch and verify run."""
    jsonl = ("\n".join(lines(record())) + "\n").encode()
    pdf = b"%PDF-1.4 fake"
    digest = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    open_source = PinnedSource(
        name="financebench-open-source",
        url="https://example.invalid/open.jsonl",
        sha256=digest(jsonl),
        licence=financebench.FINANCEBENCH_LICENCE,
    )
    info = PinnedSource(
        name="financebench-document-information",
        url="https://example.invalid/info.jsonl",
        sha256=digest(b"{}\n"),
        licence=financebench.FINANCEBENCH_LICENCE,
    )
    monkeypatch.setattr(financebench, "FINANCEBENCH_OPEN_SOURCE", open_source)
    monkeypatch.setattr(financebench, "FINANCEBENCH_DOCUMENT_INFORMATION", info)
    monkeypatch.setattr(financebench, "PDF_SHA256", {"ACME_2022_10K": digest(pdf)})
    payloads = {
        open_source.url: jsonl,
        info.url: b"{}\n",
        pdf_source("ACME_2022_10K").url: pdf,
    }
    calls: list[str] = []

    def download(url: str) -> bytes:
        calls.append(url)
        return payloads[url]

    monkeypatch.setattr(sources, "_download", download)
    return payloads, calls, tmp_path


def test_fetch_downloads_verifies_and_then_serves_from_the_cache(served):
    payloads, calls, root = served
    paths = fetch_financebench(dest=root)
    assert set(paths.pdfs) == {"ACME_2022_10K"}
    assert paths.pdfs["ACME_2022_10K"].read_bytes() == b"%PDF-1.4 fake"
    assert len(calls) == 3
    fetch_financebench(dest=root)
    assert len(calls) == 3, "a verified cached file is not downloaded again"
    assert load_queries(paths.open_source)[0].query_id == "financebench_id_1"


def test_a_changed_upstream_file_fails_the_digest(served):
    payloads, calls, root = served
    payloads[pdf_source("ACME_2022_10K").url] = b"%PDF-1.4 tampered"
    with pytest.raises(ChecksumMismatchError):
        fetch_financebench(dest=root)


def test_a_corrupted_cached_file_is_replaced(served):
    payloads, calls, root = served
    paths = fetch_financebench(dest=root)
    paths.pdfs["ACME_2022_10K"].write_bytes(b"partial")
    fetch_financebench(dest=root)
    assert paths.pdfs["ACME_2022_10K"].read_bytes() == b"%PDF-1.4 fake"
