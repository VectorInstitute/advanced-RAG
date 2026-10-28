# RI #3 corpus licensing and provenance

FinanceBench is RI #3's corpus. This record states what its sources say about
licence and what the loader pins.

## FinanceBench

150 analyst questions over SEC filings, each with a reference answer and
page-level evidence. The 84 cited filings are 64 10-Ks, 8 10-Qs, 6 8-Ks, and 6
earnings releases. The sample has 50 questions of each type
(`metrics-generated`, `domain-relevant`, `novel-generated`), cites 84 filings, and
carries 189 evidence entries. `question_reasoning` is empty for 50 questions.

### Licence

The Hugging Face dataset card for
[`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench)
states `license: cc-by-nc-4.0` (card revision `e04404e3`, modified 2024-11-17).
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) permits use with
attribution and without commercial purpose.

The GitHub repository `patronus-ai/financebench`, which hosts the annotation files
and the PDFs, carries no licence file and no licence statement. The filings
themselves are SEC public filings by public companies. The card does not say
which of those two layers its licence covers.

`Licence.permits_commercial_use` is `False` for this corpus. That records the
licence text and is not a ruling.

### Pinned version

Every file is read at commit
[`cc39aeb4`](https://github.com/patronus-ai/financebench/commit/cc39aeb4afdf33909ee1412188bf89035950c2eb)
(2024-12-03), the repository head when the loader was written.

```text
data/financebench_open_source.jsonl          929,848 bytes
sha256      a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877
data/financebench_document_information.jsonl  88,781 bytes
sha256      1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575
pdfs/<doc_name>.pdf   84 files, 165.5 MB, digests in corpora/financebench.py
```

`aieng/advanced_rag/corpora/financebench.py` verifies every digest on each fetch,
including a file already in the cache. A changed upstream file fails with
`ChecksumMismatchError`. The loader never reads a moving reference such as `main`.

Evidence page numbers are zero-based: the smallest in the sample is 0 and the
largest is 303. Pages on `FinanceQuery.gold_pages` keep that base.

### Attribution

```text
FinanceBench: A New Benchmark for Financial Question Answering, Islam,
Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI.
https://arxiv.org/abs/2311.11944
```

### What stays local

The loader fetches into the local corpus cache. Neither the PDFs nor anything
derived from them, such as section folders or retrieval chunks, is committed. The
loader is not registered in `corpora.SOURCES`, because that registry's test
requires a commercial-use licence.

## Fallback

TAT-QA (16,000 questions over financial reports, "CC-BY-4.0" per its README) is an
alternative that permits commercial use. No loader exists for it yet.

## Open questions

FinanceBench is licensed CC BY-NC 4.0, which limits use to non-commercial purposes.
