# Attribution for the results in this directory

`ingestion_summary.json` holds, for each filing, the number of pages with content,
tables, and ingestion failures. `table_retrieval_scores.json` holds retrieval
rates and table-cell attachment rates for two chunkings of the same filings. Neither
file carries question text, answer text, or filing text. A filing appears only by
the name FinanceBench gives it.

The figures derive from FinanceBench, which the Hugging Face dataset card
licenses CC BY-NC 4.0, so they carry its attribution:

```text
FinanceBench: A New Benchmark for Financial Question Answering, Islam,
Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI.
https://arxiv.org/abs/2311.11944
```

Changes made: the 84 filings the 150 questions cite were converted with Docling,
chunked into 1,200-character chunks two ways, and judged against the pages
FinanceBench's evidence entries name. The PDFs and section
folders are not committed. See `../CORPUS.md`.
