# RI #3: Table and structured-data reasoning

This reference implementation (RI) retrieves and answers over financial filings
whose answers sit in tables. It covers ingesting filings into section folders and
retrieving table cells with their headers, on what a filing corpus tests: tables
with nested headers, section structure, and questions whose answer is a cell
value.

## Where it sits in the program

This RI varies what the pipeline indexes and keeps the agent unchanged. Section
folders and table-aware chunks are built, and a database queried with SQL
(Structured Query Language) is planned. RI #5 builds on it.

## What is built

- Docling ingestion of PDFs into one folder per document, with section files and
  a setting for how tables are stored.
- Table-aware chunking that keeps a table with its heading and repeats the header
  rows when a table is split.
- Retrieval judgments derived from the pages that FinanceBench's evidence entries
  name, scored in the RI #1 harness.
- A comparison of plain and table-aware chunking on the 150 questions.

## What is planned

- A paired comparison of ingestion routes and table storage settings on the 150
  questions.
- A database extension on the `financial` database of BIRD Mini-Dev, a text-to-SQL
  benchmark, behind a read-only SQL guard.

## Datasets

- FinanceBench (`PatronusAI/financebench`, CC-BY-NC-4.0): 150 analyst questions
  over SEC (Securities and Exchange Commission) filings, with PDFs and
  pre-aligned evidence pages. The 84 filings the questions cite are 64 10-Ks, 8
  10-Qs, 6 8-Ks, and 6 earnings releases. The licence is non-commercial. TAT-QA is an alternative
  that permits commercial use.
- BIRD Mini-Dev, `financial` database: the structured-data extension.

## Prerequisites

- Python and the bootcamp environment (see `docs/`).
- Familiarity with RI #1's evaluation harness.
- The package installed with the `docling` and `local` extras. The first installs
  the converter and the second the local embedding model.
- About 1.35 GB of disk: 166 MB for the PDFs, 590 MB for the section folders, and 590 MB
  for the parse cache.

## Run it

This procedure ingests the filings and scores both chunkings. Converting takes
about half an hour split across three processes, and scoring takes about 75
minutes. To split the conversion, run step 1 as three processes with
`--shard 0/3`, `--shard 1/3`, and `--shard 2/3`.

1. Fetch the filings and convert them into section folders:
   `python implementations/table_structured_reasoning/ingest_financebench.py`.
2. Check the folders:
   `python implementations/table_structured_reasoning/ingest_financebench.py --summarise implementations/table_structured_reasoning/outputs/ingestion_summary.json`.
   It prints `84 documents, 11678 pages, 7600 tables, 0 failures`.
3. Score both chunkings:
   `python implementations/table_structured_reasoning/score_table_retrieval.py`.
   It prints retrieval rates and cell attachment for each chunking, and a second run
   reads the cached result.
4. Open `01_ingestion_and_table_retrieval.ipynb`. With `RUN_EXPENSIVE` set to `False`
   in its first code cell, it reads the results in `outputs/`.

The PDFs and section folders stay in the local cache and are never committed.
`CORPUS.md` records the licence and the pins. If a download fails its digest check,
the upstream file changed: stop and read `CORPUS.md`.

## Results

Ingesting the 84 filings produced 11,678 pages with content and 7,600 tables, with
no failures. A page with no extracted content writes no page file, so a page
number can be missing from a folder.

Both chunkings cut the same serialised text at 1,200 characters. Retrieval is
scoped to the question's own filing and judged at `k=10` against the pages that
FinanceBench's evidence names. BM25 is a keyword-ranking function, and the dense
arm is `BAAI/bge-small-en-v1.5` run locally. MRR is mean reciprocal rank.

| Chunking | Chunks | Arm | Retrieval failure | MRR |
| --- | ---: | --- | ---: | ---: |
| Plain windows | 41,401 | BM25 | 0.6267 | 0.1965 |
| Plain windows | 41,401 | Dense | 0.3400 | 0.3459 |
| Table-aware | 53,715 | BM25 | 0.6600 | 0.1677 |
| Table-aware | 53,715 | Dense | 0.4000 | 0.3542 |

Table-aware chunking keeps a numeric cell in a chunk with its column header and
row label in 94.6% of its appearances, against 51.3% for plain windows. Of the
189,040 scored numeric cells, 99.9% appear complete in some chunk, against 62.5%.
It does not improve retrieval on these questions. Dense failure rises from 0.34 to
0.40, which is nine questions of 150. That difference is unpaired and untested, and
the table-aware chunking makes more, smaller chunks. The figures are in
`outputs/table_retrieval_scores.json`.

## Status

The loader, ingestion, judgments, table-aware retrieval, and the
notebook are done. The paired comparison, the three-way ingestion comparison, and
the database extension are not.
