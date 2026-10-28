# Attribution for the results in this directory

The JSON files hold identifiers, counts, and metrics from the reference run of
`01_route_across_sources.ipynb`:

- `question_set.json`: the question set's counts per type, its generation funnel, and each
  question's identifier, type, filing name, and source identifiers.
- `results.json`: accuracy, model calls, and tokens per arm and question type, each router's
  confusion matrix and routing errors, how often each fixed route answered per question type,
  the SQL attempts per question, and one row per question and arm with the route taken and
  whether the judge marked the answer correct.

- `followup_sources.json`: every router scored again under each table source.
- `heldout_tables.json`: the held-out table run, with 20 question identifiers on filings the
  reference questions do not use, and per question and source whether the answer was correct,
  how many SQL rows came back, and whether the rows held the reference figure.

No file carries question text, answer text, SQL text, chunk text, or filing text. A chunk
appears only as `<FinanceBench doc_name>::<chunk index>`, a table cell as
`<doc_name>#i<item index>@r<row>c<column>`, and a filing by its `doc_name`. The questions,
the chunks, the embeddings, and every model response stay in the local caches.

The figures derive from FinanceBench, which the Hugging Face dataset card licenses
CC BY-NC 4.0, so they carry its attribution:

```text
FinanceBench: A New Benchmark for Financial Question Answering, Islam,
Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI.
https://arxiv.org/abs/2311.11944
```

Changes made: the 84 filings were parsed with Docling (`table_structured_reasoning`), chunked with its
table-aware chunker, embedded, loaded into SQLite as cell tables, and used to generate and
check new questions. No FinanceBench question is used. See
`../../../table_structured_reasoning/CORPUS.md`.
