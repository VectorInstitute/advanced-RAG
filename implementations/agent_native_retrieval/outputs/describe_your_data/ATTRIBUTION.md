# Sources and attribution for the describe-your-data results

`results.json` holds, for three corpora and four arms, the accuracy, model calls, tool calls,
and tokens per question, and one row per question and arm with its identifier, its type, its
cost, which tools it called, and whether the judge marked it correct. It carries no question
text, answer text, reference answer, or corpus text. `describe_your_data.py` in the parent
directory writes it from the cached runs.

The corpus folders, the question sets, the embeddings, and the full runs stay in the local cache
under `ri5/describe-your-data/`. This record follows the corpus records in
`../../../graphrag_technical_support/CORPUS.md` and `../../../table_structured_reasoning/CORPUS.md`
and does not rule on whether a corpus may be redistributed.

## ServerFault

The 10,000-question slice of the Stack Exchange dump published 2024-04-06, pinned and verified as
the RI #4 corpus record describes. `describe_your_data.py prepare serverfault` writes each of the
9,999 posts with an accepted answer as one Markdown file, and their metadata (title, tags, score,
year, link, licence) as two CSV tables. Posts are CC BY-SA at the version each post records.

```text
Stack Exchange user contributions to serverfault.com, licensed CC BY-SA at the
version recorded per post, from the Stack Exchange data dump published 2024-04-06.
https://archive.org/details/stackexchange
```

Changes made: posts were converted to plain text with Markdown headings, and their metadata was
written as tables. Twelve questions were computed from that metadata and the path index (tag
counts, top-scored post per tag, the post that mentions a path), and twelve were generated from
sampled posts by the synthetic Q&A path. Questions that quote a post are derived text under the
share-alike term, so the question set stays in the local cache.

## FinanceBench

The 84 filings the 150 FinanceBench questions cite, at commit `cc39aeb4` of
`patronus-ai/financebench`, converted by RI #3's `ingest_financebench.py`. The dataset card
states CC BY-NC 4.0, as the RI #3 corpus record describes. `describe_your_data.py prepare
financebench` joins each filing's pages into one Markdown file and writes its tables as CSV.
The 24 questions are FinanceBench's own, drawn by a fixed seed, 8 of each question type, and
are named in `results.json` by their `financebench_id` only.

```text
FinanceBench: A New Benchmark for Financial Question Answering, Islam,
Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI.
https://arxiv.org/abs/2311.11944
```

## click

The repository of [click](https://github.com/pallets/click), a Python command-line library, at
commit `d4ecb8a89a52ad3faacf3f1c65065d08c1d11c14` (tag `8.1.8`). Its `LICENSE.txt` is the
BSD 3-Clause licence, "Copyright 2014 Pallets", which permits redistribution with the copyright
notice and the licence's conditions.

```text
url         https://github.com/pallets/click/archive/d4ecb8a89a52ad3faacf3f1c65065d08c1d11c14.tar.gz
bytes       343,452
sha256      187d5ca45dec237364dff89c829c8faececafead4d5b599f008b07cb70c86a10
```

`describe_your_data.py` checks the size and digest on every use, including a cached archive, and
reads the archive as data: the source is parsed with Python's `ast` module and never run.
Sixteen questions were read from the source (parameter defaults, base classes, defining files)
and eight were generated from the documentation by the synthetic Q&A path.
