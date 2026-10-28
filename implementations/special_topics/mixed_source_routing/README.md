# Special topic: route a question across prose and tables

This special topic compares ways of answering questions over a corpus that holds both prose
documents and tables: one index over everything, a router in front of separate sources, and an
agent that picks its own tools. It is for teams whose data mixes documents with tables, a
database, or other structured records, and who need to decide whether a router improves accuracy enough to justify it.

## Notebook

`01_route_across_sources.ipynb` runs seven arms on 60 questions over the 84 FinanceBench
filings, 20 each answered by prose, by a table, or by both. It reports accuracy, model calls,
tokens, and routing errors per question type. It then follows up on the table source and tests
that follow-up on 20 held-out table questions.

The notebook follows the repository pattern: `RUN_EXPENSIVE = False` and
`MODELS = choose_models(show=True)` in the first code cell, every model step through
`load_or_run`, and `spend.report()` last. With the flag off it reads the reference run. With it
on, and the local chunks, embeddings, and question set built, it runs the routes, the LLM
router, the agent, and the judge on your models. A mid or frontier model is capped at 4
questions per type.

## Arms

| Arm | Who chooses the source | What it reads |
| --- | --- | --- |
| `U5`, `U10` | nobody | one dense index over prose and table chunks, top 5 or 10 |
| `RE` | an embedding router (the `semantic-router` method, local `bge-small-en-v1.5`) | the chosen route |
| `RL` | an LLM (large language model) router (LlamaIndex `LLMSingleSelector` style), one call | the chosen route |
| `B` | nobody: every question goes to both sources | prose top 5 and SQL rows |
| `A` | the agent from `implementations/agent_native_retrieval` (its `describe_your_data.py`, arm `P`), with the tools the FinanceBench profile generates | whatever it calls |
| `O` | the oracle: the gold label | the gold route |

The `prose` route reads the top 5 prose chunks. The `table` route writes one SQL query over the
profile's `cells`, `tables`, and `filings` tables, retries once on an error or no rows, and
reads the rows. The `both` route reads both.

## Results of the reference run

One run, no repeats, with `Qwen3_6-35B-A3B` answering and judging. `outputs/results.json`,
`outputs/followup_sources.json`, and `outputs/heldout_tables.json` hold every figure.

- The oracle router answered 20 of 60 and one index with no router answered 31 (`U10`). The
  table route answered 3 of 20 table questions because the generated SQL returned no rows for
  13 of them. The LLM router chose the gold route for 56 of 60 and scored what the oracle
  scored.
- The agent answered 45 of 60 (18 prose, 13 table, 14 two-source) at 12.8 calls and about
  209,000 prompt tokens per question, roughly 66 times the tokens of `U10`. It reached its
  30-call budget and was made to answer on 8 of the 60.
- The embedding router sent 52 of 60 questions to the table route and answered 5.
- Follow-up: giving the SQL step the filing list and three matching rules lifted table
  questions to 14 of 20, and dense search over table chunks reached 12. With either source, the
  oracle, the LLM router, and "always both" tied (31 or 33 of 60), and two-source questions
  reached only 6 of 20 against the agent's 14.
- Held-out table questions: the follow-up's rules were written after reading failed queries
  from the reference question set, so 14 of 20 is in-sample. On 20 held-out table questions over 19
  filings that no reference question uses, dense search over table chunks answered 15, the
  follow-up SQL prompt 12, and the reference SQL prompt 7. The reference prompt returned no
  rows for 11 of the 20, so its failure repeats on filings it has not seen. The follow-up prompt's gain
  over it is 5 questions out of sample against 11 in-sample, and dense search answered more.
  The two sources miss different questions: reading SQL rows beside table chunks answered 16
  of 20, but that design came after seeing these scores and has not been tested on fresh
  questions.

A hand check of 12 sampled reference questions found one ambiguous reference (a pension figure
whose plan the question does not name), and the rest were sound.

## Prior art and why this design

The two routers follow `semantic-router` (Aurelio AI, MIT) and LlamaIndex's
`RouterQueryEngine` selectors. The table route has the shape of TAG (table-augmented generation;
"Text2SQL is Not Enough", Biswal and others, 2024), over the guarded SQLite tables of the
`agent_native_retrieval` data profile. The agent is that topic's arm `P`, run on Google ADK
(Agent Development Kit) through `aieng.advanced_rag.agents.session_loop`, so you can add ADK
tools, callbacks, and plugins to it. It keeps the arm's 30-call budget, the forced final
answer, the read-before-answer prompt, and the error it returns to the model for tool-call
arguments that do not parse. ADK declares each tool from its whole docstring, parameters and
return value included, so every request carries about 5,800 characters of tool text for the
six tools. Neither
library is in the project's lock file, so `routing.py` writes the two routers out on the project's cached, metered client and local embedder. Learned
routers (RouteLLM, Adaptive-RAG, StructRAG) need training labels. The per-route outcomes in
`results.json` are the kind Adaptive-RAG trains on. The notebook has the full section.

## Files

- `routing.py`: the routers (`EmbeddingRouter`, `LLMRouter`, `OracleRouter`, `FixedRouter`)
  and the scoring (`routing_report`, `hindsight_route`, `summarise_arm`). It depends on no
  corpus, so you can use it over your own sources.
- `build_reference.py`: the chunks and index, the SQL source, the question builder, the cached
  runs, the follow-up, and the held-out run. Its docstring lists the commands.
- `tests/`: unit tests for both modules, and checks that the committed outputs agree with each
  other, name their models, and carry no corpus text.

## Rebuild the reference

Prerequisites:

- The Docling ingestion of FinanceBench from `implementations/table_structured_reasoning` in
  the cache (`ingest_financebench.py`).
- The FinanceBench folder from `implementations/agent_native_retrieval`
  (`python describe_your_data.py prepare financebench`).

Run from this folder. A stage has finished when its file exists: `prepare` writes the chunks
and embeddings to the local cache, `questions` writes the question set and
`outputs/question_set.json`, `run` writes `outputs/results.json`, `followup` writes
`outputs/followup_sources.json`, and `heldout` writes `outputs/heldout_tables.json`.

```bash
python build_reference.py prepare
```

```bash
python build_reference.py questions --per-minute 20
```

```bash
python build_reference.py run --per-minute 20
```

```bash
python build_reference.py followup --per-minute 20
```

```bash
python build_reference.py heldout --per-minute 20
```

`prepare` embeds about 54,000 chunks on CPU, which took over an hour on a shared 32-vCPU
workstation. The model steps for the questions, the arms, and the follow-up use about 2,000 calls on the
on-prem models at 20 requests per minute, about 2 hours, and `heldout` adds about 400.
The agent's responses are cached by an ADK plugin in a file of their own, and the other
steps' responses by the governed client. Every response is cached, so a
second run makes no call.

## Data and license

FinanceBench is CC BY-NC 4.0 (see `../../table_structured_reasoning/CORPUS.md`). The chunks,
embeddings, questions, answers, and SQL stay in the local caches. `outputs/` holds identifiers,
counts, and metrics only, with the attribution in `outputs/ATTRIBUTION.md`, and the committed
notebook outputs print no question or filing text.
