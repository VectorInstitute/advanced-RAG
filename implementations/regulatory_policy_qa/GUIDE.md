# Work in RI #1: set up, verify, and build the spine

Set up the environment, confirm that reference implementation (RI) #1 reproduces its committed
numbers, and find the code behind each technique, in the order the notebooks build them.

RI #1 is the Advanced RAG spine and the evaluation harness the other RIs reuse. Its pipeline
code and the harness live in the shared package, under `aieng-advanced-rag/aieng/advanced_rag/`,
with the spine itself, `AdvancedRagSpine`, in `pipeline/spine.py`. Its folder holds the measurement runners, their tests, the
committed outputs, and the memos, and its notebooks follow [the build order](#build-ri-1-in-order).

## Prerequisites

- Python 3.12, which the repository pins in `.python-version`.
- [uv](https://docs.astral.sh/uv/) for the environment.
- A Vector proxy key, for the runs that call a model. Setup and the offline checks need no key.

## Set up the environment

Run the commands from the repository root unless a step says otherwise. None of them needs root
privileges. `implementations/regulatory_policy_qa/` is its own Python project and not a workspace
member, so `uv run` inside that folder fails to resolve the shared package.

1. Create the environment with every workspace package, all their extras, and the development
   tools:

   ```bash
   $ uv sync --all-packages --all-extras --dev
   ```

   Without `--all-packages`, `--all-extras` covers only the root project, and the shared package
   installs without the extras its tests and runners need.

2. Make the proxy key available. On a Coder workspace, `onboard` exports `OPENAI_API_KEY` and
   `OPENAI_BASE_URL` into your shell and this step is done. On your own machine, add the key to
   `.env` at the repository root:

   ```text
   VECTOR_PROXY_API_KEY=<your_key>
   ```

   where `<your_key>` is your Vector proxy key.

   Every runner reads the key through `aieng.advanced_rag.proxy.resolve_api_key`: an exported
   `OPENAI_API_KEY` when `OPENAI_BASE_URL` points at the proxy, then an exported
   `VECTOR_PROXY_API_KEY`, then the same names in `.env`. An `OPENAI_API_KEY` exported without the
   proxy endpoint is never sent, because it may belong to OpenAI's own API. The GraphRAG null arm's
   `--env` flag points it at another `.env`. [`docs/setup.md`](../../docs/setup.md) covers the
   credential model and [`docs/models-and-spend.md`](../../docs/models-and-spend.md) the models the
   proxy serves and what they cost.

3. Optional: set `ADVANCED_RAG_CORPUS_CACHE` to move the corpus cache. It defaults to
   `~/.cache/aieng-advanced-rag/corpora`.

## Verify the setup

These steps confirm the environment without calling a model.

1. Move into the shared package:

   ```bash
   $ cd aieng-advanced-rag
   ```

2. Run its tests:

   ```bash
   $ uv run --all-extras pytest -m "not integration_test" tests
   ```

   The offline suite passes with the extras installed. Published-figure guards
   need their corpus and local models; fetch them with `uv run python scripts/fetch_test_data.py`
   from the repository root before checking the measured retrieval figures.

3. Return to the repository root:

   ```bash
   $ cd ..
   ```

4. Run RI #1's runner tests:

   ```bash
   $ uv run python -m pytest implementations/regulatory_policy_qa/tests implementations/regulatory_policy_qa/graphrag_null/tests
   ```

5. Reproduce the threshold memo's table from the committed outputs:

   ```bash
   $ uv run python implementations/regulatory_policy_qa/summarise_sweep.py implementations/regulatory_policy_qa/outputs/sweep_contract-nli.json --grader llm --attempts 2 --sufficient 0.6
   ```

   The first row reads 0.787 balanced accuracy at thresholds 0.60 and 0.00, and the "no grader"
   row reads 0.780. Both match [the threshold memo](THRESHOLDS.md).

## Troubleshoot the setup

| Symptom | Cause and fix |
| --- | --- |
| Test collection fails with a numpy import error | The shared package's extras are missing. A root `uv sync` without `--all-packages` installs the package without them. Run `uv sync --all-packages --all-extras --dev`. |
| `uv run` reports that `aieng-advanced-rag` was not found in the package registry | The command ran inside `implementations/regulatory_policy_qa/`. Run it from the repository root. |
| A shell tool cannot find the key after `set -a; . ./.env` | A `.env` line with spaces around `=` does not parse in a shell. The runners read the file themselves and accept the spaces. For other tools, remove the spaces or export the key by hand. |
| A model response comes back with `content` set to `None` | The reasoning model spent `max_tokens` before answering. Turn thinking off with `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, or raise `max_tokens`. |

## Read the reference documents

RI #1's documents list every default and the measurements behind the measured ones. Read them when
a step below points to them.

| Document | What it covers |
| --- | --- |
| [`README.md`](README.md) | What RI #1 demonstrates, its corpora, and its headline results. |
| [`PIPELINE.md`](PIPELINE.md) | The five pipeline stages, every default, and the model calls per question. |
| [`THRESHOLDS.md`](THRESHOLDS.md) | The grading-threshold sweep on ContractNLI, with CUAD and MultiHop-RAG as checks. |
| [`CORPUS.md`](CORPUS.md) | Licence records, pins, and the credit lines that ship with any output quoting corpus text. |
| [`GRAPHRAG_NULL_ARM.md`](GRAPHRAG_NULL_ARM.md) | LightRAG against dense retrieval on questions with no answer in the corpus. |
| [`aieng-advanced-rag/README.md`](../../aieng-advanced-rag/README.md) | The harness API: scoring a retriever, the metrics, the answer judges, and the extras. |

## Where the code lives

Package paths are relative to `aieng-advanced-rag/aieng/advanced_rag/`. RI paths are relative to
this folder.

| Technique or tool | Code |
| --- | --- |
| Dense, lexical, and hybrid retrieval | `retrieval/retrievers.py` (`DenseRetriever`, `ShuffledRetriever`); `retrieval/sparse.py` (`BM25Retriever`, `HybridRetriever`) |
| Cross-encoder reranking | `retrieval/rerankers.py` (`CrossEncoderReranker`) |
| Chunkers | `retrieval/chunkers.py`; hierarchical splitting in `pipeline/hierarchical.py`; contextual retrieval in `retrieval/contextual.py` |
| Query rewriting | `pipeline/rewrite.py` |
| Evidence grading and the opt-in corrective-loop experiment | `pipeline/grading.py` |
| Generation and abstention | `pipeline/generate.py` |
| The pipeline from question to answer | `pipeline/spine.py` (`AdvancedRagSpine`) |
| Retrieval and null-detection metrics | `evaluation/metrics.py` |
| Answer judges | `evaluation/answers.py` (`judge_answer`, `judge_coverage`) |
| Threshold sweep | `evaluation/sweep.py`; RI runners `run_threshold_sweep.py`, `summarise_sweep.py`, and `sweep_lab/` |
| Corpus loaders | `corpora/cuad.py`, `corpora/contract_nli.py`, and `corpora/multihop_rag.py`, with pinning in `corpora/sources.py` |
| GraphRAG null arm | `graphrag_null/`, described in `GRAPHRAG_NULL_ARM.md` |
| Committed results | `outputs/` |

Every loader verifies a SHA-256 digest on fetch, so an upstream change fails the run and leaves
the measurements alone.

> **Note:** Notebooks 01 to 06 and 99 cover the participant path. Added answering notebooks
> replay recorded measurements by default; see [their provenance and limits](ANSWERING_EXPERIMENTS.md).
> The full-split threshold/answering harvest remains deferred. The shared layer is merged. One retrieval interface
> is the single surface for every retrieval method, in place of the
> package's three separate retrieval shapes, and it has an asynchronous twin on the same protocol.
> It takes the number of results and a preview length, and it supports expanding a hit and reading
> a range. For graph methods it carries typed advisory hints, such as keywords or seed entities, and
> a find-then-fetch shape that returns candidates first and reads them second. One answering
> interface covers a query-analysis stage, a loop policy, and a switch between a method's own prompt
> and the project's default prompt. One agent loop runs on Google's Agent Development Kit (ADK) 2.x,
> with its tools written as framework-neutral Python. One cost meter reports model calls, tokens,
> and both service seconds and elapsed seconds per question.

## Build RI #1 in order

Start with the baseline, then change one layer at a time. Retrieval, answering, and agent tool use
have different measures; the notebook order keeps those distinctions visible.

1. Set up the harness and score the baseline in
   [`01_harness_and_baseline.ipynb`](01_harness_and_baseline.ipynb). Load the corpus, run dense and BM25 retrieval, and report
   the retrieval-failure rate, mean reciprocal rank (MRR), and null detection. The committed dense
   baseline uses 20 CUAD contracts and their 250 answerable questions, with `build_slice`'s
   default 1,200-character windows and `bge-small-en-v1.5` at k=10, as in the package README's
   example. It has a retrieval-failure rate of 0.2400 and an MRR of 0.4348. That run left out the
   model's published query prompt, which `SentenceTransformerEmbedder` applies by default and
   which raises MRR to 0.4632, so pass `query_prompt=""` to reproduce it.
   Chunk size is a confound, because a fixed number of results rewards larger chunks, so compare
   chunkings at equal characters with `budget_k`.
   The notebook also loads ContractNLI and runs the recommended single-pass pipeline on four
   sample questions. Its expensive flag ships off: the default path reads measured artifacts
   through the shared cache without a model or key. Enabling it recomputes both dense arms and
   makes four generation calls using the shared model and credential settings.
   The samples demonstrate execution and citations; their null labels and predictions are shown
   separately, and they are not an answer-correctness benchmark.
2. Add hybrid retrieval and reranking in
   [`02_retrieval_levers.ipynb`](02_retrieval_levers.ipynb). Fuse BM25 and dense rankings with
   weighted reciprocal rank fusion, then reorder the candidates with a cross-encoder. On the full
   CUAD split (1,244 questions), the shipped dense weight of 0.2 fails on 0.2436 of questions,
   against 0.2629 at equal weight and 0.2548 for BM25 alone, so a single equal-weight trial would
   suggest hybrid search does not help. The notebook also finds that SPLADE, a learned sparse model,
   beats the tuned hybrid on its own: 0.2066 failure and 0.5687 MRR. On the CUAD 20 slice,
   `bge-reranker-v2-m3` at depth 20 lowers SPLADE's MRR from 0.6139 to 0.5951 and costs 24,584 ms
   per query against 109 ms for SPLADE alone. On the ContractNLI 20 slice, it raises MRR from
   0.8180 to 0.8502 over SPLADE alone. Which lever wins depends on the metric and the slice, and
   section 7 of the notebook names each winner.
3. Change the chunking in [`03_chunking_variants.ipynb`](03_chunking_variants.ipynb).
   Hierarchical chunking ranks small chunks and returns their larger parent sections to the reader.
   Contextual retrieval adds model-written context to every chunk, at one model call per chunk.
   The result depends on whether retrieval is scoped to one contract. Across all 123 ContractNLI contracts, with questions that name their
   contract, contextual chunks written with the head-and-neighbourhood prompt cut the failure rate at
   `k=10` from 0.6473 to 0.1187, for 1,418 calls and 2.45M prompt tokens. A free metadata header
   reaches 0.1936. Inside one contract, retrieval is scoped and no arm beats plain chunks by more than
   seven questions in 1,188 at the tight character budget (the text plain chunks return at `k=3`). On CUAD, scoped to one contract,
   contextual chunking raised the failure rate from 0.2400 to 0.2640 and cost 2.09M tokens. A label
   that names the contract cannot break a tie between chunks of that contract.
4. Measure rewriting, grading, and abstention in
   [`05_answering_and_null_detection.ipynb`](05_answering_and_null_detection.ipynb), as an optional contrast.
   The participant path uses `max_attempts=1`, `SingleShotPolicy`, a model-free `HeuristicGrader`,
   and no corrective rewriter. It retrieves once and lets the generator's `not_stated` response
   handle missing evidence. The opt-in loop rewrites, grades, retries, and then abstains. At its best setting it
   scores 0.788 balanced accuracy against 0.780 with no grader. A paired test cannot tell the 0.008
   gap from zero (exact McNemar p = 0.26 over 680 questions), and it costs about 9.2 times the model calls
   ([the threshold memo](THRESHOLDS.md)). Scoping the retriever matters more: an unscoped
   dense retriever answers 75.8% of the unanswerable questions against 13.0% scoped
   ([the null-arm memo](GRAPHRAG_NULL_ARM.md)), and the change is two lines in the retriever.
   The notebook also replays threshold choices without new calls, measures citation requirements
   separately from correctness, and includes the remote's retry-without-rewriting control.
5. Study free-form regulatory answers and evaluation in
   [`06_regulatory_qa_and_ragas.ipynb`](06_regulatory_qa_and_ragas.ipynb). Inspect the public FOIA
   source, generated references, supporting quotes, and null control before comparing RAGAS
   metrics. The local embedding model avoids hosted embedding calls; RAGAS judging is separately
   opt-in. Six answerable examples demonstrate the method, not regulatory reliability.
6. Build a starter agent in [`99_starter_agent.ipynb`](99_starter_agent.ipynb). Exercise the
   shared search/fetch tools without a model, then edit the factory or supply your own documents
   for an optional live ADK run. Search and fetch use the same document boundary; the call limit
   and unsuccessful saved run are part of the lesson.

Outside this order, [`04_bring_your_own_corpus.ipynb`](04_bring_your_own_corpus.ipynb) builds a
small evaluation set from a folder of your documents, then ranks the same eight retrieval arms on
it. The questions are generated, so the table it prints gives an order of arms and not absolute
scores. Read [the special topic](../special_topics/synthetic_question_sets/README.md) first for
when a generated set can be trusted.

`temperature=0` removes sampling variance and nothing more. A server that batches requests can
return different output for the same prompt, because the batch a request lands in changes the
order of floating-point sums
([Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
Retrieval that runs before any model call reproduces, so compare runs on that evidence, and treat
answers and model-rewritten queries as samples.

## Run the paid measurements

The threshold and null-arm runners call the proxy. Print the estimate before every run, and keep concurrency at 8 or
below, because proxy throughput collapses above eight concurrent requests
([the threshold memo](THRESHOLDS.md)).

The added answering experiments also have opt-in live runs; their default is reference replay.
See [ANSWERING_EXPERIMENTS.md](ANSWERING_EXPERIMENTS.md) for their flags, report provenance,
and private variant outputs. The full-123 paid harvest remains deferred.

| Run | Estimate first | What it costs |
| --- | --- | --- |
| Threshold sweep | `uv run python implementations/regulatory_policy_qa/run_threshold_sweep.py --corpus contract-nli --documents 40 --estimate-only` | The first ContractNLI harvest made 9,232 network calls. A full-split harvest is about 31,000 calls, roughly two hours ([the threshold memo](THRESHOLDS.md)). |
| GraphRAG null arm | `uv run python implementations/regulatory_policy_qa/graphrag_null/run_null_arm.py --documents 20 --estimate-only` | Indexing 20 contracts with LightRAG cost 1,177,351 tokens ([the null-arm memo](GRAPHRAG_NULL_ARM.md)). |

To run either one, repeat its command without `--estimate-only` and add `--workers 8`. The sweep
defaults to 24 workers and also needs `--yes` to start above its call limit. The null arm defaults
to eight workers under a limit of 90 requests per minute. It skips indexing and retrieval when its
store already holds their outputs, so a re-run with an existing store re-scores without
re-indexing.

For an exported null-arm bundle, verify it, unpack it, and replay from the
repository root:

```bash
python implementations/regulatory_policy_qa/graphrag_null/export_store.py --verify null-arm-store.tar.gz
mkdir -p unpacked
tar -xzf null-arm-store.tar.gz -C unpacked
python implementations/regulatory_policy_qa/graphrag_null/run_null_arm.py --store ./unpacked
```

The bundle automatically disables indexing and retrieval. The runner uses all
five arms and the two constant controls at 0.6/0.0 with Qwen3_6-35B-A3B, and
writes to `outputs/graphrag_null/measured_thresholds/`. Cache misses make proxy
calls. Answer correctness and grounding use the separate judging command in
[the null-arm memo](GRAPHRAG_NULL_ARM.md#answer-correctness-on-the-answerable-half).

## Trace a run with Langfuse

Langfuse records what a run called, in what order, and what each call cost. The `tracing` extra
installs it, and `uv sync --all-packages --all-extras --dev` already has.

Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and, outside the default region, `LANGFUSE_HOST`
to `.env` at the repository root, as [`.env.example`](../../.env.example) lists them. Then wire the
process up in the first cell:

```python
from aieng.advanced_rag.tracing import init_langfuse_tracing

init_langfuse_tracing()
```

Wrap the expensive cell, so the sweep's calls group under one trace:

```python
from aieng.advanced_rag.tracing import langfuse_trace

with langfuse_trace("threshold-sweep", tags=["regulatory-policy-qa"]) as trace:
    answers = [loop.run(question) for question in questions]

print(trace.trace_url)
```

Both calls do nothing without the two keys, and `trace.trace_url` is then `None`. A cell that
calls them stays runnable, and commits the same output, for a participant with no Langfuse
project.

## Follow the conventions

- Embed locally. RI #1 uses `BAAI/bge-small-en-v1.5`, and `tests/test_no_paid_embeddings.py` fails
  if the sweep code calls a hosted embedding API or names any other embedding model. The proxy
  hosts no embedding model, so an embedding call through it bills an external provider.
- Turn thinking off for bulk calls with `chat_template_kwargs={"enable_thinking": False}`, or wrap
  the client with `disable_thinking` from `aieng.advanced_rag.reasoning`. The proxy ignores
  `reasoning_effort`.
- Carry the credit lines from [`CORPUS.md`](CORPUS.md) in any output that quotes corpus text.
- Put expensive notebook cells behind a flag that defaults to off, and commit their outputs.

## Models and credentials for live experiments

RI1 uses the shared `resolve_api_key()` and `resolve_base_url()` helpers. Onboarding
exports `OPENAI_API_KEY` together with `OPENAI_BASE_URL` for the Vector proxy;
`VECTOR_PROXY_API_KEY` remains a supported environment or repository `.env` alias.
Environment values take precedence. The OpenAI SDK is an OpenAI-compatible client
for the proxy, including when using another provider's models.

New live experiments use `choose_models()` and the standard
`ADVANCED_RAG_MODEL_FAMILY`, `ADVANCED_RAG_GENERATION_MODEL`,
`ADVANCED_RAG_EXTRACTION_MODEL`, and `ADVANCED_RAG_JUDGE_MODEL` settings.
Generation answers questions, extraction writes search queries and contexts, and
judging grades evidence and answers. Embeddings remain local. The shared
[model and spend guide](../../docs/models-and-spend.md) lists family defaults,
request settings, and `ADVANCED_RAG_SPEND_LIMIT_USD`.

Saved reports and cache-only replay retain their original model identities.
Changing a family or role requires a fresh matched comparison and a separate
output/cache location. The fixed GraphRAG indexing configuration describes the
reference indexes; choosing a new reader does not relabel their producer.
