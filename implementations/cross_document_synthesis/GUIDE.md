# Explore cross-document synthesis

A multi-hop question can fail because retrieval misses a bridge, the evidence
budget excludes a useful passage, or the reader combines the sources incorrectly.
RI #2 lets you inspect these failures while changing the retrieval policy and
keeping the corpus, encoder, and final answerer fixed.

Start with [the results and notebook map](README.md). The
[method reference](docs/METHODS.md) explains implementation settings;
[EXPERIMENTS.md](docs/EXPERIMENTS.md) covers commands, replay, and result contracts.

## Work through the methods

1. **Decomposition:** use [notebook 01](01_query_decomposition.ipynb) to split an
   invented question, inspect pooled evidence, and compare independent search
   questions with the original query. Change the reader budget or pooling order
   separately. Gold hop graphs help explain an error after answering; they must
   not guide a held-out search.
2. **Agent tools:** use [notebook 02](02_agentic_retrieval.ipynb) to call search and
   read by hand before inspecting recorded sessions. Follow the exact literal
   query and exposed passage IDs. Explain an abstention in terms of the evidence
   the agent actually saw.
3. **IRCoT:** use [notebook 03](03_ircot.ipynb) to follow a discovered entity into
   a second search. Change the retrieval-round limit and inspect enforced
   retrieval/update order. Brief facts guide retrieval; original passages remain
   the final reader's evidence.
4. **Graph retrieval:** use [notebook 04](04_graphrag_head_to_head.ipynb) to remove
   a bridge or change a seed in a traversable toy graph. The final reader stays
   fixed. The example isolates graph connectivity and evidence selection; its
   synthetic scores are not estimates of MuSiQue performance.
5. **Method selection:** use [notebook 05](05_decision_grid.ipynb) to compare
   quality, evidence coverage, failures, uncertainty, and calls across MuSiQue hop
   counts and the NaturalQuestions control. Read optional correction against
   both dense and graded single-pass controls.
6. **Your agent:** use [notebook 99](99_starter_agent.ipynb) to edit a retrieval
   policy, test a counterexample, and check state isolation. A separate live path
   applies editable instruction and budget settings to the shared ADK runtime.

Before editing a control, predict which passages will reach the reader.
Afterwards, explain the difference using the trace and score rather than
the answer alone.

## Read the measurements

The three core methods each have 1,000 outcomes on MuSiQue and 1,000 on
NaturalQuestions. They use pinned local BGE embeddings and the same five-passage
answer template. MuSiQue results are split by two, three, and four hops.
NaturalQuestions is a single-hop control even when several alternative contexts
are marked supporting.

Use the [full matched comparisons](docs/COMPARISONS.md) for quality claims. The
balanced 30-question decomposition/smoke examples and three-question agent lab
have different selections and serve different purposes. Invented fixtures teach
mechanics and do not supply benchmark evidence.

A report distinguishes completed coverage from successful execution. Failed
questions receive zero credit and stay in the denominator. Supporting recall,
complete support coverage, and coverage of the actual generation evidence answer
different questions. Citation validity checks source identity rather than
semantic support. The [comparison notes](docs/COMPARISONS.md#read-the-comparison)
explain the limits of paired quality and cost measurements.

Compare calls, tokens, tool usage, and service/wall time alongside quality.
The shared meter reports live usage and cache hits separately.

### Models, credentials, and spend

All live chat calls use the Vector proxy, including OpenAI-family models. An
OpenAI-compatible SDK does not imply use of OpenAI's direct API. Use the
credentials exported by onboarding (`OPENAI_API_KEY` together with
`OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1`). The shared resolver also
accepts `VECTOR_PROXY_API_KEY` in the environment or repository `.env`.
Environment values take precedence; never put keys in notebooks.

Live runners use the shared `choose_models()` settings: the default Google
family selects generation and extraction models, while grading uses the shared
judge. Set `ADVANCED_RAG_MODEL_FAMILY=onprem` for Qwen generation and extraction;
the on-prem set also uses Qwen for judging. Paid families use the shared judge.
Individual overrides are
`ADVANCED_RAG_GENERATION_MODEL`, `ADVANCED_RAG_EXTRACTION_MODEL`, and
`ADVANCED_RAG_JUDGE_MODEL`. See the repository's
[model and spend guide](../../docs/models-and-spend.md) for current model names
and prices. Embeddings remain the pinned local BGE encoder for every family.

Before optional notebook calls, `choose_models(show=True)` displays the selected
roles and budget. Live command-line runners display this information and print
`spend.report()` on exit. Set `ADVANCED_RAG_SPEND_LIMIT_USD` to bound additional
live spending. A budget stop leaves benchmark checkpoints incomplete and can be
resumed after adjusting the budget. Cached responses are reported separately.

Recorded tables describe the original Qwen reference runs. Changing models
requires fresh, matched runs for every compared method, a new output path, and
an isolated response-cache namespace. Model roles are recorded in each new
report and checked during comparison and resumption.

## Execute an experiment

Default notebooks require Python 3.12+, NumPy, `qdrant-client==1.19.1`, a Jupyter kernel,
and RI #2's reference artifacts, with no credentials, data downloads, model
weights, or graph indexes. On a Slurm cluster, start the kernel inside a compute
allocation, and run all Python, tests, embeddings, and benchmarks there.

Every model step in notebooks 01 to 04 and 99 goes through `load_or_run`. With
`RUN_EXPENSIVE = False` it replays the reference built with `Qwen3_6-35B-A3B` and
prints what a rebuild on your model would cost; with `RUN_EXPENSIVE = True` it
runs on your model and caches the result. The references hold MuSiQue question
text and model answers, so they ship as release `reference-ri2-35b43b26a35a`
rather than in the repository. `scripts/setup.sh` installs them on each workspace
start. To install them yourself, from the repository root:

```bash
python scripts/fetch_reference_artifacts.py implementations/cross_document_synthesis/reference-artifacts.json
```

Live execution additionally needs the bootcamp dependencies, local encoder
weights, and Vector proxy credentials through the shared resolver. Embeddings
stay local; model selection and spending follow the shared settings below.
Expensive cells default to off and write to ignored experiment paths.

From the repository root inside compute:

```bash
export PYTHONPATH=aieng-advanced-rag
python implementations/cross_document_synthesis/run_benchmark.py \
  --live --corpus musique --method decomposition --per-category 10 \
  --output 'Arm Store/ri2-experiments/decomposition.json'
```

Use `--per-category 0` for all questions. Use a new output path when changing a
method configuration; incompatible reports cannot resume. See
[EXPERIMENTS.md](docs/EXPERIMENTS.md) for isolated response caches and full replay.

## Code and data boundaries

| Component | Where to inspect |
| --- | --- |
| Method adapters and prompts | `synthesis_lab/decomposition.py`, `ircot.py`, `methods.py` |
| Framework-neutral tools | `synthesis_lab/agent_lab.py`, `ircot.py` |
| Local embeddings, proxy, vectors | `synthesis_lab/runtime.py`, `vectors.py` |
| Corpus annotations, aliases, rewrite pairs | `synthesis_lab/corpora.py` |
| Scores, paired intervals, decision grid | `synthesis_lab/evaluation.py`, `decision_grid.py` |
| Answering and corrective interfaces | Shared `pipeline/answering.py`, `spine.py`, `grading.py`, `rewrite.py` |
| ADK and shared usage | Shared `agents/` and `metering.py` |
| Pinned corpus loaders | Shared `corpora/musique.py`, `natural_questions.py`, `multihop_articles.py` |
| Reference replay of notebook model steps | `synthesis_lab/reference.py`, `build_reference.py` |

Shared paths are relative to `aieng-advanced-rag/aieng/advanced_rag/`. RI2 reuses
RI1's extracted shared answering components; it does not require importing the
RI1 implementation folder.

[CORPUS.md](CORPUS.md) records source pins, attribution, and rights boundaries.

### Compare the prebuilt graph indexes

Both graph arms retrieve original MuSiQue passages and use RI2's shared final
answer template with a five-passage budget. HippoRAG 2 uses **atlas-rag's
entity/relation variant**, with concepts and events disabled. LightRAG uses
`mix` mode with a retrieval depth of 40 before deduplication and selection of
20 original passages. These settings describe different retrieval methods;
the final reader and its evidence budget remain fixed.

RI #4's [release manifest](../graphrag_technical_support/reference-artifacts.json)
pins both artifact envelopes and payloads, and RI #2 reads the pins from it. The loader verifies their SHA-256 digests, corpus revision,
complete passage mapping, and precomputed stores before querying. It uses a
writable copy of each index, leaving the verified cache entry unchanged.
Query embeddings use RI2's pinned local encoder revision. The producer cache
key records the encoder model name; it does not record its embedding revision.
Graph construction cost and query-time usage are separate measurements.

Run the following from the repository root. To install from a local copy of the
release, download it with the GitHub CLI:

```bash
RI4_INDEX_DIR="$HOME/ri4-indexes"
mkdir -p "$RI4_INDEX_DIR"
gh release download reference-ri4-bd05f7bb16c6 \
  --repo VectorInstitute/advanced-RAG --dir "$RI4_INDEX_DIR"
sed "s#https://github.com/VectorInstitute/advanced-RAG/releases/download/reference-ri4-bd05f7bb16c6/#file://$RI4_INDEX_DIR/#g" \
  implementations/graphrag_technical_support/reference-artifacts.json \
  > "$RI4_INDEX_DIR/local-manifest.json"
python -m aieng.advanced_rag.cache "$RI4_INDEX_DIR/local-manifest.json"
```

Then query either installed index. Start with one question per hop category;
set `--per-hop 0` to evaluate the full 1,000-question split. Use a different
output path when changing settings; the runner rejects incompatible resumption.

```bash
python implementations/cross_document_synthesis/run_graph_comparison.py \
  --live --backend hipporag2 --per-hop 1 \
  --output 'Arm Store/ri2-graph/hipporag2-smoke.json'
python implementations/cross_document_synthesis/run_graph_comparison.py \
  --live --backend lightrag --per-hop 1 \
  --output 'Arm Store/ri2-graph/lightrag-smoke.json'
```

Each saved report includes the release identity, producer settings, actual
implementation, per-question retrieved and generation passage IDs, citations,
failures, stop reasons, and metered query usage. Replay it with
`run_evaluation.py` before interpreting a method difference. Compare identical
question IDs and keep failed questions in the denominator.


Run Python and live cells inside a requested compute allocation.
