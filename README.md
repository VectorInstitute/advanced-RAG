# Advanced RAG Bootcamp

Advanced RAG extends a foundational RAG pipeline into
enterprise retrieval: complex and multimodal documents, tables, multi-source
queries, changing corpora, and the question of *when* graph structure is
warranted. A baseline pipeline gets additive layers rather than a rewrite, and each
method choice is weighed against retrieval quality, grounding, cost, latency, and
engineering effort.

The repository pairs a shared library (`aieng-advanced-rag`) with self-contained
**reference implementations**, each a worked example of a distinct enterprise RAG
problem and the techniques that fit it. Enter at whichever one is closest to the
target system.

The repository is the course material of Vector Institute's Advanced RAG Bootcamp, published for
anyone who builds or studies retrieval systems: engineers taking a RAG pipeline past the baseline,
and researchers comparing Advanced RAG with GraphRAG on shared data. You don't need a bootcamp
account or a model key to follow it. Every notebook ships with `RUN_EXPENSIVE = False`, and in that
mode it replays the committed results instead of calling a model.
[Run without bootcamp keys](#run-without-bootcamp-keys) explains how, and how to point the
notebooks at your own endpoint.

First time here? Run
[`implementations/getting_started/00_environment_check.ipynb`](implementations/getting_started/00_environment_check.ipynb)
before anything else. It confirms that the dependencies, keys, embedding model, and data are in
place, and names the fix for anything that is not. [`docs/setup.md`](docs/setup.md) covers setup.

## Teaching model: one spine, two circles

The bootcamp is organized along one axis, **RAG becoming agentic**. RAG basics
are assumed.

> **Baseline RAG** → **Advanced RAG** (hybrid, rerank, contextual/hierarchical
> chunking, query rewriting) → **Agentic RAG** (ReAct loops, decomposition,
> self-correction, reflexion) → **Agent-native / embedding-less retrieval** (the
> agent *is* the retriever: grep/navigate/tool-call, Docling-agent- and
> coding-agent-style, no vector index in the loop).

Two overlapping circles, **Retrieval** (the "R") and **Agent** (the loop
reasoning over it). Along the spine the Agent circle grows and slides over the
Retrieval circle until, at the terminal, the two coincide and the embedding index
disappears. Quality moves by turning either circle. A richer retrieval substrate
with a simple agent (KG-RAG + basic ReAct), or a plain substrate with a deep
agentic loop, can reach comparable quality, so the portfolio holds one constant
and varies the other to read the cost, latency, and complexity trade-off
directly.

The **Advanced-RAG-vs-GraphRAG decision** is applied at each stage of the spine.
GraphRAG stacks on the Advanced RAG spine, warranted by specific capability gaps
(entity aggregation, explicit reasoning paths, corpus-scale summarization,
path-finding multi-hop). Everything sits downstream of a shared **evaluation
harness**.

## Reference implementations

Each reference implementation (RI) is independent and self-contained; see its `README.md` for the full
walkthrough. Together they span the spine from a conventional baseline to
agent-native, embedding-less retrieval, and each pairs an Advanced RAG method
with a GraphRAG contrast. Each one's `CORPUS.md` or `README.md` records its datasets, their
licences, and their pinned versions.

| # | Implementation | Spine position | The problem |
| --- | --- | --- | --- |
| 1 | [`regulatory_policy_qa/`](implementations/regulatory_policy_qa/) | Baseline → Advanced RAG | The Advanced RAG spine and shared evaluation harness, with reliable "not stated" null-answer detection on dynamic regulatory content. |
| 2 | [`cross_document_synthesis/`](implementations/cross_document_synthesis/) | Advanced → Agentic RAG | **Flagship.** Multi-hop reasoning across many documents, growing the agent loop, with the Advanced-RAG-vs-GraphRAG decision resolved on measured results. |
| 3 | [`table_structured_reasoning/`](implementations/table_structured_reasoning/) | Advanced RAG (substrate) | Retrieval and answering over tables and structured data (Docling section folders, table-aware chunking), grounded on financial filings. |
| 4 | [`graphrag_technical_support/`](implementations/graphrag_technical_support/) | Agentic RAG over a graph | The full KG-RAG track: graph construction, hybrid vector+graph retrieval with provenance, agentic traversal, and graph maintenance. |
| 5 | [`agent_native_retrieval/`](implementations/agent_native_retrieval/) | Agent ∩ Retrieval overlap | The spine terminal: embedding-less, agent-native retrieval (grep/navigate/Docling-agent), benchmarked against the vector baseline on the same corpora. |

## Design notes

Every RI starts from the same baseline pipeline and evaluation harness, and each advanced technique
is an added layer on top of it. Techniques are ordered by expected return: hybrid search, reranking,
and hierarchical chunking come before contextual retrieval, then adaptive routing, then agents, then
graphs. Each RI holds either the retrieval substrate or the agent loop fixed and varies the other,
so a change in cost, latency, or quality traces to one circle. Each RI also shows where its
technique loses, and ends with a starter-agent notebook that moves the use case one stage further
along the spine. Datasets are public and come with questions and answers, so lab time goes to
retrieval rather than data preparation.

## Repository structure

```text
aieng-advanced-rag/  # Shared library for the bootcamp
implementations/     # Environment check, self-contained reference implementations, and special topics
docs/                # Setup guide and additional documentation
scripts/             # Workspace setup and data scripts
tests/               # Onboarding gate that the platform runs (not run in CI)
```

`implementations/special_topics/` holds short, standalone studies that apply to more than one RI:
`citations` (scoring cited answers), `synthetic_question_sets` (building an evaluation set from an
unlabelled corpus), and `visual_documents` (parsing, enriching, or embedding the page).

## Getting started

On a Coder workspace, `scripts/setup.sh` runs on every start. It installs the dependencies, loads
your bootcamp keys into the shell, fetches the core data, and starts Jupyter. On your own machine,
follow [Set up on your own machine](docs/setup.md#set-up-on-your-own-machine).

1. Run [`00_environment_check.ipynb`](implementations/getting_started/00_environment_check.ipynb)
   and fix anything it reports.
2. Begin with [`regulatory_policy_qa/`](implementations/regulatory_policy_qa/) to learn the
   pipeline and evaluation harness.
3. Fetch the data for the RI closest to your use case, then move to it:
   `python -m aieng.advanced_rag.corpora.download <ri_number>`.

Bootcamp keys (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LANGFUSE_*`) live in your shell
environment. Personal keys go in `.env`, as [`.env.example`](.env.example) lists.
[`docs/setup.md`](docs/setup.md) explains the credential model and has a troubleshooting table.

### Run without bootcamp keys

You can read and run every notebook without a key. Each notebook ships with `RUN_EXPENSIVE =
False`, and in that mode it replays the committed results in the RI's `outputs/` folder through
the shared cache instead of calling a model. RIs whose results are too large to commit name a
manifest, `reference-artifacts.json`, that points at files on this repository's GitHub releases.
`python -m aieng.advanced_rag.cache <manifest>` downloads them and checks each digest, and
[Install the reference artifacts](docs/setup.md#install-the-reference-artifacts) has the details.
Embeddings are computed locally, so retrieval runs without a key too.

To make live calls with your own provider key, point the package at any OpenAI-compatible
endpoint. In `.env`, set `OPENAI_BASE_URL` to the endpoint and `VECTOR_PROXY_API_KEY` to your key.
The package reads both through `aieng.advanced_rag.proxy`, and an `OPENAI_API_KEY` exported in the
shell is ignored unless `OPENAI_BASE_URL` points at the Vector proxy, so that an OpenAI key never
reaches the proxy by mistake. Then name models your endpoint serves, either with
`choose_models(generation=..., judging=..., extraction=...)` or with the variables
`ADVANCED_RAG_GENERATION_MODEL`, `ADVANCED_RAG_JUDGE_MODEL`, and `ADVANCED_RAG_EXTRACTION_MODEL`.
`choose_models` accepts a model outside its default sets with a warning. The default sets use the
Vector proxy's model names, and the paid families share `gemini-3.8-flash` as the judge, so scores
you produce with another judge are comparable only with each other. The RI #4 graph build scripts
read `LLM_BASE_URL` and `LLM_API_KEY` instead.

## License
This project is licensed under the terms of the [LICENSE](LICENSE.md) file.

## Contribution
See our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## Contact Information
For more information, contact the Vector AI Engineering team at
advanced-rag@vectorinstitute.ai. Report a security vulnerability privately, as
[SECURITY.md](SECURITY.md) describes.
