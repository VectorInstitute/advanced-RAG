# Run the notebooks on data that stays off the platform

When your documents can't leave your organization's environment, try the methods on public
datasets on the bootcamp platform, then run the same notebooks on your own machine, against your
own model endpoint.

The package and notebooks are public, embeddings run locally, and every chat call goes to whatever
OpenAI-compatible endpoint you name, so your documents and questions reach only the machines and
endpoint you choose.

Support outside the platform is limited. On a Coder workspace, facilitators can join your session
with Live Share and look at the problem with you. In your own environment they can't, so they help
from what you send them: the error message, the cell that raised it, your settings with keys
removed, and whether the same notebook runs on a public dataset. Facilitators still help where
they can.

## Prerequisites

- A bootcamp Coder workspace, for the first step.
- For your own environment: Python 3.12, [uv](https://docs.astral.sh/uv/), git, and about 10 GB of
  free disk for the virtual environment, the core data, and the embedding model.
- An OpenAI-compatible chat endpoint and a key for it, with at least one model your organization
  allows for your data.

## Choose a reference implementation and a public dataset

Work through the reference implementation (RI) closest to your use case on its shipped data first,
on the platform, where facilitators can see what you see. Each RI's `CORPUS.md` or `README.md`
records its datasets and licences.

| Your use case | Start with | Shipped data to try it on |
| --- | --- | --- |
| Questions over policies, contracts, or regulations, where "not stated" must be a valid answer | RI #1, [`regulatory_policy_qa/`](../implementations/regulatory_policy_qa/) | CUAD and ContractNLI contracts |
| Questions whose answer combines facts from several documents | RI #2, [`cross_document_synthesis/`](../implementations/cross_document_synthesis/) | MuSiQue multi-hop questions, with NaturalQuestions as a single-hop control |
| Answers that sit in tables in reports or filings | RI #3, [`table_structured_reasoning/`](../implementations/table_structured_reasoning/) | FinanceBench filings and questions |
| Support tickets, forums, or other material where entities and their links matter | RI #4, [`graphrag_technical_support/`](../implementations/graphrag_technical_support/) | ServerFault posts, and MuSiQue for the retrieval comparison |
| An agent that searches files directly, with no vector index | RI #5, [`agent_native_retrieval/`](../implementations/agent_native_retrieval/) | MuSiQue passages, ServerFault, FinanceBench, and a code repository |
| A corpus with no labelled questions | RI #1 notebooks [`04`](../implementations/regulatory_policy_qa/04_bring_your_own_corpus.ipynb) and [`07`](../implementations/regulatory_policy_qa/07_levers_and_answers_on_your_corpus.ipynb), and the [`synthetic_question_sets`](../implementations/special_topics/synthetic_question_sets/) special topic | The ten ContractNLI contracts the notebooks use as a stand-in |

Notebooks 04 and 07 are the path most teams take to their own data: notebook 04 generates a
question set with known answers and null questions from a folder of documents, and notebook 07
scores the retrieval levers and answering with abstention on that set. Run both on the stand-in
first, so you know what their output looks like before your data goes in.

## Install the package in your own environment

Run every command from a terminal on the machine that can read your data. None needs root
privileges.

1. Clone the public repository:

   ```bash
   $ git clone https://github.com/VectorInstitute/advanced-RAG.git
   ```

2. Change into it:

   ```bash
   $ cd advanced-RAG
   ```

3. Install the dependencies, every extra, the development tools, the Jupyter kernel, and the core
   data, without the platform's key loading or Jupyter server:

   ```bash
   $ bash scripts/setup.sh --no-onboard --no-jupyter
   ```

   The script runs `uv sync --frozen --all-packages --all-extras --dev` and downloads about 165 MB
   of data, including the `BAAI/bge-small-en-v1.5` embedding model. On a machine with no internet
   access, run it where you have access and copy the repository folder, `~/.cache/aieng-advanced-rag`,
   and the Hugging Face cache across.

4. Copy the example environment file:

   ```bash
   $ cp .env.example .env
   ```

[Set up on your own machine](setup.md#set-up-on-your-own-machine) has the same steps one command
at a time.

## Point the package at your endpoint

The package reads its settings from the environment first and from `.env` at the repository root
second. Set these in `.env`:

```bash
OPENAI_BASE_URL=<endpoint_url>
VECTOR_PROXY_API_KEY=<endpoint_key>
ADVANCED_RAG_GENERATION_MODEL=<generation_model>
ADVANCED_RAG_JUDGE_MODEL=<judge_model>
ADVANCED_RAG_EXTRACTION_MODEL=<extraction_model>
```

where:

- `<endpoint_url>` is your endpoint's base URL, ending in `/v1` for most servers.
- `<endpoint_key>` is the key the endpoint expects. The package reads it under this name for any
  endpoint. An `OPENAI_API_KEY` exported in your shell is ignored unless `OPENAI_BASE_URL` points
  at the Vector proxy, so a personal OpenAI key never goes to the wrong place.
- `<generation_model>`, `<judge_model>`, and `<extraction_model>` are model names as your endpoint
  serves them. Set all three: the default judge is a model on the Vector proxy, and a run on your
  endpoint fails on the first judging call without this setting. The same model can fill every
  role.

`choose_models(show=True)` at the top of each notebook prints the models it picked. A name outside
the package's default sets prints an `UnlistedModelWarning`, which is expected here. The package
sends such a model the request shape a vLLM server takes, including vLLM's switch that turns
reasoning off.

Embeddings stay local whatever you set: retrieval embeds with `BAAI/bge-small-en-v1.5` on your CPU,
and no embedding request goes to your endpoint or anywhere else.

### Replay and the spend guard on your endpoint

With `RUN_EXPENSIVE = False`, a notebook replays the reference results built on Vector's on-prem
models and makes no model call, so it runs with no endpoint at all. RIs whose results are too large
to commit fetch them once:

```bash
$ python scripts/fetch_reference_artifacts.py
```

With `RUN_EXPENSIVE = True`, a step runs on your endpoint and stores its result in your local
cache, `~/.cache/aieng-advanced-rag` (`ADVANCED_RAG_CACHE_DIR` moves it), so the next run reads it
back without a call. Results built from your data stay in that cache and are never uploaded.

The spend guard prices calls from the Vector proxy's price list and the package's own table. A
model on your endpoint is in neither, so:

- the estimate a step prints before it runs reads `price unknown` or `no estimate`;
- the confirmation stop, which compares an estimate with your budget, never fires; and
- `ADVANCED_RAG_SPEND_LIMIT_USD` counts your calls at $0 and never stops a run.

`spend.report()` prints call and token counts but no dollars and no budget. Set a spending limit
on the endpoint itself, and read each step's call count before you set `RUN_EXPENSIVE = True`.

## Run the notebooks on your data

1. Start Jupyter from the repository root:

   ```bash
   $ uv run jupyter lab
   ```

2. Open `implementations/getting_started/00_environment_check.ipynb`, select the **Advanced RAG
   Bootcamp** kernel, and run every cell. Off the platform the Langfuse check reports `WARN`, and
   every other check must pass. With its `RUN_EXPENSIVE = True`, the notebook also sends one
   completion, to `Qwen3_6-35B-A3B` unless you export `ADVANCED_RAG_CHECK_MODEL=<generation_model>`
   in the shell before you start Jupyter.
3. Put your documents in a folder outside the repository, as `.txt`, `.md`, or `.pdf` files.
   Convert web pages to Markdown first.
4. In `implementations/regulatory_policy_qa/04_bring_your_own_corpus.ipynb`, set `FOLDER` to that
   folder, set `RUN_EXPENSIVE = True`, and run the notebook to build the question set.
5. In `07_levers_and_answers_on_your_corpus.ipynb`, set the same `FOLDER`, `CONFIG_FILE`, and
   `MAX_DOCUMENTS`, set `RUN_EXPENSIVE = True`, and run it to score the retrieval levers and the
   answering.

Other RIs take your data the same way: each RI's README names the cell or command that reads a
corpus. Executed notebooks hold questions and answers drawn from your documents, so keep them out
of any repository or channel your data may not reach.

## Verify the setup

- `choose_models(show=True)` lists your three model names.
- Notebook 04 on the stand-in, with `RUN_EXPENSIVE = False`, prints the same table as the committed
  notebook.
- A first live step prints its call count before it starts, and a second run of the same cell
  reports its results as cached.

## Troubleshoot

| Symptom | Cause | Fix |
| --- | --- | --- |
| `MissingProxyKeyError` | No key in the environment or `.env`. | Set `VECTOR_PROXY_API_KEY` in `.env` at the repository root. |
| `AuthenticationError` (HTTP 401) on the first call | The endpoint rejected the key, or `OPENAI_BASE_URL` points at the wrong server. | Check both values in `.env` for stray spaces or quotes. |
| `NotFoundError` (HTTP 404) naming a model | The endpoint does not serve that name. | Set the three model variables to names your endpoint lists. |
| HTTP 400 naming `chat_template_kwargs` | The endpoint is not a vLLM server and refuses vLLM's reasoning switch. | Serve the model through vLLM, or raise it with the facilitators with the full error. |
| `FAIL BAAI/bge-small-en-v1.5` in the environment check | The embedding model is not in the Hugging Face cache. | Run `python -m aieng.advanced_rag.corpora.download bge-small`, or copy the cache from a machine with access. |
| A judging cell raises an error naming `gemini-3.8-flash` | `ADVANCED_RAG_JUDGE_MODEL` is not set, so the default judge on the Vector proxy was called. | Set it to a model on your endpoint. |
