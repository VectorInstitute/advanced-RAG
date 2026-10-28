# Set up your workspace

Get a working environment for the Advanced RAG reference implementations, on a Coder workspace or
your own machine, and confirm it with the environment check before you run anything else.

## Start on a Coder workspace

The workspace template clones this repository and runs `bash scripts/setup.sh` from the repository
root on every start. By the time you connect, that script has:

- Installed every dependency from `uv.lock` into `.venv`, with all the extras the reference
  implementations import, and PyTorch's CPU build.
- Installed the **Advanced RAG Bootcamp** Jupyter kernel.
- Loaded your bootcamp keys with the platform's `onboard` command and run the onboarding gate,
  `tests/test_integration.py`, which records you as onboarded when it passes.
- Fetched the core corpora and the `BAAI/bge-small-en-v1.5` embedding model, each checked against
  its pinned SHA-256.
- Looked for reference-artifact manifests, as
  [Install the reference artifacts](#install-the-reference-artifacts) describes, and skipped the
  step because none exist.
- Started Jupyter Lab on port 8888 in the background.

The first start installs about 2.6 GB of packages and downloads about 165 MB of data. A later start takes seconds, because each step finds its
work already done.

To confirm the start worked:

1. Open [`implementations/getting_started/00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb).
2. Select the **Advanced RAG Bootcamp** kernel.
3. Run every cell from the top.

The last cell prints how many checks passed. Each `WARN` or `FAIL` line ends with its fix, and the
[troubleshooting table](#troubleshoot-the-setup) below covers the rest.

## Workspace requirements

The Coder template and the workspace image provide these. On your own machine, you need the same
tools on your `PATH`.

| Requirement | Needed by | Where it comes from |
| --- | --- | --- |
| `ripgrep` (`rg`) | RI #5's starter and code corpora, which the agent's search tool reads with `rg`. The MuSiQue notebooks search with Python's `re` and do not need it. | The `Dockerfile` installs it. On your own machine, install `ripgrep` with apt or Homebrew. |
| `p7zip-full` (`7z`) | RI #4's ServerFault build, which extracts an 860 MB Internet Archive dump. | The `Dockerfile` installs it. An image built before it was added needs a rebuild. |
| A Qdrant server, version 1.19 | Only the vector compression and HNSW demos, because in-process Qdrant has no HNSW graph and no quantization. Every RI runs Qdrant in process and needs no server. | Optional. Run the Qdrant server binary or container next to the workspace and set `QDRANT_URL` to its address. |
| More than 10 GB of disk under `/home/coder` | The venv (about 2.6 GB with the CPU build of PyTorch), the uv cache, the core data (about 165 MB), and the RI #3 and #4 corpora (159 MB of filings, and the 860 MB dump with its extracted slice). | The Coder workspace's home disk. Size it at 20 GB or more. |
| `GITHUB_USER` | `scripts/setup.sh`, which runs `onboard` only when `GITHUB_USER` or `GH_USER` is set. `onboard` looks up the team's keys by that GitHub handle. | The Coder template exports it as the workspace owner's Coder username, which GitHub sign-in makes their GitHub login. Without it, the start skips `onboard` and no keys load. |

## How keys reach your code

Bootcamp keys live in your shell environment, and nothing writes them to disk. `onboard` reads your
team's keys from the platform and prints `export` lines, and `scripts/setup.sh` evaluates them. The
Jupyter server that `setup.sh` starts inherits them, and a block that `setup.sh` adds to `~/.bashrc`,
`~/.zshrc`, and `~/.profile` loads them into each new terminal, VS Code session, and Jupyter
kernel.

The keys `onboard` provides:

- `OPENAI_API_KEY` and `OPENAI_BASE_URL`: your team's key for the Vector proxy, and its address.
  The proxy serves every model provider through one OpenAI-compatible endpoint.
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`: your team's Langfuse project,
  for tracing, when the bootcamp provisions one.

Every runner and notebook reads the proxy key through `aieng.advanced_rag.proxy`, which tries
`OPENAI_API_KEY` and then `VECTOR_PROXY_API_KEY`, in the environment first and the repository
`.env` second. An `OPENAI_API_KEY` exported without an `OPENAI_BASE_URL` on the proxy's host is
skipped, because it may be a key for OpenAI's own API and must not reach the proxy.
`VECTOR_PROXY_API_KEY` is the name to use in `.env` on your own machine.

Personal keys that the platform does not provide go in `.env` at the repository root, which git
ignores. [`.env.example`](../.env.example) lists the names. A value exported in the shell always
wins over the same name in `.env`.

## Set up on your own machine

Prerequisites:

- Python 3.12, which `.python-version` pins.
- [uv](https://docs.astral.sh/uv/).
- A Vector proxy key.

Run every command from the repository root. None needs root privileges.

1. Clone the repository and change into it.

2. Install the dependencies, every extra, and the development tools:

   ```bash
   $ uv sync --all-packages --all-extras --dev
   ```

3. Copy the example environment file:

   ```bash
   $ cp .env.example .env
   ```

4. In `.env`, set `VECTOR_PROXY_API_KEY=<your_key>`, where `<your_key>` is your Vector proxy key.
   If you have Langfuse keys, set those too.

5. Fetch the core data:

   ```bash
   $ uv run python -m aieng.advanced_rag.corpora.download --core
   ```

6. Run the environment check notebook, as in the Coder steps above.

`bash scripts/setup.sh --no-onboard --no-jupyter` runs steps 2 and 5 and installs the Jupyter
kernel in one command.

If the platform has provisioned keys for your GitHub handle and you have Google Cloud credentials
for it, `onboard` can load them in place of step 4:

```bash
$ eval "$(uv run onboard --bootcamp-name advanced-rag-bootcamp --test-script tests/test_integration.py)"
```

## Fetch the data for a reference implementation

`python -m aieng.advanced_rag.corpora.download` fetches what a reference implementation reads. It
verifies every file against its pinned SHA-256, prints each path with its digest, and skips work
already done, so you can run it any number of times. `--list` prints what each name covers.

| Reference implementation | Command | Adds to the core set |
| --- | --- | --- |
| Every RI | `python -m aieng.advanced_rag.corpora.download --core` | CUAD, ContractNLI, MultiHop-RAG queries, MuSiQue, FinanceBench questions, `bge-small-en-v1.5`: about 35 MB plus 130 MB |
| #1 `regulatory_policy_qa` | `python -m aieng.advanced_rag.corpora.download 1` | Nothing |
| #2 `cross_document_synthesis` | `python -m aieng.advanced_rag.corpora.download 2` | Nothing |
| #3 `table_structured_reasoning` | `python -m aieng.advanced_rag.corpora.download 3` | The 84 FinanceBench filings, 159 MB |
| #4 `graphrag_technical_support` | `python -m aieng.advanced_rag.corpora.download 4` | The ServerFault 10,000-question slice, built from an 860 MB Internet Archive dump; needs `7z` |
| #5 `agent_native_retrieval` | `python -m aieng.advanced_rag.corpora.download 5` | Nothing |

The corpora land in `~/.cache/aieng-advanced-rag/corpora`. `ADVANCED_RAG_CACHE_DIR` moves every
cache the package keeps, and `ADVANCED_RAG_CORPUS_CACHE` moves the corpora alone. The model lands
in the Hugging Face cache, which `HF_HOME` moves.

## Install the reference artifacts

The reference implementations reuse graphs, indexes, and judged runs built on Vector's on-prem
models. They are published as assets on GitHub releases of this repository, and each reference
implementation that has them names a manifest, `implementations/<ri>/reference-artifacts.json`,
which pins the URL and SHA-256 of every file.

Manifests exist for RI #2, RI #3, RI #4, RI #5, and the visual-documents and citations special topics.

`scripts/setup.sh` installs every manifest in the repository on each start, unless you pass
`--no-fetch`. An artifact already installed with its pinned digest is skipped, so a warm start
downloads nothing. A failed download is retried three times, and then the start continues with a
warning. The script's `--retries <count>` option (default 3) changes the retry count.

To install them yourself, run from the repository root:

```bash
$ python scripts/fetch_reference_artifacts.py
```

To install one reference implementation's artifacts, name its manifest:

```bash
$ python -m aieng.advanced_rag.cache implementations/<ri>/reference-artifacts.json
```

where `<ri>` is the reference implementation's folder, such as `graphrag_technical_support`.
This command downloads every artifact in the manifest again, while the script skips the ones you
have.

Both check each file's digest and its cache key before anything lands in the cache, and print what
they installed. The artifacts go to `~/.cache/aieng-advanced-rag/artifacts/fetched-global`, which
`ADVANCED_RAG_CACHE_DIR` moves with the rest of the cache. A result you build yourself goes to the
local tier and takes precedence over the reference one.

Artifacts derived from corpus text under a non-commercial licence, such as the FinanceBench
filings, are not published. The notebooks that use them rebuild them on your machine, from the
corpus you fetched. ServerFault-derived artifacts are published under CC BY-SA, with
`ATTRIBUTION-serverfault.md` in each release that holds them; anything you share that is built
from them carries the same licence and attribution.

## Run the onboarding gate from a terminal

`tests/test_integration.py` is the check `onboard` runs, with one test per credential. The proxy
test resolves the key and gets one chat completion, and fails when the key is missing, because a
skipped test would let `onboard` record you as onboarded. The Langfuse test checks the keys when
they are set and skips when neither is, since Langfuse keys are provisioned only when a bootcamp
uses tracing.

```bash
$ uv run pytest tests/test_integration.py -v
```

The completion goes to `Qwen3_6-35B-A3B`, an on-prem model every bootcamp key can call at no
cost. `ADVANCED_RAG_CHECK_MODEL` names a different one.

## Options for `scripts/setup.sh`

| Option | Environment variable | Default | Effect |
| --- | --- | --- | --- |
| `--fetch <spec>` | `ADVANCED_RAG_FETCH` | `core` | What to prefetch: `core`, RI numbers such as `"core 3"`, or `none`. Any value but `none` also installs every reference-artifact manifest. A failed download prints a warning and the start continues. |
| `--no-fetch` | | | Same as `--fetch none`. |
| `--no-onboard` | | | Skips `onboard`, and leaves `~/.bashrc`, `~/.zshrc`, and `~/.profile` alone. |
| `--no-jupyter` | `ADVANCED_RAG_START_JUPYTER=0` | Jupyter starts | Skips starting Jupyter Lab. |
| | `ADVANCED_RAG_BOOTCAMP_NAME`, then `BOOTCAMP_NAME` | `advanced-rag-bootcamp` | The bootcamp name passed to `onboard`. The Coder template sets `BOOTCAMP_NAME`. |

`setup.sh` runs `onboard` only when `GITHUB_USER` or `GH_USER` is set and no `OPENAI_API_KEY` is
exported yet. Jupyter's log is `~/.cache/aieng-advanced-rag/jupyter.log`.

## Troubleshoot the setup

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FAIL proxy key` in the notebook, or `MissingProxyKeyError` from a runner | No key in the environment or in `.env`. On Coder, the Jupyter server started before the keys loaded. | On Coder, run `eval "$(onboard --bootcamp-name advanced-rag-bootcamp --skip-test)"` in a terminal, then restart Jupyter from that terminal. Locally, set `VECTOR_PROXY_API_KEY` in `.env`. |
| `FAIL proxy completion` with `AuthenticationError` | The proxy rejected the key (HTTP 401). | Check the key for stray spaces or quotes. Reload it with `onboard`. |
| `FAIL proxy completion` with `PermissionDeniedError` | The key cannot use the model (HTTP 403 `model_access_denied`). | Set `ADVANCED_RAG_CHECK_MODEL` to a model your key can call. |
| `FAIL` on an import, or `ModuleNotFoundError` in a notebook | An extra is missing, or the notebook runs on a kernel from another environment. | Run `uv sync --all-packages --all-extras --dev` and select the **Advanced RAG Bootcamp** kernel. |
| `FAIL BAAI/bge-small-en-v1.5` | The model is not in the Hugging Face cache. | Run `python -m aieng.advanced_rag.corpora.download bge-small`. |
| `WARN` on a corpus | The core data has not been fetched. | Run `python -m aieng.advanced_rag.corpora.download --core`. |
| `ChecksumMismatchError` | A cached or downloaded file differs from its pin. The fetch deletes a bad cached copy and downloads again, so a repeat means the upstream file changed. | Re-run the fetch once. If it fails again, report it, and do not use the file. |
| `<manifest>: 0 installed, 0 already present, 2 failed` with `HTTP Error 404` | The manifest names a release that is not published, or is still a draft. | Pull the repository again. If the error stays, open an issue. |
| `Downloaded payload checksum mismatch` or `Downloaded envelope checksum mismatch` | A download was cut short, or the release file differs from its pin. Nothing was installed. | Run `python scripts/fetch_reference_artifacts.py` again. If it fails again, report it. |
| `extracting the dump needs the 7z program` | RI #4's ServerFault build needs `7z`. | Install `p7zip-full` (Debian and Ubuntu) or `p7zip` (Homebrew). The workspace image includes it. |
| `onboard` fails at "Fetch Environment Variables" | No keys are provisioned for your GitHub handle, or `GITHUB_USER` is wrong. | Check `echo $GITHUB_USER` against the GitHub handle the keys were provisioned for. |
| A shell tool cannot read `.env` after `set -a; . ./.env` | A line with spaces around `=` does not parse in a shell. The package parses it. | Remove the spaces, or export the key by hand. |
