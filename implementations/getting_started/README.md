# Getting started

Confirm that your environment can run every reference implementation before you open one. This
folder holds the environment check, and it is the first thing to run on a new Coder workspace or a
new local checkout.

## Run the environment check

Open [`00_environment_check.ipynb`](00_environment_check.ipynb) with the **Advanced RAG Bootcamp**
kernel and run every cell from the top. Each check prints `PASS`, `WARN`, or `FAIL`, and a line that
is not `PASS` ends with the command that fixes it. The last cell counts the results.

The notebook checks:

- The Python version and that `aieng.advanced_rag` imports from this checkout.
- The libraries each reference implementation imports, and that PyTorch is the CPU build.
- Where the proxy key, the proxy endpoint, and the Langfuse keys come from, without printing them.
- One chat completion through the proxy, when `RUN_EXPENSIVE = True`.
- That Langfuse accepts the keys.
- That `BAAI/bge-small-en-v1.5` loads from the local cache and returns 384-dimension vectors.
- That the core corpora are cached and match their pinned SHA-256 digests.
- That the cache directories are writable.

The proxy completion is the only check that calls a model, so it sits behind `RUN_EXPENSIVE`, which
ships `False`. Set it to `True` in the first code cell to include it. The call asks a free on-prem
model for two letters. From a terminal at the repository root, the onboarding gate makes the same
call:

```bash
pytest tests/test_integration.py -v
```

## Fix a failed check

[`docs/setup.md`](../../docs/setup.md) covers the workspace start, the credential model, local setup,
and a troubleshooting table for the failures this notebook reports.

## Continue

Start with [`regulatory_policy_qa/`](../regulatory_policy_qa/), which teaches the pipeline and the
evaluation harness, then move to the reference implementation closest to your use case. Each reference implementation's
data comes from one command, which `docs/setup.md` lists. That page also says which reference
artifacts the setup script installs.
