# Contributing to Advanced RAG Bootcamp

Thanks for your interest in contributing to the Advanced RAG Bootcamp!

To submit a PR, fill out the PR template. If the PR fixes an issue, link the PR to
the issue.

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

The code uses type hints, which [mypy](https://mypy.readthedocs.io/en/stable/) checks.

## Run the tests

CI runs two test suites and so must you. Both need every extra installed, because
some tests import numpy, sentence-transformers, or Google ADK.

Install everything from the repository root:

```bash
uv sync --all-packages --all-extras --dev
```

Run the package tests from `aieng-advanced-rag`:

```bash
pytest -m "not integration_test" tests
```

Run the implementation tests from the repository root, because the
implementations are not workspace members and cannot find the package from
inside their own folders:

```bash
pytest implementations/regulatory_policy_qa/tests implementations/regulatory_policy_qa/graphrag_null/tests
```

Four guards re-measure published retrieval figures, and they need the CUAD test
split and the `BAAI/bge-small-en-v1.5` embedding model. Fetch both once:

```bash
python scripts/fetch_test_data.py
```

Without that data a guard skips on your machine and fails in CI, because a
skipped guard checks nothing. The fetch takes seconds and about 130 MB.

## Notebook conventions

Every reference implementation ends in notebooks a participant runs, so a notebook has to work on a
machine that is not yours. Five rules, each of which a test in
`implementations/regulatory_policy_qa/tests/test_notebooks.py` checks.

Name a notebook `NN_lower_snake_case.ipynb`, numbered from `01_` in the order a reader works
through it.

Put every expensive step behind one module-level flag named `RUN_EXPENSIVE`, defaulting to `False`.
Expensive means it calls a model, embeds a corpus, or takes more than about a minute. With the flag
off the notebook still runs end to end and still prints the numbers, by reading the result the
expensive path produced. State the measured configuration next to any figure read that way.

Commit the outputs, with every cell executed in order. A reader who opens the notebook without
running it has to see what it produces, and a diff of the outputs is how a changed result becomes
visible in review.

Reach cached work through `aieng.advanced_rag.cache` rather than an ad hoc path, so a fresh
environment finds the committed result instead of paying for it again.

`load_or_run` hashes the file that defines a step's `compute`. When that file has changed since
the build, it rebuilds a live step and labels a replayed one. When the prompts or logic live in other files, list those modules in the
step's `code` field so an edit to them also counts. After changing a producer, rebuild and
republish its reference artifacts.

Keep credentials and trace identifiers out of the outputs. No proxy key, no Langfuse public or
secret key, no project URL or host. `aieng.advanced_rag.tracing.init_langfuse_tracing` does nothing
without credentials, which is what makes it safe to leave in a committed cell.

`implementations/regulatory_policy_qa/01_harness_and_baseline.ipynb` is the worked example. The
hackable starter template that each implementation's closing notebook adapts is separate, and is
tracked under Participant setup.
