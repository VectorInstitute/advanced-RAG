# Answering methods and their evidence

The answering additions extend the remote-main retrieval curriculum. They use the shared spine,
answering policies, judges, usage meter, result cache, and ADK loop. They do not replace the
retrieval and chunking measurements in notebooks 02 and 03.

## Participant path

| Notebook | Question it investigates | Evidence and limits |
| --- | --- | --- |
| 05: answering and null detection | Do rewriting, grading, retries, or citation requirements earn their calls? | Replays threshold traces and a 40-query policy comparison; includes the remote retry-without-rewrite control. Small same-model comparisons are descriptive. |
| 06: regulatory QA and RAGAS | Is an answer correct, supported, relevant, and willing to abstain? | Seven generated FOIA questions, six answerables; public source and references are inspectable. RAGAS evaluates the fixed shipped answers. |
| 99: starter agent | Can a bounded agent search and fetch evidence without crossing document scope? | Four recorded ADK cases, manual tool exercise, editable factory and own-corpus extension. Tool traces demonstrate behaviour, not agent accuracy. |

`RUN_EXPENSIVE = False` is the default. The notebooks read small shared-cache envelopes and
perform model-free exercises. `RUN_RAGAS = False` is a separate opt-in because local embeddings
do not make RAGAS judging free. Default replay needs no proxy key, model download, or RAGAS/ADK
installation. Notebook 99 imports ADK only in its live branch.

The reports were measured before this integration, using `Qwen3_6-35B-A3B` with reasoning disabled
and temperature zero. Their original timestamps, costs, and failures are preserved. They have
not been relabelled as fresh runs against the newer remote shared code. See
[the provenance record](outputs/advanced/PROVENANCE.md). Temperature zero removes sampling
variance only; cached outputs reproduce one observed run, not deterministic model behaviour.

## Replay and opt-in variants

From the repository root, the following runner prints a shipped report without model calls:

```bash
python implementations/regulatory_policy_qa/run_advanced.py policies
```

Other report names are `scope`, `regulatory`, `ragas`, and `agent`. Adding `--run` explicitly
executes a variant. Live variants use the existing proxy credential resolver and optional model
dependencies; RAGAS uses the RI project's `evaluation` extra, pinned to its measured 0.4.1 API.
No environment installation command is embedded in the participant notebooks.

Run these commands from the repository root with the repository's Python environment selected.
This RI remains a source-checkout project, rather than a new workspace member. See
[GUIDE.md](GUIDE.md#where-the-code-lives) for the shared implementation locations and the
repository [setup guide](../../docs/setup.md) for the environment. The added `advanced_lab/`
modules orchestrate experiments; reusable retrieval, answering, agent, cache and metric code
continues to live in `aieng-advanced-rag/`.

| Task | Live work | Additional dependencies |
| --- | --- | --- |
| `scope` | Local embedding/retrieval only; no hosted model calls | Shared `local` and `qdrant` extras |
| `policies` | 40-question policy comparison, generation/loop calls and two judge passes | Shared `local`, `qdrant`; RI `proxy` extra |
| `regulatory` | Question generation, answering and judging on the pinned FOIA snapshot | Shared `local`, `qdrant`; RI `proxy` extra |
| `ragas` | Automated judging on the selected QA report; local CPU embeddings | RI `evaluation` extra, including pinned RAGAS/LangChain versions |
| `agent` | Four starter cases, at most six model calls per case | Shared `agents`, `local`, `qdrant`; RI `proxy` extra |

Shared extras belong to `aieng-advanced-rag/pyproject.toml`; RI extras belong to this folder's
`pyproject.toml`. RAGAS is optional and is not installed by the root workspace sync. Live model
tasks use `Qwen3_6-35B-A3B` and the repository's proxy credential resolver. Reference replay
does not import the live experiment modules. Model calls and tokens depend on cache reuse,
retries and generated questions, so the recorded report usage is the observed cost of that run.

Live reports go to `ADVANCED_RAG_CACHE_DIR/ri1-answering/runs/<run-id>/` under the shared cache
root (the usual user cache when the variable is unset), with a local result-cache envelope.
`--output-dir PATH` chooses another private cache/run directory. The shipped reports and global
cache envelopes are preserved. Prompt caches and evidence tables can contain corpus text and
must stay private. On a Slurm cluster, run all Python, tests, and notebooks on a compute
allocation, not on a login node.

The runner prints the saved report path to stderr, leaving stdout as JSON for redirection.
Progress files, input snapshots and raw answer diagnostics live beside that report under the
same run ID. The reusable `responses.sqlite` cache stays at the store root. Choose the same
`--output-dir` on a later invocation to reuse responses; each process gets a new run directory.
Output/input flags require `--run`, so a misspelled execution mode cannot silently ignore them.

For example, explicitly run a regulatory variant into a private cache directory:

```bash
python implementations/regulatory_policy_qa/run_advanced.py regulatory --run --output-dir /path/to/private/ri1-cache
```

The reported directory contains both `regulatory.json` and `regulatory_source.json`. To evaluate
those new answers, pass that exact run directory to RAGAS:

```bash
python implementations/regulatory_policy_qa/run_advanced.py ragas --run --input-dir /path/to/private/ri1-cache/runs/RUN_ID --output-dir /path/to/private/ri1-cache
```

Without `--input-dir`, a live RAGAS run deliberately evaluates the shipped reference answers.
The input hashes in the RAGAS report identify which pair of files was evaluated. The source
snapshot is read from the repository and validated; a missing snapshot fails rather than
silently replacing the experiment's source with a new download.

For a model-cache-only policy recomputation, use `policies --run --cache-only --output-dir
/path/to/existing/private/cache`. This rebuilds local retrieval and fails if a required model
response is absent. It differs from the default report replay, which needs no local models.
`--cache-only` applies only to the policy runner; the other runners' default report replay is
their credential-free path. Both cache-only modes preserve the historical reference reports.

`require_citations` is measured alongside the other policies in `outputs/advanced/policies.json`.
Every answerable receives separate `judge_answer` correctness and grounding evaluations,
including abstentions. Presence of a citation is not evidence that its claim is correct or grounded.

## Portable public-source evaluation

`advanced_lab.regulatory_inputs.regulatory_inputs()` reconstructs the exact seven-row fixture
from `regulatory_source.json` and the ordered evidence IDs in `regulatory.json`. It validates the
source digest, unique question IDs, answerability labels, and evidence membership before any
optional model imports. Six answerables go to RAGAS; the null remains in the decision evaluation.
New RAGAS reports record both input-file digests, the library version, local encoder, per-row
metrics, and usage. A missing, invalid, or nonfinite score prevents publication.

Faithfulness, answer relevance, semantic similarity, and reference correctness answer different
questions. Generated references and a shared generator/judge can share mistakes. Inspect the
individual failures; do not interpret the means over six examples as regulatory reliability.
Repeated identical RAGAS prompts reuse the response cache under a fixed temperature-zero
protocol, so the repetitions are not independent samples.

## Full-split and CUAD follow-ups

The threshold runner supports `--coverage-only --replay-dir PATH` for a zero-model-call full-split
cache audit and `--cache-only --replay-dir PATH` for strict trace replay. The null runner supports
`--corpus contract-nli --documents 123 --arms dense --score-only --cache-only` to fail on missing saved evidence/responses rather than index or
call a model. Use `--help` for output and store paths. The committed
[coverage audit](outputs/full_contract_nli/cache_status.json) is incomplete: 1,411 traces per grader
are missing. This integration does not authorize or perform the deferred paid harvest.

To audit the full threshold population without a model client:

```bash
python implementations/regulatory_policy_qa/run_threshold_sweep.py --corpus contract-nli --documents 123 --coverage-only --replay-dir implementations/regulatory_policy_qa/outputs --output-dir /path/to/private/full-split-audit
```

Optionally add `--dense-evidence /path/to/store/evidence/dense.json` to count saved dense evidence
in the same audit. Coverage does not validate or score the traces. A subsequent strict cache-only
replay of the full population is expected to fail while the missing traces remain unavailable;
it will not silently switch to paid calls or publish a partial full-split score.

The [CUAD pilot](outputs/cuad_null/COMPARISON.md) measures five fixed contracts, not the whole
split. It reports paired outcomes and separately judged correctness, exact input hashes, unequal
realized text lengths under matched caps, extraction warnings, and indexing/retrieval cost lower
bounds. The runner restores only an exact stripped document ID within its contract workspace;
it rejects broader scope leakage. Historical prompt and evidence caches are not redistributed.
