# Choose a model and track what it costs

Pick a provider family for the notebooks, reuse the reference results built on Vector's on-prem
models, and see what a live run on your model costs before you start it.

Every chat model call goes through the Vector proxy with your team's key. The reference graphs,
indexes, and results in this repository were built on the on-prem Qwen models, which cost nothing.
Your own runs use the family you pick: the Anthropic, Google, and OpenAI families spend your team's
credits, and the `onprem` family costs nothing. Embeddings run locally on `BAAI/bge-small-en-v1.5`
in every family and never reach a paid API.

## Choose a model set

Put this at the top of a notebook that calls a model:

```python
from aieng.advanced_rag.models import choose_models

RUN_EXPENSIVE = False
MODELS = choose_models(show=True)
```

`choose_models` returns one model for each of three roles:

- `generation` answers questions, on the family's cheapest model.
- `extraction` builds graphs, indexes, and contextual chunks, on the same cheap model, because a
  build makes tens of thousands of short calls.
- `judging` grades answers. Every family judges with `gemini-3.8-flash`, whichever model generates,
  so scores compare across families. Scores from an overridden judge compare only with scores from
  the same judge.

With `show=True` it prints each model's price per million tokens as your key sees it and the
budget left on the key, and it starts the spend session that the notebook's last cell reports on.

To pick a family, pass it or set it once in your shell or `.env`:

```python
MODELS = choose_models("anthropic", show=True)  # or "google", "openai", "onprem"
```

```bash
export ADVANCED_RAG_MODEL_FAMILY=openai
```

With no argument and no variable, `choose_models` uses the `google` family: `gemini-3.5-flash-lite`
for generation and extraction, and the shared judge `gemini-3.8-flash` for judging.

To change one role, pass it or set its variable. Each paid family's frontier model is available this
way and is never a default: `claude-opus-5-5`, `gemini-4-argon`, and `gpt-6-astra`.

```python
MODELS = choose_models("anthropic", judging="claude-opus-5-5", show=True)
```

| Variable | Effect |
| --- | --- |
| `ADVANCED_RAG_MODEL_FAMILY` | `anthropic`, `google`, `openai`, or `onprem`. |
| `ADVANCED_RAG_GENERATION_MODEL` | Overrides the generation model. |
| `ADVANCED_RAG_JUDGE_MODEL` | Overrides the judging model. |
| `ADVANCED_RAG_EXTRACTION_MODEL` | Overrides the extraction model. |
| `ADVANCED_RAG_SPEND_LIMIT_USD` | Stops a run once its estimate passes this many dollars. Unset means no limit beyond the key's budget. |
| `ADVANCED_RAG_KEY_BUDGET_USD` | The budget, in US dollars, the spend guards assume when your key reports none. Default 300. Ignored when the key reports a budget. |

An argument wins over a variable, and a variable wins over the family default. A model name
outside the default sets works if the proxy grants it to your key; you get a warning, and its
price comes from the proxy.

The `onprem` family uses `Qwen3_6-35B-A3B`, which built most of the reference results. Some
reference artifacts were built with another on-prem model, and each one's label names it. Whether
your key reaches the on-prem models depends on how your project is set up on the proxy.

## Reuse a reference result or run your own

Each model-produced step in a notebook goes through one call:

```python
from aieng.advanced_rag.runs import load_or_run

result = load_or_run(step, build, models=MODELS, live=RUN_EXPENSIVE)
```

With `RUN_EXPENSIVE = False`, you get your own earlier result for your model if you have one, and
the reference result otherwise. Either way, the cell prints where the result came from:

```text
graph   reference artifact, built with Qwen3_8-27B on Vector's cluster at $0
        20,077 calls, 10,960,000 prompt + 15,170,000 completion tokens
        rebuilding with claude-haiku-5-5: about 20,077 calls, est. $8.68 (set RUN_EXPENSIVE = True to rebuild)
```

With `RUN_EXPENSIVE = True` and no result of yours, the step prints the estimate, runs on your
model, and stores the result in your local cache, so the next run reads it back at no cost. A
step whose estimate is more than 10% of the budget left on the key stops until you pass
`confirm=True`.

If your key reports no budget, the guard assumes $300 and the cell prints a line that says so, for
example `key budget not set on the key; assuming $300.00 (set ADVANCED_RAG_KEY_BUDGET_USD to
override), spent $12.40, left $287.60`. While the budget is assumed, any step estimated at more
than $1 also stops until you pass `confirm=True`. Cached and reference results never stop, so a
notebook with `RUN_EXPENSIVE = False` runs through without a prompt.

Later steps run on your model over whichever result came back. On each family's default
models, answering questions over the reference graph costs cents. Rebuilding the graph costs about
$8.68 on `claude-haiku-5-5`, for example.

A live run records a digest of the producer's source file in the result's `produced_by` block, as
`code_hash`. If you edit that file, for example a prompt or a retriever setting, a cached result no
longer matches your code. With `RUN_EXPENSIVE = True`, the step rebuilds your own result from your
code. With `RUN_EXPENSIVE = False`, and for the reference result, the cell returns the cached result
and the label adds a line:

```text
answers code changed: built from 3f9c0a1e2b4d5c6f, the producer is now 8a7b6c5d4e3f2a1b, so this result does not reflect the code on disk (set RUN_EXPENSIVE = True to rebuild it)
```

Results built before the digest existed replay with no such line.

The estimate scales the reference build's token counts to your model's price. Tokenizers differ,
and a model can write longer or shorter completions than the one that built the reference, so read
it as a guide. The live run replaces it with metered figures.

## Where the reference results come from

The reference results live in your cache's fetched-global tier. They reach it from GitHub release
assets on `VectorInstitute/advanced-RAG`: each reference implementation with published results has
a manifest, `implementations/<ri>/reference-artifacts.json`, that pins every file's URL and
SHA-256, and `scripts/setup.sh` installs every manifest on start.
[Install the reference artifacts](setup.md#install-the-reference-artifacts) covers installing them
by hand.

A release carries no artifact that holds text from a corpus under a non-commercial licence, which
leaves out the FinanceBench ingestion (CC BY-NC 4.0) and the runs and page texts built from it. A
step over one of those has no reference result to fall back on, so you build it on your machine:
with `RUN_EXPENSIVE = True` for a model step, which prints its estimate first. ServerFault-derived
artifacts (CC BY-SA) are published under that licence: RI #4's ServerFault knowledge graph and
RI #5's ServerFault runs. Their releases carry `ATTRIBUTION-serverfault.md`, which links each
post and names and links its author. The visual-documents release holds FinanceBench-derived
artifacts with no filing text, the page-embedding index among them, under CC BY-NC 4.0 and only
for non-commercial use.

## Publish the reference artifacts

`scripts/publish_artifacts.py` turns what a cache holds into a release and its manifest. It is for
whoever builds the reference results. It never creates a release, and it uploads only with
`--upload`.

1. Check what the cache holds, with no files written:

   ```bash
   $ python scripts/publish_artifacts.py all --dry-run
   ```

   For each reference implementation, the report lists every artifact type with its count, asset
   count, size, licence, and decision. A problem line means an envelope or payload failed its
   checks. A `LIMIT` line means the set breaks GitHub's limits of 1,000 assets per release and
   2 GiB per file.

2. Stage one reference implementation's artifacts and write its manifest:

   ```bash
   $ python scripts/publish_artifacts.py <ri>
   ```

   where `<ri>` is an RI number from 1 to 5, a folder name, or `visual-documents`. Pass
   `--skip-refused` to leave out the artifacts the licence policy refuses, rather than stop. The
   script stages the cached bytes under content-addressed names in
   `~/.cache/aieng-advanced-rag/publish/<tag>`, beside a `SHA256SUMS` file and release notes that
   carry each corpus's attribution, and writes `implementations/<ri>/reference-artifacts.json`.
   The tag is derived from the artifacts, so the same set always gets the same tag and manifest.

3. Run the `gh release create` command the script printed. It creates a draft.

4. Upload the files:

   ```bash
   $ python scripts/publish_artifacts.py <ri> --upload
   ```

   The script uploads in batches of 40 files and then compares GitHub's recorded digest for every
   asset with the manifest. `--verify-release` runs that comparison without uploading.

5. Publish the draft. A draft's download URLs return 404 to everyone else, so the installer reports
   the release as not published until you do.

6. Commit the manifest. Workspaces install it on their next start.

The licence policy is `scripts/reference-artifact-licences.json`. Each rule names the corpora an
artifact type derives from and whether its payload holds corpus text. An artifact no rule matches
is refused. An artifact a rule calls text-free is still refused if its value holds a string of 400
characters or more, or its payload holds a text file. Text under CC BY-NC or CC BY-SA is refused
unless an `allowlist` entry matches it, and every allowlist entry needs a `reason`. The shipped
allowlist admits ServerFault-derived artifacts, and a release that holds them gets a per-post
attribution file built from the slice and the dump's `Users.xml`.

RI #5's per-question `agent-question` entries are left out by default: the `agent-runs` steps hold
the same results, and the 1,653 entries pass the 1,000-asset limit. `--include-type` and
`--exclude-type` choose types, and `--key` chooses single artifacts by key digest.

## Read what a run cost

A cell that calls a model can print one line from the meter:

```python
from aieng.advanced_rag import metering, spend

print(spend.usage_line(metering.default_meter()))
```

```text
592 calls, 434,921 in + 18,000 out tokens on gemini-3.5-flash-lite, est. $0.18
```

The last cell of a notebook prints the run's estimate beside the proxy's own figure and the budget:

```python
print(spend.report())
```

```text
this run, estimated   est. $1.94 over 3 models
this run, proxy usage $2.07 (team key: includes teammates' calls in the same window)
key budget $300.00, spent $43.19, left $256.81 (86%)
```

The two figures differ when a teammate runs at the same time, because the proxy reports usage per
key and a team shares one key. When the proxy does not answer, for example on a personal key, the
report shows the estimate alone, and prices come from a table in the package.

The model cell and the report warn when the key has spent 50% of its budget and again at 80%.
When the key reports no budget, the warnings use the assumed $300, or
`ADVANCED_RAG_KEY_BUDGET_USD`, and say the budget is assumed. The proxy sends no alerts of its own.

## Fix errors from model cells

- A cell raises `ReferenceMissingError`: the reference artifact is not in your cache. Run
  `python scripts/fetch_reference_artifacts.py` from the repository root, which installs every
  manifest, or set `RUN_EXPENSIVE = True` to build it on your model. Some artifacts are never
  published because of their corpus licence; see
  [Where the reference results come from](#where-the-reference-results-come-from).
- A cell raises `ConfirmationRequiredError`: the step would spend more than 10% of what is left on the
  key, or more than $1 while the key reports no budget. Read the estimate, then pass `confirm=True`
  if you want the run. If your key has a budget the proxy does not report, set
  `ADVANCED_RAG_KEY_BUDGET_USD` to it.
- A cell raises `SpendLimitReached`: the run passed `ADVANCED_RAG_SPEND_LIMIT_USD`. The call that
  passed it was sent and billed; every later call is refused before it is sent. Raise or unset the
  limit to continue.
- A paid model returns 403: your key has not been granted that model. Choose another family.
- A paid model returns 402 `key_budget_exceeded`: the key's spend limit is used up. If your key
  reaches the on-prem models, the `onprem` family still works.
- The price shows as unknown: the proxy does not list the model for your key, and the package's
  table has no entry for it. Spend still counts on the proxy, and only the estimate is missing.
- The label says a cached result is stale: the proxy resolves the model name to a different
  snapshot than the one that built your result. Run the step live to rebuild it.
- The label says the code changed: the file that produced the result differs from the one that
  built it. If you edited that file, set `RUN_EXPENSIVE = True` to rebuild the result from your
  code. Without budget for the rebuild, the result shown is the one from the unedited code. If a
  pull changed the file, the reference result is older than the code and still reflects the code
  that built it.
