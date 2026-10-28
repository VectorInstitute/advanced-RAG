# Running and evaluating experiments

The later notebooks extend the dense and decomposition references with bounded
IRCoT, an optional corrective comparison, graph retrieval, and a decision grid.
On a Slurm cluster, run everything inside a compute allocation.

Method behavior, bounds, and evidence flow are explained in
[METHODS.md](METHODS.md). The commands below keep those settings fixed.

## Run and resume

Use the bootcamp environment with local pinned BGE embeddings and the Vector
proxy. From the repository root inside compute:

```bash
export PYTHONPATH=aieng-advanced-rag
.venv/bin/python implementations/cross_document_synthesis/run_benchmark.py \
  --live --corpus musique --method ircot --per-category 10 --workers 4 \
  --output 'Arm Store/ri2-complete/musique-smoke-ircot.json'
```

Set `--per-category 0` for the complete split.
Use `--response-cache-dir path` to isolate a new cohort from reference responses.
The resolved namespace is part of resume identity; omitting the option preserves
the reference cache behavior. Dense, graded single-pass, and corrective controls
may share a fresh cohort cache, so exact-request reuse is recorded rather than
charged as a new call. Each worker keeps separate last-response usage state.
 Methods are `dense`,
`decomposition`, `ircot`, `graded-single-pass`, and `corrective`; corpora are
`musique`, `nq`, and the opt-in `multihop-rag`. Every method searches the
entire selected corpus. New queries use the same pinned local encoder.
Omitting `--live` inspects an existing report without model calls.

Reports record source/configuration identity, corpus pins, question selection,
predictions, evidence, stop reasons, and per-question shared-meter usage. A
completed question is saved atomically. Repeating the command resumes only if
the report configuration matches. Source or prompt changes require another
output path. Complete coverage means every selected question has an outcome;
`execution_success` is separate and stays false when any session failed. Failed
sessions retain zero answer and retrieval credit in the full denominator. The
grid marks affected cells `measured_with_failures`. Incomplete coverage cannot
enter a completed comparison. Cached generation responses have separate
cache-hit counts. ADK sessions are resumed at completed-question boundaries,
not individual model turns.

For a timed-out session, elapsed time and usage from completed responses remain
recorded. Usage from an unfinished provider request is unknown. Failed sessions
are not rerun selectively to improve the reported answer scores.

The independent decomposition prompt and validation define the reference arm.
It splits once into independent retrieval questions; IRCoT instead conditions
later searches on evidence. Do not change both methods while evaluating one.

## Corrective comparison

The optional arm reuses the shared RI #1 `AdvancedRagSpine`, `LLMGrader`,
`LLMRewriter`, and `CorrectivePolicy`. Thresholds are fixed before evaluation at
sufficient 0.6 and insufficient 0.0; they were not calibrated on MuSiQue.
It grades at most five passages, proposes one rewrite, and permits two retrieval
attempts. Generation is still single-pass cited answering after retrieval settles.
This is evidence correction, not repeated answer drafting.

Compare with dense single-pass and `graded-single-pass` to distinguish grading
cost from the corrective retry. Report paired quality differences and additional
calls/tokens. RI #1's outcome does not establish a gain on MuSiQue. A recommendation
requires a positive paired F1 interval and an explicit assessment of added cost.

## Replay and compare

```bash
.venv/bin/python implementations/cross_document_synthesis/run_evaluation.py \
  --reports 'Arm Store/ri2-complete/musique-dense.json' \
            'Arm Store/ri2-complete/musique-decomposition.json' \
            'Arm Store/ri2-complete/musique-ircot.json' \
  --output implementations/cross_document_synthesis/outputs/evaluation/summary.json
```

Replay verifies input provenance, selected question text, coverage, and saved
metrics against the pinned annotations. Comparisons require matched question
IDs, answer model, prompt, embedding configuration, and endpoint. The decision
grid accepts completed, replayed results; unmeasured cells have no score.

Answer EM and token F1 use all aliases. Supporting-passage recall and all-supports
coverage are reported at 5/10/20, alongside generation coverage, citation validity,
abstentions, failures, stop reasons, calls, tokens, tool calls, and service/wall
time. Citation validity checks source identity, not semantic entailment.
Failures receive zero answer and retrieval credit and remain in denominators.
Short rankings are scored as returned rather than padded with invented hits.

Paired percentile 95% bootstrap intervals use 5,000 resamples and seed 20261020.
Rewrite variants stay together under their original question ID. With very small
samples the intervals cannot establish population performance.
[The comparison notes](COMPARISONS.md#read-the-comparison) explain provider
variation and the difference between recorded usage and cold-run cost. Wall time
also depends on worker count and service load.

`export_results.py --input path.json --output compact.json` keeps replayable
scores, source IDs, and usage while removing full agent traces, replacing each
answer with its exact match and token F1, and replacing question text with
SHA-256 digests. It reads the pinned corpus to score the answers. `run_evaluation.py` verifies these digests
against the pinned loader before scoring. Original execution configuration
digests remain attached to the exported records. Exports also retain worker count
and elapsed batch time when recorded. Missing historical fields stay unknown;
summed per-question time is not a substitute for elapsed batch time.

## Paired wording comparisons

Pass `--rewrites path.json` to each core method. The input is:

```json
{"version":"experiment-v1","pairs":[
  {"question_id":"existing-musique-id","plain":"Plain question?","obfuscated":"Obfuscated question?"}
]}
```

The file digest is recorded. IDs must exist in MuSiQue and occur once, and both
wordings must be nonempty. Gold answers and supports come from the original
corpus. Keep the wording variants paired throughout evaluation. Evaluation
reports each wording separately and resamples original IDs for paired results.

## Prebuilt graph comparison

`run_graph_comparison.py --live --backend hipporag2` loads RI4's released
**atlas-rag HippoRAG 2 variant**. Choose `--backend lightrag` for LightRAG in
`mix` mode. The [installation guide](../GUIDE.md#compare-the-prebuilt-graph-indexes)
uses the [pinned release manifest](../reference-artifacts.json) to populate the
shared artifact cache. An optional `--artifact-manifest path.json` imports local
release downloads through the same checksum-verifying cache.

Each release entry pins its complete producer key, envelope SHA-256, and payload
SHA-256. The native loader verifies those pins before extraction, then verifies
the extracted tree, full corpus identity, and original-passage mappings. The
released archives need no RI2-specific sidecar. HippoRAG maps GraphML passage-node
text back to the pinned corpus. LightRAG verifies original document bytes,
processed status, and every chunk's document and file-path identifiers.

HippoRAG uses the producer's `musique` keyword, BGE encoder name, and
entity/relation settings with concepts and events disabled. Its graph and all
six required FAISS/embedding stores must already exist; loading cannot silently
re-embed the corpus. LightRAG also requires its complete graph and vector stores.
It retrieves at depth 40 before deduplicating to at most 20 original passages.
Both query encoders use RI2's pinned local revision. The release key records the
producer encoder name but does not pin its Hugging Face revision.

The final answer uses original corpus passages and the shared five-source
template. Query model calls use the Vector proxy, response cache, and shared
usage meter. Reports preserve the actual implementation, release identity,
settings, query usage, and per-question evidence. Construction costs are kept
separate; an unrecorded construction cost is unknown rather than zero.
Query-time writes use a temporary working copy, leaving the verified index
unchanged. Existing `--handoff` descriptors remain supported for explicitly
versioned custom artifacts with an in-archive `ri2_handoff.json` provenance
record.

## Article inputs

MultiHop-RAG article execution uses `--article-ruling path.json`. The loader checks
source authorization, immutable input digests, and evidence-to-article mappings.
[CORPUS.md](../CORPUS.md) describes the required authorization record and rights.

## Run and replay the shared-300 correction comparison

The committed correction cohort answers the shared 300 MuSiQue questions that
RI #4 and RI #5 report on. Run the three arms in order, so the graded control
reuses dense answers and correction reuses first-pass grades, sharing one
response-cache namespace:

```bash
for method in dense graded-single-pass corrective; do
  python implementations/cross_document_synthesis/run_benchmark.py \
    --live --corpus musique --method "$method" --shared-300 --workers 6 \
    --response-cache-dir 'Arm Store/my-corrective-study/responses' \
    --output "Arm Store/my-corrective-study/$method.json"
done
```

`--shared-300` selects `stratified_test_set` (150 two-hop, 75 three-hop, and 75
four-hop questions with its fixed seed) and ignores `--per-category`. Export each
report with `export_results.py` before committing it. The export replaces answer
text with scores and hashes question text.

`build_summaries.py` replays the committed core, graph, and correction reports
against pinned annotations and rewrites `outputs/evaluation/summary.json`,
`graphrag/summary.json`, `corrective/summary.json`, and
`outputs/validation/graph_release.json`. `run_shared_300.py` then rewrites
`shared_300.json`. Neither makes model calls. The correction interface requires
all three methods, one explicit cache namespace, matched answer settings, and
identical selected MuSiQue questions, and stores paired correction comparisons
under `corrective_cohort`, with its subset grid separate from full-split results.

`export_results.py --first-questions <n>` still projects the first `<n>` outcomes
of a raw checkpoint in pinned corpus order, for studies that use that slice.

### Incomplete-index exploration

`run_graph_comparison.py --allow-incomplete-index` permits a checksum-pinned
LightRAG release with missing passage chunks to run. Strict validation stays the
default, and the flag records incomplete coverage in the configuration identity;
resume cannot mix strict and exploratory settings. Decision-grid cells for this
mode are labelled `exploratory_incomplete_index`. The pinned release
`reference-ri4-bd05f7bb16c6` indexes every passage, so the committed LightRAG run
uses strict validation.

## Provider configuration

Use the shared [model, credential, and spend settings](../GUIDE.md#models-credentials-and-spend)
for live runs. New report identities include all three model roles. Keep the
same role settings across comparison arms, even when a particular arm does not
use every role. Live outputs default to ignored experiment storage; the
committed reference predictions remain the score-replay inputs.
