# RI #1: Advanced RAG Spine (Regulatory & Policy Q&A)

## Introduction
The foundational reference implementation of the Advanced RAG Bootcamp. It builds
the multi-stage RAG pipeline and the shared **evaluation harness** that every
other RI reuses, running on dynamic regulatory and policy content. It includes
**"not stated" / null-answer detection**, the number that decides whether a
pipeline can be trusted on a corpus where nearly half the questions have no
answer. Start here to fix what
"correct" and "grounded" mean for a corpus before adding advanced techniques.

## Where it sits on the spine
**Baseline RAG to Advanced RAG.** The start of the "RAG becoming agentic" spine
(see the [repository README](../../README.md)). The agent loop stays small, and the lift comes
from the retrieval substrate. Later RIs grow the agent circle.

## What the participant notebooks demonstrate

Notebook [`01_harness_and_baseline.ipynb`](01_harness_and_baseline.ipynb) demonstrates:

- loading the pinned CUAD and ContractNLI corpora from a fresh environment;
- BM25, dense retrieval, and a shuffled negative control;
- the evaluation harness's retrieval-failure rate and mean reciprocal rank (MRR);
- null-detection metrics and the always-answer and always-abstain controls;
- why chunk size is a confound at fixed `k`, with an equal-character-budget comparison; and
- a single-pass answering pipeline over four sample questions, using scoped BM25,
  `SingleShotPolicy`, and `Qwen3_6-35B-A3B` with reasoning disabled. Its measured outcomes
  and dense retrieval reports replay through the shared result cache when expensive work is off.

[`02_retrieval_levers.ipynb`](02_retrieval_levers.ipynb) scores each retrieval lever on the same
contract questions and sets what it gains against its CPU time per question and index size. It covers:

- weighted reciprocal rank fusion (RRF) of BM25 and dense rankings, with the dense weight chosen by a sweep;
- SPLADE (a learned sparse model) and ColBERT late interaction beside BM25, dense, and hybrid first stages;
- reranking with two cross-encoders and with ColBERT, at several depths; and
- exact search, HNSW (hierarchical navigable small world) graph search, and 1-bit and 2-bit quantization on a Qdrant server.

On the full CUAD split, SPLADE alone fails on 0.2066 of questions against 0.2436 for the tuned
BM25 and dense hybrid, with an MRR of 0.5687 against 0.4969. The notebook runs on three slices and
names the winning lever for each metric.

[`03_chunking_variants.ipynb`](03_chunking_variants.ipynb) compares plain, hierarchical,
header-labelled, and contextual chunks of the ContractNLI test split, scored with retrieval scoped to
the question's contract and across all 123 contracts. Inside one contract no arm beats plain chunks
by more than seven questions in 1,188 at a tight character budget. Across all contracts, contextual chunking with the
head-and-neighbourhood prompt cuts the failure rate at `k=10` from 0.6473 to 0.1187, for 1,418 model
calls. The notebook also reports each arm's indexing cost in calls, tokens, and an estimated price,
and replays an earlier CUAD and ServerFault measurement.

[`04_bring_your_own_corpus.ipynb`](04_bring_your_own_corpus.ipynb) builds an evaluation set from a
folder of your documents and ranks the retrieval arms on it. It generates questions from quoted
statements, filters them, adds null questions, and scores the eight arms against the quotes' character
offsets. With the expensive flag off it uses ten ContractNLI contracts and replays a reference set
built on Vector's on-prem models. The method and its limits are in
[the synthetic question sets special topic](../special_topics/synthetic_question_sets/README.md).

[`05_answering_and_null_detection.ipynb`](05_answering_and_null_detection.ipynb) separates query
rewriting, evidence grading, retries, citation requirements, and abstention. It replays the measured
threshold traces, includes a retry-without-rewriting control, and makes the dense-unscoped scope
confound an exercise. The corrective loop remains an opt-in negative-result experiment.

[`06_regulatory_qa_and_ragas.ipynb`](06_regulatory_qa_and_ragas.ipynb) follows seven free-form
questions over a pinned public-domain federal FOIA source, audits source quotes, and separates
answer correctness from grounding. It reconstructs the evaluation inputs from shipped artifacts
and explains RAGAS faithfulness, relevance, and semantic similarity with local embeddings.
The six answerable examples are a teaching fixture, not a regulatory accuracy benchmark.

[`99_starter_agent.ipynb`](99_starter_agent.ipynb) adapts the shared ADK `AgentLoop` and
framework-neutral retrieval tools. Participants can edit the agent factory and supply their own
corpus. Both search and fetch stay within the selected document, model calls are bounded, and
the saved examples include an unsuccessful run that reaches the call limit.

The added answering notebooks replay historical measurements by default. Their provenance,
limits, and opt-in runners are documented in [ANSWERING_EXPERIMENTS.md](ANSWERING_EXPERIMENTS.md).

The stages between a question and a grounded answer, and the numbers each one
exposes, are in [the pipeline reference](PIPELINE.md). The corrective loop's
thresholds are measured in [the threshold memo](THRESHOLDS.md).

## Run and adapt experiments

Open the notebooks in order with expensive flags off to inspect the methods and recorded results.
For the added answering experiments, the command-line equivalent is:

```bash
python implementations/regulatory_policy_qa/run_advanced.py policies
python implementations/regulatory_policy_qa/run_advanced.py --help
```

Run those commands from the repository root using the repository's Python environment. The default
prints a reference report without credentials or model downloads. `--run` explicitly executes a
variant, keeps new outputs in the shared private cache, and prints the saved report path.
For dependencies, individual experiment commands, cache-only recomputation, and passing a new QA
run to RAGAS, follow [the experiment guide](ANSWERING_EXPERIMENTS.md).

The notebooks explain and vary methods. `advanced_lab/` orchestrates these RI-specific experiments
over the shared package. `outputs/` holds attributed reference results; response databases, indexes
and per-run diagnostics belong in the private cache. Environment setup stays in [GUIDE.md](GUIDE.md)
and the [repository setup guide](../../docs/setup.md).

## Datasets
Licences, pinned versions, and the attribution that has to ship are recorded in
[`CORPUS.md`](./CORPUS.md). Loaders are in
`aieng-advanced-rag/aieng/advanced_rag/corpora/`, and every one verifies a
SHA-256 digest on fetch, so an upstream change fails the run instead of altering
a measurement.

- **CUAD** (CC BY 4.0): SQuAD-style clause QA over 102 contracts, with 2,938
  labelled unanswerable questions as the in-domain null cases. Pinned to
  `data.zip` at commit `7aad6491`. The Hugging Face mirror ships a loading
  script that resolves the data through a moving reference, so it cannot be the
  pinned source.
- **ContractNLI** (CC BY 4.0): 123 non-disclosure agreements with
  human-annotated evidence spans and a 43% `NotMentioned` rate. A second
  document genre and a second null operating point. Pinned by archive digest.
- **MultiHop-RAG** (ODC-BY 1.0): the 301 null queries, used for the
  out-of-domain **Null** control. The article corpus is not used and is not
  redistributed; ODC-BY does not cover the publishers' copyright in it.

PolicyQA is not in the built set and has no loader. Its answer spans derive
from OPP-115, which Carnegie Mellon releases "for research, teaching, and
scholarship purposes only", with a separate commercial licence sold for the
annotation files the spans come from. ContractNLI covers the same capability gap
and is already built, so nothing waits on the answer.
[`CORPUS.md`](./CORPUS.md) quotes the terms.

## Prerequisites
- Python and the bootcamp environment (see `docs/`)
- Completion of the technical onboarding / environment check

## Status
Notebooks 01 to 06 and 99 cover the participant methods. The remote's notebooks 02 to 04 are
retained unchanged; the baseline's code and measured outputs are retained too. Added notebooks
ship with executed outputs and expensive flags off.

The full 123-document threshold/answering measurement is **not complete**. Its runner supports
cache-only coverage checks and fails on missing replay data. The available threshold traces cover
680 of 2,091 queries, and the original dense evidence covers 340. No full-split quality figure is
inferred from that coverage. The all-123 retrieval experiment in notebook 03 is a different task.
See [the experiment guide](ANSWERING_EXPERIMENTS.md#full-split-and-cuad-follow-ups).

The CUAD null check is a bounded five-contract pilot, with correctness judged on all 70 answerables
per arm. It does not establish a full-CUAD or population result. Its paired outcomes, reader budgets,
graph extraction warnings, and indexing-cost lower bound are in
[the CUAD comparison](outputs/cuad_null/COMPARISON.md).

The shared harness, the corpus loaders, and the pipeline stages are in
`aieng-advanced-rag/aieng/advanced_rag/` under `retrieval/`, `evaluation/`,
`corpora/` and `pipeline/`. The harness reproduces the retrieval baselines it
was consolidated from, exact to four decimal places. The fresh-environment verification
record is in [`outputs/baseline_validation.json`](outputs/baseline_validation.json).

The corrective loop's thresholds are measured, and the loop is not part of RI #1's recommended
participant path. [`THRESHOLDS.md`](THRESHOLDS.md) carries the sweep, the controls, and what the
operating point costs on answerable questions. The generator-only control scores 0.780 balanced
accuracy against 0.788 for the best `LLMGrader` loop. A paired test cannot tell the 0.008 gain
from zero (exact McNemar p = 0.26 over 680 paired questions, 95% paired bootstrap interval -0.004 to 0.021), while calls rise from 680 to 6,259, about 9.2 times. The generator's own `not_stated` path,
not the grader, detects the missing answers. The loop remains available only as an opt-in A/B
measurement and as reusable package machinery for another RI to evaluate on its own corpus.
The single-pass participant configuration uses `SingleShotPolicy`, `max_attempts=1`, a model-free
`HeuristicGrader`, and no corrective rewriter, so only generation calls a model and the grade cannot
pre-empt the generator's abstention decision.

The headline null set is ContractNLI, at a 43% `NotMentioned` rate. CUAD's 70%
`is_impossible` rate makes it a secondary in-domain check rather than a headline,
because always abstaining scores 0.70 there.

The original five-arm experiment did not reproduce the claim that naive GraphRAG collapses on
null answers. It identified retrieval scope as the main confound. Its unscoped dense arm answered
75.8% of null questions against 13.0% scoped, at the historical 0.6/0.3 operating point.
The two-arm replay uses the measured 0.6/0.0 thresholds and the original
Qwen3_6-35B-A3B model identifier: dense scores 0.7572 balanced decision
accuracy and LightRAG 0.7458, with paired McNemar p=0.6201.
Both replayed verified cached evidence with no indexing or retrieval calls.
`judge_answer` scores every answerable question, including abstentions, with
separate semantic correctness and evidence-grounding passes:
dense 122/179 (0.6816),
LightRAG 109/179 (0.6089) correct.
Correctness measures agreement with ContractNLI annotations, using the same model
for generation and judging. The sample does not establish a population ranking.
GraphRAG stays out of RI #1 on its historical indexing cost: 1.18M tokens for 20 contracts
against zero hosted-model calls for local dense indexing. See
[`GRAPHRAG_NULL_ARM.md`](./GRAPHRAG_NULL_ARM.md).

To replay an exported store, verify `null-arm-store.tar.gz`, unpack it into
`./unpacked`, and run `run_null_arm.py --store ./unpacked`. The bundle disables
indexing and retrieval automatically; defaults score all five arms at 0.6/0.0
and write to `outputs/graphrag_null/measured_thresholds/`. See the
[export-and-replay commands](GRAPHRAG_NULL_ARM.md#score-it-on-another-machine).

When the optional loop is enabled, its fallback deviates from CRAG deliberately: CRAG falls back to
web search on incorrect evidence, while this implementation rewrites, retries, then abstains,
because fetching a contract clause from the web is wrong on a closed corpus. No rewriter is
configured in the recommended RI #1 path, which also selects `SingleShotPolicy` and one attempt, so
that retry does not run there.

Live experiments follow the repository’s shared [model and credential settings](GUIDE.md#models-and-credentials-for-live-experiments). Recorded tables retain their original model identities; embeddings stay local.
