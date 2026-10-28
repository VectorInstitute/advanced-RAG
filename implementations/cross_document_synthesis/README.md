# RI #2: Cross-document synthesis

This reference implementation asks when an agent needs more than one retrieval
step to answer a question spanning several documents. It keeps the passage
collection, local embedding model, and final answerer fixed while comparing dense
retrieval, query decomposition, and IRCoT. Optional evidence correction tests
whether grading and rewriting justify their additional cost.

MuSiQue supplies questions requiring two, three, or four linked passages.
NaturalQuestions supplies the single-hop control. The notebooks explain how each
method assembles evidence, let you change its retrieval policy, and separate
retrieval failures from answer-generation failures.

## Results on MuSiQue and NaturalQuestions

Each core method answers the same 1,000 questions within each corpus. All search
the complete passage collection and generate from at most five original passages.
Answers are scored against every accepted alias. Rates below are fractions.

| Corpus | Method | Exact match | F1 | Supporting recall@5 | Generation all-supports |
| --- | --- | ---: | ---: | ---: | ---: |
| MuSiQue | Dense | 0.2010 | 0.2733 | 0.5060 | 0.1710 |
| MuSiQue | Decomposition | 0.2060 | 0.2805 | 0.5540 | 0.1880 |
| MuSiQue | IRCoT | 0.4090 | 0.4854 | 0.6823 | 0.4300 |
| NaturalQuestions | Dense | 0.5070 | 0.6302 | 0.6824 | 0.3510 |
| NaturalQuestions | Decomposition | 0.5010 | 0.6249 | 0.6739 | 0.3440 |
| NaturalQuestions | IRCoT | 0.4840 | 0.6036 | 0.6812 | 0.3230 |

IRCoT alternates brief, source-supported evidence updates with targeted searches.
Its paired F1 gain over dense is 0.2122 on MuSiQue (95% interval 0.1828–0.2418),
while its difference on NaturalQuestions is −0.0266 (−0.0415 to −0.0115).
The decomposition differences include zero. Evidence-dependent retrieval helps
this multi-hop workload; the single-hop control shows why additional steps should
not be the default for every question.

### Cost and limits

| Corpus | Method | Recorded model calls + cache hits per question |
| --- | --- | ---: |
| MuSiQue | Dense | 1.000 |
| MuSiQue | Decomposition | 2.000 |
| MuSiQue | IRCoT | 6.335 |
| NaturalQuestions | Dense | 1.000 |
| NaturalQuestions | Decomposition | 2.000 |
| NaturalQuestions | IRCoT | 4.664 |

Cache hits and original execution usage are reported separately in the
[comparison](docs/COMPARISONS.md). These counts are logical request counts, not
cold-run billing estimates. Local embeddings make no hosted embedding calls.

Nine MuSiQue and six NaturalQuestions IRCoT sessions timed out. They remain in
the 1,000-question denominators with zero credit. Citation validity establishes
source identity, not that the source entails the answer. NaturalQuestions can
mark alternative supporting contexts, so requiring every marked context is a
strict diagnostic rather than a necessary condition for answering.

See [COMPARISONS.md](docs/COMPARISONS.md) for hop-count breakdowns, paired
intervals, optional correction results, and the limits of these measurements.

### The shared 300 MuSiQue questions

RI #4, RI #5, and the special topics report MuSiQue on the same 300 questions
(150 two-hop, 75 three-hop, and 75 four-hop, drawn by `stratified_test_set`).
`run_shared_300.py` selects those questions from the committed 1,000-question
records and rescores them with no model calls. Every arm reads with
`Qwen3_6-35B-A3B` and a five-passage budget, and the scores are lexical, with no
judge model.

| Method | Exact match | F1 | F1 minus dense (95% interval) | Calls + cache hits per question |
| --- | ---: | ---: | --- | ---: |
| Dense | 0.200 | 0.275 | | 1.00 |
| Decomposition | 0.227 | 0.294 | +0.019 (-0.013 to 0.053) | 2.00 |
| IRCoT | 0.420 | 0.509 | +0.234 (0.184 to 0.284) | 6.58 |
| HippoRAG 2 (atlas-rag) | 0.320 | 0.418 | +0.143 (0.098 to 0.189) | 2.00 |
| LightRAG (mix) | 0.213 | 0.284 | +0.010 (-0.016 to 0.036) | 2.00 |

Correction and its two controls ran on these 300 questions directly, in their
own response-cache namespace; the next section reports them.

RI #4's own rows on the same questions use the same reader model with its own
answer prompt and a rank depth of 10, so its F1 differs: 0.250 for dense and
0.390 for HippoRAG 2. Compare methods within one RI's table. The
[shared-300 summary](outputs/evaluation/shared_300.json) holds the hop-count
breakdown and RI #4's figures.

## Prebuilt graph retrieval

HippoRAG 2's atlas-rag variant, using the released full-corpus index, scores
0.4073 F1 on the same 1,000 MuSiQue questions. Its paired difference
from dense is 0.1340 (95% interval 0.1088 to 0.1600),
with the same five-passage final reader. Graph retrieval adds model calls and
an upfront index-building cost; the notebook separates these costs.

LightRAG in `mix` mode, over RI #4's release `reference-ri4-bd05f7bb16c6`, which
indexes all 11,656 passages, scores 0.2655 F1 on the same 1,000 questions. Its
paired difference from dense is -0.0077 (95% interval -0.0236 to 0.0079), so
these runs do not establish a difference. See the
[graph comparison](docs/COMPARISONS.md#lightrag-over-the-complete-index) for
per-hop scores and cost.

## Does evidence correction help?

Correction runs on the shared 300 MuSiQue questions with matched dense and
graded single-pass controls, fixed thresholds (0.6 sufficient, 0.0 insufficient),
and a two-attempt limit. Every role uses `Qwen3_6-35B-A3B`, and the three arms
share one response-cache namespace.

| Method | EM | F1 | F1 minus correction's dense control (95% interval) | Live calls per question | Calls + cache hits per question |
| --- | ---: | ---: | --- | ---: | ---: |
| Dense | 0.200 | 0.274 | | 1.00 | 1.00 |
| Graded single pass | 0.200 | 0.274 | 0.000 | 5.00 | 6.00 |
| Corrective | 0.223 | 0.296 | +0.022 (-0.009 to 0.052) | 6.11 | 11.25 |

The paired F1 interval includes zero against both controls, so these runs do not
support an improvement. Correction remains an optional experiment and is not
recommended as the default. One corrective question hit the proxy's rate limit
and keeps zero credit in the denominator. This cohort's dense control, executed
separately from the core dense arm, matches it on the same questions within
-0.0004 F1. An earlier cohort on the first 300 questions in corpus order gave
+0.019 (-0.017 to 0.056).

See [the matched correction comparison](docs/COMPARISONS.md#matched-musique-evidence-correction-comparison)
for paired intervals, added cost, failures, and cache reuse. These controls are
kept separate from the full core comparison.

## Notebooks

Start with the [guide](GUIDE.md), then use the notebooks in order. Each combines
method explanations, worked examples, recorded results, and editable experiments.
Live execution is explicitly gated.

| Notebook | What you can explore |
| --- | --- |
| [01 · Query decomposition](01_query_decomposition.ipynb) | Independent search questions, pooling, fallback, and hop-count scores |
| [02 · Agentic retrieval](02_agentic_retrieval.ipynb) | Search/read tools, evidence gaps, bounded sessions, and abstentions |
| [03 · IRCoT](03_ircot.ipynb) | Evidence-dependent queries, enforced tool order, and retrieval budgets |
| [04 · GraphRAG head-to-head](04_graphrag_head_to_head.ipynb) | Traversable toy graph, full-split HippoRAG results, index coverage, and build/query cost |
| [05 · Decision grid](05_decision_grid.ipynb) | Corpus complexity, paired uncertainty, quality, failures, and cost |
| [99 · Starter agent](99_starter_agent.ipynb) | Editable retrieval policy, fresh question state, and a metered ADK experiment |

Default execution needs Python 3.12+, NumPy, `qdrant-client==1.19.1`, a Jupyter kernel,
and RI #2's reference artifacts, which `scripts/setup.sh` installs from release
`reference-ri2-35b43b26a35a`. It uses invented examples, committed results, and
the release's worked examples, with no credentials or model calls: each model step
goes through `load_or_run`, which replays the reference while `RUN_EXPENSIVE` is
`False`. Optional live cells use local pinned BGE embeddings and the Vector
proxy. On a Slurm cluster, launch kernels and all Python commands inside a compute
allocation, not on a login node.

## Datasets

- **MuSiQue:** pinned HippoRAG 2 packaging with 1,000 questions, 11,656 passages,
  and per-question hop graphs. The core comparison reports results by hop count.
- **NaturalQuestions:** pinned, digest-checked single-hop control with 1,000
  questions over 9,633 passages, accepted answer aliases, and support mappings.

[CORPUS.md](CORPUS.md) records source revisions, digests, attribution, and rights.

## Run an experiment

Inside compute, from the repository root with the bootcamp environment:

```bash
export PYTHONPATH=aieng-advanced-rag
python implementations/cross_document_synthesis/run_benchmark.py \
  --live --corpus musique --method ircot --per-category 10 \
  --output 'Arm Store/ri2-experiments/ircot.json'
```

Use `--per-category 0` for the full split. Live calls use the shared Vector
credential resolver, model-family selection, and spending controls described in
the [guide](GUIDE.md#models-credentials-and-spend). The [experiment reference](docs/EXPERIMENTS.md) covers
resumption, cache namespaces, score replay, and custom inputs.

## Files

- `synthesis_lab/` contains reusable methods, local embedding/runtime helpers,
  corpus adapters, and scoring.
- `run_benchmark.py`, `run_evaluation.py`, and `export_results.py` run matched
  experiments, replay comparisons, and export compact predictions.
- `run_dense_baseline.py`, `run_decomposition.py`, and `run_agent_lab.py` expose
  focused method demonstrations; `run_graph_comparison.py` compares retrieval
  using the checksum-pinned RI4 release. See the
  [graph comparison instructions](GUIDE.md#compare-the-prebuilt-graph-indexes)
  for atlas-rag HippoRAG 2 and LightRAG.
- [docs/METHODS.md](docs/METHODS.md) explains method settings and evidence flow;
  [outputs/](outputs/README.md) separates evaluations, examples, validation, and
  reference records.
- `synthesis_lab/reference.py` puts each notebook model step behind `load_or_run`;
  `build_reference.py` stores the reference results for the release, and
  `build_summaries.py` and `run_shared_300.py` rebuild the committed summaries.
- `tests/` checks contracts and replay; `benchmarks/` holds specialized retrieval,
  scorer, and answer audits.

Raw corpora, model weights, vectors, response caches, and operational logs stay in
ignored local storage.
