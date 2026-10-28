# Matched full-corpus comparisons

Dense, the independent decomposition reference, and bounded ADK IRCoT answer the
same 1,000 questions within each pinned corpus. Every method searches the full
corpus and uses the same final five-passage answer template. Rates are fractions.

| Corpus | Method | Questions | EM | F1 | Recall@5 | Generation all-supports | Failure rate | Live calls | Cache hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | dense | 1000 | 0.2010 | 0.2733 | 0.5060 | 0.1710 | 0.0000 | 0 | 1000 |
| musique | decomposition | 1000 | 0.2060 | 0.2805 | 0.5540 | 0.1880 | 0.0000 | 0 | 2000 |
| musique | ircot | 1000 | 0.4090 | 0.4854 | 0.6823 | 0.4300 | 0.0090 | 6262 | 73 |
| nq | dense | 1000 | 0.5070 | 0.6302 | 0.6824 | 0.3510 | 0.0000 | 1000 | 0 |
| nq | decomposition | 1000 | 0.5010 | 0.6249 | 0.6739 | 0.3440 | 0.0000 | 1261 | 739 |
| nq | ircot | 1000 | 0.4840 | 0.6036 | 0.6812 | 0.3230 | 0.0060 | 4444 | 220 |

## Paired differences

Differences are arm minus dense. The 95% intervals resample the same question
IDs together, with 5,000 bootstrap samples and the fixed evaluation seed.

| Corpus | Method | F1 difference | 95% interval |
| --- | --- | ---: | --- |
| musique | decomposition | 0.0072 | [-0.0127, 0.0272] |
| musique | ircot | 0.2122 | [0.1828, 0.2418] |
| nq | decomposition | -0.0053 | [-0.0149, 0.0036] |
| nq | ircot | -0.0266 | [-0.0415, -0.0115] |

## What the results support

IRCoT improves MuSiQue F1 and supporting-passage recall, while reducing F1
on the single-hop control. The additional retrieval loop is useful for this
multi-hop workload, with substantial additional model calls. It is not a
general accuracy improvement across both corpora. The decomposition intervals
include zero, so these runs do not establish a decomposition gain.

## Read the comparison

The [machine-readable summary](../outputs/evaluation/summary.json) contains
hop/category breakdowns, retrieval at 5/10/20, generation coverage, citations,
abstentions, stop reasons, and usage. All six final reports cover every selected
question. Failed sessions receive zero answer and retrieval credit and remain
in the denominators. Execution-success flags remain separate from coverage.
Compact predictions replay against pinned annotations; full agent traces remain
in the local cache.

Live calls and cache hits describe recorded execution rather than cold-run cost.
[Original reference usage](../outputs/reference/execution_usage.json) matches the
replayed reference predictions. Unfinished provider requests can add unrecorded
usage, so totals are not complete billing estimates.

The served model alias does not pin weight revisions. Independently executed arms
can span backend deployments, and temperature zero does not guarantee identical
responses. Paired intervals describe the recorded predictions; they do not remove
provider drift or isolate every retrieval-policy choice.

NaturalQuestions is the single-hop control, despite multiple alternative marked
supporting passages. Its all-supports coverage is a strict diagnostic, not a
claim that every marked passage is necessary to answer.

## Prebuilt graph retrieval on MuSiQue

The released HippoRAG 2 index uses **atlas-rag's entity/relation variant with
query2edge**, rather than the authors' paper configuration. It covers all 11,656
original passages. This arm answers the same 1,000 MuSiQue questions as dense,
retrieves 20 passages, and uses the shared answer template with five original
passages. Intermediate graph facts guide ranking; final citations refer to the
original corpus.

| Method | Questions | EM | F1 | Recall@5 | Generation all-supports | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 1000 | 0.2010 | 0.2733 | 0.5060 | 0.1710 | 0.0000 |
| HippoRAG 2 (atlas-rag) | 1000 | 0.3140 | 0.4073 | 0.6254 | 0.3230 | 0.0000 |

The paired F1 difference, graph minus dense, is 0.1340 (95% interval [0.1088, 0.1600]). This interval supports a quality gain for these recorded predictions.

The query uses the pinned local BGE revision. The producer artifact records the
encoder name but not its revision, so identical producer and query embedding
weights cannot be established from its metadata. The served answer-model alias
also does not pin weights. These paired intervals describe recorded predictions
and do not remove provider drift between independently executed arms.

The graph arm records 1,998 live model calls, two cache hits, and 2,523,840
live tokens: two logical requests per question, compared with one for dense.
These totals include retrieval and final generation. The
[replayable graph summary](../outputs/evaluation/graphrag/summary.json) records
live calls, cache hits, tokens, wall time, per-hop paired intervals, and the dense
reference's original execution separately from its cached replay. Cache hits are
logical requests, not free cold-run work. Index construction is reported
separately in the notebook and is not included in query cost.

### LightRAG over the complete index

LightRAG in `mix` mode runs over RI #4's release `reference-ri4-bd05f7bb16c6`,
whose index holds chunks and vectors for all 11,656 passages. Strict validation
passed: every passage maps to stored chunks, so no question is set aside. It
answers the same 1,000 MuSiQue questions with the shared five-passage reader,
retrieving 40 chunk hits and deduplicating them to at most 20 original passages.
No reranking model is configured.

| Method | Questions | EM | F1 | Recall@5 | Generation all-supports | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 1000 | 0.2010 | 0.2733 | 0.5060 | 0.1710 | 0.0000 |
| LightRAG (mix) | 1000 | 0.1950 | 0.2655 | 0.4663 | 0.1370 | 0.0000 |

The paired F1 difference, LightRAG minus dense, is -0.0077 (95% interval
[-0.0236, 0.0079]). The interval includes zero, so these predictions do not
establish a difference from dense; supporting recall at 5 is 0.040 lower. F1 by
hop count is 0.3048, 0.2320, and 0.2069 for two, three, and four hops.

The run records 1,998 live calls, one cache hit, and 1,629,760 live tokens: one
keyword-extraction call and one answer per question. It ran with
`Qwen3_6-35B-A3B` for every role, after the HippoRAG 2 and core arms, so provider
drift between independently executed arms is not removed.

The [LightRAG report](../outputs/evaluation/graphrag/lightrag.json) and the
[graph summary](../outputs/evaluation/graphrag/summary.json) hold per-question
evidence, metering, per-hop metrics, and the index coverage check.

## Matched MuSiQue evidence-correction comparison

This separately executed cohort answers the shared 300 MuSiQue questions (150
two-hop, 75 three-hop, and 75 four-hop, drawn by `stratified_test_set`) with
dense, graded single-pass, and corrective answering. It uses the fixed thresholds
(0.6 sufficient, 0.0 insufficient), at most two attempts, and the same encoder,
final answer prompt, five-passage budget, and `Qwen3_6-35B-A3B` for generation,
extraction, and grading. The three arms ran in one response-cache
namespace, isolated from the core comparison. No thresholds were selected using
these answers.

| Method | Questions | EM | F1 | Recall@20 | Abstained | Failure | Live calls | Cache hits | Recorded tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 300 | 0.2000 | 0.2742 | 0.6275 | 0.4267 | 0.0000 | 300 | 0 | 269,117 |
| graded-single-pass | 300 | 0.2000 | 0.2742 | 0.6275 | 0.4267 | 0.0000 | 1,500 | 300 | 514,287 |
| corrective | 300 | 0.2233 | 0.2959 | 0.6250 | 0.3733 | 0.0033 | 1,832 | 1,544 | 715,037 |

Differences below are corrective minus its matched control, from
[`shared_300.json`](../outputs/evaluation/shared_300.json). The
[cohort summary](../outputs/evaluation/corrective/summary.json) resamples the same
pairs in corpus order, and its upper bound differs in the third decimal (0.0526).

| Control | Paired F1 difference | 95% interval | Added live calls | Added calls + hits |
| --- | ---: | --- | ---: | ---: |
| dense | 0.0217 | [-0.0093, 0.0519] | 1,532 | 3,076 |
| graded-single-pass | 0.0217 | [-0.0093, 0.0519] | 332 | 1,576 |

The paired F1 intervals do not support an improvement against either control.
Correction remains an optional experiment and is not recommended as the default.
One corrective question failed on a proxy rate limit (HTTP 429) and keeps zero
answer and retrieval credit; failed questions are not rerun selectively. This
cohort's dense control matches the core dense arm on the same 300 questions:
paired F1 difference -0.0004 (-0.0118 to 0.0097), although the two ran
separately with separate response caches.

An earlier cohort on the first 300 questions in pinned corpus order, chosen after
execution began, gave +0.0194 (-0.0171 to 0.0556) against both controls. Only 97
of those questions are in the shared 300. The shared-300 run replaces it so that
correction sits on the draw RI #4 and RI #5 report.

Dense and graded single-pass share identical final-generation requests, so the
graded control can reuse the dense answers. Their identical answers do not imply
that grading is free. Correction also reuses exact first-pass grading requests;
new rewrites and second-pass grading/generation are charged when executed.
The response cache is isolated from the core comparison and shared only
within this cohort. Request counts including hits describe logical work; recorded
live tokens exclude the original cost of cache hits and are not a cold-cost estimate.

The grader pools single-passage scores. That can misjudge a question whose support
is sufficient only when passages are read together. Inspect whether a retry found
a missing bridge rather than assuming every low grade diagnoses a retrieval error.
The [compact cohort reports](../outputs/evaluation/corrective/) and
[comparison summary](../outputs/evaluation/corrective/summary.json) replay against
pinned annotations. The subset has its own decision grid and matched controls.
The [30-question smoke run](../outputs/examples/smoke/comparison.json) demonstrates
integration; use the matched shared-300 comparison to assess correction.
