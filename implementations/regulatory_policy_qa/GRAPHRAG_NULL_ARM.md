# Null detection on ContractNLI: dense retrieval or naive GraphRAG?

## Measured-threshold replay

The replay at `sufficient=0.6`, `insufficient=0.0` scores dense and LightRAG on
the same 340 questions from the first 20 ContractNLI test contracts (179
answerable, 161 null). Verified cached evidence was reused without indexing
or retrieval.

The grader and generator use **Qwen3_6-35B-A3B**, matching the original model
identifier, with reasoning disabled, temperature 0, and a 512-token completion
cap. The original run used 0.6/0.3. Both current arms share the same answering
model, prompts, five-passage evidence budget, and decision policy.
Matching the model identifier and removing
sampling variance does not guarantee identical runs across serving backends.

| Arm | Abstain precision | Abstain recall | False-answer rate | Answerable abstention | Balanced decision accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.7241 | 0.7826 | 0.2174 | 0.2682 | 0.7572 |
| `lightrag-mix` | 0.6872 | 0.8323 | 0.1677 | 0.3408 | 0.7458 |
| `always-abstain` | 0.4735 | 1.0000 | 0.0000 | 1.0000 | 0.5000 |
| `always-answer` | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 |

The paired abstention decisions disagree on 65 questions. Dense alone
makes the correct decision on 35; LightRAG alone on 30.
Exact two-sided McNemar p = 0.6201. This comparison measures answer-versus-abstain
decisions; answer correctness is measured separately below.

The replay incurred 406 new model calls and 553,525
tokens, with 3,521 response-cache hits and zero empty responses.
Cache hits are reported separately, so paid calls are not an estimate of each
arm's independent cold cost. No new indexing or retrieval calls were made;
the historical LightRAG indexing bill remains 1,177,351 tokens.

Results: [`null-arm summary`](outputs/graphrag_null/measured_thresholds/null_arm_summary.json),
[`paired comparison`](outputs/graphrag_null/measured_thresholds/paired_comparison.json),
and [`provenance`](outputs/graphrag_null/measured_thresholds/replay_provenance.json).

Visible gold-span reach is 151/179 (0.8436) for dense and 133/179 (0.7430) for
LightRAG, measured within the generator's five passages of 1,200 characters each.
Span containment measures retrieval coverage, not answer correctness.

## Answer correctness on the answerable half

`evaluation.answers.judge_answer` now makes two separate assessments for each
of the **179 answerable questions per arm**, including pipeline abstentions:

- Semantic correctness receives only the question, gold reference, and saved
  answer; classification labels and verbatim supporting spans are not required.
- Grounding receives only the question, saved answer, and actual top five cached
  passages, truncated to 1,200 characters each as in generation. It receives no
  gold reference and can disagree with correctness.

Both passes use Qwen3_6-35B-A3B with reasoning disabled, temperature 0, and a
512-token completion cap. Judging uses the saved answers and evidence.

| Arm | Correct / all answerable | Correctness | Pipeline abstentions | Grounding | Hallucination among attempted answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | 122/179 | 0.6816 | 48 | 0.9832 | 0.0229 |
| `lightrag-mix` | 109/179 | 0.6089 | 61 | 0.9944 | 0.0085 |

Correctness uses all 179 answerables, including abstentions. Grounding includes
canonical pipeline abstentions as vacuously grounded; hallucination excludes
those same abstentions and uses 131 dense and 118 LightRAG attempted answers.
Grounding does not establish answer correctness. These automated judgments measure
agreement with ContractNLI annotations; generator and judges use the same model,
and no human adjudication was performed. The 20-contract sample does not establish
a population ranking. The McNemar test above applies only to abstention decisions.

Saved reports include judging usage and provenance. Cached costs do not estimate
independent cold-run costs. Prompts, gold passages, and response caches are not
published.

Judge answers after the replay as a separate step:

```bash
python implementations/regulatory_policy_qa/graphrag_null/judge_null_arm.py \
  --store ./unpacked
```

This scores correctness and grounding for dense and LightRAG using the saved
answers in `measured_thresholds/`. It makes API calls on judge-cache misses.
After judging once, add `--cache-only` to verify the saved reports without API
calls or replacing their provenance. The exported store holds the grading and
generation cache, not the separate judge cache.

Reports: [`judging summary and provenance`](outputs/graphrag_null/measured_thresholds/answer_judging_summary.json),
[`validation audit`](outputs/graphrag_null/measured_thresholds/answer_judging_validation.json),
[`dense verdicts`](outputs/graphrag_null/measured_thresholds/answer_judgements_dense.json),
and [`LightRAG verdicts`](outputs/graphrag_null/measured_thresholds/answer_judgements_lightrag-mix.json).

## Original five-arm experiment (0.6/0.3)

The remainder of this memo records the original five-arm experiment at 0.6/0.3
with Qwen3_6-35B-A3B. Its scope-control numbers apply to that run.

## Short answer

**The README's claim did not reproduce.** On ContractNLI, LightRAG does not
collapse on "not stated" questions. It matches dense retrieval: 0.614 abstain
precision against 0.611, 0.888 abstain recall against 0.870, and a 0.112
false-answer rate against 0.130. The two arms disagree on 60 of 340 abstain
decisions and are right and wrong in equal measure, 31 to 29, an exact
two-sided McNemar p of 0.897.

What decides null detection here is retrieval scope rather than retrieval
method. The same dense retriever, told nothing about which contract a question
belongs to, answers 75.8% of the unanswerable questions instead of 13.0%. That
control is the only arm in the table that collapses.

The claim's cost argument survives and is worth keeping. LightRAG buys nothing
on this task and charges 371 model calls and 1,177,351 tokens to index 20
contracts, against zero model calls for the dense index, which is local
embeddings. GraphRAG is still the wrong default for RI #1. The reason is price,
and the README has to stop saying it is accuracy. Replacement wording is in
[Replacement wording for the README](#replacement-wording-for-the-readme).

Raw reports: [`outputs/graphrag_null/`](./outputs/graphrag_null/). Every number
below comes from `null_arm_summary.json` or from the per-question records in
`outcomes_<arm>.json` beside it.

## What was measured

The corpus is the first 20 contracts of the pinned ContractNLI test split, in
file order: 214 chunks, 340 questions, 161 of them `NotMentioned`, a null rate
of 0.4735. The full test split is 123 contracts at a 0.4319 null rate, so this
slice is 16% of it and runs a little heavier on nulls, which raises the
always-abstain floor rather than favouring any arm. Documents were taken in file
order and not selected on anything the measurement is about. The cut is the
LightRAG indexing bill: projecting the measured rate to all 123 contracts gives
2,471 calls and 7.8M tokens, at 4.3 hours of indexing on this machine.

The pipeline is `AdvancedRagSpine` at the shipped defaults: `top_k` 10,
`evidence_budget` 5, `require_citations` on, `LLMGrader` with `max_chunks` 5,
and `GradingThresholds(0.6, 0.3)`. No rewriter is configured. The one model-free
rewriter that ships carries the CUAD pattern list and changes 0 of these 340
questions, so `max_attempts=2` settles on its final-attempt rule and the
corrective retry never fires. Every arm reaches the same grader, the same
generation template, and the same abstention paths. Only the evidence differs.

The two thresholds are unmeasured placeholders. They ship at 0.6 and 0.3 and are
under separate measurement in the threshold sweep, so every number here is
conditional on that pair. A sweep that moves them moves every row, and it moves
them together: nothing in this comparison rests on the thresholds being right,
only on both arms meeting the same ones.

Model calls go to `Qwen3_6-35B-A3B` through the Vector proxy at temperature 0,
with reasoning disabled on every call through `reasoning.disable_thinking`.
Embeddings are local `BAAI/bge-small-en-v1.5` on CPU with its published query
prompt. This ran on one developer machine, so the elapsed times hold for that
machine only. The proxy's rate limit is
per key and was shared with another job throughout, which puts 429s and their
retries inside every elapsed figure here. Call counts and token counts are
unaffected.

| Arm | Evidence |
| --- | --- |
| `dense` | Harness dense retrieval, scoped to the question's own contract. The RI #1 substrate. |
| `lightrag-mix` | LightRAG `mix` mode over a per-contract graph, the adapter's default. |
| `shuffled` | The right contract's chunks in a fixed random order: same candidate pool, no ranking signal. |
| `dense-unscoped` | Dense retrieval over all 20 contracts at once. The scope control. |
| `no-retrieval` | Nothing retrieved, so the pipeline abstains with no model call. |
| `always-abstain` | Decides abstain on every question. |
| `always-answer` | Decides answer on every question. |

LightRAG was given one graph per contract, which hands it the same retrieval
scope the dense arm gets for free. That is the configuration most favourable to
it, and it was chosen deliberately: a corpus-wide graph would have lost on the
scope confound that `dense-unscoped` measures separately, and the result would
have said nothing about graphs. It also matters mechanically, because LightRAG
keeps its default storages in process-global state, so per-contract graphs need
distinct workspaces or each one silently accumulates every contract indexed
before it. The evidence tables confirm the isolation held: none of the 2,499
retrieved passages came from a contract other than the question's.

## Null detection

`evaluation.metrics.score_null_detection`, on all 340 questions.

| Arm | Abstain precision | Abstain recall | False-answer rate | Decision accuracy |
| --- | ---: | ---: | ---: | ---: |
| `dense` | 0.611 | 0.870 | 0.130 | 0.676 |
| `lightrag-mix` | 0.614 | 0.888 | 0.112 | 0.682 |
| `shuffled` | 0.547 | 0.932 | 0.068 | 0.603 |
| `dense-unscoped` | 0.481 | 0.242 | 0.758 | 0.518 |
| `no-retrieval` | 0.473 | 1.000 | 0.000 | 0.474 |
| `always-abstain` | 0.473 | 1.000 | 0.000 | 0.474 |
| `always-answer` | 0.000 | 0.000 | 1.000 | 0.526 |

Decision accuracy is the share of questions where the abstain decision matched
the label, computed from the per-question records. It is the one column a row
cannot win by refusing to answer.

Read the bottom three rows first. Always abstaining scores a perfect abstain
recall and a zero false-answer rate, with precision equal to the null rate. Two
of the columns that look like the headline are free. The `shuffled` control,
which has no ranking signal at all, beats both real arms on recall and on
false-answer rate for the same reason.

Between the two arms the claim is about, there is nothing to choose. LightRAG is
ahead on all three columns by between 0.003 and 0.018, which on 161 null
questions is two or three questions. The 95% intervals overlap almost entirely:
dense abstains on 140 of 161 nulls, CI 0.809 to 0.913; LightRAG on 143, CI 0.830
to 0.928. Paired on the same questions, they disagree 60 times, LightRAG right
on 31 of those and dense right on 29.

## What each arm costs on the answerable half

The 179 questions ContractNLI labels `Entailment` or `Contradiction`. Span reach
is the share where a human-annotated evidence span reached the generator;
answered with span is the share where it reached the generator **and** the arm
answered.

| Arm | Answerable abstention | Span reach | Answered with span |
| --- | ---: | ---: | ---: |
| `dense` | 0.497 | 0.844 | 0.453 |
| `lightrag-mix` | 0.503 | 0.866 | 0.475 |
| `shuffled` | 0.693 | 0.503 | 0.274 |
| `dense-unscoped` | 0.235 | 0.168 | 0.134 |
| `no-retrieval` | 1.000 | 0.000 | 0.000 |
| `always-abstain` | 1.000 | 0.000 | 0.000 |
| `always-answer` | 0.000 | 0.000 | 0.000 |

This is the column that disposes of `shuffled`. It abstained on 69.3% of the
questions that had an answer, which is what bought its 0.932 abstain recall.
`always-abstain` shows the same trade at its limit. Neither is a null detector;
both are one refusal counted twice.

It also shows how much both real arms give up. Half the answerable questions get
no answer: dense abstains on 89 of the 179 and LightRAG on 90, and in both arms
70 of those abstentions happened with a gold span already retrieved. Those are
grading-threshold losses rather than retrieval losses, because the pooled grade
fell under 0.3 and the pipeline abstained before generating. That is the number
the threshold sweep exists to move, and it is the largest quality defect visible
in this run.

Span reach is measured by text containment rather than by chunk identifier,
because LightRAG chunks a contract its own way and its identifiers are not the
harness's. On the three arms where both measures exist, the containment rule
agrees with the harness judgment set on 0.911, 0.883, and 0.978 of answerable
questions, and it errs low every time: 0.844 against a 0.933 identifier hit rate
on the dense arm. Treat the span-reach column as a floor.

## Why the arms land where they do

The pooled evidence grade separates null from answerable questions in every arm
scoped to the right contract, and in none that is not.

| Arm | Mean pooled grade, null | Mean pooled grade, answerable | Separation |
| --- | ---: | ---: | ---: |
| `dense` | 0.106 | 0.475 | 0.369 |
| `lightrag-mix` | 0.112 | 0.436 | 0.324 |
| `shuffled` | 0.065 | 0.263 | 0.198 |
| `dense-unscoped` | 0.708 | 0.704 | -0.004 |

`dense-unscoped` is the whole mechanism in one row. ContractNLI asks the same 17
hypotheses of every contract, so for almost any question there is a contract in
the slice that does state the hypothesis. An unscoped retriever finds that
passage, the grader reads it as strong evidence, and the pipeline answers a
question about contract A out of contract B. Its grades on unanswerable
questions are indistinguishable from its grades on answerable ones, so it has no
signal to abstain on, and it answers 122 of the 161 nulls.

Dense and LightRAG separate the two classes by a similar margin and make similar
decisions. LightRAG puts about 30% more text in front of the reader at the same
k, 1,522 characters per passage against 1,175, because its 300-token chunking
produced wider passages than the harness's 1,200-character chunks. It does not
spend that extra context on false answers.

## Cost

Counted at the client. Cached calls are recorded as hits and excluded from the
call and token totals, so every figure is the cost of a cold run.

| Phase | Calls | Tokens | Elapsed |
| --- | ---: | ---: | ---: |
| LightRAG indexing, 20 contracts | 371 | 1,177,351 | 2,342 s |
| LightRAG querying, 340 questions | 17 distinct | 12,403 | 46 s |
| `dense` pipeline | 1,580 | 797,665 | 1,413 s |
| `lightrag-mix` pipeline | 1,645 | 842,222 | 1,355 s |
| `shuffled` pipeline | 646 | 348,152 | 510 s |
| `dense-unscoped` pipeline | 52 | 79,504 | 47 s |
| `no-retrieval` pipeline | 0 | 0 | 0 s |

Indexing works out at 18.6 calls and 58,868 tokens per contract, or 1.79 calls
and 5,683 tokens per thousand characters. Per document that is twice what the
ServerFault comparison measured, and per character it is less than half, because these
contracts are about five times longer and LightRAG's fixed per-document work
amortises. Dense indexing costs **zero model calls**: the index is local
sentence-transformers embeddings, and the same 207,179 characters embed on CPU
with no proxy traffic at all.

Querying is one call per question, LightRAG's keyword extraction, made before
generation is reached at all. On this corpus that collapses to 17 distinct calls
because the 17 hypotheses repeat verbatim across contracts and a response cache
answers the rest; the 323 cache hits are recorded in the summary. Do not read 17
as the cost on a corpus with distinct questions. The fan-out experiment
disqualified Microsoft GraphRAG at 45 calls and 590k tokens per question, and
LightRAG's query side is nothing like that. It is not what rules LightRAG out
here either.

The falling call counts down the pipeline column are cache hits rather than
cheaper arms. Arms that retrieve overlapping passages share graded prompts: the
`dense-unscoped` arm cost 52 calls because 1,908 of its prompts had already been
answered for an earlier arm. Read about 1,600 calls as the figure for an arm run
first. A question costs one grading call per retrieved passage up to `max_chunks`
of 5, and one generation call only if it reaches generation, which on the dense
arm 136 of 340 did.

## What this means for RI #1

Three things follow.

The accuracy half of the claim is gone. There is no measured sense in which a
tuned stack answers where naive GraphRAG collapses on this corpus. Teaching that
order on this justification would be teaching a result the repository's own
harness contradicts.

The cost half is stronger than the claim stated. A graph over these contracts
costs 1.18M tokens to build and buys a difference of two questions in 340. RI #1
should still come before the GraphRAG material, and the reason to say out loud
is that a plain chunk index scoped to the right document already does this job
for nothing. That is the same finding the ServerFault comparison reached by a different
route, and it now holds on a second corpus and a different task.

Retrieval scope is the teachable result here, and it is not in the curriculum.
The gap between `dense` and `dense-unscoped` is 0.130 against 0.758 on the
false-answer rate, far larger than anything separating the retrieval methods.
Scoping a retriever to the document a question is about is worth more on a null
detector than every technique RI #1 layers on top of it, and
`outcomes_dense-unscoped.json` is a ready demonstration.

## What this result means for RI #1

RI #1's README describes null detection on its own terms, and its status section
carries this result: LightRAG and dense retrieval score the same abstain
decision, retrieval scope is what separates the arms, and GraphRAG stays out of
RI #1 on cost.

## Limits

- 20 contracts of 123, one run. The slice was cut to fit LightRAG's indexing
  bill and is 16% of the test split, at a 0.4735 null rate against the split's
  0.4319. The dense and LightRAG arms are indistinguishable at this sample size,
  which bounds how large any real difference can be rather than proving there is
  none: the confidence interval on abstain recall spans about ten points.
- The thresholds are placeholders. 0.6 and 0.3 ship untuned and are under
  separate measurement. They are the largest single lever on these numbers,
  since 70 of each real arm's 89 or 90 abstentions on answerable questions
  happened with a gold span already retrieved.
- One LightRAG mode, one GraphRAG library. `mix` only, which is the adapter's
  default and combines graph and vector retrieval. `local`, `global`, and
  `naive` were not run, and neither were HippoRAG 2 or Microsoft GraphRAG.
- LightRAG got the retrieval scope for free, through per-contract graphs. A
  corpus-wide LightRAG index was not built, on the superlinear indexing time seen in the
  ServerFault comparison, so how a genuinely naive deployment behaves is answered only
  indirectly, through `dense-unscoped`.
- No judge on answer correctness. Answered with span says a gold span was in the
  prompt and the arm answered. It does not say the answer was right.
  `evaluation.answers.judge_answer` reads the same evidence and would close
  this.
- The question is an NLI hypothesis. ContractNLI states a claim rather than
  asking a question, and the shipped generation template asks the model to
  answer it as one. Both arms get the identical prompt, so the comparison is
  unaffected, but the absolute correctness numbers would move under a template
  written for this corpus.
- Elapsed figures are not measurements. The proxy's rate limit is per key and
  was shared with another job throughout, so 429s and their retries are inside
  every elapsed time here. Calls and tokens are unaffected.

## Follow-ups

1. **Done:** re-run the two real arms at the measured thresholds
   from the copied store without re-indexing. The current results are above.
2. **Done:** `evaluation.answers.judge_answer` scored all 179
   answerables per arm in the measured-threshold replay, including abstentions,
   at the same five-passage evidence budget. Verdicts and provenance are above.
3. **Runner ready; measurement deferred:** full 123-contract dense-arm and threshold
   measurements support cache-only audit/replay. Existing traces cover 680 of 2,091 queries,
   and dense evidence covers 340. No full-split quality score is inferred. See
   [the coverage audit](outputs/full_contract_nli/cache_status.json).
4. **Done:** [`05_answering_and_null_detection.ipynb`](05_answering_and_null_detection.ipynb)
   makes the dense-unscoped scope confound an inspectable exercise. The historical false-answer
   rates of 0.130 and 0.758 belong to the original operating point, not a new full-split run.
5. **Bounded pilot complete:** the first five CUAD test contracts supply 205 queries, including
   70 answerables judged per arm. [The comparison](outputs/cuad_null/COMPARISON.md) records
   paired decisions, source budgets, indexing-cost lower bounds, and extraction warnings.
   This nonrandom pilot does not establish a whole-CUAD or population ranking.

## Reproducing this

```bash
python implementations/regulatory_policy_qa/graphrag_null/run_null_arm.py --documents 20
```

Needs `lightrag-hku`, `sentence-transformers`, `numpy`, and `openai`, plus a
proxy key: `VECTOR_PROXY_API_KEY` in the environment or the repository `.env`,
or `OPENAI_API_KEY` exported together with an `OPENAI_BASE_URL` on the proxy. Add `--estimate-only` to print
the projected LightRAG indexing bill without spending it. The three phases skip
themselves when their outputs exist, so a re-run scores without re-indexing, and
`--force-index` and `--force-retrieve` override that. Offline tests for the
measurement pieces are in `graphrag_null/tests/`.

## Score it on another machine

The store is not committed, because it holds contract text and whole model
prompts. Scoring needs only the retrieval tables and response cache. An unpacked
score-only bundle automatically forbids indexing and retrieval, and fails before
model access if a requested evidence table is missing or invalid.

Build and check a bundle:

```bash
python implementations/regulatory_policy_qa/graphrag_null/export_store.py --out null-arm-store.tar.gz
```

```bash
python implementations/regulatory_policy_qa/graphrag_null/export_store.py --verify null-arm-store.tar.gz
```

Verification checks the archive against its manifest; it does not extract it.
Unpack it and run from the repository root:

```bash
mkdir -p unpacked
tar -xzf null-arm-store.tar.gz -C unpacked
python implementations/regulatory_policy_qa/graphrag_null/run_null_arm.py --store ./unpacked
```

The runner defaults to all five retrieval arms plus always-answer/always-abstain
controls, 20 contracts, Qwen3_6-35B-A3B, eight workers, and thresholds 0.6/0.0.
It writes results to `outputs/graphrag_null/measured_thresholds/`, preserving the
historical reports in the parent directory. `--out` chooses another destination.
The pinned corpus downloads on first use; grading and generation reuse the
response cache and call the proxy on misses, using `VECTOR_PROXY_API_KEY`.
Answer correctness and grounding judging remain a separate step described above.

`--full` adds the LightRAG indexes for stores that need to retrieve again.

### The durable path

Handing a file over works once. For repeated use the bundle goes through the
shared artifact cache, which is the same two calls RI #4's two MuSiQue indexes
take, so solving it here solves it there.

The producing machine writes its local tier out with
`TieredResultCache.export_local(destination)`. The files are uploaded, and the
URLs and digests go into a manifest built from
`aieng/advanced_rag/cache/global-artifacts.example.json`. Any other machine then
runs `python -m aieng.advanced_rag.cache <manifest>`, which fetches each
artifact and verifies its digest before writing it.

The identity of this artifact is fixed, so a manifest entry can be written
before anything is hosted:

```python
CacheKey(
    ri_id="ri-1",
    artifact_type="graphrag-null-store",
    producer_id="lightrag-contract-nli",
    config={"documents": 20, "split": "test", "bundle": "score-only"},
)
```

Nothing is hosted yet, so the manifest ships empty and the handover is the
bundle above. The key and the procedure are what make the switch a change of
manifest rather than a change of code.
