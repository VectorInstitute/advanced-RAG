# RI #1: the corrective loop's grading thresholds

## Short answer

**Curriculum decision: do not configure the corrective loop in RI #1's recommended
participant path.** Retain it only as an opt-in A/B measurement. The generator-only control scores
0.780 balanced accuracy with 680 model calls; the best `LLMGrader` loop scores 0.788 with 6,259.
A paired test cannot tell the 0.008 gain from zero (exact McNemar p = 0.26 over 680 paired questions, 95% paired bootstrap interval -0.004 to 0.021), and the loop makes about 9.2 times the calls.
The comparison is in `outputs/paired_corrective_loop.json`, written by `paired_corrective_loop.py`.
The corresponding runtime configuration is one attempt with `SingleShotPolicy`, a model-free
`HeuristicGrader`, and no corrective rewriter; the grade is recorded but cannot pre-empt generation.

If the optional experiment is run, lower the rewrite threshold from 0.3 to 0.0 and leave the answer threshold at
0.6. The rewrite threshold is the only one of the two that moves the result, and
every setting of it that makes the grader abstain costs more answerable
questions than it saves false answers. On ContractNLI at `max_attempts` 2 with
`LLMGrader`, the old 0.3 catches 20 more of 322 null questions than 0.0 does and
refuses 43 more of 358 answerable ones. CUAD orders the grid the same way by a
wider margin, and the out-of-domain control is unaffected because the generator
already declines every far null on its own. The answer threshold is flat from
0.2 to 1.0 on ContractNLI, so there is no measured reason to move it.

The larger finding is that the grader is not what detects a missing answer. The
same pipeline with no grader at all, one retrieval and one generation, scores
0.780 balanced accuracy on ContractNLI. The best `HeuristicGrader` setting
reaches 0.783 and the best `LLMGrader` setting 0.788, and `LLMGrader` spends
about nine model calls per query to get there. The generator's own "not stated"
path is doing nearly all of the work, and the citation gate inside it is a worse
trade than either threshold.

The defaults in `pipeline/grading.py` are changed to match for callers that explicitly enable the
loop. `PIPELINE.md` and the README record why RI #1 itself does not recommend
that configuration.

## What was measured

ContractNLI test split, the first 40 documents in file order: 680 queries, 451
chunks of 1,200 characters with 200 of overlap, 322 queries labelled
`NotMentioned` for a 47.4% null rate. The full split is 43.2% null, so this
slice is a little more null-heavy than the corpus it comes from.

Retrieval is dense over `BAAI/bge-small-en-v1.5`, run locally, scoped to one
document, top-10 with the top 5 placed in the generation prompt. Grading and
generation go through `Qwen3_6-35B-A3B` on the Vector proxy with reasoning
disabled, `max_tokens` 512 and temperature 0. `LLMGrader` grades the top 5
passages, one call each. The corrective retry uses `LLMRewriter`.

The grid is 12 answer thresholds from 0.2 to 1.0 against 9 rewrite thresholds
from 0.0 to 0.8, dropping inverted pairs, at `max_attempts` of 1 and 2. Both
axes carry the values that shipped before this run, 0.6 and 0.3.

The sweep does not re-run the pipeline once per grid point. Thresholds only turn
a pooled grade into a verdict, and nothing upstream of that turn depends on
them, so one harvest records the per-passage scores and the generation outcome
for both attempts of every query, and the grid is replayed over those records.
`aieng-advanced-rag/tests/aieng/advanced_rag/evaluation/test_sweep.py` asserts
that the replay reproduces `AdvancedRagSpine.answer` at every point on the grid,
because a replay that drifted from the spine would report an operating point the
pipeline does not have. Re-scoring all three corpora from the cached harvest
takes about twenty seconds and no model calls.

Raw results are in `outputs/sweep_contract-nli.json`, `outputs/sweep_cuad.json`
and `outputs/sweep_multihop-null.json`, and the harvested traces every row is
computed from are in `outputs/traces_*.json`. Every table below is rendered from
those files by `summarise_sweep.py`.

Reproduce a table with, for example:

```bash
python3 summarise_sweep.py outputs/sweep_contract-nli.json --grader llm --attempts 2 --sufficient 0.6
```

### Reading the columns

Abstention scored on its own is maximised by abstaining on everything, so every
row carries both sides. "Answerable lost" is the share of questions with a gold
span that the pipeline declined to answer, which is what the operating point
costs. "Grounded answers" is the share answered with a gold chunk among the five
passages the generator was shown, which is the stricter reading of the same
side. Balanced accuracy is the mean of abstention recall and the answerable
answer rate: it weights the two errors equally, ignores the base rate, and puts
both trivial controls on 0.500.

Three controls appear under every table. "No grader" is the pipeline with the
grading verdict removed, one retrieval and one generation per query. It is the
number a threshold has to beat, and the two trivial controls cannot supply it.

### ContractNLI, the rewrite threshold, `HeuristicGrader`

Answer threshold held at 0.6, `max_attempts` 2.

| Thresholds | Attempts | Abstain recall | Abstain precision | False-answer rate | Answerable lost | Grounded answers | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 / 0.00 | 2 | 0.829 | 0.738 | 0.171 | 0.265 | 0.715 | 0.782 |
| 0.60 / 0.10 | 2 | 0.829 | 0.738 | 0.171 | 0.265 | 0.715 | 0.782 |
| 0.60 / 0.20 | 2 | 0.829 | 0.738 | 0.171 | 0.265 | 0.715 | 0.782 |
| 0.60 / 0.30 | 2 | 0.829 | 0.738 | 0.171 | 0.265 | 0.715 | 0.782 |
| 0.60 / 0.40 | 2 | 0.832 | 0.736 | 0.168 | 0.268 | 0.712 | 0.782 |
| 0.60 / 0.50 | 2 | 0.848 | 0.695 | 0.152 | 0.335 | 0.648 | 0.756 |
| 0.60 / 0.60 | 2 | 0.904 | 0.648 | 0.096 | 0.441 | 0.545 | 0.731 |
| no grader | 1 | 0.845 | 0.727 | 0.155 | 0.285 | 0.698 | 0.780 |
| always answer | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.922 | 0.500 |
| always abstain | 1 | 1.000 | 0.473 | 0.000 | 1.000 | 0.000 | 0.500 |

The old 0.3 is indistinguishable from 0.0 for this grader. Term coverage never
scores a ContractNLI passage below 0.20, so any rewrite threshold at or below
0.2 is already "never insufficient", and 0.3 reaches only the few queries
scoring between 0.2 and 0.3.

### ContractNLI, the rewrite threshold, `LLMGrader`

Answer threshold held at 0.6.

| Thresholds | Attempts | Abstain recall | Abstain precision | False-answer rate | Answerable lost | Grounded answers | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 / 0.00 | 2 | 0.845 | 0.737 | 0.155 | 0.271 | 0.718 | 0.787 |
| 0.60 / 0.10 | 2 | 0.907 | 0.676 | 0.093 | 0.391 | 0.598 | 0.758 |
| 0.60 / 0.30 | 2 | 0.907 | 0.676 | 0.093 | 0.391 | 0.598 | 0.758 |
| 0.60 / 0.50 | 2 | 0.907 | 0.676 | 0.093 | 0.391 | 0.598 | 0.758 |
| 0.60 / 0.60 | 2 | 0.966 | 0.576 | 0.034 | 0.640 | 0.355 | 0.663 |
| 0.60 / 0.00 | 1 | 0.845 | 0.727 | 0.155 | 0.285 | 0.698 | 0.780 |
| 0.60 / 0.30 | 1 | 0.901 | 0.629 | 0.099 | 0.478 | 0.514 | 0.711 |
| 0.60 / 0.60 | 1 | 0.972 | 0.555 | 0.028 | 0.701 | 0.296 | 0.635 |
| no grader | 1 | 0.845 | 0.727 | 0.155 | 0.285 | 0.698 | 0.780 |
| always answer | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.922 | 0.500 |
| always abstain | 1 | 1.000 | 0.473 | 0.000 | 1.000 | 0.000 | 0.500 |

Rows between 0.10 and 0.50 are identical because `LLMGrader` returns only the
three values its prompt anchors, 0.0, 0.5 and 1.0. A threshold matters here only
by which side of 0.5 and 1.0 it falls on, so 86 valid grid points collapse to
five distinct outcomes at `max_attempts` 2 and three at `max_attempts` 1.

### ContractNLI, the answer threshold

Rewrite threshold held at the old 0.3, `HeuristicGrader`, `max_attempts` 2.

| Thresholds | Attempts | Abstain recall | Abstain precision | False-answer rate | Answerable lost | Grounded answers | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 / 0.30 | 2 | 0.845 | 0.727 | 0.155 | 0.285 | 0.698 | 0.780 |
| 0.40 / 0.30 | 2 | 0.845 | 0.729 | 0.155 | 0.282 | 0.698 | 0.781 |
| 0.50 / 0.30 | 2 | 0.839 | 0.730 | 0.162 | 0.279 | 0.701 | 0.780 |
| 0.60 / 0.30 | 2 | 0.829 | 0.738 | 0.171 | 0.265 | 0.715 | 0.782 |
| 0.70 / 0.30 | 2 | 0.832 | 0.732 | 0.168 | 0.274 | 0.710 | 0.779 |
| 0.75 / 0.30 | 2 | 0.842 | 0.732 | 0.158 | 0.277 | 0.710 | 0.783 |
| 0.80 / 0.30 | 2 | 0.842 | 0.731 | 0.158 | 0.279 | 0.707 | 0.781 |
| 0.85 / 0.30 | 2 | 0.835 | 0.727 | 0.165 | 0.282 | 0.704 | 0.777 |
| 0.90 / 0.30 | 2 | 0.835 | 0.733 | 0.165 | 0.274 | 0.712 | 0.781 |
| 1.00 / 0.30 | 2 | 0.835 | 0.733 | 0.165 | 0.274 | 0.712 | 0.781 |

The whole axis spans 0.777 to 0.783, inside the sampling error of this slice.

## What the operating point costs on answerable questions

At the recommended 0.6 / 0.0 with `max_attempts` 2 and `LLMGrader`, the pipeline
answers "Not stated in the provided sources" to 97 of the 358 answerable
questions, 27.1% of them, and answers 71.8% with a gold chunk among the five
passages the generator was shown. The ceiling is 92.2%: that is the share of
answerable questions where retrieval put a gold chunk in the top 5 at all, so
20.4 points of the loss are retrieval and the rest is the generator declining
evidence it was given. On the other side it answers 50 of the 322 null
questions, a false-answer rate of 15.5%.

The old 0.3 buys 20 more null questions caught, 292 against 272, and pays 43
more answerable questions refused, 140 against 97. That is one false answer
prevented for every 2.15 answerable questions turned away. Moving further, to
0.6, prevents one for every 3.4 turned away. Balanced accuracy weights the two
equally, so it prefers 0.0. A deployment that weighs a false answer above twice
a refused answerable question should read the table rather than the
recommendation: 0.60 / 0.60 with `LLMGrader` at `max_attempts` 2 is the
low-false-answer end of it, at 3.4% false answers for 64.0% of answerable
questions refused.

Cost at the operating point, over the 680-query slice: 7,173 model calls with
`LLMGrader` and `LLMRewriter` at `max_attempts` 2, a mean of 10.5 per question,
against 959 with `HeuristicGrader` and 680 for the no-grader control.

## Reading the numbers

### The controls are the point of the table

Always answering scores 0.500 and always abstaining scores 0.500, which is what
makes a 47.4% null rate legible. Always abstaining also has an abstention
precision of 0.473, exactly the null rate, which is the reminder that precision
alone cannot separate a detector from a refusal to work.

The third control matters more. "No grader" scores 0.780 with 680 model calls
over the slice, so everything the corrective loop does here has to be read
against that. `HeuristicGrader` at its best setting reaches 0.783 for 1,165
calls and `LLMGrader` at its best reaches 0.788 for 6,259 calls. The grader plus
the rewrite-and-retry loop buys 0.008 balanced accuracy for roughly nine times the
model calls, and a paired test cannot tell that gain from zero: the loop is right
on 13 questions the control gets wrong and wrong on 7 it gets right (exact
McNemar p = 0.26). The best row records no grader abstentions, so the gain comes
from rewriting and retrying, not from the grade.

### The generator is the null detector

At the recommended thresholds the grader never returns `insufficient` on this
slice, so every abstention comes from `generate`. Measured on the first attempt
alone, before any threshold is applied, the generator declines 272 of 322 null
questions and 102 of 358 answerable ones. That is the 0.780 baseline, and it is
what the thresholds are competing with rather than adding to.

The reason the threshold loses is structural. The grader scores each passage
alone and pools the scores with a maximum, and it decides before generation, so
its abstention cannot be revised. The generator sees the five passages together
with the question in one prompt and can still answer from evidence the grader
scored zero.

### The citation gate is the worse trade

Of the 374 abstentions the generator made on the first attempt, 318 came from
the model setting `not_stated` and 56 from the citation gate, which turns an
answer citing no usable source into an abstention. Split by label, the gate
accounts for 38 of the 102 answerable questions the generator refused and only
18 of the 272 null ones it caught. That is a rate of roughly one false answer
prevented for every two answerable questions lost, worse than the rewrite
threshold's 2.15 at the point this memo recommends removing.

`require_citations` is therefore the more promising lever, and it is one harvest
to measure. This memo does not measure it: turning the gate off returns answers
whose correctness is not scored here, so the counts above are evidence for
running that experiment rather than a result from it.

### `LLMGrader` separates the two classes and the threshold cannot use it

On the first attempt the pooled `LLMGrader` score is 0.0 for at least three
quarters of null questions and has a median of 0.5 on answerable ones. The
grader is reading the corpus correctly. The problem is that its zero means two
different things: no retrieved passage supports the question, and the question
has no answer anywhere. Abstaining on the first is what costs the 43 answerable
questions, because a passage that scores zero in isolation is often enough for
the generator in context.

`HeuristicGrader` has no such separation to waste. Its pooled score has a median
of 0.62 on null questions and 0.64 on answerable ones, so term coverage carries
almost no signal about whether an answer exists.

### The corrective retry moves the evidence for a quarter of queries

`LLMRewriter` returned a usable rewrite for all 680 queries, but the rewritten
query changed the top 5 chunks for only 166 of them with `HeuristicGrader` and
181 with `LLMGrader`. Retrieval is already scoped to one document of 11 chunks
on average, so there is little for a rewrite to find. Raising `max_attempts`
from 1 to 2 is still worth about 0.007 balanced accuracy for `LLMGrader` at 0.0,
0.787 against 0.780, and costs 80% more calls.

## Secondary check: CUAD

CUAD is a check rather than a headline. Its test split is 70% `is_impossible`,
and the always-abstain control below shows what that does: an abstention
precision of 0.668, which is the corpus null rate and not a measurement of
anything the pipeline did. Balanced accuracy is what keeps the two corpora on
one scale.

Measured on the first 8 contracts in file order: 328 queries, 428 chunks, 219 of
the queries unanswerable for a 66.8% null rate. Retrieval reaches a lower
ceiling here than on ContractNLI, 65.1% against 92.2%, because CUAD's 41
questions repeat verbatim across contracts and carry no signal about which
clause of which contract they belong to.

`LLMGrader`, answer threshold held at 0.6:

| Thresholds | Attempts | Abstain recall | Abstain precision | False-answer rate | Answerable lost | Grounded answers | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 / 0.00 | 2 | 0.813 | 0.848 | 0.187 | 0.294 | 0.550 | 0.760 |
| 0.60 / 0.10 | 2 | 0.895 | 0.800 | 0.105 | 0.450 | 0.468 | 0.723 |
| 0.60 / 0.30 | 2 | 0.895 | 0.800 | 0.105 | 0.450 | 0.468 | 0.723 |
| 0.60 / 0.60 | 2 | 0.959 | 0.764 | 0.041 | 0.596 | 0.358 | 0.681 |
| no grader | 1 | 0.794 | 0.841 | 0.205 | 0.303 | 0.523 | 0.746 |
| always answer | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.651 | 0.500 |
| always abstain | 1 | 1.000 | 0.668 | 0.000 | 1.000 | 0.000 | 0.500 |

The rewrite threshold behaves the same way it does on ContractNLI and the gap is
wider: 0.0 scores 0.760 against 0.723 at the old default of 0.3, and the whole
grid is ordered the same. The generator-only control again sits within 0.014 of
the best arm, for a tenth of the calls, 328 against 3,576.

CUAD disagrees with ContractNLI on the answer threshold, and only for
`HeuristicGrader`. At the shipped 0.6 that grader scores 0.682 here, below the
no-grader control, and its best point on the grid is 0.753 at 0.20 / 0.20. CUAD
questions are long templated ones, so term coverage against a passage is lower
across the board and a high answer threshold rejects passages that do support
the question. `LLMGrader` runs the other way and prefers 0.6 or above. With the
two graders disagreeing on one corpus and the axis flat on the other, 0.6 stays
for want of a reason to move it, and a model-free arm on CUAD-shaped questions
should set its own.

## Out-of-domain control: MultiHop-RAG

150 of MultiHop-RAG's 301 null queries, each scoped onto one of the 40
ContractNLI documents in round-robin order so that retrieval returns real
contract clauses. The questions ask about news events from late 2023 and the
passages are non-disclosure agreements, so these are far nulls: separable on
topic alone. Only the null-query slice of MultiHop-RAG is used, and no article
text is read; `CORPUS.md` records why.

The pipeline declines all 150 at every one of the 172 grid points, for both
graders and both attempt budgets. There is nothing to tune here, which is the
result the control exists to produce: a detector that handles the far case and
not the near one has not been shown to work, and this says the far case is
handled.

What the control does add is a check on the recommendation. At the recommended
rewrite threshold of 0.0 the grader abstains on none of the 150 and the
generator abstains on all 150. Lowering the threshold therefore costs nothing on
the out-of-domain side: the stage that stops answering these questions is the
one that was always going to.

The answerable-side columns and balanced accuracy are undefined for an all-null
set, so this arm is reported through `evaluation.sweep.null_only_sweep` rather
than as sweep rows that would have to invent both.

## Measurement conditions

Every quality figure above is counted rather
than timed, so none of it is affected by the machine. Two timing facts are worth
recording anyway.

Embeddings ran on this machine's RTX 4090, so the 0.8 s index time for 451
chunks is a GPU figure. The bootcamp container's default is CPU, and that is the
number a participant sees, so do not quote this one in teaching material without
re-measuring it there.

Proxy throughput collapses above eight concurrent requests, and the collapse is
large enough to change how a sweep is run. Measured on this key with a
grading-sized prompt of about 300 tokens and `max_tokens` 512: eight concurrent
requests returned in 1.5 s mean and sustained 5.3 calls per second, while
twenty-four concurrent requests returned in 32 s mean and sustained 0.75. The
first ContractNLI harvest was run at twenty-four and took 65 minutes for 9,232
network calls. Anyone re-running this should stay at or below eight. Both
figures are shared-proxy measurements taken on one evening and are not a
property of the deployment.

## Cost

16,209 distinct completions were paid for across the three corpora, held in
`outputs/responses.sqlite`, which is gitignored because it carries whole prompts
and therefore corpus text. The three sweeps report 9,232, 3,150 and 2,108
network calls respectively; the remainder is a smoke run, the cached overlap
between grader arms, and one CUAD harvest abandoned at 24 workers and resumed
from cache.

Re-scoring every grid point on all three corpora from that cache costs nothing
and takes about twenty seconds, which is the whole point of harvesting traces
instead of re-running the pipeline per grid point.

## Limits

- One slice of each corpus carries the recommendation. ContractNLI at 40 of 123
  documents gives 358 answerable and 322 null queries, so a rate near 0.85 has a
  standard error of about 0.02 and balanced accuracy about 0.015. Differences
  below 0.03 in the tables are not separable, which is why the answer threshold
  is reported as flat rather than as having a winner, and why the 0.008 between
  the no-grader control and the best `LLMGrader` arm is not claimed as a real
  gap.
- The 40-document slice is 47.4% null against the full split's 43.2%. Every
  comparison is between rates over their own class and balanced accuracy ignores
  the base rate, so the ordering is unaffected. The absolute abstention
  precision would move.
- The MultiHop-RAG control was cut from 301 queries to 150 when proxy throughput
  dropped. Since every one of the 150 is declined at every grid point, the
  remaining 151 would have to behave differently from all of them to change the
  reading, but they were not run.
- One grading prompt and one model. `GRADE_PROMPT` anchors 0.0, 0.5 and 1.0 by
  name, and `Qwen3_6-35B-A3B` returns those three values and nothing else. A
  prompt that asked for a continuous score might give the threshold something to
  separate, and that is a different experiment.
- One rewriter. `LLMRewriter` at its defaults produced a usable rewrite for
  every query, so the unavailable-rewrite path in `decide` is never exercised on
  these corpora.
- Answer correctness is not measured. "Grounded answers" says a gold chunk was
  among the passages the generator read, which is necessary for a correct answer
  and not sufficient. `evaluation.answers.judge_answer` would close that and
  costs another call per answered question.

## Follow-ups

1. Measure `require_citations`. The counts above put the citation gate at 38
   answerable questions lost against 18 null ones caught, a worse trade than the
   threshold this memo just moved. One harvest with the gate off, scored with
   `judge_answer` so the recovered answers are checked rather than assumed.
2. `SpineConfig.thresholds` is carried on the config and never read.
   `AdvancedRagSpine.answer` takes its verdict from `grade.verdict`, which the
   grader produced from its own thresholds, so setting the config field does
   nothing and does it silently. Either wire it through or remove it; a test
   pins the current behaviour either way.
3. Re-measure on the full 123-document ContractNLI split, at eight workers. The
   harvest is 2,091 queries, about 31,000 calls, roughly two hours.
4. Retrieval is the larger lever on the answerable side. The ceiling is 92.2% on
   ContractNLI and 65.1% on CUAD, and the recommended operating point reaches
   71.8% and 55.0%. A whole extra retrieval moves ContractNLI from 69.8% to
   71.8%, so the corrective loop is not that lever; the harness's hybrid and
   reranking arms move the ceiling itself.
