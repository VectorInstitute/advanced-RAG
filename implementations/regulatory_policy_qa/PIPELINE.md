# Pipeline stages

Five stages turn a question into a grounded answer or an abstention. Each stage
below names the module it lives in and the numbers you can turn, and the tables
at the end list every default. The code is in
`aieng-advanced-rag/aieng/advanced_rag/pipeline/`.

## Stage order

1. Rewrite the question, before the first retrieval.
2. Retrieve, through a caller-supplied callable.
3. Grade the retrieved evidence.
4. Act on the grade: answer, rewrite and retrieve again, or abstain.
5. Generate an answer with citations, or abstain.

Retrieval enters the pipeline as a callable of `(question, scope, k)`, so the
pipeline is independent of which retriever is behind it. Every retriever in the
package satisfies that callable itself, through
`retrieval.interface.RetrieverBase`, so the dense, BM25, hybrid, and reranking
arms of the harness are passed in as they are and the pipeline imports none of
them.

Passages move between stages as `graphrag.evidence.Evidence`, the shared
boundary type every retrieval method in this repository returns. Citations are
`provenance.Provenance` records.

## Query rewriting

`pipeline/rewrite.py`

`TemplateRewriter` deletes a corpus's known boilerplate. It is deterministic and
needs no model. CUAD's 41 questions are templated, so the shared wrapper is most
of the term mass a lexical retriever sees and most of the sentence a dense
encoder embeds; `CUAD_BOILERPLATE` is the pattern list that removes it.

`LLMRewriter` asks a model for alternative phrasings. Passing the grader's
reason as `feedback` puts it in the prompt, so a retry aims at the gap the
grader named.

Both return a `RewritePlan`. An empty plan means "keep the question as asked",
and the spine reads it that way.

The spine uses at most one question per attempt. Retrieving with several
variants and fusing the rankings is a different technique and is not
implemented here.

## Parent-child chunking

`pipeline/hierarchical.py`

`split_document` cuts a document into parents, the units a reader gets, and
children, the units a retriever indexes. Parents align to detected headings when
`use_headings` is on, and no child straddles two parents, which is what makes
the parent lookup single-valued.

`small_to_big` runs between retrieval and generation. It swaps each retrieved
child's text for its parent's and keeps the child's `chunk_id`, so retrieval
metrics still score the unit the retriever ranked. The offsets and section on
the returned provenance move to the parent, because that is the text being read;
`provenance.extra` records both identifiers.

`contextualize` prefixes each child with its document and section before
indexing. Offsets are left alone, so span-derived relevance judgments stay
valid even though the indexed text no longer starts at `start`. Passing
`describe` adds a model-written sentence per child, which is one call per chunk
in the corpus.

`to_chunks` converts children to the harness chunk type. The harness names the
document field `contract_id`; the mapping happens there.

## Evidence grading

`pipeline/grading.py`

A grader returns an `EvidenceGrade`: a per-passage score list, the pooled score,
the verdict that score falls under, the reason, and the thresholds it was read
against. The pool is the maximum: one passage that states the answer is enough to answer
from, and averaging it against the filler retrieved alongside it would score a
successful retrieval as a failure.

`HeuristicGrader` scores term coverage between the question and the passage,
after stop-word removal and crude suffix stripping. It needs no model, which
makes it the grader for tests and offline threshold sweeps. It cannot tell a
passage that states the answer from one that repeats the question's nouns, so
numbers reported from it understate the pipeline.

`LLMGrader` asks a model to score each passage from 0.0 to 1.0. One call per
passage: grading several at once is cheaper, and the question-generator verifier
in this project already showed what that does to a judge, dropping its
acceptance rate to 5% when one bad item poisoned the verdicts on the rest.
`max_chunks` bounds the cost instead. Passages past that bound are recorded with
a score of 0.0 and no call, so an ungraded passage can never raise the pool.
Measured on ContractNLI it returns only the three values its prompt anchors,
0.0, 0.5 and 1.0, so a threshold matters there only by which side of 0.5 and 1.0
it falls on.

The default loop policy reads the pooled grade against `SpineConfig.thresholds`
to decide the action. The grader records an advisory verdict using its own
`GradingThresholds`; give it the same thresholds when measuring that policy.
An explicit policy owns its decision rules, and the spine rejects combining it
with non-default config thresholds that would otherwise go unread.

## The corrective decision

`grading.decide` maps a verdict and the retries remaining onto one action.

RI #1 does not configure this loop in its recommended participant path. On the measured
ContractNLI slice, the generator-only control scored 0.780 balanced accuracy with 680 model calls;
the best `LLMGrader` plus rewrite-and-retry arm scored 0.788 with 6,259 calls. A paired test
cannot tell the 0.008 gain from zero (exact McNemar p = 0.26 over 680 paired questions, 95% paired bootstrap interval -0.004 to 0.021), and it costs about 9.2 times the calls. The loop stays implemented for an
optional A/B measurement and for other RIs to test on their own tasks, not as a claimed quality
default. See [`THRESHOLDS.md`](THRESHOLDS.md).

The recommended single-pass configuration is `SpineConfig(max_attempts=1)`, a model-free
`HeuristicGrader`, `SingleShotPolicy`, and no `rewriter`. `SingleShotPolicy` makes the recorded grade
advisory, so it cannot abstain before generation; the generator's own `not_stated` path decides
whether to answer. The grader preserves the spine's trace shape without adding a model call.

| Verdict | Attempts left | Action |
| --- | --- | --- |
| sufficient | any | answer |
| ambiguous | one or more | rewrite |
| ambiguous | none | answer |
| insufficient | one or more | rewrite |
| insufficient | none | abstain |

Ambiguous evidence on the last attempt is answered rather than abstained on: the
generator has its own abstention path and can read the passages the grader only
scored. Insufficient evidence on the last attempt abstains with no generation
call.

With no rewriter configured—the recommended RI #1 configuration—a retry has nothing new to retrieve
with, so the loop settles immediately on the rule its final attempt would have applied. Enabling the
optional experiment requires configuring a rewriter explicitly.

## Grounded generation

`pipeline/generate.py`

`generate` renders the evidence as a numbered source list, asks for one JSON
object, and returns a `GroundedAnswer`. The model cites source numbers, never
chunk identifiers: asked to reproduce a long opaque string, a model invents
plausible ones, and a fabricated citation that parses is worse than none. Cited
numbers outside the range are dropped and recorded in `dropped_citations`.

Four paths abstain, and each names itself in `reason`:

1. Nothing was retrieved, so no generation call is made.
2. The model set `not_stated` in its response.
3. The model ignored the response format and wrote a refusal in prose.
4. The answer cited no usable source, with `require_citations` on.

The third path consults a phrase list, and only when the model's own flag is
absent. An answer can legitimately contain "not specified" in a subordinate
clause, so where the flag is present it is authoritative.

On any abstention, `text` is `ABSTENTION_TEXT` and the rejected draft stays in
`raw`. A draft the pipeline rejected must not reach a reader or a judge as
though it had been accepted.

`ChatCompleter` is the only place in the package that knows about a client. It
disables reasoning through `reasoning.disable_thinking` and turns `content=None`
into an empty string, counted in `empty_responses`. Both matter on the Vector
proxy: the default bulk model bills reasoning as completion tokens, at 1,312 of
them against 16 for the same answer, and it returns `content=None` with a normal
finish reason when `max_tokens` runs out mid-reasoning.

## Decision points

| Knob | Where | Default | Effect |
| --- | --- | --- | --- |
| `top_k` | `SpineConfig` | 10 | Passages kept from each retrieval and handed to the grader. |
| `evidence_budget` | `SpineConfig` | 5 | Passages placed in the generation prompt. |
| `max_attempts` | `SpineConfig` | 2 | Library compatibility default and maximum retrievals per question. It does not enable a retry without a `rewriter`. Set `max_attempts=1` with `SingleShotPolicy` for the recommended RI #1 configuration; use two only for the optional A/B measurement. |
| `require_citations` | `SpineConfig` | `True` | Turns an uncited answer into an abstention. Trades correctness on answerable questions for a lower false-answer rate on null ones. |
| `thresholds.sufficient` | `GradingThresholds` | 0.6 | At or above this pooled score the pipeline answers. Measured flat from 0.2 to 1.0 on ContractNLI. |
| `thresholds.insufficient` | `GradingThresholds` | 0.0 | Below this the pipeline rewrites, or abstains when out of attempts. At 0.0 no score is below it, so the grader never abstains on its own. See [`THRESHOLDS.md`](THRESHOLDS.md). |
| `prepare` | `AdvancedRagSpine` | `None` | Rewriter run before the first retrieval. `TemplateRewriter` belongs here. |
| `rewriter` | `AdvancedRagSpine` | `None` | Rewriter run on a corrective retry. Without one, there is no retry. |
| `grader` | `AdvancedRagSpine` | required | `HeuristicGrader` offline, `LLMGrader` for reported numbers. |
| `max_chunks` | `LLMGrader` | 5 | Passages graded per attempt, one model call each. |
| `max_chars` | `LLMGrader`, `generate` | 1200 | Per-passage truncation in the prompt. |
| `variants` | `LLMRewriter` | 2 | Rewrites requested per call. Does not change the call count. |
| `phrase_fallback` | `generate` | `True` | Reads a refusal out of prose when the response format is ignored. Turn off to measure the format contract alone. |
| `parent_chars` | `split_document` | 3000 | Size of a read unit. |
| `child_chars` | `split_document` | 600 | Size of an indexed unit. |
| `overlap_chars` | `split_document` | 100 | Overlap between neighbouring children. |
| `use_headings` | `split_document` | `True` | Aligns parents to detected headings instead of fixed windows. |
| `describe` | `contextualize` | `None` | Model-written context per child. One call per chunk in the corpus. |
| `max_tokens` | `ChatCompleter` | 512 | Completion cap. |
| `temperature` | `ChatCompleter` | 0.0 | Removes sampling variance and nothing more. See the note below. |

Zero temperature does not make a graded run reproducible. Variation remains, because a request's
result depends on the batch it lands in, and it is not portable: some providers accept only a
temperature of 1.0. The `ChatCompleter.temperature` docstring in
`aieng/advanced_rag/pipeline/generate.py` carries both citations. Compare evidence sets, which do
reproduce, rather than answers.

## Model calls per question

With `HeuristicGrader` and `TemplateRewriter`, only generation calls a model: at
most one call per question, and none when the pipeline abstains before
generating.

With `LLMGrader` and `LLMRewriter` at the defaults, one question costs
`max_chunks` calls per attempt, one call per rewrite, and one for generation:
6 calls when the first attempt is sufficient, 12 when it retries and answers,
and 11 when it retries and abstains. Measured over 680 ContractNLI queries at
the shipped thresholds and `max_attempts` 2, the mean is 10.5 calls per
question. The same slice with `HeuristicGrader` and no rewriter costs 1.0, since
a retry with nothing new to retrieve with settles immediately.

Those counts are why the `LLMGrader` plus `LLMRewriter` configuration is not recommended for RI #1.
The measured generator-only control uses one retrieval and at most one generation call; the
multi-call configuration above is kept to reproduce the experiment, not as the participant default.

## Feeding the evaluation harness

`spine.abstentions(results)` returns the mapping `score_null_detection` reads.
`spine.retrieval_runs(results)` returns the runs `score_retrieval` reads, taken
from the final attempt; score `results[i].attempts[0].evidence_ids` the same way
to price what the corrective loop bought. Answer correctness and grounding come
from `evaluation.answers.judge_answer`, which reads the same `Evidence` list.

Relevance judgments are derived from the chunks you index. Swapping flat chunks
for hierarchical children changes the identifiers, so rebuild the judgment set
from the children before comparing a number from the two setups.

## Known limits

- The thresholds are measured on ContractNLI and CUAD only, over a slice of
  each. [`THRESHOLDS.md`](THRESHOLDS.md) records the sweep, what the operating
  point costs on answerable questions, and the sample it rests on.
- Heading detection is a regular expression over line shapes, with a spacing
  filter to stop capitalised body text from becoming a heading on every line. It
  is not a document-structure parser.
- The abstention phrase list is English and hand-written.
- `HeuristicGrader` measures vocabulary overlap and nothing else.
