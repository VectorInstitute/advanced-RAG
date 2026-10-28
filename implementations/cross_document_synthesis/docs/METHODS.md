# Retrieval and answering methods

The experiments vary how evidence is collected while keeping the final answering
contract fixed. Methods receive passage text and the question. Gold answers,
aliases, supporting IDs, and hop graphs belong to the scorer.

## Shared retrieval and reader

Every dense method searches the complete selected corpus with
`BAAI/bge-small-en-v1.5`, pinned to revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Inputs are title, newline, and passage
text, with no additional chunking. The CPU encoder produces 384-dimensional,
L2-normalized float32 vectors, uses CLS pooling, and truncates to 512 tokens.
Queries use BGE's query instruction; passages have no instruction prefix.
The vector cache records the encoder configuration and verifies the file digest.

`DenseIndex` ranks by dot product with deterministic passage-ID tie breaks.
Retrieval retains up to 20 passages; generation receives at most five complete
original passages without character truncation. `ANSWER_TEMPLATE` requests JSON
with an answer, numbered sources, and an explicit not-stated flag. Invalid or
missing citations trigger abstention. Citation provenance does not test entailment.

The recorded reference runs use `Qwen3_6-35B-A3B` at the Vector proxy,
temperature zero, thinking disabled, and a 256-token completion cap. Each
completion has bounded transport retries and shared usage accounting. These
settings describe the saved core methods and corrective comparison. New live
runs select generation, extraction, and judging roles through the shared
`choose_models()` configuration. Decomposition planning and IRCoT retrieval
reasoning use extraction; final answers use generation; corrective evidence
grading uses judging. All chat stays on the Vector proxy. See the
[provider settings](../GUIDE.md#models-credentials-and-spend) before comparing
new runs with the recorded reference.
See [the measurement limits](COMPARISONS.md#read-the-comparison) when comparing
provider responses and cached execution costs.

## Dense baseline

`DenseAnswerer` retrieves the original question once and delegates cited answering
to `AdvancedRagSpine` with `SingleShotPolicy`. Its heuristic grade is recorded
without deciding whether to retry. The original full baseline also demonstrates
the shared evaluation harness through `run_dense_baseline.py`.

The [reference records](../outputs/reference/dense_baseline/) retain rankings,
answer scores, and stage timings. The 1,000 cited answers ship in RI #2's
reference release as the `dense-predictions` artifact. The compact [evaluation reports](../outputs/evaluation/)
are the source for matched method comparisons. A baseline failure can occur in
retrieval, selection into the five-passage reader budget, or generation; a correct
answer alone does not establish that its evidence was complete.

## Query decomposition

`DecompositionAnswerer` implements the shared answering interface. One planning
completion requests two to four standalone search questions without guessed
intermediate answers. Validation requires distinct nonempty strings of at most
40 words and rejects numbered placeholders. An empty list declines decomposition;
malformed plans fall back to the original question. Provider failures remain
execution failures.

Each accepted subquestion and the original question retrieves 20 passages.
Rank-first round-robin pooling interleaves the rankings, with subquestions before
the original query, deduplicates source IDs, and retains 20. The first five feed
the common final reader. Pooling before the spine allows several subquestions to
contribute evidence within the fixed reader budget.

This reference method plans independent searches. It does not answer intermediate
questions or substitute discovered entities. Its measured effect combines decomposition
and pooling; the comparison does not isolate those choices.

[Notebook 01](../01_query_decomposition.ipynb) demonstrates the actual adapter on
an invented chain, inspects a scorer-only gold hop graph, and contrasts a balanced
30-question teaching example with the full matched results. Gold decomposition
steps never enter held-out planning.

## Agent lab

`MusiqueTools` provides framework-neutral plan, search, read, and reflect tools.
The ADK adapter uses local embeddings, the Vector proxy, and fresh state per
question. Search and read budgets bound the amount of corpus text exposed.
The parser validates final citations against passages actually read.

The three-question record, which `synthesis_lab.reference.agent_lab` loads from
RI #2's reference release, contains three abstentions, including one model-call-limit stop. It demonstrates bounded execution
and diagnosis, not improved quality. [Notebook 02](../02_agentic_retrieval.ipynb)
lets you inspect how literal searches, missing bridges, and tool budgets affect
which evidence reaches the agent.

## IRCoT

`IRCoTAnswerer` uses the shared ADK loop with plain Python retrieval and evidence
update tools. The original question is retrieved first. Thereafter an accepted
evidence update must precede each search. Updates retain brief facts with source
IDs, selected evidence IDs, and the next query or a terminal status. A discovered
entity can therefore guide the next retrieval.

Defaults are four retrieval rounds, 20 ranked hits per round, five exposed passages
per round, ten ADK model calls, and a 180-second session timeout. The separate final
reader still uses at most five original passages through the common answer template.
Intermediate facts guide retrieval and cannot replace source text as final evidence.

Stop reasons distinguish sufficient evidence, explicit insufficiency, repeated
queries, retrieval limits, call limits, and incomplete updates. Insufficient or
incomplete sessions abstain. Budget stops may answer from available evidence.
Provider failures retain zero scoring credit. Fresh tool and session state prevent
one question from supplying evidence to another.

[Notebook 03](../03_ircot.ipynb) demonstrates evidence-dependent retrieval with a
scripted policy and uses recorded live trajectories to explain the measured arm.
A valid source ID does not establish that an intermediate fact is supported;
inspect the passage before attributing a gain to reasoning.

## Optional evidence correction

The corrective arm reuses RI #1's extracted `AdvancedRagSpine`, `LLMGrader`,
`LLMRewriter`, and `CorrectivePolicy`. It grades the first five passages, pooling
by the highest score. Fixed thresholds are sufficient 0.6 and insufficient 0.0.
A borderline result permits one rewritten retrieval, with at most two attempts.
The final reader uses the settled attempt's first five passages. Retrieval
metrics also use that settled ranking; earlier attempts remain in local traces.

The dense control isolates the whole correction policy. The graded single-pass
control adds grading without a retry, isolating the extra effect of rewriting
and retrieval. Thresholds are not tuned on benchmark answers. This is evidence
correction, not repeated answer drafting, and a single passage's grade can miss
whether the full multi-hop chain is present.

Use the paired scores on the shared 300 questions and the added cost to assess the policy.
The 30-question smoke run demonstrates integration. See
[the comparisons](COMPARISONS.md) for the measured quality and cost.

## Starter agent

[Notebook 99](../99_starter_agent.ipynb) exposes an editable offline rule policy
and a separately gated ADK instruction, question ID, and tool/model-call budgets.
Both paths use the same bounded IRCoT answering adapter and fresh state.
The recorded ADK check, which `synthesis_lab.reference.starter_development`
loads from RI #2's reference release, answered one
pinned question with complete generation support and valid source citations,
using five model calls, three tool calls, and 11,519 tokens. This validates the
live wiring and metering; one question cannot establish method performance.

## Graph retrieval

The [graph notebook](../04_graphrag_head_to_head.ipynb) compares dense retrieval
with a traversable invented graph under the same final reader budget. Removing
a bridge or a retrieval seed demonstrates where graph connectivity can fail.
It is a mechanics exercise, not a MuSiQue graph benchmark.

The graph runner queries both checksum-pinned native RI4 indexes through the
shared artifact cache. It verifies complete original-passage mappings and uses
the same five-passage answerer for atlas-rag HippoRAG 2 and LightRAG. See
[the input contract](EXPERIMENTS.md#prebuilt-graph-comparison) for release loading,
encoder settings, retrieval depth, and query-cost accounting.

RI #4's recorded HippoRAG 2 build uses the atlas-rag variant. Label that variant
explicitly rather than attributing its results to the paper configuration.
Construction cost and query cost measure different work and are reported separately.
