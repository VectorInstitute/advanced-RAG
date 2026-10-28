# Special topic: how much an LLM-built graph moves between builds

This special topic rebuilds the same knowledge graph many times and measures how far the graphs
and their retrieval move. It is for teams that maintain an LLM-built graph and need to know what
a rebuild, a new builder model, a schema, or an incremental update does to it. It covers three
kinds of graph: LLM-built and schema-free (atlas-rag's HippoRAG 2 extraction and LightRAG),
schema-guided extraction with provenance (RI #4's ServerFault extractor), and a graph built with
no model.

## Results

Every figure comes from a handful of builds per setting, with no repeats beyond those and no seed
controls. Each
result is scoped to the builder, model, and corpus named with it. The MuSiQue figures use 100
questions (50 two-hop, 25 three-hop, 25 four-hop) over the 1,825 passages those questions list,
the reader is always `Qwen3_6-35B-A3B` at temperature 0, and an F1 interval is a paired bootstrap
against the anchor build (`atlas-Qwen3_6-35B-A3B-t0.7-r1`).

### Surprise: a rebuild with identical settings moves retrieval as much as deleting a quarter of the graph

Three builds of the MuSiQue sub-corpus with atlas-rag's prompt on `Qwen3_6-35B-A3B` at atlas-rag's
default temperature of 0.7 (RI #4's reference build and two rebuilds made on one later day) have
the same size to within 1%: 17,000 to 17,100 entities and 18,200 to 18,400 triples. They share
0.63 of their entities, 0.45 of their entity pairs, and 0.23 of their labelled triples. Their
all-supporting recall at 5 is 0.39, 0.46, and 0.34, a 12-point range from rebuilding alone, and
their F1 is 0.406, 0.420, and 0.388. The top five passages are the same set for 29% of questions.

Deleting triples at random from the anchor gives the scale. Removing 25% leaves recall at 5 at
0.37 and F1 at 0.429; removing 50% gives 0.29 and 0.349; removing 75% gives 0.25 and 0.327
(interval against the anchor -0.180 to -0.004). A rebuild that replaces 77% of the triples
therefore costs about what deleting 25% costs: most replaced triples state the same facts in
other words, so the passage graph that PageRank walks barely changes, but which passage lands in
fifth place does.

The sixteen-day-old reference agrees with the new builds (0.63 entities, 0.23 triples) exactly
as well as the two new builds agree with each other (0.63, 0.23). Nothing drifted over time
beyond the sampling at temperature 0.7.

- What this means for maintenance: a team comparing two graph builds on 100 questions cannot read
  an all-supporting-recall difference under about 12 points, or an F1 difference under about 0.07,
  as a property of the change. Any comparison of graph types or build controls on this corpus
  has to clear that noise floor.

### Temperature 0 steadies the MuSiQue graph but narrows retrieval only partly

Three more builds of the same MuSiQue sub-corpus with the same prompt and model, sent at
temperature 0 on the same day as those two rebuilds, agree far more than the temperature-0.7 builds, and their retrieval
still moves. Figures are means over each setting's pairs of builds; the temperature-0.7 column
is the three builds of the first section, except the response comparison, which uses the two
same-day builds because the reference came from atlas-rag's own pipeline.

| Measure | Temperature 0 (3 builds) | Temperature 0.7 (3 builds) |
|---|---|---|
| Entities shared | 0.86 | 0.63 |
| Entity pairs shared | 0.78 | 0.45 |
| Labelled triples shared | 0.64 | 0.23 |
| Passages with a byte-identical response | 0.45 | 0.03 |
| Where differing responses split (share of the response) | 0.37 | 0.11 |
| Questions with the same top five passages | 0.48 | 0.29 |
| All-supporting recall at 5 | 0.39, 0.41, 0.34 | 0.39, 0.46, 0.34 |
| All-supporting recall at 10 | 0.57, 0.53, 0.50 | 0.55 to 0.59 |
| F1 | 0.411, 0.403, 0.417 | 0.406, 0.420, 0.388 |

Labelled-triple agreement nearly triples and the F1 range halves (0.014 against 0.032), but
recall at 5 and recall at 10 each still span 7 points, and at 10 that is wider than the 4 points
at 0.7. No temperature-0 build differs from the anchor in F1 (differences -0.017 to -0.003, every
interval spanning zero). Each build is 1,825 calls on the on-prem model and the three cost
nothing.

The cause of the remaining churn is the server, not sampling: every temperature-0 request is
byte-identical between builds, so each of the 55% of passages whose responses differ was
changed by the serving stack, most likely by batch composition on a vLLM server shared with
other jobs. Request settings do not stop it. Six passages sent three times each, while the
builds and other jobs loaded the server, gave on average 2.7 distinct responses at temperature
0, 2.3 with a fixed seed, 2.7 with `top_k` 1, and 2.7 at temperature 0.7 with a fixed seed. No
passage returned the same response all three times under any setting.

- Scoped: with atlas-rag's prompt on `Qwen3_6-35B-A3B` behind the shared on-prem server,
  temperature 0 is the better default for a graph build, because it removes most of the label
  churn and halves the F1 range, but it does not make a rebuild a replay. A team comparing two
  builds on these 100 questions still cannot read a recall-at-5 difference under about 7
  points as a property of the change.
- What would have to be true for temperature 0 to remove the spread: the server would have to
  return the same tokens for the same request, which needs batch-invariant kernels or a server
  with no concurrent traffic. Cheapest test: send 50 passages twice, one at a time, to a vLLM
  instance with batch-invariant mode on (about 100 calls), and count identical responses.

### Temperature bounds build variance; the schema bounds labels

On 100 ServerFault posts with `Qwen3_8-27B`, three prompts were each built three times at
temperature 0.7 and twice at temperature 0. `schema` is RI #4's extractor (seven entity and seven
relation types, alias map, quotes); `free` is the same prompt with "any short type label you
choose" in place of the two type lists; `atlas` is atlas-rag's triple prompt.

| Prompt | Temperature | Entities | Entity pairs | Labelled edges | Type agreement |
|---|---|---|---|---|---|
| `schema` | 0.7 | 0.62 | 0.29 | 0.20 | 0.89 |
| `free` | 0.7 | 0.66 | 0.35 | 0.15 | 0.56 |
| `atlas` | 0.7 | 0.50 | 0.28 | 0.15 | n/a |
| `schema` | 0 | 0.94 | 0.75 | 0.68 | 0.98 |
| `free` | 0 | 0.94 | 0.84 | 0.67 | 0.96 |
| `atlas` | 0 | 0.91 | 0.86 | 0.80 | n/a |

Figures are graph-level Jaccard between repeats, after name normalisation. At temperature 0 the
schema-free prompt is as stable as the schema prompt, and atlas-rag's prompt, with no types
at all, keeps the most labelled edges (0.80 against 0.68). At temperature 0.7 the schema raises type
agreement from 0.56 to 0.89 and labelled-edge agreement from 0.15 to 0.20, and lowers entity
agreement from 0.66 to 0.62: it fixes the vocabulary of labels, not which facts get extracted.
99.8% of the schema prompt's labels already sit on the schema, so the post-hoc fold changes
almost nothing; applied to the free prompt's output it keeps only 12% of relations.

On MuSiQue, adding a closed list of twelve relation types to atlas-rag's prompt (two builds on
`Qwen3_6-35B-A3B` at 0.7) raises labelled-triple agreement from 0.23 to 0.31 and entity
agreement from 0.63 to 0.66, and leaves entity-pair agreement at 0.43 against 0.45. Its F1 is
0.336 and 0.397 against 0.388 to 0.420 without the schema, and the first schema build is below
the anchor (-0.084, interval -0.167 to -0.001).

- Negative, scoped: a closed relation schema on atlas-rag's prompt with `Qwen3_6-35B-A3B` on
  MuSiQue does not reduce retrieval variance and costs some F1. For it to win, the questions would
  have to turn on relation types (a "who directed" question where the graph also holds "who
  produced"), which PageRank over passages never reads. Cheapest test: retrieve by typed
  traversal (a Cypher-style hop on `created_by`) on the 50 two-hop questions, with both graphs,
  about 100 reader calls.
- Negative, scoped: the RI #4 schema on ServerFault with `Qwen3_8-27B` does not bound which facts
  are extracted. For a schema to bound content it would have to constrain instances, not types:
  a closed entity list per post (from an earlier build or a product catalogue) that the model
  must link to. Cheapest test: a fourth ServerFault arm that lists build 1's entity names in the
  prompt, three repeats, about 600 calls.

### Builder model: rebuild stability differs by model; retrieval does not separate

One atlas-rag build per model on the MuSiQue sub-corpus, with a second build for five models.
The hosted models ran at their provider's default temperature, because Claude 4.7 and later and
the GPT-6 models reject one; the others ran at 0.7.

| Builder | Rebuild labelled-triple agreement | Triples shared with the anchor | Recall at 5 | F1 | F1 against anchor | Build cost |
|---|---|---|---|---|---|---|
| `Qwen3_6-35B-A3B` (anchor) | 0.23 | 1.00 | 0.34 to 0.46 | 0.388 to 0.420 | | $0 |
| `Nemotron-3-Nano-Omni-30B-A3B` | | 0.09 | 0.38 | 0.367 | -0.053 (-0.129 to 0.024) | $0 |
| `gpt-oss-120b` | | 0.10 | 0.38 | 0.376 | -0.044 (-0.116 to 0.026) | $0 |
| `Qwen3_8-Flash-Next` | | 0.14 | 0.45 | 0.440 | 0.020 (-0.056 to 0.096) | $0 |
| `gemini-3.5-flash-lite` | 0.38 | 0.12 | 0.42 | 0.410 | -0.010 (-0.093 to 0.075) | $1.25 |
| `gemini-3.8-flash` | | 0.12 | 0.42 | 0.446 | 0.026 (-0.043 to 0.095) | $2.57 |
| `gemini-4-argon` | | 0.12 | 0.38 | 0.377 | -0.043 (-0.118 to 0.029) | $8.97 |
| `claude-haiku-5-5` | 0.37 | 0.12 | 0.46 | 0.424 | 0.004 (-0.079 to 0.088) | $0.48 |
| `claude-sonnet-5-5` | | 0.10 | 0.40 | 0.454 | 0.034 (-0.040 to 0.111) | $10.89 |
| `claude-opus-5-5` | | 0.10 | 0.41 | 0.408 | -0.012 (-0.099 to 0.076) | $22.11 |
| `gpt-6-luna` | 0.33 | 0.07 | 0.44 | 0.429 | 0.009 (-0.084 to 0.102) | $0.33 |
| `gpt-6.1-sol` | 0.44 | 0.05 | 0.43 | 0.424 | 0.004 (-0.068 to 0.077) | $7.46 |

Every hosted model rebuilds more consistently than `Qwen3_6-35B-A3B` (0.33 to 0.44 of labelled
triples against 0.23) even at a higher temperature, and two different builders share only 0.05
to 0.14 of their triples. Downstream, every builder's F1 lies within the anchor's own rebuild
range or its interval, and Opus 5.5 at $22 a build answers as well as the free on-prem model.

- Negative, scoped: on 100 MuSiQue questions with atlas-rag's prompt and HippoRAG 2 retrieval,
  builder scale from a 30B mixture-of-experts model to Opus 5.5 does not move F1 beyond rebuild
  noise. For scale to win, the graph would have to carry facts a small model misses; MuSiQue's
  bridges are mostly named entities in one sentence, which every model extracts. Cheapest test:
  the four-hop questions over the 139,416-passage pool, where distractors share names, with the
  Haiku and A3B builds of those passages only (about 2,000 calls each).

### A graph built with no model loses; ensembling builds does not help

The title-mention graph (LLM-free: each passage's article title is an entity, linked to every other
title its text names; 0 model calls) scores recall at 5 of 0.20 and F1 0.284, 0.136 below the
anchor (-0.224 to -0.050) and level with dense retrieval over the same passages (0.20, F1 0.271).
The union of the three A3B builds' triples scores 0.37 and 0.406, inside the members' range, so
the variance is not recall that a union can harvest.

- Negative, scoped: title mentions are too sparse a graph (877 of 1,825 passages have no link).
  For an LLM-free graph to win, it needs the links the LLM finds: noun-phrase co-occurrence in
  the FastGraphRAG style. Cheapest test: a spaCy noun-phrase graph through the same indexer,
  0 model calls and about 200 reader calls.

### Invented names do not raise build variance

Replacing every article title (1,616 of them) with an invented name in the passages, two builds
on `Qwen3_6-35B-A3B` at 0.7 share 0.63 of entities and 0.23 of labelled triples, the same as with
the real names. On this corpus, rebuild variance does not come from the model recognising
entities. Other names (people not titled in the sub-corpus, places) stayed real, so the control
is partial. Cheapest stronger test: rename every capitalised span the anchor build extracted,
two builds, about 3,650 calls.

### LightRAG: incremental against all at once

Three LightRAG builds of the 60-question sub-corpus (1,153 passages) on `Qwen3_8-27B` at
temperature 0, in LightRAG's `mix` mode with RI #4's settings: two inserting every passage at
once and one inserting them in five batches of about 230.

| Pair | Entities | Edges | Degree distance | Top-5 Jaccard |
|---|---|---|---|---|
| once r1, once r2 | 0.935 | 0.839 | 0.003 | 0.69 |
| incremental, once r1 | 0.935 | 0.838 | 0.003 | 0.76 |
| incremental, once r2 | 0.933 | 0.831 | 0.004 | 0.71 |

The incremental build differs from each all-at-once build exactly as much as the two
all-at-once builds differ from each other, so inserting in batches adds no drift beyond rebuild
noise on this corpus and model. Each build cost about 2,570 calls; LightRAG marked 1 to 10
passages failed per build after a retry pass, which is part of the rebuild noise. Retrieval does
not separate the three (all-supporting recall at 5 of 0.17 to 0.18, F1 0.286 to 0.289) or from
dense retrieval over the same passages (0.18, F1 0.265), which repeats RI #4's finding that
LightRAG ties dense on MuSiQue.

LightRAG's graph is far more stable between rebuilds than atlas-rag's (0.84 of edges against
0.45 of entity pairs), because it runs at temperature 0 and merges each entity's descriptions
into one node. The comparison crosses model and temperature, so it says where to look, not
which builder is steadier.

- Negative, scoped: five equal batches on 1,153 MuSiQue passages with `Qwen3_8-27B` show no
  order effect. For order to matter, later batches would have to revise entities that earlier
  batches created, which LightRAG does through description summaries once an entity collects
  enough mentions; few MuSiQue entities recur that often in 1,153 passages. Cheapest test:
  insert the ServerFault sample, where product names recur across posts, in five batches and
  once, about 1,000 calls each.

## How the builds were kept independent

A repeat that reads its responses from a cache is a replay, not a second build. Every
extraction call in this topic goes to the model with no response cache in front of it: the
runner writes each raw response to the build's `records.jsonl` as it arrives. Two machine
restarts interrupted the block; every repeat build that was cut off was moved aside and run
again from an empty directory, so no repeat holds responses from an earlier attempt. Every
build's record shows zero cache hits, and the committed results test that. LightRAG builds go through the shared proxy client,
which needs a cache, so each LightRAG build gets its own empty cache file; its hits are repeats
of one request inside that build.

Temperature is the second half of independence. atlas-rag's HippoRAG 2 pipeline sends
temperature 0.7 by default, and RI #4's reference HippoRAG 2 index was built that way, so a
rebuild at the library default is a fresh sample. The RI #4 ServerFault extractor and RI #4's
LightRAG wrapper send temperature 0. Anthropic models from 4.7 on and the GPT-6 models reject a
temperature, so the paid builds ran at their provider default, which each build's meta
records as `temperature_sent: null`.

Evaluation answers fixed questions over a finished graph, so it is not repeated and its calls
share one cache. The reader is always `Qwen3_6-35B-A3B` at temperature 0, so only the graph
changes between rows.

## Corpora and measures

The MuSiQue sub-corpus is the first 50 two-hop, 25 three-hop, and 25 four-hop questions of RI
#4's stratified test set and every candidate paragraph those questions list, supporting and
distracting: 1,825 passages. LightRAG builds use the first 30, 15, and 15 of those questions
and their 1,153 passages, because a LightRAG build costs about 2.2 calls a passage, and ran on
`Qwen3_8-27B`, whose request limit is separate from the atlas-rag builds' model. The
ServerFault sample is 100 posts drawn with seed 33 from RI #4's 10,000-question slice, cut into
203 windows the way RI #4's extractor cuts them.

Two builds are compared after normalising names: lowercase, accents and punctuation removed,
whitespace collapsed, and a leading article dropped. The measures are the Jaccard index of the
merged node sets, of the labelled edge sets (head, relation, tail), and of the unlabelled edge
sets (the pair of entities, direction and label dropped), at graph level and per document;
entity-type agreement on shared entities; the Kolmogorov-Smirnov distance between degree
distributions; and overlap of the ten highest-degree entities. Downstream, each graph is
scored with RI #4's HippoRAG 2 adapter on all-supporting recall at 5 and 10, exact match, and
token F1, and two graphs' rankings are compared by the Jaccard of each question's top five.

The closed MuSiQue schema adds twelve relation types (`located_in`, `part_of`, `born_in`,
`died_in`, `citizen_of`, `family_of`, `member_of`, `employed_by`, `educated_at`, `created_by`,
`owned_by`, `happened_on`) and seven entity kinds to atlas-rag's own prompt, then folds the
model's labels onto the twelve and drops any relation that maps to none. Keeping atlas-rag's
prompt otherwise unchanged isolates the schema. On ServerFault the same isolation compares RI
#4's schema prompt with a copy whose two type lists say "any short type label you choose" and
"any short verb phrase you choose", with every other rule, the evidence quotes, and the
twelve-item cap kept.

## Designs considered

Before running, twelve directions were ranked by what they would teach. The block ran seven: the
rebuild census (repeats per builder), the schema test on its home domain (ServerFault), the same
test on MuSiQue, a triple-removal ladder that gives graph churn an exchange rate in retrieval, a
graph built with no model, a union of builds, and the builder scale ladder, plus an invented-names
control. LightRAG's incremental build is the eighth. Not run: a closed entity list as an
instance-level schema, and ensembling by majority vote.

The temperature-0 MuSiQue block weighed five designs: three plain temperature-0 builds; a build
with a fixed seed; a byte-level comparison of the builds' raw responses, which separates server
nondeterminism from sampling at no extra cost; a build sent one request at a time, so this
runner adds no batching of its own; and a majority-vote graph over the temperature-0 builds. It
ran the three builds and the response comparison, and replaced the seeded build with a
72-request probe once the probe showed a seed does not make the server repeat itself. The
one-at-a-time build was dropped because the shared server answered about one request every 15
seconds that day, about 7.5 hours a build, and other jobs would still have shared its batches.

## Files

- `stability.py`: name normalisation, the overlap measures, the derived graphs (title mentions,
  union, triple removal, invented names), and the prompts. No model or network.
- `run_stability.py`: builds, indexes, and evaluates; its docstring lists the subcommands.
- `stability_report.py`: gathers every build into `outputs/graph_build_stability.json`.
- `outputs/graph_build_stability.json`: every figure on this page, with `produced_by`.
- `tests/`: the measures and prompts, and checks that the committed results agree with
  themselves and carry no corpus text.

No notebook ships with this topic: the results are tables, and a participant run of even one
MuSiQue rebuild is 1,825 calls.

## Rebuild the results

Run from this folder with the bootcamp environment. Each build writes its raw responses to
`~/.cache/aieng-advanced-rag/special_topics/graph_build_stability/` as they arrive, so a stopped
build resumes; a build that should be independent of an interrupted one must start in a fresh
directory.

```bash
python run_stability.py mq-build --builder atlas --model Qwen3_6-35B-A3B --temperature 0.7 --rep 1
```

```bash
python run_stability.py mq-index --build atlas-Qwen3_6-35B-A3B-t0.7-r1
```

```bash
python run_stability.py mq-eval --build atlas-Qwen3_6-35B-A3B-t0.7-r1
```

```bash
python run_stability.py report
```

The greedy probe sends six extraction prompts three times under each setting:

```bash
python run_stability.py mq-greedy
```

A MuSiQue build is 1,825 calls (about 30 minutes at 65 requests a minute on the on-prem model),
an evaluation about 200, and a LightRAG build about 2,600. A paid build needs
`ADVANCED_RAG_SPEND_LIMIT_USD`; the block's paid builds cost about $67 in all: $63.56 for the complete builds, $0.76 for
20-passage pilots, and about $2.50 for a repeat that a machine restart interrupted and that was
not rerun.

## Limits

- 100 questions give F1 intervals about 0.15 wide, so only large differences separate.
- One reader answers every graph; another reader changes absolute scores.
- Two or three builds per setting estimate variance roughly.
- The renamed-corpus control renames titles only.

## Data and licence

MuSiQue is CC BY 4.0 and ServerFault posts are CC BY-SA (see
`../../graphrag_technical_support/CORPUS.md`). `outputs/` holds identifiers, counts, and metrics
only; `outputs/ATTRIBUTION.md` has the attribution.
