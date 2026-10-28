# Special topic: graph or not?

This special topic tests three claims a team makes before it pays for a knowledge graph. One:
that a graph index beats dense retrieval on multi-hop questions. Two: that a governed graph
answers relational questions (counts, complete lists, relations across documents, paths) that
retrieval cannot. Three: that cheap statistics of a corpus say in advance whether a graph will
pay. It is for participants who have worked through RI #4 and want to decide whether their own
corpus earns a graph. Every run is one run per arm with no repeats and no seeds, so treat a
difference under about 10 points as noise.

## Notebook

`01_graph_or_not.ipynb` reads the committed results in `outputs/` and ends with a readiness
check you run on your own documents, which makes no model call. The first code cell sets
`RUN_EXPENSIVE = False` and `MODELS = choose_models(show=True)`; with the flag on it reruns the
experiments on your models.

## Results of the reference run

The surprise comes first, as the project's research rules ask.

On ServerFault, for "which posts mention X" questions, the governed graph loses to `grep`.
Over 15 count questions the graph cites 37% of the posts that contain the name, and its exact
counts land within 20% of the true count for 3 of 15; `grep` lands on all 15 with no model
call. The facts are mostly in the graph under other names (`sysctl` appears as 34 entities,
from `sysctl.conf` to `sysctl -w net.ipv4...`): matching names by substring recovers 63%.
On MuSiQue, a bridge step with no graph and no model call, `chain-rerank`, retrieves every
supporting passage as often as HippoRAG 2 (0.387 against 0.390 at 10). With the participant
default reader, `gemini-3.5-flash-lite`, it still ties, and the reader itself moves more than
the arms do: rerun at temperature 0 on the same passages, it changes the score on 10 to 22 of
300 questions and moves `hipporag2`'s F1 by up to 0.020.

### X1: HippoRAG 2 against a stronger local retriever (MuSiQue)

RI #4's `hipporag2` beats a `bge-small-en-v1.5` dense retriever by 0.140 F1 on 300 MuSiQue
questions. The question here is how much of that is the graph and how much the small embedder.
The arms use only local models:

- `hybrid3`: reciprocal rank fusion of `bge-small-en-v1.5`, `Splade_PP_en_v1`, and
  `answerai-colbert-small-v1`, over all 11,656 passages.
- `hybrid3-rerank`: `bge-reranker-v2-m3` over the top 100 of `hybrid3`.
- `chain-rerank`: `hybrid3-rerank`, then for each of its top two passages a second retrieval
  with the question plus that passage's text, reranked against the same expanded query, and
  interleaved with the first ranking.
- `union-rerank`: the cross-encoder over `hipporag2`'s top 10 and `hybrid3`'s top 100.
- `fusion`: reciprocal rank fusion of `hybrid3-rerank` and `hipporag2`.

`Qwen3-Embedding-0.6B`, the strongest local embedder in our CUAD retrieval measurements (a
retrieval-failure rate of 0.1480, against 0.2400 for `bge-small-en-v1.5`), and `bge-base` and `bge-large` are in the package registry and outside
this study's model set. The reader is RI #4's: `Qwen3_6-35B-A3B` answers from an
arm's top five passages with RI #4's prompt, and `gemini-3.8-flash` judges, given the question,
the reference, and the five passages. The `dense` and `hipporag2` rows are RI #4's committed
rows; the reproduced dense ranking is identical to RI #4's for 300 of 300 questions.

| Arm | All-supporting @5 | All-supporting @10 | Exact match | F1 | Judged | F1 by hops (2 / 3 / 4) |
|---|---|---|---|---|---|---|
| `dense` | 0.163 | 0.237 | 0.193 | 0.250 | 0.243 | 0.323 / 0.165 / 0.188 |
| `hipporag2` | 0.320 | 0.390 | 0.297 | 0.390 | 0.377 | 0.491 / 0.374 / 0.204 |
| `hybrid3` | 0.217 | 0.303 | 0.240 | 0.310 | 0.317 | 0.392 / 0.239 / 0.216 |
| `hybrid3-rerank` | 0.233 | 0.283 | 0.260 | 0.338 | 0.333 | 0.406 / 0.315 / 0.226 |
| `chain-rerank` | 0.290 | 0.387 | 0.317 | 0.388 | 0.393 | 0.486 / 0.307 / 0.272 |
| `union-rerank` | 0.233 | 0.290 | 0.260 | 0.338 | 0.333 | 0.406 / 0.315 / 0.226 |
| `fusion` | 0.327 | 0.433 | 0.303 | 0.381 | 0.373 | 0.506 / 0.317 / 0.194 |

Each arm against `hipporag2` on the same questions: a paired bootstrap interval on F1, and the questions where only one of the two has every supporting passage in its top 10.

| Arm | F1 difference (95% interval) | Complete @10, only this arm / only `hipporag2` | F1 gain over `dense` |
|---|---|---|---|
| `hybrid3` | -0.080 (-0.129 to -0.034) | 17 / 43 | +0.060 |
| `hybrid3-rerank` | -0.052 (-0.103 to -0.001) | 19 / 51 | +0.088 |
| `chain-rerank` | -0.003 (-0.053 to +0.047) | 29 / 30 | +0.138 |
| `union-rerank` | -0.052 (-0.103 to -0.001) | 19 / 49 | +0.088 |
| `fusion` | -0.009 (-0.045 to +0.027) | 18 / 5 | +0.131 |

Retrieval alone, with no answer step: `splade` has every supporting passage in its top 10 for
0.243 of questions, `colbert` for 0.300, and `hybrid3` with the `ms-marco-MiniLM-L-6-v2`
reranker for 0.287.

What this says, scoped to this reader, these 300 questions, and RI #4's atlas-rag index:

- A stronger local-model stack closes about 63% of the F1 gap: `hybrid3-rerank` gains
  0.088 F1 over dense and `hipporag2` still leads it by 0.052 (95% interval 0.001 to 0.103).
  On retrieval the gap holds better: all-supporting recall at 10 rises from 0.237 to 0.283,
  against 0.390 for `hipporag2`. The graph wins against a small embedder by 0.140 and against
  this stack by about a third of that.
- The reranker does not keep the graph's candidates. `union-rerank` puts the same five
  passages in front of the reader as `hybrid3-rerank` for 299 of 300 questions, because a
  cross-encoder scores a hop-two passage against a question that does not name its subject.
- The bridge is what matters, and it does not need a graph. `chain-rerank` retrieves the
  second hop from the first hop's text and ties `hipporag2`: F1 0.388 against 0.390 (difference
  -0.003, interval -0.053 to +0.047), judged 0.393 against 0.377, and complete evidence on 29
  questions only it finds against 30 only `hipporag2` finds. It leads on four-hop questions
  (F1 0.272 against 0.204) and trails on three-hop (0.307 against 0.374). It costs three
  first-stage retrievals and about 200 cross-encoder pairs per question, no model call, and no
  index build, against 11,285 calls to build the HippoRAG 2 index.
- Fusing the strong stack with the graph retrieves the most complete evidence (0.433 at 10;
  18 questions complete only for `fusion` against 5 only for `hipporag2`, p = 0.011) but
  answers no better (F1 0.381). RI #4's fusion with the small dense retriever kept a fifth of
  the graph's gain.

### X1 with `gemini-3.5-flash-lite` as the reader (MuSiQue)

The same 300 questions and rankings, answered by `gemini-3.5-flash-lite` from each arm's top
five passages and judged by `gemini-3.8-flash`. RI #4's `standard` prompt offers "unknown";
its `best-guess` prompt does not, and RI #4's reader swap found that Gemini's gap is mostly
abstention. This run answers `hipporag2` again from RI #4's ranking, so each arm is compared
with a `hipporag2` answered in the same run.

| Prompt | Arm | All-supporting @10 | Exact match | F1 | Judged | F1 by hops (2 / 3 / 4) | Answered "unknown" |
|---|---|---|---|---|---|---|---|
| `standard` | `hipporag2` | 0.390 | 0.200 | 0.256 | 0.263 | 0.386 / 0.168 / 0.084 | 0.63 |
| `standard` | `hybrid3-rerank` | 0.283 | 0.180 | 0.224 | 0.233 | 0.326 / 0.180 / 0.062 | 0.63 |
| `standard` | `chain-rerank` | 0.387 | 0.217 | 0.263 | 0.283 | 0.362 / 0.180 / 0.148 | 0.61 |
| `best-guess` | `hipporag2` | 0.390 | 0.317 | 0.420 | 0.413 | 0.526 / 0.367 / 0.260 | 0.00 |
| `best-guess` | `hybrid3-rerank` | 0.283 | 0.280 | 0.376 | 0.357 | 0.421 / 0.348 / 0.316 | 0.00 |
| `best-guess` | `chain-rerank` | 0.387 | 0.330 | 0.433 | 0.443 | 0.515 / 0.396 / 0.307 | 0.00 |
| `best-guess`, top 10 read | `hipporag2` | 0.390 | 0.320 | 0.441 |  | 0.537 / 0.387 / 0.302 | 0.00 |
| `best-guess`, top 10 read | `chain-rerank` | 0.387 | 0.350 | 0.466 |  | 0.550 / 0.387 / 0.374 | 0.00 |

`chain-rerank` against `hipporag2` from the same run, paired bootstrap on F1:

| Prompt | F1 difference (95% interval) | F1 difference by hops (2 / 3 / 4) |
|---|---|---|
| `standard` | +0.007 (-0.039 to +0.054) | -0.024 / +0.012 / +0.064 |
| `standard`, repeat run | +0.010 (-0.033 to +0.055) | -0.024 / +0.038 / +0.051 |
| `best-guess` | +0.013 (-0.038 to +0.063) | -0.011 / +0.028 / +0.046 |
| `best-guess`, top 10 read | +0.025 (-0.021 to +0.071) | +0.014 / +0.000 / +0.072 |

What this says, scoped to `gemini-3.5-flash-lite`, these 300 questions, and RI #4's atlas-rag
index:

- The tie holds with this reader under both prompts. `chain-rerank` leads `hipporag2` by
  0.007 to 0.025 F1 in every setting, with every interval spanning zero, and leads on
  four-hop questions by 0.046 to 0.072 each time. With the on-prem reader, `chain-rerank`
  trailed on three-hop questions by 0.067; with this reader it does not trail there.
- The reader does not answer at temperature 0 the same way twice. Over three runs of
  `hipporag2` with the `standard` prompt (RI #4's committed run and two here, on identical
  passages), F1 is 0.236, 0.256, and 0.245, and 10 to 13 of 300 questions change score
  between any two runs; with `best-guess`, 22 change. Differences between arms smaller than
  about 0.02 F1 are within this reader's own run-to-run spread.
- The `standard` prompt hides retrieval differences. The reader answers "unknown" for 61% to
  63% of questions whatever the arm, including `chain-rerank` and `hipporag2`, which hold
  every supporting passage at 10 for 39% of questions against 28% for `hybrid3-rerank`.
- The bridge step still pays with a reader that guesses: `chain-rerank` gains 0.057 F1 over
  `hybrid3-rerank` with `best-guess` (0.433 against 0.376). On four-hop questions it does
  not: `hybrid3-rerank` scores 0.316, above both `chain-rerank` (0.307) and `hipporag2`
  (0.260). No arm has every supporting passage in its top 10 for more than 2 of the 75
  four-hop questions, so four-hop scores measure answering from partial evidence, and the
  four-hop lead of `chain-rerank` over `hipporag2` says more about that than about retrieval.
- Reading ten passages instead of five adds 0.033 F1 to `chain-rerank` and 0.021 to
  `hipporag2`. The best setting here, `chain-rerank` with `best-guess` over ten passages
  (F1 0.466), uses no graph and no model call before the answer.

### X3: relational questions over the ServerFault graph

Sixty questions, fifteen per family, generated from graph patterns with a fixed seed and no
model call:

- `count`: "How many posts in the collection mention X?"
- `co_mention`: "Which posts mention both X and Y?"
- `relation`: "Across all posts, which errors are reported as caused by X?" and the
  `depends_on` and `fixes` variants.
- `path`: "Which concepts, tools, or settings link X and Y?", where no single post mentions
  both, so only a link across posts answers it.

`count` and `co_mention` have two golds: what the graph returns, and the posts whose text
contains the name as a whole word (`lexical`), which does not depend on the graph. `relation`
and `path` have only the graph gold, so a hand check bounds what it is worth. The arms read
the same 9,999 posts or the same graph. `dense` reads the top 10 posts by
`bge-small-en-v1.5`; `dense-map` asks the reader about each of the top 50 in turn. `grep`
finds the posts that contain the names; `grep-map` asks the reader about each (up to 50).
`graph-template` runs one fixed SQL query per family over the graph's tables, and
`graph-like` matches entity names by substring. `text2sql` has the reader write the query from
the schema and 25 sample names, and `text2sql-repair` adds one retry when the query fails or
returns nothing. SQL over SQLite stands in for Cypher because the locked environment has no
graph database; the failure under test, a query that runs and returns nothing, is the same.

| Arm | `count` within 20%, graph / lexical | `co_mention` F1, graph / lexical | `relation` F1 | `path` F1 | Model calls |
|---|---|---|---|---|---|
| `dense` | 0.00 / 0.00 | 0.31 / 0.24 | 0.09 | 0.07 | 60 |
| `dense-map` | 0.13 / 0.07 | 0.32 / 0.38 | 0.14 | 0.02 | 3,000 |
| `grep` | 0.20 / 1.00 | 0.80 / 1.00 | 0.23 | 0.09 | 30 |
| `grep-map` | 0.20 / 0.80 | 0.74 / 0.92 | 0.18 | 0.00 | 1,268 |
| `graph-template` | 1.00 / 0.20 | 1.00 / 0.80 | 1.00 | 1.00 | 0 |
| `graph-like` | 0.33 / 0.60 | 0.91 / 0.86 | 1.00 | 1.00 | 0 |
| `text2sql` | 1.00 / 0.20 | 1.00 / 0.80 | 0.73 | 0.07 | 60 |
| `text2sql-repair` | 1.00 / 0.20 | 1.00 / 0.80 | 0.85 | 0.07 | 77 |

The `graph` columns score access to the graph, so the template arm scores 1.00 by
construction. Model calls count every call an arm needs from an empty cache.

The hand check (`outputs/x3_hand_check.json`, one reader, at the cited span of each post):

- 12 of 20 relation facts behind the `relation` golds are supported by their post. The eight
  failures repeat two patterns: a symptom seen while running a tool stored as caused by it
  (`ping` causes `100% packet loss`), and a question or a rejected option stored as a fact.
- 14 of 14 sampled posts that `grep` finds and the graph's exact entity does not cite do
  mention the name, three of them in passing (a tag, a file path, an Ansible keyword).
- 9 of 15 path connectors are a meaningful link between the two ends, judged from the names;
  the rest are hubs such as `bash` or `nagios`.

What this says, scoped to RI #4's graph and these generated questions:

- For questions about which posts mention a name, the corpus is its own database and `grep`
  is exact at no cost. The graph's exact-name answer undercounts because entity resolution
  left one thing under many names, which `outputs/maintenance/dedup.json` in RI #4 measures
  from the other side.
- Relations across posts and paths are where only the graph answers: no retrieval arm passes
  0.23 F1 against the graph gold. That gold is about 60% right on relations, so the graph's
  lead is real and its answers need checking.
- Generated SQL matches the templates on counts and co-mentions, recovers most relation
  answers after one repair (0.85), and fails on paths: 14 of 15 first queries error or return
  nothing, and one retry fixes 1 of 14. Mixed-source routing saw the same empty-result failure
  with SQL over tables (13 of 20).

### X2: a readiness check before you build

`gon.readiness.graph_readiness` takes documents, and optionally questions, gold documents,
a search function, and entity names from any extractor. It makes no model call. Without names
it uses capitalised spans and code-like tokens (`sysctl.conf`, `/etc/hosts`).

| Corpus, questions, names | Names recurring | Documents linked | Relational questions | Bridge questions | Dense all gold @10 |
|---|---|---|---|---|---|
| musique, multi-hop questions, graph names | 0.26 | 1.00 | 0.11 | 0.45 | 0.24 |
| musique, multi-hop questions, regex names | 0.27 | 1.00 | 0.11 | 0.48 | 0.24 |
| serverfault, support questions, graph names | 0.40 | 1.00 | 0.00 | 0.17 | 0.93 |
| serverfault, support questions, regex names | 0.29 | 1.00 | 0.00 | 0.07 | 0.93 |
| serverfault, X3 relational questions, graph names | 0.40 | 1.00 | 1.00 |  |  |
| contractnli, hypotheses, regex names | 0.27 | 1.00 | 0.00 |  |  |

What the statistics separate on these corpora:

- Bridge share and dense headroom separate the corpora where a graph paid from the one where
  it had nothing to do. The ServerFault row uses RI #5's 30 error-string support questions. MuSiQue needs a document that names nothing in the question for 45%
  of questions (48% with regex names) and dense retrieval holds all gold for 24%; the
  ServerFault support questions need one for 7% to 17%, and dense holds all gold for 93%.
- Regex names give the same reading as the graph's names on bridge share, so the check needs
  no extraction pass to make that call.
- Name recurrence does not separate anything. With recurrence counted lexically, every
  corpus links all its documents and 26% to 40% of names recur. A recurrence statistic that
  separates corpora needs names weighted by how rare they are, which this version does not do.
- The relational-question share reads the question set, not the corpus: 0 for the support
  questions, 1.00 for the X3 questions, 0.11 for MuSiQue.

## How each could still win

- The strong stack does not close the gap to `hipporag2` (F1 0.052 behind; 0.283 against
  0.390 at 10). It
  would close it if a stronger embedder found the hop-two passage from the question alone.
  Cheapest test: the same arms with `Qwen3-Embedding-0.6B`, about one GPU
  hour and 900 on-prem calls.
- `union-rerank` gains nothing from the graph's candidates. A reranker would keep them if it
  scored a passage against the question plus the first hop. Cheapest test: rerank the union
  against the `chain-rerank` expanded queries, 300 questions, no model calls.
- The graph loses mention counts to `grep`. It would win if entity resolution folded name
  variants onto one node. Cheapest test: rerun the 15 count questions after RI #4's
  name-folding deduplication, no model calls.
- Text2SQL fails on paths. It would succeed with a path template or a recursive-query
  example in the prompt. Cheapest test: one worked path example in the schema prompt, 15
  questions, about 30 calls.
- With `gemini-3.5-flash-lite` reading, `hipporag2` does not beat `chain-rerank` under
  either prompt. The graph would win if its passages answered questions the bridge step
  cannot reach, which the 30 questions only `hipporag2` holds complete evidence for suggest.
  Cheapest test: answer from the union of both arms' top five, 300 questions, about 300
  reader calls and $0.10.
- Name recurrence predicts nothing. It might with names weighted by inverse document
  frequency. Cheapest test: recompute on the four rows here, no calls.

## Files

- `run_graph_or_not.py` runs X1, X3, and X2 and writes `outputs/x1_musique.json`,
  `outputs/x3_serverfault.json`, and `outputs/x2_readiness.json`; `x3-finalize` merges the
  hand check into the X3 file.
- `outputs/x1_musique_gemini_standard.json`, `x1_musique_gemini_standard_repeat.json`,
  `x1_musique_gemini_best_guess.json`, and `x1_musique_gemini_best_guess_top10.json` hold
  the `gemini-3.5-flash-lite` rows.
- `rank_x1.py` ranks the X1 questions; it needs only `torch` and `sentence-transformers`, so
  it runs on a GPU machine outside the bootcamp environment, and resumes from a partial file.
- `gon/x1.py` holds the retrieval arms, `gon/relational.py` the question generator, golds,
  scorers, and X3 arms, and `gon/readiness.py` the readiness check.
- `outputs/x3_hand_check.json` holds the hand labels; `outputs/ATTRIBUTION.md` the licences.

## Rebuild the reference run

On-prem calls cost nothing and the judge is paid, so the judge needs a response cache of its
own and a spend ceiling:

```bash
python run_graph_or_not.py x1 --write-inputs
```

```bash
python rank_x1.py --inputs <work>/x1-inputs-300.json --out <work>/x1-rankings-300.json --device cuda --colbert-device cuda
```

```bash
ADVANCED_RAG_SPEND_LIMIT_USD=10 python run_graph_or_not.py x1 --judge --paid-cache <paid_responses.sqlite>
```

The `gemini-3.5-flash-lite` rows reuse the rankings and take the reader, the prompt, and the
RI #4 file with that reader's `dense` and `hipporag2` rows:

```bash
ADVANCED_RAG_SPEND_LIMIT_USD=2 python run_graph_or_not.py x1 --reader gemini-3.5-flash-lite --arms hipporag2 hybrid3-rerank chain-rerank --answer-prompt best-guess --baseline musique_reader_swap_best_guess_gemini.json --judge --paid-cache <paid_responses.sqlite> --out x1_musique_gemini_best_guess.json
```

```bash
python run_graph_or_not.py x3
```

```bash
python run_graph_or_not.py x2
```

where:

- `<work>` is `$ADVANCED_RAG_CACHE/special_topics/graph-or-not`, the cache folder, which also
  holds the inputs file with passage text and the full X3 payload with question wording;
  neither is committed.
- `<paid_responses.sqlite>` is a response cache file kept apart from the on-prem one.

Ranking took under an hour on one shared GPU with ColBERT scoring on it, split across three
processes, and takes several hours on a CPU. X1 answering makes 1,500 on-prem calls; judging makes 1,500 calls to
`gemini-3.8-flash`. Each Gemini prompt with three arms and the judge cost about $0.75. X3 makes about 4,500 on-prem calls from an empty cache. X2 makes none.
