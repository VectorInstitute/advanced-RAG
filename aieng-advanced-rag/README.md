## aieng-advanced-rag

Shared library for the Advanced RAG Bootcamp: the evaluation harness every
reference implementation reports against, and the boundary a GraphRAG method
implements.

## Evaluation harness

The harness scores a retriever against judgments derived from a dataset rather
than from a model, so a retrieval number is measured. `retrieval` holds the
corpus loader, the chunkers, the embedders, and the retrievers. `evaluation`
holds the metrics. The default install pulls no dependencies: the corpus loader,
every metric, both offset-preserving splitters, the table-aware chunker, BM25, and
the pure-Python dense retriever all run under it.

### Score a retriever

```python
from pathlib import Path

from aieng.advanced_rag.evaluation import score_retrieval
from aieng.advanced_rag.retrieval import (
    DenseRetriever,
    SentenceTransformerEmbedder,
    build_slice,
    run_retriever,
)

sliced = build_slice(Path("data/cuad/test.json"), max_contracts=20)
retriever = DenseRetriever(SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"))
retriever.index(sliced.chunks)
runs = run_retriever(retriever, sliced.answerable, k=10)
report = score_retrieval(runs, sliced.qrels, k=10)
print(report.to_dict())
```

`build_slice` reads a CUAD SQuAD file, chunks each contract into overlapping
character windows, and derives a judgment set per query: a chunk is relevant when
its window overlaps a gold answer span. `Slice.answerable` is the queries with a
gold span, `Slice.null_queries` is the rest, and `Slice.qrels` maps a query
identifier to its relevant chunk identifiers.

`run_retriever` returns ranked chunk identifiers per query, by driving the
`index` and `search` pair. `score_retrieval` takes that mapping and the judgments
and returns a `RetrievalReport`. Scoring reads ranked identifiers and nothing
else, so anything that returns `Evidence`, including a graph traversal or an
agent's tool trace, is scored by mapping it through `to_runs` instead:
`{query.query_id: to_runs(retriever.retrieve(query.question, 10, scope=query.doc_id))}`.

### Retrieval interface

`retrieval.interface.Retriever` is the surface to program against. A `name`, one
method, and its asynchronous twin are required:

```python
retrieve(question, k=10, *, scope=None, hints=None) -> list[Evidence]
```

`Evidence` carries the passage text and its provenance, so a caller never joins a
chunk identifier back to the corpus. `scope` is the corpus's own filter and `None`
means the whole corpus; a method that cannot filter raises `ScopeUnsupported`
rather than ignoring the argument or dropping passages from a finished ranking,
which would return fewer than `k`. `hints` are advisory: an implementation that
ignores every field returns the ranking it would have returned for `hints=None`,
which is what stops a hint from moving a measured number. `aretrieve` is the same
call for an asynchronous caller, and it is the real method of the LightRAG
adapter. Calling a retriever directly, as `retriever(question, scope, k)`, is the
shape `pipeline.spine` takes, and `k` defaults to 10 for a caller that has no
opinion. The spine passes `SpineConfig.top_k`, so a configured depth reaches the
method.

The interface promises no ordering. Passages arrive in the order the method ranked
them, and scores are not promised to descend: `ShuffledRetriever` scores every hit
0.0, and the LightRAG and HippoRAG 2 adapters report no per-passage score at all.

Controls beyond `k` are optional capabilities an implementation offers by defining
their methods: `Indexable`, `Findable`, `Fetchable`, `Expandable`, `Readable`, and
`Greppable`. Ask `declares(retriever, "fetch")` rather than `isinstance`, because
`index`, `fetch`, `find`, and `read` are names unrelated objects carry: a list has
an `index` method and does not retrieve. Each harness retriever gains find, fetch,
expand, read, and grep from the `ChunkStore` it builds at index time, which is
also what lets `retrieve` return text. `index` and `search` stay as each harness
retriever's internal ranking core, typed as `Ranker`, and `run_retriever` still
drives them.

`retrieval.conformance` checks an implementation against all of this:

```python
from aieng.advanced_rag.retrieval.conformance import assert_conforms

assert_conforms(
    my_retriever, "which law governs?", chunks=my_chunks, scope="contract-1"
)
```

It checks shape and provenance, and never order, because a descending-score rule
would rewrite the negative control's published numbers. `CHECKS` lists what it
looks for and `NOT_CHECKED` lists what it leaves alone.

### Retrieval metrics

Every metric takes ranked chunk identifiers and a set of relevant identifiers.
Queries with an empty judgment set are excluded from `score_retrieval`, because a
question with no answer in the corpus has no relevant chunk to retrieve and
averaging it in would report a failure for correct behaviour.

| Metric | Definition |
| --- | --- |
| `retrieval_failure` | True when the top-k holds nothing relevant. Its mean over queries is the retrieval-failure rate, the share of answerable questions where nothing relevant reached the reader. |
| `hit_at_k` | 1.0 when the top-k holds something relevant. The per-query complement of `retrieval_failure`. |
| `recall_at_k` | Share of the relevant chunks that appear in the top-k. |
| `reciprocal_rank` | 1/rank of the first relevant chunk over the full ranking, 0.0 when none was retrieved. Its mean is mean reciprocal rank (MRR). |
| `ndcg_at_k` | Binary-gain normalized discounted cumulative gain (nDCG) at k. Binary because span-derived judgments are binary: a chunk either overlaps a gold answer or it does not. |
| `retrieved_chars` | Mean characters the reader receives from the top-k. Compare chunk sizes at equal characters with `budget_k`, since a fixed k rewards large chunks for retrieving more of the document. |

`score_null_detection` scores the other half of the CUAD task. Around 70% of the
test split is unanswerable, so a system that always answers scores well on
retrieval and still fails. It takes a mapping from query identifier to whether
the system abstained, and returns a `NullReport` with abstain precision, abstain
recall, and the false-answer rate.

### Answer metrics

`judge_answer` scores one generated answer on correctness, grounding, and
abstention as independent fields, against a reference answer and the evidence the
system was given. `AnswerReport` aggregates the verdicts and derives a
hallucination rate over attempted answers only, so an abstention is not counted as
a hallucination. `judge_coverage` scores a fan-out answer against what each gold
document was required to contribute, and `CoverageReport` reports both the mean
share of contributions stated and the share of questions where every contribution
was stated.

Both judges take a completion function, so they depend on no client and can be
driven by a cached, metered one. Neither uses embeddings.

### Retrievers, embedders, and chunkers

`DenseRetriever` scores cosine similarity over normalised vectors held in memory,
scoped to the query's own document, with ties broken on chunk identifier.
`BM25Retriever` is the lexical arm, fitted over the whole slice so a term's rarity
is a property of the corpus. `HybridRetriever` fuses two indexed arms with
weighted reciprocal rank fusion. `ShuffledRetriever` is the negative control: the
same candidate pool in a fixed random order, so its failure rate is the floor a
baseline has to beat. `CrossEncoderReranker` reorders first-stage candidates.

`HashEmbedder` produces deterministic hashed bag-of-words vectors with no
dependencies, for tests and smoke runs. `SentenceTransformerEmbedder` runs a local
checkpoint and applies the model's published query prompt to queries and not to
documents; pass `query_prompt=""` for a deliberately prefix-free arm. `MODELS`
records the published prompts per Hugging Face identifier.

Chunkers all preserve character offsets, which is what lets judgments be derived
per chunking. `fixed_chars` is the control, `recursive_tokens` splits on a
separator hierarchy to a token budget, `semantic` cuts where consecutive sentence
embeddings stop agreeing, and `table_aware_chunks` chunks a converted filing
without splitting a table away from its header rows.

`retrieval.index.DenseIndex` is the same ranking in a numpy matrix, for sweeps
that re-index a corpus many times. It ranks identically to `DenseRetriever`,
tie-breaks included.

### Bulk model calls

`batch.ProxyBatch` sends many independent calls to one model through
[infermesh](https://github.com/VectorInstitute/infermesh), Vector's batching client on
LiteLLM. infermesh holds the batch to a request rate, keeps a bounded number of
requests in flight, and retries only transient errors (429, connection failures,
timeouts, and 5xx). The package's response cache answers first and stores each
result as it arrives, so a repeated or interrupted run sends only what is missing,
and a fully cached run needs no key. Calls report to the usage meter, and the spend
limit and the confirmation check apply.

```python
from pathlib import Path

from aieng.advanced_rag.batch import ProxyBatch
from aieng.advanced_rag.graphrag.extraction import build_graph

with ProxyBatch("Qwen3_6-35B-A3B", cache=Path("responses.sqlite")) as batch:
    outcomes = batch.complete_batch(prompts, max_tokens=200, temperature=0.0)
    graph = build_graph(documents, batch.completer(max_tokens=2000, temperature=0.0))
```

`graphrag.extraction.build_graph`, `evaluation.answers.judge_answers`,
`retrieval.contextual.generate_contexts` and `name_documents`, and the synthetic
question generator's `proxy_model` send their calls this way. Keep
`graphrag.proxy.govern` for a library that needs an `OpenAI` client, such as atlas-rag.

### Extras

| Extra | Adds |
| --- | --- |
| `local` | `SentenceTransformerEmbedder`, on a local checkpoint. |
| `retrieval` | `retrieval.index` and semantic chunking. |
| `chunking` | Token-budget chunking, counted with the embedding model's own tokenizer. |
| `rerank` | `CrossEncoderReranker`. |
| `filings` | `table_aware_chunks` over a converted document. |
| `late` | `retrieval.late`, which pools token vectors from a long-context encoder. |
| `lightrag`, `hipporag`, `graphrag` | The GraphRAG adapters. |
| `batch` | `batch.ProxyBatch`, on infermesh. |
