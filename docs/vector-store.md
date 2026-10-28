# Vector store reference

Every dense arm in the package keeps its vectors in Qdrant through
`aieng.advanced_rag.retrieval.stores`. This page lists the modes, the settings,
the cache format, and the measured trade-offs of each setting.

Install the `qdrant` extra (`qdrant-client==1.19.1`). `uv sync --all-extras`
includes it.

## Modes

| Mode | How to choose it | Search | Needs |
| --- | --- | --- | --- |
| In memory | `":memory:"`, the default | Exact numpy scan | Nothing beyond the extra |
| On disk | A directory path | Exact numpy scan | A writable directory |
| Server | An `http://` or `https://` URL, or `QDRANT_URL` with `connect(None)` | Exact by default; HNSW and quantization when `QdrantSettings` ask | A running Qdrant server |

In-process mode reproduces the in-memory rankings every committed figure was
measured with, and it is what every RI uses. qdrant-client recommends it below
about 20,000 points: MuSiQue's 11,656 passages fit, the ServerFault 10k slice's
23,525 chunks do not.

In-process mode has no HNSW graph and no quantization, and qdrant-client ignores
those settings there without a warning. `QdrantStore` raises `ValueError`
instead, so a notebook cannot report float32 results under a quantized label.

A directory path is a convenience for one machine. It holds qdrant-client's own
SQLite file of pickled points, which a server cannot read and another client
version might not, so it is never the shipping format.

## Classes and functions

`QdrantStore(client=":memory:", *, collection="chunks", settings=None)` holds
one collection. `client` is a `QdrantClient` or a location string.
`add(keys, vectors, doc_ids=None)` stores normalized vectors under string keys,
and refuses a repeated key or a changed width. `search(vector, k, *, scope=None,
within=None)` returns `(key, score)` pairs, and `search_batch(vectors, k, *,
scopes=None, within=None)` runs several queries in one call. `scope` filters on
the `doc_id` payload; `within` restricts to point identifiers. `vectors()` reads
the stored matrix back in point order.

`QdrantRetriever(embedder, store=":memory:", *, settings=None)` is the dense arm.
`DenseRetriever` is the same class under the name every RI imports, and its
`name` stays `"dense"`, so cache keys and evidence labels do not change. It
satisfies the `Retriever` protocol unchanged. Without a `scope`, `retrieve` runs
one search over every document.

`DenseIndex(chunk_ids, vectors, doc_ids=None, *, store=None)` builds a store
from vectors computed elsewhere, such as late chunking, and ranks a batch of
queries in one call. The graph dense control
(`graphrag.adapters.dense.DenseRetriever`) and RI #4's `DenseArm` use the same
store.

The store scores by dot product, and every embedder in the package normalizes
its output. Cosine distance would normalize a second time in float32 and move
scores in the last bit.

## Settings

`QdrantSettings` fields are all off by default, which gives an exact search over
the float32 vectors. Every field other than the defaults needs a server.

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `quantization` | `"binary"`, `"turbo"`, or `None` | `None` | Keeps a compressed copy of each vector in RAM and scores against it. The float32 vectors stay on disk for rescoring. |
| `bits` | float | `1.0` | Bits per dimension. TurboQuant takes 1, 1.5, 2, or 4; binary takes 1, 1.5, or 2. |
| `asymmetric` | bool | `False` | Binary only: encodes the query as 8-bit scalars. |
| `oversampling` | float or `None` | `None` | Fetches `oversampling * k` candidates by compressed score before rescoring. Needs `quantization`. |
| `rescore` | bool or `None` | `None` | Re-ranks the oversampled candidates with the float32 vectors. `None` keeps the server default, which on Qdrant 1.19 is on for binary and for TurboQuant below 4 bits. |
| `hnsw_ef` | int or `None` | `None` | Searches the HNSW graph with this beam width. `None` keeps an exact scan. |
| `hnsw_m` | int or `None` | `None` | Graph degree at build time. |
| `ef_construct` | int or `None` | `None` | Beam width at build time. |
| `indexing_threshold_kb` | int or `None` | 1 when any server setting is on | Segment size above which the optimizer builds the graph and the quantized copy. |
| `full_scan_threshold_kb` | int or `None` | 10 when `hnsw_ef` is set | Segment size below which a search scans exactly whatever its graph says. |

The last two exist because of a trap the scale experiment found. Qdrant's
`indexing_threshold` (20 MB) and `full_scan_threshold` (10 MB) apply per
segment. A 23,525-vector collection arrived as eight segments of about 4.5 MB,
so under the defaults it built no graph and no quantized copy, and every search
was an exact scan whatever the settings said. Lowering only the indexing
threshold builds the graph, and searches still skip it. After each `add`, a
server store waits until the server reports every vector indexed.

With quantization and no `hnsw_ef`, the store leaves the search path to the
server, which is how the compression sweep below was measured.

## Measured trade-offs

The numbers below come from two sweeps, one on vector compression and one on
retrieval at scale, run against a Qdrant 1.19.0 server with qdrant-client 1.19.0. Embeddings are `BAAI/bge-small-en-v1.5`
at 384 dimensions. "Agreement" is the share of the float32 top-10 an arm also
returns. "Ratio" divides the float32 vector bytes on disk by the quantized
vector bytes.

### Compression on the CUAD slice

CUAD test split, first 20 contracts, 954 chunks, 250 answerable queries, k=10,
each query scoped to its contract (about 48 candidates). Query-time settings are
the server defaults.

| Setting | Ratio | Agreement@10 | Top-1 match | Retrieval failure |
| --- | ---: | ---: | ---: | ---: |
| float32 | 1.0 | 1.000 | 1.000 | 0.240 |
| `turbo`, 4 bits | 7.7 | 0.924 | 0.896 | 0.252 |
| `turbo`, 2 bits | 14.8 | 0.887 | 1.000 | 0.248 |
| `binary`, 2 bits | 16.0 | 0.758 | 0.964 | 0.252 |
| `turbo`, 1.5 bits | 19.2 | 0.787 | 0.992 | 0.268 |
| `turbo`, 1 bit | 27.4 | 0.742 | 0.960 | 0.256 |
| `binary`, 1 bit | 32.0 | 0.682 | 0.892 | 0.268 |
| `binary`, 1 bit, `asymmetric` | 32.0 | 0.764 | 0.944 | 0.252 |

TurboQuant beat binary at every matched width. Retrieval failure stayed between
0.240 and 0.268 across the whole table while agreement fell from 0.924 to 0.682,
so a report that shows only task metrics cannot see quantization damage on this
slice. Rebuilding `turbo` at 2 bits three times moved retrieval failure by 0.004
and agreement by 0.014.

One `turbo` 1-bit collection, queried with different settings:

| Query setting | Agreement@10 | Top-1 match | Retrieval failure |
| --- | ---: | ---: | ---: |
| `rescore=False` | 0.690 | 0.492 | 0.260 |
| `rescore=True`, `oversampling=1` | 0.746 | 0.976 | 0.248 |
| `rescore=True`, `oversampling=2` | 0.930 | 0.992 | 0.244 |
| `rescore=True`, `oversampling=4` | 0.988 | 1.000 | 0.232 |

At `oversampling=4` with rescoring, 1-bit storage at 27.4x returned 98.8% of the
float32 top-10. Without rescoring, the top hit matched on half the queries, so
anything that reads only the first chunk is more exposed than the set metrics
show. Median latency was 1.4 to 1.5 ms for every row, because the scoped filter
leaves about 48 points and the server scans them exactly.

### HNSW and quantization on ServerFault

9,999 ServerFault documents chunked to 23,525 chunks, searched corpus-wide, with
both thresholds lowered and the HNSW path confirmed from the server's telemetry.
Agreement is against the same collection's exact search.

| Setting | Agreement@10 | Retrieval failure | Median ms |
| --- | ---: | ---: | ---: |
| exact | 1.0000 | 0.0781 | 10.3 |
| `hnsw_ef=16` | 0.9781 | 0.0857 | 7.7 |
| `hnsw_ef=64` | 0.9988 | 0.0787 | 15.6 |
| `hnsw_ef=128` | 0.9998 | 0.0782 | 16.2 |
| `hnsw_ef=256` | 1.0000 | 0.0782 | 18.9 |

Only `hnsw_ef=16` beat the exact scan, by 2.5 ms, and the same search repeated
with the graph unused spread over 4.3 ms, so that saving is at the edge of the
noise. Every larger `hnsw_ef` was slower than exact. At this size HNSW buys
nothing; the curve is real, and the accuracy is paid for in latency.

On the same collection with HNSW on, `turbo` at 2 bits returned 99.86% of the
float32 top-10, and `turbo` at 1 bit returned 97.66%, 99.51%, and 99.85% at
`oversampling` 1, 2, and 4 with rescoring, and 67.56% with rescoring off.
Median latency stayed within 9.7 to 12.1 ms against a run-to-run spread of 1.8
to 2.9 ms, and the server's resident memory stayed within 259 to 271 MiB, so
neither latency nor memory moved measurably with 36 MB of vectors.

### Reading these numbers

All of it is one encoder at 384 dimensions, one machine, and a server reached
over HTTP. A 1024-dimension encoder might bend the compression curve later, and
a corpus of a few hundred thousand points is where latency and memory start to
separate the settings.

## Ties and reproducibility

Qdrant returns equal scores in no promised order, and qdrant-client's
in-process scan returns them in reverse insertion order. CUAD repeats
boilerplate, so exact ties are common. `QdrantStore.search` fetches `k + 16`
points, widens the fetch while its weakest point ties the k-th, and sorts on
`(-score, key)`. That reproduces the chunk-ID tie-break the published figures
used.

Vectors are stored as float32. In-process mode multiplies them by the query in
float64, as the list-based retriever this store replaced did, and on the CUAD
slice the two disagreed by at most 4.4e-16 on any score. A server scores in
float32, so a score there can differ in the seventh decimal place.

Every committed dense ranking was re-run through the store and compared with
the code it replaced, query by query, at k=10:

| Ranking | Queries | Differing from the old code | Differing from the committed output |
| --- | ---: | ---: | ---: |
| CUAD 20 contracts, no query prompt (`test_published_baselines.py`) | 250 | 0 | 0 (failure 0.2400, MRR 0.4348) |
| CUAD 20 contracts, default query prompt | 250 | 0 | 0 (failure 0.2400, MRR 0.4632) |
| RI #1 threshold sweep, first retrieval, CUAD 8 contracts | 328 | 0 | 0 |
| RI #1 threshold sweep, ContractNLI 40 documents | 680 | 0 | 0 |
| RI #1 threshold sweep, MultiHop-RAG nulls over ContractNLI | 150 | 0 | 0 |
| RI #1 null arm, dense scoped to the contract | 340 | 0 | 0 |
| RI #1 null arm, dense unscoped | 340 | 0 | 0 |
| RI #4 MuSiQue dense arm, 11,656 passages | 300 | 0 | 0 |
| RI #3 FinanceBench, plain chunking, 41,401 chunks | 150 | 0 | 0 (failure 0.3400, MRR 0.3459) |
| RI #3 FinanceBench, table-aware chunking, 53,715 chunks | 150 | 0 | 0 (failure 0.4000, MRR 0.3542) |

The committed outputs are the trace files and result JSON under
`implementations/`, and the stored evidence tables the RI #1 null arm replays.
The old unscoped control sorted with an unstable `argsort` and so left ties in
an unspecified order; no query in the 340 had a tie that moved.

An in-process search over the 11,656 MuSiQue passages took 28 ms against 13 ms
for the numpy matrix product it replaced, on a shared, loaded machine. Encoding
the question costs more than either, so no measured reason kept the numpy
path.

## Cached collections

`export_collection(store, directory, metadata=..., payloads=...)` writes three
files, and `load_collection(directory, store)` rebuilds them into any empty
store, in memory, on disk, or on a server:

- `vectors.npy`: float32, row i is point i.
- `points.jsonl`: one payload per point, in point order, with `key`, `doc_id`,
  and any extra fields, such as chunk offsets.
- `collection.json`: the export format, the width, the distance, the
  qdrant-client version, and the caller's metadata.

`cache_collection(cache, key, store)` and `fetch_collection(cache, key, store)`
put the directory through the artifact cache as one archive
(`store_directory` and `fetch_directory`). Use the artifact type
`"vector-collection"`, and put what changes the vectors in the key's `config`:
the corpus digest, the chunker, the embedding model, and the query prompt. Leave
the store mode out, so one artifact loads into every mode.

`QdrantRetriever.load_or_index(cache, key, chunks)` does both for the dense arm:
on a hit it loads the vectors and skips the encoder, on a miss it embeds and
caches. `save(directory)` and `load(directory, chunks)` are the same without the
cache. The export holds vectors and offsets and no text, so `load` takes the
chunks and refuses them if their identifiers or spans differ from the export,
or if the retriever's embedder or prompts differ from the recorded ones.

## Start a server

A server is needed only for the settings above. The release binary runs without
Docker. Download `qdrant-x86_64-unknown-linux-musl.tar.gz` for v1.19.1 from the
[Qdrant releases page](https://github.com/qdrant/qdrant/releases/tag/v1.19.1),
extract it, and start it:

```bash
$ QDRANT__STORAGE__STORAGE_PATH=./qdrant-storage ./qdrant
```

Then pass `"http://localhost:6333"` as the store location, or set `QDRANT_URL`.
The server tests in `tests/aieng/advanced_rag/retrieval/test_stores.py` run when
`QDRANT_URL` is set.

## Stores this package does not own

LightRAG and atlas-rag keep their own stores: NanoVectorDB JSON files for
LightRAG, and a FAISS `IndexHNSWFlat` for atlas-rag's HippoRAG 2. Changing them
would mean re-running the paid graph extraction or rewriting library
internals, and the published graph numbers come from those stores. LightRAG
1.5.6 ships `QdrantVectorDBStorage`, which reads `QDRANT_URL` and needs a
server, for a new build that wants Qdrant.
