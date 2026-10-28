# RI #2: Corpus provenance and loader validation

MuSiQue is the main corpus and NaturalQuestions is the single-hop control,
both using pinned HippoRAG 2 packaging. MultiHop-RAG is an opt-in secondary
corpus. This record distinguishes source pins, annotation semantics, packaging
terms and article rights. The loaders
live in the shared package's `corpora/` directory.

## Corpus overview

| Corpus | Role | Loader |
| --- | --- | --- |
| MuSiQue | Main, multi-hop Wikipedia questions | Pinned files and hop graphs |
| NaturalQuestions | Single-hop control | Pinned files, aliases, and support mappings |
| MultiHop-RAG | Secondary, answerable query types | Authorization-checked article parser for three answerable query types |

Source licence labels describe the underlying records. Raw files, article bodies, vectors, and graph indexes are not committed.

## Pinned files

Revision: `5ec05b38deecc3318bb432c69865959c56058990`.
The source URLs are stored beside the digests in `MUSIQUE_QUESTIONS` and
`MUSIQUE_CORPUS`.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `musique.json` | 12,543,629 | `98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46` |
| `musique_corpus.json` | 6,239,261 | `73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb` |

`fetch_musique()` follows the shared pinned-source download and cache pattern.
`load_musique()` also verifies the bytes it parses, including explicitly supplied
`MusiquePaths`. A mismatched digest raises `ChecksumMismatchError` before parsing.
Raw files stay in the local corpus cache and are not committed.

## Hop graphs

Passage IDs are revision-specific corpus positions, from `musique::00000` through
`musique::11655`. Candidate paragraphs map to corpus passages by both title and
text. The supporting IDs agree with the decomposition's supports.

`question.hop_graph.nodes` contains the decomposition in source order, including
each step's original `step_id`, question, answer, supporting passage ID, and
`depends_on` identifiers. `question.hop_graph.edges` contains directed
`(prerequisite_id, dependent_id)` pairs. A `#1` placeholder means the first
decomposition step, whose original ID need not be 1. Repeated references produce
one dependency edge. Independent branches are preserved.

Parsing rejects duplicate question IDs, paragraph indices, step IDs, and supports;
invalid hop counts; missing supports; decomposition/support disagreement; and
zero, self, forward, or out-of-range references. Backward references make the
source order topological and prevent cycles. Gold sub-answers and graphs are
evaluation annotations, not inputs to held-out answering.

## Validation

The full-data test verifies 1,000 unique questions, 11,656 passages, the
518 two-hop / 316 three-hop / 166 four-hop split, 2,648 supporting passage
occurrences, and 1,648 directed edges. All question graphs are connected and
acyclic, and every supporting passage resolves to a corpus ID. The machine-readable
record is [MuSiQue validation](outputs/validation/musique.json).

To repeat, request a compute allocation first. Inside that allocation, set
`ADVANCED_RAG_CORPUS_CACHE` to a writable local cache, then run:

```bash
python -m pytest aieng-advanced-rag/tests/aieng/advanced_rag/corpora -m 'not integration_test' -q
python -m pytest aieng-advanced-rag/tests/aieng/advanced_rag/corpora/test_musique_integration.py -m integration_test -q
```

The integration test fetches and verifies the pinned files when they are absent.

## Licence status and attribution

The existing source record states CC BY 4.0 for MuSiQue. The HippoRAG 2 packaging
card states no licence. Source licensing does not establish packaging rights.
The loader is not registered in `corpora.SOURCES`.

Carry the source credit line with outputs that quote corpus text:

```text
MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi,
Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique
```

## NaturalQuestions control

The shared `corpora/natural_questions.py` loader uses HippoRAG 2 revision
`5ec05b38deecc3318bb432c69865959c56058990`.

| File | SHA-256 |
| --- | --- |
| `nq_rear.json` | `8acadaeff625790e8e10764d660d9ec46de6ac24eb241ac584c0171deca1d9cd` |
| `nq_rear_corpus.json` | `c672a7c1813f1a3e1e7a0b1a29784bf7c5cf2d7a2934cb3ffc3b347843b42864` |

Both downloaded and explicit local paths are checked before parsing. Passage IDs
are `nq::00000` onward; question IDs are `nq::0000` onward, stable within this
revision. References retain every accepted answer string. Supporting contexts map
by title and exact text, with repeated identical contexts deduplicated.
The pinned files contain 1,000 questions and 9,633 passages.
Full-data validation resolves 3,794 supporting-passage occurrences after
deduplication. The [validation record](outputs/validation/nq.json)
includes counts, source digests, and the finite 9,633 × 384 local vector matrix.

This is the single-hop control by dataset category. Multiple contexts may each
support an answer. All-supports coverage measures every marked context and should
not be interpreted as requiring all of them jointly for a correct answer.

The existing source card record states CC BY-SA 3.0. The packaging states no
licence. Redistribution of derived corpus text requires the applicable share-alike
and packaging rights. The loader stays
outside `corpora.SOURCES`. Credit: Natural Questions, Kwiatkowski et al., TACL
2019, https://ai.google.com/research/NaturalQuestions.

## MultiHop-RAG article boundary

RI #1's existing null-only loader is unchanged. The separate
`corpora/multihop_articles.py` surface handles `inference_query`,
`comparison_query`, and `temporal_query` with article-level support IDs.
Its synthetic tests validate title/author mappings and reject ambiguous evidence.
Article inputs are checked against the authorization record and pinned source
bytes before use.

The query file retains the shared revision
`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82` and SHA-256
`03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff`.
ODC-BY 1.0 covers the database; it does not grant rights in individual article
contents. Distributing article text needs the publishers' permission.

An article ruling file must contain `status: approved`,
`scope: multihop-rag-articles`, a `reference` to the permission, an immutable revision
`url`, the article file `sha256`, and a `redistribution` description. The loader
checks this before fetching either file and verifies bytes before parsing.
The reference and redistribution conditions provide the provenance
needed to interpret and distribute article-derived outputs.

Source attribution: MultiHop-RAG, Tang and Yang, COLM 2024,
https://huggingface.co/datasets/yixuantt/MultiHopRAG, ODC Attribution License.
