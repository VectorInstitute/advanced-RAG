# RI #4 corpus licensing and provenance

This record states what RI #4's sources say about licence and what the loaders
pin. It does not rule on whether a corpus may be used or redistributed.

## MuSiQue, as HippoRAG 2 packages it

1,000 multi-hop questions over 11,656 Wikipedia passages. Each question lists
about twenty candidate paragraphs, marks the ones that support the answer, and
carries the single-hop questions it composes. The corpus is exactly the union of
those paragraphs. The packaging has 518 two-hop, 316 three-hop, and 166 four-hop
questions, where the hop count is the number of supporting paragraphs.

### Licence

The [MuSiQue repository](https://github.com/StonyBrookNLP/musique) states in its
README that "MuSiQue is distributed under a CC BY 4.0 License", and GitHub
reports the repository licence as CC-BY-4.0.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits use with
attribution.

The Hugging Face dataset `osunlp/HippoRAG_2`, which repackages the questions and
passages, states no licence: its card holds a language tag and links to the
paper and code.

### Pinned version

Both files are read at revision
[`5ec05b38`](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/5ec05b38deecc3318bb432c69865959c56058990)
(modified 2025-04-02).

```text
musique.json          12,543,629 bytes
sha256      98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46
musique_corpus.json    6,239,261 bytes
sha256      73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb
```

`aieng/advanced_rag/corpora/musique.py` verifies both digests on each fetch,
including a cached file. Passages get identifiers from their position in the
corpus file (`musique::00000` to `musique::11655`), which is stable at this
revision. Every question's supporting paragraph matches exactly one corpus
passage by title and text, and the decomposition agrees with the marked
paragraphs, which the loader checks.

The test set is a stratified draw of 300 questions with a fixed seed: 150
two-hop, 75 three-hop, and 75 four-hop.

### Attribution

```text
MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi,
Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique
```

### What stays local

The loader fetches into the local corpus cache. Nothing derived from the
passages, including indexes built over them, is committed. The loader is not
registered in `corpora.SOURCES`, because the packaging's licence is unstated.

## ServerFault

The technical-support corpus: Stack Exchange questions with accepted answers from
serverfault.com. A document is one question with its accepted answer. The slice is
the top 10,000 questions that have an accepted answer, ordered by score and then by
ascending identifier, so a prefix of it is exactly a smaller slice. It holds 10,000
questions and 9,999 accepted answers, and 9,999 documents once a question without
a kept answer is dropped.

### Sources

Three sources have been named for this corpus, and only one is used.

- The Internet Archive `stackexchange` dump of serverfault.com is the source. It is
  frozen, checksummed, and citable.
- The Hugging Face `common-pile/stackexchange` copy pins no dump and is not used.
- The live Stack Exchange API, sorted by votes,
  is not reproducible: scores move and posts are edited or deleted, so the same
  query returns different documents each month. It also caps anonymous use at 300
  requests a day. It is not used.

Every document that the four existing ServerFault question sets name, 70 distinct
posts across the multi-hop, 10,000-slice, fan-out, and obfuscated sets, is in the
slice built from the dump. Those question sets therefore apply to this corpus
without a mapping.

### Licence

Posts are CC BY-SA at the version each post records. The slice holds 11,835 posts
under 3.0, 5,900 under 2.5, and 2,264 under 4.0. Every post keeps its author, its
link, and its licence version.
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) permits use, with
attribution, by anyone, and requires that adaptations carry the same licence.
Share-alike applies to derived text, extracted graphs, and questions that quote a
post.

### Pinned version

```text
url         https://archive.org/download/stackexchange/serverfault.com.7z
published   2024-04-06
bytes       859,606,467
md5         1b075d4c7ae7396da965874f156e0fdf
sha256      8a2fed8620c93c9d94de7fa1722b24e77f802f64cc5ca652652af3adfd946cd8
```

The MD5 is the one archive.org publishes, and the SHA-256 was computed when the pin
was written. `aieng/advanced_rag/corpora/serverfault.py` checks the size and both
digests on each fetch, including a cached file. Extracting `Posts.xml` needs the
`7z` program.

The 10,000-question slice is checked two ways after it is built. Its aggregate
statistics must equal those recorded when a slice of this size was first built from
the same dump: 10,000 questions, 9,999 answers, 10,957,078
question characters, 11,517,692 answer characters, 2,082 distinct tags, scores 9 to
1,831, and creation dates 1,217,608,671 to 1,707,778,079. A rebuild here matches all
of them. Its SHA-256 must equal `7bfb51e5...1dd36`, the digest of this module's own
serialisation. The earlier slice file was never committed, so its bytes cannot be
compared, and the statistics are the check that does not depend on serialisation.

### Attribution

```text
Stack Exchange user contributions to serverfault.com, licensed CC BY-SA at the
version recorded per post, from the Stack Exchange data dump published 2024-04-06.
https://archive.org/details/stackexchange
```

### What stays local

The dump, `Posts.xml`, and the built slice stay in the local corpus cache. Nothing
derived from the posts is committed. The loader is not registered in
`corpora.SOURCES`, because the share-alike question is open.

## Code licences

Each project's repository states its licence. A package's own metadata can omit it.

| Project | Repository | Licence |
| --- | --- | --- |
| HippoRAG 2 | `OSU-NLP-Group/HippoRAG` | MIT |
| LightRAG | `HKUDS/LightRAG` | MIT |
| atlas-rag | `HKUST-KnowComp/AutoSchemaKG` | MIT |

The `atlas-rag` package on PyPI lists no licence in its metadata, so its repository
is the source.

## Open questions

The HippoRAG 2 packaging of MuSiQue states no licence, and derivatives of CC BY-SA
ServerFault posts carry the share-alike term. Both are unresolved here.
