# Attribution for the question sets in this directory

Small question sets for comparing retrieval with and without a fixed index. Each row
names the model that wrote it (`generator`) and the model that checked it (`checker`). Each
set has a `.summary.json` beside it with the build's funnel and the calls and tokens each
model spent. `question_sets.py` in the parent directory loads them, and the `build_*.py`
scripts there rebuild them.

## MuSiQue paraphrase control

`musique_paraphrase_control.jsonl` holds 50 MuSiQue questions (21 four-hop, 29 three-hop)
reworded by `Qwen3_8-27B` so that no rare capitalized span of their gold passages appears in
them. `Qwen3_8-Flash-Next` answered each reworded question from the gold passages alone, and
judged that answer against MuSiQue's reference, and only matches were kept. A capitalized word
that occurs in more than 1% of the 11,656 passages, such as "League" or "Italian", does not
count as a leak; each row lists the ones it keeps in `common_spans_kept`. Rows carry MuSiQue
question text, the model's rewrite, MuSiQue's answers, and passage identifiers. They hold no
passage text.

MuSiQue's repository README licenses it CC BY 4.0, so the set carries its attribution:

```text
MuSiQue: Multihop Questions via Single-hop Question Composition,
Trivedi, Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique
```

Changes made: the questions come from the 1,000 that the HippoRAG 2 packaging
(`osunlp/HippoRAG_2`) draws from MuSiQue's answerable dev split, all of which are in the
authors' own CC BY 4.0 release. A model reworded each one, as the row records.

## MuSiQue near-miss nulls

`musique_near_miss_nulls.jsonl` holds 50 questions (25 two-hop, 12 three-hop, 13 four-hop)
that `Qwen3_8-27B` wrote by changing one detail of a question from the shared 300-question
MuSiQue test. `Qwen3_8-Flash-Next` read the BM25 and `bge-small-en-v1.5` top 10 over the
11,656 passages, plus up to 10 passages a `grep` for each changed term returns, and kept a
question only when it judged that none of them answers it, fully or in part. A null is
checked against the 11,656 passages only, not against the 139,416-passage pool. Each row
lists the capitalized spans and years the edit introduced with their `grep` hit counts at
both corpus sizes. Rows carry MuSiQue question text, the model's edit, and no passage text.
They carry the MuSiQue attribution above, with the same changes noted, and each question
was edited by a model as the row records.

## ServerFault set

`serverfault_error_strings.jsonl` holds questions keyed on an exact error string or
identifier from one to three posts of the 10,000-question ServerFault slice. A row holds no
post text: it gives the gold documents, their links and licence versions, and the key's
location as character offsets with a SHA-256 digest. The question wording is a fixed
template from `question_sets.TEMPLATES`, and `question_sets.load_serverfault_questions`
fills in the key from the locally built slice and checks it against the digest. This
follows `../../graphrag_technical_support/CORPUS.md`, which keeps post text out of the
repository while the share-alike question is open.

The keys come from Stack Exchange posts, licensed CC BY-SA at the version each post
records, and any materialised question carries this attribution:

```text
Stack Exchange user contributions to serverfault.com, licensed CC BY-SA at the
version recorded per post, from the Stack Exchange data dump published 2024-04-06.
https://archive.org/details/stackexchange
```

See `../CORPUS.md` for the corpora's provenance and pins.
