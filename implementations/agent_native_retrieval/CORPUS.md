# RI #5 corpus licensing and provenance

This record states what RI #5's sources say about licence, what the loaders pin, and what
this directory commits. It does not rule on whether a corpus may be used or redistributed.
RI #5 shares MuSiQue's 11,656-passage packaging and the ServerFault 10,000-question slice
with RI #4, and `../graphrag_technical_support/CORPUS.md` records both.

## MuSiQue pool, 139,416 passages

The larger corpus of the scale ladder. It holds every distinct paragraph across the six
question files of the authors' MuSiQue v1.0 release: answerable and full, each in train,
dev, and test. IRCoT (Trivedi et al., ACL 2023) indexes the same set for MuSiQue, and its
README states the same count. Paragraphs are distinct by title and text. IRCoT's rule,
which hashes the title and text joined by a space, gives the same 139,416.

The pool opens with the 11,656 packaged passages, in their own order and with their own
identifiers (`musique::00000` to `musique::11655`). The other 127,760 follow in the order
the six files list them (`musique::11656` to `musique::139415`), with the files read in
IRCoT's order: answerable dev, test, train, then full dev, test, train. A question's
supporting passages therefore carry the same identifiers at both corpus sizes. Every
packaged passage is in the release, which the builder checks.

### Licence

The [MuSiQue repository](https://github.com/StonyBrookNLP/musique) states in its README
that "MuSiQue is distributed under a CC BY 4.0 License".
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits use with attribution.
The pool comes from the authors' own release, so the unstated licence of the HippoRAG 2
packaging does not apply to the 127,760 passages it adds.

### Pinned version

The MuSiQue README at commit
[`922ac98f`](https://github.com/StonyBrookNLP/musique/tree/922ac98f19a201998dbdae6d7f2887a5258dbdeb)
links the release as Google Drive file `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`. Drive keeps no
revision a URL can name, so the SHA-256 of the archive and of each member it reads are the
pin.

```text
musique_v1.0.zip                    272,049,578 bytes
sha256  98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd

data/musique_ans_v1.0_dev.jsonl     15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b
data/musique_ans_v1.0_test.jsonl    92ee3067957b8ebce885baa71156a7c0c29f3bc72cca04510aef506911dd768f
data/musique_ans_v1.0_train.jsonl   83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a
data/musique_full_v1.0_dev.jsonl    8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a
data/musique_full_v1.0_test.jsonl   01d7b0f752bdc1fc98856d605560e021a89be903baeb9daff11adeb7fac61328
data/musique_full_v1.0_train.jsonl  b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034

built pool, musique-pool-139416.json  67,544,486 bytes
sha256  777bd36d26d28b232c0bdc5e6587c93551d1b101447babd192c2127138fa959d
```

`aieng.advanced_rag.corpora.musique.fetch_pool` downloads the archive, verifies it and the
six members, builds the pool, checks its size and that it opens with the packaged corpus,
and verifies the serialised pool's digest before caching it. A cached pool is verified on
every call. The pool holds about 64 million characters of title and text, roughly 16
million tokens at four characters a token, against about 1.5 million for the packaged
corpus.

### What stays local

The archive and the built pool stay in the local corpus cache. Nothing in this directory
holds pool passage text.

## ServerFault error-string set

`outputs/serverfault_error_strings.jsonl` draws on the 10,000-question ServerFault slice that
`../graphrag_technical_support/CORPUS.md` records: the Internet Archive dump published
2024-04-06, pinned by size, MD5, and SHA-256, and the slice pinned by its statistics and
SHA-256. Posts are CC BY-SA at the version each post records.

That record keeps post text out of the repository while the share-alike question is open, and
this set follows it. A row holds the gold document identifiers, links, and licence versions,
the key's character offsets in the gold post's title or question, and the key's SHA-256. It
holds no post text, and the checker's written reasons, which quote the key, are dropped. The
question text is built locally from a fixed template and the key read out of the slice, and
the loader fails when the key at the offsets no longer matches its digest or when the set of
documents containing it changes.

## Attribution

```text
MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi,
Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique
```

```text
Stack Exchange user contributions to serverfault.com, licensed CC BY-SA at the
version recorded per post, from the Stack Exchange data dump published 2024-04-06.
https://archive.org/details/stackexchange
```
