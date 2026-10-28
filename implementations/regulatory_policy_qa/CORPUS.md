# RI #1 corpus licensing and provenance

What each corpus behind RI #1 is licensed for, which version the code pins, and
the attribution that has to ship with any derived slice. Read this before adding
a corpus, redistributing a slice, or publishing notebook output that quotes
corpus text.

Two engineering conventions apply to every corpus here: pin to a stable reproducible version
rather than to whatever is live, and ship the attribution the licence asks for
with any derived slice.

This file records what the terms say, quoted from a primary source. Nothing here
is legal advice, and nothing here is a verdict on any use.

One distinction is worth keeping straight when reading the entries below,
because it changes which terms are engaged. Teaching a retrieval technique on a
corpus, and redistributing that corpus or a slice of it, are different acts. A
dataset licence reaches the data and works derived from the data. It does not
reach the method, so a technique demonstrated on one corpus carries no licence
condition into a participant's own corpus.

## Licence summary

Terms as recorded, with the engineering status of each corpus. The commercial-use
column states what the terms say, and carries no judgment about any particular
use.

| Corpus | Licence as stated | Commercial use | Status in the code |
| --- | --- | --- | --- |
| CUAD | CC BY 4.0 | Permitted | Loader and pinned source, in use |
| ContractNLI | CC BY 4.0 | Permitted | Loader and pinned source, in use |
| MultiHop-RAG | ODC-BY 1.0 over the database only | Permitted for the database | Loader for the null queries; the article corpus has no loader, because ODC-BY does not cover the publishers' copyright in the article text |
| PolicyQA | Non-commercial, via OPP-115 | Not permitted by those terms | No loader |

Loaders live in `aieng-advanced-rag/aieng/advanced_rag/corpora/`. Each pinned
source carries its licence terms in code, and every fetch verifies a SHA-256
digest, so an upstream change fails loudly instead of altering a measurement.

## CUAD

The Contract Understanding Atticus Dataset, SQuAD-style extractive question
answering over 510 commercial contracts, of which the pinned test split holds
102.

Licence: Creative Commons Attribution 4.0 International
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

The Atticus Project states on the
[dataset card](https://huggingface.co/datasets/theatticusproject/cuad-qa):
"CUAD is licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) license
and free to the public for commercial and non-commercial use." The
[Zenodo record](https://zenodo.org/records/4595826) for CUAD v1
(DOI [10.5281/zenodo.4595826](https://doi.org/10.5281/zenodo.4595826)) displays
the same licence.

What it permits: redistribution and modification, commercial use included, on
condition of attribution and a statement of changes.

### Pinned version

`data.zip` at commit
[`7aad6491`](https://github.com/The-Atticus-Project/cuad/commit/7aad649183846688381a147eeb3601aff590fe4a),
the only commit that has ever written that file.

```text
url         https://github.com/The-Atticus-Project/cuad/raw/7aad649183846688381a147eeb3601aff590fe4a/data.zip
sha256      f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a
member      test.json
sha256      007b6a40b0c65247f881627375c3d2e9b6aeeb5dfa89957494feed0765a1a073
```

Verified on download: 102 contracts, 4,182 questions, 2,938 of them
`is_impossible`, internal version string `aok_v1.0`.

Do not depend on the Hugging Face mirror for the data. `theatticusproject/cuad-qa`
ships a loading script, not the corpus, and that script's `_URL` points at
`https://github.com/TheAtticusProject/cuad/raw/main/data.zip`. Pinning the
Hugging Face revision `6d9a118fb629814c4384a1e63cfc84f5c01894b3` pins the script
and leaves the data on a moving reference. The commit-pinned zip is the stable
source.

The package loader, `aieng/advanced_rag/corpora/cuad.py`, reads the pinned zip above
and checks both digests.

### Attribution

```text
Contract Understanding Atticus Dataset (CUAD) v1, The Atticus Project,
licensed CC BY 4.0. https://doi.org/10.5281/zenodo.4595826
```

Cite the paper in written material: Hendrycks, Burns, Chen, and Ball, "CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021,
[arXiv:2103.06268](https://arxiv.org/abs/2103.06268).

### Two gaps worth recording

The GitHub repository carries no `LICENSE` file. The GitHub API reports
`"license": null` for `The-Atticus-Project/cuad`, and the repository README
states no terms. The CC BY 4.0 grant rests on the dataset card and the Zenodo
record, both published by the rightsholder, rather than on a licence file beside
the data.

The contract text originates in filings to EDGAR, the U.S. Securities and
Exchange Commission's electronic filing system, and the contracts within those
filings were authored by the filing companies. The CC BY 4.0 grant over the
compiled dataset is The Atticus Project's own assertion over the compilation and
its annotations. Nothing in the sources above contradicts it, and the
dataset has been redistributed on those terms since 2021.

## PolicyQA

Reading-comprehension question answering over
website privacy policies, 25,017 examples derived from 714 human-written
questions. The pinned test split holds 20 policies and 4,152 questions, every one
of them answerable, so PolicyQA offers free-form questions rather than null
cases. It is often labelled CC-BY-NC-SA. The terms differ from
that label, and are non-commercial either way. The chain below records what they
say.

### The chain of rights

The [PolicyQA repository](https://github.com/wasiahmad/PolicyQA) carries an MIT
`LICENSE` file, copyright Wasi Ahmad, 2020, and the data is committed directly
under `data/` as `train.json`, `dev.json`, and `test.json`. That MIT grant covers
the BERT fine-tuning code the repository was published to share. It cannot
relicense the data, because the repository owner is not the rightsholder of what
the data is derived from.

The [paper](https://aclanthology.org/2020.findings-emnlp.66/) states PolicyQA was
"curated from an existing corpus of 115 website privacy policies". That corpus is
OPP-115, and its terms are set by Carnegie Mellon University's Usable Privacy
Policy Project. From
[usableprivacy.org/data](https://www.usableprivacy.org/data), verbatim:

> The dataset is made available for research, teaching, and scholarship purposes
> only, with further parameters in the spirit of a Creative Commons
> Attribution-NonCommercial License.

and, on the same page:

> A commercial license for the annotation files ("annotations" and
> "consolidation" sub-folders in the zip file) is available here.

Two things follow. The restriction is non-commercial, as that label says. And OPP-115 is not released *under* a Creative Commons
licence at all: it is released "in the spirit of" one, which is a description of
intent rather than a grant of rights. There is no CC-BY-NC-SA licence to comply
with, so compliance cannot be established by following CC-BY-NC-SA. The offer of
a separate commercial licence confirms that the default terms do not reach
commercial use.

### What the terms reach

Redistribution in this repository is the first question. The repository is
public, and the default terms cover research, teaching, and scholarship only.

Use in the bootcamp turns on the same sentence. "Teaching" appears in the
permitted purposes, but the sentence is a single grant limited to those purposes,
and CMU sells a commercial licence for exactly the annotation files PolicyQA's
answer spans derive from. Paid corporate training is the case that commercial
licence exists for, so reading it as covered teaching reads the grant against its
own commercial carve-out.

A separate problem survives even with a commercial licence. Every `context` field
in PolicyQA is verbatim text from a live website's privacy policy: the first
document in the test split is `neworleansonline.com`. Those policies are the
copyright of the site operators. CMU's commercial licence is offered for the
annotation files, and OPP-115's terms never purported to license the underlying
policy text.

### What would change the answer

Buying the commercial licence for the OPP-115 annotation files, which the
usableprivacy.org page routes through Prof. Norman Sadeh, and getting a written
position on the third-party policy text. RI #1 uses ContractNLI instead.

## MultiHop-RAG

Use with conditions. RI #1 uses this dataset for one thing, the Null control, and
only the null-query slice may be used. Do not redistribute `corpus.json`.

Licence: [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/), declared on
the [dataset card](https://huggingface.co/datasets/yixuantt/MultiHopRAG) and in
the [repository README](https://github.com/yixuantt/MultiHop-RAG): "MultiHop-RAG
is licensed under ODC-BY".

### What ODC-BY does not cover

ODC-BY grants rights in a database. It says twice that it does not reach the
database's contents. From the preamble:

> Databases can contain a wide variety of types of content (images, audiovisual
> material, and sounds all in the same database, for example), and so this
> license only governs the rights over the Database, and not the contents of the
> Database individually.

> Licensors may therefore wish to use this license together with another license
> for the contents.

and from section 2.4:

> The individual items of the Contents contained in this Database may be covered
> by other rights, including copyright, patent, data protection, privacy, or
> personality rights, and this License does not cover any rights (other than
> Database Rights or in contract) in individual Contents contained in the
> Database.

No second licence for the contents is offered. That matters here because of what
the contents are. Measured against the pinned revision, `corpus.json` holds 609
news articles published between 2023-09-26 and 2023-12-25, carrying full article
bodies averaging 10,340 characters, from 49 named publishers. The largest
contributors are Sporting News (101 articles), TechCrunch (97), The Verge (45),
and Polygon (44), and the set also includes The New York Times, The Guardian, the
Financial Times, Wired, Fortune, and BBC News. Each article is the copyright of
its publisher. Redistributing them is not something ODC-BY grants.

### The slice that is clean

The 301 queries whose `question_type` is `null_query` carry an empty
`evidence_list` and the fixed answer `Insufficient information.` A slice built
from them contains the question text and nothing else: no article body, no
quoted fact, no headline. That is the entire Null control RI #1 asks for.

The other 2,255 queries carry 6,084 evidence entries, each quoting a sentence
from an article. Those quotations are not in the null slice and must not be
pulled into one.

`aieng/advanced_rag/corpora/multihop_rag.py` enforces this. It exposes no loader
for `corpus.json`, and `load_null_queries` raises if a query typed as null
arrives carrying evidence, which would mean the upstream file changed shape and
the guarantee no longer holds.

### Pinned version

```text
url         https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/MultiHopRAG.json
sha256      03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff
```

Revision `71ac0d0b`, last modified 2024-01-30, is the pinned revision. Verified on download: 2,556 queries, of which 856 comparison, 816
inference, 583 temporal, and 301 null.

The loader, `aieng/advanced_rag/corpora/multihop_rag.py`, reads the null queries
only and raises if one carries evidence. The other three query types carry the
evidence entries; RI #2's `CORPUS.md` records their opt-in article loader.

### Attribution

ODC-BY specifies the wording of the notice, with the database name linking to the
database and the licence name linking to the licence:

```text
Contains information from MultiHop-RAG
(https://huggingface.co/datasets/yixuantt/MultiHopRAG) which is made available
under the ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).
```

Cite the paper in written material: Tang and Yang, "MultiHop-RAG: Benchmarking
Retrieval-Augmented Generation for Multi-Hop Queries", COLM 2024,
[arXiv:2401.15391](https://arxiv.org/abs/2401.15391).

### What the control measures

The null queries ask about news events, so against a contract corpus they are far
nulls, separable on topic alone. CUAD's 2,938 `is_impossible` questions and
ContractNLI's `NotMentioned` labels are the near nulls, where the clause category
is plausible for the document and absent. A null detector that handles the
far case and not the near one has not been shown to work, so report the two
separately.

## ContractNLI

Recommended replacement for PolicyQA. Document-level natural language inference
over 607 non-disclosure agreements, with human-annotated evidence spans.

Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
[project site](https://stanfordnlp.github.io/contract-nli/) states "Our dataset
is released under CC BY 4.0", and the distribution ships the full licence text as
`contract-nli/LICENSE`. A `contract-nli/TERMS` file in the same archive confirms
the grant:

> The Dataset is being made available to you pursuant to, and you understand and
> agree that you will have the rights to access, review, download or otherwise
> make use of the Dataset, in accordance with the terms and conditions of the
> Creative Commons Attribution 4.0 International Public License.

The rightsholder is Hitachi America, Ltd., which also names a copyright agent for
infringement notices under 17 U.S.C. 512(c)(3). That is a considered corporate
release with a takedown route for the source documents, which is a stronger
position than a licence field on a dataset card.

### Why it fits RI #1

Each document carries its full text plus a list of character offset pairs, and
each annotation references those offsets by index. Gold evidence therefore maps
onto chunks by the same overlap rule CUAD uses, with no fuzzy text matching, so
retrieval judgments stay exact.

`NotMentioned` is a labelled in-domain null. Measured on the pinned test split:
123 documents, 2,091 annotations, 903 of them `NotMentioned`, a 43.2% null rate.
Chunked at 1,200 characters with 200 of overlap that gives 1,418 chunks, 1,188
answerable queries, and 1.94 gold chunks per answerable query. The 43% rate sits
alongside CUAD's 70% and gives the null detector a second operating point on a
different document genre.

Know the limitation before adopting it. The 17 hypotheses are fixed across all
607 documents, so a query carries no signal about which document it belongs to.
That is the same constraint CUAD has, and it means ContractNLI does not recover
the one thing PolicyQA offered that CUAD does not: free-form questions where
retrieval has to select the document rather than the chunk. The section on
recovering free-form questions below covers that gap.

### Pinned version

```text
url          https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip
sha256       e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b
train.json   dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5
dev.json     310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e
test.json    460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387
```

The published download link at `stanfordnlp.github.io/contract-nli/resources/`
resolves to the same bytes; the `gh-pages` commit is pinned because the published
path is not immutable. The archive is 65 MB because it also carries the source
PDFs. The loader extracts only the requested JSON split.

The loader reads Stanford's distribution, which states CC BY 4.0. A third-party
Hugging Face mirror labels its copy NC-SA, and nothing here relies on that label:
the loader never touches the mirror. A reader who wants the mirror has to check its
terms against Stanford's.

Split sizes, verified on download: train 423 documents and 7,191 annotations,
dev 61 and 1,037, test 123 and 2,091.

### Attribution

```text
ContractNLI: A Dataset for Document-level Natural Language Inference for
Contracts, Koreeda and Manning, Findings of EMNLP 2021,
(c) Hitachi America, Ltd., licensed CC BY 4.0.
https://stanfordnlp.github.io/contract-nli/
```

## Alternatives evaluated

Named candidates for the PolicyQA slot, judged on licence first and on whether
they arrive with usable ground truth second. RI #1 needs answer spans, so a
corpus of documents with no annotations is a different kind of asset.

PrivacyQA (Ravichander et al., EMNLP 2019) is the closest domain match: 1,750
crowdsourced free-form questions over mobile app privacy policies, with
sentence-level relevance judged by experts, and questions marked unanswerable.
The annotations were created by its own authors rather than inherited from
OPP-115, so the non-commercial problem does not carry over. Its licence is
ambiguous and cannot be relied on as it stands. GitHub reports the
[repository](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) as MIT, but
the `LICENSE` file's copyright line reads "Copyright (c) 2010-2019 Google, Inc.
http://angularjs.org". That file grants rights in AngularJS. It names neither the
dataset nor its rightsholders, so there is no MIT grant in PrivacyQA to rely on,
whatever the licence detector reports. What would resolve it: the authors
replacing the `LICENSE` file with one naming the dataset and its copyright
holders, or usableprivacy.org publishing terms for the corpus, which its
[data page](https://www.usableprivacy.org/data) lists without stating a licence.
Until one of those happens, treat PrivacyQA as unlicensed.

ObliQA (Gokhan et al., [arXiv:2409.05677](https://arxiv.org/abs/2409.05677)) has
the right shape: 27,869 free-form compliance questions over 40 Abu Dhabi Global
Markets financial regulation documents, with passage-level ground truth. The
[repository](https://github.com/RegNLP/ObliQADataset) has no `LICENSE` file and
the GitHub API reports no licence, so there is no grant to redistribute it under.
The questions are also generated by `gpt-4-turbo-1106` and validated by a natural
language inference model, which is a weaker ground truth than human annotation.

Terms of Service; Didn't Read holds its ratings data in
[tosdr-declarations](https://github.com/tosdr/tosdr-declarations), whose
`LICENSE` file is AGPL-3.0 and whose README states only that "the code in this
repository is distributed under the GNU Affero General Public Licence (AGPL)
v3.0". Secondary sources describe the project's JSON as CC BY-SA 3.0; the
repository does not say so, and a copyleft software licence applied to a data
repository leaves the terms for the data unstated. That ambiguity does not need
resolving, because the data carries quoted excerpts and topic labels rather than
answer spans against a document and does not supply the ground truth RI #1
needs.

EUR-Lex permits reuse of EU legislation under Commission Decision 2011/833/EU,
for commercial purposes included, with source acknowledgement; EU-owned content
is CC BY 4.0 and the metadata is CC0
([legal notice](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html)).
The Federal Register, the Code of Federal Regulations, and everything else on
govinfo are U.S. Government works, and
[govinfo's policy page](https://www.govinfo.gov/about/policies) records that
"Copyright protection under this title is not available for any work of the
United States Government" under 17 U.S.C. 105, with the caveat that a government
publication "may contain copyrighted material which was used with permission of
the copyright owner". Both are clean sources of regulatory documents and neither
comes with question answering ground truth.

## Recovering free-form questions

ContractNLI closes the licence gap PolicyQA opened, and leaves one capability
gap: every query set now in RI #1 is templated, so retrieval is scoped to a
document and never has to select one. The path to closing it without a licence
question is to generate questions over public domain regulatory text.

EUR-Lex and the Federal Register supply the documents on the terms cited above.
`aieng.advanced_rag.evaluation.synthetic` holds the machinery for multi-hop
questions: pair sampling, a question taxonomy, an LLM verifier, and a
single-document defeat test that drops any question one document answers.

The trade-off is that ground truth becomes generated rather than human-annotated,
which is why this is a complement to CUAD and ContractNLI and not a replacement
for them.

## Attribution that ships

Any published artifact that quotes corpus text, including notebooks with rendered
output, carries the credit line for each corpus it draws on. The lines are stored
on the source records in `aieng/advanced_rag/corpora/` so code can read them.

```text
Contract Understanding Atticus Dataset (CUAD) v1, The Atticus Project,
licensed CC BY 4.0. https://doi.org/10.5281/zenodo.4595826

ContractNLI: A Dataset for Document-level Natural Language Inference for
Contracts, Koreeda and Manning, Findings of EMNLP 2021,
(c) Hitachi America, Ltd., licensed CC BY 4.0.
https://stanfordnlp.github.io/contract-nli/

Contains information from MultiHop-RAG
(https://huggingface.co/datasets/yixuantt/MultiHopRAG) which is made available
under the ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).
```

CC BY 4.0 also requires a statement of changes. Chunking, slicing, and deriving
retrieval judgments are modifications, so say so where a derived slice is
published: "chunked and sliced for retrieval evaluation".

## Related records

FinanceBench's record is `implementations/table_structured_reasoning/CORPUS.md`.
Its licence is non-commercial and its source is pinned to a commit and per-file
digests.
