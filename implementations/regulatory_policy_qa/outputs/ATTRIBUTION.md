# Attribution for the results in this directory

These files are derived from three corpora. The `traces_*.json` files carry each
corpus's question text, the chunk identifiers retrieval returned, per-passage
grades, and the grader's one-line reasons. They carry no passage text: a chunk
appears only as its identifier, and the reasons are model-written summaries
bounded at 200 characters. The `sweep_*.json` files carry aggregate rates only.

Changes made: the source documents were chunked into 1,200-character windows
with 200 of overlap, relevance judgments were derived from the datasets' own
answer offsets, and a slice of each corpus was taken for retrieval evaluation.

`responses.sqlite` is deliberately not committed. It holds whole prompts and so
holds passage text.

## CUAD

Used for the secondary in-domain check. The first 8 contracts of the test split.
The notebook's cached baseline under `baseline_cache/` uses the first 20.

```text
Contract Understanding Atticus Dataset (CUAD) v1, The Atticus Project,
licensed CC BY 4.0. https://doi.org/10.5281/zenodo.4595826
```

## ContractNLI

The headline null set. The first 40 documents of the test split.
The GraphRAG experiment under `graphrag_null/` uses the first 20 test documents;
its `measured_thresholds/` subdirectory preserves the two-arm replay at the measured thresholds separately
from the original results. Per-question records include model-written answers
and reasons derived from the source contracts. The credit line below applies to
both runs.

`answer_judgements_*.json` in `measured_thresholds/` records automated correctness,
grounding, and abstention verdicts for all 179 answerable questions per arm,
including abstentions. References were constructed from the annotation label
and gold spans, retaining Entailment/Contradiction polarity in natural language.
Correctness is judged against the reference independently of retrieved evidence;
grounding is judged against the retrieved passages without the reference. The public files
contain verdicts and short model-written reasons, with the same credit below;
gold passages, full prompts, and the judge response cache are not redistributed.

```text
ContractNLI: A Dataset for Document-level Natural Language Inference for
Contracts, Koreeda and Manning, Findings of EMNLP 2021,
(c) Hitachi America, Ltd., licensed CC BY 4.0.
https://stanfordnlp.github.io/contract-nli/
```

## MultiHop-RAG

The out-of-domain null control, 150 of the 301 `null_query` questions. The
article corpus is not used and is not redistributed; ODC-BY covers the database
and not the publishers' copyright in its contents. `corpora/multihop_rag.py`
raises if a query typed as null arrives carrying evidence, so no article
sentence can reach these files.

```text
Contains information from MultiHop-RAG
(https://huggingface.co/datasets/yixuantt/MultiHopRAG) which is made available
under the ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).
```

Full licence records, pinned versions, and digests are in
[`../CORPUS.md`](../CORPUS.md).

## Harness-and-baseline notebook artifacts

`baseline_cache/` holds shared-cache envelopes measured on the first 20 CUAD and
ContractNLI test contracts. They contain retrieval aggregates, corpus digests,
model configuration, software versions, and four single-pass execution summaries.
The summaries identify questions and record abstention, citation counts, attempts,
and usage; they contain no contract passages or generated answer text. The corpus
credit lines above apply to these derived results as well.

## Additional answering exercises

`advanced/policies.json`, `advanced/scope.json`, and `advanced/agent.json` derive from
ContractNLI; the corpus credit lines above apply to their diagnostics and model-written
answers. Shared-cache envelopes under `advanced/cache/` reproduce these public reports.

`advanced/regulatory_source.json` snapshots the official 2024 edition of 5 U.S.C. §552
from the U.S. Government Publishing Office. It records the source URL, raw HTML and
extracted-text SHA-256 digests, and federal-government attribution. The extracted federal
statutory text is public domain in the United States. `advanced/regulatory.json` contains
generated questions, explanatory references, exact supporting quotations, and model outputs
from that snapshot. `advanced/ragas.json` contains automated evaluation scores.

`cuad_null/` contains derived results from the first five public CUAD test contracts.
The CUAD credit above applies. Corpus copies, graph indexes, full prompts, and response
databases remain private and are not included here.

`full_contract_nli/cache_status.json` records coverage only, with no inferred full-split
quality score. Historical measurement provenance is preserved; see
[`advanced/PROVENANCE.md`](advanced/PROVENANCE.md).
