# Attribution

The questions, labels, and evidence spans come from ContractNLI:

> ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts, Koreeda and
> Manning, Findings of EMNLP 2021, (c) Hitachi America, Ltd., licensed
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source:
> [stanfordnlp.github.io/contract-nli](https://stanfordnlp.github.io/contract-nli/).

The reference run uses the first 24 agreements of the test split, as pinned in
`aieng.advanced_rag.corpora.contract_nli`. `results.json` holds identifiers, labels, counts, and
scores. The notebook's committed outputs quote a few hypotheses, contract sentences, and model
answers about them, which the licence permits with this attribution. Model answers and verdicts are
derived from the agreements and were produced by the models named in `results.json`.

The local support checker is MiniCheck-Flan-T5-Large
([`lytang/MiniCheck-Flan-T5-Large`](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large), MIT
licence; Tang, Laban, and Durrett, EMNLP 2024).
