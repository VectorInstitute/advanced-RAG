# Attribution for the generated ContractNLI question set

`contract_nli_generated.jsonl` holds questions generated from ContractNLI test
documents. Each answerable question carries the verbatim document text it was
written from, with character offsets into the document.

ContractNLI: A Dataset for Document-level Natural Language Inference for
Contracts, Koreeda and Manning, Findings of EMNLP 2021, (c) Hitachi America,
Ltd., licensed CC BY 4.0. https://stanfordnlp.github.io/contract-nli/

Changes: passages were selected from the documents, quoted, and paired with
model-generated questions and answers for retrieval evaluation. The questions,
reference answers, and null questions are model output, not part of
ContractNLI, and ContractNLI's own annotations are not reproduced here.

`contract_nli_validation.json` records the models that generated and checked
the set, the acceptance rate at each filter, the call and token counts, and the
rank agreement between generated and real labels. `validate_synthetic_qa.py`
in the parent directory produces both files.
