# Attribution for the results in this directory

`x1_musique.json` holds retrieval and answer scores for five arms over 300 MuSiQue
questions. Each row names a question by its MuSiQue id and lists the ids of the ten passages
the arm ranked first, with exact match and F1. The file carries no question, answer, or
passage text. The figures derive from MuSiQue, whose repository README licenses it CC BY 4.0:

```text
MuSiQue: Multihop Questions via Single-hop Question Composition,
Trivedi, Balasubramanian, Khot, and Sabharwal, 2022.
https://arxiv.org/abs/2108.00573
```

The questions and passages come from the HippoRAG 2 packaging of MuSiQue
(`osunlp/HippoRAG_2`), whose dataset card states no licence. Changes made: the 300 questions
are RI #4's fixed-seed draw, and five retrieval arms were scored against the supporting
passages MuSiQue marks.

`x3_serverfault.json` and `x3_hand_check.json` hold entity names and relation types from the
RI #4 ServerFault graph, ServerFault post ids, counts, and scores. They carry no post text,
no generated question wording, no model answers, and no generated SQL. ServerFault posts are
licensed CC BY-SA at the version each post records (2.5, 3.0, or 4.0); a post id `q<id>`
resolves to `https://serverfault.com/questions/<id>`. Changes made: questions were generated
from graph patterns, and the hand check records one reader's verdict on each sampled fact.

`x2_readiness.json` holds shares and counts computed over MuSiQue (attribution above),
ServerFault (as above), and the ContractNLI test split, which is licensed CC BY 4.0:

> ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts, Koreeda
> and Manning, Findings of EMNLP 2021, (c) Hitachi America, Ltd.
