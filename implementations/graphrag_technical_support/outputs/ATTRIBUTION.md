# Attribution for the results in this directory

`musique_graph_test.json` holds retrieval and answer scores for five arms over 300
MuSiQue questions. Each row names a question by its MuSiQue id and lists the ids of
the ten passages the arm ranked first, with exact match and F1. The file carries no
question text, answer text, or passage text. `musique_hipporag2_index_cost.json` and
`musique_lightrag_index_cost.json` hold model calls and tokens spent building the two
graph indexes.
`serverfault_graph_summary.json` and `serverfault_graph_sample.json` describe the
ServerFault knowledge graph by counts, entity names, relation types, and character
offsets. They carry no post text.

The MuSiQue figures derive from MuSiQue, whose repository README licenses it
CC BY 4.0, so they carry its attribution:

```text
MuSiQue: Multihop Questions via Single-hop Question Composition,
Trivedi, Balasubramanian, Khot, and Sabharwal, 2022.
https://arxiv.org/abs/2108.00573
```

The questions and passages come from the HippoRAG 2 packaging of MuSiQue
(`osunlp/HippoRAG_2`), whose dataset card states no licence.

Changes made: 300 questions were drawn by a fixed seed (150 two-hop, 75 three-hop,
75 four-hop) from the packaging's 1,000, and five retrieval arms were scored against
the supporting passages MuSiQue marks.

ServerFault posts are licensed CC BY-SA at the version each post records. The
ServerFault files hold derived counts and names and no post text. See
`../CORPUS.md`.

`maintenance/` holds the results of `02_graph_maintenance.ipynb`. The ServerFault
files hold counts, entity names, relation types, and post ids, and no post text.
`revision.json` derives from earlier revisions of 25 posts in the same dump's
`PostHistory.xml`, under each post's CC BY-SA version, and names each post by its
id, which resolves to `https://serverfault.com/questions/<id>`. `lightrag.json`
holds entity names from five MuSiQue passages and carries MuSiQue's attribution
above. Changes made: one sentence of passage `musique::00002` was edited to test
re-extraction, and the edit is described in the notebook.
