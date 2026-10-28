# Attribution for the results in this directory

`graph_build_stability.json` holds counts, overlaps, scores, costs, and identifiers from the
builds and evaluations that `run_stability.py` ran:

- `serverfault`: for each arm, model, and temperature, the agreement between repeated builds of
  the same 100 ServerFault posts (Jaccard of nodes, labelled edges, and unlabelled edges, and
  entity-type agreement), each build's graph shape, and how many of the model's labels sat on
  the schema.
- `musique`: for each build of the 1,825-passage MuSiQue sub-corpus, its graph shape, cost, and
  agreement with the anchor build; its retrieval and answer scores on 100 questions; paired
  comparisons; and one row per question with the ranked passage identifiers, exact match, and
  F1.
- `lightrag`: the same for the LightRAG builds over the 60-question sub-corpus.

No field carries post text, passage text, question text, answer text, or entity names. A
passage appears as its `musique::<position>` identifier and a question by its MuSiQue
identifier.

The results derive from two corpora:

- MuSiQue (Trivedi, Balasubramanian, Khot, and Sabharwal, TACL 2022), licensed CC BY 4.0 by its
  repository, read from the HippoRAG 2 packaging at the revision pinned in
  `../../graphrag_technical_support/CORPUS.md`.
- ServerFault posts from the Stack Exchange data dump, licensed CC BY-SA at the version each
  post records (2.5, 3.0, or 4.0). The sample is 100 posts drawn with seed 33 from the
  10,000-question slice described in `../../graphrag_technical_support/CORPUS.md`.
