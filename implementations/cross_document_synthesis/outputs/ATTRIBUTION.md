# Output attribution and distribution boundaries

## MuSiQue

MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi,
Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique

The source records CC BY 4.0. HippoRAG 2's packaging card states no licence;
source licensing and packaging rights must be assessed separately. Reports contain numerical results,
predictions, original-passage IDs, and configuration/usage records. Compact full
reports hash question text and omit corpus passages and full agent traces.

No file here holds question text, gold answers, model answers, decomposition
plans, or agent traces; `tests/test_committed_records.py` checks that. The worked
examples the notebooks show, including one scorer-only gold hop graph with short
sub-answers, and the dense baseline's 1,000 cited answers ship in RI #2's
reference release (`../reference-artifacts.json`), which carries this
attribution in its release notes. The hop graph illustrates evaluation and must
not enter held-out answering.

## NaturalQuestions

Natural Questions: a Benchmark for Question Answering Research, Kwiatkowski
et al., TACL 2019. https://ai.google.com/research/NaturalQuestions

The recorded source card states CC BY-SA 3.0; the packaging states no licence.
Committed compact reports contain scores, model predictions,
passage IDs, hashed question text, and usage. Full passages and downloaded
annotation files remain in the local corpus cache.

## Article and graph assets

ODC-BY 1.0 covers the MultiHop-RAG database and does not establish publisher
article rights. Distribution of article content and derived outputs must follow
the applicable authorization record. Prebuilt graph assets use the shared artifact
cache. [CORPUS.md](../CORPUS.md) describes source attribution and rights boundaries.
