# Saved results

The notebooks use these small, replayable artifacts. Corpus text, model weights,
indexes, full agent traces, response caches, and operational logs remain local.
The worked examples with question and answer text ship in RI #2's reference
release instead; see [the manifest](../reference-artifacts.json) and
`synthesis_lab/reference.py`.

| Folder | Contents |
| --- | --- |
| [evaluation](evaluation/) | Six full 1,000-question core reports, their decision grid, and paired comparisons |
| [evaluation/graphrag](evaluation/graphrag/) | Full 1,000-question atlas-rag HippoRAG 2 and LightRAG predictions, paired comparisons, and combined decision grid |
| [evaluation/shared_300.json](evaluation/shared_300.json) | Every core and graph arm rescored on the shared 300 MuSiQue questions, with RI #4's figures for the same draw |
| [evaluation/corrective](evaluation/corrective/) | Dense, graded single-pass, and corrective answering on the shared 300 questions, with paired comparisons |
| [reference](reference/) | Dense reference rankings, answer scores, and recorded execution usage |
| [examples](examples/) | Smoke runs and RI #4's graph build costs |
| [validation](validation/) | Pinned corpus integrity, evidence mapping, and released graph-index coverage checks |

The correction cohort answers the shared 300 MuSiQue questions. Its grid and
controls stay separate from the full core comparison.

Scores replay against pinned annotations. Failures remain in the denominator;
cache hits are separate from live usage. See
[comparison details and cost limits](../docs/COMPARISONS.md#read-the-comparison)
and [provenance](../CORPUS.md).
