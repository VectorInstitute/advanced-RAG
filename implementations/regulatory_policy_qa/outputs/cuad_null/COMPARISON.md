# CUAD test split: matched null-arm comparison

The fixed leading 5 contracts contain 205 questions: 70 answerables and 135 nulls. This is a bounded, nonrandom pilot.

Generation uses `Qwen3_6-35B-A3B` and judging uses `Qwen3_6-35B-A3B`. Both retrieval arms use contract scope, top-k 10, a 5-passage reader cap, and thresholds 0.6/0.0. Reader text is capped at 1,200 characters per passage. No answer rewriter is configured.

| Arm | Balanced decision accuracy | False answers among nulls | Correct answerables | Answerable abstentions |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.7831 | 0.1481 | 39/70 (0.5571) | 20/70 |
| lightrag-mix | 0.7442 | 0.1259 | 29/70 (0.4143) | 27/70 |

Both trivial decision controls score 0.5 balanced accuracy. Correctness includes every answerable question, including pipeline abstentions.

Decision discordances: dense alone is correct on 19 questions; LightRAG alone on 15. The exact question-level McNemar p-value is 0.6076. Questions within a contract are correlated, so this is descriptive rather than contract-level population inference.

## Reader budgets and cost

- dense: mean 5886.8 visible source characters; maximum 6000.
- lightrag-mix: mean 5626.6 visible source characters; maximum 6000.

Observed indexing lower bound: 723 completed responses and 2,359,835 tokens. Interrupted in-flight request billing is unknown. The cumulative resume ledger is included once.

Graph retrieval cost was recovered from 41 unique completed keyword responses (32,170 tokens). Cache-hit counts, timing, and possible retry billing were not persisted; this is a lower bound.

LightRAG stripped whitespace from one source ID. Only the exact stripped identifier was restored within its per-contract workspace, preserving text and ranking. The original table hash and repair audit are retained.

## Interpretation limits

- Fixed leading contracts, not a random or full-corpus sample.
- Same-model generation and judging can share errors; annotated null labels may omit valid answers.
- Equal passage caps do not guarantee equal realized text lengths; both arm budgets are reported.
- Embedding query conventions differ between adapters; this compares implemented configurations and does not isolate graph structure alone.
- Index logs contain 99 format-warning events and 12 zero-relation extraction events. Events can repeat during recovery. These diagnostics limit conclusions about graph quality.
- Canonical abstentions are incorrect on answerables and vacuously grounded. Grounding is measured against the returned evidence, separately from reference correctness.

These results describe the measured configuration and cohort. They do not establish a universal GraphRAG null-answer claim or a population ranking.

Input hashes, exact counts, reader budgets, and index diagnostics are in [`paired_comparison.json`](paired_comparison.json). Separate correctness and grounding reports are in [`answer_judging_summary.json`](answer_judging_summary.json). Corpus credit is recorded in [`../ATTRIBUTION.md`](../ATTRIBUTION.md).
