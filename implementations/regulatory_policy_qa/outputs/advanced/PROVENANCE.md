# Provenance of the answering teaching fixtures

These files hold RI #1 measurements, with their recorded measurement artifacts, made with an
earlier version of the code. They were not re-measured against the code in this repository.

Each JSON report retains its original `generated_at`, host fields, model, parameters, and
recorded usage. The model is `Qwen3_6-35B-A3B`, reasoning is disabled, and
temperature is zero. The local encoder is `BAAI/bge-small-en-v1.5`.

| Files | What was measured |
| --- | --- |
| `policies.json` | Forty balanced ContractNLI questions, matched policy variants including citation requirements, separate correctness/grounding judges |
| `scope.json` | Matched dense scoped/unscoped retrieval and replays of the historical null-arm outcomes |
| `regulatory_source.json`, `regulatory.json` | Pinned public-domain federal FOIA text and seven generated/audited free-form questions, pipeline answers, decisions, and answer judgments |
| `ragas.json` | RAGAS 0.4.1, six answerable regulatory examples, local CPU embeddings, fixed zero-temperature cached judging |
| `agent.json` | Four shared-ADK-loop executions with search/fetch traces, usage, and bounded stopping |
| `cache/` | Shared result-cache envelopes keyed by each exact report digest; replay requires a matching envelope |

The new portable input builder reconstructs the RAGAS fixture from the shipped public source
and ordered retrieved IDs. The historical RAGAS report predates its input-fingerprint field;
fresh reports include those fingerprints. Preserving the historical file also preserves the
identity of its existing cache envelope.

The source text and generated references are inspectable teaching data, not independently
adjudicated legal guidance. See [the attribution record](../ATTRIBUTION.md). ContractNLI-derived
agent responses can quote contract text and carry the corpus's CC BY 4.0 attribution.
Private corpus archives, prompt/response databases, and graph indexes are not included.

The repository's `produced_by.json` sidecars label artifacts whose historical payloads do not
carry a model in the standard location. Cache envelopes keep their original `producer_id`
(`ri1-lab-v1` identifies the cache format); the adjacent sidecar names the model that produced
the cached measurement. Source extraction, local retrieval audits, deterministic controls,
identifier restoration, cache coverage, and offline validation explicitly record a null chat
model and zero model calls. No measured payload or cache key is rewritten to add these labels.

The CUAD pilot and full-split coverage audit in sibling output folders were recorded the same
way. Their original provenance and limitations remain intact. Coverage is not
an accuracy measurement. New notebook executions only replay these fixtures and exercise
model-free methods; they do not reset the historical experiment timestamps or cost counters.
