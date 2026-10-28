# Special topic: synthetic question sets

This special topic builds an evaluation set from a corpus that has no labeled questions, and
measures how far the set can be trusted. It is for teams whose documents come without gold
questions: each generated question carries its gold evidence as a character span quoted from a
document, so RI #1's harness scores any chunker against it, and the set includes in-domain null
questions for abstention scoring.

## Notebook

`01_build_and_trust_a_question_set.ipynb` covers quote-first generation, the filter funnel, null
questions, and measured difficulty, then the validation on ContractNLI: generated labels against
human labels, over RI #1's eight retrieval arms. It ends with when to trust a generated set. To
build a set from your own folder, use
`../../regulatory_policy_qa/04_bring_your_own_corpus.ipynb`, which calls the same functions.

The notebook follows the repository pattern: `RUN_EXPENSIVE = False` and
`MODELS = choose_models(show=True)` in the first code cell, every model step through `load_or_run`,
and `spend.report()` last. With the flag off it reads the reference run, and with it on it repeats the
build and the validation on your models over 8 documents. The generator is your generation model and
the judge is your judging model, the shared judge `gemini-3.8-flash` for every paid family. The
checker comes from another family than the generator: `gpt-6-luna` when the generator is a Google
model, `gemini-3.5-flash-lite` for any other paid generator, and a different on-prem Qwen model for an on-prem generator. Set `CHECKER` to override it.

## Results of the reference run

One run, no repeats: one generation seed, one configuration, the first 28 ContractNLI test
documents (3 held out for nulls, 25 scored), and 219 human-labeled answerable queries.
`Qwen3_8-27B` generated, `Qwen3_6-35B-A3B` checked, and `Qwen3_8-Flash-Next` judged, all Qwen
models, so the check is within one family.

- The set holds 139 questions: 111 single-hop, 14 near-miss nulls, and 14 held-out nulls, plus a
  paraphrase variant of 65 of the single-hop questions.
- With plain wording, Kendall's tau between the human and the generated ranking of the eight arms is
  0.86 on normalized discounted cumulative gain at rank 10 (nDCG@10), and the generated labels give
  21 of the 22 pairs the human labels separate the same sign, with no significant reversal.
- With paraphrased wording, tau is 0.57 and two pairs reverse significantly, so plain wording is
  the default.
- Generated labels score every arm higher, by 0.14 nDCG@10 on average, and rate 600-character
  dense retrieval level with 1,200-character dense, where human labels put it 0.10 lower.

The build made 1,007 requests on free on-prem models: 391 to the generator, 415 to the checker,
and 201 to the judge.

## Follow-up run: two statements per question

Generated labels overrate small chunks, and the likely cause is that a generated question has one
short gold span (1.1 spans and 281 characters on average) where a human ContractNLI query has 1.9
spans and 401 characters. One follow-up run tested two corrections on the same arms and human
labels, recorded in `outputs/baseline_cache` under `explore-adjacent-statements`:

- Widened spans, with no model call: each plain gold span shorter than 401 characters, the
  human mean, widened around itself to that length. Tau stays 0.86 on nDCG@10, all 22 human-separated pairs keep the human sign, and
  600-character dense returns to sixth place, its human rank.
- Pairs, 389 on-prem calls: `Qwen3_8-27B` wrote one question from two statements of the same
  passage, giving 50 accepted questions with 2.1 spans each. Statement A alone answered 10% of
  them. Tau falls to 0.79 with no significant reversal, and the BM25 gap between 600- and
  1,200-character chunks rises from 0.06 to 0.10 nDCG@10 (human labels: 0.13).

Pairs did not raise rank agreement in this run. They would win if the lower tau came from the
smaller set rather than the pairing, or if pairs drawn from different passages matched the spread
of human gold better. The cheapest test is to score a random 50 of the plain questions, which
needs no model call.

## Prior art and why this design

The RAGAS test-set generator, DeepEval's `Synthesizer`, and LlamaIndex's
`generate_question_context_pairs` record gold as context text or a chunk id, which ties the labels
to one chunker, and none writes in-domain null questions. ARES keeps a question only when a
retriever returns its passage first, which tilts a comparison toward that retriever. This design
takes the weighted user and question categories of DataMorgana (a hosted service, so not wrapped),
the personas of RAGAS, the statement-first generation of Know Your RAG, the round-trip check of
Alberti et al., and the corpus-level null check of CRUMQs. It exports the Benchmarking Information
Retrieval (BEIR) layout of `corpus.jsonl`, `queries.jsonl`, and qrels TSV per chunking, which
`pytrec_eval`, `ranx`, and LlamaIndex's `EmbeddingQAFinetuneDataset` read. Generation is custom for
three reasons: span gold that every chunker can use, filters in which no retriever decides what is
kept, and in-domain null questions for abstention scoring. Every call goes through the proxy's
response cache, so a cached run replays offline.

## Data and license

The questions are generated from ContractNLI (Koreeda and Manning, Findings of EMNLP 2021, CC BY
4.0). The committed artifacts keep each question's verbatim gold quote, and the notebook prints a
few; `../../regulatory_policy_qa/outputs/synthetic_qa/ATTRIBUTION.md` states the changes. The
questions, reference answers, and null questions are model output, not part of ContractNLI.

## Reference artifacts and rebuilding

The reference artifacts sit in `outputs/baseline_cache`, the shipped tier of the artifact cache,
so a fresh workspace reads them with no model call and no download beyond ContractNLI itself.
Rebuild them from this folder:

```bash
$ python build_reference.py validation explore --env-file <env_file>
```

where:

- `<env_file>` is a `.env` holding `VECTOR_PROXY_API_KEY`, needed only when the key is not in
  the environment and a response is not in the shared response cache.

`validation` makes the same calls as `../../regulatory_policy_qa/validate_synthetic_qa.py`, so
with that script's response cache present it makes no new call. Both stages keep the on-prem
models at 15 requests per minute or fewer. Scoring the arms embeds the documents locally and took
about 15 minutes on a shared 32-vCPU workstation.
