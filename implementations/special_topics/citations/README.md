# Special topic: score cited answers

This special topic measures citation precision and recall for the answers of a retrieval-augmented generation (RAG) system. It scores
the same citations two ways: against human-annotated evidence spans, and with a model that checks
whether each cited text supports its statement. Then it measures how often the two agree. It is
for teams that need cited answers and a number for how good the citations are, including on a
corpus with no gold evidence.

## Notebook

`01_score_cited_answers.ipynb` runs on ContractNLI, whose annotators marked the sentences that
justify each label, so the gold measure and the model-judged measure can be compared citation by
citation. It follows the repository pattern: `RUN_EXPENSIVE = False` and
`MODELS = choose_models(show=True)` in the first code cell, every model step through
`load_or_run`, and `spend.report()` last. With the flag off it reads the reference run. With it
on, it answers a 40-question stratified subset with your generation model and judges it with your
judging model, which defaults to the shared judge `gemini-3.8-flash`. `RUN_LOCAL_CHECKER = True`
adds MiniCheck on the CPU.

The scoring code is in the package, at `aieng.advanced_rag.evaluation.citations`, so it runs on
any answer text and the passages behind it. The module docstring records the prior art and why
each part is reused or custom.

## Arms

| Arm | Sources shown | Citation | Built by |
| --- | --- | --- | --- |
| `answer` | Top 5 chunks | One source list for the whole answer (the shipped `pipeline.generate`) | Generator |
| `inline` | Top 5 chunks | `[n]` on every sentence, as in ALCE (Automatic LLMs' Citation Evaluation) | Generator |
| `quotes` | Top 5 chunks | Verbatim quotes per statement, located in the source | Generator |
| `fulldoc` | The whole agreement | Verbatim quotes per statement | Generator |
| `shuffled` | Top 5 chunks | `inline`'s citations moved to another shown chunk | Control, no model |
| `rank1` | Top 5 chunks | Every `inline` statement cites the top-ranked chunk | Control, no model |
| `posthoc` | Top 5 chunks | `inline`'s statements, cited after the fact by MiniCheck's best sentence | MiniCheck |

Retrieval is the dense baseline of reference implementation 1: `bge-small-en-v1.5` over 1,200-character chunks with 200
characters of overlap, scoped to the question's agreement.

## Scorers

- Gold: a citation hits when its character range overlaps an annotated span by half of the
  shorter range. Gold precision is the share of citations that hit, and gold recall the share
  of gold spans hit.
- ALCE, judged: a statement has recall when its cited texts together fully support it. A
  citation is precise unless it fails alone while the others still support the statement. The
  large language model (LLM) judge uses the three support levels of the Text Retrieval Conference (TREC) RAG 2024 track. MiniCheck-Flan-T5-Large reads the
  statement without the question.
- Agreement: per citation, the gold hit against the judge's single-citation verdict, with
  accuracy and Cohen's kappa.

## Results of the reference run

One run, no repeats: `Qwen3_6-35B-A3B` generated and judged, MiniCheck-Flan-T5-Large judged
on a CPU, over 257 questions from the first 24 agreements of the ContractNLI test split (171
entailed, 38 contradicted, 48 not mentioned). `outputs/results.json` holds every figure.

- Quoting broke abstention. With the whole agreement and a quote prompt, the generator answered
  46 of the 48 questions the agreement does not address, and the LLM judge rated those statements
  supported by their quotes (mean judged recall 0.77). With `[n]` markers over retrieved chunks,
  it declined 34 of the 48. Citation scores do not measure whether the answer is right.
- A second quote prompt can bring abstention back. Over the same 257 questions, asking first for
  the clauses that settle the question and declining when none exist (`fulldoc-extract`) declines
  37 of the 48 unaddressed questions instead of 2. It also declines 19 of the 209 answerable ones,
  and its quotes are more precise (gold precision 0.932 against 0.893). A quote that means "no
  clause supports this" in the one-call prompt (`fulldoc-sentinel`) declines 20 of 48, and a
  second read of the quotes alone (`fulldoc-check`) declines 5. A wording match finds 28 of the 46
  answers the reference prompt gave to those questions saying in the statement that the agreement
  has no such clause, so part of the failure is the "Yes or No" format.
- The model's own citations carry information beyond the ranking. Inline citations hit gold 0.873
  of the time, against 0.468 for `rank1` and 0.093 for `shuffled`.
- The LLM judge tracked gold at arm level in every row, falling from 0.946 judged recall on
  inline citations to 0.086 when they were shuffled. Per citation it agreed with gold on 77% to
  87% of citations, with kappa 0.31 to 0.48, and most disagreements were citations it accepted
  that miss gold.
- MiniCheck scored the shipped generator's answers at 0.099 judged recall against the LLM judge's
  0.788, on the same citations, because those statements are a bare "Yes" or "No". Writing the
  claim as the question plus the statement lifts it to 0.528 (kappa against gold 0.218), and
  rewriting each statement to stand alone lifts it to 0.613 (kappa 0.174). On `inline`, `quotes`,
  and `fulldoc`, whose statements already stand alone, the question makes MiniCheck stricter:
  judged recall falls by 16 to 28 points, and kappa rises on `inline` (0.177 to 0.367) only.
- Of 292 quotes, 287 were found in their source, 257 of them character for character.

## Live check on paid models

The notebook's live path ran once on `gpt-6-luna` (answering) and `claude-haiku-5-5` (judging),
over the 40-question subset (33 labelled, 7 not mentioned), with `live_check.py`. The run used no key for the shared judge `gemini-3.8-flash`, so the judge differs from the notebook's default.
`outputs/live_check.json` holds the figures beside the on-prem ones for the same subset. The
reference figures in `outputs/results.json` are unchanged.

The run made 160 answering calls (250,471 prompt and 11,904 completion tokens) and 245 judging
calls (149,798 prompt and 17,632 completion tokens). At the proxy's rates of $0.10 and $0.50 per
million input and output tokens for both models, that prices at $0.055. The estimate before the
run was $0.053, scaled from the reference token counts, and the proxy's key usage rose by $0.06.
The key is shared, so that figure can include other callers.

The arm ordering held on the paid pair. Gold precision was 0.935
to 1.000 on the four generated arms, against 0.909 to 1.000 on-prem, and the controls scored
badly (judged recall 0.059 for `shuffled` and 0.441 for `rank1`). `fulldoc` declined 1 of the 7
unaddressed questions on both. With the same retrieved chunks, the paid generator declined 4 of 7
under `inline` and `quotes`, where the on-prem one declined 6 and 3. These are 7 questions and
one run, so a difference of one question is not a finding.

## Data and licence

ContractNLI is CC BY 4.0, (c) Hitachi America, Ltd.; see `outputs/ATTRIBUTION.md`.
`outputs/results.json` holds identifiers, labels, counts, and scores. The notebook's committed
outputs quote a few hypotheses, contract sentences, and model answers, which the licence permits
with attribution. MiniCheck's weights are MIT-licensed.

## Reference artifacts and cost

The reference run made 3,070 requests to `Qwen3_6-35B-A3B` through the proxy at 20 requests per
minute or fewer, at no charge: 1,028 to answer and 2,042 to judge, a few of them repeats served
from the response cache. MiniCheck ran on the CPU of a shared workstation with 8 threads, at about
0.6 seconds for a short premise and 1.2 seconds for a chunk-length one. The notebook prints the
estimated cost of rebuilding each step on your models: $0.12 to $0.29 per answering arm and $0.08
to $0.18 per judged arm on the google defaults, for the full set.

The abstention and MiniCheck variants made 1,985 further requests to the same model, none charged:
1,184 to answer, rewrite, or re-read, and 801 to judge the three whole-agreement variants.
The question-aware MiniCheck passes took under 45 minutes on the shared CPU.

The answers, judgements, and MiniCheck verdicts sit in the local tier of the artifact cache under
the `citations` reference implementation identifier, and every response sits in the response
cache. `reference-artifacts.json` installs all of them into a fresh workspace, and
`build_reference.py` rebuilds them.

## Rebuild the reference run

Prerequisites:

- A proxy key that reaches the on-prem models.
- About 3 GB of disk for MiniCheck's weights.

Run the stages in order from this folder; each is cached and resumable:

```bash
python build_reference.py rankings answers posthoc judge minicheck explore report --env-file <env_file>
```

where:

- `<env_file>` is a `.env` holding `VECTOR_PROXY_API_KEY`, needed only when the key is not in
  the environment.

The `explore` stage builds the abstention variants (`cites/explore.py`) and the question-aware
MiniCheck checks. `--rpm <n>` sets the requests per minute for each model.

## Rerun the live check

Prerequisites:

- A proxy key that reaches the two models you name.

Print the estimate and stop:

```bash
python live_check.py --estimate-only --env-file <env_file>
```

Run it with a ceiling in dollars:

```bash
python live_check.py --generation <answering_model> --judging <judging_model> --cap <dollars> --env-file <env_file>
```

The script refuses to start when its estimate is over the cap, and stops mid-run when the running
estimate passes it. It writes `outputs/live_check.json`. The paid responses go to their own cache
file, so a rerun costs nothing.
