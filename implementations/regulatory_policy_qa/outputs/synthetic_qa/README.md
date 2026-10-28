# Synthetic question set validation on ContractNLI

These files record whether generated labels order RI #1's retrieval arms as
ContractNLI's human labels do. `validate_synthetic_qa.py` in the parent
directory produced them in one run: one generation seed, one
configuration, no few-shot examples.

The surprise is the paraphrase variant. Rewording questions away from their
evidence's vocabulary, which the design expected to remove a lexical bias,
made agreement worse and produced the only significant reversals. The plain
variant therefore stays the default.

## Setup

The corpus is the first 28 documents of the ContractNLI test split. Three are
held out for null questions, and both label sets are scored on the other 25,
which carry 219 answerable ContractNLI queries. `Qwen3_8-27B` generates,
`Qwen3_6-35B-A3B` checks, and `Qwen3_8-Flash-Next` judges. All three are Qwen
models, so the results JSON flags the checker as the same family as the
generator. Retrieval is scoped to each question's document, as in RI #1.

The generated set holds 139 accepted questions: 111 single-hop, 14 near-miss
nulls, and 14 held-out nulls. 65 of the single-hop questions also have a
paraphrase variant. The eight arms are BM25, dense BGE-small, two hybrid
weights, and dense with a MiniLM reranker at 1,200-character chunks; BM25 and
dense at 600-character chunks; and a shuffled control.

## Results

With plain wording, Kendall's tau over the eight arms is 0.86 on nDCG@10,
0.64 on recall@3, and 0.50 on MRR. The real labels separate 22 of the 28 arm
pairs on nDCG@10 with a paired bootstrap, and the generated labels give the
same sign on 21 of them, with no significant reversal. Generated labels score
every arm higher than real labels, by 0.14 nDCG@10 on average, so only the
ordering carries over.

The paraphrase variant agrees less: tau is 0.57 on nDCG@10, and two pairs
reverse with both intervals excluding zero. Rewording away the terms a
question shares with its evidence lifts both dense arms above both hybrids,
and 600-character dense to third place where real labels put it sixth.
ContractNLI's real queries share vocabulary with their evidence, so a lexical
penalty the real queries do not have is the likely cause.

Both variants overrate small chunks. A generated question usually has one
short gold span, while a real ContractNLI query averages about two gold
chunks, so 600-character chunks lose less under generated labels.

## What would change the result

The paraphrase variant could agree better on a corpus whose real queries do
not echo document wording, such as questions written by end users. Testing
that needs a corpus with real free-form questions, which ContractNLI lacks.
Gold sets with more than one span per question might correct the chunk-size
bias. The cheapest test is to generate from two adjacent statements per
question and rerun this script.

## Cost

The run made 1,007 requests: 391 to the generator, 415 to the checker, and
201 to the judge. It used 582,215 prompt tokens and 48,109 completion tokens
on free on-prem models, and the whole script took 43 minutes at 18 requests
per minute per model. The results JSON carries the acceptance rate at each filter.
