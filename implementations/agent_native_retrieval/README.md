# RI #5: agent-native retrieval

Reference implementation 5 (RI #5) gives an agent tools over a corpus (`search`, `grep`, and
`open`) and lets it decide what to read, with no fixed retrieval stage. It measures when that
beats a fixed index and what it costs per question. It is for participants who want to compare
an agent that navigates a corpus with the dense and graph retrievers of the earlier RIs, on the
same questions and with the model held fixed.

The work is grounded in published agentic-search results (direct corpus interaction, grep agents
against BM25 at scale, and the coding agents that dropped their vector index) and is under
measurement here. Every number in this directory is labelled with the model that produced it.

Status: the tool server, the arms `C0`, `C5`, `C7`, and `grep-iterative`, and notebooks 01, 02,
03, and 99 are built. Arms `C1` to `C4` and `C6` are planned.

## Results on the shared 300

Every row uses the same 300 MuSiQue questions of RI #2 and RI #4, the same indexes and corpus, the same 30-call budget for the RI #5 agents, and thinking off. The on-prem rows are the reference (`produced_by.label` "reference trajectories", `Qwen3_6-35B-A3B`). The Gemini rows are the same RI #5 arms and RI #4's `dense`, `dense-iterative`, and `hipporag2` rerun on `gemini-3.5-flash-lite`, the default model for participants ("paid-model rerun"), over the same indexes. Both sets of answers are judged by the shared judge `gemini-3.8-flash`. "Read" and "seen" are all-supporting recall; notebook 01 defines both. "Judged" is the share of answers the judge marks correct, given the question, the reference answer, and the passages the agent read.

### Result on the default model

On `gemini-3.5-flash-lite`, the agents and the fixed pipelines move in opposite directions. RI #4's `dense`, `dense-iterative`, and `hipporag2` lose 0.05 to 0.13 F1 against `Qwen3_6-35B-A3B`, while `C0` gains 0.062 and `C5` 0.016. On Qwen, `C0` beats `hipporag2` by 0.115 F1; on Gemini it beats it by 0.302, and judged accuracy is 0.593 against 0.273. The pipelines' retrieval does not get worse (`hipporag2`'s all-supporting recall at 10 is 0.397 against 0.390). A follow-up that handed Gemini the exact five passages Qwen answered from shows the loss is abstention: Gemini replies "unknown" to 65% to 76% of the 300 questions against Qwen's 38% to 55%, and on the questions both answer their F1 matches within 0.03. With an answering prompt that drops the "unknown" option, Gemini's `hipporag2` on Qwen's passages reaches F1 0.425 (judged 0.417), above Qwen's 0.390 with the standard prompt. RI #4's README has the per-arm detail.

`C5` and `C0` tie on four-hop questions on both models. On Gemini, `C5` reaches F1 0.472 and `C0` 0.498 on the 75 four-hop questions (paired difference -0.026, 95% bootstrap interval -0.110 to +0.052); on `Qwen3_6-35B-A3B` the difference is +0.022 (-0.054 to +0.104). Over all 300 questions on Gemini they are level (+0.000, interval -0.041 to +0.040). On this model and set, the agent loop with dense search alone is as accurate as the grep-and-open loop, at 6.5 calls and 27,900 tokens per question against 10.5 calls and 31,800. These are single runs with no repeats. The result is scoped to one question set and the two answering models. For `C5` to win, `C0`'s top-5 dense search would have to miss supporting passages that grep finds. The scale ladder below tests that with twelve times the distractors.

| Arm | Source | Model | F1 | EM | Judged | Calls per question | Tokens per question | Exhausted |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | RI #4 | `Qwen3_6-35B-A3B` | 0.250 | 0.193 | 0.243 | 1.00 | 789 | - |
| `dense` | RI #4 | `gemini-3.5-flash-lite` | 0.170 | 0.140 | 0.180 | 0.98 | 748 | - |
| `dense-iterative` | RI #4 | `Qwen3_6-35B-A3B` | 0.297 | 0.220 | 0.283 | 4.53 | 2,149 | - |
| `dense-iterative` | RI #4 | `gemini-3.5-flash-lite` | 0.245 | 0.190 | 0.243 | 4.54 | 2,083 | - |
| `hipporag2` | RI #4 | `Qwen3_6-35B-A3B` | 0.390 | 0.297 | 0.377 | 1.97 | 2,421 | - |
| `hipporag2` | RI #4 | `gemini-3.5-flash-lite` | 0.265 | 0.207 | 0.273 | 1.91 | 2,227 | - |
| `C0` | RI #5 | `Qwen3_6-35B-A3B` | 0.505 | 0.393 | 0.573 | 8.67 | 77,956 | 12% |
| `C0` | RI #5 | `gemini-3.5-flash-lite` | 0.567 | 0.443 | 0.593 | 6.51 | 27,919 | 1% |
| `C5` | RI #5 | `Qwen3_6-35B-A3B` | 0.551 | 0.450 | 0.600 | 12.80 | 47,619 | 12% |
| `C5` | RI #5 | `gemini-3.5-flash-lite` | 0.567 | 0.453 | 0.573 | 10.50 | 31,779 | 1% |
| `grep-iterative` | RI #5 | `Qwen3_6-35B-A3B` | 0.240 | 0.183 | 0.247 | 4.66 | 2,172 | 0% |
| `grep-iterative` | RI #5 | `gemini-3.5-flash-lite` | 0.230 | 0.197 | 0.233 | 4.72 | 2,029 | 0% |

On the four-hop questions, judged accuracy is `C0` 0.493 and `C5` 0.520 on Qwen, and 0.507 for both on Gemini. The judge is `gemini-3.8-flash` for both models. Gemini answers use fewer calls and fewer tokens per question (`C0` 36% and `C5` 67% of the on-prem count), and Gemini's implicit prompt caching served 3.7 million of `C0`'s 8.3 million prompt tokens and 2.7 million of `C5`'s 9.5 million; the token counts above include cached tokens. The RI #5 rerun cost about $5 on the proxy, answering and judging both models, the RI #4 rerun about $1.30, and judging the on-prem RI #4 answers about $0.90.

`grep-iterative` loses to `C0` and `C5` on both models and falls to F1 0.027 on four-hop questions on Gemini, so its negative holds for this span rule on both sets of answers. RI #4's rows come from `graphrag_technical_support/outputs/musique_graph_test_gemini.json` and, for the on-prem judged column, `musique_graph_test.json`; `lightrag` and `hybrid` were not rerun.

### F1 by hop count on the shared 300

| Arm | Model | Two-hop F1 (150) | Three-hop F1 (75) | Four-hop F1 (75) | All 300 |
| --- | --- | ---: | ---: | ---: | ---: |
| `C0` | `Qwen3_6-35B-A3B` | 0.594 | 0.400 | 0.434 | 0.505 |
| `C5` | `Qwen3_6-35B-A3B` | 0.625 | 0.496 | 0.457 | 0.551 |
| `C0` | `gemini-3.5-flash-lite` | 0.636 | 0.496 | 0.498 | 0.567 |
| `C5` | `gemini-3.5-flash-lite` | 0.662 | 0.472 | 0.472 | 0.567 |

`C5` minus `C0` on the four-hop questions is +0.022 F1 on `Qwen3_6-35B-A3B` (95% bootstrap interval
-0.054 to +0.104) and -0.026 on `gemini-3.5-flash-lite` (-0.110 to +0.052). The rows come from
`outputs/musique_agent_arms.json` and `outputs/musique_agent_arms_gemini.json`; notebook 01
prints the all-300 and four-hop columns.

### Scale ladder on the default model

On `gemini-3.5-flash-lite`, `C7` is the least accurate arm at 139,416 passages (F1 0.483) and ties
`C5` at 11,656 (0.555), at three to four times the tokens of `C0` or `C5`. It sees every supporting
passage in a preview for 72% of the questions at 11,656 passages and opens all of them for 14%. A
follow-up below tests whether the previews are the cause.

The question the ladder answers: does `C5`'s lead over `C0` return when dense top-5 search faces
twelve times the distractors? On the four-hop questions, yes. On all 75 four-hop questions at
139,416 passages, a separate run, `C5` leads by +0.115 F1 (95% interval +0.030 to +0.205) and
judged accuracy is 0.573 against 0.440, where at 11,656 passages the two tie (-0.026, interval
-0.110 to +0.052). On the 120 ladder questions over all hop counts the lead stays inside the noise:
-0.009 F1 at 11,656 and +0.015 at 139,416 (-0.059 to +0.090). From the smaller corpus to the
larger, `C0` loses 0.030 F1 and `C5` 0.006, and on the 40 ladder four-hop questions the
difference moves from +0.000 to +0.095 (-0.020 to +0.217). At 139,416 `C0` still reads every
supporting passage for 42% of questions, against 25% for `C5`. That fits dense search ranking a
four-hop question's later passages below its top 5, which these runs do not test directly.

| Arm | Passages | F1 on Qwen | F1 on Gemini | EM | Judged | Four-hop F1 | Calls per question | Tokens per question | Exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C0` | 11,656 | 0.517 | 0.564 | 0.475 | 0.583 | 0.541 | 6.47 | 22,551 | 0% |
| `C0` | 139,416 | 0.469 | 0.534 | 0.442 | 0.542 | 0.524 | 7.37 | 31,355 | 1% |
| `C5` | 11,656 | 0.553 | 0.555 | 0.467 | 0.583 | 0.542 | 11.15 | 34,964 | 1% |
| `C5` | 139,416 | 0.556 | 0.549 | 0.467 | 0.558 | 0.619 | 12.62 | 41,496 | 4% |
| `C7` | 11,656 | 0.559 | 0.555 | 0.467 | 0.567 | 0.529 | 10.13 | 87,552 | 1% |
| `C7` | 139,416 | 0.513 | 0.483 | 0.400 | 0.475 | 0.430 | 11.57 | 129,697 | 8% |

All columns after "F1 on Qwen" are on Gemini, judged by `gemini-3.8-flash`. The ladder is the
first 40 questions of each hop count in the shared 300, so its four-hop F1 is over 40 questions.
These are single runs with no repeats on one answering model and one question set.
`outputs/musique_scale_ladder_gemini.json` and `outputs/musique_agent_arms_4hop_139416_gemini.json`
hold the runs, and notebook 03 prints the table. The `C0` and `C5` runs at 11,656 passages are the
shared-300 runs above, read back from the per-question cache; the ladder and the four-hop run cost
about $6.

### Follow-ups on the default model

Two single runs on `gemini-3.5-flash-lite`, judged by `gemini-3.8-flash`, cost about $1.10 together.

`C7-top5-full` is `C7` with each search returning five passages at 512-token previews (whole MuSiQue passages, so a preview counts as read) instead of twenty at 40 tokens. On the 40 four-hop ladder questions at 11,656 passages it reaches F1 0.584 and judged 0.625, against `C7`'s 0.529 on the same questions (paired F1 difference +0.055, 95% interval -0.053 to +0.168). It reads every supporting passage for 28% of questions against `C7`'s 2.5%, at 80,200 tokens per question against 121,800. The direction says the 40-token previews cost `C7` accuracy, because the agent answered from snippets instead of opening the passages they pointed to, and 40 questions cannot separate it from noise. It is level with `C0` on these 40 (F1 0.541, difference +0.043, interval -0.034 to +0.136). For a `C7`-style arm to win here, grep inside the searched set would have to find hops that `C0`'s rewording misses; the cheapest test is `C7-top5-full` at 139,416 passages on the same 40, where `C0` loses more.

`C0-top2` is `C0` with search cut to the top 2. On the 75 four-hop questions at 139,416 passages it does not lose: F1 0.472 and judged 0.467, against `C0`'s 0.428 and 0.440 at top 5 (paired difference +0.044, interval -0.034 to +0.124), at 10.4 calls per question against `C0`'s 8.9 and fewer tokens (25,800 against 41,000). The agent searches more often when each search returns less, and `C5`'s lead over it is +0.070 (interval -0.009 to +0.155), against +0.115 over `C0` at top 5. Starving dense search did not widen `C5`'s lead on this model. For `C5` to win by more, the hops would have to be unreachable by any dense query, not only ranked lower; the cheapest test is the paraphrase control (`--set paraphrase`) on Gemini, where wording no longer matches the passages.

`outputs/musique_agent_arms_c7_top5_full_4hop40_gemini.json` and `outputs/musique_agent_arms_c0_top2_4hop_139416_gemini.json` hold the runs, made with `--arms C7-top5-full --set 4hop --limit 40` and `--arms C0-top2 --set 4hop --corpus 139416`, plus `--family google --judge` and a paid response cache.

### Reference table on the on-prem model

With `Qwen3_6-35B-A3B` for every row (thinking off, 30-call budget for the RI #5 agents):

| Arm | Source | F1 | EM | Read | Seen | Calls per question | Tokens per question |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | RI #4 | 0.250 | 0.193 | 0.163 | 0.237 | 1.00 | 789 |
| `dense-iterative` | RI #4 | 0.297 | 0.220 | 0.197 | 0.233 | 4.53 | 2,149 |
| `hipporag2` | RI #4 | 0.390 | 0.297 | 0.320 | 0.390 | 1.97 | 2,421 |
| `C0` | RI #5 | 0.505 | 0.393 | 0.537 | 0.537 | 8.67 | 77,956 |
| `C5` | RI #5 | 0.551 | 0.450 | 0.223 | 0.547 | 12.80 | 47,619 |
| `grep-iterative` | RI #5 | 0.240 | 0.183 | 0.157 | 0.157 | 4.66 | 2,172 |

On the 75 four-hop questions, where `dense` has every supporting passage in its top 10 for none,
`C5` reaches F1 0.457 against `C0`'s 0.434 (paired difference +0.022, 95% bootstrap interval
-0.054 to +0.104) and sees every supporting passage for 29% of them. Over all 300, `C5` leads by
+0.045 (+0.006 to +0.086). `C0`, the agent loop with dense search only, beats `hipporag2` (F1
0.204 on these questions) by 0.230, so most of the gain on four-hop questions comes from the loop,
not from grep. `C0` reads every supporting passage for 54% of the 300 questions, more than any
other arm; the agent loop describes `search` to the model from its whole docstring, including
what each result holds. A second ADK run of both arms on these 75 questions, with an empty
response cache, gave `C0` 0.375 and `C5` 0.461 (difference +0.086, interval +0.009 to +0.169), so
`C0`'s four-hop F1 moves by about 0.06 between identical runs, and a difference under about 0.1
on these 75 questions is within one run's noise. `grep-iterative` loses to `dense-iterative`: in a scripted loop, a grep for each step's
rarest span finds the passage less often than dense retrieval does. These are single runs with no
repeats on one model and one corpus. `outputs/musique_agent_arms.json` holds the per-question
rows, and `outputs/musique_agent_arms_gemini.json` holds the Gemini rows.

The paraphrase control did not hold. On 50 rewrites with no literal span of their supporting
passages, `C5` still beats `C0` (F1 0.514 against 0.304), because the model names the entity from
its own knowledge and greps the name: in 29 of the 50, `C5`'s first grep that matched used a
pattern the question does not contain, such as "The Book Thief" for "the novel about a girl who
steals books". On these Wikipedia-derived questions, part of `C5`'s edge is the model knowing the
entities. On 120 questions at 139,416 passages, `C5` goes from F1 0.553 to 0.556 and `C0` from 0.517 to 0.469, and `C5`'s budget exhaustion goes from 12% to 23%. So on `Qwen3_6-35B-A3B` too, `C5` leads at 139,416 passages, by +0.087 F1 over the 120 (95% interval +0.018 to +0.155) and +0.142 on their 40 four-hop questions (+0.020 to +0.275), where at 11,656 the two are within noise (+0.036, -0.033 to +0.105). On the 50 near-miss null questions, `C5`
abstains on 14% and `C0` on 6%. On the 30 ServerFault error strings, `C5` opens the post that holds
the string for all 30 at 4,192 tokens per question, and `C0` for 19 at 17,918. Notebooks 02
and 03 carry these results, the ServerFault error strings, and the near-miss nulls. The paraphrase
control, the nulls, and the ServerFault runs are on the on-prem model only; the ladder's Gemini
rerun is above.

## Arms

Each arm is a configuration of one tool server and one agent loop, so switching arms changes
configuration only. The agent loop allows 30 model calls per question; a run that reaches the cap
answers from what it has read and is counted as exhausted.

| Arm | Tools | Built |
| --- | --- | --- |
| `C0` | `search` with k fixed at 5 and 512-token previews | Yes |
| `C1` | `C0` plus `get_document` | No |
| `C2` | `search` with agent-set k and preview length, plus `get_document` | No |
| `C3` | `C2` plus paging, the one control with a documented failure | No |
| `C4` | `C2` plus `expand` and `open` | No |
| `C5` | `grep` and `open` only | Yes |
| `C6` | `C4` plus `grep`, and `read_section` on FinanceBench | No |
| `C7` | `search` at k 20, then `grep` inside what search returned, then `open` | Yes |
| `grep-iterative` | RI #4's `dense-iterative` pipeline with `grep` in place of dense retrieval; the model decomposes the question and names each bridge, and does not choose tools | Yes |

The four guards are switches on every arm: a 4,000-token cap on tool output with a hint naming
where to continue, a "narrow the query" reply when a grep matches more than 200 documents, an
"already read" notice when the agent opens the same lines twice, and read-before-answer, which
asks the agent once to open its sources when it answers without opening anything.

Each agent arm runs as a Google Agent Development Kit (ADK) agent through
`aieng.advanced_rag.agents.session_loop.run_session`, with thinking off. The arm sets the tools,
the 30-call budget with a forced final answer, and the guards, including a 60,000-token prompt
guard that makes the next call final. A run stops when the model answers without a tool call, and
its record says whether it reached the budget (`exhausted`) and whether the read-before-answer
guard fired (`nudged`). The record holds the answer, calls, tokens, seconds, the tool trace, the
documents read and seen, and the transcript in chat-completions form. ADK describes each tool to
the model from the tool function's whole docstring, including what a result holds, and LiteLLM
repairs tool-call arguments written as Python literals or with unquoted keys before the tool
runs. `grep-iterative` is no agent: its decomposition, step, and answer calls go straight to the
chat-completions endpoint.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| `01_tools_and_harness.ipynb` | The tools on one question by hand, the arms as configuration, one reference trajectory step by step, the arm table on the shared 300 MuSiQue questions beside RI #4's `dense`, `dense-iterative`, and `hipporag2` rows, and the same RI #5 arms rerun on `gemini-3.5-flash-lite` |
| `02_where_grep_must_win.ipynb` | The 4-hop existence proof, the paraphrase control where grep must lose, near-miss null questions, and exact error strings on ServerFault |
| `03_scale_and_cost.ipynb` | `C0`, `C5`, and `C7` at 11,656 and 139,416 passages: accuracy, calls, and tokens per question as the corpus grows |
| `04_describe_your_data.ipynb` | Tools generated from a short data profile (`outline`, `open_section`, `sql`, `find_identifier`, `find_symbol`) against the generic `C7` arm on ServerFault, FinanceBench, and a code repository |
| `99_starter_agent.ipynb` | The tool server pointed at a folder of your own files, with the ripgrep backend |

Each notebook reads the reference runs with `RUN_EXPENSIVE = False`. With it set to `True`, the
live cells run on the model `choose_models()` returns for your family. On the cheap tier, a live
cell runs 30 questions (10 per hop count). On a mid-tier or frontier model it runs 10. Answers
are judged by the shared judge that `choose_models()` returns.

## Data

- MuSiQue, as packaged by HippoRAG 2: 11,656 passages and the shared 300-question draw (150
  two-hop, 75 three-hop, 75 four-hop) that RI #2 and RI #4 also use.
- The MuSiQue pool of 139,416 passages, every distinct paragraph of the authors' v1.0 release,
  for the scale ladder. It opens with the 11,656 passages under the same identifiers.
- ServerFault, the 10,000-question slice RI #4 uses.

`CORPUS.md` records the licence and pinned version of each source. Committed results hold
question and passage identifiers and scores, and no question, answer, or passage text. The
MuSiQue results derive from MuSiQue, licensed CC BY 4.0:

```text
MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi,
Balasubramanian, Khot, and Sabharwal, TACL 2022.
https://github.com/StonyBrookNLP/musique
```

## Run the reference arms

The reference runs use the on-prem `Qwen3_6-35B-A3B`, the model RI #4's MuSiQue rows use, at no
more than 45 requests a minute, with every response cached. From this directory:

```bash
python run_agent_arms.py --set shared --report outputs/musique_agent_arms.json
```

`--set 4hop`, `--set ladder`, `--corpus 139416`, and `--corpus serverfault` select the other
question sets and corpora. `python grep_probe.py` rebuilds the model-free reachability probe.

A second run reads every response from the cache and makes no model call.

The Gemini table comes from the same script with the paid family, its own response cache, and the
shared judge. The indexes and corpus stay as built on-prem:

```bash
ADVANCED_RAG_SPEND_LIMIT_USD=50 python run_agent_arms.py --family google --judge --set shared \
  --arms C0 C5 grep-iterative --rpm 120 --workers 12 \
  --cache <paid_responses.sqlite> --report outputs/musique_agent_arms_gemini.json
```

where `<paid_responses.sqlite>` is a response cache file separate from the on-prem one.
The scale ladder on the paid model writes both corpus sizes to one file:

```bash
ADVANCED_RAG_SPEND_LIMIT_USD=20 python run_agent_arms.py --family google --judge --set ladder \
  --sizes 11656 139416 --arms C0 C5 C7 --rpm 100 --workers 12 \
  --cache <paid_responses.sqlite> --report outputs/musique_scale_ladder_gemini.json
```

`--judge-model gemini-3.8-flash` with `--cached-only` adds the judged column to the on-prem file
without a new answering call. A step whose pre-run estimate is over the confirmation threshold stops before
it calls a model; pass `--confirm` to run it, for example to rebuild a step whose responses are
all cached.

## Code

- `aieng.advanced_rag.agents.corpus_tools`: the tool server, the grep backends (Python regular
  expressions over documents in memory, and ripgrep over files), dense search, `ArmConfig`, and
  the trace of every tool call.
- `aieng.advanced_rag.agents.session_loop`: runs one question's tool session as a Google
  Agent Development Kit (ADK) agent, with the arm's call budget, the guards, and every model
  response cached by an ADK plugin.
- `agent_arms.py`: corpora, the `grep-iterative` pipeline, and each arm as a `load_or_run` step.
- `agent_report.py`: per-hop summaries and the table joined to RI #4's rows.
- `run_agent_arms.py`: builds the reference runs and writes `outputs/musique_agent_arms.json`.
- `grep_probe.py`: how far literal grep can reach on MuSiQue, with no model call.

## Extend the arms with ADK

Every agent arm is an ADK `LlmAgent`, so you can give it ADK tools, callbacks, and plugins.
`run_session` takes any `AgentLoop` argument: `plugins` adds your plugins after the response cache
and the call guard, and `agent_options` passes `LlmAgent` arguments such as
`before_model_callback`. A plain function appended to a session's tools becomes one more tool, and
the model reads its docstring as the tool's description. This example adds a tool to `C5` and a
plugin that counts tool calls:

```python
from agent_arms import MUSIQUE_DESCRIPTION, arm_config, musique_corpus, tool_server
from aieng.advanced_rag.agents.session_loop import run_session
from google.adk.plugins.base_plugin import BasePlugin

corpus = musique_corpus()


def passage_title(doc_id: str) -> dict:
    """Return the title of one passage, without its text.

    Parameters
    ----------
    doc_id : str
        A passage identifier, such as one ``grep`` returned.
    """
    found = corpus.get(doc_id)
    return {"doc_id": doc_id, "title": found.text.splitlines()[0] if found else None}


class ToolCounter(BasePlugin):
    """Count the tools the agent calls."""

    def __init__(self) -> None:
        super().__init__(name="tool_counter")
        self.counts: dict[str, int] = {}

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        self.counts[tool.name] = self.counts.get(tool.name, 0) + 1


session = tool_server(corpus).session(arm_config("C5"))
session.tools.append(passage_title)
counter = ToolCounter()
run = run_session(
    "<question>",
    session,
    model="gemini-3.5-flash-lite",
    corpus_description=MUSIQUE_DESCRIPTION,
    plugins=[counter],
)
print(run.answer, run.model_calls, counter.counts)
```

where `<question>` is any question about the corpus.

The run is cached and metered like the reference runs and stops at the arm's 30-call budget. A new
tool changes what the model is told, so these runs are not comparable with the committed arms
until you rerun the arm you compare against with the same tool. A tool you add does not appear in
the session's trace unless it records itself, as the built-in tools do.

## Describe your data

`04_describe_your_data.ipynb` turns a short description of a corpus into tools for the agent. A
data profile is a YAML file that names which files are documents or code, how documents split into
sections, the identifier patterns the text uses, and any tables, with a description of each
column. `aieng.advanced_rag.agents.profile` reads it and registers one tool family per part on the
same tool server as `search`, `grep`, and `open`:

| Profile part | Tools |
| --- | --- |
| `identifiers` | `find_identifier`, `list_identifiers` |
| `tables` | `sql` (read-only SQLite; the schema is in the tool description) |
| `sections` | `outline`, `open_section` |
| `code` (Python) | `find_symbol`, `open_symbol` |

The notebook writes profiles for three public corpora (the ServerFault slice, the FinanceBench
filings, and `click` at a pinned commit) and compares the generic `C7` arm with the same arm given
the profile's description, given the generated tools, and given the generated tools alone.
`profiles/` holds the three profiles, and `outputs/describe_your_data/ATTRIBUTION.md` records each
source's licence.

To describe your own folder, copy `profiles/template.yaml`, edit it, and build the server:

```python
from aieng.advanced_rag.agents import STANDARD_ARMS, build_server, load_profile

profile = load_profile("my-profile.yaml", root="path/to/folder")
server, tools = build_server(profile)
session = server.session(tools.arm(STANDARD_ARMS["C5"]))
```

The loader refuses unknown keys, so a misspelt key or an unquoted comma in a YAML `{...}` mapping
shows up as an error with the key's location.

## Prerequisites

- The bootcamp environment (see `docs/setup.md`).
- RI #1's evaluation harness and RI #4's MuSiQue results, which the arm table joins.
