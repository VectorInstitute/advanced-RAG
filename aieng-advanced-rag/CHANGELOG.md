# Changelog

Notable changes to `aieng-advanced-rag`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every name 0.1.1 retires stays importable from the module that defines it, carries
a deprecation note naming its replacement, and is left out of `__all__`. The
aliases go away one release later.

## [0.1.1] - Unreleased

### Added

- `batch.ProxyBatch` sends bulk calls to one model through infermesh (the new `batch` extra):
  a bounded in-flight window, a request rate that follows any `x-ratelimit-*` headers, and
  retries of transient errors only (429, connection failures, timeouts, 500, 502, 503, 504).
  It keys requests with `graphrag.proxy.request_key`, the key `govern` uses, so existing response
  caches answer it, and it stores each result as it settles, so an interrupted run resumes.
  Calls report to the usage meter, `metering.before_call` runs before live calls, and a paid
  batch whose estimate needs it stops for `confirm=True`. `ProxyBatch.completer` returns a
  `complete(prompt)` callable with a `batch` method. `ProxyBatch(key=..., payload="text")` serves
  a cache with its own keys and text payloads, and `ProxyBatch.acreate` sends one request on the
  caller's event loop.
- `graphrag.proxy.async_llm` takes `key`, a cache key function of the caller's own.
- `batch.EmptyReplyError`: a `ProxyBatch` reply with no text and no tool call is not cached under
  either payload and comes back as an outcome with this error and its usage, so a rerun asks the
  model again.
- `evaluation.answers.judge_answers` judges many answers at once, as one batch when `complete`
  has a `batch` method, and gives None for a call that failed instead of stopping. `judge_prompt`
  and `AnswerCase` go with it.
- `runs.step_checked` says whether the current code runs inside a live `load_or_run` step that
  has already passed the confirmation check.
- `agents.plugins` holds two ADK plugins. `ResponseCachePlugin` answers a model request it has
  seen from a `graphrag.proxy.ResponseCache` and stores the response to one it has not, keyed on
  the request without `config.labels`, `config.http_options`, and run-specific handles, and with
  tool-call ids renumbered, so ADK's random `adk-` ids for calls a model returns without one do not
  break a replay. A hit is reported to `metering` as a cache hit. `QuestionGuardPlugin` holds a
  question to a model-call cap that counts cached responses too, can make the last allowed call
  with tools off and a budget message so the run still answers, makes a call final once the
  request passes a size estimate, returns a tool's exception to the model as `{"error": ...}`, and
  records the run's transcript in chat-completions form.
- `agents.model.GovernedLlmClient` is the LiteLLM client under an ADK agent: it passes each request
  through `models.adapt_request`, runs the spend guard's pre-call hooks, waits on a shared
  `RateLimiter`, retries transient errors, and reports the call to `metering` with the model name.
- `agents.session_loop.run_session` answers one question over a `ToolSession` on ADK and returns
  an `AgentRun` record, with the arm's call budget, the read-before-answer guard, and the
  prompt-size guard. `AgentRun`, `parse_answer`, `MeasuredLimiter`, and the instruction and guard
  messages live in `agents.records`. RI #5's arms and the mixed-source router run on it, and the
  chat-completions tool loop added earlier in this release is gone.

- `runs.load_or_run` records `code_hash`, a digest of the producer's source file, in each new
  artifact's `produced_by` block. When the hash differs from the code on disk, a live step rebuilds
  your own result. A replay with `live` off, or of the reference, returns the result with a "code
  changed" line in its label and `StepResult.code_changed` set. An artifact without a hash replays
  as before. `Step.code` names other files to hash when the
  step's logic lives outside the file that passes `compute`.
- `proxy` reads the Vector proxy key and endpoint in one place. `resolve_api_key` tries
  `OPENAI_API_KEY`, which the platform's `onboard` exports, then the `VECTOR_PROXY_API_KEY`
  alias, in the environment first and the repository `.env` second, and treats a template
  placeholder as missing. `resolve_base_url` reads `OPENAI_BASE_URL` the same way and falls back
  to `PROXY_BASE_URL`. `agents.loop.proxy_api_key` and `AgentLoop` use them, so the loop also
  finds a key held only in `.env`.
- `corpora.download` fetches what a reference implementation reads, through the pinned loaders,
  and prints each path with its SHA-256: `python -m aieng.advanced_rag.corpora.download --core`
  for the small set every RI starts from, or an RI number for that RI's own data. It also fetches
  `BAAI/bge-small-en-v1.5`. One failed item does not stop the others, and the exit status says
  whether any failed.
- `corpora.cached_path` says where `fetch` keeps a source, and `corpora.is_cached` says whether
  that file is there and matches its pin, without downloading anything.
- `evaluation.multihop` scores multi-hop question answering: `all_gold_recall_at_k` is the share of
  questions with every supporting passage in the top `k`, `exact_match` and `token_f1` are the
  SQuAD-style answer measures taken over the gold answer and its aliases, and `paired_bootstrap`
  and `mcnemar_exact` compare two arms on the same questions.
- `graphrag.extraction` builds a knowledge graph by direct extraction, one call per question or
  answer. It enforces the schema of seven entity types and seven relation types after the call
  through alias maps, coerces an unknown entity type to `Concept`, drops a relation whose type
  maps to nothing or whose endpoint is missing, and counts each in a `SchemaReport`. Each model
  quotes the words behind a fact, and the quote is located as a `Provenance` naming the document,
  section, block, and character span. `KnowledgeGraph.validate` checks the graph against its own
  rules, and `save` and `load` round-trip it as JSON and GraphML. It runs concurrently and folds
  in document order, so the graph does not depend on completion order.
- `corpora.serverfault` builds the ServerFault technical-support corpus from the Internet
  Archive dump of 2024-04-06, pinned by size, MD5, and SHA-256. A document is a question with its
  accepted answer, and the slice is the top 10,000 by score then identifier. `fetch_slice` checks
  the 10,000-question slice against the statistics recorded for a first build of the slice and against
  its own digest. Each post keeps its author, link, and CC BY-SA version, and the module is not in
  `corpora.SOURCES` because the share-alike question is open.
- `corpora.musique` loads MuSiQue as HippoRAG 2 packages it: 1,000 questions and 11,656
  passages, pinned to a Hugging Face revision and a SHA-256 per file. Each question's
  supporting paragraphs are corpus passages with stable identifiers, `stratified_test_set`
  draws the 150, 75, and 75 two-, three-, and four-hop questions reproducibly, and the
  module is not in `corpora.SOURCES` because the packaging states no licence.
- `cache.store_directory` and `cache.fetch_directory` keep a directory, such as a built graph
  index, as one artifact. The archive is a deterministic tar.gz, so the same files always give the
  same digest. Extraction goes into a temporary sibling renamed into place, refuses a path outside
  the destination and anything that is not a file or a directory, and checks the extracted tree
  against the stored `tree_digest`. `pack_directory` and `tree_digest` are exported with them.
- `graphrag.proxy` is the path to the Vector proxy for long indexing runs: `ResponseCache` (keyed on
  the whole request, so a run resumes and a repeat reports its cold cost), `RateLimiter` (a shared
  per-minute budget), `CallLedger` (calls and tokens counted at the client), `govern` (applies all
  three to an `openai.OpenAI` client in place, which `atlas-rag` requires), and `async_llm` (the model
  function LightRAG takes). The three classes moved here from the GraphRAG null arm, which imports
  them unchanged.
- `cache.store_directory` and `cache.fetch_directory` keep a directory, such as a built graph
  index, as one artifact. The archive is a deterministic tar.gz, so the same files always give the
  same digest. Extraction goes into a temporary sibling renamed into place, refuses a path outside
  the destination and anything that is not a file or a directory, and checks the extracted tree
  against the stored `tree_digest`. `pack_directory` and `tree_digest` are exported with them.
- `corpora.musique` loads MuSiQue as HippoRAG 2 packages it: 1,000 questions and 11,656
  passages, pinned to a Hugging Face revision and a SHA-256 per file. Each question's
  supporting paragraphs are corpus passages with stable identifiers, `stratified_test_set`
  draws the 150, 75, and 75 two-, three-, and four-hop questions reproducibly, and the
  module is not in `corpora.SOURCES` because the packaging states no licence.
- `retrieval.filings` chunks an ingested filing two ways over the same serialised text
  (`plain` windows and `table_aware` chunks), maps each chunk to the zero-based pages
  it covers, and builds a `FilingSlice` whose judgments come from FinanceBench's gold
  pages. A question no chunk can answer is listed in `uncovered` instead of dropping out
  of the score. `FinanceQuery.as_query` gives the harness's retrievers their query type.
- `corpora.financebench` loads the FinanceBench open sample: 150 questions, 189
  page-level evidence entries, and the 84 filings they cite, all pinned to a commit of
  `patronus-ai/financebench` and verified by SHA-256 on every fetch. `FinanceQuery`
  carries a question's zero-based gold pages and evidence text. The module is not in
  `corpora.SOURCES`, whose test requires a commercial-use licence.
- The `docling` extra installs the Docling converter that the ingestion harness takes as
  its `convert` callable.
- `evaluation.trajectory.score_agent_run` runs an agent loop inside a Langfuse trace and
  records how the run got to its answer: tool calls, repeated calls, model calls,
  tokens, elapsed time, whether the expected tools appear in order, and a judge's
  `correct` verdict when a reference is given. `tracing.record_scores` attaches named
  scores to a trace and does nothing without credentials.
- `retrieval.representations` adds `SpladeRetriever` and `ColbertRetriever`, both
  `ChunkStoreRetriever` subclasses that rank within the query's document. They take
  an encoder, so tests run without a model, and the default encoders use fastembed
  through the new `representations` extra. `max_sim` is the late-interaction score.
  There is no MUVERA retriever: its recall loss at scale rules it out.
- `evaluation.runner.Experiment` holds a query set, its judgments, and one pipeline
  configuration. `with_(...)` swaps one part and holds the rest, `run(name)` scores
  the result with the shared metrics and reports the meter's change, and
  `compare({...})` runs several variants. A misspelt knob raises instead of being
  ignored.
- `tracing.project_traces` reads Langfuse traces into the mappings `score_retrieval`
  and `score_null_detection` take, skips a trace that lacks a query identifier,
  a `RETRIEVER` observation, or an abstention signal, and records why.
  `tracing.DROPPED` lists what the projection discards.
- `retrieval.interface.declares(retriever, capability)` and
  `capabilities_of(retriever)` report which optional capabilities a retriever
  offers, with `CAPABILITY_METHODS` recording the methods each one needs.
- `metering.using(meter)` installs a meter as the default for a block and restores
  the previous one, so two runs in one process keep their figures apart without
  either emptying the other's.
- `metering.unmetered_calls()` and `reset_unmetered_calls()` report the calls that
  were reported with no meter in the call path.
- `__all__` on `metering`, `retrieval.interface`, `retrieval.conformance`, and the
  five `agents` modules.
- `pipeline` exports the answering stages: `LoopPolicy`, `CorrectivePolicy`,
  `SingleShotPolicy`, `PromptSelector`, `FixedPromptSelector`,
  `RetrieverPromptSelector`, `QueryAnalysis`, `RewriterAnalysis`,
  `PassthroughAnalysis`, `QueryPlan`, `RetrievalStep`, `LoopDecision`,
  `LoopState`, `Answerer`, and `merge_steps`.
- `aieng.advanced_rag` exports `metering` and the meter's own surface, because
  `pipeline.generate` imports it and it pulls in nothing beyond the standard
  library.
- `cache`, a two-tier content-addressed artifact cache, with `TieredResultCache`,
  `CacheKey`, and `load_or_compute` over results and over file payloads. It
  verifies a digest on every fetch and reports every resolved hit to the meter,
  so a warm re-run stays comparable to the cold one that filled it. Exported
  from the package root, because it needs nothing beyond the standard library.
- `cache.paths.root()` resolves one cache root, `~/.cache/aieng-advanced-rag`
  under `ADVANCED_RAG_CACHE_DIR`, which `corpora` and the artifact cache both
  read. `ADVANCED_RAG_CORPUS_CACHE` still wins for the corpus cache where it is
  set.
- `python -m aieng.advanced_rag.cache <manifest>` fetches the artifacts a
  manifest names, with `global-artifacts.example.json` as the template.
- `tracing`, a Langfuse bootstrap with `init_langfuse_tracing()` and the
  `langfuse_trace(...)` context manager. Both do nothing without credentials,
  which is what makes them safe to leave in a committed notebook cell. It needs
  the new `tracing` extra, so it is a named subpackage rather than a root
  export.
- `pipeline.answering.DecompositionAnalysis`, an analysis stage that is
  multi-step by construction, for a stage that splits a question into
  sub-questions and retrieves for each.
- `pipeline.answering.step_bound(analysis)`, so the spine and the sweep's
  harvest read a stage's step bound the same way.
- `RewriterAnalysis(rewriter, max_steps=...)` retrieves one step per rewrite
  candidate. It defaults to 1, which is the behaviour it had, because grading
  pools by maximum and widening the plan would move every committed figure.
- `RewritePlan.outcome` and `Attempt.rewrite_outcome` say what a rewriter did:
  `rewritten`, `no-change`, `emptied`, or `all-rejected`.
- `tracing` extra for `langfuse>=4.5.1`.

### Changed

- `graphrag.proxy.ResponseCache` adds the `prompt_tokens` and `completion_tokens` columns, as it
  adds `model` and `response_model`, when it opens a file written without them.
- `graphrag.extraction.build_graph` sends every window's prompt as one batch when `complete` has
  a `batch` method, such as `ProxyBatch.completer`, and keeps its thread pool otherwise.
- `retrieval.contextual.generate_contexts` and `name_documents` take a `ProxyBatch` as the client
  and send one batch. LiteLLM removes `cache_control` from messages, so a Claude model loses its
  prompt cache on that path; pass a governed client for a full-document prompt on a family that
  caches.
- The synthetic question generator's `proxy_model` sends through `ProxyBatch`. Its requests, cache
  keys, ledger, and tally are unchanged, and a fully cached run no longer reads a key.
- `AgentLoop` caches responses by default, in `agents/responses.sqlite` under the cache root, so a
  repeated question makes no model call; pass `cache=False` to send every request, or a
  `ResponseCache` of your own. It counts cached responses against `max_model_calls`, so a replay
  stops where the run it repeats stopped. It takes `limiter`, `max_retries`, `on_limit`,
  `max_prompt_tokens`, `read_first`, your own ADK `plugins`, ADK tools beside plain functions, and
  `agent_options` for `LlmAgent` callbacks, and `build_plugins` gives a runner of your own the same
  plugins. Live calls are metered by the model client with the model name, so per-model totals
  cover agent runs. `AgentAnswer` gains `final_text`, `exhausted`, `nudged`, `cache_hits`,
  `seconds`, `transcript`, and `tools_run`, all with defaults.

- `retrieval.interface.Retrieval` is `Retriever`, the one interface every
  retrieval method implements. `graphrag.protocol.GraphRetriever` is an alias of
  it, and its first parameter is `question` rather than `query`.
- `retrieval.interface.RetrievalMixin` is `RetrieverBase`. It defines the primary
  method, which is a base class rather than a mixin.
- The capability protocols are adjectives: `Indexable`, `Findable`, `Fetchable`,
  `Expandable`, `Readable`, and `Greppable`. None of them is `runtime_checkable`
  any more, so `isinstance` against one raises `TypeError`. Ask `declares`
  instead.
- `read` and `read_lines` take their bounds by keyword only, because one counts
  0-based half-open characters and the other 1-based inclusive lines.
- `retrieval.conformance.run_checks` is `assert_conforms`.
- `metering.CostMeter`, `QuestionCost`, and `RunCost` are `UsageMeter`,
  `QuestionUsage`, and `RunUsage`. They count calls, tokens, and seconds, and
  report no price, currency, or model rate.
- `metering.QuestionScope` is `QuestionMeter`, `UsageMeter.scope` is
  `for_question`, and the metering keyword `scope=` is `charge_to=`. A scope
  elsewhere in the package is the corpus's own filter.
- `metering.CompletionFacts` and `agents.usage.EventFacts` are
  `ParsedCompletion` and `ParsedEvent`.
- `pipeline.answering.Move` and `Step` are `LoopDecision` and `RetrievalStep`.
- `pipeline.answering.PromptSource`, `FixedPrompt`, and `MethodPrompt` are
  `PromptSelector`, `FixedPromptSelector`, and `RetrieverPromptSelector`, and
  `template_for` is `get_template`. Both shipped selectors keep `template_for` as
  an alias of it.
- `pipeline.answering.NoAnalysis` is `PassthroughAnalysis`. Its `name` stays
  `"original"`, because that label is recorded on every attempt.
- `agents.toolset.PythonToolset` is `FunctionToolset`, matching ADK's own
  `FunctionTool`.
- `agents.tools.retrieval_tools` takes its first argument as `retriever`.
- The `docling` and `all` extras name `torchvision>=0.20`, which Docling already pulled in, so
  the repository workspace can route it to the PyTorch CPU index with `torch`.

### Removed

- `retrieval.retrievers.Retriever`, the identifier-and-score protocol, is
  `Ranker` with no alias. Keeping the old name would leave `Retriever` meaning
  both the interface and the ranking core inside one package.
- `pipeline.answering.QueryPlan.hints`, a documented field that nothing wrote and
  nothing read. `retrieval.interface.Hints` is the advisory record, passed to
  `retrieve(hints=...)`.
- `AdvancedRagSpine.template`. It reported the losing half of the
  `template`/`prompt` pair while the prompt rendered the other one.

### Fixed

- A proxy key that reports no budget skipped the 10% confirmation gate and the 50% and 80%
  warnings without a word. `spend.read_budget`, `describe`, and `SpendSession.report` assume $300,
  a team's budget for the bootcamp, or the dollars in `ADVANCED_RAG_KEY_BUDGET_USD`, run the gate
  and warnings against it, and say in each line that the budget is assumed. While the budget is assumed,
  `runs.load_or_run` and the synthetic-set replay also stop for `confirm=True` before any live step
  estimated at more than $1 (`runs.ASSUMED_CONFIRM_USD`). Cached and reference results never stop.
- `ADVANCED_RAG_SPEND_LIMIT_USD` stops calls that are already queued. The ceiling raised only after
  a call returned, so a script that sent calls from a thread pool kept sending and paying for every
  queued call after the first one passed the limit. `SpendSession` also installs a pre-call check
  (`metering.add_pre_call_hook`, run by `metering.before_call`), and `graphrag.proxy.govern`,
  `graphrag.proxy.async_llm`, and the metered clients run it before each live request, so a run
  past its ceiling raises `SpendLimitReached` without sending the request.
- `RewriterAnalysis` kept only the first rewrite candidate and discarded the
  rest, so a stage that produced several questions could not retrieve for them.
  The same truncation was in `AdvancedRagSpine._next_steps` and in the sweep's
  `harvest_traces`, where it would have made a recorded retry narrower than the
  one the spine runs.
- `replay` rebuilt a plan with no retrieval steps, so a multi-step analysis
  agreed with the spine only by never being asked. The recorded step questions
  now travel in the trace and the replay rebuilds the plan from them.
- `TemplateRewriter` returned the same empty plan whether stripping changed
  nothing or removed everything, and `LLMRewriter` whether a completion was
  unparseable or every candidate was rejected. Four outcomes, two of which had
  no way to be told apart.
- The four published-figure guards skipped unless the CUAD slice sat in an
  experiment checkout or an environment variable named it, which is not the
  common case, so the figures went unchecked while the suite reported them as
  skipped. They now fall back to the corpus cache.
- `CallUsage(10, 2).total_tokens` was 0 while the field documented a fallback to
  the sum of the other two counts, so a total assembled from records built by
  hand under-reported while `tokens_complete` still read true.
- A `MeteredCompleter` nested in another one counted the same call twice and
  booked the outer count with no tokens. Nesting now raises.
- `report_call` dropped a report silently when nothing was metering, so a run that
  never opened a question reported zero calls, which reads as free rather than as
  unmeasured.
- `Hints` was frozen with a mutable `extra`, so `hints.extra["x"] = 1` flipped
  `is_empty()` after the request was built, and `hash(Hints())` raised
  `TypeError`.
- `isinstance([], SupportsIndex)` was `True`, because a list has an `index`
  method, and the conformance kit gated a real `index(chunks)` call on that check.
- `AdvancedRagSpine` took `analysis` over `prepare` and `prompt` over `template`
  in silence, and accepted a `policy` beside a `config.thresholds` that nothing
  would then read. Each pair raises.
- `Answerer.answer` returned `Any`, which checks nothing. It returns
  `SpineResult`.

### Deprecated

- Every name under **Changed** stays importable under its old spelling until one
  release after 0.1.1. `UsageMeter.scope` and the two selectors' `template_for`
  are method aliases of the same kind.
