# Chatbot Project — Iteration 24 Spec (RAG Pipeline Factory + Eval Diagnostics Sweep)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Captures iter-11 (end-to-end QA eval) through iter-23 (full-7k SOTA confirmation) — everything in the implementation that hasn't yet landed in the main SPEC. The previous focus doc only covered iter-11; this one sweeps 12 subsequent iterations.

## Overview

iter-11 introduced an end-to-end QA accuracy eval (`answer_f1`, `exact_match`, `build_qa_prompt`, `ask_llm`, `--compare-baseline`). Subsequent iterations built the production-shaped RAG surface on top of it:

- **iter-12**: pluggable pipeline factory with named presets; one CLI switch runs any preset
- **iter-13**: hybrid BM25+dense via Reciprocal Rank Fusion
- **iter-14**: `gold_paragraph_in_top_k` failure-mode instrumentation; iter-12/13 "retrieval saturation" claims were wrong — most gain was still on the retrieval side at k=4
- **iter-15**: CoT prompt scaffold (`CoTExtractPromptBuilder`) — first non-trivial prompt-only lever; becomes the comparison baseline for everything after
- **iter-16 → iter-19**: rule-heavy prompt variants (canonical-name post-processing, two-step extraction, yes/no discipline, v2 nudge) — all regressed
- **iter-20**: Anthropic extended thinking mode (`thinking_budget`) — tied with iter-15
- **iter-21**: title-strip breakthrough — removing the `[title]:` heading prefix on context paragraphs forces the model to extract canonical entity names from the body
- **iter-22**: combine title-strip with thinking — new SOTA at 0.934 (n=334 sample)
- **iter-23**: full-7k SOTA confirmation at 0.937 (n=7369 of 7405)

Headline metric throughout this sweep is `contains_gold` — substring containment of the normalized gold answer in the normalized prediction. It's the most user-relevant metric and was promoted from a derived field to the primary reported number in iter-14.

This focus doc captures all of the above as requirements so the implementation can be cleanly merged into `SPEC.md` without losing context.

## Functional Requirements

### FR-40: HotpotQA Standard Answer F1

*(Unchanged from previous focus — already implemented in iter-11; included for completeness.)*

| ID | Requirement |
|----|-------------|
| FR-40.1 | `backend.eval.metrics.answer_f1(predicted: str, gold: str) -> float` computes the SQuAD-style token-F1 that HotpotQA's official eval script uses: lowercase, strip punctuation, remove articles (`a`, `an`, `the`), tokenize on whitespace, compute precision/recall/F1 over token multisets. Returns 0.0 for empty predicted or gold. |
| FR-40.2 | `backend.eval.metrics.exact_match(predicted: str, gold: str) -> bool` returns True iff token lists are identical after normalization. Returns False for empty predicted or gold. |
| FR-40.3 | Both functions are pure: no I/O, no LLM, no Anthropic imports. |

### FR-41: QA Judge Module

*(Unchanged from previous focus — already implemented in iter-11; included for completeness.)*

| ID | Requirement |
|----|-------------|
| FR-41.1 | `backend.eval.qa_judge.build_qa_prompt(question, context_docs)` returns `[system (RAG), user (<context>...</context> + question)]` with context, or `[user (question only)]` without. The system message is `RAG_SYSTEM_PROMPT_HERE` (a local mirror of `backend.chat.chain.RAG_SYSTEM_PROMPT`, kept as a separate string constant to preserve the FR-32 isolation rule). |
| FR-41.2 | `backend.eval.qa_judge.ask_llm(client, model, messages, max_tokens=200, thinking_budget=None)` calls the LLM once with `temperature=0`, extracts the visible text from `text` blocks (skipping `thinking` blocks), joins multiple text blocks with newlines, trims whitespace. Returns the cleaned answer string. |
| FR-41.3 | When `thinking_budget` is a positive int, enables Anthropic extended thinking mode (`thinking={"type": "enabled", "budget_tokens": thinking_budget}`). `max_tokens` must be `>= thinking_budget` so the visible answer has room to render. |
| FR-41.4 | `ask_llm` is fully async and takes an `AsyncAnthropic` client. |

### FR-42: End-to-End Eval CLI

| ID | Requirement |
|----|-------------|
| FR-42.1 | `scripts/eval_qa_hotpotqa.py` is a standalone CLI. It does **not** register any HTTP route. It does **not** import from `backend.chat.*`. |
| FR-42.2 | The CLI exposes `--subset N | --full` mutually-exclusive group; `--full` is the default. |
| FR-42.3 | The CLI exposes `--k N` for retrieval depth. Default 4. Overridden automatically when `--pipeline` selects a preset that fixes `top_k`. |
| FR-42.4 | The CLI exposes `--no-cache`, `--fixture PATH`, `--paraphrase-set PATH`, `--compare-baseline`, `--llm-model NAME` flags. |
| FR-42.5 | The CLI exposes `--pipeline NAME` to run any preset from `backend.rag.pipeline.PRESETS`. When set, `--k` is overridden by `pipeline_cfg.top_k`, `embedding_model` by `pipeline_cfg.embedding_model`, and `prompt_template` by `pipeline_cfg.prompt_template`. The CLI also surfaces `pipeline_cfg.thinking_budget` to the LLM caller. |
| FR-42.6 | The CLI exposes `--list-pipelines` to print the available preset names and exit 0. |
| FR-42.7 | The CLI exposes `--dump-results PATH` to write per-question results as JSON Lines. Each line includes `qid, question, variant, mode, predicted, gold, contains_gold, answer_f1, answer_em, gold_in_top_k, gold_paragraph_titles, retrieved_titles`. Used for offline failure-mode inspection. |
| FR-42.8 | The CLI exposes `--batch-size N` (default 1) for concurrent LLM calls via `asyncio.gather`. `--batch-size 2` achieves ~2× throughput on hot-cache runs. |
| FR-42.9 | The CLI exposes `--start-from N` (default 0) and `--max-items N` (default: all remaining) for resumable runs after a crash. The CLI prints the next `--start-from` value on resume. |
| FR-42.10 | The CLI reuses the existing per-question FAISS cache from `backend.eval.cache`. Cache key is `({dataset_sha}_{embedding_tag}, qid)` — same dataset SHA, plus embedding-tag suffix so a switch from MiniLM to mpnet does not silently reuse a stale 384-dim index. |
| FR-42.11 | The CLI applies 1-second pacing between consecutive LLM calls within a question's flow; batched parallel runs pace between batches, not within. |
| FR-42.12 | The CLI exits 0 on completion. Per-question LLM errors are logged at WARNING and counted toward `errors`; they do not affect exit code. The CLI exits 1 only on setup failure (dataset missing, JSONDecodeError, embedding model load, missing API key, unknown `--pipeline`). |

### FR-43: Output Format

| ID | Requirement |
|----|-------------|
| FR-43.1 | The CLI prints `Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)` attribution once before the metric block. |
| FR-43.2 | The CLI prints a header line: `HotpotQA End-to-End QA Eval — subset={...}, k={...}, dataset_sha={...}`. |
| FR-43.3 | The CLI prints a `with_context:` block with `contains_gold`, `answer_f1`, `answer_em` averages. The CLI always prints `contains_gold` first because it is the user-relevant headline metric. |
| FR-43.4 | The CLI prints a `failure-mode breakdown (with_context):` block with `success / extraction miss / retrieval miss` buckets (and `unknown` if any gold_in_top_k was not recorded). The bucket definition: `success` = contains_gold ≥ 1.0; `extraction miss` = contains_gold < 1.0 AND gold_in_top_k is True; `retrieval miss` = contains_gold < 1.0 AND gold_in_top_k is False. |
| FR-43.5 | If `--compare-baseline` is set, the CLI prints a `without_context:` block and a `delta (retrieval helps):` block for each of the three metrics. |
| FR-43.6 | If `--paraphrase-set` is given, the CLI prints a `-- by variant --` block with per-variant `n`, `cg`, `f1`, `em` for `original`, `lexical`, `structural`, `casual`. |
| FR-43.7 | The CLI prints a footer with: total LLM calls, cache hits / builds, errors, elapsed seconds, and the `--dump-results` path (if set). |

### FR-44: Backward Compatibility

| ID | Requirement |
|----|-------------|
| FR-44.1 | `backend.eval.metrics.answer_coverage_at_k`, `paragraph_recall_at_k`, `supporting_fact_metrics`, `gold_paragraph_in_top_k` are unchanged from iter-10. |
| FR-44.2 | `scripts/eval_hotpotqa.py` (retrieval-only eval) is unchanged. |
| FR-44.3 | `scripts/eval_qa_hotpotqa.py` imports from `backend.eval.*` + `anthropic` + `backend.rag.config` + `backend.rag.embeddings`. It does NOT import from `backend.chat.*` (FR-32 isolation). |

### FR-45: RAG Pipeline Factory

| ID | Requirement |
|----|-------------|
| FR-45.1 | `backend.rag.pipeline` exposes a `PipelineConfig` dataclass (frozen) that names each stage's implementation: `embedding_backend`, `embedding_model`, `retriever`, `reranker`, `rerank_top_k`, `top_k`, `prompt_template`, `thinking_budget`, `llm_model`. |
| FR-45.2 | `build_pipeline(config, vectorstore, llm_client, corpus=None) -> RagPipeline` is the top-level factory. Users call `build_pipeline(PRESETS[name], ...)` and switch presets by changing only the name argument. |
| FR-45.3 | The factory dispatches each stage through a dedicated `build_*` function (`build_embedder`, `build_retriever`, `build_reranker`, `build_prompt_builder`, `build_llm`). Each `build_*` raises `ValueError` for unknown stage names. |
| FR-45.4 | `RagPipeline.run(question) -> str` orchestrates `retrieve → optional rerank → build prompt → ask LLM` and returns the visible answer text. |
| FR-45.5 | `PRESETS` is a `dict[str, PipelineConfig]` of every named pipeline variant. `list_presets() -> list[str]` returns sorted preset names. |
| FR-45.6 | Adding a new preset is a one-line addition to `PRESETS` — no other code changes required. |

### FR-46: Pipeline Stage Protocols

| ID | Requirement |
|----|-------------|
| FR-46.1 | `Retriever` is a duck-typed protocol with `retrieve(query: str, k: int) -> list[Document]`. |
| FR-46.2 | `Reranker` is a duck-typed protocol with `rerank(query, candidates, top_k) -> list[Document]`. |
| FR-46.3 | `PromptBuilder` is a duck-typed protocol with `build(question, context_docs) -> list[dict]` returning the messages list passed to the LLM. |
| FR-46.4 | `LLM` is a duck-typed protocol with `async ask(messages, max_tokens=200) -> str`. |
| FR-46.5 | Concrete implementations live in `backend.rag.pipeline`: `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `NoOpReranker`, `CrossEncoderReranker`, `DefaultPromptBuilder`, `ExtractSpanPromptBuilder`, `CoTExtractPromptBuilder`, `CoTExtractV2PromptBuilder`, `CoTExtractNoTitlesPromptBuilder`, `AnthropicLLM`. |

### FR-47: Retriever Implementations

| ID | Requirement |
|----|-------------|
| FR-47.1 | `DenseRetriever(vectorstore)` wraps any object with `.similarity_search(query, k)`. |
| FR-47.2 | `BM25Retriever(docs)` tokenizes the corpus at construction (lowercase, `[a-z0-9]+`), builds a `rank_bm25.BM25Okapi` index. Empty docs after tokenization are replaced with a `["_empty_"]` placeholder so `BM25Okapi` doesn't raise. |
| FR-47.3 | `HybridRetriever(dense_retriever, bm25_retriever, rrf_k=60)` fuses results via Reciprocal Rank Fusion: for each retriever, `score(d) += 1 / (rrf_k + rank + 1)`. The fused list is sorted by total score and trimmed to `k`. Uses `id(doc)` for identity-based fusion. Candidate depth per retriever is `max(k * 4, 20)` (paper-recommended). |
| FR-47.4 | `HybridRetriever` requires the raw corpus (not just FAISS) so the BM25 index can be built. The eval CLI passes `with_corpus=True` to `ev_cache.load_or_build` when the active preset is hybrid. |

### FR-48: Reranker Implementations

| ID | Requirement |
|----|-------------|
| FR-48.1 | `NoOpReranker` returns the first `top_k` candidates unchanged. |
| FR-48.2 | `CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")` uses sentence-transformers' `CrossEncoder`. Model is lazy-loaded on first call. Pairs `(query, doc.page_content)` are scored; top_k by score. |
| FR-48.3 | `RagPipeline.run` retrieves up to `rerank_top_k` candidates when a reranker is configured; otherwise retrieves `top_k` directly. |

### FR-49: Prompt Builders

| ID | Requirement |
|----|-------------|
| FR-49.1 | `DefaultPromptBuilder` delegates to `backend.eval.qa_judge.build_qa_prompt` (no system-message change beyond the RAG prompt). |
| FR-49.2 | `ExtractSpanPromptBuilder` adds an instruction asking the LLM to begin its visible answer with the extracted span (in quotation marks if a phrase), then explain. Does not paraphrase. Goal: lift `contains_gold` at the cost of conversational fluency (which dilutes `answer_f1`). |
| FR-49.3 | `CoTExtractPromptBuilder` (iter-15 SOTA scaffold) instructs the model to (1) identify entities, (2) find relevant paragraphs, (3) chain multi-hop facts in order, (4) decide the exact span. Then begin with the extracted span in quotes + brief reasoning. The four-step scaffold exists because at k≥8 retrieval is saturated; the residual failures are LLM extraction/reasoning errors, half of them multi-hop. |
| FR-49.4 | `CoTExtractV2PromptBuilder` (iter-19, regressed) tightens step 4 + closing directive to nudge toward "the most complete form of an entity name as written in the context." Kept in `PRESETS` as a documented variant even though it regressed by 0.9 pp vs `cot_extract_k10`. |
| FR-49.5 | `CoTExtractNoTitlesPromptBuilder` (iter-21 breakthrough) inherits the CoT instruction but strips the `[title]:` heading prefix on each context paragraph. The body text is what gets sent to the LLM. Hypothesis (from iter-20 thinking audit): when paragraphs are introduced with their Wikipedia article heading as a prefix, the model uses the heading as the entity label when emitting its answer. HotpotQA gold uses the full canonical name (typically in the article body opening). Stripping the heading forces the model to extract the canonical form from the body. |

### FR-50: LLM Wrapper

| ID | Requirement |
|----|-------------|
| FR-50.1 | `AnthropicLLM(client, model, thinking_budget=None)` is an async wrapper that calls `backend.eval.qa_judge.ask_llm` with the configured `thinking_budget`. |
| FR-50.2 | When `thinking_budget is None` or 0, calls `ask_llm` without thinking. When set, calls with `thinking={"type": "enabled", "budget_tokens": thinking_budget}`. |
| FR-50.3 | Thinking content is consumed by the model and discarded by the scoring path; only `text` blocks reach the caller. |

### FR-51: Named Pipeline Presets

| ID | Requirement |
|----|-------------|
| FR-51.1 | `naive_dense` — MiniLM, dense, k=4, default prompt. Baseline. |
| FR-51.2 | `large_dense` — mpnet, dense, k=4, default prompt. Larger embedding only. |
| FR-51.3 | `dense_then_ce` — mpnet + cross-encoder rerank (50→4), default prompt. |
| FR-51.4 | `extract_span_prompt` — mpnet, dense, k=4, extract_span prompt. |
| FR-51.5 | `extract_span_k8` — MiniLM, dense, k=8, extract_span prompt. |
| FR-51.6 | `extract_span_k10` — MiniLM, dense, k=10, extract_span prompt. Recommended for QA-style tasks before iter-15. |
| FR-51.7 | `cot_extract_k10` — MiniLM, dense, k=10, CoT extract scaffold. iter-15 SOTA at 0.904. |
| FR-51.8 | `cot_extract_v2_k10` — same as `cot_extract_k10` with the iter-19 v2 nudge. Kept as a documented variant even though it regressed. |
| FR-51.9 | `cot_thinking_k10` — MiniLM, dense, k=10, extract_span prompt, `thinking_budget=4096`. iter-20; tied with iter-15 on `contains_gold` (same 0.904, only 25 of 32 failures overlap). |
| FR-51.10 | `cot_extract_notitles_k10` — MiniLM, dense, k=10, CoT-no-titles prompt. iter-21; 0.925. |
| FR-51.11 | `cot_extract_notitles_thinking_k10` — MiniLM, dense, k=10, CoT-no-titles prompt, `thinking_budget=4096`. iter-22 / iter-23 SOTA at **0.937** on n=7369. |
| FR-51.12 | `hybrid_bm25_dense` — MiniLM + BM25 via RRF, k=4, default prompt. iter-13. |

### FR-52: Failure-Mode Diagnostics

| ID | Requirement |
|----|-------------|
| FR-52.1 | `backend.eval.metrics.gold_paragraph_in_top_k(retrieved_titles, gold_titles) -> bool` returns True iff at least one gold paragraph title appears in retrieved. Vacuous (True) on empty gold; False on empty retrieved. |
| FR-52.2 | The eval CLI records `gold_in_top_k` per question. With-context mode always records it; without-context mode records `None`. |
| FR-52.3 | The CLI's failure-mode breakdown reports success / extraction miss / retrieval miss / unknown counts. This split lets the operator decide whether further work should target retrieval (k, embedder, reranker) or extraction (prompt scaffold, model). |
| FR-52.4 | `contains_gold` is reported alongside `answer_f1` and `answer_em` in `with_context` and `without_context` blocks. `contains_gold` is the user-relevant headline; `answer_f1` is reported for HotpotQA-strict benchmark parity. |

### FR-53: Embedding Backends

| ID | Requirement |
|----|-------------|
| FR-53.1 | `backend.rag.embeddings.make_embeddings(backend, *, model_name="all-MiniLM-L6-v2", api_key="", base_url=...)` dispatches to either `_build_huggingface(model_name)` (returns `HuggingFaceEmbeddings`) or `_build_minimax(api_key, base_url)` (returns `MiniMaxEmbeddings`). |
| FR-53.2 | `MiniMaxEmbeddings` calls the vendor's `/embeddings` endpoint via `httpx`. Reuses `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` from the project's `.env` (the same env-var names the chat model uses — see CLAUDE.md). |
| FR-53.3 | `MiniMaxEmbeddings.embed_query` and `embed_documents` share `_embed`, which POSTs to `/embeddings` with `{"model": ..., "input": text}` and returns `data[0].embedding`. |
| FR-53.4 | When the MiniMax endpoint is unavailable, the import or first call raises; the operator falls back to `EMBEDDING_BACKEND=sentence-transformers`. |

### FR-54: Format-Aware Loaders

| ID | Requirement |
|----|-------------|
| FR-54.1 | `backend.rag.loaders.REGISTRY` maps extension → loader function. Loaders self-register via the `@register(ext)` decorator when their module is imported. |
| FR-54.2 | `ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())`. Currently: `.txt`, `.md`, `.pdf`, `.html`, `.docx`, `.csv`. |
| FR-54.3 | `RawDocument(text, metadata)` is the loader output type. For PDFs, each `RawDocument` is one page. For CSV, one row. For other formats, one per file. |
| FR-54.4 | `backend.rag.splitter.pick_splitter(extension, chunk_size, chunk_overlap)` returns `MarkdownTextSplitter` for `.md` and `RecursiveCharacterTextSplitter` for everything else. Markdown chunks respect header / code-block / list boundaries. |
| FR-54.5 | `split_into_documents(path, source_type, conversation_id, chunk_size, chunk_overlap)` dispatches to the registered loader, runs the format-appropriate splitter, and yields chunk `Document`s with propagated metadata. Metadata guaranteed on every chunk: `source`, `source_type`, `filename`, `format`, `chunk_id`. Per-format extras: `.md` → `header_path`, `.pdf` → `page_number`, `.html` → `title`, `.docx` → `paragraph_number`, `.csv` → `row_number`. |
| FR-54.6 | `_md_header_path(full_text, offset)` returns the breadcrumb (e.g. `Intro / Setup / Install`) by walking back from `offset` collecting the most recent header at each Markdown level. |

### FR-55: Cache Safety

| ID | Requirement |
|----|-------------|
| FR-55.1 | `backend.eval.cache.embedding_tag(embeddings)` returns a stable string identifier: tries `model_name` / `model` attributes first, then probes a query to read the output dimension, then falls back to the class name. The output is intended as a stable distinguisher between vector spaces, not a cryptographic fingerprint. |
| FR-55.2 | `load_or_build` accepts `embedding_tag_override` so tests can force a specific tag (e.g., `fake64`) and avoid class-name collisions. |
| FR-55.3 | `load_or_build` accepts `with_corpus=True` to also return the raw paragraph corpus (in the same order as the FAISS index). Required by `HybridRetriever` to build its BM25 index. |
| FR-55.4 | On cache load failure, the cache dir is `shutil.rmtree`'d, rebuilt, saved, and the run continues with a WARNING log. A corrupt cache never blocks a run. |

### FR-56: RAG Service Lifecycle

*(Existing iter-8 surface — captured here for completeness; not changed in this sweep.)*

| ID | Requirement |
|----|-------------|
| FR-56.1 | `RagService` owns two FAISS indexes: `library_index` (global, per-conversation-agnostic) and `uploads_index` (per-conversation). Both are tagged with the embedding backend name to prevent silent-failure on backend switch. |
| FR-56.2 | `ingest_file(conversation_id, path)` copies the file into `uploads_dir/<conversation_id>/`, chunks it, and `add_documents` to the uploads index. |
| FR-56.3 | `reindex_library()` walks `library_dir` recursively (allowlisted extensions), re-splits, and overwrites the library index in place. |
| FR-56.4 | `list_library_files()` returns metadata for every allowlisted file at any depth under `library_dir`. |
| FR-56.5 | `save_library_file(filename, content)` resolves the path safely (no `..` traversal), writes atomically via `mkstemp + os.replace`, and triggers an auto-reindex so the file is queryable immediately. |
| FR-56.6 | `delete_library_file(filename)` removes the file and auto-reindexes. Returns True if the file existed. |
| FR-56.7 | `_safe_library_path(library_dir, filename)` rejects filenames with leading separators, drive letters, or `..` components. Used by save/delete to accept subpath filenames like `hotpotqa/<id>.md` while still blocking path-traversal. |
| FR-56.8 | `retrieve_by_scope(conversation_id, query, top_k)` returns hits grouped by scope so the chat service can emit a sources event even when both scopes return zero hits. |

## Non-Functional Requirements

### NFR-19: Cost ceiling (updated)

The `--subset 1000 --pipeline cot_extract_notitles_thinking_k10` run issues ~1000 LLM calls (one per question; one mode). At `minimax-3` pricing with thinking budget 4096, expect $8-12.

The `--full --pipeline cot_extract_notitles_thinking_k10 --batch-size 2` run (iter-23's full-7k confirmation) issues 7369 LLM calls over ~12h wall-clock (detached subprocess on Windows). End-to-end cost ~$60-80.

### NFR-20: Latency target (updated)

Per-question end-to-end latency: 2-5 seconds for non-thinking presets, 8-15 seconds for thinking-mode presets (LLM call dominates; FAISS retrieval is ~50ms; answer extraction is negligible). With 1-second pacing and `--batch-size 1`, wall-clock for `--subset 1000 --pipeline cot_extract_k10` is ~17 minutes. With `--batch-size 2`, ~9 minutes.

### NFR-21: Reproducibility

LLM calls use `temperature=0` for determinism. Cache hits for FAISS indices make retrieval deterministic. The MiniMax endpoint is not 100% reproducible (network jitter, server-side variance); the `without_context` baseline in iter-12/13 runs moved by ±6 pp across runs — see iter-12 lesson #1. `contains_gold` is more stable than `answer_f1` across re-runs because it requires only substring containment, not exact-token-set match.

### NFR-22: Test isolation

All new code is testable without real LLM calls:
- `metrics` tests use string inputs.
- `qa_judge.build_qa_prompt` is pure.
- `qa_judge.ask_llm` is tested via `AsyncMock` for `AsyncAnthropic`.
- `pipeline` tests use `FakeVectorStore` / mocked clients.
- `pipeline.EmbeddingBackend` selection is tested via `embedding_tag_override` to bypass real probe calls.
- The CLI is tested via subprocess + small fixture JSON + mocked LLM via env-var stub.

### NFR-23: Eval CLI scaling

The CLI must handle full-dataset (7405 question) runs without dropping a single line:
- `--dump-results` writes JSON Lines incrementally so partial dumps survive crashes.
- `--start-from` + `--max-items` resume from any offset.
- `--batch-size N` (default 1, max recommended 2-4) uses `asyncio.gather` for concurrent LLM calls. Per-item setup (cache load + retrieval) is sequential within a batch; the concurrency is over LLM round-trips.
- On Windows, the detached-subprocess pattern (Python `subprocess.Popen` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`) keeps the eval worker alive even if the parent bash is reaped.

## Out of Scope (deferred to future iterations)

- **Rule-heavy prompt variants** (canonical-name post-processing, two-step extraction, yes/no discipline) — iter-15 → iter-19 all regressed. The LLM applies conditional rules inconsistently; small targeted nudges (iter-21 title-strip, iter-22 thinking) compound; rule-pile does not.
- **Larger embedding models beyond mpnet** — on HotpotQA, the mpnet upgrade costs 3× embedding time for +0.9 pp at k=4 (within noise).
- **Cross-encoder rerank beyond k=50 candidates** — rerank needs ≥20 candidates to express itself; k=50 already does, and HotpotQA's retrieval is saturated at k≥8.
- **Multi-shot prompting** — HotpotQA is zero-shot in this setup; few-shot is a known win for QA but adds prompt complexity.
- **Sentence-level supporting-fact scoring at the QA level** — would require the LLM to emit supporting-fact lists, not just answers.
- **Calibration metrics** (does the LLM's confidence match its correctness?) — orthogonal to this eval.
- **Per-(type, level) breakdown** for the QA eval — sample-size for `comparison/hard` in n=334 is small; defer until full-7k numbers are needed (now available).
- **RAGAS-style metrics** (faithfulness, answer relevance) — need a separate reference answer or LLM-as-judge pipeline.
- **LLM-as-judge** for subjective quality (e.g., "is the answer helpful?").
- **Top-k sweep beyond k=10** — HotpotQA's distractor setting provides exactly 10 paragraphs; k>10 is meaningless for this benchmark.