# Chatbot Project — Iteration 24 Design (RAG Pipeline Factory + Eval Diagnostics Sweep)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> See [SPEC_focus.md](SPEC_focus.md) for requirements. This document covers the architectural choices, module changes, and component skeletons that produced iter-11 through iter-23's implementation.

---

## 1. Architecture Decisions

### 1.1 Reuse the Eval Pipeline, Add a Pipeline Factory

**Choice**: The RAG pipeline (`backend/rag/pipeline.py`) is a thin orchestration layer that reuses the existing `backend.eval.qa_judge` LLM caller and the existing `backend.eval.cache` per-question FAISS cache. The factory composes embedder + retriever + optional reranker + prompt builder + LLM into a `RagPipeline` from a `PipelineConfig`.

**Rationale**:
- One-switch API: `build_pipeline(PRESETS[name], ...)` lets the chat service, eval CLI, and tests all run any preset by changing only the name.
- The factory wires existing primitives; no new evaluation surface area is needed.
- A new preset is a one-line addition to `PRESETS` — no other code changes.

**Trade-off**: Adding a new retriever or reranker requires editing `build_retriever` / `build_reranker` dispatch. This is one place to edit and one place to test, which is the right balance.

### 1.2 Pluggable Stages via Duck-Typed Protocols

**Choice**: `Retriever`, `Reranker`, `PromptBuilder`, `LLM` are `typing.Protocol` classes (structural subtyping — no inheritance required). Concrete implementations live in `backend/rag/pipeline.py`.

**Rationale**:
- Duck-typed protocols keep the contract minimal: each stage has one method.
- A new stage implementation (e.g., `HyDERetriever`) doesn't need to import or inherit from anything in the pipeline module.
- Tests can use lightweight fakes (`FakeVectorStore`, `AsyncMock`) without standing up FAISS / BM25 / cross-encoder machinery.

**Trade-off**: No static enforcement that an implementation honors its protocol. Mitigated by `backend/tests/rag/test_pipeline.py` (71 tests covering each implementation's contract).

### 1.3 Stage Dispatch via Factory Functions

**Choice**: `build_embedder`, `build_retriever`, `build_reranker`, `build_prompt_builder`, `build_llm` are top-level functions that take a `PipelineConfig` and return the corresponding implementation. Each raises `ValueError` for unknown stage names.

**Rationale**:
- One function per stage keeps each dispatch small and testable in isolation.
- The pipeline orchestrator (`RagPipeline.run`) calls each `build_*` once at construction time, so per-request overhead is zero.
- `ValueError` for unknown stages surfaces a config typo at startup, not on the first user request.

**Trade-off**: Five separate functions instead of one big `build_pipeline_internal` switch. Worth it — each is <20 lines and trivially mockable.

### 1.4 Reciprocal Rank Fusion for Hybrid Retrieval

**Choice**: `HybridRetriever(dense_retriever, bm25_retriever, rrf_k=60)` fuses the two ranked lists via RRF: `score(d) += 1 / (rrf_k + rank + 1)`. Uses `id(doc)` for identity-based fusion. Candidate depth per retriever is `max(k * 4, 20)` (paper-recommended).

**Rationale**:
- RRF doesn't require learning weights — robust to score-scale differences between dense and BM25.
- Identity-based fusion: a Document appearing in both lists scores higher than one appearing in only one. Cheap and correct.
- `cand_k = max(k * 4, 20)` gives the fusion room to express itself while staying bounded.

**Trade-off**: BM25 indexing happens at construction time per question (10 docs each — trivial). On HotpotQA at k=4, hybrid is **worse** than dense alone (0.769 vs 0.778) — HotpotQA's distractors are too lexically similar to the gold paragraphs for BM25 to add information. Kept as a documented preset because hybrid wins on technical-jargon / large-corpus workloads.

### 1.5 Cross-Encoder Reranking on Top of Dense

**Choice**: `dense_then_ce` preset uses mpnet (110M params, 768-dim) for dense retrieval of top-50, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks to top-4.

**Rationale**:
- Joint (query, document) encoding finds the right needle in a much larger haystack than top-4 alone.
- The cross-encoder is ~50ms per question on a CPU. Cold start ~5s to load the model.

**Trade-off**: On HotpotQA at k=4, dense+ce matches naive_dense within noise (+0.8 pp with 12 errors). The iter-12 report flagged this; iter-14 confirmed. Rerank needs k≥20 to express itself; HotpotQA retrieval is saturated at k≥8, so rerank's main value (re-ordering rank-10-hits) is moot when k already includes everything.

### 1.6 CoT Prompt Scaffold (iter-15 SOTA)

**Choice**: `CoTExtractPromptBuilder` instructs the LLM to (1) identify entities, (2) find relevant paragraphs, (3) chain multi-hop facts in order, (4) decide the exact span. The visible output begins with the extracted span in quotes + brief reasoning.

**Rationale**:
- At k≥8 retrieval is saturated (0 retrieval misses). The remaining failures are LLM extraction/reasoning errors, half of them multi-hop.
- A bare `extract_span` instruction doesn't scaffold that reasoning — the model sees one paragraph and quotes from it. The four-step scaffold forces the model to walk through the reasoning before quoting.
- The model is asked to lead with the span, so `contains_gold` stays safe even when the reasoning is wrong (the model quotes from somewhere plausible).

**Trade-off**: More tokens per call (~30% more than bare extract_span). Lifts `contains_gold` from 0.889 → 0.904 at k=10 (+1.5 pp; iter-15 SOTA).

### 1.7 Title-Strip (iter-21 Breakthrough)

**Choice**: `CoTExtractNoTitlesPromptBuilder` inherits the CoT instruction but strips the `[title]:` heading prefix on each context paragraph before sending. The body text is what gets sent to the LLM.

**Rationale** (from iter-20 thinking audit):
- When paragraphs are introduced with their Wikipedia article heading as a prefix (e.g., `[Hector Berlioz]: ...body...`), the model uses the heading as the entity label when emitting its answer.
- HotpotQA gold uses the full canonical name found in the article body opening (e.g., `Louis-Hector Berlioz`), not the colloquial heading form (`Hector Berlioz`).
- Stripping the heading forces the model to extract the canonical form from the body, where Wikipedia puts it in the first sentence.

**Trade-off**: A separate prompt builder (preserves the existing `CoTExtractPromptBuilder` for everything else). Title-strip is benchmark-specific: HotpotQA gold uses canonical names; other benchmarks may not have this gap. Worth checking before adopting as a non-HotpotQA default.

**Result**: `cot_extract_notitles_k10` lifts `contains_gold` from 0.904 → 0.925 (+2.1 pp).

### 1.8 Anthropic Extended Thinking Mode (iter-20 / iter-22)

**Choice**: `PipelineConfig.thinking_budget` enables Anthropic extended thinking mode. When set, `AnthropicLLM.ask` passes `thinking={"type": "enabled", "budget_tokens": thinking_budget}` to `messages.create`. The visible answer remains in `text` blocks; `thinking` blocks are discarded.

**Rationale**:
- The model reasons internally before emitting the visible answer.
- Internal reasoning doesn't dilute the visible answer with conversational wrapper tokens (which would lower `contains_gold` if the reasoning paraphrased instead of quoted).
- Combined with the title-strip + CoT scaffold, gives the model budget to reason about canonical-form choice on hard multi-hop questions.

**Trade-off**: ~50% more wall-clock (1939s vs 1012s on n=334) because thinking-mode emits 5-10× more output tokens per call. 2× LLM cost vs non-thinking CoT-only presets.

**Result**: `cot_extract_notitles_thinking_k10` lifts `contains_gold` from 0.925 → 0.934 (+0.9 pp on n=334), confirmed at **0.937 on n=7369** in iter-23's full-dataset run.

### 1.9 Why `contains_gold` Is the Headline Metric

**Choice**: The eval CLI reports `contains_gold` first, ahead of `answer_f1` and `answer_em`. `contains_gold` = 1.0 if the normalized gold answer appears as a substring of the normalized prediction; 0.0 otherwise.

**Rationale**:
- User-relevant: "did the user see the right answer?" — not "did the model output exactly the gold string?"
- Robust to conversational wrappers and quote-padded spans.
- `answer_f1` is reported for HotpotQA-strict benchmark parity, but it dilutes with verbatim-extraction prompts (which quote the span, lowering token-level F1 even though the answer is right).

**Trade-off**: `contains_gold` is binary, so per-variant breakdowns are coarse. The failure-mode breakdown (`success / extraction miss / retrieval miss`) is the finer-grained diagnostic on top of `contains_gold`.

### 1.10 Failure-Mode Instrumentation (`gold_in_top_k`)

**Choice**: `gold_paragraph_in_top_k(retrieved_titles, gold_titles)` returns True iff at least one gold paragraph title appears in retrieved. Recorded per question; the CLI's failure-mode breakdown bucket-defines failures by `contains_gold=0 + gold_in_top_k is True → extraction miss` vs `contains_gold=0 + gold_in_top_k is False → retrieval miss`.

**Rationale**:
- Without this split, every retrieval lever looks "saturated" when most of the gain is still on the retrieval side.
- The iter-12/13 "0.787 ceiling" claim was based on this missing split. iter-14 added `gold_in_top_k` and the ceiling jumped to 0.889 at k=10.

**Trade-off**: One extra bool per result. ~negligible cost.

### 1.11 Cache Key Includes Embedding-Tag Suffix

**Choice**: `load_or_build(item, dataset_sha, embeddings)` derives an `embedding_tag(embeddings)` (tries `model_name` / `model` attributes, then probes a query for output dimension, then falls back to class name) and uses `(dataset_sha, embedding_tag, qid)` as the cache key.

**Rationale**:
- Prevents silent-failure when switching embedding models: a 384-dim index would silently produce garbage for 768-dim queries.
- `embedding_tag_override` parameter lets tests force a specific tag (e.g., `fake64`) and avoid class-name collisions.

**Trade-off**: Tests with fake embedders need explicit `embedding_tag_override`. Documented in the helper.

### 1.12 Resumable Eval Runs (`--start-from` + `--max-items`)

**Choice**: The CLI accepts `--start-from N` (skip the first N items) and `--max-items N` (process at most N items from the start position). Combined with `--dump-results PATH`, this lets a crashed run resume without redoing finished work.

**Rationale**:
- iter-23's full-7k run died several times mid-flight on Windows (parent bash reaped the worker). Without resume, every crash meant redoing 8+ hours of work.
- The `iter22-full-7k-batch2-dump.jsonl` dump file is the authoritative source of completed results; the resume path lets new runs append to it.

**Trade-off**: The CLI doesn't auto-resume from a previous dump. Operator must read the last `qid` in the dump and pass `--start-from N` manually. (Auto-resume is out of scope.)

### 1.13 Detached Subprocess for Long Runs (iter-23)

**Choice**: For multi-hour runs, launch the eval via Python `subprocess.Popen` with `creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` (Windows). The detached process survives the parent bash being reaped.

**Rationale**:
- iter-23's first two full-7k attempts died silently at 43.7% and 13.4% completion — both attributed to the parent bash session being reaped.
- Detached subprocess + redirect stdout to a log file is the only pattern that survives the Claude Code shell session lifecycle.

**Trade-off**: Windows-specific. On Linux/macOS, `nohup` + `&` works equivalently.

### 1.14 One-Second Pacing Between LLM Calls

**Choice**: `PACING_SECONDS = 1` between LLM calls within a question's flow; batched parallel runs pace between batches, not within.

**Rationale**:
- LLM calls are heavier than paraphrase calls; rate-limit (429) is more sensitive.
- 1s is enough to keep the burst rate low without doubling wall-clock.
- The Anthropic SDK retries transparently on 429, so we don't see hard failures.

**Trade-off**: Slightly higher 429 risk than the paraphrase generator. Worth it for the eval throughput.

### 1.15 Subset Defaults (Updated)

**Choice**: Default subset is still 100 (cheap smoke test). For full-dataset runs, the recommendation is `--full --pipeline cot_extract_notitles_thinking_k10 --batch-size 2` (iter-23's SOTA confirmation command). Expect ~12h wall-clock, ~$60-80.

**Rationale**:
- `--subset 100` smoke test: ~$5, ~5 min.
- `--subset 1000` (stratified 334 effective): ~$15, ~30 min. Standard for iter-12 through iter-22 measurements.
- `--full` (7405 questions): only economically viable with `cot_extract_*` presets at k=10 because that's where retrieval is saturated and extraction is the residual problem.

---

## 2. Module Layout

### 2.1 New Files (since iter-9)

```
backend/eval/
└── qa_judge.py                  # ask_llm + build_qa_prompt (FR-41)

backend/rag/
├── pipeline.py                  # PipelineConfig + stage impls + PRESETS (FR-45..51)
├── embeddings.py                # make_embeddings factory (FR-53)
├── vector_store.py              # load_or_init / save / rebuild_filtered
├── retriever.py                 # ScopedRetriever (chat-side)
├── service.py                   # RagService (chat-side lifecycle)
├── routes.py                    # /api/rag/* HTTP routes
├── config.py                    # RagSettings (env-driven)
├── splitter.py                  # pick_splitter + split_into_documents (FR-54)
└── loaders/                     # format-aware loader registry
    ├── __init__.py              # REGISTRY + RawDocument + ALLOWED_EXTENSIONS
    ├── text.py                  # .txt / .md
    ├── pdf.py                   # .pdf (one page per RawDocument)
    ├── html.py                  # .html (with title metadata)
    ├── docx.py                  # .docx
    └── csv.py                   # .csv (one row per RawDocument)

scripts/
├── eval_qa_hotpotqa.py          # end-to-end QA accuracy eval CLI (FR-42)
└── run_rag.py                   # standalone RAG runner for testing

backend/tests/rag/
├── test_pipeline.py             # 71 tests covering presets + factory + each impl
├── test_loaders.py
├── test_splitter.py
├── test_routes.py
├── test_service.py
├── test_retriever.py
├── test_vector_store.py
├── test_config.py
├── test_embeddings_factory.py
└── conftest.py

backend/tests/eval/
├── test_qa_judge.py             # build_qa_prompt + ask_llm (mocked client)
└── test_metrics.py              # 208 lines, covers FR-40..41 + FR-52
```

### 2.2 Modified Files

| File | Change |
|---|---|
| `backend/eval/qa_judge.py` | Added `thinking_budget` parameter to `ask_llm` (iter-20). |
| `backend/eval/cache.py` | Added `embedding_tag()`, `embedding_tag_override`, `with_corpus=True` option (FR-55). |
| `backend/main.py` | Lifespan builds `RagService` from settings; injects into chat routes; mounts `/api/rag/*`. Wires delete-conversation callback chain (`rag.purge_uploads` → `chat_service.clear_pending_inline_files`). |
| `scripts/eval_qa_hotpotqa.py` | Added `--pipeline`, `--list-pipelines`, `--dump-results`, `--batch-size`, `--start-from`, `--max-items`. Added `contains_gold` + `gold_in_top_k` to per-question output and failure-mode breakdown block. |
| `requirements.txt` | Added `rank-bm25>=0.2.0` for `HybridRetriever` (iter-13). |

### 2.3 Unchanged (since iter-9)

- `backend/eval/hotpotqa.py` — dataset loader + `gold_paragraph_titles` + `dataset_sha` + `sample()`.
- `backend/eval/paraphrases.py` — paraphrase set loader.
- `scripts/eval_hotpotqa.py` — retrieval-only eval (FR-31 surface).
- `scripts/ingest_hotpotqa.py` — HotpotQA → markdown library ingest (FR-30 surface).
- `scripts/generate_paraphrases_hotpotqa.py` — paraphrase generator (FR-34 surface).
- `backend/chat/*` — chat chain, service, routes, stream_manager. RAG_SYSTEM_PROMPT is mirrored in `qa_judge.py` to preserve FR-32 isolation.
- `frontend/*` — frontend is unchanged in this sweep.

---

## 3. Component Skeletons

### 3.1 `backend/rag/pipeline.py` — PipelineConfig + factory

```python
@dataclass(frozen=True)
class PipelineConfig:
    name: str
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever: str = "dense"          # 'dense' | 'hybrid'
    reranker: str | None = None        # None | 'cross_encoder'
    rerank_top_k: int = 50
    top_k: int = 4
    prompt_template: str = "default"   # 'default' | 'extract_span' | 'cot_extract' | ...
    thinking_budget: int | None = None
    llm_model: str = "minimax-3"


class RagPipeline:
    def __init__(self, config, retriever, reranker, prompt_builder, llm):
        self.config = config
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm

    async def run(self, question: str) -> str:
        retrieve_k = self.config.rerank_top_k if self._reranker else self.config.top_k
        candidates = self._retriever.retrieve(question, k=retrieve_k)
        if self._reranker:
            final_docs = self._reranker.rerank(question, candidates, top_k=self.config.top_k)
        else:
            final_docs = candidates[: self.config.top_k]
        messages = self._prompt_builder.build(question, final_docs)
        return await self._llm.ask(messages)


def build_pipeline(config, vectorstore, llm_client, corpus=None) -> RagPipeline:
    return RagPipeline(
        config=config,
        retriever=build_retriever(config, vectorstore, corpus=corpus),
        reranker=build_reranker(config),
        prompt_builder=build_prompt_builder(config),
        llm=build_llm(config, llm_client),
    )
```

### 3.2 `backend/rag/pipeline.py` — Retrievers

```python
class DenseRetriever:
    def __init__(self, vectorstore):
        self._vs = vectorstore
    def retrieve(self, query, k):
        return self._vs.similarity_search(query, k=k)


class BM25Retriever:
    def __init__(self, docs):
        import re
        from rank_bm25 import BM25Okapi
        self._docs = list(docs)
        self._tokenize = lambda s: [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t]
        if self._docs:
            tokenized_corpus = [self._tokenize(d.page_content) for d in self._docs]
            self._bm25 = BM25Okapi(
                [toks if toks else ["_empty_"] for toks in tokenized_corpus]
            )
        else:
            self._bm25 = None

    def retrieve(self, query, k):
        if not self._docs or self._bm25 is None:
            return []
        q_tokens = self._tokenize(query) or ["_empty_"]
        scores = self._bm25.get_scores(q_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._docs[i] for i in ranked_indices[:k]]


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, rrf_k=60):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query, k):
        cand_k = max(k * 4, 20)
        dense_hits = self._dense.retrieve(query, k=cand_k)
        bm25_hits = self._bm25.retrieve(query, k=cand_k)
        scores, docs_by_id = {}, {}
        for rank, doc in enumerate(dense_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc
        for rank, doc in enumerate(bm25_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [docs_by_id[doc_id] for doc_id, _ in ranked[:k]]
```

### 3.3 `backend/rag/pipeline.py` — Rerankers

```python
class NoOpReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # lazy-loaded on first call

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        self._ensure_model()
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_k]]
```

### 3.4 `backend/rag/pipeline.py` — Prompt builders

```python
class DefaultPromptBuilder:
    """Wraps backend.eval.qa_judge.build_qa_prompt."""
    def build(self, question, context_docs):
        from backend.eval.qa_judge import build_qa_prompt
        return build_qa_prompt(question, context_docs)


class ExtractSpanPromptBuilder:
    EXTRACT_INSTRUCTION = (
        "Read the <context>...</context> block carefully and extract the "
        "exact span that answers the question. Begin your response with the "
        "extracted span (in quotation marks if it is a phrase), then briefly "
        "explain. Do not paraphrase the answer — quote it verbatim."
    )
    def __init__(self):
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.EXTRACT_INSTRUCTION

    def build(self, question, context_docs):
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class CoTExtractPromptBuilder:
    """Iter-15 SOTA: CoT scaffold + verbatim-span extraction."""
    COT_INSTRUCTION = (
        "Read the <context>...</context> block carefully. Some questions "
        "require combining facts from multiple paragraphs (multi-hop reasoning).\n\n"
        "Think step by step:\n"
        "1. Identify the entities and facts the question asks about.\n"
        "2. Find the relevant paragraph(s) in the context.\n"
        "3. If multi-hop reasoning is needed, chain together the supporting "
        "facts in order.\n"
        "4. Decide which exact span answers the question.\n\n"
        "Begin your response with the extracted span (in quotation marks "
        "if it is a phrase), then briefly explain your reasoning. "
        "Do not paraphrase the answer — quote it verbatim from the context."
    )
    def __init__(self):
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.COT_INSTRUCTION

    def build(self, question, context_docs):
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class CoTExtractV2PromptBuilder(CoTExtractPromptBuilder):
    """Iter-19: minimal targeted refinement (regressed, kept as documented variant)."""
    COT_INSTRUCTION_V2 = (
        # Same as CoTExtractPromptBuilder.COT_INSTRUCTION but step 4 and the
        # closing directive add "using the most complete form as written in
        # the context for entity-name answers." Total ~25 word growth.
    )


class CoTExtractNoTitlesPromptBuilder(CoTExtractPromptBuilder):
    """Iter-21: CoT scaffold with [title]: prefix stripped from each context paragraph."""
    def build(self, question, context_docs):
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
```

### 3.5 `backend/rag/pipeline.py` — LLM wrapper

```python
class AnthropicLLM:
    def __init__(self, client, model, thinking_budget=None):
        self._client = client
        self._model = model
        self._thinking_budget = thinking_budget

    async def ask(self, messages, max_tokens=200):
        from backend.eval.qa_judge import ask_llm
        return await ask_llm(
            self._client,
            self._model,
            messages,
            max_tokens=max_tokens,
            thinking_budget=self._thinking_budget,
        )
```

### 3.6 `backend/rag/pipeline.py` — PRESETS

```python
PRESETS = {
    "naive_dense": PipelineConfig(
        embedding_model="all-MiniLM-L6-v2", retriever="dense",
        top_k=4, prompt_template="default",
    ),
    "large_dense": PipelineConfig(
        embedding_model="all-mpnet-base-v2", retriever="dense",
        top_k=4, prompt_template="default",
    ),
    "dense_then_ce": PipelineConfig(
        embedding_model="all-mpnet-base-v2", retriever="dense",
        reranker="cross_encoder", rerank_top_k=50,
        top_k=4, prompt_template="default",
    ),
    "extract_span_prompt": PipelineConfig(
        embedding_model="all-mpnet-base-v2", retriever="dense",
        top_k=4, prompt_template="extract_span",
    ),
    "extract_span_k8": PipelineConfig(
        top_k=8, prompt_template="extract_span",
    ),
    "extract_span_k10": PipelineConfig(
        top_k=10, prompt_template="extract_span",
    ),
    "cot_extract_k10": PipelineConfig(  # iter-15 SOTA at 0.904
        top_k=10, prompt_template="cot_extract",
    ),
    "cot_extract_v2_k10": PipelineConfig(  # iter-19, regressed, documented variant
        top_k=10, prompt_template="cot_extract_v2",
    ),
    "cot_thinking_k10": PipelineConfig(  # iter-20, tied at 0.904
        top_k=10, prompt_template="extract_span",
        thinking_budget=4096,
    ),
    "cot_extract_notitles_k10": PipelineConfig(  # iter-21, 0.925
        top_k=10, prompt_template="cot_extract_no_titles",
    ),
    "cot_extract_notitles_thinking_k10": PipelineConfig(  # iter-22/23 SOTA, 0.937
        top_k=10, prompt_template="cot_extract_no_titles",
        thinking_budget=4096,
    ),
    "hybrid_bm25_dense": PipelineConfig(
        retriever="hybrid", top_k=4, prompt_template="default",
    ),
}
```

### 3.7 `backend/eval/qa_judge.py` — `ask_llm` with thinking

```python
async def ask_llm(client, model, messages, max_tokens=200, thinking_budget=None):
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=0, messages=messages)
    if thinking_budget is not None and thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    response = await client.messages.create(**kwargs)
    parts = []
    for block in response.content:
        if _block_type(block) == "text":
            text = _block_text(block)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()
```

### 3.8 `backend/eval/cache.py` — embedding-tag-aware cache

```python
def embedding_tag(embeddings):
    for attr in ("model_name", "model"):
        if hasattr(embeddings, attr):
            value = getattr(embeddings, attr)
            if isinstance(value, str) and value:
                return value.replace("/", "_").replace("\\", "_")
    try:
        vec = embeddings.embed_query("embedding-tag-probe")
        return f"dim{len(vec)}"
    except Exception:
        pass
    return type(embeddings).__name__


def load_or_build(item, dataset_sha, embeddings, no_cache=False,
                  embedding_tag_override=None, with_corpus=False):
    tag = embedding_tag_override or embedding_tag(embeddings)
    cache_dir = EVAL_CACHE_ROOT / f"{dataset_sha}_{tag}" / item.id
    if no_cache or not cache_dir.exists():
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        if with_corpus:
            return index, False, _build_corpus(item)
        return index, False
    try:
        index = load_or_init(cache_dir, embeddings)
        if with_corpus:
            return index, True, _build_corpus(item)
        return index, True
    except Exception as e:
        log.warning("cache corrupt for %s (%s); rebuilding", item.id, e)
        shutil.rmtree(cache_dir, ignore_errors=True)
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        if with_corpus:
            return index, False, _build_corpus(item)
        return index, False
```

### 3.9 `backend/rag/embeddings.py` — factory

```python
class MiniMaxEmbeddings(Embeddings):
    """httpx POST to {base_url}/embeddings with the same auth as the chat model."""
    def __init__(self, api_key, base_url, model="minimax-3"):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _embed(self, text):
        import httpx
        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_query(self, text):
        return self._embed(text)
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]


def make_embeddings(backend, *, model_name="all-MiniLM-L6-v2",
                    api_key="", base_url="https://api.minimax.chat/v1"):
    if backend == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    if backend == "minimax":
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return MiniMaxEmbeddings(api_key=api_key, base_url=base_url)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
```

### 3.10 `backend/rag/loaders/` — registry skeleton

```python
@dataclass
class RawDocument:
    text: str
    metadata: dict = field(default_factory=dict)

REGISTRY: dict[str, LoaderFn] = {}   # extension (lowercase) → loader

def register(extension):
    def _decorator(fn):
        REGISTRY[extension.lower()] = fn
        return fn
    return _decorator

def load(path, source):
    ext = path.suffix.lower()
    loader = REGISTRY.get(ext)
    if loader is None:
        raise UnsupportedFormatError(ext)
    yield from loader(path, source)

# Self-registering imports at module bottom:
from backend.rag.loaders import text, pdf, html, docx, csv  # noqa
ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())
```

### 3.11 `backend/rag/splitter.py` — format-aware splitter

```python
def pick_splitter(extension, chunk_size, chunk_overlap):
    if extension.lower() == ".md":
        from langchain_text_splitters import MarkdownTextSplitter
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_into_documents(path, source_type, conversation_id, chunk_size, chunk_overlap):
    """Dispatches to the registered loader, runs the format-appropriate splitter,
    yields chunk Documents with metadata guaranteed: source, source_type, filename,
    format, chunk_id. Per-format extras: .md→header_path, .pdf→page_number,
    .html→title, .docx→paragraph_number, .csv→row_number."""
    ext = path.suffix.lower()
    splitter = pick_splitter(ext, chunk_size, chunk_overlap)
    full_text = path.read_text(encoding="utf-8") if ext == ".md" else ""
    for raw in registry_load(path, source_type):
        if not raw.text.strip():
            continue
        for chunk_text in splitter.split_text(raw.text):
            meta = dict(raw.metadata)
            meta["source"] = source_type
            meta["source_type"] = source_type
            meta["filename"] = path.name
            meta["format"] = ext
            if conversation_id is not None:
                meta["conversation_id"] = conversation_id
            if ext == ".md":
                snippet = chunk_text[:80].strip()
                if snippet:
                    idx = full_text.find(snippet)
                    if idx >= 0:
                        meta["header_path"] = _md_header_path(full_text, idx)
            meta["chunk_id"] = hashlib.sha256(
                f"{path.name}:{chunk_text}".encode()
            ).hexdigest()[:16]
            yield Document(page_content=chunk_text, metadata=meta)
```

### 3.12 `scripts/eval_qa_hotpotqa.py` — key signatures

```python
PACING_SECONDS = 1

async def _evaluate_one(client, model, item, retrieved_docs, question_text,
                        variant_name, mode, prompt_template="default",
                        gold_in_top_k=None, gold_titles=None,
                        retrieved_titles=None, thinking_budget=None,
                        max_tokens=None) -> dict:
    """One LLM call + scoring. Records contains_gold + answer_f1 + answer_em +
    gold_in_top_k. Always sleeps PACING_SECONDS before the LLM call."""
    ...


async def _process_one_item(client, item) -> list[dict]:
    """Process one item: cache load (with corpus if hybrid) → variants × modes.
    For hybrid preset: build HybridRetriever per item (BM25 + Dense)."""
    ...


async def run():
    """Batches items via asyncio.gather(batch_size=2). Pacing between batches."""
    ...
```

---

## 4. Configuration

| Env var | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Chat + MiniMax embeddings (when EMBEDDING_BACKEND=minimax). |
| `ANTHROPIC_BASE_URL` | `https://api.minimax.chat/v1` | Used by both chat and MiniMax embeddings. |
| `ANTHROPIC_MODEL` | `minimax-3` | Override via `--llm-model` or this env var. |
| `RAG_ENABLED` | `false` | When true, `/api/rag/*` routes are mounted at startup. |
| `RAG_EMBEDDING_BACKEND` | `sentence-transformers` | One of `sentence-transformers`, `minimax`. |
| `RAG_SENTENCE_TRANSFORMERS_MODEL` | `all-MiniLM-L6-v2` | The HF model name. |
| `RAG_LIBRARY_DIR` | `storage/library` | Relative to `<repo>/backend/`. |
| `RAG_UPLOADS_DIR` | `storage/uploads` | Relative to `<repo>/backend/`. |
| `RAG_INDEX_DIR` | `storage/rag` | Relative to `<repo>/backend/`. |
| `RAG_CHUNK_SIZE` | `800` | RecursiveCharacterTextSplitter chunk_size. |
| `RAG_CHUNK_OVERLAP` | `200` | RecursiveCharacterTextSplitter chunk_overlap. |
| `RAG_TOP_K` | `4` | Per-scope top_k for chat retrieval. |
| `RAG_INLINE_CONTEXT_THRESHOLD_BYTES` | `8192` | Files smaller than this go inline (UTF-8 text); larger go through FAISS. |

## 5. Error Handling

| Stage | Failure | Behavior |
|---|---|---|
| Eval: missing `ANTHROPIC_API_KEY` | env var unset | Exit 1 with hint. |
| Eval: dataset missing | path doesn't exist | Exit 1 with download instructions. |
| Eval: dataset corrupt | JSON parse fails | Exit 1. |
| Eval: embedding model load | sentence-transformers not installed | Exit 1. |
| Eval: per-question cache corrupted | `load_local` raises | `shutil.rmtree`, rebuild, WARNING log, continue. |
| Eval: per-question LLM call fails | API error / network error | Log WARNING, count as errored, skip rest of run unaffected. |
| Eval: LLM returns empty content | text extraction yields "" | `answer_f1("", gold) == 0.0`; counted normally. |
| Eval: unknown `--pipeline` | name not in `PRESETS` | Exit 1 with `Available: [...]` list and `--list-pipelines` hint. |
| Eval: rate-limited (429) | Anthropic SDK retries internally | Transparent. |
| Eval: detached subprocess reaped | parent shell terminates | Resume from `--start-from`. |
| RAG service: missing files / IO error | upload, delete, reindex | 400/409/500 with diagnostic; auto-reindex failure logged but does not roll back the write. |
| RAG service: path traversal in `filename` | `_safe_library_path` detects `..` | 400 from routes layer; service raises ValueError. |

## 6. Testing Strategy

### 6.1 Layers

| Layer | Files | Speed |
|---|---|---|
| Metrics unit | `backend/tests/eval/test_metrics.py` (208 lines) | <10 ms each |
| QA judge unit | `backend/tests/eval/test_qa_judge.py` (159 lines, mocked AsyncAnthropic) | <100 ms each |
| Pipeline unit | `backend/tests/rag/test_pipeline.py` (806 lines, 71 tests, FakeVectorStore) | <100 ms each |
| Loaders unit | `backend/tests/rag/test_loaders.py` (119 lines) | <50 ms each |
| Splitter unit | `backend/tests/rag/test_splitter.py` (157 lines) | <50 ms each |
| RAG service unit | `backend/tests/rag/test_service.py` (254 lines) | <500 ms each |
| RAG routes unit | `backend/tests/rag/test_routes.py` (401 lines) | <200 ms each |
| Eval integration | `backend/tests/eval/test_eval_integration.py` | <5 s |
| Manual smoke | operator runs `python scripts/eval_qa_hotpotqa.py --subset 10` | ~30 s |

### 6.2 Manual Smoke Tests

```bash
# 1. Smoke test — pipeline mode
python scripts/eval_qa_hotpotqa.py --subset 10 --pipeline naive_dense

# 2. SOTA benchmark — n=334 sample
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline cot_extract_notitles_thinking_k10
# expect: ~30 min wall-clock, contains_gold ~ 0.93

# 3. Full-7k confirmation — multi-hour run, detached subprocess on Windows
python -c "
import subprocess, sys
DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP = 0x08, 0x0200
subprocess.Popen(
    [sys.executable, 'scripts/eval_qa_hotpotqa.py',
     '--full', '--pipeline', 'cot_extract_notitles_thinking_k10',
     '--batch-size', '2',
     '--dump-results', 'docs/eval-results/iter24-full-7k-dump.jsonl'],
    stdout=open(r'C:/Users/Administrator/AppData/Local/Temp/full_eval.log', 'wb'),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
"
# expect: ~12h wall-clock, contains_gold ~ 0.937 on n=7369
```

### 6.3 Isolation Guard

```bash
grep -rn "backend\.chat" backend/eval/ scripts/eval_qa_hotpotqa.py
# Expected: no matches in eval scripts (qa_judge intentionally duplicates the prompt
# as a string constant; scripts/eval_qa_hotpotqa.py imports from backend.eval.* only).
```

---

## 7. Out of Scope (Deferred to Future Iterations)

1. **Rule-heavy prompt variants** — canonical-name post-processing, two-step extraction, yes/no discipline, longer rule sets. All regressed in iter-15 → iter-19; the LLM applies conditional rules inconsistently.
2. **Larger embedding models beyond mpnet** — diminishing returns at k≥8.
3. **Cross-encoder rerank beyond k=50 candidates** — rerank's value is moot when k already covers the whole corpus.
4. **Multi-shot prompting** for QA.
5. **Sentence-level supporting-fact scoring** at the QA level.
6. **Calibration metrics** (confidence vs correctness).
7. **Per-(type, level) breakdown** for the QA eval — sample-size for `comparison/hard` in n=334 is small; full-7k numbers now available.
8. **RAGAS-style metrics** (faithfulness, answer relevance).
9. **LLM-as-judge** for subjective quality.
10. **Top-k sweep beyond k=10** — HotpotQA's distractor setting provides exactly 10 paragraphs.
11. **Auto-resume from previous dump** — currently operator must read the last qid and pass `--start-from N` manually.
12. **Cross-platform detached subprocess pattern** — current pattern is Windows-specific; `nohup &` on Linux/macOS works equivalently but isn't packaged.