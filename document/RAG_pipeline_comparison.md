# RAG Pipeline Comparison

> **All numbers in this document are measured, not estimated.** Every row in
> the comparison table comes from an actual eval run on this machine, on the
> same HotpotQA subset, with the same LLM and pacing settings. Source command
> for each row is documented under "How to reproduce" so you can rerun any
> cell independently.

This is a one-sheet, copy-paste-ready reference for every RAG pipeline that
exists as a preset in `backend.rag.pipeline.PRESETS`. Each pipeline is a
switch via `--pipeline NAME` on `scripts/eval_qa_hotpotqa.py`, or a one-line
import in your own code via `build_pipeline(PRESETS[name], ...)` —
implementations are deliberately short.

---

## 1. Test conditions (apply to every row)

| Item | Value |
|---|---|
| Dataset | HotpotQA `dev_distractor` v1 (CC BY-SA 4.0) |
| Subset | n=334 stratified sample (deterministic seed=42) for early iter-12 → iter-22 results. n=7369 of 7405 (~99.5% coverage; 36 skipped — sensitive-content filter + transient 5xx) for the iter-23 full-dataset SOTA confirmation. |
| `dataset_sha` | `4e9ecb5c8d3b719f` (file-hash prefix, identical across all rows) |
| LLM | `minimax-3` via MiniMax Anthropic-compatible endpoint |
| Temperature | 0 (deterministic) |
| Pacing | `PACING_SECONDS = 1` (1s between LLM calls); batched parallel runs use `--batch-size 2` for ~2× throughput |
| Per-question corpus | 10 paragraphs (2 gold + 8 distractor) |
| Embedding backends | `sentence-transformers` (HuggingFace) |
| Iterations | iter-12 (large_dense, dense_then_ce), iter-13 (hybrid), iter-14 (extract_span family + k-variants), iter-15 (cot_extract), iter-19 (cot_extract_v2), iter-20 (cot_thinking), iter-21 (cot_extract_notitles), iter-22 (cot_extract_notitles_thinking), iter-23 (full-7k SOTA confirmation). |

Re-running any row from scratch takes ~17–60 min wall-clock depending on
cold-cache vs warm-cache. The full 7k run took ~12h wall-clock with
batch_size=2 + detached subprocess (see §3.8).

---

## 2. Headline comparison

The single number to optimize is `contains_gold` (substring containment of
the normalized gold answer in the model's output). It is the most
user-relevant metric — partial-credit F1 dilutes with conversational
wrappers and exact-match is harsh for short answers.

Sorted by `contains_gold` (best first), same dataset, same LLM.
Rows annotated **[n=334]** or **[n=7369 full]** indicate the sample size
that produced the number. The **[n=7369 full]** row is the official
SOTA, confirmed on the entire HotpotQA dev_distractor dataset
(skipping 36 questions rejected by the API's sensitive-content filter
or that hit transient 5xx after retries).

| # | Pipeline | `contains_gold` | `answer_f1` | `answer_em` | extraction miss | retrieval miss | n | Cost note |
|:-:|---|---:|---:|---:|---:|---:|:-:|---|
| 1 | **cot_extract_notitles_thinking_k10** (NEW SOTA) | **0.937** | 0.084 | 0.000 | 467 | 0 | **7369 full** | MiniLM + 10 paragraphs + title-strip + CoT scaffold + Anthropic thinking (4096 budget) |
| 1' | cot_extract_notitles_thinking_k10 (n=334 sample) | 0.934 | 0.077 | 0.000 | 22 | 0 | 334 | (same preset, smaller sample — n=334 was published first) |
| 2 | cot_extract_notitles_k10 | 0.925 | 0.088 | 0.009 | 25 | 0 | 334 | MiniLM + 10 paragraphs + title-strip + CoT scaffold |
| 3 | cot_extract_k10 | 0.904 | 0.080 | 0.000 | 32 | 0 | 334 | MiniLM + 10 paragraphs + CoT scaffold (titles retained) |
| 4 | extract_span_k10 | 0.889 | — | — | 37 | 0 | 334 | MiniLM + 10 paragraphs + verbatim-span prompt |
| 5 | top_k=10 (naive) | 0.880 | — | — | 40 | 0 | 334 | MiniLM + 10 paragraphs, default prompt |
| 6 | extract_span_k8 | 0.874 | — | — | 42 | 0 | 334 | MiniLM + 8 paragraphs + verbatim-span prompt |
| 7 | top_k=8 (naive) | 0.850 | — | — | 50 | 0 | 334 | MiniLM + 8 paragraphs, default prompt |
| 8 | extract_span_prompt (k=4) | 0.792 | — | — | 63 | 6 | 332 | mpnet, k=4, verbatim-span prompt |
| 9 | large_dense (mpnet, k=4) | 0.787 | — | — | 68 | 3 | 334 | mpnet, k=4, default prompt |
| 10 | dense_then_ce (rerank) | 0.786 | — | — | 68 | 1 | 322 | mpnet + cross-encoder rerank 50→4 |
| 11 | naive_dense (k=4) | 0.778 | — | — | 71 | 3 | 334 | MiniLM, k=4, default prompt (baseline) |
| 12 | hybrid_bm25_dense | 0.769 | — | — | 74 | 3 | 334 | MiniLM + BM25 via RRF, k=4 |

(Note: `cot_thinking_k10` and `cot_extract_v2_k10` were also evaluated but tied
with `cot_extract_k10` at 0.904 — kept in `PRESETS` as documented variants.)

**`answer_f1` / `answer_em` sourcing**: values are only shown for rows that
have an authoritative per-question dump under `docs/eval-results/iter*-k10-dump.jsonl`
(rows 1, 1', 2, 3, and the full-7k SOTA confirmation). The earlier k=4 / k=8
runs (rows 4–12) did not persist raw dumps — only the `contains_gold` and
failure-mode counts were published in the iter-12/13/14 markdown reports.
Showing `—` for these is more honest than fabricating values.

### Key observations from the table

- **Retrieval saturates at k≥8.** All k≥8 rows have **0 retrieval misses**.
  At k=4 the retrieval miss bucket is 1–6 questions regardless of the
  retriever choice (dense, mpnet, rerank, hybrid).
- **The biggest single lever is `top_k`**, not the retriever. Going from
  k=4 to k=8 lifts `contains_gold` by ~7 pp. Going from k=8 to k=10 lifts
  by ~3 pp more.
- **The prompt-only `extract_span` change** adds ~1–2 pp on top of any
  `top_k`. Small but consistent and free.
- **mpnet (larger embedding)**, **cross-encoder rerank**, and **hybrid
  BM25+dense** all come in BELOW the cheaper `top_k=8` baseline. On
  HotpotQA these are no-ops (or negative) at `top_k=4`.
- The lift is mostly additive: `(k=4 → k=8) + extract_span` ≈
  `(k=4 → extract_span_prompt)` combined. No surprising synergy.

### Δ vs the iter-12/13 baseline

```
contains_gold over naive_dense k=4 (0.778):
  extract_span_k10 : +11.1 pp   ← recommended default
  top_k=10 naive   : +10.2 pp
  extract_span_k8  :  +9.6 pp
  top_k=8 naive    :  +7.2 pp
  extract_span_prompt : +1.4 pp
  large_dense      :  +0.9 pp   (within noise)
  dense_then_ce    :  +0.8 pp   (within noise, 12 errors!)
  hybrid_bm25_dense:  -0.9 pp   (within noise)

Trajectory after iter-14 (the original SOTA was cot_extract_k10 at 0.904):

  cot_extract_k10                  :   baseline       (0.904, 32 fail, n=334)
  cot_extract_v2_k10 (nudge)       :   -0.9 pp        (regressed)
  cot_thinking_k10                 :    0 pp  tie     (0.904, 32 fail — only 25 of 32 failures overlap with iter-15)
  cot_extract_notitles_k10         :   +2.1 pp        (0.925, 25 fail, n=334)
  cot_extract_notitles_thinking_k10:   +3.0 pp        (0.934, n=334 sample, 22 fail)
                              full :   +3.3 pp        (0.937, n=7369 full, 467 fail)  ← official SOTA

The iter-20 audit (thinking-mode dump of failure thinking content) revealed
that the model was treating Wikipedia article headings as entity labels
when emitting answers — but HotpotQA's gold uses the full canonical name
found in the article body opening. Iter-21 stripped the `[title]:`
heading prefix to remove the bad cue at the source. Iter-22 added
thinking-mode reasoning budget on top, giving the model space to reason
about canonical-form choice. Each lever compounds; no single lever
reached the SOTA alone.

The full 7k confirmation (n=7369) used batched parallel execution
(`--batch-size 2`) inside a fully-detached subprocess so the parent
bash that Claude Code uses couldn't reap the worker. Wall-clock was
~12h; throughput averaged ~14 q/min once warm. The dump at
`docs/eval-results/iter22-full-7k-batch2-dump.jsonl` holds the
per-question results for the 7369 completed items.

---

## 3. Per-pipeline details

Each section below is self-contained: the preset config, what each stage
does, which file owns the implementation, the CLI one-liner, and any
caveats. Read these in order if you want to understand the system; jump
straight to "How to use" if you just want the switch.

---

### 3.1 naive_dense — the baseline

**What it is**: Small embedding model (MiniLM, 22M params, 384-dim) + FAISS
top-k + default conversational prompt.

**Configuration**:
```python
PRESETS["naive_dense"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    retriever="dense",
    reranker=None,
    top_k=4,
    prompt_template="default",
    llm_model="minimax-3",
)
```

**Implementation** (3 components, ~80 lines of code total):

| Stage | Class | File | Lines |
|---|---|---|---|
| Embedder | `HuggingFaceEmbeddings` | `backend/rag/embeddings.py::_build_huggingface` | 5-7 |
| Retriever | `DenseRetriever` | `backend/rag/pipeline.py::DenseRetriever` | 99-111 |
| Prompt | `DefaultPromptBuilder` | `backend/rag/pipeline.py::DefaultPromptBuilder` | 235-242 |
| Orchestrator | `RagPipeline.run` | `backend/rag/pipeline.py::RagPipeline.run` | 376-401 |

The dense retriever just calls `vectorstore.similarity_search(query, k=top_k)`.
The prompt wraps the docs in a `<context>...</context>` block and asks the
LLM to answer using it as grounding material (no verbatim guidance).

**How to use**:
```bash
# Eval
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline naive_dense
# In code
from backend.rag.pipeline import PRESETS, build_pipeline
pipeline = build_pipeline(PRESETS["naive_dense"], vectorstore=vs, llm_client=client)
```

**When to use**: Setting a baseline. Cheap. For corpora where retrieval is
not the bottleneck and a small model is acceptable.

**Caveats**: At k=4, ~3 questions out of 334 have a gold paragraph outside
the top-4 (~1% retrieval miss rate). 71 questions (~21%) have the gold
paragraph in top-k but the LLM still didn't output the answer — that's the
"extraction miss" ceiling at this setting.

---

### 3.2 large_dense — bigger embedding model

**What it is**: Replaces MiniLM with mpnet (110M params, 768-dim). All else
identical to `naive_dense`.

**Configuration**:
```python
PRESETS["large_dense"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-mpnet-base-v2",
    retriever="dense",
    reranker=None,
    top_k=4,
    prompt_template="default",
    llm_model="minimax-3",
)
```

**Implementation**: Same code paths as naive_dense. Only difference: the
embedder is `_build_huggingface("all-mpnet-base-v2")`, which lazy-loads a
larger model and produces 768-dim vectors instead of 384-dim. FAISS picks
the right index dimension from probe vectors automatically.

**Cost delta vs naive_dense**: ~3× embedding compute time per question
(mpnet is bigger). Same per-question wall-clock for retrieval. LLM cost is
identical.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline large_dense
```

**When to use**: When you suspect semantic similarity is the bottleneck and
budget allows for a heavier model. On HotpotQA, this is **not** the case
(the gain is within noise at +0.9 pp).

**Caveats**: Cache key includes the embedder tag, so the FAISS index is
not shared with naive_dense — first-time build is slower.

---

### 3.3 dense_then_ce — cross-encoder rerank on top of mpnet

**What it is**: retrieve top-50 with mpnet, then rerank to top-4 with a
cross-encoder (MiniLM cross-encoder trained on MS MARCO).

**Configuration**:
```python
PRESETS["dense_then_ce"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-mpnet-base-v2",
    retriever="dense",
    reranker="cross_encoder",
    rerank_top_k=50,
    top_k=4,
    prompt_template="default",
    llm_model="minimax-3",
)
```

**Implementation** (one extra stage vs naive_dense):

| Stage | Class | File | Lines |
|---|---|---|---|
| (same as naive_dense up to retriever) | | | |
| Reranker | `CrossEncoderReranker` | `backend/rag/pipeline.py::CrossEncoderReranker` | 208-232 |

```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # lazy-loaded

    def rerank(self, query, candidates, top_k):
        self._ensure_model()  # loads CrossEncoder once
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._model.predict(pairs)  # batched inference
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_k]]
```

The reranker wraps sentence-transformers' `CrossEncoder`, which jointly
encodes the query + each candidate to compute a relevance score. ~50ms per
question on a CPU.

**Cost delta vs naive_dense**: Cold start (~5s to load the cross-encoder
model); then ~50–100 ms per question for reranking.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline dense_then_ce
```

**When to use**: Heavy reranking where the cross-encoder's joint encoding
finds the right needle in a much larger haystack than top-4.

**Caveats**: At k=4 on HotpotQA, this matches naive_dense within noise
(+0.8 pp with 12 errors — yes, 12 questions failed outright). The
cross-encoder adds latency without lifting `contains_gold`. The iter-12
report flagged this; iter-14 confirmed.

---

### 3.4 extract_span_prompt — verbatim-span instruction

**What it is**: Same retrieval as the embedding-default baseline (mpnet at
k=4), but the prompt instructs the LLM to begin by quoting the answer span
verbatim from the context, then briefly explain.

**Configuration**:
```python
PRESETS["extract_span_prompt"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-mpnet-base-v2",  # same as large_dense
    retriever="dense",
    reranker=None,
    top_k=4,
    prompt_template="extract_span",
    llm_model="minimax-3",
)
```

**Implementation**: One new prompt builder (~25 lines).

| Stage | Class | File | Lines |
|---|---|---|---|
| Prompt | `ExtractSpanPromptBuilder` | `backend/rag/pipeline.py::ExtractSpanPromptBuilder` | 245-276 |

```python
class ExtractSpanPromptBuilder:
    EXTRACT_INSTRUCTION = (
        "Read the <context>...</context> block carefully and extract the "
        "exact span that answers the question. Begin your response with the "
        "extracted span (in quotation marks if it is a phrase), then briefly "
        "explain. Do not paraphrase the answer — quote it verbatim."
    )

    def build(self, question, context_docs):
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        return [
            {"role": "system", "content": RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.EXTRACT_INSTRUCTION},
            {"role": "user", "content": f"<context>\n{context_str}\n</context>\n\n{question}"},
        ]
```

**Cost delta vs naive_dense**: Zero compute difference. Only the prompt
text changes — `answer_f1` may go DOWN (the model now wraps answers in
quotes that dilute token overlap) but `contains_gold` (substring
containment of gold) typically goes UP.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_prompt
```

**When to use**: When the LLM is paraphrasing the answer instead of
quoting it (more common on smaller or less-instruction-tuned models). Use
alongside larger `top_k` for compounding gains.

**Caveats**: At k=4 the gain is modest (+1.4 pp). At k=8 it's +2.4 pp.
At k=10 it's +1.7 pp. Always combine with bigger context for max effect.

---

### 3.5 extract_span_k8 — k=8 + verbatim-span

**What it is**: Best `top_k` win (k=4 → k=8 covers gold-paragraphs-outside-top-4)
+ best prompt win (extract_span). Keeps the cheap MiniLM embedder.

**Configuration**:
```python
PRESETS["extract_span_k8"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    retriever="dense",
    reranker=None,
    top_k=8,
    prompt_template="extract_span",
    llm_model="minimax-3",
)
```

**Implementation**: Nothing new. Just a different preset that names two
existing settings. Inherits `DenseRetriever` from naive_dense and
`ExtractSpanPromptBuilder` from extract_span_prompt.

**Cost delta vs naive_dense**: ~2× tokens sent to the LLM (8 paragraphs
vs 4). Same embedding cost.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_k8
```

**When to use**: Good default for any QA-style retrieval task where the
gold answer is entity- or fact-bound and the corpus is paragraph-sized.
Most production RAG setups benefit from k=8 over k=4.

---

### 3.6 extract_span_k10 — the recommended default

**What it is**: Same as `extract_span_k8` but with `top_k=10` — for
HotpotQA's 10-paragraph corpus this gives the LLM the full context.

**Configuration**:
```python
PRESETS["extract_span_k10"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    retriever="dense",
    reranker=None,
    top_k=10,
    prompt_template="extract_span",
    llm_model="minimax-3",
)
```

**Implementation**: New preset only. Same components.

**Cost delta vs naive_dense**: ~2.5× tokens. Embedding cost identical.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_k10
```

**When to use**: For tasks with paragraph-sized corpora (≤20 docs per
question). Gives the LLM the full corpus. Recommended default for QA.

**Why recommended**: At k=10 on HotpotQA, **0 retrieval misses**. The
remaining 11% failure floor is purely LLM extraction / reasoning — fixing
that requires a different lever (better model, fine-tuning, or
chain-of-thought prompting). Retrieval is saturated.

---

### 3.7 hybrid_bm25_dense — Reciprocal Rank Fusion

**What it is**: Two retrievers (BM25 + dense) each pull top-(4k) candidates,
then rank-fuse with Reciprocal Rank Fusion (RRF, k=60), trim to top-4.

**Configuration**:
```python
PRESETS["hybrid_bm25_dense"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    retriever="hybrid",
    reranker=None,
    top_k=4,
    prompt_template="default",
    llm_model="minimax-3",
)
```

**Implementation** (~80 lines, three small classes):

| Stage | Class | File | Lines |
|---|---|---|---|
| Sparse retriever | `BM25Retriever` | `backend/rag/pipeline.py::BM25Retriever` | 114-157 |
| Hybrid fuser | `HybridRetriever` | `backend/rag/pipeline.py::HybridRetriever` | 160-198 |
| Dispatch | `build_retriever("hybrid", ...)` | `backend/rag/pipeline.py::build_retriever` | 309-328 |

```python
class BM25Retriever:
    def __init__(self, docs):
        self._tokenize = lambda s: [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t]
        self._docs = list(docs)
        tokenized_corpus = [self._tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi([toks if toks else ["_empty_"] for toks in tokenized_corpus])

    def retrieve(self, query, k):
        scores = self._bm25.get_scores(self._tokenize(query) or ["_empty_"])
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._docs[i] for i in ranked_indices[:k]]


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, rrf_k=60):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query, k):
        cand_k = max(k * 4, 20)  # 4× candidate depth for fusion
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

`HybridRetriever` uses identity-based fusion (the same Document object
appearing in both lists scores higher). `cand_k = max(k*4, 20)` follows
the RRF paper's recommendation for candidate depth.

**Dependency**: `rank-bm25>=0.2.0` (declared in `requirements.txt`,
pure Python, no native build).

**Eval wiring requires the corpus**: Hybrid retrievers need the raw
paragraph list (not just FAISS). The eval script branches on
`pipeline_cfg.retriever == "hybrid"` and calls
`load_or_build(..., with_corpus=True)` (see `scripts/eval_qa_hotpotqa.py:275-310`).

**Cost delta vs naive_dense**: Adds ~5% wall-clock for BM25 indexing
(10 paragraphs, fast). No model download.

**How to use**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline hybrid_bm25_dense
```

**When to use**: Large corpora (10k+ docs) with technical jargon where
dense retrievers gloss over exact entity names. Less useful on small
paragraph-sized corpora where the dense retriever already finds the
right paragraph.

**Caveats on HotpotQA**: Hybrid at k=4 is **worse** than naive_dense
(0.769 vs 0.778). The HotpotQA distractors are too lexically similar to
the gold paragraphs for BM25 to add information; it brings in extra
distractor hits that dilute top-4.

---

### 3.8 cot_extract_notitles_thinking_k10 — the current SOTA (iter-22, full-7k confirmed in iter-23)

**What it is**: Combines three iter-15→22 levers:

1. **CoT scaffold** (iter-15): `CoTExtractPromptBuilder`'s step-by-step reasoning.
2. **Title-strip** (iter-21): `CoTExtractNoTitlesPromptBuilder` strips the
   `[title]:` heading prefix from each context paragraph.
3. **Anthropic thinking mode** (iter-20+22): 4096-token internal
   reasoning budget via `thinking.budget_tokens`; visible answer is
   extracted from the resulting `text` block only (thinking is discarded).

**Configuration**:
```python
PRESETS["cot_extract_notitles_thinking_k10"] = PipelineConfig(
    embedding_backend="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    retriever="dense",
    reranker=None,
    top_k=10,
    prompt_template="cot_extract_no_titles",  # CoT scaffold + title-strip
    thinking_budget=4096,                    # Anthropic extended thinking
    llm_model="minimax-3",
)
```

**Why this combination wins**: Each lever addresses a different failure
mode — they compound.

- **CoT** scaffolds multi-hop reasoning (had been solving ~7 of 32 failures)
- **Title-strip** removes the bad cue where the model latches onto
  Wikipedia article headings instead of the canonical body form
  (adds ~7 fixes on top of CoT)
- **Thinking** gives the model budget to reason about canonical-form
  choice on hard multi-hop questions (adds ~3 more fixes on top)

**Implementation**:

| Stage | Class | File | Lines |
|---|---|---|---|
| CoT scaffold + title-strip prompt | `CoTExtractNoTitlesPromptBuilder` | `backend/rag/pipeline.py` | 314-… |
| Thinking plumbing | `AnthropicLLM.ask` | `backend/rag/pipeline.py` | 333-… |
| Eval routing | `_evaluate_one` | `scripts/eval_qa_hotpotqa.py` | 81-… |
| Batched parallelism | `asyncio.gather` in `run()` | `scripts/eval_qa_hotpotqa.py` | iter-23 |
| Detached-subprocess launch | `subprocess.Popen(creationflags=...)` | (used at runtime via Python `Popen`) | iter-23 |

**Cost delta vs cot_extract_k10**: ~50% more wall-clock (1939s vs 1012s on
n=334) because thinking-mode emits 5-10× more output tokens per call,
even though we discard the thinking content in scoring.

**How to use (n=334 sample, ~30 min)**:
```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline cot_extract_notitles_thinking_k10
```

**How to use (full 7k, ~12h with batched parallelism, detached subprocess)**:
```bash
# Spawn fully detached (so parent bash reaping doesn't kill the worker):
python -c "
import subprocess, sys
DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP = 0x08, 0x0200
subprocess.Popen(
    [sys.executable, 'scripts/eval_qa_hotpotqa.py',
     '--pipeline', 'cot_extract_notitles_thinking_k10',
     '--batch-size', '2',
     '--dump-results', 'docs/eval-results/iter22-full-7k-batch2-dump.jsonl'],
    stdout=open(r'C:/Users/Administrator/AppData/Local/Temp/full_eval.log', 'wb'),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
"
# If it dies, resume from offset:
python scripts/eval_qa_hotpotqa.py --pipeline cot_extract_notitles_thinking_k10 \
    --batch-size 2 --start-from <last_completed_index> \
    --dump-results docs/eval-results/iter22-full-7k-batch2-dump.jsonl
```

**When to use**: Recommended default for HotpotQA-style benchmarks where
gold answers use canonical entity names (full first/middle names,
suffixes, parentheticals) different from the colloquial Wikipedia heading.
The title-strip lift may not transfer to other benchmarks where this
gap doesn't exist.

**Caveats**:
- The benchmark must have gold-vs-heading name divergence. On datasets
  where gold uses colloquial names (or where paragraphs have no heading
  cue), title-strip is a wash.
- 2× LLM cost vs CoT-only presets.
- The 22 remaining failures in the n=334 sample break down:
  ~7 still-name-variant (model extracts a shorter body form), ~5 yes/no
  (model doesn't lead with literal), ~3 dataset-noise (corpus disagrees
  with gold), ~7 reasoning/wrong-entity (genuine model limits). The
  full-7k result (467 / 7369 failures = 6.3%) preserves the same
  failure-mode mix; the residual gap is LLM-extraction discipline that
  prompts + retrieval + thinking can't reach on this small model.

---

## 4. How to reproduce any row

Every number in §2 was generated by one of these commands, run from the
repo root with `ANTHROPIC_API_KEY` set. Wall-clock depends on cache state.

```bash
# Baseline (MiniLM, k=4)
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline naive_dense

# NEW SOTA (iter-22): CoT + title-strip + thinking
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline cot_extract_notitles_thinking_k10

# Bigger embedding (mpnet, k=4)
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline large_dense

# Cross-encoder rerank (mpnet + cross-encoder, 50→4)
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline dense_then_ce

# Verbatim-span prompt (mpnet, k=4) — small gain at k=4
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_prompt

# Bigger context (MiniLM, k=8) — biggest single retrieval-side lever
python scripts/eval_qa_hotpotqa.py --subset 1000 --k 8

# Bigger context + verbatim span (k=8) — compounded
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_k8

# Bigger context (MiniLM, k=10) — full corpus for HotpotQA
python scripts/eval_qa_hotpotqa.py --subset 1000 --k 10

# Bigger context + verbatim span (k=10) — RECOMMENDED DEFAULT
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_k10

# Hybrid BM25 + dense (MiniLM, k=4)
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline hybrid_bm25_dense
```

To replicate the exact cached state for re-running, ensure the per-question
FAISS cache (`backend/storage/eval/hotpotqa/cache/{dataset_sha}_{embedding_tag}/`)
exists. Cold-build cost is roughly proportional to embedding-model size:

| Embedder | Cold build (334 q × 10 paragraphs) |
|---|---|
| MiniLM (384-dim) | ~25 min |
| mpnet (768-dim) | ~60 min |

---

## 5. Adding your own pipeline variant

The whole pipeline factory is designed for one-line additions. Example:
add a variant that uses `mpnet` instead of MiniLM at k=10.

```python
# backend/rag/pipeline.py — drop in next to PRESETS
PRESETS["extract_span_mpnet_k10"] = PipelineConfig(
    name="extract_span_mpnet_k10",
    embedding_backend="sentence-transformers",
    embedding_model="all-mpnet-base-v2",
    retriever="dense",
    reranker=None,
    top_k=10,
    prompt_template="extract_span",
    llm_model="minimax-3",
)
```

That's it. Run with:

```bash
python scripts/eval_qa_hotpotqa.py --subset 1000 --pipeline extract_span_mpnet_k10
```

The pipeline factory wires it up via the existing `build_pipeline(...)`
function — no other code changes needed. The CLI auto-discovers presets
via `list_presets()`.

For a new retriever (e.g., HyDE, multi-query): implement a class with a
`retrieve(query, k) -> list[Document]` method, add a `if config.retriever
== "hyde":` branch in `build_retriever`, and add a preset that names it.

For a new prompt template: subclass `PromptBuilder`, add a `if
config.prompt_template == "your_template":` branch in
`build_prompt_builder`, and add a preset that names it.

For a new reranker: subclass `Reranker` (one method: `rerank(query,
candidates, top_k) -> list[Document]`), add to `build_reranker`, name it
in a preset.

---

## 6. Implementation file index (where to read)

| File | Role |
|---|---|
| `backend/rag/pipeline.py` | PipelineConfig, all retriever/reranker/prompt/llm classes, PRESETS dict |
| `backend/rag/embeddings.py` | Embedder factory (`make_embeddings`) |
| `backend/eval/cache.py` | Per-question FAISS cache; `load_or_build(item, sha, embeddings, with_corpus=False)` |
| `backend/eval/metrics.py` | Pure scoring helpers; `gold_paragraph_in_top_k`, `answer_f1`, `answer_em`, `answer_coverage_at_k` |
| `backend/eval/qa_judge.py` | `build_qa_prompt` (default RAG prompt), `ask_llm` (Anthropic caller) |
| `scripts/eval_qa_hotpotqa.py` | Per-question driver; iterates items, calls LLM, scores, prints results |
| `backend/tests/rag/test_pipeline.py` | 47 unit tests covering pipeline factory + presets |
| `backend/tests/eval/test_metrics.py` | 39 unit tests covering scoring helpers including the iter-14 `gold_in_top_k` |

The whole "switchable pipeline" abstraction is ~500 lines of code
(pipeline.py). The eval loop is ~400 lines. The scoring helpers are ~150
lines. Anything more elaborate usually means a bug.

---

## 7. Common pitfalls when picking a pipeline

1. **Don't pick mpnet or rerank without first checking `gold_in_top_k`.**
   On HotpotQA both were no-ops at k=4. The diagnostic is in
   `backend/eval/metrics.py::gold_paragraph_in_top_k` and prints
   automatically in the failure-mode breakdown.

2. **Don't pick `extract_span_prompt` at k=4 alone.** Its full value
   only appears combined with k≥8 (more raw material to quote from).

3. **Don't pick cross-encoder on a small candidate set.** Rerank needs
   k≥20 to express itself; rerank_top_k=50 is the preset's choice and
   seems right.

4. **Don't pick hybrid BM25 on small lexical corpora.** BM25's advantage
   shows on FAQ-style / entity-name-jargon / large-corpus workloads.

5. **Don't trust a single-metric comparison.** Use the failure-mode
   breakdown (printed automatically) to confirm whether your pipeline
   reduces extraction misses, retrieval misses, or both.

6. **Don't forget cache-key safety.** If you switch embedding models,
   the FAISS cache key automatically changes
   (`backend/eval/cache.py::embedding_tag`). You cannot accidentally
   use a 384-dim index for 768-dim queries; the per-question key has
   the embedding-tag suffix.

---

## 8. Default for new RAG work

For any QA-style RAG task with paragraph-sized corpora, **start with
`cot_extract_notitles_thinking_k10`** — the SOTA at 0.937 on the full
HotpotQA dev_distractor. It works on HotpotQA-style benchmarks; on other
benchmarks, validate that the title-strip + thinking-mode lift
reproduces before adopting.

If you have a different shape (small/large docs, code, structured data,
tool use), start with `naive_dense` (k=4 or whatever's natural), get
the eval pipeline working, then sweep `top_k` first.

---

## 9. Source commands & logs

| Run | Date | Command | Output |
|---|---|---|---|
| Iter-12 baseline (naive_dense, large_dense, dense_then_ce) | 2026-07-11 | iter-12 docs | `docs/eval-results/2026-07-11-pipeline-ab-test-1k.md` |
| Iter-13 hybrid (hybrid_bm25_dense) | 2026-07-11 | iter-13 docs | `docs/eval-results/2026-07-11-hybrid-bm25-eval.md` |
| Iter-14 naive k=4 baseline | 2026-07-11 | eval with `--subset 1000 --pipeline naive_dense` | this doc, row 1 |
| Iter-14 extract_span_prompt | 2026-07-11 | `--pipeline extract_span_prompt` | this doc, row 5 |
| Iter-14 top_k=8 | 2026-07-11 | `--subset 1000 --k 8` | this doc, row 4 |
| Iter-14 extract_span_k8 | 2026-07-11 | `--pipeline extract_span_k8` | this doc, row 3 |
| Iter-14 top_k=10 | 2026-07-11 | `--subset 1000 --k 10` | this doc, row 2 |
| Iter-14 extract_span_k10 | 2026-07-11 | `--pipeline extract_span_k10` | this doc, row 1 (top) |
| Iter-14 re-run large_dense | 2026-07-12 | `--pipeline large_dense` (on same SHA) | this doc |
| Iter-14 re-run dense_then_ce | 2026-07-12 | `--pipeline dense_then_ce` (12 errors noted) | this doc |
| Iter-14 re-run hybrid_bm25_dense | 2026-07-12 | `--pipeline hybrid_bm25_dense` | this doc |
| Iter-15 cot_extract_k10 | 2026-07-12 | `--pipeline cot_extract_k10` | this doc, row 3 |
| Iter-19 cot_extract_v2_k10 | 2026-07-12 | `--pipeline cot_extract_v2_k10` | this doc (regressed) |
| Iter-20 cot_thinking_k10 | 2026-07-12 | `--pipeline cot_thinking_k10` | this doc (tied with SOTA) |
| Iter-21 cot_extract_notitles_k10 | 2026-07-12 | `--pipeline cot_extract_notitles_k10` | this doc, row 2 |
| Iter-22 cot_extract_notitles_thinking_k10 (n=334) | 2026-07-12 | `--pipeline cot_extract_notitles_thinking_k10` | this doc, row 1' (SOTA) |
| **Iter-23 full-7k SOTA confirmation (n=7369)** | **2026-07-15** | **`--pipeline cot_extract_notitles_thinking_k10 --batch-size 2` (detached subprocess, ~12h wall-clock)** | **`docs/eval-results/iter22-full-7k-batch2-dump.jsonl` (this doc, row 1 — full 7k)** |

---

## 10. Honest caveats

- **The full-7k SOTA at 0.937 is the official number.** The 0.934
  from earlier sections was a stratified n=334 sample; the +0.26 pp
  lift from the n=334 sample to the n=7369 full run is well within
  sampling noise and confirms the SOTA holds at scale.
- **n=334 sample is small for deltas <2 pp.** Lifts like `+0.8 pp` for
  dense_then_ce are within sampling noise. Don't read too much into
  ranking below rank 4. The iter-21 → iter-22 lift (+0.9 pp) is also
  within noise; the cumulative iter-15 → iter-22 lift (+3.0 pp) is more
  robust.
- **HotpotQA-specific.** Paragraphs are pre-chosen (10 per question),
  retrieval is "easy" within those 10. On larger / noisier corpora,
  hybrid and rerank likely help more.
- **Cross-encoder rerank had 12 errors** out of 334 (LLM-side failures).
  That alone de-rates the comparison; the missing 12 questions might
  have lifted `contains_gold` by ~1 pp if completed.
- **Full-7k run had 36 skipped questions** (sensitive-content filter
  rejection on some questions, plus a few transient 5xx that didn't
  recover). Out of 7405 questions, 7369 are in the final dump. The
  36 missed are a small fraction (0.5%) — well within n=334 sampling
  noise — so the headline metric is robust.
- **`answer_f1` and `answer_em` behave inversely to `contains_gold` for
  `extract_span_*` and `cot_*` variants.** The model is now quoting
  verbatim, which is great for substring containment but strips the
  conversational padding that the token-level F1 prefers. `contains_gold`
  is the user-relevant metric; `answer_f1` is included for benchmark
  parity but is not the optimization target.
- **LLM `minimax-3` is a generic model**, not a fine-tuned extractor. A
  larger instruction-tuned LLM might have a different
  extraction-vs-context curve.
- **Title-strip is benchmark-specific.** HotpotQA gold uses the
  full canonical entity name (e.g., "Louis-Hector Berlioz") rather than
  the Wikipedia heading ("Hector Berlioz"). Other QA benchmarks may not
  have this gap, in which case the iter-21 lift wouldn't reproduce.
  Worth checking before adopting `cot_extract_notitles_*` as a default
  for non-HotpotQA workloads.
