# Iter-25 — MultiHop-RAG Cross-Dataset Validation

**Date**: 2026-07-16
**Iteration**: Cross-dataset validation of the iter-22 / iter-23 SOTA on a second RAG benchmark
**Goal**: Test whether the iter-23 SOTA (`cot_extract_notitles_thinking_k10` at 0.937 on HotpotQA) generalizes to MultiHop-RAG, a different RAG benchmark with different question styles

---

## TL;DR — SOTA generalizes, with a much larger lift on harder multi-hop questions

| Dataset | n | SOTA (iter-23) | Baseline (extract_span_k10) | SOTA lift |
|---|---:|---:|---:|---:|
| HotpotQA n=334 sample | 334 | **0.937** (full 7k) / 0.934 (n=334) | 0.889 | +4.5 pp |
| **MultiHop-RAG n=100 (overall)** | 100 (90 completed) | **0.756** | 0.589 | +16.7 pp |
| **MultiHop-RAG n=100 (non-null)** | 100 (73 completed) | **0.932** | 0.726 | **+20.6 pp** |

**The SOTA pipeline transfers cleanly.** Excluding MultiHop-RAG's unanswerable null queries (a category HotpotQA doesn't have), the SOTA hits **0.932 contains_gold** — within sampling noise of the HotpotQA n=334 result. More importantly, the **SOTA-vs-baseline lift is 4-5× larger on MultiHop-RAG** than on HotpotQA, indicating the levers (CoT scaffold + title-strip + thinking mode) compound more on harder temporal/comparison questions.

---

## 1. Setup

### Dataset

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (Tang & Yang, COLM 2024, [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)) is a multi-hop QA dataset over 609 news articles. It ships 2,556 queries across 4 question types:

| Type | Count | Description |
|---|---:|---|
| `inference_query` | 816 | "Who is the individual associated with the cryptocurrency industry...?" — entity + fact linkage |
| `comparison_query` | 856 | "Compare X and Y on metric Z" — multi-entity comparison |
| `temporal_query` | 583 | "Which company was acquired first, A or B?" — time-ordered reasoning |
| `null_query` | 301 | "Insufficient information." — designed to be unanswerable from the corpus |

Each query has 2-4 evidence documents (or 0 for null queries). The corpus is news articles from 49 outlets (Mashable, The Verge, TechCrunch, NYT, etc.) across 6 categories (business, science, health, entertainment, technology, sports). Body length is ~7-10K characters per article — much longer than HotpotQA's 1-2K-character Wikipedia paragraphs.

### Adapter

`scripts/ingest_multihop_rag.py` maps MultiHop-RAG records to our `HotpotQaItem` shape so the existing eval pipeline works without modification:

| MultiHop-RAG field | HotpotQaItem field | Notes |
|---|---|---|
| `query` | `question` | Direct |
| `answer` | `answer` | Direct |
| `question_type` | `type` | Mapped: inference/comparison/temporal/null |
| `len(evidence_list)` | `level` | Mapped: 0→hard, 2→easy, 3→medium, 4→hard |
| `evidence_list[].title` (joined with distractor titles) | `context` | 10 paragraphs per question: 2-4 gold + 6-8 sampled distractor |
| `evidence_list[].title` | `supporting_facts` | Title-only; sentence index 0 (the `gold_paragraph_in_top_k` metric only checks title membership) |

Distractor sampling: random from the 609-doc corpus excluding the evidence titles, deterministic via `random.Random(seed)`. The "10 paragraphs per question" setting matches HotpotQA's distractor count, so the per-question FAISS cache size is comparable.

Stratified sampling by `question_type` (4 buckets), `random.Random(42)`, per-bucket cap `min(ceil(n/4), len(bucket))` — mirrors `backend.eval.hotpotqa.sample` for cross-dataset consistency.

### Eval command

```bash
# SOTA preset
python scripts/eval_qa_hotpotqa.py --subset 100 \
    --fixture scripts/.cache/multihop_rag_fixture_100.json \
    --pipeline cot_extract_notitles_thinking_k10

# Baseline preset (for comparison)
python scripts/eval_qa_hotpotqa.py --subset 100 \
    --fixture scripts/.cache/multihop_rag_fixture_100.json \
    --pipeline extract_span_k10
```

Same LLM, same temperature, same pacing, same prompt-template logic — only the preset name changes.

---

## 2. Headline results (n=100, 90 completed)

The eval CLI reported 10 questions not processed (8 null + 2 comparison). All 10 were the same questions skipped across both presets — consistent with the iter-23 observation that the API's sensitive-content filter rejects certain questions and the comparison prompt is a forced-extract that can't say "I don't know" for null queries. See §4 for details.

| Preset | n | contains_gold | extraction miss | retrieval miss | Wall-clock |
|---|---:|---:|---:|---:|---:|
| **SOTA** (cot_extract_notitles_thinking_k10) | 90 | **0.756** | 22 (24.4%) | 0 (0%) | 1135s |
| Baseline (extract_span_k10) | 90 | 0.589 | 37 (41.1%) | 0 (0%) | 446s |

Both runs hit **0 retrieval misses** (all gold paragraphs are in the per-question context by construction — this is the same as HotpotQA's k=10 setting). The story is purely **extraction**: the SOTA's CoT + title-strip + thinking cuts the extraction-miss rate from 41.1% to 24.4% (-16.7 pp).

Wall-clock: SOTA is 2.5× slower because the thinking-mode preset emits 5-10× more output tokens per call.

---

## 3. Breakdown by question type

The 8 null queries always fail (the prompt forces extract-the-span; "Insufficient information." is a refusal-shaped answer, not a span). Excluding null queries gives a clean apples-to-apples comparison:

| Type | n (SOTA) | SOTA | Baseline | Δ |
|---|---:|---:|---:|---:|
| `inference` | 25 | **1.000** | **1.000** | 0 (tied at ceiling) |
| `comparison` | 23 | **0.870** | 0.652 | +21.8 pp |
| `temporal` | 25 | **0.920** | 0.520 | **+40.0 pp** |
| `null` | 17 | 0.000 | 0.000 | 0 (unanswerable by design) |
| **Non-null total** | **73** | **0.932** | 0.726 | **+20.6 pp** |

**Three findings**:

1. **`inference` is at ceiling for both presets** (25/25 = 1.000). The model is already extracting entity + fact linkages correctly with the simpler prompt. The SOTA's extra reasoning budget has no headroom to add value.

2. **`temporal` sees the biggest lift** (+40 pp). Temporal questions require the model to identify time-related facts and order them — exactly the kind of multi-step reasoning that CoT scaffold + thinking mode are designed for. The baseline struggles because `extract_span` gives the model no structure to work through the ordering.

3. **`comparison` sees a substantial lift** (+22 pp). Multi-entity comparison requires tracking which entity each fact belongs to — the CoT scaffold's "identify the entities and facts the question asks about" step is exactly the disambiguation primitive that the baseline lacks.

---

## 4. Caveats and what didn't work

### 4.1 Null queries fail by design (n=17, all fail)

`null_query` questions have empty `evidence_list` and gold answer "Insufficient information." The SOTA prompt's "begin your response with the extracted span (in quotation marks)" directive forces the model to either:
- Refuse with text that doesn't contain the gold string "Insufficient information." → `contains_gold = 0`
- Output "I don't know" or similar → `contains_gold = 0`

**To handle null queries, the prompt would need to allow an "I don't know" path** — but that's a different design question (and one that hurts the verbatim-extraction discipline that drives HotpotQA's gain). Out of scope for iter-25.

### 4.2 Sensitive-content filter rejections (n=10 across both presets)

The 10 questions not processed by the CLI (8 null + 2 comparison) are consistent across SOTA and baseline — i.e., they're filtered by the LLM API's content safety filter, not by the preset. Iter-23 saw the same pattern (36 / 7405 HotpotQA questions rejected). The CLI's per-item exception handler logs WARNING but the `errors` counter only catches setup errors (a separate bug to fix in a later iteration).

### 4.3 The "non-null" denominator is the right frame

Reporting the 0.756 number without the null-query context is misleading. The non-null 0.932 is the more meaningful comparison to HotpotQA's 0.937 — the two are now within sampling noise (n=73 vs n=7369 of course, but the n=100 stratified sample preserves type distribution).

### 4.4 FAISS cache hits for the baseline run

The baseline run's 90/0 cache hit/build count is the SOTA run's FAISS index reused — the dataset_sha is identical (same fixture file), and the embedding model is the same (MiniLM). This is the expected behavior of the iter-12+ cache-key fix.

---

## 5. Cross-dataset generalization

| Preset | HotpotQA n=334 sample | HotpotQA n=7369 full | MultiHop-RAG n=100 (non-null) | Cross-dataset gap |
|---|---:|---:|---:|---:|
| SOTA | 0.934 | 0.937 | 0.932 | -0.5 pp |
| extract_span_k10 | 0.889 | — | 0.726 | -16.3 pp |
| SOTA lift over extract_span_k10 | +4.5 pp | — | +20.6 pp | **+16.1 pp** |

The **SOTA pipeline transfers cleanly** — the 0.937 → 0.932 cross-dataset gap is within sampling noise (n=73 has SE ~3 pp). The SOTA's prompt-and-thinking architecture isn't overfit to HotpotQA's specific question style.

The **SOTA's lift over the baseline is 4.5× larger on MultiHop-RAG** than on HotpotQA. This is the more interesting finding: the levers (CoT scaffold + title-strip + thinking mode) compound more on harder multi-hop questions. On HotpotQA's already-easy bridge/comparison questions, most of the gain is from retrieval saturation at k=8; on MultiHop-RAG's harder temporal/comparison questions, the model needs the extra reasoning budget to do the ordering / disambiguation correctly.

### Implication for production RAG

If you're building a RAG system on harder multi-hop content (e.g., news articles, legal documents, medical records), the iter-23 SOTA pipeline's lift is likely **larger** than the +4.5 pp measured on HotpotQA would suggest. If your content is short, fact-bound, and similar in difficulty to HotpotQA, the lift will be closer to the +4.5 pp.

---

## 6. Implementation notes

### 6.1 Why per-question context (not open-domain)

MultiHop-RAG's natural evaluation is open-domain retrieval over the 609-doc corpus — the system must retrieve the right 2-4 evidence docs from 605 distractors. But our existing `scripts/eval_qa_hotpotqa.py` is built around the HotpotQA "10 paragraphs per question" model: it pre-builds a per-question FAISS index from `item.context` and only tests answer quality given the context.

To reuse the pipeline without modification, the adapter pre-samples 6-8 distractor docs per question and folds them into the 10-paragraph context. This tests the same extraction/reasoning pipeline (the iter-22 levers) on MultiHop-RAG's question styles — but it does **not** test open-domain retrieval. A true open-domain eval would require a separate `RagService`-style path that searches the full corpus.

### 6.2 Why n=100, not n=334 or full

n=100 × 4 question types = 25 per type, which is enough to see per-type trends. The full n=2556 would be ~3 hours per preset at the SOTA's 19s/question rate; n=100 is enough to validate the headline. If iter-25 results are interesting, the next iteration can scale to n=334 or full with `batch_size=2`.

### 6.3 Why `extract_span_k10` as the baseline

It's the iter-14 SOTA — the simpler "verbatim span extraction" prompt that was the best non-CoT result on HotpotQA. It also matches the k=10 setting that the SOTA uses, so the only thing that differs is the prompt template (not the retrieval setup). This isolates the **prompt-and-thinking** contribution of the SOTA from the **k=10 retrieval-saturation** contribution.

---

## 7. Files produced

| Path | Contents |
|---|---|
| `scripts/ingest_multihop_rag.py` | Adapter: MultiHop-RAG → HotpotQaItem-shaped fixture |
| `scripts/.cache/multihop_rag/corpus.json` | Original 609-doc corpus (copied from upstream) |
| `scripts/.cache/multihop_rag/MultiHopRAG.json` | Original 2,556-query QA file (copied from upstream) |
| `scripts/.cache/multihop_rag_fixture_100.json` | 100-query stratified fixture (25 per type, 90 completed) |
| `scripts/.cache/multihop_rag_fixture_334.json` | 334-query stratified fixture (84 per type, not yet run) |
| `docs/eval-results/iter25-multihop-rag-sota-k10-dump.jsonl` | Per-question SOTA results (90 records) |
| `docs/eval-results/iter25-multihop-rag-sota-k10.log` | SOTA eval log |
| `docs/eval-results/iter25-multihop-rag-baseline-k10-dump.jsonl` | Per-question baseline results (90 records) |
| `docs/eval-results/iter25-multihop-rag-baseline-k10.log` | Baseline eval log |
| `docs/eval-results/2026-07-16-iter25-multihop-rag-cross-dataset.md` | This report |

---

## 8. Recommended next steps

1. **Scale to n=334** (or full n=2556) on the SOTA preset to get a tighter confidence interval on the 0.932 non-null result. Expected wall-clock: ~6 hours for n=334 with `batch_size=2`.
2. **Track B** (build a small heterogeneous-format corpus with actual PDF/DOCX/HTML files) — exercises the loader pipeline directly, which the n=100 fixture doesn't do.
3. **Add an "I don't know" path** to the SOTA prompt for null queries — likely hurts HotpotQA's gain but unlocks MultiHop-RAG's null_query category. Worth a separate iteration to measure the trade-off.
4. **Fix the per-item error counter** in `scripts/eval_qa_hotpotqa.py` — the `errors` variable is initialized but never incremented, so per-item failures only show up as missing qids in the dump.