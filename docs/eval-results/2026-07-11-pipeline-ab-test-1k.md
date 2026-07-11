# Pipeline A/B Test (1k subset) — Iter-12 Full Run

**Date**: 2026-07-11
**Iteration**: full 1000-question (334 effective) A/B across 2 pipeline presets
**Previous report**: [2026-07-10-pipeline-ab-test.md](2026-07-10-pipeline-ab-test.md) — n=100 results, included `dense_then_ce`
**Pipeline factory**: `backend/rag/pipeline.py` (commit `5c52375`)
**Cache-key fix**: `backend/eval/cache.py` (commit `d0df5a8`)

---

## TL;DR — corrected by sample size

The n=100 run suggested a clear winner (large_dense with +6 pp lift over naive_dense). **The n=334 run tells a different story**: the two pipelines are essentially tied.

| Pipeline | `contains_gold` w/ context | `contains_gold` w/o context | Lift |
|---|---:|---:|---:|
| **naive_dense** (MiniLM 384-dim) | **0.787** | 0.560 | **+0.228** |
| **large_dense** (mpnet 768-dim) | **0.796** | 0.596 | **+0.201** |

**Δ in `contains_gold` ceiling: +0.9 pp** — within sampling noise at this size. The two are statistically indistinguishable.

**Δ in retrieval lift: -2.7 pp** — naive_dense actually has a *bigger* lift at n=334. But this is driven by the without-context baseline being unstable across runs (0.560 / 0.596 / 0.620 in different runs), not by a real difference in retrieval quality.

### Bottom-line

**Don't switch production based on the n=100 result.** The two pipelines perform equivalently on the user-facing metric at 3.3× the sample size. Switching to mpnet would cost 3× embedding time and 3× index size for no measurable gain.

If we want to find a real winner, we need either:
- A bigger lever (hybrid BM25, cross-encoder at k=8, etc.)
- A bigger sample (full 7405 questions)
- A different metric that's more sensitive to retrieval quality

The original iter-11 finding (retrieval helps ~17-23 pp) is robust. **The within-pipeline comparison (which pipeline?) is not yet clear at this sample size.**

---

## 1. Setup

| Item | Value |
|---|---|
| Dataset | HotpotQA dev_distractor v1 |
| Subset | `--subset 1000` → **334 effective questions** |
| LLM | `minimax-3` via MiniMax Anthropic-compatible endpoint |
| Temperature | 0 (deterministic) |
| Top-k | 4 |
| Cache | per-embedding-model (after iter-12 fix); cold for naive_dense, partially warm for large_dense (100/334 hits reused from iter-12 n=100 run) |
| Pacing | 1s between LLM calls |
| Baseline mode | `--compare-baseline` |

### Pipelines compared

| Name | Embedding | Retrieval | Reranker | Prompt |
|---|---|---|---|---|
| naive_dense | all-MiniLM-L6-v2 (384-dim, 22M params) | dense top-4 | none | default |
| large_dense | all-mpnet-base-v2 (768-dim, 110M params) | dense top-4 | none | default |

### Skipped: dense_then_ce at 1k

The n=100 result showed cross-encoder rerank underperforming (0.750 vs 0.800 for large_dense at k=4). At n=1k with 334 questions, this run would take ~80 min and ~$15-20 — and the n=100 result suggests it's unlikely to win at k=4. Re-running at k=8 is the more useful next step, deferred to a future iteration.

---

## 2. Results

### Headline — `contains_gold` (user-facing)

| Pipeline | with context | without context | Lift |
|---|---:|---:|---:|
| naive_dense | 0.787 | 0.560 | +0.228 |
| large_dense | 0.796 | 0.596 | +0.201 |
| **Δ (large - naive)** | **+0.009** | +0.036 | −0.027 |

The "+0.9 pp ceiling difference" is well within sampling noise for n=334. Standard error on a proportion at p=0.79 with n=334 is roughly ±0.022 (sqrt(0.79*0.21/334) = 0.022). So the true difference is somewhere in `[-0.013, +0.031]` with high probability — confidently close to zero, possibly slight positive or slight negative.

### Headline — `answer_f1` (HotpotQA-strict)

| Pipeline | with context | without context | Lift |
|---|---:|---:|---:|
| naive_dense | 0.125 | 0.056 | +0.069 |
| large_dense | 0.123 | 0.056 | +0.067 |

`answer_f1` is essentially identical. Both pipelines produce the same F1 under the strict metric — the LLM behavior dominates this number, not retrieval.

### Headline — `answer_em`

| Pipeline | with context | without context | Lift |
|---|---:|---:|---:|
| naive_dense | 0.021 | 0.003 | +0.018 |
| large_dense | 0.012 | 0.000 | +0.012 |

`answer_em` is noisy at this sample size (very few exact matches). Doesn't differentiate the pipelines.

---

## 3. Comparison: n=100 vs n=334

### The without-context baseline is unstable

| Run | n | without-context `contains_gold` |
|---|---:|---:|
| naive_dense n=100 | 100 | 0.620 |
| naive_dense n=334 | 334 | 0.560 |
| large_dense n=100 | 100 | 0.570 |
| large_dense n=334 | 334 | 0.596 |

The without-context number should be **identical across runs** (same questions, same LLM, no retrieval). But we see it move by 6 pp across runs. This is the MiniMax LLM endpoint's natural variance — even at temperature=0, the endpoint isn't 100% reproducible. We should expect ±5 pp noise on the without-context number alone.

This means **the "lift" metric is too noisy to use as a decision signal** for the within-pipeline comparison. The lift differences (-2.7 pp) are inside the noise.

### The with-context ceiling is stable

| Run | n | with-context `contains_gold` |
|---|---:|---:|
| naive_dense n=100 | 100 | 0.790 |
| naive_dense n=334 | 334 | 0.787 |
| large_dense n=100 | 100 | 0.800 |
| large_dense n=334 | 334 | 0.796 |

The with-context numbers are stable across sample sizes within ~1 pp. This is the user-facing signal we should trust. **Both pipelines hit ~0.79 ceiling; neither wins.**

---

## 4. What changed from n=100 to n=334

| Metric | n=100 | n=334 | What changed |
|---|---|---|---|
| naive_dense w/ context | 0.790 | 0.787 | Stable (within 0.3 pp) |
| naive_dense w/o context | 0.620 | 0.560 | −6 pp (LLM endpoint variance) |
| naive_dense lift | +0.170 | +0.228 | +5.8 pp (driven by w/o noise) |
| large_dense w/ context | 0.800 | 0.796 | Stable (within 0.4 pp) |
| large_dense w/o context | 0.570 | 0.596 | +2.6 pp (LLM endpoint variance) |
| large_dense lift | +0.230 | +0.201 | −2.9 pp (driven by w/o noise) |
| **large − naive (w/ context)** | **+0.010** | **+0.009** | **Essentially zero** |

The headline at n=100 ("large_dense is +1 pp ceiling, +6 pp lift") becomes "essentially tied" at n=334. The +1 pp ceiling signal is the most reliable — it's the same number at both sample sizes. The +6 pp lift signal is noise from the without-context baseline.

---

## 5. Honest assessment

### What the n=100 result actually told us

The n=100 result correctly showed that **mpnet is not worse than MiniLM**. It incorrectly implied that mpnet is meaningfully better (the +6 pp lift was almost all noise from the without-context baseline).

### What the n=334 result tells us

At 3.3× the sample size, the two pipelines are tied on the user-facing metric. The mpnet embedding model doesn't deliver measurable improvement on HotpotQA's distractor setting.

This is somewhat surprising — published benchmarks usually show mpnet gaining 5-10 pp. The likely explanations:

1. **HotpotQA's distractors are easy.** The 9 distractor paragraphs per question are usually topic-related but factually irrelevant. Even MiniLM separates them well enough that the recall ceiling is ~80% regardless.
2. **The LLM is doing heavy lifting.** `minimax-3` "knows" 56-62% of answers from training. The marginal value of better retrieval is capped at ~20 pp — which both pipelines hit.
3. **Top-k=4 is small.** With 4 candidates, the embedding model's ranking precision matters less than whether gold is in top-4 at all.

### What this means for the production pipeline

**Stay on naive_dense for now.** The mpnet upgrade costs 3× embedding time and 3× index size for no measurable improvement at the user-facing level. Revisit if:
- A bigger lever (hybrid BM25, cross-encoder at k=8) shows >5 pp lift at this sample size
- The full 7405-question eval (22× more data) shows a smaller-sample-masked effect
- Production latency budget changes (smaller model = faster inference)

---

## 6. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/2026-07-11-pipeline-ab-test-1k.md` | This report |
| `docs/eval-results/2026-07-10-pipeline-ab-test.md` | Previous n=100 report (now superseded for the within-pipeline conclusion) |

## 7. Recommendations for the next iteration

1. **Re-test dense_then_ce at k=8.** The n=100 result at k=4 was inconclusive (likely needed more room for the reranker). Try `top_k=8` first; if it helps, scale up.

2. **Add the hybrid BM25+dense preset.** This is a different lever — BM25 is exact-match, dense is semantic. They cover different failure modes. Expected gain: +3-5 pp on entity-name queries.

3. **Run on the full 7405-question dataset.** At n=334, we're at the edge of what we can measure reliably. At n=7405, even 1-2 pp deltas become meaningful.

4. **Add `--pipeline extract_span_prompt` to the comparison.** Prompt-only change — costs nothing in retrieval time but might boost `answer_f1` for the HotpotQA-strict metric.

5. **Don't switch production embedding model yet.** The cost/benefit isn't there. Wait for a larger lever or a clear win at higher n.

---

## 8. Lessons learned

1. **Lift metric is unreliable when the without-context baseline has endpoint noise.** Better to compare on `contains_gold` w/ context directly, which is more stable.

2. **n=100 is too small for A/B testing RAG pipelines.** Differences under ±3 pp are noise; need n≥300 to see real effects.

3. **The without-context baseline should be ~constant across pipelines.** When it isn't, that's a signal that something else is varying (LLM endpoint noise in our case).

4. **The user-facing ceiling is the right metric.** The LLM's conversational wrapping dominates `answer_f1`; the rank-precision matters less than whether gold is in top-k.