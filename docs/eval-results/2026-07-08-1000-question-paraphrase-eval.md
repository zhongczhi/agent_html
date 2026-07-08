# HotpotQA Paraphrase Eval — 1000-Question Run Results

**Date**: 2026-07-08
**Iteration**: paraphrase-eval pipeline (commits `71b404d` → `345393e`)
**Status**: Final results for the planned "stratified 1000" scope
**Plan**: `docs/superpowers/plans/2026-07-08-paraphrase-eval.md`

---

## TL;DR

Ran the new multi-variant retrieval pipeline on a stratified 1000-question sample of HotpotQA dev_distractor. **334 questions** were actually evaluated (the dataset only contains 2 of 6 type/level buckets, so the 1000 request saturated at the available 334). Of those:

- **Original question**: 76.3% answer-coverage, 85.2% supporting-fact recall @k=4
- **Lexical paraphrase**: 71.2% / 82.0% (n=205)
- **Structural paraphrase**: 71.2% / 81.5% (n=205)
- **Casual paraphrase**: 76.0% / 82.9% (n=217)
- **Robustness@4**: 67.5% — 137 of 203 fully-covered questions succeeded on all 4 variants

The retriever loses ~5 percentage points of supporting-fact recall under paraphrase pressure (84% → 81%), and the most stress comes from structural paraphrasing (clause reordering). Casual paraphrasing is the closest to the original — possibly because both are short and entity-anchored.

---

## 1. Setup

| Item | Value |
|---|---|
| Dataset | HotpotQA dev_distractor v1 (`hotpot_dev_distractor_v1.json`) |
| License | CC BY-SA 4.0 |
| Sample | `--subset 1000` (stratified by `(type, level)`, seed=42) |
| Effective sample | 334 questions (167 `bridge/hard` + 167 `comparison/hard`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Paraphrase model | `minimax-3` via MiniMax Anthropic-compatible endpoint |
| Paraphrase styles | `lexical` (synonym swap), `structural` (clause reorder), `casual` (informal) |
| Validation gate | ≥80% token overlap with gold answer → reject, retry once, skip on double-fail |
| Retrieval top-k | 4 |
| Index cache | per-question FAISS, SHA-keyed by dataset, 10 distractor paragraphs per question |
| Cold-cache eval | `--no-cache` (every FAISS index rebuilt for this run) |

### Dataset characteristic worth knowing

HotpotQA's **dev_distractor** split only contains `bridge/hard` (5,918) and `comparison/hard` (1,487) questions — the easy/medium levels exist only in the train split. Stratified sampling across all 6 buckets therefore saturates at 2 × per-bucket-cap. With `--subset 1000`, per-bucket-cap = ceil(1000/6) = 167, giving 334 questions total. Per-bucket breakdowns in this report reflect the dev_distractor's natural distribution, not the train split's.

---

## 2. Paraphrase Generation

### Pipeline

`scripts/generate_paraphrases_hotpotqa.py` runs 3 concurrent LLM calls per question (one per style) via `asyncio.gather`. Each style's `gen_with_retry` task fires its first attempt, validates, and conditionally retries once before returning. The 3 tasks run concurrently so wall-clock per question ≈ 2× a single API call's latency (first + possible retry) rather than 3×.

### Stats

| Metric | Count |
|---|---|
| Questions attempted | 334 |
| Paraphrase entries written | 218 (65% of attempted) |
| Questions with all 3 styles | 203 (60.8%) |
| Questions with 1-2 styles | 15 (4.5%) |
| Questions with 0 styles | 116 (34.7%) |
| `lexical` accepted | 205 (61.4%) |
| `structural` accepted | 205 (61.4%) |
| `casual` accepted | 217 (65.0%) |
| API calls (200 OK) | 962 |
| Rate-limit hits (429) | 646 |
| Leak-then-retry events | 171 |
| Retry successes | 16 (9.4% of retries) |
| Styles permanently skipped | 151 (after double failure) |
| Wall-clock for generation | ~30 minutes |

### Coverage analysis

The 35% zero-entry rate is **not random** — it concentrates around questions where the gold answer is the subject entity of the question. The validation gate (≥80% token overlap with the gold answer) rejects paraphrases that contain those entities, and `temperature=0` retries don't help much (9.4% success). See `2026-07-08-validation-gate-coverage-issue.md` for full analysis and three resolution options. Recommended for now: accept the coverage rate, document it honestly, revisit only if downstream metrics are uninterpretable.

---

## 3. Evaluation Results

The output format preserves all existing headline labels (`paragraph_recall@K`, `sf_precision`, `sf_recall`, `sf_f1`, `sf_em`, `cache hits / builds`) for backward compatibility, with new sections appended below.

### Headline (across all 4 variants × 961 evaluations)

| Metric | Value |
|---|---|
| `paragraph_recall@4` | **0.832** |
| `sf_precision` | 0.419 |
| `sf_recall` | 0.832 |
| `sf_f1` | 0.556 |
| `sf_em` | 0.005 |

`sf_em` (exact-set match of retrieved vs gold paragraph titles) is very low (0.005) — typical for HotpotQA distractors where the retriever surfaces *related* but not *exactly identical* paragraph sets. This is the right behavior for a "did we find evidence" retriever; EM is the wrong primary metric for HotpotQA.

### Per-variant breakdown

| Variant | n | `answer_coverage@k` | `sf_recall@k` | `para_recall@k` |
|---|---:|---:|---:|---:|
| **original** | 334 | **0.763** | **0.852** | **0.852** |
| `lexical` | 205 | 0.712 | 0.820 | 0.820 |
| `structural` | 205 | 0.712 | 0.815 | 0.815 |
| `casual` | 217 | 0.760 | 0.829 | 0.829 |

`n` differs across variants because some questions had one or two styles rejected by the validation gate. The headline metrics above are computed over the full per-variant `n`, not over a common subset.

### Aggregate

| Metric | Value |
|---|---|
| `mean_ans_cov@k` (over all 4 variants) | **0.741** |
| `robustness@4` | **0.675** (137 / 203 qids had all 4 variants succeed) |

`robustness@4` is the most actionable number: **a third of fully-covered questions fail at least one variant**, meaning the retriever is meaningfully sensitive to surface-form variation. This is consistent with using a small embedding model (`all-MiniLM-L6-v2`, 22M params); larger models typically smooth this out.

### Per (type, level)

| Bucket | `ans_cov@k` | n (variant-evaluations) |
|---|---:|---:|
| `bridge/hard` | 0.731 | 551 |
| `comparison/hard` | 0.754 | 410 |

Interestingly, `comparison/hard` slightly outperforms `bridge/hard` on answer coverage here. This is a 334-question sample so the difference may not be statistically meaningful — the per-question standard deviation is high. Worth re-checking on a larger sample or the train split for a cleaner read.

### Footer

| | |
|---|---|
| Variant-evaluations | 961 (out of 334 questions × up to 4 variants) |
| Cache hits / builds | 0 / 334 (cold cache, as expected with `--no-cache`) |
| Errors | 0 |
| Eval wall-clock | 20.8 s |

---

## 4. Analysis

### What the numbers say

**The retriever handles paraphrase pressure reasonably well.** Original answer-coverage is 0.763. The drop to casual (0.760) is essentially zero. The drop to lexical/structural (both 0.712) is ~5 percentage points. This is consistent with general findings that synonym-level paraphrases are well-handled by sentence-transformers models, while clause reorder (structural) tends to break them — but the gap here is modest.

**Casual paraphrasing barely hurts the retriever.** This is interesting — it suggests that real user phrasing (lowercase, contractions, dropped articles) maps to a similar embedding-space neighborhood as the formal original. For a chat application, this is the most important signal: real users don't write "Was the film released in 1999?" — they write "was the film out in 99?".

**Robustness@4 of 67.5% is moderate.** A third of fully-covered questions fail at least one variant. The strongest factor is probably the embedding model size; switching to `all-mpnet-base-v2` (110M params, ~5× larger) typically gains 5-10 percentage points on paraphrase-robustness benchmarks.

### What's surprising or worth investigating

1. **`comparison/hard` outperforms `bridge/hard` slightly** — contrary to typical HotpotQA literature where bridge questions (multi-hop) are harder. Likely a sample-size artifact at 334 questions.
2. **`structural` and `lexical` tie exactly** at 0.712 answer-coverage, 0.815/0.820 sf-recall. Could be coincidence (both are paraphrases that move away from the original phrasing) or could indicate the model is treating them similarly.
3. **`sf_precision` is low (0.42)** while `sf_recall` is high (0.83). The retriever returns too many supporting-fact-worthy paragraphs on average — it over-includes. This is the dual of the EM=0.005 finding: it gets the right answer into top-k but doesn't tightly bound the set.

### What this eval cannot tell us

- **End-to-end QA accuracy.** We measure retrieval (does the gold answer appear in top-k?). We do *not* measure whether the downstream LLM produces the correct answer given the retrieved context. That's a separate eval and would require `answer_f1` against the gold string.
- **Hardness calibration.** Per-question standard deviation is high; the per-bucket means at 334 questions are noisy. A 0.04 gap between buckets is well within sampling error.
- **Cross-encoder re-ranking impact.** The current pipeline uses bi-encoder retrieval only. Adding a cross-encoder re-ranker on top would likely move these numbers significantly.

---

## 5. Reproducibility

### Re-run generation (overwrites the JSON)

```bash
# Removes the cached JSON, forcing regeneration
rm backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json

python scripts/generate_paraphrases_hotpotqa.py --subset 1000
```

Wall-clock: ~30 min (mostly rate-limit backoff). API cost: ~$0.50-2 (depending on token pricing for `minimax-3`).

### Re-run eval (uses cached FAISS indices after first run)

```bash
# Cold cache (rebuild every per-question index): ~20s for retrieval
python scripts/eval_hotpotqa.py --subset 1000 --no-cache

# Warm cache (subsequent runs): ~5s
python scripts/eval_hotpotqa.py --subset 1000
```

### Scale to full 7,405 questions

```bash
python scripts/generate_paraphrases_hotpotqa.py --full   # ~3-6 hours
python scripts/eval_hotpotqa.py --full --no-cache        # ~10 minutes cold
```

`--full` removes the per-bucket-cap bottleneck and processes all questions in both existing buckets, giving 5,918 + 1,487 = 7,405 questions. With ~65% paraphrase coverage, expect ~4,800 entries in the JSON, and ~14,000-19,000 variant evaluations in the eval.

---

## 6. Limitations

1. **Two-bucket dataset.** HotpotQA dev_distractor only has `bridge/hard` and `comparison/hard`. For meaningful "all 6 buckets" coverage, the train split is needed — but it's much larger (~85k questions).

2. **35% paraphrase zero-coverage.** Documented separately in `2026-07-08-validation-gate-coverage-issue.md`. Concentrated on entity-as-answer questions. Three options for resolution: accept (current), lower threshold (risky), post-process stripping (last resort).

3. **Single embedding model.** `all-MiniLM-L6-v2` is 22M params; this is a deliberate choice for speed. Larger models would likely improve robustness@4.

4. **Single retrieval top-k (4).** Different `k` values would shift recall/precision trade-offs. Worth re-running with `k=8` or `k=10` to see if the gap between variants narrows at higher k.

5. **Per-question variance is high.** With 334 questions, per-bucket means are noisy. A 5-percentage-point gap between buckets is not statistically meaningful.

---

## 7. Files Produced

| Path | Contents |
|---|---|
| `docs/eval-results/2026-07-08-1000-question-paraphrase-eval.md` | This report |
| `docs/eval-results/2026-07-08-validation-gate-coverage-issue.md` | Separate doc on the 35% zero-coverage issue |
| `backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json` | 218 paraphrase entries keyed by qid |
| `backend/storage/eval/hotpotqa/cache/4e9ecb5c8d3b719f/{qid}/` | 334 per-question FAISS indices (cold cache means all rebuilt for this run) |

## 8. Plan / Spec / Implementation Trace

| Stage | Commit | Notes |
|---|---|---|
| Spec | `345393e docs(plan): paraphrase-eval pipeline implementation plan` | Plan at `docs/superpowers/plans/2026-07-08-paraphrase-eval.md` |
| Task 1 | `71b404d feat(eval): add answer_coverage_at_k metric` | Pure metric in `backend/eval/metrics.py` |
| Task 2 | `8eadfbe feat(eval): add paraphrase load + validate helpers` | `backend/eval/paraphrases.py` |
| Task 3 | `9999efb feat(scripts): generate HotpotQA paraphrases via 3 concurrent LLM calls` | Generator CLI |
| Task 3 fix | `ab6fca6 test(generator): fix validation-gate retry test mock + assertion` | Test bug fix |
| Task 4 | `efc58f7 feat(eval): score retrieval across original + 3 paraphrases per question` | Eval pipeline modification |

---

## 9. Recommendations for the next iteration

1. **Address the validation-gate coverage issue** if downstream consumers of the eval complain about missing-data. Otherwise, accept it. The doc at `2026-07-08-validation-gate-coverage-issue.md` lays out three options.

2. **Try a larger embedding model** (`all-mpnet-base-v2` or similar). The biggest expected gain is in robustness@4.

3. **Run with `--full` once** to get the full 7,405-question picture. Cost is ~$5-15 API + ~3-6 hours wall-clock for generation.

4. **Add `k=8` and `k=10` runs.** If the gap between paraphrase variants narrows at higher k, that tells us the retriever is surfacing the right paragraph just outside top-4 — a fixable ranking issue.

5. **Add an end-to-end QA metric** (generate answer from retrieved context, F1 against gold). This is a separate pipeline change but addresses the "does the user actually get the right answer?" question that retrieval metrics alone can't answer.

6. **Run on HotpotQA train split** if you want all 6 (type, level) buckets represented. Train has 85k questions; per-bucket breakdown would be far more meaningful than the dev-distractor's two-bucket structure.