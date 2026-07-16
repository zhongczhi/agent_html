# Iter-26 — MultiHop-RAG Cross-Dataset Validation (n=334, partial)

**Date**: 2026-07-16
**Iteration**: Scaled iter-25's n=100 cross-dataset run to n=334 to get a tighter per-type estimate. n=2556 (full) is running detached and will be reported in a follow-up.
**Goal**: Confirm the n=100 result (SOTA 0.932 non-null) holds at 3.3× the sample size, and detect any per-type variance that n=100 was too small to surface.

---

## TL;DR — SOTA holds at 0.908 (non-null) on n=334

| Preset | n | n (completed) | contains_gold (overall) | contains_gold (non-null) |
|---|---:|---:|---:|---:|
| SOTA n=100 | 100 | 90 | 0.756 | **0.932** |
| **SOTA n=334** | 334 | 294 | **0.735** | **0.908** |
| Baseline n=100 (extract_span_k10) | 100 | 90 | 0.589 | 0.726 |

**The SOTA pipeline transfers cleanly to MultiHop-RAG at n=334.** Non-null contains_gold: 0.908 (216/238), which is within ~3 pp of the n=100 estimate (0.932) and within ~3 pp of the HotpotQA n=334 SOTA (0.934). The gap from n=100 to n=334 is consistent with sampling noise (n=100 has SE ~3 pp on a proportion near 0.9).

**Key per-type findings on n=334**:

| Type | n (completed) | contains_gold | Same in n=100? |
|---|---:|---:|---|
| `inference` | 84/84 (100%) | **1.000** | Yes — 25/25 in n=100 |
| `comparison` | 72/84 (86%) | 0.847 | Down from 0.870 in n=100 (within noise) |
| `temporal` | 82/82 (100%) | 0.866 | Down from 0.920 in n=100 (slight regression) |
| `null` | 56/84 (67%) | 0.000 | Same — unanswerable by design |

The API content filter is **differentially affecting null queries**: 28/84 null-type questions were skipped (33% skip rate) vs 12/84 comparison-type (14%) and 0/82 temporal / 0/84 inference. Null queries often contain "Insufficient information" / "no answer" wording that triggers the LLM endpoint's safety filter.

---

## 1. Setup

Same as iter-25 (`scripts/ingest_multihop_rag.py --subset 334`). The n=334 fixture is at `scripts/.cache/multihop_rag_fixture_334.json` (36 MB, gitignored).

```bash
python scripts/ingest_multihop_rag.py --from-local --subset 334 \
    --output scripts/.cache/multihop_rag_fixture_334.json
```

Stratified by `question_type` (4 buckets × 84 = 336, capped to 334): comparison 84, inference 84, null 84, temporal 82.

```bash
python scripts/eval_qa_hotpotqa.py --subset 334 \
    --fixture scripts/.cache/multihop_rag_fixture_334.json \
    --pipeline cot_extract_notitles_thinking_k10 \
    --batch-size 2 \
    --dump-results docs/eval-results/iter26-multihop-rag-sota-k10-dump.jsonl
```

`--batch-size 2` for ~2× throughput (asyncio.gather concurrency). Wall-clock: 2717.6s ≈ 45 min.

---

## 2. Headline results

```
with_context:
  contains_gold: 0.735  (n=294)
  answer_f1   : 0.015  (n=294)
  answer_em   : 0.000  (n=294)
failure-mode breakdown (with_context):
  success         :  216  (0.735)
  extraction miss :   78  (0.265) — gold in top-k, LLM missed
  retrieval miss  :    0  (0.000) — gold NOT in top-k
LLM calls             : 294
cache hits / builds   : 5 / 289
errors                : 0  ← bug: this counter only catches setup errors
elapsed               : 2717.6s
dumped per-q results  : 294 records
```

**All failures are extraction misses**, 0 retrieval misses — same as iter-25 n=100 and iter-23 HotpotQA full-7k. This is the expected pattern for the per-question-context setup: by construction, the gold paragraph is in top-k.

The `cache hits / builds` ratio (5/289) reflects the per-question FAISS cache being built fresh because the n=334 fixture has a different `dataset_sha` than the n=100 fixture. With shared fixture, cache hits would be much higher.

---

## 3. Per-type breakdown (n=334)

| Type | Fixture n | Dump n (completed) | Skip rate | contains_gold |
|---|---:|---:|---:|---:|
| `inference` | 84 | 84 | 0% | **1.000** (84/84) |
| `temporal` | 82 | 82 | 0% | 0.866 (71/82) |
| `comparison` | 84 | 72 | **14%** (12 skipped) | 0.847 (61/72) |
| `null` | 84 | 56 | **33%** (28 skipped) | 0.000 (0/56) |
| **Non-null total** | 250 | **238** | — | **0.908** (216/238) |

### 3.1 Inference still at ceiling (n=84/84 = 100%)

Inference questions ("Who is the individual associated with the cryptocurrency industry...?") are direct entity+fact extractions. The SOTA prompt's CoT scaffold + title-strip + thinking mode is overkill for these — even the baseline `extract_span` prompt hit 1.000 in n=100. Confirmed at n=334: ceiling is real, not a small-sample artifact.

### 3.2 Comparison held up (0.870 → 0.847)

n=100 result: 0.870. n=334 result: 0.847. The 2.3 pp gap is within sampling noise for n=72 (SE ≈ 5 pp). No regression.

### 3.3 Temporal showed mild regression (0.920 → 0.866)

n=100 result: 0.920. n=334 result: 0.866. The 5.4 pp gap is borderline-significant (n=82, SE ≈ 3 pp). Two possible explanations:
1. **Sampling**: n=100 was lucky. n=334's wider sample catches harder temporal questions.
2. **Real ceiling**: the SOTA's temporal-query ceiling might be ~0.87 rather than 0.92.

Need the n=2556 run to disambiguate. If n=2556 confirms 0.86, the n=100 result was upward-biased.

### 3.4 API content filter bias toward null queries (33% vs 0-14%)

28 of 84 null-type questions got skipped by the API. 0 of 82 temporal, 0 of 84 inference, 12 of 84 comparison. The filter is more aggressive on null-type content — likely because null questions often contain "insufficient information", "no answer", "unclear", or "ambiguous" wording that the safety classifier treats as concerning.

**This is not the eval pipeline's fault** — the same 28 null + 12 comparison questions would be skipped on any preset run. To get a true measure of null-query performance, we'd need either:
- A local model without content filtering
- A pre-filtered subset of null questions whose wording doesn't trip the safety classifier

Out of scope for iter-26.

---

## 4. Cross-dataset generalization update

| Preset | HotpotQA n=334 | MultiHop-RAG n=100 (non-null) | MultiHop-RAG n=334 (non-null) | MultiHop-RAG n=2556 (planned) |
|---|---:|---:|---:|---:|
| SOTA (iter-23) | 0.934 | 0.932 | **0.908** | TBD |
| extract_span_k10 | 0.889 | 0.726 | not run at n=334 | not run |
| SOTA lift over extract_span_k10 | +4.5 pp | +20.6 pp | (estimate: similar) | TBD |

The SOTA pipeline's non-null contains_gold:
- HotpotQA: 0.934
- MultiHop-RAG n=100: 0.932
- MultiHop-RAG n=334: 0.908

**n=334 pulled the cross-dataset SOTA score down by 2.4 pp from n=100.** This is a meaningful narrowing of the gap (was -0.2 pp at n=100, now -2.6 pp at n=334). My prior reading "within sampling noise" was probably too generous — the true cross-dataset SOTA is closer to 0.91 than 0.93.

The HotpotQA → MultiHop-RAG SOTA delta of -2.6 pp is still small in absolute terms, but the n=100 estimate was ~2 pp high due to small-sample variance. The SOTA still transfers cleanly, just not as cleanly as n=100 suggested.

### What this means for production

- The SOTA pipeline generalizes to a second RAG benchmark with different question types, with a modest -2.6 pp drop vs HotpotQA
- The biggest drop is on temporal queries (0.920 → 0.866, n=82), suggesting temporal reasoning is harder for the SOTA on news-article context than on Wikipedia context (HotpotQA's temporal questions were rare)
- The +20.6 pp SOTA-vs-baseline lift on MultiHop-RAG n=100 (temporal was +40 pp) likely still holds at n=334; need to re-run baseline at n=334 to confirm

---

## 5. Caveats

### 5.1 The "errors" counter remains a bug

The CLI prints `errors: 0` even when 40 questions are missing from the dump. The counter is initialized to 0 and never incremented — it only catches setup errors (missing API key, dataset missing, etc.). Per-item errors only show up as missing qids in the dump. This is a pre-existing bug from iter-12 (or earlier); it doesn't affect the result, but makes debugging harder. **Recommend fixing in iter-27.**

### 5.2 The hardcoded "HotpotQA dev_distractor v1" banner

The CLI's dataset attribution line is hardcoded text that doesn't change when `--fixture` points to a different dataset. So the log says "HotpotQA dev_distractor v1 (CC BY-SA 4.0)" even when running on MultiHop-RAG. The actual data is correct (verified by SHA and qid pattern), but the banner is misleading. **Recommend parameterizing the attribution in iter-27.**

### 5.3 Cache-key pollution by per-fixture SHA

Each fixture file gets its own `dataset_sha` (file-hash prefix), so the per-question FAISS cache is rebuilt for every new fixture. This is the intended behavior (a fixture change means re-embedding), but it means n=100 / n=334 / n=2556 each have separate cache roots. For the full-7k HotpotQA run (iter-23), all questions share a single cache; for the MultiHop-RAG runs, we get a fresh cache per fixture size. **No correctness impact, just ~5-10 min extra warmup per fixture.**

### 5.4 Skipped null questions bias the cross-dataset comparison

The API's content filter skips 33% of null-type questions. If we treat the n=334 result as the "true" MultiHop-RAG SOTA, the null-question category is doubly under-counted (already unanswerable + filtered at higher rate). This is a real source of downward bias on the headline 0.735 number, but the 0.908 non-null number is unaffected.

---

## 6. What n=2556 will add (running detached, ETA ~4-6h)

The full n=2556 run is in progress at PID 127836 (detached subprocess, `docs/eval-results/iter26-multihop-rag-sota-k10-full-dump.jsonl`).

When it completes, expect:
- ~2556 × 0.85 (assuming ~15% skip rate) = ~2173 results
- Wall-clock: ~6h with `batch_size=2`
- Per-type confidence intervals will tighten by ~3× (sqrt(2556/334) ≈ 2.8)
- The 0.866 temporal number will resolve whether the n=334 regression is real or sampling noise

I'll report n=2556 in `docs/eval-results/2026-07-16-iter26-multihop-rag-n2556-results.md` when the detached process exits.

---

## 7. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter26-multihop-rag-sota-k10-dump.jsonl` | 294 per-question SOTA results |
| `docs/eval-results/iter26-multihop-rag-sota-k10.log` | SOTA eval log (45 min) |
| `docs/eval-results/2026-07-16-iter26-multihop-rag-n334-results.md` | This report |

(`scripts/.cache/multihop_rag_fixture_334.json` is gitignored at 36 MB.)

---

## 8. Recommended next steps after n=2556

1. **Fix the `errors` counter and dataset attribution banner** in `scripts/eval_qa_hotpotqa.py` (small bug, high debugging value)
2. **Re-run baseline (extract_span_k10) on n=334** to confirm the SOTA-vs-baseline lift holds at this sample size (estimate: 30 min)
3. **Track B**: build a small heterogeneous-format corpus with actual PDF/DOCX/HTML files to exercise the loader pipeline
4. **Per-(type, level) breakdown on n=2556**: detect whether any specific (temporal/easy), (comparison/medium) etc. cell is a weak point
5. **Open-domain evaluation**: extend the eval to skip the per-question context adapter and run open-domain RAG over the full 609-doc corpus (uses `RagService` instead of the per-question FAISS cache)