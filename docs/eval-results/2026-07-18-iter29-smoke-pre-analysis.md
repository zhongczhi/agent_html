# Iter-29 Smoke — Pre-Analysis Prompt Instruction

**Date**: 2026-07-18
**Iteration**: First eval of the iter-29 `pre_analysis_extract_thinking_k10` preset — same as iter-22 SOTA but with a pre-analysis prefix added to the user message
**Goal**: Measure whether asking the model to "briefly analyze the question before reading the context" improves extraction on the questions iter-22 SOTA already gets wrong

---

## TL;DR — +0.5 pp overall, but the per-type picture is split

| Preset | contains_gold (n=200) | Wall-clock | Per-call cost |
|---|---:|---:|---|
| iter-22 SOTA (`cot_extract_notitles_thinking_k10`) | 0.620 (124/200) | 21.3 min | baseline |
| **iter-29 (`pre_analysis_extract_thinking_k10`)** | **0.625 (125/200)** | **19.5 min** | similar |
| Net | **+0.5 pp** | -8% | ~same |

**The aggregate lift is within noise**, but the per-type breakdown shows the pre-analysis instruction helps one type and hurts another:

| Type | n | SOTA | iter-29 | Δ |
|---|---:|---:|---:|---:|
| inference | 36 | 94.4% | 94.4% | 0 |
| **comparison** | 74 | 56.8% | 52.7% | **-4.1 pp** |
| **temporal** | 75 | 64.0% | 69.3% | **+5.3 pp** |
| null | 15 | 0.0% | 0.0% | 0 |

**The pre-analysis instruction trades comparison questions for temporal questions.** Hypothesis-driven and consistent with the iter-26 per-type analysis: temporal questions need explicit ordering reasoning (which pre-analysis primes), while comparison questions need direct yes/no adjudication (which pre-analysis disrupts by forcing the model to write a "supporting analysis").

---

## 1. Subset

Built by `scripts/build_iter29_smoke_subset.py`. 200 qids = 100 from the iter-22 SOTA's 237 real failures (excluding null) + 100 random from the iter-22 SOTA's 2302 completed MultiHop-RAG dump, stratified proportionally by failure type for the failure set.

| Set | comparison | inference | null | temporal | total |
|---|---:|---:|---:|---:|---:|
| failure | 48 | 3 | 0 | 49 | 100 |
| random | 26 | 33 | 15 | 26 | 100 |
| **TOTAL** | **74** | **36** | **15** | **75** | **200** |

Sanity check: 100/100 of the failure set were fails in the iter-22 n=2556 SOTA run; 22/100 of the random set were fails (matches the 22.0% baseline fail rate).

---

## 2. Per-set breakdown

| Subset | n | SOTA pass | iter-29 pass | Δ | flip-up | flip-down |
|---|---:|---:|---:|---:|---:|---:|
| failure | 100 | 45 (45.0%) | 45 (45.0%) | 0 | +15 | -15 |
| random | 100 | 79 (79.0%) | 80 (80.0%) | +1 | +3 | -2 |
| **TOTAL** | **200** | **124 (62.0%)** | **125 (62.5%)** | **+1** | **+18** | **-17** |

**Failure set**: 0 net change. The pre-analysis doesn't recover any of the iter-22 SOTA's hard extraction failures — it just trades different questions (15 newly correct, 15 newly wrong within the failure set). The model's failure modes shift but don't improve.

**Random set**: +1 net. Mostly noise but the right direction (3 newly correct, 2 newly wrong).

---

## 3. What worked, what didn't

### 3.1 Temporal questions: +5.3 pp (the hypothesis was right)

The pre-analysis instruction is forcing the model to think about "what kind of material would answer this" before seeing the context. For temporal questions, that translates to: identify time anchors, look for ordering. The flip-up details show the model saying things like "Yes, there was a notable shift in the narrative about Sam Bankman-Fried's management" instead of just outputting a wrong answer.

12 temporal questions flipped up vs 8 flipped down. The 8 that flipped down look like the model over-reasoning into a wrong answer (e.g., "partial consistency in how Kenneth Walker" instead of "Yes").

### 3.2 Comparison questions: -4.1 pp (the hypothesis was wrong)

Pre-analysis hurts comparison adjudication. The flip-up details show iter-29 writing "**SUPPORTED:** The Sporting News MLB free agency article..." (a long analysis) instead of emitting the bare "Yes" that the gold substring-matches against. 9 comparison questions flipped down vs 6 flipped up.

The mechanism: the pre-analysis instruction primes the model to produce a multi-sentence "analysis" before answering, which uses up output token budget and dilutes the verbatim-extract discipline. Comparison answers on MultiHop-RAG are usually one-word ("Yes", "no", "Different") and need the model to commit early.

### 3.3 Inference and null: 0 change (expected)

Inference is at 94.4% (close to the 99.1% ceiling on n=2556). The pre-analysis instruction has nowhere to add value. Null is at 0% because the SOTA has no refusal path — pre-analysis doesn't add one.

---

## 4. Decision: do not promote iter-29 to the default SOTA

The iter-29 preset's per-type tradeoff (helps temporal, hurts comparison) is real but the **net lift is +0.5 pp — within the run-to-run variance** of a 200-question sample. For the full n=2556 dataset, we'd expect the per-type effects to roughly scale, with the comparison regression likely to grow (because comparison is 33% of the dataset vs 37% in this subset).

**Three concrete reasons not to promote**:
1. **The temporal lift (+5.3 pp) is offset by the comparison regression (-4.1 pp)**. Net zero on a 200-question sample.
2. **The comparison regression is more dangerous than the temporal lift**. MultiHop-RAG comparison questions are filtered by the API on political content (29.4% skip rate). The questions that DO get through are the "easy" comparison cases; pre-analysis makes them harder.
3. **The flip-up/flip-down rate is symmetric (18 vs 17)** — the pre-analysis instruction is shuffling answers, not improving them. The 18 newly-correct questions are mostly temporal (12 of 18); the 17 newly-wrong are mostly comparison (9 of 17).

---

## 5. Recommended next steps

1. **Keep iter-22 SOTA as the production default.** The iter-29 preset stays in the registry for A/B testing but is not promoted.

2. **Investigate the temporal-vs-comparison tradeoff.** The pre-analysis instruction helps temporal reasoning but hurts direct adjudication. A possible refinement: make the pre-analysis **type-aware** — e.g., for comparison questions, instruct the model to emit the answer first, then justify. That's a separate iteration (iter-30+).

3. **Don't run iter-29 at full n=2556.** The aggregate +0.5 pp wouldn't justify the $80-100 cost. The right next experiment is the type-aware variant from step 2.

4. **Document this finding in `document/RAG_pipeline_comparison.md`** as a known "lever tried but not promoted" item — the iter-29 preset stays in `PRESETS` for reproducibility but isn't the default.

---

## 6. Files produced

| Path | Contents |
|---|---|
| `scripts/build_iter29_smoke_subset.py` | Subset builder: 100 failure + 100 random |
| `scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json` | 200-question smoke fixture |
| `docs/eval-results/iter29-smoke-sota-baseline-dump.jsonl` | iter-22 SOTA results on smoke 200 (n=200) |
| `docs/eval-results/iter29-smoke-iter29-candidate-dump.jsonl` | iter-29 candidate results on smoke 200 (n=200) |
| `docs/eval-results/2026-07-18-iter29-smoke-pre-analysis.md` | This report |
| `backend/rag/pipeline.py` | `PreAnalysisExtractPromptBuilder` + new preset |
| `backend/tests/rag/test_pipeline.py` | 6 new tests for iter-29 |

Total wall-clock: ~41 min (21 min baseline + 20 min iter-29, both with batch=4). Total cost: ~$5-7 (small subset, no per-question retrieval rebuild on iter-29 thanks to FAISS cache reuse).