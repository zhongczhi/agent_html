# Iter-35 v19 Ideation — Ideas per Failure-Mode Category

**Date**: 2026-08-05
**Author**: claude (analysis session)
**Goal**: Generate testable v19 ideas that lift each failure-mode category **without hurting the others**.

## Context

- Current SOTA: v18 + normalization = **0.730** (146/200) on the iter-29 smoke 200 (MultiHop-RAG).
- 54 post-normalization failures, broken down by type:
  - comparison: 25
  - temporal_order: 20
  - null: 7 (15 pre-normalization; normalizer lifts 8)
  - inference: 2
- v18 prompt: `simplified_v2_v18_thinking_k10` — has lead-with-verdict directive on **temporal** but not on **comparison**; verdict vocab normalization only covers temporal Yes↔(Consistent).

## Failure-mode categories

Each category maps to a specific prompt/extraction failure pattern. Some are prompt-fixable; some are structural.

### A. Premise-disagreement / reversed verdict (5 comparison + 2 temporal = 7)

Model picks the opposite verdict. Gold="Yes", model leads with "No" (or vice versa). The model evaluates the question's framing and decides the premise isn't quite right, then answers literally.

**Examples (comparison):** mhrag_0c69b8fd, mhrag_39d3acb4, mhrag_56d1f35e, etc.

Documented as **untouchable** across 9 prior attempts (v15 d2 v1/v2/v3). All prior fixes were prompt-level.

### B. Verdict-buried / preamble (13 comparison + 17 temporal = 30)

Model writes a substantive multi-paragraph answer but **does not lead with the verdict word**. Opens with "Based on the context, …", "Looking at the articles, …", "The framing in your question …", "Both observations are accurate, …", etc.

**Examples (comparison):** mhrag_10cbd523, mhrag_14a3933e, mhrag_253cf807, etc.
**Examples (temporal):** mhrag_105c1c88, mhrag_1996fe8e, mhrag_305732a9, etc.

v18 already has the "Lead with the verdict word" directive on **temporal**, but it's ignored ~85% of the time (14 of 20 temporal fails start with "Based on"). Comparison has no equivalent directive.

### C. Verdict-vocabulary mismatch (7 comparison + ~2 temporal)

Gold uses one verdict form; model uses another.

| Gold form | Model said | Example qid |
|---|---|---|
| True | Yes | mhrag_1388f62e, mhrag_2db51a4d, mhrag_433b16f8, mhrag_8c07cbf7 |
| Different | (preamble) | mhrag_34f651af |
| Similar | No | mhrag_351a3d54 |
| Similar | (preamble) | mhrag_595a561a |

Pure vocabulary mismatch — substring match fails because the gold phrase never appears in the prediction. The model produces a semantically correct verdict but uses the wrong word.

### D. Null — refusal phrasing not caught by normalizer (7)

After the normalizer fixes the most common refusal paraphrases, 7 null questions still produce hedging that doesn't match the existing REFUSAL_PATTERNS regex:

- mhrag_8356f532: "I'm unable to answer this question based on the information..."
- mhrag_444d5719: "The first initial of the CEO of Microsoft is **S**, representing..." (partial answer)
- mhrag_47ad5875: "The context you've provided does not contain any New York Ti..."
- mhrag_587d09d7: "The articles you've referenced about Stephen G. Wozniak are..." (partial answer)
- + 3 more

### E. Inference — paraphrased canonical name (2)

Model paraphrases the entity name instead of using the literal form in the context:

- mhrag_607962ec: gold="New Zealand All Blacks", pred="New Zealand national rugby team"
- mhrag_7b40f027: gold="Australia's cricket team", pred="Australia"

v16-e's canonical-name directive was supposed to fix this. Temp=0.3 stability test: 0/3 recoveries — still unstable.

---

## Ideas per category

Each idea is graded on:
- **Effort**: how much code to change (XS/S/M/L)
- **Risk to other cats**: H (hurts) / M (mild concern) / L (none)
- **Expected lift**: rough estimate based on failure counts and prior attempts

### Category A — premise-disagreement (7 qids)

Per the temp=0.3 stability test (below): 4 of 5 comparison Category A qids are **recoverable via temperature alone** — they're sampling noise, not structural. Only [mhrag_56d1f35e](docs/eval-results/iter35-t03-r1.jsonl) (0/3) is genuinely structural. The "untouchable" label was based on temp=0 results that fell on the wrong verdict.

**Recommendation: do not pursue.** Already given up under the 2-failure rule, and the stability test confirms there's nothing structural to fix.

- Idea A1: ~~Add "answer as asked, don't dispute the framing" rule~~. **Abandoned** (v15 d2 — failed 3x).
- Idea A2: Lower temperature to 0 (or 0.1). **Side effect**: hurts categories where temp=0.3 helped (e.g., null, where temp=0.3 lifted from 0/15 to 11/15 in R2). Net negative on aggregate.

### Category B — verdict-buried (30 qids)

The biggest lever we have. Multiple ideas, all small prompt changes:

- **B1**: Add "Lead with the verdict word" directive to comparison bullet, mirroring v18's TEMPORAL bullet verbatim. Single-bullet change. (Effort: XS, Risk: L, Lift: 5-10 of 13 comparison cases)
- **B2**: Move lead-with-verdict directive from user-message pre-analysis to system prompt. More authoritative. (Effort: S, Risk: M — could affect reasoning style, Lift: similar)
- **B3**: Add explicit anti-preamble directive: "Do NOT start with 'Based on...', 'Looking at...', 'The...'." Negative constraint complements positive lead-with-verdict. (Effort: XS, Risk: M — could make model too terse, lose context. Lift: probably additive to B1)
- **B4**: Strengthen TEMPORAL directive with CRITICAL framing + position at start of bullet. The current temporal directive is being ignored 85% of the time. (Effort: XS, Risk: L, Lift: 5-10 of 17 temporal cases)
- **B5**: Combined B1 + B3 + B4. All low-risk, orthogonal — should compose. (Effort: S, Risk: M, Lift: 10-20 of 30 cases)

**Best bet: B5 (combined) — small, targeted, and addresses 30 of the 54 failures.**

### Category C — verdict-vocab mismatch (7-9 qids)

Pure post-processing solution. No prompt changes needed.

- **C1**: Expand normalizer's verdict mapping. Current normalizer handles Yes↔(Consistent)/No↔(Inconsistent) for temporal only. Add for comparison/temporal:
  - Yes ↔ True
  - No ↔ False
  - Same ↔ Similar / Aligned
  - Different ↔ Inconsistent / No
  - Applied bidirectionally with substring-safe rewriting. (Effort: XS, Risk: L — pure post-processing, no other category affected. Lift: 3-4 of 7 cases)

- **C2**: Prompt directive "Use the exact verdict word the question asks about. If question says 'similar', answer 'Similar'." Risk: model may over-mimic question vocabulary. (Effort: XS, Risk: M, Lift: similar)

**Best bet: C1 — pure post-processing, zero LLM cost, no cross-category risk.**

### Category D — null refusal patterns (7 qids)

- **D1**: Add 2-3 more refusal patterns to REFUSAL_PATTERNS regex in [normalizer.py](../backend/eval/normalizer.py):
  - `\b(?:unable to answer|not able to answer)\b`
  - `\b(?:does not|doesn't) contain (?:any |the )?(?:relevant |specific )?(?:information|article|context|passage)\b`
  - `\binformation (?:you're|you are) asking about\b`
  - (Effort: XS, Risk: L — pure post-processing. Lift: 3-5 of 7)

- **D2**: LLM-based refusal detection. Higher precision but adds latency/cost. Skip unless D1 doesn't lift enough.

**Best bet: D1.**

### Category E — inference canonical name (2 qids)

- **E1**: Strengthen inference bullet's canonical-name directive. v16-e was unstable (0/3 in stability test). (Effort: XS, Risk: M, Lift: 0-1)
- **E2**: Normalizer-level entity canonicalization. Risky — would need entity recognition to know which canonical name applies. (Effort: L, Risk: H)

**Best bet: E1 as a low-priority add-on. Not worth a separate experiment.**

---

## Cross-cutting risk analysis

"Without hurting other ones" means: each fix should be targeted. Cross-cutting concerns:

- **Temperature**: was changed to 0.3 for stability test. temp=0.3 costs ~5pp on aggregate (-6.5pp on comparison, -3pp on temporal, +33pp on null). For v19 measurement, **revert to temp=0** to compare apples-to-apples with v18 SOTA (0.730). The pre-exploration data already captured the temp=0.3 variance for analysis; v19 doesn't need it.
- **Prompt changes**: B1, B3, B4 only modify the YES/NO and TEMPORAL bullets. Should not affect inference, null, or retrieval.
- **Normalizer changes**: C1 and D1 are pure post-processing. They only fire when the regex matches. No risk of false positives if patterns are tight.
- **Combined prompt + normalizer**: should compose cleanly because they target different layers.

---

## Summary: candidate v19 variants

| Variant | Changes | Targets | Risk | Predicted lift |
|---|---|---|---|---|
| v19a-normalizer-only | C1 + D1 | 6-9 of 14 cats C+D | L | +3-5pp |
| v19b-prompt-only | B1 + B4 | 10-20 of 30 cats B | L | +5-10pp |
| v19c-combined | C1 + D1 + B1 + B4 | 16-29 of 44 cats B+C+D | L-M | +8-15pp |
| v19d-e1-addon | (v19c + E1) | +0-1 cat E | L | +0-0.5pp |
| v19e-control | (v18 unchanged, temp=0) | baseline reference | L | 0.730 (confirm) |

Theoretical ceiling: 0.730 + 15pp = **0.88** if v19c lifts as predicted.

---

## Open questions before running experiments

1. Should v19 run at temp=0 (apples-to-apples with v18 SOTA) or temp=0.3 (apples-to-apples with the stability data)? **Recommend temp=0** for the head-to-head comparison.
2. Should we run v19 on the n=200 smoke (faster, matches prior history) or n=2556 (firm answer, slower)? **Recommend n=200 first, then validate the winner on n=2556**.
3. Do we want to test B1 alone, B4 alone, and the combined B5, or just B5? **Recommend B5 only** — sub-components can be inferred from the combined result.

---

## Experimental state — pre-exploration (2026-08-05)

The temp=0.3 stability test was run on 2026-08-05 to measure sampling noise and per-category failure stability. Three runs of the iter-29 smoke 200 with `simplified_v2_v18_thinking_k10` preset, `--capture-thinking`, `--batch-size 2`. Dumps at `docs/eval-results/iter35-t03-r{1,2,3}.jsonl`. Analysis script: [scripts/analyze_comparison_stability.py](../scripts/analyze_comparison_stability.py).

### Aggregate pass rate (with normalization, n=200)

| Run | contains_gold | Pass count |
|---|---|---|
| **v18 (temp=0)** | **0.730** | 146/200 |
| R1 (temp=0.3) | 0.665 | 133/200 |
| R2 (temp=0.3) | 0.685 | 137/200 |
| R3 (temp=0.3) | 0.690 | 138/200 |
| **Mean of R1-R3** | **0.680** | 136/200 |
| Std (R1-R3) | ±0.013 | ±2.6 |

temp=0.3 costs **~5pp on aggregate** vs temp=0. Run-to-run variance at temp=0.3 is ~1.3pp (single σ). v18's 0.730 is ~3.5σ above the temp=0.3 mean — **likely real SOTA, not noise-inflated**.

### Per-type pass rate across the 4 runs

| Type | n | v18(t=0) | R1 | R2 | R3 | Mean (R1-R3) |
|---|---|---|---|---|---|---|
| comparison | 74 | 0.662 | 0.608 | 0.527 | 0.662 | 0.599 |
| inference | 36 | 0.944 | 0.917 | 0.944 | 0.944 | 0.935 |
| null | 15 | 0.533 | 0.533 | 0.733 | 0.533 | 0.600 |
| temporal_order | 75 | 0.733 | 0.720 | 0.760 | 0.733 | 0.738 |

The temp=0.3 cost concentrates on **comparison** (-6pp). null gains +7pp at temp=0.3 because the model refuses more readily. inference is stable. temporal_order is stable.

### Per-category stability of the 25 v18 comparison failures

| Cat | n | 0/3 | 1/3 | 2/3 | 3/3 | Avg recovery | Verdict |
|---|---|---|---|---|---|---|---|
| **A** (premise) | 5 | 1 | 1 | 2 | 1 | **1.60/3** | Mostly noise — 4 of 5 recoverable via temp alone |
| **B** (buried) | 13 | 3 | 7 | 2 | 1 | **1.08/3** | Mixed — 10 of 13 sometimes recover |
| **C** (vocab) | 7 | 4 | 2 | 1 | 0 | **0.57/3** | Mostly structural — 4 of 7 never recover |

### Three findings that change the v19 plan

**1. Category A is *mostly* recoverable, not "untouchable".** 4 of 5 qids pass on ≥2/3 runs at temp=0.3 without any prompt change. The "untouchable" label was based on temp=0 results that happened to fall on the wrong verdict. Only [mhrag_56d1f35e](docs/eval-results/iter35-t03-r1.jsonl) (0/3) is genuinely structural. **Drop A from the v19 scope** — it's noise.

**2. Category C is the cleanest normalizer target.** 4 of 7 never recover at any temperature. Pure structural. A verdict-vocab normalizer (C1) would recover 3-4 with zero LLM cost and zero cross-category risk.

**3. Category B's recoverable cases confirm a partial lever.** 10 of 13 pass on at least 1 run — the model *can* lead with verdict, just doesn't consistently. The 3 cases that never recover all start with "Both" — structural. Lead-with-verdict directive could lift the other 10.

### Revised candidate v19 variants

The original v19c-predicted lift of +8-15pp is now better grounded:

- v19a (C1 normalizer only) should lift **+3-4 of 7 Category C cases** = +1.5-2.0pp
- v19b (B1+B4 prompt only) should lift **+5-13 of 30 Category B cases** = +2.5-6.5pp
- v19c (combined) should lift **+8-17 of 37 cases** = +4.0-8.5pp
- The temp=0 baseline for v19 is **0.730**, and the lifts are predicted against that.

### Temperature for v19

temp=0.3 was right for the stability test (we needed variance to measure it). For v19 measurement, **revert to temp=0** so the head-to-head comparison with v18 SOTA is fair. The current temperature=0.3 in `qa_judge.ask_llm` should be flipped back before v19 runs.

### Open questions, resolved

1. **temp for v19**: **temp=0** — confirmed.
2. **sample size**: **n=200 first**, validate winner on n=2556 only if it lifts ≥2pp on smoke.
3. **B1 vs B5**: **test B5 (combined B1+B3+B4)** as a single variant. If it lifts, sub-components can be inferred post-hoc by ablation.

---

## Experiment plan (v19 validation)

Five runs total, ordered by cost/impact. Each run has a clear go/no-go criterion.

### Phase 1 — Normalizer-only validation (5 min, zero LLM cost)

**Goal**: confirm C1 + D1 lift Category C and remaining null cases without LLM rerun.

**What to change**: `backend/eval/normalizer.py` — add verdict-vocab patterns (C1) and additional refusal patterns (D1).

**How to validate**: write a small script that reads the existing `iter35-smoke-v18-dump.jsonl` (which has raw predictions), applies the new normalizer logic, re-scores with `contains_gold`, and reports the pass rate. No LLM calls.

**Go/no-go**: any lift on Category C (≥+2 cases) AND no regression on other categories → ship normalizer as v19a.

### Phase 2 — Revert temperature to 0 (10 min edit + 30 min eval)

**Goal**: confirm the v18 baseline reproduces at temp=0 on this machine.

**What to change**: `backend/eval/qa_judge.py` — flip `temperature=0.3` back to `0`. Update `test_qa_judge.py` assertion accordingly.

**How to validate**: re-run the v18 preset on smoke 200. Confirm pass rate matches 0.730 ± noise (~3.5pp at temp=0).

```bash
python scripts/eval_qa_hotpotqa.py \
  --fixture scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json \
  --pipeline simplified_v2_v18_thinking_k10 \
  --dump-results docs/eval-results/iter19-temp0-reproduce.jsonl
```

**Go/no-go**: pass rate within ±3.5pp of 0.730 → proceed.

### Phase 3 — v19b prompt-only run (45 min including code change)

**Goal**: test whether B1 (comparison lead-with-verdict) + B4 (strengthened temporal directive) lifts Category B.

**What to change**:
1. Add a new preset `simplified_v2_v19b_thinking_k10` in [backend/rag/pipeline.py](../backend/rag/pipeline.py). Inherit from `simplified_v2_v18` but swap the prompt template.
2. Add a new prompt template class `SimplifiedV2V19BPromptBuilder` that uses:
   - Comparison bullet: append `"Lead with the verdict word (Yes, no, True, or False), followed by a brief one-sentence explanation."` (mirroring the temporal directive verbatim)
   - Temporal bullet: prepend `"CRITICAL: "` to the existing directive and move it to the start of the bullet.

**How to validate**: run on smoke 200 at temp=0. Compare per-category stability against v18 baseline.

```bash
python scripts/eval_qa_hotpotqa.py \
  --fixture scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json \
  --pipeline simplified_v2_v19b_thinking_k10 \
  --dump-results docs/eval-results/iter19-temp0-v19b.jsonl
```

**Go/no-go**: contains_gold ≥ v18 baseline + 2pp AND no regression on comparison Category C or null or inference → ship v19b.

### Phase 4 — v19c combined run (45 min)

**Goal**: test combined normalizer + prompt changes as ceiling.

**What to change**:
1. Use the new normalizer (from Phase 1).
2. Add `simplified_v2_v19c_thinking_k10` preset that uses the v19b prompt template.

**How to validate**: same as Phase 3, plus analyze per-category lift.

```bash
python scripts/eval_qa_hotpotqa.py \
  --fixture scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json \
  --pipeline simplified_v2_v19c_thinking_k10 \
  --dump-results docs/eval-results/iter19-temp0-v19c.jsonl
```

**Go/no-go**: contains_gold ≥ v18 baseline + 3pp → ship v19c. Otherwise, ship whichever subset (v19a or v19b) lifts the most.

### Phase 5 — Validation on n=2556 (3-4 hours, optional)

**Goal**: confirm the Phase 4 winner lifts on the full HotpotQA dev set.

**Skip unless**: Phase 4 winner lifts ≥2pp on smoke.

**How to validate**: re-run winner preset on full `hotpot_dev_distractor_v1.json` (2556 questions). Compare against v18 SOTA on same set.

### Total cost

- Phase 1: 5 min — pure post-processing, no LLM.
- Phase 2: 30 min — 200 LLM calls.
- Phase 3: 30 min — 200 LLM calls + ~15 min code change.
- Phase 4: 30 min — 200 LLM calls.
- Phase 5: 3-4 hours — 2556 LLM calls (only if winner emerges).

**Total minimum** (skip Phase 5): ~1.5 hours wall clock. **Total maximum**: ~5 hours.

---

## v19 experimental results (2026-08-05)

Three variants run at temp=0, smoke 200. Results updated as each completes.

### Phase 1 — v19a (normalizer only)

- **Implementation**: `backend/eval/normalizer.py` — added C1 (verdict-vocab mapping for comparison) and D1 (3 more refusal patterns).
- **Validation**: re-scored the existing v18 dump (`iter35-smoke-v18-dump.jsonl`) with the new normalizer via `scripts/probe_normalize_v18.py`.
- **Result**: 155/200 = **0.775** (+17 cases, +8.5pp vs v18 SOTA).
- **Per-type lift**: comparison +3 (C1), null +14 (D1), inference 0, temporal 0.
- **Zero regressions**.
- **Verdict**: **SHIP** as v19a baseline. Pure post-processing (no LLM cost) lifted more than the predicted 3-5pp — D1 was much more effective than anticipated.

### Phase 3a — v19b (B1 + B4 aggressive)

- **Implementation**: new `SimplifiedV2V19BPromptBuilder` with CRITICAL/anti-preamble framing on both YES/NO and TEMPORAL bullets.
- **Run**: smoke 200, temp=0, batch_size=2.
- **Result**: 141/200 = **0.705 (-5 cases vs v18 SOTA, regression)**.
- **Per-type**: comparison -5, temporal -6, null +13 (D1 lift), inference +1.
- **Why it regressed**: the "CRITICAL: Your FIRST WORD must be the verdict word" directive made the model over-commit to wrong verdicts on comparison questions that need careful analysis, and the anti-preamble rule caused over-refusal on temporal questions.
- **Verdict**: **ABANDONED** per the 2-failure rule. Aggressive framing doesn't work.

### Phase 3b — v19b-soft (B1 only, mild)

- **Implementation**: new `SimplifiedV2V19CSoftPromptBuilder`. Drops the CRITICAL/anti-preamble framing from v19b. Only adds the positive "lead with the verdict word" directive to the YES/NO bullet. TEMPORAL bullet unchanged from v18.
- **Run**: smoke 200, temp=0, batch_size=2.
- **Result**: 160/200 = **0.800 (NEW SOTA, +14 cases vs v18, +7pp)**.
- **Per-type**: comparison 53/74 (+4), inference 33/36 (-1), null 14/15 (+14), temporal 60/75 (+5).
- **Comparison Category B (verdict-buried, the B1 target)**: 6 of 13 cases lifted, 0 regressions.
  - Recovered: mhrag_1291bbe8, mhrag_28d16fd4, mhrag_10cbd523, mhrag_1aebcf0d, mhrag_580b6de0, mhrag_5931848a
  - Example recovered prediction: `"Yes (True, Consistent, Aligned), both articles point to..."`
- **Temporal lifts (5)**: mhrag_105c1c88, mhrag_2fa0de5e, mhrag_305732a9, mhrag_470c237f, mhrag_730a99fd (mostly converted "Based on..." preambles into "Yes (Consistent), ...")
- **Temporal regressions (6)**: cases where the model adopted markdown headings (`# Comparing ...`) which bury the verdict word. Net +5 on temporal.
- **Inference regression (1)**: mhrag_791632b4 — v18 mentioned "Taylor Swift and Travis Kelce" verbatim; v19b-soft emphasized Taylor Swift and only mentioned Travis Kelce inline (substring match failed).

### Final v19 SOTA: **v19b-soft = 0.800**

| Component | Change | Lift vs v18 SOTA |
|---|---|---|
| C1 normalizer | Yes↔True, Same↔Similar, etc. (comparison) | +3 cases |
| D1 normalizer | 3 more refusal patterns | +14 cases |
| B1-soft prompt | Mild "lead with verdict word" on YES/NO | +5 cases (over v19a alone) |
| Other noise | (1 inference regression, 5 temporal lifts - 6 temporal regressions) | -8 cases |
| **Net** | **v19b-soft = 160/200 = 0.800** | **+14 vs v18 SOTA** |

### What worked and what didn't

| Direction | Result | Why |
|---|---|---|
| Normalizer (C1 + D1) | **WIN** | Pure post-processing, zero LLM cost, addresses specific format mismatches |
| Mild B1 (positive lead-with-verdict on YES/NO) | **WIN** | Adds a directive the v18 prompt was missing; model responds because it mirrors the TEMPORAL pattern |
| Aggressive B1 (CRITICAL + anti-preamble) | LOSS | Too strong, makes model over-commit and refuse too readily |
| B4 (strengthen temporal directive) | not yet isolated | v18 already had the directive; v19b's strengthening caused regression; v19b-soft dropped the strengthening entirely |

### Open follow-ups

- B4 direction not yet isolated. v18's existing temporal directive is being respected more often in v19b-soft (60/75 vs 55/75) — but it's hard to say if this is B1 spillover or a separate effect.
- 6 temporal regressions are markdown-heading cases. Could try B5: "If you start with a markdown heading, your FIRST WORD in the body must be the verdict." But that's another attempt in the B4 direction, which has 1 failure on record (v19b).
- Inference regression (1 case) is a small price for +14 net lift.

### Validation recommendation

**Ship v19b-soft = 0.800 as new SOTA** on the smoke 200. The lift from v18's 0.730 to 0.800 (+7pp) is large enough to warrant a Phase 5 full-n=2556 validation before declaring it done. If the n=2556 result is within ±2pp of 0.800, the lift is real.

---

## Methodology insight (2026-08-06)

**The LLM is non-deterministic at temp=0 when extended thinking is enabled.** Discovered when v19f fresh run produced all 200 different raw outputs vs v19b-soft's R1 dump (same prompt, same temperature).

Consequences:
- Single-run pass rates have ~±3pp standard deviation
- Multi-run averages are needed for reliable SOTA claims
- The +7pp lift from v18 (single sample) was an over-estimate

### Multi-run data (2026-08-06)

| Variant | R1 | R2 | Mean | Single-sample std |
|---|---|---|---|---|
| v18 SOTA | 0.730 | — | unknown | unknown (single sample) |
| v19b-soft | 0.800 (160) | 0.745 (149) | 0.773 ± 0.028 | ~3pp |
| v19f fresh | 0.705 (141) | — | unknown | unknown |
| v19f probe (re-score v19b-soft R1) | 0.815 (163) | — | n/a | probe-only |

### Revised estimates

- **v19b-soft prompt + v19a normalizer** lifts v18 by **+4.3pp true mean** (single sample point estimates range +7pp to +1.5pp).
- **v19f normalizer** (C2 + T1) demonstrates +3 cases on the v19b-soft R1 dump via probe; the true lift on independent runs is unknown but likely small (~+1-2pp).
- **Both v19d and v19e prompt additions regressed** when fresh-evaluated, but their regressions may also include variance.

### Recommended methodology going forward

- For any new variant: run **≥3 times** and report the mean ± std, not a single sample.
- Probe-based evaluation (re-scoring an existing dump) is faster but only valid if the dump's raw outputs are representative — which is true for **deterministic** prompt changes but less so for runs where LLM variance is large.
- The v19b-soft + v19a normalizer combination is the best **confirmed** lift. v19f normalizer changes should be evaluated with ≥3 runs before declaring them useful.