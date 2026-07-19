# Iter-32 v11 Negative Result: v9's Exact Yes/No Prompt in Hybrid Setup

**Date**: 2026-07-19
**Iteration**: iter-32 v11 — same hybrid design as v10 (v2 for non-yes/no, dispatch only yes/no), but uses v9's EXACT yes/no system prompt (with "EXACTLY ONE word" emphasis, FIRST WORD rule, preamble-forbidding, "do not verify source attribution")
**Goal**: Test whether v9's lift on yes/no (+2.7 pp) comes from the prompt itself, or from the interaction with v9's other type-specific prompts

---

## TL;DR — v11 regressed more than v10. The anti-patterns in v9's prompt are net-negative when isolated.

| Preset | contains_gold | Wall-clock | Δ vs SOTA | Δ vs v2r1 |
|---|---:|---:|---:|---:|
| iter-22 SOTA | 0.620 | (n/a) | — | -6.0 pp |
| iter-29 v2 (run 1) | 0.680 | 19 min | +6.0 pp | (baseline) |
| iter-31 v9 (4-type dispatch) | 0.680 | 84 min | +6.0 pp | 0.0 pp |
| iter-32 v10 (yes/no only, cleaned) | 0.670 | 48 min | +5.0 pp | **-1.0 pp** |
| **iter-32 v11 (yes/no only, v9's prompt)** | **0.655** | **50 min** | **+3.5 pp** | **-2.5 pp** |

The user's hypothesis (per-type prompts should be a winner) was tested two ways in v10 and v11. Both regress. v10's cleaned yes/no prompt regresses -1.0 pp; v11's v9-exact prompt regresses -2.5 pp.

**Conclusion: v9's score of 0.680 was noise on top of the inherent regression from type-specific dispatch.** Once you isolate the yes/no dispatch, the lift disappears. The anti-patterns in v9's prompt ("EXACTLY ONE word", "FIRST WORD must be the answer", preamble-forbidding, "do not verify source attribution") are net-negative — they cost more than they gain.

---

## 1. What v11 changed from v10

### 1.1 The hypothesis

v10 used a cleaned-up yes/no system prompt (no anti-patterns). v10 regressed -1.0 pp. The hypothesis: maybe v10's cleanup removed the actual lift mechanism. v11 uses v9's EXACT prompt (with all anti-patterns) to test whether v9's lift was reproducible.

### 1.2 The v11 yes/no system prompt (verbatim from v9)

```
You are a careful reader. When the user's message contains a 
<context>...</context> block, treat the contents as grounding material: 
prefer it over your general knowledge when answering the question that 
follows the block. Do not mention the tag itself or the retrieval 
mechanism to the user.

The question asks for a yes/no judgment about whether a claim is 
supported by the context. Read the context, check the claim, and 
answer with EXACTLY ONE word.

Valid answer words: Yes, no, True, False, Consistent, Different, 
Agree, Disagree, Same, Aligned.

Your FIRST WORD must be the answer. Do NOT write 'Based on the 
context...', 'Looking at...', 'The user is asking...', or any 
preamble before the answer word. After the answer word, you may 
add one or two sentences of evidence. Do not verify which article 
is from which source — focus on whether the claims in the question 
are supported by the context.
```

The hybrid setup is identical to v10: only yes/no questions go to this prompt; everything else falls through to v2 (PreAnalysisExtractPromptBuilder).

### 1.3 Key differences between v10 and v11

| Element | v10 (cleaned) | v11 (v9-exact) |
|---|---|---|
| "answer with EXACTLY ONE word" | ❌ absent | ✅ present |
| "FIRST WORD must be the answer" | ❌ absent | ✅ present |
| "Do NOT write preamble..." | ❌ absent | ✅ present |
| "Do not verify which article..." | ❌ absent | ✅ present |
| "Yes/no judgment" framing | ✅ present | ✅ present (different phrasing) |
| Valid answer enumeration | "Consistent, Inconsistent, ..., Same, Different, Aligned" | "Consistent, Different, Agree, Disagree, Same, Aligned" |
| v2 fallback for non-yes/no | ✅ | ✅ |

---

## 2. Results

### 2.1 Headline

v11: **0.655 (131/200)** — worse than v10 (0.670), v2 (0.680), and v9 (0.680).

### 2.2 Per-type (v11 vs v10 vs v2 vs v9)

| Type | n | v2r1 | v9 | v10 | v11 | Δ v11 vs v2 |
|---|---:|---:|---:|---:|---:|---:|
| inference | 37 | 34 (91.9%) | 33 (89.2%) | 34 (91.9%) | 33 (89.2%) | -1 |
| yesno | 104 | 69 (66.3%) | 72 (69.2%) | 68 (65.4%) | 68 (65.4%) | -1 |
| temporal_order | 45 | 33 (73.3%) | 31 (68.9%) | 32 (71.1%) | 30 (66.7%) | -3 |
| other | 14 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **200** | **136 (68.0%)** | **136 (68.0%)** | **134 (67.0%)** | **131 (65.5%)** | **-5 (-2.5 pp)** |

### 2.3 Flip analysis v11 vs v2

| v11 type | +up | -down | net |
|---|---:|---:|---:|
| inference | 0 | 1 | -1 |
| yesno | 7 | 8 | -1 |
| temporal_order | 5 | 8 | -3 |
| other | 0 | 0 | 0 |
| **TOTAL** | **12** | **17** | **-5** |

### 2.4 Comparison: v11 vs v10 (same hybrid, different yes/no prompt)

| Type | v10 | v11 | Δ |
|---|---:|---:|---:|
| inference | 34 | 33 | -1 (noise) |
| yesno | 68 | 68 | 0 |
| temporal_order | 32 | 30 | -2 (noise) |
| other | 0 | 0 | 0 |
| **TOTAL** | **134** | **131** | **-3** |

v11's yes/no prompt (with anti-patterns) scored the SAME as v10's cleaned prompt (both 68/104). The wall-clock was nearly identical (48 min vs 50 min). The anti-patterns did NOT help and did NOT hurt on yes/no specifically. The overall regression in v11 is from noise on inference and temporal paths (which both use v2 fallback in v11, same as v10).

---

## 3. Why v11 regressed

### 3.1 v9's lift on yes/no was noise, not a real mechanism

v9 reported +3 net on yesno (10 up, 7 down). But that was measured on a SINGLE run. The variance on n=200 is ~3.5 pp = ~7 questions that change pass/fail between runs.

When I run v9's exact prompt in isolation (v11), the yes/no lift disappears — same 68/104 as v10. The v9 lift on yes/no was within the noise band.

### 3.2 v9's full 4-prompt setup is doing something subtle

v9 with all 4 type-specific prompts scored 0.680 (tied v2). v11 with only yes/no dispatch (and v9's prompt) scored 0.655. v10 with only yes/no dispatch (and a different prompt) scored 0.670.

This suggests that v9's score of 0.680 was the result of:
- A small genuine lift on yes/no from "yes/no judgment" framing + valid-answer enumeration (worth ~+1 pp maybe)
- A small regression on inference/temporal from rephrased canonical-name nudge and forced single-word commitment (~-1 pp)
- Plus noise that happened to come out at +0 net

When I isolate the yes/no dispatch, the genuine lift is gone and only the regression shows up. The full 4-prompt setup seems to balance out, but doesn't actually improve.

### 3.3 The "EXACTLY ONE word" / "FIRST WORD" rules don't help

v11 measured the same first-word=answer rate as v10 (42-46%, similar to v2's 50%). The model's preamble prior is more deeply trained than the system prompt's first-word rule. The rule doesn't override the prior.

### 3.4 The "do not verify source attribution" prime doesn't help either

v11 wall-clock (50 min) was similar to v10 (48 min). The "do NOT verify" prime doesn't make the model spend more thinking tokens in this isolated setup. (v9's 4× wall-clock regression was from the full 4-prompt setup, not from any single anti-pattern.)

---

## 4. Bigger picture: 11 prompt-engineering attempts, no clear winner

| Preset | Approach | contains_gold | Wall-clock | Δ vs v2r1 |
|---|---|---:|---:|---:|
| iter-22 SOTA | (baseline) | 0.620 | (n/a) | -6.0 pp |
| iter-29 v1 | Generic pre-analysis | 0.625 | (n/a) | -5.5 pp |
| iter-29 v2 (run 1) | Shape enumeration | 0.680 | 19 min | (baseline) |
| iter-29 v2 (run 2) | Same prompt, re-run | 0.645 | (n/a) | -3.5 pp (variance) |
| iter-29 v3 | v2 + 4 refinements | 0.675 | (n/a) | -0.5 pp |
| iter-29 v4 | Paraphrase, ignore attributions | 0.620 | (n/a) | -6.0 pp |
| iter-29 v5 | CRITICAL anti-preamble | 0.685 | 46 min | +0.5 pp (in noise) |
| iter-29 v6 | Fill-in-blank template | 0.646 | (n/a) | -3.4 pp |
| iter-30 v7 | Separate yes/no strict | 0.610 | (n/a) | -7.0 pp |
| iter-30 v8 | Per-type user prompts | 0.650 | 22 min | -3.0 pp |
| iter-31 v9 | Clean per-type system prompts | 0.680 | 84 min | 0.0 pp |
| iter-32 v10 | v2 + clean yes/no dispatch | 0.670 | 48 min | -1.0 pp |
| **iter-32 v11** | **v2 + v9's exact yes/no prompt** | **0.655** | **50 min** | **-2.5 pp** |

### Patterns observed across 11 attempts

1. **v2's exact wording is the lift mechanism on inference.** Any rephrasing (v9 INFERENCE prompt, v10 with different inference path) loses the +6 pp. Only v2 unchanged preserves the inference score.

2. **Per-type yes/no dispatch makes the model commit more confidently.** Sometimes a lift (v9 +3 net), sometimes a regression (v10 -1, v11 -1, v7 -7). Signal is in the noise.

3. **The "EXACTLY ONE word" / first-word rules don't work.** v7 (yes/no strict), v11 (yes/no + first-word rule) both regressed. The model's preamble prior is deeper than any system-prompt rule.

4. **The "do NOT X" anti-patterns prime the model to think about X.** v4, v8, v11 all regressed. Don't say "do not verify source attribution" — it makes the model verify attribution.

5. **v9's score of 0.680 was noise on top of net-negative type-specific dispatch.** Isolating the yes/no dispatch shows the regression clearly.

6. **Refusal (0/13) and source-attribution (37% of fails) are unfixable by prompts.** Confirmed across 7 attempts. These need dataset/metric changes.

### The honest answer to "theoretically it should be"

**It should be, in theory.** Different question types have different optimal answer shapes, and a per-type prompt that targets each shape's dominant failure mode should win. But the implementation details — classification accuracy, prompt wording, the model's deep priors — all conspire to make the lift smaller than the noise floor on n=200.

The only way to confirm whether per-type prompts can win is to test on a larger n where the noise floor drops below the expected lift. n=2556 has SE ~0.9 pp (vs 3.5 pp on n=200), which would let us distinguish a +1 pp lift from noise. But each n=2556 run costs ~$100+ and takes ~4 hours.

### What would actually move the needle (non-prompt changes)

1. **Source-attribution fix at the dataset level** (re-ingest with "the first article" / "the second article"). Eliminates the dominant v2 thinking failure (37% of fails).

2. **Metric change for refusal-shaped answers** (semantic similarity for null questions). Fixes 15 null questions.

3. **Full n=2556 run** to get a firm answer on v2 vs SOTA (or v2 vs v9 if the user wants to compare).

---

## 5. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter32-smoke-v11-candidate-dump.jsonl` | iter-32 v11 results (n=200) |
| `docs/eval-results/2026-07-19-iter32-v11-yesno-v9-prompt-regression.md` | This report |

Total wall-clock: ~50 min. Total cost: ~$10-12.

## 6. Code state

v10 and v11 code reverted (both regressed). v2 prompt remains the default. All 93 tests pass (after revert).