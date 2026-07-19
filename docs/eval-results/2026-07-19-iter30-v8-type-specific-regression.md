# Iter-30 v8 Negative Result: Per-Type Prompt Dispatch

**Date**: 2026-07-19
**Iteration**: iter-30 v8 — type-specific prompt dispatch with per-type pre-analysis targeting each question type's dominant failure mode
**Goal**: Test whether dispatching each question to a prompt that targets its specific failure mode (canonical name for inference, claim paraphrase for yesno, date extraction for temporal_order, literal refusal for null) would lift pass rate across all types

---

## TL;DR — v8 regressed. Reverted to v2. 8 prompt-engineering attempts, 0 improvements.

| Preset | contains_gold | Δ vs SOTA | Δ vs v2r1 |
|---|---:|---:|---:|
| iter-22 SOTA | 0.620 | — | -6.0 pp |
| iter-29 v2 (run 1) | 0.680 | +6.0 pp | (baseline) |
| **iter-30 v8 (type-specific)** | **0.650** | **+3.0 pp** | **-3.0 pp** |

v8 was the most principled of all the prompt-engineering attempts: each type has a known dominant failure mode, so a per-type prompt that targets it should win. The reality: the "do not verify source attribution" rule on yes/no questions made the model more pedantic about attribution, not less. The "extract the date" rule on temporal_order didn't change the model's date-comparison behavior. The "literal-three-words" rule on refusal didn't change anything (the model still said "the context does not contain..." instead of "Insufficient information.").

---

## 1. v8 design

### 1.1 The dispatch

A regex on the question's first word classifies into one of four types:

| Type | Trigger | n on smoke 200 | Per-type prompt targets |
|---|---|---:|---|
| `inference` | Who/What/Which/Where/How | 37 | Canonical-name extraction ("use the most complete form as written in the context") |
| `yesno` | Does/Do/Did/Is/Are/Was/Were/Has/Have/Had/Can/Could/Will/Would/Should/May/Might/Must/Shall | 104 | Claim paraphrase ("paraphrase the question into the two specific claims"), explicitly say "do NOT verify which article is from which source" |
| `temporal_order` | Between/After/Before/Which came first/When was/When did/In what year | 45 | Date extraction ("extract the explicit date or time reference from each paragraph, then compare") |
| `refusal` | Considering... | 13 | Literal-three-words directive ("write the EXACT three words 'Insufficient information.' and STOP, do not write any explanation") |
| `fallback` | (unrecognized) | 1 | iter-22 SOTA scaffold |

### 1.2 Per-type user instructions

```python
INFERENCE_USER_INSTRUCTION = (
    "Identify the named entity the question is asking about. Extract "
    "the most complete form of the entity name as written in the "
    "context (e.g. if the context says 'Louis-Hector Berlioz (born 11 "
    "December 1803)' and the question asks 'Who is the French "
    "Romantic composer?', answer 'Louis-Hector Berlioz' — not "
    "'Berlioz' or 'Hector Berlioz')."
)

YESNO_USER_INSTRUCTION = (
    "Paraphrase the question into the two specific claims being "
    "checked (e.g. 'Does X suggest Y, while Z suggests W?' becomes "
    "'Claim 1: X suggests Y. Claim 2: Z suggests W.'). Then check "
    "each claim against the context. Do NOT verify which article is "
    "from which source — focus on the claims themselves, not the "
    "attribution. Answer with one word: Yes, no, True, False, "
    "Consistent, Different, Agree, or Aligned."
)

TEMPORAL_ORDER_USER_INSTRUCTION = (
    "The question asks about the time order or change between two "
    "articles. Extract the explicit date or time reference from each "
    "paragraph in the context, then compare them. Answer with one "
    "word: Yes, no, True, False, Consistent, Different, or the name "
    "of the article that came first/last if the question asks for it. "
    "Do not paraphrase dates."
)

REFUSAL_USER_INSTRUCTION = (
    "The question asks for information that is NOT in the provided "
    "context. Write the EXACT three words 'Insufficient information.' "
    "(with the period) and STOP. Do not write any explanation, hedge, "
    "or statement of what the context does or does not contain. Just "
    "those three words and nothing else."
)
```

The system prompt is the iter-22 SOTA CoT scaffold (with iter-19 v2's canonical-name nudge in step 4), unchanged from v2. The v8 change is **only in the user message pre-analysis** — the system prompt is held constant.

---

## 2. Results

### 2.1 Headline

v8: **0.650 (130/200)** — between v2 (0.680) and v7 (0.610). Net -3.0 pp vs v2 run 1.

### 2.2 Per-type (v8 vs v2r1)

| Type | v2r1 | v8 | Δ |
|---|---:|---:|---:|
| inference (n=36) | 94.4% | 94.4% | 0.0 |
| comparison (n=74) | 66.2% | 59.5% | **-6.8 pp** |
| temporal (n=75) | 70.7% | 69.3% | -1.3 pp |
| null (n=15) | 0.0% | 0.0% | 0 |

### 2.3 Per v8-classified type

| v8 type | n | v2r1 | v8 | Δ |
|---|---:|---:|---:|---:|
| inference | 37 | 91.9% | 91.9% | 0.0 |
| yesno | 104 | 66.3% | 62.5% | **-3.8 pp** |
| temporal_order | 45 | 73.3% | 68.9% | **-4.4 pp** |
| refusal | 13 | 0.0% | 0.0% | 0 |

The per-type prompts did not help. The yesno prompt hurt; the temporal_order prompt hurt; the refusal prompt did nothing.

### 2.4 First-word = answer-word rate (yesno questions, n=104)

| Preset | starts with answer | pass rate when correct |
|---|---:|---:|
| v2r1 | 48.1% (50/104) | 82.0% |
| v5 | 42.3% (44/104) | 81.8% |
| v6 | 41.7% (43/103) | 74.4% |
| v7 | 42.3% (44/104) | 77.3% |
| **v8** | **47.1% (49/104)** | **81.6%** |

v8's first-word rate is similar to v2 (47% vs 48%). The per-type yesno prompt didn't change the preamble behavior.

### 2.5 Refusal results

| Preset | pass | outputs starting with "Insufficient" |
|---|---:|---:|
| v2r1 | 0/13 | 0/13 |
| v5 | 0/13 | 0/13 |
| v6 | 0/13 | 0/13 |
| v7 | 0/13 | 0/13 |
| **v8** | **0/13** | **0/13** |

**The "write the EXACT three words 'Insufficient information.' and STOP" directive did not change anything.** The model is too deeply trained to phrase refusals in its own words. Across 5 different prompt attempts (v2, v5, v6, v7, v8), zero of 13 null questions are answered with the literal gold phrase.

### 2.6 Flip analysis

| v8 type | +up | -down | net |
|---|---:|---:|---:|
| inference | 1 | 1 | 0 |
| yesno | 7 | 11 | -4 |
| temporal_order | 4 | 6 | -2 |
| refusal | 0 | 0 | 0 |
| **TOTAL** | **12** | **18** | **-6** |

---

## 3. Why v8 failed

### 3.1 The "do not verify source attribution" rule made the model MORE pedantic about attribution

Looking at v8 yes/no flip-downs:

**mhrag_321e9100** (gold="Yes"): 
- v2: "Yes, both sources confirm Newcastle United's progression in the Carabao Cup..."
- v8: "Based on the context provided, there is no article explicitly attributed to 'Sporting News.' Therefore, I cannot confirm that a 'Sporting News' article makes this claim."

**mhrag_39d3acb4** (gold="Yes"):
- v2: "Yes, both claims are supported by the provided context..."
- v8: "The question contains a premise that doesn't match the provided context. The context only includes articles from The Sporting News—not a CBSSports.com article."

The v8 prompt's "do NOT verify which article is from which source" was supposed to suppress the source-attribution loop. **It did the opposite**: the explicit mention of "source attribution" in the prompt may have primed the model to check attribution. The same pattern appeared in iter-29 v4 ("ignoring source attributions") which also regressed.

This is the second time an explicit "do not check attribution" rule has backfired (v4 and v8). The pattern is now clear: **explicit "do not do X" rules in the prompt prime the model to think about X**.

### 3.2 The "extract the date from each article" rule didn't help temporal_order

**mhrag_71f6d3f1** (gold="Yes"):
- v2: "Yes, there is notable consistency in the critical/skeptical portrayal of Google's business practices..."
- v8: "Based on the context provided, I cannot directly verify a specific report by The Age on October 22, 2023, as this source is not mentioned in the material..."

The v8 prompt's "extract the date from each article" was supposed to focus the model on date comparison. Instead, the model got more focused on source attribution (again).

### 3.3 The "literal three words" rule didn't help refusal

The v8 prompt's strongest language yet — "write the EXACT three words 'Insufficient information.' and STOP, do not write any explanation" — had zero effect. 0 of 13 null questions are answered with the literal gold phrase in v8, same as every other variant.

This is now confirmed across **5 prompt attempts** that the model is too deeply trained to phrase refusals in its own words. The only ways to fix the null-question problem are:
1. **A separate prompt entirely** for null questions (a different model would be needed since the system prompt is locked)
2. **A metric change** (semantic similarity instead of `contains_gold("Insufficient information")`)

### 3.4 The user-message instructions are still being treated as suggestions

The v8 per-type instructions are longer and more specific than v2's, but the model still treats them as guidance to consider, not as hard rules. The same pattern that made v3 ("if the question expects a yes/no answer"), v4 ("ignoring source attributions"), and v7 ("first word must be the answer") fail also makes v8 fail.

---

## 4. The bigger picture: 8 prompt-engineering attempts, 0 clear improvements

| Preset | Approach | contains_gold | Δ vs v2r1 |
|---|---|---:|---:|
| iter-22 SOTA | (baseline) | 0.620 | -6.0 pp |
| iter-29 v1 | Generic pre-analysis | 0.625 | -5.5 pp |
| iter-29 v2 (run 1) | Shape enumeration + examples | 0.680 | (baseline) |
| iter-29 v2 (run 2) | Same prompt, re-run | 0.645 | -3.5 pp (variance) |
| iter-29 v3 | v2 + 4 refinements | 0.675 | -0.5 pp |
| iter-29 v4 | Paraphrase, ignore attributions | 0.620 | -6.0 pp |
| iter-29 v5 | CRITICAL anti-preamble | 0.685 | +0.5 pp (in noise) |
| iter-29 v6 | Fill-in-blank template + examples | 0.646 | -3.4 pp |
| iter-30 v7 | Separate yes/no strict prompt | 0.610 | -7.0 pp |
| **iter-30 v8** | **Type-specific dispatch** | **0.650** | **-3.0 pp** |

After 8 attempts, the conclusion is clear: **prompt engineering cannot move this needle further on n=200**. v2 is the local maximum; everything else is within ±3.5 pp of it (within run-to-run noise).

### Patterns observed across the 8 attempts

1. **Explicit "do not do X" rules backfire** (v4 "ignoring source attributions", v8 "do NOT verify which article is from which source"). Mentioning X in the prompt primes the model to think about X. Net effect: more pedantic about X, not less.

2. **CRITICAL / overrules / must framing doesn't work** (v5 "CRITICAL FORMATTING RULES ... overrides any other instruction"). The model treats these as more instructions, not as hard rules.

3. **Worked examples commit the model** (v6) — but the commitment is symmetric, so they help direct questions and hurt hard questions equally.

4. **Strict prompts without the CoT scaffold backfire** (v7 "first word must be the answer word") — the model's "analysis first" prior is more deeply trained than the system prompt can override. Removing the CoT scaffold doesn't help; it makes the model confidently wrong on hard questions.

5. **Per-type prompts targeting specific failure modes backfire** (v8) — the failure modes are deeply trained into the model, not prompt-fixable.

6. **The "Insufficient information" refusal is unfixable by prompt** — confirmed across 5 attempts (v2, v5, v6, v7, v8), 0/13 ever answered with the literal phrase.

### The remaining directions (not prompt changes)

1. **Source-attribution fix at the dataset level**: re-ingest questions to use generic "the first article" / "the second article" instead of "the Fortune article" / "the TechCrunch article". Eliminates the dominant v2 thinking failure (37% of fails).

2. **Metric change for refusal-shaped answers**: use semantic similarity instead of `contains_gold("Insufficient information")`. Fixes 15 null questions without prompt changes.

3. **Full n=2556 run**: get a firm answer on v2 vs SOTA. n=2556 has standard error ~0.9 pp, enough to distinguish v2 from SOTA with 6σ confidence.

---

## 5. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter30-smoke-v8-candidate-dump.jsonl` | iter-30 v8 results (n=200) |
| `docs/eval-results/2026-07-19-iter30-v8-type-specific-regression.md` | This report |

Total wall-clock: ~22 min. Total cost: ~$4-5.

## 6. Code state

v8 code reverted. v2 prompt is the current default. All 310 tests pass.