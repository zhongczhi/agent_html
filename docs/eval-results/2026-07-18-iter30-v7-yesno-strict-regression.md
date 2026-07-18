# Iter-30 v7 Negative Result: Separate Yes/No Strict Prompt

**Date**: 2026-07-18
**Iteration**: iter-30 v7 — separate prompt for yes/no questions (bypass the iter-22 SOTA CoT scaffold)
**Goal**: Test the hypothesis that bypassing the CoT scaffold via a separate, simpler system prompt for yes/no questions would suppress the preamble and lift pass rate

---

## TL;DR — v7 regressed. Reverted to v2.

| Preset | contains_gold | Δ vs SOTA | Δ vs v2r1 |
|---|---:|---:|---:|
| iter-22 SOTA | 0.620 | — | -6.0 pp |
| iter-29 v2 run 1 | 0.680 | +6.0 pp | (baseline) |
| **iter-30 v7 (yes/no strict)** | **0.610** | **-1.0 pp** | **-7.0 pp** |

v7 was the highest-priority next direction from the iter-29 attempt log. The hypothesis was: "bypass the CoT scaffold via a separate system prompt, and the model will lead with the answer word." **The hypothesis was wrong.** The model's "analysis first" pattern is more deeply trained than the system prompt; removing the CoT scaffold didn't help and the strict prompt made the model more confidently wrong on hard comparison questions.

---

## 1. v7 design

### 1.1 The hypothesis

From the iter-29 v2 thinking analysis: the system prompt's CoT scaffold "Begin your response with the extracted span... then briefly explain your reasoning" is the strongest source signal that the model writes analysis-then-answer. User-message rules (v3, v4, v5, v6) couldn't override it.

**v7 fix**: dispatch yes/no questions to a DIFFERENT builder that uses a SIMPLER, NON-COT system prompt. The strict system prompt has no CoT scaffold, no "begin with extracted span" directive, and an explicit "first word must be the answer word" rule.

### 1.2 The dispatch

A narrow regex on the question's first word classifies yes/no (Does/Do/Did/Is/Are/Was/Were/Has/Have/Had/Can/Could/Will/Would/Should/May/Might/Must/Shall) from entity (Who/What/Which) and other. On the smoke 200 fixture, 104 of 200 questions are classified as yes/no (74/74 comparison + 30/75 temporal). 96 fall through to the iter-22 SOTA CoT scaffold.

### 1.3 The strict system prompt (yes/no path)

```
You are a careful reader. Read the <context>...</context> block and decide
whether the claim in the question is supported by the context. Answer with
exactly ONE word — the first word of your response must be the answer.

Valid answers: Yes, no, True, False, Consistent, Inconsistent, Agree,
Disagree, Same, Different, Aligned, Insufficient information.

If the question's claim is supported by the context, the answer is Yes
(or True/Consistent/Agree/Same/Aligned). If contradicted, the answer is
no (or False/Inconsistent/Disagree/Different). If neither paragraph
contains the information, the answer is Insufficient information.

After the answer word, you may add a brief sentence of evidence. Do not
write 'Based on the context...', 'Looking at...', or any preamble before
the answer word.
```

The non-yes/no path uses the iter-21/22 SOTA CoT scaffold unchanged.

---

## 2. Results

### 2.1 Headline

v7: **0.610 (122/200)** — within noise of SOTA (0.620) but well below v2 run 1 (0.680).

### 2.2 Per-type (v7 vs v2r1)

| Type | v2r1 | v7 | Δ |
|---|---:|---:|---:|
| inference (n=36) | 94.4% | 91.7% | **-2.8 pp** |
| comparison (n=74) | 66.2% | 56.8% | **-9.5 pp** |
| temporal (n=75) | 70.7% | 62.7% | **-8.0 pp** |
| null (n=15) | 0.0% | 0.0% | 0 |

The strict prompt was supposed to help comparison and temporal. It hurt both — by 9.5 and 8.0 pp respectively.

### 2.3 First-word = answer-word rate

| Type | v2r1 | v5 | v6 | v7 |
|---|---:|---:|---:|---:|
| comparison | 51.4% | 47.3% | 49.3% | **41.9%** |
| temporal | 45.3% | 36.0% | 40.0% | 41.3% |

**The strict prompt did NOT make the model lead with the answer word more often.** v7's first-word=answer rate is actually LOWER than v2r1 for comparison. The "first word must be the answer" rule didn't override the "analysis first" deep behavior.

### 2.4 Flip analysis

| Type | +up | -down | net |
|---|---:|---:|---:|
| inference | 0 | 1 | -1 |
| comparison | 5 | 12 | -7 |
| temporal | 7 | 13 | -6 |
| null | 0 | 0 | 0 |
| **TOTAL** | **12** | **26** | **-14** |

17 of the 26 flip-downs are on yes/no questions (the questions the strict prompt was supposed to help).

---

## 3. Why v7 failed

### 3.1 The strict prompt made the model more confidently wrong

Looking at the yes/no flip-downs:

**mhrag_440e84e8** (gold="Yes"): 
- v2: "Yes, this is clearly supported. The article quotes Alex de Minaur saying, 'Hopefully we can go one better this year'..."
- v7: "Based on my reading of both articles, the comparison isn't quite accurate: ..."

**mhrag_321e9100** (gold="Yes"):
- v2: "Yes, both sources confirm Newcastle United's progression in the Carabao Cup..."
- v7: "Based on the context provided, there is no article explicitly attributed to 'Sporting News'..."

**mhrag_2b8acb60** (gold="Yes"):
- v2: "Looking at both articles: Epic Games article: The article does mention a loss for Epic Games..."
- v7: "Based on the context provided, the answer to both parts of your question is essentially **no**..."

The strict prompt's "first word must be the answer" was supposed to commit the model. Instead, it made the model **commit to confidently-wrong rejections** on hard comparison questions. v2 was uncertain on these same questions and sometimes substring-matched the gold by accident. v7 confidently says "No" or "isn't quite accurate" and then writes a long justification.

### 3.2 The preamble rate went UP, not down

v7 first words on comparison:
- "yes" (24), "based" (19), "the" (11), "no" (7) — yes/no rate is 41.9%

The strict prompt told the model "Do not write 'Based on the context...' or 'Looking at...'" — but v7 still wrote 19 "Based on" preambles on comparison questions. The model treats "Do not write X" as another instruction in a long list, not as a hard rule. Removing the CoT scaffold from the system prompt did not remove the model's "analysis first" prior.

### 3.3 Source-attribution confusion got worse

On mhrag_321e9100, v2 said "Yes, both sources confirm..." (passing by ignoring the source-attribution issue). v7 said "there is no article explicitly attributed to 'Sporting News'" — the strict prompt made the model more, not less, likely to raise source-attribution issues. v2 was using the pre-analysis to commit despite the attribution; v7 was being asked to commit cleanly, which forced the model to address the attribution question first.

### 3.4 The non-yes/no path should be unchanged but isn't

Inference questions fall through to the CoT scaffold (same system prompt as v2). But inference dropped from 94.4% to 91.7% (-2.8 pp). The 1-question difference is within noise, but it's a tiny signal that the user-message-level dispatch (regex on question start) may be affecting the model in subtle ways. Not significant enough to investigate.

---

## 4. The bigger lesson: 7 prompt-engineering attempts, 0 improvements

After iter-29 v1-v6 and iter-30 v7, I have 7 prompt-engineering attempts that have not produced a clear improvement over v2. The full table:

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
| **iter-30 v7** | **Separate yes/no strict prompt** | **0.610** | **-7.0 pp** |

What I've learned:

1. **The system's CoT scaffold dominates the user message.** All 7 attempts to change output format via user-message rules (v1, v3, v4, v5, v6) or via a separate system prompt (v7) failed to move the needle.

2. **Pattern matching beats explicit rules.** v2's lift came from implicit pattern matching to the example phrasings ("Does X suggest Y?"). v5's explicit "CRITICAL" rules and v6's worked examples did not produce a clear lift.

3. **The model's "analysis first" prior is more deeply trained than any prompt instruction can override.** The preamble rate ("Based on the context...") is 30-40% across all variants. Removing the CoT scaffold (v7) did not reduce it.

4. **The model's calibration on hard questions is bad.** When v7 made the model commit to "No" on hard comparison questions (where v2 was uncertain), the lift went negative. v2's "Based on... partially accurate" sometimes substring-matched the gold by accident.

5. **Variance is ~3.5 pp on n=200.** v2's 0.680 vs v2's re-run 0.645 is 3.5 pp. v5's 0.685 vs v2's 0.645 is 4.0 pp. These are within noise; no prompt change has demonstrated a clear improvement.

### What would actually work

The iter-29 attempt log identified three directions that aren't prompt changes:

1. **Source-attribution fix at the dataset level**: re-ingest the questions to use generic "the first article" / "the second article" instead of "the Fortune article" / "the TechCrunch article". This is a one-time data prep change that would eliminate the dominant v2 thinking failure (source-attribution confusion in 37% of fails).

2. **Metric change for refusal-shaped answers**: instead of `contains_gold("Insufficient information")`, use a semantic-similarity check or a dedicated "refused" classification. This would fix all 15 null questions without prompt changes.

3. **A full n=2556 run to get a firm answer on v2.** The v2 vs SOTA +6.0 pp on n=200 is suggestive (1.8σ) but not conclusive. n=2556 would have standard error ~0.9 pp, enough to distinguish v2 from SOTA with 6σ confidence.

### What's committed

- v7 prompt code reverted; v2 is the current default
- All 310 tests pass
- v7 eval dump at `docs/eval-results/iter30-smoke-v7-candidate-dump.jsonl` (kept for the analysis)

---

## 5. Decision: v7 not promoted, v2 stays

- v2 (`cot_extract_notitles_thinking_k10` + `pre_analysis_extract` user-message pre-analysis) is the most-tested prompt.
- 7 attempts to improve on v2 have all failed to produce a clear lift on n=200.
- Run-to-run variance is ~3.5 pp on n=200, which means single-run comparisons are noisy.
- The next step should be a different direction entirely (dataset, metric, or n=2556 confirmation), not another prompt-engineering iteration.

---

## 6. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter30-smoke-v7-candidate-dump.jsonl` | iter-30 v7 results (n=200) |
| `docs/eval-results/2026-07-18-iter30-v7-yesno-strict-regression.md` | This report |

Total wall-clock: ~23 min. Total cost: ~$4-5.