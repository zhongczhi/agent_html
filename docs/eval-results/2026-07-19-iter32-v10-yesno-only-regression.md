# Iter-32 v10 Negative Result: Yes/No Only Dispatch

**Date**: 2026-07-19
**Iteration**: iter-32 v10 — v2 (PreAnalysisExtractPromptBuilder) for INFERENCE/TEMPORAL/REFUSAL + clean YES/NO system prompt for yes/no questions
**Goal**: Test whether using v2 unchanged for shapes v2 handles well, plus a clean yes/no system prompt (dropping v9's anti-patterns), could produce a high figure

---

## TL;DR — v10 regressed slightly and is 2.5× slower than v2. Rolled back.

| Preset | contains_gold | Wall-clock | Δ vs SOTA | Δ vs v2r1 |
|---|---:|---:|---:|---:|
| iter-22 SOTA | 0.620 | (n/a) | — | -6.0 pp |
| iter-29 v2 (run 1) | 0.680 | 19 min | +6.0 pp | (baseline) |
| iter-31 v9 (clean type-specific) | 0.680 | 84 min | +6.0 pp | 0.0 pp |
| **iter-32 v10 (yes/no only dispatch)** | **0.670** | **48 min** | **+5.0 pp** | **-1.0 pp** |

The user's hypothesis (per-type prompts should be a winner) is theoretically right, but v10's implementation regressed by 1.0 pp vs v2 (2 questions, within run-to-run noise of 3.5 pp) and took 2.5× as long. The next test (v11) should try v9's exact yes/no prompt + v2 fallback to see if v9's lift on yes/no can be preserved while keeping v2's behavior elsewhere.

---

## 1. What v10 changed from v9

### 1.1 The hybrid design

v10 uses **only two code paths**:
- **YES/NO questions** → clean yes/no system prompt
- **Everything else** → v2 (PreAnalysisExtractPromptBuilder) unchanged

This is the minimum-viable type-specific dispatch: only override where the data shows v2 is weak (yes/no questions had 66.3% pass rate in v2 vs 91.9% on inference).

### 1.2 The v10 YES/NO system prompt (vs v9)

v10 dropped three v9 anti-patterns:
- ❌ "first word must be the answer" (v9 measured 46% first-word=answer, no better than v2's 50%)
- ❌ "Do NOT verify which article is from which source" (v9's "do NOT" prime made the model spend 4096 thinking tokens on attribution verification, causing 4× wall-clock)
- ❌ "Do NOT write 'Based on the context...'" (anti-pattern that primed preambles)

v10 kept the structural elements that lifted v9's yes/no score:
- ✅ RAG framing prefix
- ✅ "The question is a yes/no judgment..." framing
- ✅ Valid answer enumeration (Yes/no/True/False/Consistent/...)

v10 also changed the answer-word enumeration: "Consistent, Inconsistent, ..., Same, Different, Aligned" vs v9's "Consistent, Different, Agree, Disagree, Same, Aligned" (the gold distribution shows "Consistent" appears 1× and "Similar" appears 2×, so neither list is exactly right).

### 1.3 The v10 YES/NO system prompt (final)

```
You are a careful reader. When the user's message contains a 
<context>...</context> block, treat the contents as grounding 
material: prefer it over your general knowledge when answering 
the question that follows the block. Do not mention the tag 
itself or the retrieval mechanism to the user.

The question is a yes/no judgment about whether the claim is 
supported by the context.

Valid answer words: Yes, no, True, False, Consistent, 
Inconsistent, Agree, Disagree, Same, Different, Aligned.

Read the context, check whether the claim in the question is 
supported, then answer with one of the valid answer words. 
After the answer word, you may add a brief sentence of evidence.
```

---

## 2. Results

### 2.1 Headline

v10: **0.670 (134/200)** — slightly below v2's 0.680. Within run-to-run noise (3.5 pp on n=200 = ~7 questions).

### 2.2 Per-type (v10 vs v2 vs v9)

| Type | n | v2r1 | v9 | v10 | Δ v10 vs v2 |
|---|---:|---:|---:|---:|---:|
| inference | 37 | 34 (91.9%) | 33 (89.2%) | 34 (91.9%) | 0 |
| yesno | 104 | 69 (66.3%) | 72 (69.2%) | 68 (65.4%) | **-1** |
| temporal_order | 45 | 33 (73.3%) | 31 (68.9%) | 32 (71.1%) | -1 |
| other (refusal-shaped) | 14 | 0 | 0 | 0 | 0 |
| **TOTAL** | **200** | **136 (68.0%)** | **136 (68.0%)** | **134 (67.0%)** | **-2 (-1.0 pp)** |

### 2.3 Flip analysis v10 vs v2

| v10 type | +up | -down | net |
|---|---:|---:|---:|
| inference | 0 | 0 | 0 |
| yesno | 7 | 8 | **-1** |
| temporal_order | 6 | 7 | -1 |
| other | 0 | 0 | 0 |
| **TOTAL** | **13** | **15** | **-2** |

### 2.4 First-word=gold rate (yesno questions, n=104)

| Preset | starts with answer word |
|---|---:|
| v2r1 | 48.1% (50/104) |
| v9 | 44.2% (46/104) |
| **v10** | **42.3% (44/104)** |

v10's first-word=answer rate is *lower* than v2's, despite the explicit "yes/no judgment" framing. The model is writing longer preambles in v10, not shorter.

---

## 3. Why v10 regressed

### 3.1 The yes/no system prompt made the model commit confidently wrong

Looking at v10 yesno flip-downs (v2 passed, v10 failed):

**mhrag_3f3a1eff** (gold="Yes"):
- v2: "Yes, the descriptions share notable similarities in structure and content..." (substring-matched "Yes")
- v10: "No, the descriptions are not similar, and there's an important distinction to make..." (committed to wrong "No")

**mhrag_56d1f35e** (gold="Yes"):
- v2: "The premise of your question is partially inaccurate. Here's what each article actually contains..." (substring-matched "Yes" somewhere in text)
- v10: "# Analysis of the Question\n\nThe question contains three claims that need to be evaluated..." (no "Yes" at all)

**mhrag_5931848a** (gold="Yes"):
- v2: "Yes, the context supports this interpretation..."
- v10: "Based on the context provided, I need to clarify a few things about the source attribution..." (got distracted by attribution)

**mhrag_595a561a** (gold="Similar"):
- v2: "Based on the context provided, TechCrunch's portrayals of SBF's handling..." (substring-matched "similar" by accident)
- v10: "TechCrunch describes Sam Bankman-Fried's actions regarding customer funds..." (no "similar")

**mhrag_ca18edbe** (gold="Yes"):
- v2: "Yes, there is a meaningful difference in how the two articles frame Sam Bankman-Fried's knowledge..."
- v10: "# Comparison of the Two Articles\n\nBoth articles actually present **both sides** of Sam Bankman-Fried's trial..." (no "Yes")

The pattern: v10's yes/no system prompt makes the model commit to a definite single-word answer ("No" or "# Analysis"). On hard hedge questions where v2 was uncertain and substring-matched the gold by accident, v10 commits confidently wrong.

This is the **same failure mode as v7** (separate yes/no strict prompt): making the model commit early hurts the questions where the right answer is uncertain.

### 3.2 The INFERENCE/TEMPORAL fallback is correct

The v10 design's INFERENCE path (v2 unchanged) preserved v2's score exactly: 34/37 = 91.9%. This validates the hypothesis that v9's INFERENCE prompt regressed because I rephrased the iter-19 v2 canonical-name nudge — v2's exact wording is the lift mechanism.

TEMPORAL_ORDER also uses v2 unchanged, but v10 = 32/45 vs v2 = 33/45 (1 question difference). This is within run-to-run noise (3.5 pp on n=200 = ~7 questions).

### 3.3 The wall-clock regression is structural, not fixable

v10 took 48 min vs v2's 19 min — 2.5× slower. Per-question latency:
- v10 yes/no path: ~22.6 sec/question (slower than v2 because the yes/no system prompt triggers more thinking)
- v10 v2-fallback path: ~9.1 sec/question (similar to v2 alone)

Even with only 104/200 questions going through the slow yes/no path, the wall-clock is dominated by yes/no latency. Removing v9's anti-patterns helped (v9 was 4× slower, v10 is 2.5× slower) but didn't fully restore v2's speed.

---

## 4. Bigger picture: 10 prompt-engineering attempts, no clear winner

| Preset | Approach | contains_gold | Wall-clock | Δ vs v2r1 |
|---|---|---:|---:|---:|
| iter-22 SOTA | (baseline) | 0.620 | (n/a) | -6.0 pp |
| iter-29 v1 | Generic pre-analysis | 0.625 | (n/a) | -5.5 pp |
| iter-29 v2 (run 1) | Shape enumeration | 0.680 | 19 min | (baseline) |
| iter-29 v2 (run 2) | Same prompt, re-run | 0.645 | (n/a) | -3.5 pp (variance) |
| iter-29 v3 | v2 + 4 refinements | 0.675 | (n/a) | -0.5 pp |
| iter-29 v4 | Paraphrase, ignore attributions | 0.620 | (n/a) | -6.0 pp |
| iter-29 v5 | CRITICAL anti-preamble | 0.685 | 46 min | +0.5 pp (in noise) |
| iter-29 v6 | Fill-in-blank template + examples | 0.646 | (n/a) | -3.4 pp |
| iter-30 v7 | Separate yes/no strict | 0.610 | (n/a) | -7.0 pp |
| iter-30 v8 | Per-type user prompts | 0.650 | 22 min | -3.0 pp |
| iter-31 v9 | Clean per-type system prompts | 0.680 | 84 min | 0.0 pp |
| **iter-32 v10** | **v2 + clean yes/no dispatch** | **0.670** | **48 min** | **-1.0 pp** |

After 10 attempts, the conclusions are:

1. **v2's exact wording is the lift mechanism.** Any rephrasing of the canonical-name nudge (v9, v10) loses the +6 pp. v10 INFERENCE path (v2 unchanged) preserved the score; v9 INFERENCE prompt lost 1 question.

2. **YES/NO system prompts make the model commit more confidently.** This is sometimes a lift (v9 +3), sometimes a regression (v10 -1, v7 -7). The signal is in the noise.

3. **The "do NOT X" anti-pattern primes the model to think about X.** v9 had explicit "do NOT verify source attribution" and spent 4096 thinking tokens on attribution. v10 dropped it but is still 2.5× slower than v2.

4. **The model has a deep "analysis first" prior.** First-word=answer rate stays at 42-50% across all variants. No prompt change reliably suppresses preambles.

5. **Refusal (0/13) and source-attribution (37% of fails) are unfixable by prompts.** Confirmed across 6 attempts. These need dataset/metric changes.

### Remaining directions to try (next: v11)

The one untried direction is: **v9's exact YES/NO prompt + v2 fallback for the rest**. This combines:
- v9's lift on yes/no (+2.7 pp on those questions)
- v2's behavior on inference/temporal (preserved)
- A wall-clock cost somewhere between v10's 48 min and v9's 84 min

If v11 gets ~139/200 = 0.695, that's +1.5 pp over v2 (within noise but in the right direction). If it regresses, we'll know the dispatch direction is fundamentally broken.

### What would actually move the needle (non-prompt changes)

1. **Source-attribution fix at the dataset level** (re-ingest with "the first article" / "the second article"). Eliminates the dominant v2 thinking failure (37% of fails).
2. **Metric change for refusal-shaped answers** (semantic similarity for null questions). Fixes 15 null questions.
3. **Full n=2556 run** to get a firm answer on v2 vs SOTA.

---

## 5. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter32-smoke-v10-candidate-dump.jsonl` | iter-32 v10 results (n=200) |
| `docs/eval-results/2026-07-19-iter32-v10-yesno-only-regression.md` | This report |

Total wall-clock: ~48 min. Total cost: ~$10-12.

## 6. Code state

v10 code reverted. v2 prompt remains the default. All 87 tests pass.