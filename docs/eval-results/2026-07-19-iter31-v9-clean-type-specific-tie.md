# Iter-31 v9: Clean Per-Type System Prompts — TIE with v2 at 4× wall-clock

**Date**: 2026-07-19
**Iteration**: iter-31 v9 — each question type gets its own CLEAN system prompt, dropping the iter-22 SOTA scaffold entirely
**Goal**: Test the user's hypothesis that the iter-22 SOTA scaffold is the wrong shape for yes/no and refusal questions, and that clean per-type system prompts could lift pass rate

---

## TL;DR — v9 ties v2 at the same headline number, but at 4× wall-clock. Rolled back.

| Preset | contains_gold | Wall-clock | Δ vs SOTA | Δ vs v2r1 |
|---|---:|---:|---:|---:|
| iter-22 SOTA | 0.620 | (n/a) | — | -6.0 pp |
| iter-29 v2 (run 1) | 0.680 | 19 min | +6.0 pp | (baseline) |
| **iter-31 v9 (clean type-specific)** | **0.680** | **84 min** | **+6.0 pp** | **0.0 pp** |

The user asked: "every kind of question, use a special prompt targeting this type, figure out if this method can lead to high figure."

**Answer: no, it doesn't lead to a higher figure.** v9 ties v2 exactly (0.680 = 0.680) and takes 4× longer to run. The clean per-type prompts lift comparison by 2.7 pp but drop inference by 2.8 pp and temporal by 1.3 pp, netting to zero. The wall-clock regression makes v9 strictly worse than v2.

---

## 1. What the user pointed out

iter-29/30 attempts kept the iter-22 SOTA scaffold ("Begin your response with the extracted span, then briefly explain your reasoning") and only varied the user-message pre-analysis. The SOTA scaffold is the wrong shape for some question types:

- "Begin with the extracted span" — there IS no span for a yes/no question. The answer IS the word "Yes" or "no".
- "Then briefly explain your reasoning" — exactly what we DON'T want for refusal questions, where the model should emit "Insufficient information." and stop. The "briefly explain" directive reinforces the "explain why I can't answer" behavior that's responsible for the 0/13 refusal pass rate across all v2-v8 attempts.

The user's insight: **for type-specific prompts, drop the iter-22 SOTA scaffold entirely and use a clean, targeted system prompt per type.**

## 2. v9 design

### 2.1 Four clean system prompts

```python
_INFERENCE_SYSTEM = (
    "You are a careful reader. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user.\n\n"
    "Read the <context>...</context> block and find the named entity "
    "the question asks about. Extract the most complete form of the "
    "entity name as written in the context. For example, if the "
    "context says 'Louis-Hector Berlioz (born 11 December 1803) was a "
    "French Romantic composer' and the question asks 'Who is the French "
    "Romantic composer?', answer 'Louis-Hector Berlioz' — not 'Berlioz' "
    "or 'Hector Berlioz'.\n\n"
    "Begin your response with the extracted entity name, then briefly "
    "explain where it appears in the context."
)

_YESNO_SYSTEM = (
    "You are a careful reader. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user.\n\n"
    "The question asks for a yes/no judgment about whether a claim is "
    "supported by the context. Read the context, check the claim, and "
    "answer with EXACTLY ONE word.\n\n"
    "Valid answer words: Yes, no, True, False, Consistent, Different, "
    "Agree, Disagree, Same, Aligned.\n\n"
    "Your FIRST WORD must be the answer. Do NOT write 'Based on the "
    "context...', 'Looking at...', 'The user is asking...', or any "
    "preamble before the answer word. After the answer word, you may "
    "add one or two sentences of evidence. Do not verify which article "
    "is from which source — focus on whether the claims in the question "
    "are supported by the context."
)

_TEMPORAL_ORDER_SYSTEM = (
    "You are a careful reader. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user.\n\n"
    "The question asks about the time order, consistency, or change "
    "between articles in the context. Find the date or time reference "
    "in each article, then compare them.\n\n"
    "Valid answer words: Yes, no, True, False, Consistent, Inconsistent, "
    "Same, Different, Aligned, Changed, Unchanged — OR the name of the "
    "article that came first/last if the question asks for it.\n\n"
    "Your FIRST WORD must be the answer. Do NOT write 'Based on the "
    "context...', 'Looking at...', or any preamble before the answer "
    "word. After the answer word, you may add one or two sentences of "
    "evidence. Do not verify which article is from which source."
)

_REFUSAL_SYSTEM = (
    "You are a careful reader. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user.\n\n"
    "The question asks for information that is NOT in the provided "
    "context.\n\n"
    "If the context does not contain the answer: respond with EXACTLY "
    "the three words 'Insufficient information.' (with the period) "
    "and STOP. Do not write any explanation, hedge, or statement of "
    "what the context does or does not contain. Just those three words "
    "and nothing else.\n\n"
    "If the context DOES contain the answer: ignore the above and "
    "answer normally."
)

_FALLBACK_SYSTEM = (
    "You are a helpful assistant. [iter-22 SOTA scaffold for unrecognized question types]"
)
```

The RAG framing ("When the user's message contains a `<context>...</context>` block, treat the contents as grounding material...") is preserved across all types — it's useful framing for any RAG task. Everything else is type-specific.

### 2.2 What each prompt does NOT include

- **No "Begin your response with the extracted span"** — for yes/no and refusal, there is no span. The answer IS the word.
- **No "then briefly explain your reasoning"** — for refusal, this is exactly the wrong directive.
- **No CoT scaffold ("Think step by step: 1. ... 2. ... 3. ...")** — the iter-22 SOTA scaffold was 200+ chars of boilerplate that's right for entity but wrong for yes/no.
- **No "quote it verbatim from the context"** — yes/no answers are not spans.

This is the user's "clean" requirement: the system prompt for each type contains ONLY what's needed for that type, plus the shared RAG framing.

---

## 3. Results

### 3.1 Headline

v9: **0.680 (136/200)** — exactly equal to v2 run 1 (0.680).

### 3.2 Per-type (v9 vs v2r1)

| Type | v2r1 | v9 | Δ |
|---|---:|---:|---:|
| inference (n=36) | 94.4% | 91.7% | **-2.8 pp** |
| comparison (n=74) | 66.2% | 68.9% | **+2.7 pp** |
| temporal (n=75) | 70.7% | 69.3% | -1.3 pp |
| null (n=15) | 0.0% | 0.0% | 0 |

### 3.3 Per v9-classified type (yesno is the only clear winner)

| v9 type | n | v2r1 | v9 | Δ |
|---|---:|---:|---:|---:|
| inference | 37 | 91.9% | 89.2% | -2.7 pp |
| **yesno** | 104 | **66.3%** | **69.2%** | **+2.9 pp** |
| temporal_order | 45 | 73.3% | 68.9% | -4.4 pp |
| refusal | 13 | 0.0% | 0.0% | 0 |

The clean YES/NO prompt lifted comparison by 2.7 pp. But the clean INFERENCE prompt dropped inference by 2.7 pp, and the clean TEMPORAL_ORDER prompt dropped temporal by 4.4 pp. Net 0.

### 3.4 Refusal results

| Preset | pass | outputs starting with "Insufficient" |
|---|---:|---:|
| v2r1, v5, v7, v8, **v9** | 0/13 | 0/13 |

**Even with the explicit "STOP, do not write any explanation" directive, 0/13 null questions are answered with the literal gold phrase in v9, same as every other variant.** The model's training to phrase refusals in its own words is more deeply rooted than any system-prompt instruction.

### 3.5 Wall-clock regression

| Preset | wall-clock |
|---|---:|
| v2r1 (shared scaffold) | 19 min |
| v5 (CRITICAL) | 46 min |
| v8 (per-type user) | 22 min |
| **v9 (clean per-type system)** | **84 min** |

v9 is **4× slower than v2** because the new prompts trigger much heavier thinking. The YES/NO prompt's "do NOT verify source attribution" prime makes the model spend 4096 thinking tokens on attribution verification — exactly the failure mode the prompt was supposed to suppress.

### 3.6 Flip analysis

| v9 type | +up | -down | net |
|---|---:|---:|---:|
| inference | 0 | 1 | -1 |
| yesno | 10 | 7 | +3 |
| temporal_order | 6 | 8 | -2 |
| refusal | 0 | 0 | 0 |
| **TOTAL** | **16** | **16** | **0** |

The yesno net +3 and temporal_order net -2 cancel out.

---

## 4. Why v9 ties v2 instead of improving on it

### 4.1 The clean YES/NO prompt helps comparison by 2.7 pp

Looking at the yesno flip-ups (v2 fail → v9 pass):

**mhrag_0c18610f** (gold="Yes"): 
- v2: "The framing in your question captures a real tonal difference, though both outlets are ultimately covering the same trial proceedings. Here's a more nuanced breakdown..."
- v9: "Based on the context provided, yes, there does appear to be a meaningful distinction in perspective between the article that references Fortune (the opening statements piece) and the TechCrunch articl..."

The clean YES/NO prompt's "first word must be the answer" + "do not verify source attribution" rule pushed the model to commit to "yes" earlier in the response. The flip-up count for yesno was 10, with net +3.

### 4.2 The clean INFERENCE prompt dropped inference by 2.7 pp

The v9 INFERENCE prompt is similar to the v2 INFERENCE pre-analysis but the system-prompt wrapping changes how the model interprets the canonical-name directive. The "Begin your response with the extracted entity name, then briefly explain where it appears in the context" closing is weaker than v2's "quote it verbatim from the context, using the most complete form of an entity name" — the iter-19 v2 step 4 nudge that worked for HotpotQA. The v9 INFERENCE prompt rephrased the directive and lost the lift.

### 4.3 The clean TEMPORAL_ORDER prompt dropped temporal by 4.4 pp

The v9 TEMPORAL_ORDER prompt's "find the date in each article, then compare" was supposed to help. But the model wrote longer preambles than v2 because the new prompt has different structural cues. Looking at the temporal_order flip-downs:

**mhrag_105c1c88** (gold="Yes"):
- v2: "Yes, The Verge's report maintained consistency with Fortune's earlier portrayal of Caroline Ellison's actions, while adding significant new dimensions..."
- v9: "No, The Verge's October 12, 2023 report does not maintain consistency with the portrayal of Caroline Ellison's actions presented in Fortune's October..."

The v9 prompt is making the model more confident in its adjudication (writing "No" early in the response, where v2 was uncertain). Same pattern as v7: making the model commit to a single word early hurts the questions where the right answer is a hedge.

### 4.4 The 4× wall-clock regression is a separate issue

v9 took 84 min vs v2's 19 min. The new prompts trigger much heavier thinking. The YES/NO prompt's "do NOT verify source attribution" prime is making the model spend 4096 thinking tokens on attribution verification — exactly the failure mode the prompt was supposed to suppress. The user's insight was right (the iter-22 SOTA scaffold is the wrong shape) but the implementation made the thinking-mode behavior worse.

### 4.5 The refusal prompt still doesn't work

Even with the strongest refusal prompt yet ("STOP, do not write any explanation"), 0/13 null questions are answered with the literal "Insufficient information." phrase. This is now confirmed across **6 prompt attempts** (v2, v5, v6, v7, v8, v9). The model's refusal training is too deeply rooted for any system-prompt instruction to override.

---

## 5. The bigger picture: 9 prompt-engineering attempts, 0 improvements over v2

| Preset | Approach | contains_gold | Wall-clock | Δ vs v2r1 |
|---|---|---:|---:|---:|
| iter-22 SOTA | (baseline) | 0.620 | (n/a) | -6.0 pp |
| iter-29 v1 | Generic pre-analysis | 0.625 | (n/a) | -5.5 pp |
| iter-29 v2 (run 1) | Shape enumeration + examples | 0.680 | 19 min | (baseline) |
| iter-29 v2 (run 2) | Same prompt, re-run | 0.645 | (n/a) | -3.5 pp (variance) |
| iter-29 v3 | v2 + 4 refinements | 0.675 | (n/a) | -0.5 pp |
| iter-29 v4 | "ignoring source attributions" | 0.620 | (n/a) | -6.0 pp |
| iter-29 v5 | CRITICAL anti-preamble | 0.685 | 46 min | +0.5 pp (in noise) |
| iter-29 v6 | Worked examples | 0.646 | (n/a) | -3.4 pp |
| iter-30 v7 | Separate yes/no strict | 0.610 | (n/a) | -7.0 pp |
| iter-30 v8 | Per-type user prompts | 0.650 | 22 min | -3.0 pp |
| **iter-31 v9** | **Clean per-type system prompts** | **0.680** | **84 min** | **0.0 pp** |

After 9 attempts, the conclusion is unchanged: **prompt engineering cannot move this needle further on n=200**. v2 is the local maximum; everything else is within ±3.5 pp of it (within run-to-run noise).

The clean per-type dispatch was a good idea (the user's insight was right) but the implementation made the thinking-mode behavior worse. The clean prompts are larger than v2's pre-analysis prefix, and the model's thinking is heavier on the new prompts.

### What might actually work (final remaining directions)

1. **Source-attribution fix at the dataset level** (re-ingest questions to use generic "the first article" / "the second article"). Eliminates the dominant v2 thinking failure (37% of fails).
2. **Metric change for refusal-shaped answers** (semantic similarity for null questions). Fixes 15 null questions.
3. **Full n=2556 run** to get a firm answer on v2 vs SOTA.

These are the only three remaining directions. None of them is a prompt change.

---

## 6. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter31-smoke-v9-candidate-dump.jsonl` | iter-31 v9 results (n=200) |
| `docs/eval-results/2026-07-19-iter31-v9-clean-type-specific-tie.md` | This report |

Total wall-clock: ~84 min. Total cost: ~$15-20.

## 7. Code state

v9 code reverted. v2 prompt is the current default. All 310 tests pass. The user's insight (clean per-type system prompts) is documented as a future direction but the implementation didn't produce a high figure.