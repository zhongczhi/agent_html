# Iter-29 Smoke v3 — Negative Result (Reverted to v2)

**Date**: 2026-07-18
**Iteration**: iter-29 v3 — based on inspection of v2 outputs, four targeted refinements to the pre-analysis prompt
**Goal**: Address four misleading aspects of v2 (1) shape-name emission was a no-op, (2) yes/no word list incomplete, (3) HotpotQA comparison prompt would mislead, (4) refusal phrase not used verbatim

---

## TL;DR — v3 regressed vs v2 by 0.5 pp. Reverting to v2.

| Preset | contains_gold | Δ vs SOTA | Δ vs v2 |
|---|---:|---:|---:|
| iter-22 SOTA | 0.620 (124/200) | — | — |
| iter-29 v1 (generic) | 0.625 (125/200) | +0.5 pp | — |
| iter-29 v2 (shape-enumerated) | 0.680 (136/200) | +6.0 pp | (baseline) |
| **iter-29 v3 (v2 + 4 refinements)** | **0.675 (135/200)** | **+5.5 pp** | **-0.5 pp** |

v3 was an honest attempt to address four misleading aspects I identified in v2 by inspecting actual model outputs. Three of the four changes were either no-ops or caused regressions. v2 stays the candidate SOTA.

---

## 1. The four misleading aspects I tried to fix

### 1.1 Shape-name emission was a no-op (FIX: dropped)

Inspection of all 200 v2 outputs: **zero of them emitted a shape name first.** The model ignored the "One short sentence naming the shape is enough" instruction entirely. The lift was coming from implicit pattern-matching to the example phrasings, not from a labeled first-line.

**v3 fix**: removed the "naming the shape" requirement.
**Result**: No measurable effect. v3 still doesn't emit shape names — but neither did v2, so this was a no-op.

### 1.2 Yes/no word list was incomplete (FIX: expanded)

I measured the actual gold distribution: ~7% of MultiHop-RAG temporal golds are agreement words ("Consistent", "Agreement", "Agree", "Different", "Similar"). v2's "answer with one word (Yes, no, True, or False)" didn't list these.

**v3 fix**: expanded the yes/no word list for TEMPORAL ORDERING to include 14 alternatives (Consistent, Inconsistent, Agreement, Agree, Disagree, Same, Different, Aligned, Changed, Unchanged, ...).
**Result**: Net +1 on temporal (52 → 53). v3 got 1 more temporal question right than v2 because the model could now emit "Consistent" as a valid answer.

### 1.3 HotpotQA comparison prompt would mislead (FIX: caused regression)

I measured HotpotQA comparison golds: 30.7% yes/no, 69.3% entity-type ("director", "musician", "China", "rock"). The v2 "answer with one word (Yes, no, True, or False)" for YES/NO ADJUDICATION would mislead on HotpotQA comparison.

**v3 fix**: merged YES/NO ADJUDICATION into ENTITY LOOKUP and added "if the question expects a yes/no answer, the answer may be 'Yes' or 'no' — also one word, also from the context."

**Result: -1.4 pp on comparison, -2.8 pp on inference.** The "if the question expects a yes/no answer" hint made the model start comparison responses with meta-commentary about the question's premise instead of with the answer word. Examples:
- "The premise of your question contains a misattribution that I should correct based on the context provided."
- "The user's question contains a few inaccuracies when compared to the provided context."
- "The premise of your question contains a misattributions" (and similar)
- "The question contains a premise I need to address: there is no article from 'The Roar' in the provided context."

v3 was making the model **over-cautious** about question premises, which caused it to write premise-corrections instead of the answer.

### 1.4 Refusal phrase not used verbatim (FIX: no effect)

I observed that all 15 null-question v2 outputs used phrases like "The context provided does not contain..." instead of the literal gold "Insufficient information." (with period). This is a metric-vs-judge issue: the model is semantically refusing correctly, but the gold is a specific phrase.

**v3 fix**: emphasized "use the exact words 'Insufficient information.'" in the REFUSAL directive.

**Result: 0 of 15 null questions newly correct.** The model is even more deeply trained to phrase refusals in its own words than the v3 directive can override. The "use the exact words" emphasis didn't help.

This null problem is a deeper architectural issue (the SOTA's verbatim-extract discipline is fundamentally incompatible with refusal answers). The fix would be a separate prompt with a dedicated refusal path — out of scope for iter-29.

---

## 2. Per-type lift (v3 vs v2)

| Type | v2 | v3 | v3-v2 |
|---|---:|---:|---:|
| inference (n=36) | 94.4% | 91.7% | -2.8 pp |
| comparison (n=74) | 66.2% | 64.9% | -1.4 pp |
| temporal (n=75) | 70.7% | 72.0% | +1.3 pp |
| null (n=15) | 0.0% | 0.0% | 0 |
| **TOTAL** | **68.0%** | **67.5%** | **-0.5 pp** |

The 1.3 pp temporal lift from agreement words was eaten by 2.8 pp inference regression + 1.4 pp comparison regression. Net negative.

---

## 3. Why v3 regressed

The core problem: **the v2 lift came from the model pattern-matching the question wording to the four shape bullets**. v3's changes disrupted this in two ways:

1. **The ENTITY LOOKUP "if the question expects a yes/no answer" hint** taught the model to think more carefully about whether the question expects a yes/no answer — which caused it to write premise-verification instead of the answer.
2. **The TEMPORAL ORDERING expanded word list** did help with "Consistent" golds (+1) but didn't help with the bulk of temporal questions that want "Yes/no".

v2's simpler prompt (4 shapes, each with concrete example phrasings) was the sweet spot. Adding more guidance made the model overthink.

---

## 4. Decision: revert to v2

- **v2 stays the candidate SOTA** for promotion after n=2556 confirmation.
- **v3 is a documented negative result** — useful for understanding which prompt-engineering levers work and which don't.
- **The null-question refusal issue is a separate problem** (out of scope for iter-29 prompt tweaks; would need a dedicated refusal path).

---

## 5. What I learned (process notes)

1. **Always inspect actual model outputs** before iterating. I would not have caught the "premise-correction" failure mode by just looking at aggregate metrics.
2. **Adding more guidance can hurt**. The v2 prompt's four shapes were the right level of detail; v3's expanded lists and edge-case hints introduced regression.
3. **The "if the question expects a yes/no answer" hint was the regression trigger**. v3's ENTITY LOOKUP bullet contained a conditional that made the model second-guess question premises. v2's bullets are unconditional shape matchers.
4. **Null questions need a separate path**, not a prompt-engineering fix. The SOTA's verbatim-extract discipline will continue to fail the `contains_gold` substring check until a dedicated refusal-shaped output is added.

---

## 6. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter29-smoke-v3-candidate-dump.jsonl` | iter-29 v3 results (n=200) |
| `docs/eval-results/2026-07-18-iter29-smoke-pre-analysis-v3-regression.md` | This report |

Total wall-clock: ~22.4 min. Total cost: ~$3-4. **Code reverted to v2** — no net code change from this iteration.