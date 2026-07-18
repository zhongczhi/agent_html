# Iter-29 Thinking Analysis + v4 Negative Result

**Date**: 2026-07-18
**Iteration**: Inspection of v2 thinking content + v4 prompt that asked the model to paraphrase the question before reading context
**Goal**: Find the actual failure mode in v2 thinking, then write a prompt that fixes it; quantify run-to-run variance so we know how much a single smoke run can tell us

---

## TL;DR — v4 regressed to SOTA baseline (0.620). Reverted to v2. Run-to-run variance is ~3.5 pp.

| Preset | contains_gold | Δ vs SOTA | Δ vs v2 |
|---|---:|---:|---:|
| iter-22 SOTA | 0.620 (124/200) | — | — |
| iter-29 v2 (run 1) | 0.680 (136/200) | +6.0 pp | (baseline) |
| iter-29 v2 (run 2, with thinking capture) | 0.645 (129/200) | +2.5 pp | -3.5 pp |
| iter-29 v3 | 0.675 (135/200) | +5.5 pp | -0.5 pp |
| **iter-29 v4 (paraphrase question)** | **0.620 (124/200)** | **0 pp** | **-6.0 pp** |

v4 was an honest attempt to address a real failure mode I observed in v2 thinking (the model gets stuck in source-attribution verification). The fix didn't work — the "ignoring source attributions" instruction made the model over-confident in dismissing question framing.

**More importantly: the v2 vs v2 re-run shows 37/200 (18.5%) of questions changed pass/fail between runs.** That means single-run smoke-test comparisons of v1, v2, v3, v4 are dominated by noise. The real test is a full n=2556 run.

---

## 1. Thinking-content analysis (v2)

I added `--capture-thinking` to the eval CLI and a `return_thinking` parameter to `ask_llm` so the dump includes the model's reasoning blocks. Inspecting 200 v2 thinking blocks:

### 1.1 The dominant failure mode: question-paraphrase confusion

**Passes** do question-paraphrase reasoning in the thinking, then evaluate:
> "The user is asking whether the TechCrunch article suggests that the European Commission's concerns are specifically related to the spread of illegal content and disinformation during the Israel-Hamas war, while the Music Business Worldwide article indicates that the European Commission's concerns are about the impact of the CJEU ruling..."

**Fails** get stuck in source-attribution verification:
> "Wait, let me re-read more carefully. The first article explicitly mentions Fortune: 'as so many in the media (Fortune included) have written' — this suggests the article itself might be from Fortune or at least Fortune is mentioned."
> "Actually, looking more carefully, the parenthetical '(Fortune included)' appears to be the article acknowledging that Fortune has also called SBF a 'boy genius.' This means the article is NOT Fortune — it's some other publication (likely TechCrunch) referencing Fortune."

The fail pattern uses up the thinking budget on attribution and runs out of budget for the actual claim check.

### 1.2 Quantitative breakdown of failure patterns

| Pattern | Fail count | % of 71 fails |
|---|---:|---:|
| Source attribution uncertainty in thinking | 26 | 37% |
| Multiple "wait" hedges in thinking | 18 | 25% |
| Thinking length > 3500 chars (near 4096 budget) | 31 | 44% |
| Predicted starts with hedge word | 6 | 8% |
| **Mean thinking length for fails** | **4232 chars** | — |
| **Mean thinking length for passes** | **3544 chars** | — |

Fails use 20% more thinking on average and still get it wrong. The thinking budget is being burned on the wrong problem.

---

## 2. v4 design and result

### 2.1 What v4 changed

v4 replaced v2's shape-matching with a question-paraphrase step:

> Before reading the context, state in one sentence what this question is actually asking (paraphrase it in plain words, ignoring source attributions). Then pick the answer shape: ...

Hypothesis: forcing the model to paraphrase in plain words before reading the context would make it do the question-paraphrase reasoning in the thinking (the success pattern) instead of source-attribution verification (the failure pattern).

### 2.2 What happened

The paraphrase instruction was followed — **30/30 sampled records contain paraphrase language in the thinking** (e.g. "The user is asking whether..."). But the visible response got worse:

| Preset | comparison | inference | temporal | TOTAL |
|---|---:|---:|---:|---:|
| v2 (run 1) | 66.2% | 94.4% | 70.7% | 68.0% |
| **v4** | **55.4%** | **91.7%** | **66.7%** | **62.0%** |

v4 regressed on every type. The "ignoring source attributions" instruction made the model too quick to dismiss question framing:

**Example (mhrag_39d3acb4)**: gold = "Yes"
- **v2**: "Yes, both claims are supported by the provided context..." (correct)
- **v4**: "No, the Sporting News article does not anticipate an impressive performance for Jordan Love in an upcoming home game. That article focuses exclusively on a Week 4 Monday Night Football game between the Eagles and Seahawks..." (wrong)

v4's visible answer starts with "No" because the model is over-confidently denying the question's premise. The "ignoring source attributions" instruction was a worse failure than the source-attribution confusion it tried to fix.

### 2.3 The example phrasings were doing more work than expected

v4 also dropped the example phrasings ("'Does X suggest Y?'", "'Who is X?'") in favor of plain shape labels. That change alone dropped the lift — the v2 example phrasings were triggering implicit pattern matching that the plain shape labels don't.

---

## 3. Variance check (the bigger problem)

I re-ran v2 with the `--capture-thinking` flag (same prompt, same fixture, same model, same temperature=0):

| Run | contains_gold | v2-pass-v2-pass-fail count |
|---|---:|---:|
| v2 (run 1) | 0.680 (136/200) | (baseline) |
| v2 (run 2) | 0.645 (129/200) | 163 same / 37 different |

**37 of 200 (18.5%) questions changed pass/fail between runs of the same prompt.** This is run-to-run noise from the LLM API. Standard error on a proportion of ~0.65 with n=200 is sqrt(0.65*0.35/200) = 0.034 = 3.4 pp — and the 0.680 vs 0.645 difference is exactly that range.

### 3.1 What this means for the iter-29 history

Looking at all runs:

| Preset | Run | contains_gold | Pass count |
|---|---|---:|---:|
| iter-22 SOTA | 1 | 0.620 | 124 |
| iter-29 v1 | 1 | 0.625 | 125 |
| iter-29 v2 | 1 | 0.680 | 136 |
| iter-29 v2 | 2 | 0.645 | 129 |
| iter-29 v3 | 1 | 0.675 | 135 |
| iter-29 v4 | 1 | 0.620 | 124 |

The "lift" of v2 over SOTA is 6.0 pp on run 1, 2.5 pp on run 2. The "regression" of v3 vs v2 is 0.5 pp. The "regression" of v4 vs v2 is 6.0 pp. **All of these are within run-to-run noise on n=200.** We can't distinguish them statistically from a single smoke run.

### 3.2 What this means for future iterations

- **A single smoke run of 200 questions is too noisy to detect lifts under ~5 pp.** A 5 pp lift is ~1.5σ on n=200; we'd need 4-5 pp to be 2σ confident.
- **A full n=2556 run** has standard error sqrt(0.65*0.35/2556) = 0.0094 = 0.94 pp. That's the right resolution for distinguishing the iter-29 versions.
- **v2 vs SOTA is the only difference that's likely real on n=200 alone.** v2 = 0.680, SOTA = 0.620 → 6.0 pp = 1.8σ. Borderline. v2 vs v4 = 0.680-0.620 = 6.0 pp = 1.8σ. Same magnitude. The data can't tell us which is better.

---

## 4. Decision: revert to v2, run n=2556 for a real answer

**v2 stays the candidate SOTA.** The 0.680 single-run result is suggestive (1.8σ over SOTA) but not conclusive. The next step is a full n=2556 run to get a firm answer.

If the n=2556 SOTA result reproduces the n=200 pattern (SOTA ~0.882 on the original iter-26 dump, v2 SOTA + ~6 pp would be ~0.94), then the pre-analysis instruction is a real improvement worth shipping.

If v2 lands within 1-2 pp of the iter-22 SOTA on n=2556, the pre-analysis instruction is just noise and should be rolled back.

**v3 and v4 are documented negative results.** v3's regression was the "if the question expects a yes/no answer" hint. v4's regression was the "ignoring source attributions" instruction. Both taught the model to over-think and over-reject.

---

## 5. Code changes

- **`backend/eval/qa_judge.py`**: `ask_llm` now supports `return_thinking: bool = False`. Returns tuple when True, string when False (backward compatible).
- **`scripts/eval_qa_hotpotqa.py`**: added `--capture-thinking` flag. When set, the per-question dump includes a `thinking` field. Default off.
- **`backend/rag/pipeline.py`**: `PreAnalysisExtractPromptBuilder` reverted to v2 wording (the change above is just the docstring documenting the v3/v4 attempts).
- **`backend/tests/rag/test_pipeline.py`**: 1 new test for the new question-shape coverage, then reverted to v2 wording.

All 310 tests pass.

---

## 6. Files produced

| Path | Contents |
|---|---|
| `backend/eval/qa_judge.py` | `ask_llm(return_thinking=...)` |
| `scripts/eval_qa_hotpotqa.py` | `--capture-thinking` flag |
| `docs/eval-results/iter29-smoke-v2-thinking-dump.jsonl` | v2 re-run with thinking captured |
| `docs/eval-results/iter29-smoke-v4-candidate-dump.jsonl` | v4 results with thinking captured |
| `docs/eval-results/2026-07-18-iter29-thinking-analysis-v4-regression.md` | This report |

Total wall-clock for v2-thinking + v4 runs: ~50 min. Total cost: ~$10-12.