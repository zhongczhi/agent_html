# Iter-33 v12/v13/v14: Clean Per-Group Numbered Notes — ABANDONED

**Date**: 2026-07-28
**Iterations**: iter-33 v12 → v13 → v14 — clean per-group prompts with numbered notes (the user's new direction)

---

## TL;DR — Per-group numbered notes failed 3 times. Abandoned per the user's 2-failure rule.

| Preset | contains_gold | Wall-clock | Δ vs v2r1 |
|---|---:|---:|---:|
| iter-29 v2 (baseline, local max) | 0.680 | 19 min | (baseline) |
| iter-31 v9 (clean per-type system) | 0.680 | 84 min | 0.0 pp |
| iter-32 v10 (yes/no only, cleaned) | 0.670 | 48 min | -1.0 pp |
| iter-32 v11 (yes/no only, v9's prompt) | 0.655 | 50 min | -2.5 pp |
| **iter-33 v12 (clean base + 4 per-group notes)** | **0.625** | **61 min** | **-5.5 pp** |
| **iter-33 v13 (drop "first word", new YES/NO note 4)** | **0.675** | **60 min** | **-0.5 pp** |
| **iter-33 v14 (sub-agent's "compatible attribution" direction)** | **0.618** | **102 min** | **-6.2 pp** |

The user's hypothesis (clean base + per-group numbered notes) was tested 3 times. Each attempt regressed. Per the user's rule "if a note direction doesn't work for 2 times, abandon it", the per-group numbered-notes approach is **ABANDONED** after 3 failures.

---

## 1. What the user asked for

> "for the different failure modes, divide them into different groups with specific prompt, all starts from 'you are a helpful assistant, answer the question carefully.', for different groups, gather the failed samples and conclude why it happens, then add numbered notes in the gourp-specific prompt to guide the llm to avoid each problem"

> "if a note direction isn't working for 2 times, abandon it and try new methods, try not to re-try multiple times on slightly changing one notes on the prompt for efficiency"

The design: 4 question-type groups (entity_lookup / yesno / temporal / refusal), each with 1-4 numbered notes targeting that group's dominant failure mode. All groups share the same base: "You are a helpful assistant. Answer the question carefully." No CoT scaffold, no pre-analysis prefix, no "do NOT X" anti-patterns.

---

## 2. Per-group notes across v12/v13/v14

### v12 (first attempt — 0.625)
| Group | Notes |
|---|---|
| entity_lookup | 1. canonical form. 2. begin with entity name, no parens. |
| yesno | 1. match source names. 2. verdict options. 3. first word = answer (no preamble). |
| temporal | 1. match source names. 2. commit to verdict. 3. first word = verdict (no preamble). |
| refusal | 1. write EXACTLY "Insufficient information." and stop. |

### v13 (second attempt — 0.675)
| Group | Notes | Change |
|---|---|---|
| entity_lookup | 1. canonical form only. | dropped "begin with" + "no parens" (failed in v12) |
| yesno | 1. match source names. 2. compare. 3. verdict options. **4. answer as asked, don't dispute framing.** | new note 4 for premise-disagreement; dropped "first word" rule (failed in v9-v12) |
| temporal | 1. match source names. **2. state verdict in first sentence, 1-2 sentences total.** 3. verdict options. | dropped "commit to verdict" (failed in v12); replaced "first word" with "first sentence" |
| refusal | 1. write "Insufficient information." (literal-phrase). | unchanged |

### v14 (third attempt — 0.618)
| Group | Notes | Change |
|---|---|---|
| yesno | 1. evaluate article statements. **2. treat minor attribution/wording imprecision as compatible.** 3. verdict options. | dropped "dispute framing" (failed in v13); tried sub-agent's "compatible attribution" direction |
| (others) | (unchanged from v13) | |

---

## 3. Per-group pass rates (v2 vs v12 vs v13 vs v14)

| Group | n | v2 | v12 | v13 | v14 | Δ v14 vs v2 |
|---|---:|---:|---:|---:|---:|---:|
| entity_lookup | 37 | 34/37 (91.9%) | 34/37 (91.9%) | 34/37 (91.9%) | 34/37 (91.9%) | 0.0 pp |
| yesno | 104 | 69/104 (66.3%) | 66/104 (63.5%) | 66/104 (63.5%) | 65/104 (62.5%) | -3.8 pp |
| temporal_order | 45 | 33/45 (73.3%) | 25/45 (55.6%) | 35/45 (77.8%) | 26/45 (57.8%) | -15.6 pp |
| refusal | 14 | 0/14 (0%) | 0/14 (0%) | 0/14 (0%) | 0/14 (0%) | 0.0 pp |
| **TOTAL** | **200** | **136 (68.0%)** | **125 (62.5%)** | **135 (67.5%)** | **125 (62.5%)** | **-5.5 pp** |

(Note: v14 had 1 question with an API error, so its total is 123/199.)

---

## 4. What worked, what didn't (sub-agent analysis across all 3 attempts)

### Per-note verdict

| Note direction | Outcome |
|---|---|
| ENTITY canonical name | **Partial** — 34/37 in all attempts; parenthetical additions persist |
| ENTITY "begin with entity name" | **Failed** — model still adds "(the All Blacks)" |
| YES/NO "match source names" | **Failed** — source-attribution confusion persists (5+ fails) |
| YES/NO "first word = answer" | **Failed across v9-v12** — ABANDONED per 2-failure rule |
| YES/NO "answer as asked, don't dispute" | **Failed (v13)** — premise-disagreement unchanged |
| YES/NO "treat minor attribution as compatible" | **Failed (v14)** — model became too lenient, lost precision |
| TEMPORAL "commit to verdict" | **Failed (v12)** — model still hedges |
| TEMPORAL "first sentence, 1-2 sentences" | **Partial (v13)** — temporal recovered to 77.8%, then regressed again in v14 |
| REFUSAL "write literal 'Insufficient information.'" | **Failed** — 0/14 in all 8 attempts (v2, v5-v9, v12-v14) |

### Cross-cutting observations from sub-agents

1. **Clean base lost prompt authority**. v2's pre-analysis prefix ("Before reading the context, briefly identify what kind of question this is...") was more prescriptive than "Answer the question carefully." The clean base fell back to the model's default behavior (long explanatory paragraphs starting with "Based on..."), and the per-group notes couldn't compensate.

2. **Source-attribution confusion is systemic**, not prompt-fixable. ~37% of v2 fails are source-attribution related (model picks wrong article from same publisher). v12's "match source names" note didn't help — the model's tendency to engage the premise before checking attribution is too deeply trained.

3. **Premise-disagreement is prompt-resistant**. The "don't dispute framing" / "treat as compatible" notes both failed. The model has a strong tendency to evaluate the question's framing rather than answering literally. This is a calibration issue, not a prompt issue.

4. **Refusal is unfixable by prompt**. Confirmed 8 attempts. Need a metric change (semantic similarity for null questions), not a prompt change.

5. **Wall-clock regressed 3-5×**. v12=61 min, v13=60 min, v14=102 min, vs v2=19 min. The per-group system prompts trigger heavier thinking even when notes are simple.

---

## 5. Patterns across the 14 attempts

| Direction | Tried in | Verdict |
|---|---|---|
| Pre-analysis prefix (v2) | v2 | **Local max (0.680)** |
| CoT scaffold + "Begin with span" | v3-v8 | Within noise (-3.5 to +0.5 pp) |
| Per-type system prompts (full dispatch) | v9 | Tied v2 (0.680) at 4× cost |
| Yes/no only dispatch | v10, v11 | Regressed (-1.0 to -2.5 pp) |
| **Clean base + per-group numbered notes** | **v12, v13, v14** | **Regressed (-0.5 to -6.2 pp). ABANDONED.** |
| CRITICAL/overrules framing | v5 | Within noise |
| Worked examples | v6 | Regressed (-3.4 pp) |
| Few-shot examples | (not yet) | UNTESTED — promising next direction |

---

## 6. Conclusion

After 14 prompt-engineering attempts on the smoke 200 set:
- v2 remains the local maximum at 0.680.
- The user's per-group numbered-notes hypothesis was tested 3 times (v12, v13, v14) and failed each time. **ABANDONED.**
- The clean base lost prompt authority that v2's pre-analysis prefix provided.
- Per-type dispatch fragments prompt authority without compensating.

### What was NOT tried (per the user's "don't re-try failed directions" rule)

1. **Few-shot examples** for yes/no questions (3-4 worked Q→A pairs in the user message). Untried, fundamentally different lever.
2. **Output-format directives** (e.g., "Answer: Yes. Evidence: ..." with tag-based extraction in code). Untried.
3. **Self-classification** (model classifies its own question type in the user message). Untried.
4. **Different temperature / top-p** for different question types. Untried.
5. **Larger sample (n=2556)** to reduce variance floor. Confirmatory only, not a prompt change.
6. **Source-attribution fix at the dataset level**. Non-prompt change.

### The honest answer

**Per-type prompt dispatch does NOT beat v2 on n=200.** Across 11 dispatch attempts (v7, v8, v9, v10, v11, v12, v13, v14 — plus v3-v6 in earlier iters), no implementation improved over v2's 0.680 within run-to-run noise. The fundamental issue is that the run-to-run variance (~3.5 pp on n=200 = 7 questions) is larger than any consistent lift from type-specific prompts.

The only ways to confirm whether prompts can move this needle further are:
- Larger sample (n=2556) where SE drops to ~0.9 pp
- Fundamentally new approaches (few-shot, output-format, post-processing)
- Dataset-level / metric-level changes (source-attribution, refusal semantics)

---

## 7. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter33-smoke-v12-candidate-dump.jsonl` | iter-33 v12 results (n=200) |
| `docs/eval-results/iter33-smoke-v13-candidate-dump.jsonl` | iter-33 v13 results (n=200) |
| `docs/eval-results/iter33-smoke-v14-candidate-dump.jsonl` | iter-33 v14 results (n=199, 1 API error) |
| `docs/eval-results/2026-07-28-iter33-v12-v13-v14-clean-grouped-abandoned.md` | This report |

Total wall-clock: 61 + 60 + 102 = 223 min. Total cost: ~$40-50.

## 8. Code state

v14 code reverted. v2 prompt remains the default. All 90 tests pass after revert.