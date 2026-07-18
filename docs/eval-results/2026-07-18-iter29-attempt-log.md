# Iter-29 Attempt Log: Pre-Analysis Prompt Engineering

**Date**: 2026-07-17 → 2026-07-18
**Iteration**: Iter-29 (pre-analysis prompt instruction), versions 1-6 + thinking analysis
**Goal**: Find a pre-analysis prompt that lifts `cot_extract_notitles_thinking_k10` SOTA on MultiHop-RAG n=200 smoke set

---

## TL;DR

| Version | Approach | Result | Net change | Key lesson |
|---|---|---:|---:|---|
| iter-22 SOTA | (baseline) | 0.620 | — | HotpotQA-tuned; doesn't transfer to MultiHop-RAG's question style |
| iter-29 v1 | Generic pre-analysis | 0.625 | +0.5 pp | Pattern-matched to "temporal" but missed comparison |
| iter-29 v2 | Shape enumeration + example phrasings | 0.680 (run 1) / 0.645 (run 2) | +6.0 / +2.5 pp | Implicit pattern-matching to example phrasings works |
| iter-29 v3 | v2 + 4 refinements | 0.675 | -0.5 pp | "if the question expects a yes/no answer" hint was the regression trigger |
| iter-29 v4 | Paraphrase question, ignore attributions | 0.620 | -6.0 pp | "ignoring attributions" made model over-confidently reject framing |
| iter-29 v5 | CRITICAL anti-preamble rules | 0.685 | +0.5 pp | Rules didn't change preamble rate; temporal lift mostly noise |
| iter-29 v6 | Fill-in-the-blank template + worked examples | 0.646 (n=198) | -0.5 pp | Made model over-confident in rejecting comparison framing |

Run-to-run variance is ~3.5 pp on n=200 (37/200 questions change pass/fail between runs of the same prompt). Single-run comparisons of v3/v4/v5/v6 are within noise. **v2 is the most-tested prompt and the best single-run result, but no prompt change has demonstrated a clear improvement over v2 on a single smoke run.** The next step is a full n=2556 run to get a firm answer on v2, or a different direction entirely (dataset-level source-attribution fix, or metric change for refusals).

---

## Per-attempt deep dives

### iter-22 SOTA: `cot_extract_notitles_thinking_k10` (baseline reference)

- **What it does**: CoT scaffold + title-strip + thinking mode at 4096 budget.
- **What it produces**: long analyses (avg 1500+ chars), 0.620 on the smoke 200.
- **Failure modes**:
  - **Source-attribution confusion** in thinking: ~26 of 71 v2-fails (37%) involve the model unsure which article is from which source ("Wait, let me re-read... the parenthetical '(Fortune included)' suggests..."). Uses 4096 thinking tokens on attribution verification, runs out of budget for claim check.
  - **Hedge preambles**: 25 of 71 v2-fails (35%) start with "Based on the context provided...". The system prompt's "Begin your response with the extracted span... then briefly explain your reasoning" makes the model write analysis first.
  - **Semantic refusals**: 23 of 71 v2-fails (32%) say "I cannot confirm" or "context does not contain" when the answer is actually present. The model's training pulls toward "I don't have enough info" framing.
- **Why it works on HotpotQA**: HotpotQA's question style (bridge/comparison on Wikipedia) doesn't trigger attribution confusion (Wikipedia titles are obvious in the context) and rarely has null questions (no I-don't-know paths needed).

### iter-29 v1: generic pre-analysis

```
Before reading the context, briefly analyze the question:
(1) what entities, facts, or attributes does it ask about,
(2) what kind of material would answer it (a date, a name, a yes/no adjudication, etc.).
One short sentence for each. Then read the <context>...</context> block and answer.
```

- **Result**: 0.625 (+0.5 pp over SOTA)
- **Per-type**: temporal +5.3 pp, comparison -4.1 pp
- **Why it worked on temporal**: "what kind of material would answer it (a date, a name, a yes/no adjudication)" got the model to commit to "yes/no" format for temporal questions like "Was there a change between X and Y".
- **Why it hurt comparison**: the same instruction backfired on "Does X suggest Y, while Z" — the model wrote a multi-sentence analysis preamble instead of emitting "Yes" or "no".
- **Lesson**: vague instructions help when the model already has the right behavior, but hurt when the instruction is generic enough to apply to multiple formats.

### iter-29 v2: shape enumeration + example phrasings

```
Before reading the context, briefly identify what kind of question this is.
Pick the shape that matches, then extract accordingly:
- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): extract a single named entity (1-3 words) verbatim from the context.
- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?', 'Was there...?'): compare both sides of the claim, then answer with one word (Yes, no, True, or False).
- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', 'Was there a change between...?', 'Was X consistent with Y?'): check whether the time order or consistency holds across the two articles, then answer Yes or no.
- REFUSAL (the context may not contain the answer): if neither paragraph states what's asked, answer 'Insufficient information' rather than guessing.
One short sentence naming the shape is enough; do not re-read the question. Then read the <context>...</context> block and answer.
```

- **Result**: 0.680 run 1 / 0.645 run 2 (+6.0 / +2.5 pp over SOTA)
- **Per-type run 1**: comparison +9.5 pp, temporal +6.7 pp, inference 0 pp, null 0 pp
- **Per-type run 2**: comparison -4.1 pp, temporal +6.7 pp, inference -2.8 pp, null 0 pp
- **The lift mechanism (run 1)**: implicit pattern-matching to the example phrasings. The model sees "Does X suggest Y" in the YES/NO bullet, recognizes the same pattern in the question, and pattern-matches to emitting "Yes" early in the response.
- **Why it worked on comparison (run 1)**: the YES/NO bullet's example phrasings (Does X suggest Y? / Are A and B both?) are the exact patterns in MultiHop-RAG comparison questions. The model sees the pattern and emits "Yes" first.
- **The variance problem (run 1 vs run 2)**: 37/200 questions change pass/fail. The implicit pattern-matching is fragile — sometimes the model catches the pattern, sometimes it doesn't.
- **Failure analysis on run 2 (with thinking capture)**:
  - 25 of 71 fails start with "Based on the context..." preamble (despite the system prompt's "Begin with extracted span" + v2's "answer with one word" — the model follows the system prompt more strongly)
  - 18 of 71 fails have multiple "wait" hedges in the thinking (attribution verification loops)
  - 31 of 71 fails use >3500 thinking chars (near the 4096 budget)
  - Mean thinking for fails: 4232 chars vs passes: 3544 chars (fails burn 20% more thinking and still get it wrong)
  - **First-word analysis**: on temporal questions, pass rate is 96.8% when first word IS the answer, 38.6% when it isn't (58 pp gap). v2 doesn't actually achieve first-word=answer reliably.
- **Lesson 1**: example phrasings drive pattern matching, but the pattern matching is fragile. Variance between runs is high.
- **Lesson 2**: the system prompt's "Begin with extracted span... then briefly explain" overrides the user message's "answer with one word" for many failure cases. The system prompt has stronger training signal.
- **Lesson 3**: thinking content matters. The 37% source-attribution-confusion rate is the dominant v2 failure mode, and it happens in the thinking block, not the visible response.

### iter-29 v3: v2 + 4 refinements (regression)

- **What I tried**:
  1. Drop the "naming the shape" requirement (model ignored it anyway)
  2. Expand yes/no word list to include "Consistent", "Agreement", "Agree", "Different", etc. (for temporal/comparison golds that aren't pure yes/no)
  3. For ENTITY LOOKUP, add "if the question expects a yes/no answer, the answer may be 'Yes' or 'no'" (to cover HotpotQA comparison which is 70% entity-typed)
  4. For REFUSAL, emphasize "use the exact words 'Insufficient information.'" (to fix the metric-vs-judge issue)
- **Result**: 0.675 (-0.5 pp vs v2 run 1)
- **Per-type**: inference -2.8 pp, comparison -1.4 pp, temporal +1.3 pp, null 0 pp
- **The regression trigger**: the "if the question expects a yes/no answer" hint in ENTITY LOOKUP. The model started writing premise-correction meta-commentary: "The premise of your question contains a misattribution that I should correct based on the context provided." This is the same behavior v4 triggered.
- **The "Insufficient information" emphasis didn't help**: all 15 null questions still failed. The model is too deeply trained to phrase refusals in its own words.
- **Lesson**: explicit hints in the wrong place backfire. The "if the question expects..." parenthetical made the model over-think the question premise. Less is more.

### iter-29 v4: paraphrase question, ignore attributions (regression)

```
Before reading the context, state in one sentence what this question is actually asking
(paraphrase it in plain words, ignoring source attributions). Then pick the answer shape:
- ENTITY: the question asks for a named thing. Answer is 1-3 words from the context.
- YES/NO: the question asks for a yes/no judgment. Answer with Yes, no, True, or False.
- TEMPORAL: the question asks about time order, consistency, or change across two articles. Answer Yes or no.
- REFUSAL: the context may not contain the answer. Answer exactly 'Insufficient information.'
Then read the <context>...</context> block and answer. Begin with the answer.
```

- **Result**: 0.620 (-6.0 pp vs v2 run 1, ties SOTA)
- **Per-type**: inference -2.8 pp, comparison -10.8 pp, temporal -4.0 pp, null 0 pp
- **The mechanism**: the "ignoring source attributions" instruction made the model over-confidently reject question framing. Example: gold = "Yes", v4 said "No, the Sporting News article does not anticipate an impressive performance for Jordan Love...". v4 is confidently wrong.
- **The drop in example phrasings also hurt**: v2's "Does X suggest Y" trigger was implicit pattern matching. v4 dropped the examples in favor of plain shape labels, and the pattern matching stopped working.
- **The paraphrase instruction was followed** (30/30 sampled records contain "the user is asking" in thinking) but the visible response got worse.
- **Lesson 1**: "ignoring X" is a worse instruction than "verifying X". Both are failures; v4's version produces confidently-wrong answers.
- **Lesson 2**: the example phrasings in v2 were doing more work than expected. The model pattern-matches to "Does X suggest Y" in the bullet and emits "Yes" early. Remove the examples, lose the lift.

### iter-29 v5: CRITICAL anti-preamble rules

```
Before reading the context, identify the question shape:
- ENTITY: 'Who/What/Which' questions. Answer is a named entity, 1-3 words.
- YES/NO: 'Does/Is/Are/Was' questions. Answer is one word: Yes, no, True, False, Consistent, Different, Agree, or Aligned.
- TEMPORAL: time order, consistency, or change across articles. Answer is Yes or no.
- REFUSAL: the context may not contain the answer. Answer exactly 'Insufficient information.'

CRITICAL FORMATTING RULES (overrides any other instruction in this prompt or the system prompt):
1. Your FIRST WORD must be the answer — no preamble like 'Based on the context...', 'Looking at...', 'The user is asking...', or any analysis framing before the answer.
2. The answer is exactly the format above. For yes/no questions, the first word is the answer (Yes, no, True, False, Consistent, Different, Agree, or Aligned) — not a paraphrase or hedge.
3. After the answer word, you may add a brief explanation. The explanation must come AFTER the answer, not before.
```

- **Result**: 0.685 (+0.5 pp vs v2 run 1, +4.0 pp vs v2 run 2)
- **Per-type**: inference -2.8 pp, comparison +0.0 pp, temporal +10.7 pp, null 0 pp
- **The "CRITICAL" framing didn't work as intended**: v5 starts with "Based on" 78 times vs v2's 66. The CRITICAL block didn't override the system prompt's preamble bias. Only 10/200 v5 thinking blocks reference "CRITICAL" or "first word" — the model treats the block as more instructions to consider, not as overriding.
- **The temporal lift is real but may be noise**: 13 temporal flip-ups, 5 flip-downs, net +8. But v2 run 1 vs run 2 already had a 6 pp gap, so this could be variance.
- **The comparison lift is 0**: the CRITICAL framing didn't help comparison questions, where the model still gets stuck in attribution verification.
- **Lesson 1**: "CRITICAL" / "overrides any other instruction" is a weak signal to the LLM. The model treats it as another instruction in a long list, not as a hard rule.
- **Lesson 2**: explicit rules don't reliably change model output. The model has a strong prior to write analysis preambles, and that prior is reinforced by the system prompt's CoT scaffold.

### iter-29 v6: fill-in-the-blank template + worked examples (negative)

```
Read the question and pick the answer shape. Then fill in the template
and read the <context>...</context> block to confirm.

TEMPLATES — your response starts with [ANSWER] and continues after:
ENTITY: '[ANSWER] is the [entity type].'  e.g. 'Sam Bankman-Fried is the individual.'
YES/NO: '[Yes/no/True/False/Consistent/Different/Agree/Aligned].'  Then one sentence of evidence.
TEMPORAL: '[Yes/no].'  Then one sentence on the time order or change.
REFUSAL: 'Insufficient information.'  Then stop.

EXAMPLES OF CORRECT RESPONSES:
Q: Who is the individual associated with the cryptocurrency industry?
A: Sam Bankman-Fried is the individual. Based on the context...

Q: Does the Fortune article suggest a different perspective?
A: Yes. The Fortune article frames SBF as a 'boy' while the TechCrunch article frames him as a 'man'...

Q: Was there a change in portrayal between October 7 and October 28?
A: Yes. The October 7 article focused on opening statements, while the October 28 article focused on the verdict...

Q: What is the name of the project, as reported by Bloomberg?
A: Insufficient information. The context does not contain any Bloomberg articles about this project.
```

- **Result**: 0.646 (n=198 — 2 questions filtered by API) (-0.5 pp vs v2 run 1)
- **Per-type vs v2 run 1**: inference 0.0, comparison -8.2 pp, temporal 0.0, null 0.0
- **The comparison regression mechanism**: the worked examples include "A: Yes. The Fortune article..." but no examples of "I can't determine" or "Let me check both articles". The model became over-confident in committing to "No" on comparison questions. Example: gold = "Yes", v6 said "No, this claim is inaccurate based on the provided context." for 5 of 11 flip-downs. The "Yes/no/True/False" template taught the model to commit; it didn't teach it when to commit vs hedge.
- **The temporal effect is 0**: v6 didn't change temporal. The "Based on the context..." preamble rate did go down (78 → 59 for v5→v6 across all questions) but this didn't translate to temporal lift because the model was already emitting "Yes" on temporal when it could.
- **The inference effect is 0**: ENTITY LOOKUP template worked as expected.
- **The opening pattern "Based on the context..." went DOWN slightly** (from 72 in v5 to 59 in v6) but the comparison regression is bigger than the preamble suppression gain.
- **Lesson 1**: worked examples commit the model more strongly. If the worked examples are "Yes"-shaped (which they were in v6's examples), the model will commit to "Yes" more often — but also to "No" more often on harder comparison questions. The "Yes/no" is symmetric.
- **Lesson 2**: the "A: Yes. The Fortune article frames SBF as a 'boy'..." worked example trained the model to start with "Yes." but didn't teach the model when NOT to commit. The model's calibration on comparison is "when in doubt, commit" — and worked examples reinforce commitment, not calibration.

---

## Cross-attempt synthesis

### What's been learned about prompt engineering for LLM RAG

1. **Pattern matching beats explicit rules.** v2's example phrasings drove the lift via implicit pattern matching. v5's explicit "CRITICAL" rules did not. Worked examples (v6) extend pattern matching to output format, but commit the model more strongly — which is good for some questions, bad for others.

2. **The system prompt dominates the user message.** The system prompt's "Begin with extracted span, then briefly explain" is reinforced by the model's training (analysis-then-conclusion is the natural output pattern). User-message rules that conflict with the system prompt get ignored or partially followed. The v5 "CRITICAL ... overrides any other instruction" was too weak to overcome the system prompt.

3. **Variance is large on small smoke sets.** 37/200 questions change pass/fail between runs of the same prompt. Single-run comparisons of v3, v4, v5, v6 are within noise. v2 vs SOTA is the only difference that's likely real on n=200 alone (~1.8σ).

4. **The first-word = answer-word signal is huge on temporal questions.** 96.8% pass rate when first word is "Yes/no", 38.6% when it isn't. But making the model actually start with the answer word reliably has been hard — the model has a strong prior to preamble. v6 reduced preamble slightly but didn't move the needle on temporal.

5. **"Less is more" is mostly right but has a floor.** v3 (more rules) and v4 (paraphrase + ignore) both regressed. v2 (4 shape bullets with examples) is the sweet spot for the question-shape side.

6. **The "Based on the context..." preamble is a deep model behavior.** v2, v3, v4, v5 all had 30-40% of failures starting with this preamble. v6 reduced it to 30% (small improvement), but the comparison regression was larger.

7. **Source-attribution confusion is a thinking problem, not a prompt problem.** 26 of 71 v2 fails (37%) have the model spending its thinking budget trying to figure out which context chunk matches "the Fortune article". No prompt change has been able to fix this — the retrieved chunks don't include source attributions, so the model has to guess. A real fix would require either (a) including source attributions in the retrieved context, or (b) removing source names from the questions. Both are dataset-level changes, not prompt-level.

8. **The "Insufficient information" refusal is a metric-vs-judge issue.** The model is semantically refusing correctly but uses its own phrasing ("the context does not contain...") instead of the gold phrase. Three prompt iterations (v2, v3, v4) tried to fix this and all failed. The real fix is a separate prompt with a dedicated refusal path that emits the exact gold phrase, OR a metric change that uses semantic similarity instead of substring matching for refusal-shaped answers.

9. **Worked examples commit the model more strongly.** This is a double-edged sword: it makes the model commit to "Yes" more often (good for direct questions), but also to "No" more often on hard comparison questions (bad). The model needs calibration, not just commitment.

### What might actually work (next directions to try)

Given everything above, here are directions worth trying:

1. **A separate prompt for yes/no questions** (HIGHEST PRIORITY):
   - Detect question type at prompt-construction (regex on "Does/Is/Are/Was" at start).
   - Use a different, MUCH simpler prompt for yes/no questions.
   - The yes/no prompt could be: "Answer with exactly one word: Yes, no, True, False, Consistent, Different, Agree, Aligned, or 'Insufficient information.' The word must be the first thing in your response. No analysis, no preamble."
   - For non-yes/no questions, use the iter-22 SOTA prompt unchanged.
   - This avoids the conflict between "extract verbatim span" (good for entity) and "answer with one word" (good for yes/no). The system prompt stays untouched.

2. **Few-shot with real Q→A pairs from the dataset**:
   - Instead of synthetic worked examples (v6), give 5-10 actual examples of Q→A pairs from the training data.
   - The model would see real "Yes" / "Different" / "Sam Bankman-Fried" responses and pattern-match more reliably.
   - Adds ~500-1000 chars to the prompt; one-time dataset prep.

3. **Source-attribution fix at the dataset level**:
   - Re-ingest the questions to use generic "the first article" / "the second article" instead of "the Fortune article" / "the TechCrunch article".
   - This is a one-time data prep change that would eliminate the dominant v2 thinking failure.
   - Doesn't require re-running the SOTA on the new fixture; the v2 prompt and SOTA pipeline stay the same.

4. **Metric change for refusal-shaped answers**:
   - Instead of `contains_gold("Insufficient information")`, use a semantic-similarity check or a dedicated "refused" classification.
   - This would fix all 15 null questions without prompt changes.
   - Risk: changes the metric means previous eval results are no longer directly comparable.

5. **A "calibrated commitment" prompt**:
   - Tell the model that the answer is "Yes" or "Different" ONLY if it's clearly supported by both sources; otherwise hedge with "the context does not contain..." or similar.
   - This is a meta-rule that the previous attempts (v3, v4, v5, v6) all tried in different ways, and all failed at.
   - May not be solvable with prompt engineering alone.

### What NOT to try (don't waste cycles)

- More "CRITICAL" / "overrides" / "must" framing in the user message. The model treats these as more instructions, not as hard rules.
- Adding more example phrasings to v2. v2 has enough; the lift isn't from more examples, it's from a different mechanism.
- Changing the system prompt's CoT scaffold. The iter-21 SOTA works on HotpotQA. Don't break it for an unclear gain on MultiHop-RAG.
- More "v2 + one more rule" iterations. We've done v2, v3 (more rules, regression), v4 (paraphrase, regression), v5 (CRITICAL, no change), v6 (worked examples, regression). The space of v2-style variations is exhausted.

---

## Code state

- `backend/rag/pipeline.py`: `PreAnalysisExtractPromptBuilder` reverted to v2 (the most-tested prompt with the best single-run result).
- `backend/tests/rag/test_pipeline.py`: tests at v2.
- All 310 tests pass.
- v2 prompt is in git history as `998c7f2`.

## Files produced (per attempt)

| Attempt | Eval dump | Report |
|---|---|---|
| iter-29 v1 (smoke) | `iter29-smoke-iter29-candidate-dump.jsonl` | `2026-07-18-iter29-smoke-pre-analysis.md` |
| iter-29 v2 (smoke) | `iter29-smoke-v2-candidate-dump.jsonl` | `2026-07-18-iter29-smoke-pre-analysis-v2.md` |
| iter-29 v2 (with thinking) | `iter29-smoke-v2-thinking-dump.jsonl` | — |
| iter-29 v3 (negative) | `iter29-smoke-v3-candidate-dump.jsonl` | `2026-07-18-iter29-smoke-pre-analysis-v3-regression.md` |
| iter-29 v4 (negative) | `iter29-smoke-v4-candidate-dump.jsonl` | `2026-07-18-iter29-thinking-analysis-v4-regression.md` |
| iter-29 v5 (CRITICAL, in noise) | `iter29-smoke-v5-candidate-dump.jsonl` | — |
| iter-29 v6 (worked examples, negative) | `iter29-smoke-v6-candidate-dump.jsonl` | — |
| Cross-attempt summary | — | this document |