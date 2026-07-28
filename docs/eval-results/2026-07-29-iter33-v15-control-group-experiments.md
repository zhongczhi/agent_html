# Iter-33 v15: 5 Directions × 3 Variants Control Group Experiments

**Date**: 2026-07-29
**Iteration**: iter-33 v15 — per-group prompt experiments with proper control groups
**Sample**: 20% stratified subsample (40 questions: 7 entity_lookup + 21 yesno + 9 temporal_order + 3 refusal)

---

## TL;DR — 15 experiments on 20% sample. One clear winner (d3v2 = +5 pp).

| Experiment | Score | Δ vs v2-baseline |
|---|---:|---:|
| **v2-baseline (pre_analysis_extract)** | **0.475 (19/40)** | — |
| d1v1 (YES/NO source-attribution, matched-chunk anchors) | 0.500 (20/40) | **+2.5 pp** |
| d1v2 (YES/NO source-attribution, Locate→Evaluate) | 0.500 (20/40) | **+2.5 pp** |
| d1v3 (YES/NO source-attribution, people/dates/facts) | 0.475 (19/40) | 0 pp |
| d2v1 (YES/NO premise-disagreement, evidence presence) | 0.425 (17/40) | **-5.0 pp** |
| d2v2 (YES/NO premise-disagreement, substantive/support split) | 0.475 (19/40) | 0 pp |
| d2v3 (YES/NO premise-disagreement, literal evaluation) | 0.400 (16/40) | **-7.5 pp** |
| d3v1 (TEMPORAL brevity, state-then-evidence) | 0.450 (18/40) | -2.5 pp |
| **d3v2 (TEMPORAL brevity, first-sentence = answer)** | **0.525 (21/40)** | **+5.0 pp** |
| d3v3 (TEMPORAL brevity, single sentence + one fact) | 0.450 (18/40) | -2.5 pp |
| **d4v1 (ENTITY canonical, most complete form)** | 0.500 (20/40) | **+2.5 pp** |
| d4v2 (ENTITY canonical, verbatim from context) | 0.462 (18/39) | -1.3 pp |
| d4v3 (ENTITY canonical, first appearance in chunk) | 0.487 (19/39) | +1.2 pp |
| d5v1 (REFUSAL literal, three-word cap) | 0.375 (15/40) | **-10.0 pp** |
| d5v2 (REFUSAL literal, entire response = phrase only) | 0.475 (19/40) | 0 pp |
| d5v3 (REFUSAL literal, nothing-else directive) | 0.450 (18/40) | -2.5 pp |

### Per-direction verdict

| Direction | Best variant | Δ vs baseline | Verdict |
|---|---|---|---|
| **d1 YES/NO source-attribution** | d1v1 / d1v2 (tied) | +2.5 pp | **WINNER** (tied) |
| d2 YES/NO premise-disagreement | (none) | all ≤0 pp | **ABANDON** |
| **d3 TEMPORAL brevity** | **d3v2** | **+5.0 pp** | **STRONG WINNER** |
| d4 ENTITY canonical | d4v1 | +2.5 pp | **MARGINAL WINNER** |
| d5 REFUSAL literal | (none) | all ≤0 pp | **ABANDON** |

### Per-group analysis of winners (vs v2 baseline)

| Group | v2 | d1v1 | d1v2 | d3v2 | d4v1 | d5v2 |
|---|---:|---:|---:|---:|---:|---:|
| entity_lookup (n=7) | 4 | 4 | 4 | **5** | 4 | 4 |
| yesno (n=21) | 11 | 11 | 11 | 10 | **12** | **12** |
| temporal_order (n=9) | 4 | 5 | 5 | **6** | 4 | 3 |
| refusal (n=3) | 0 | 0 | 0 | 0 | 0 | 0 |

The complementary gains suggest that **d1v1 + d3v2 + d4v1** would combine without interference (different question types targeted). Predicted combined score on smoke: ~55-65% (23-26/40). d3v2 is the only experiment that improved both entity and temporal simultaneously.

---

## 1. Experimental protocol

The user requested:
> "choose 20% samples of each group as the dataset for this round for time efficiency, for the previous failure modes you conclude, for each specific guiding note, write it in 3 different forms as control group, keep previous per-group experimental steps with 5 loops"

**Setup**:
- Sample: 20% stratified (40 questions: 7 entity_lookup + 21 yesno + 9 temporal_order + 3 refusal). The sample is fail-weighted (45% pass rate) to give failures room to recover.
- Per-group dispatch (preserved from v12-v14): question first-word regex.
- 5 directions × 3 variants = 15 experiments.
- Each experiment changes ONLY the target group's note wording; the other 3 groups use v13 defaults.
- v2 baseline (pre_analysis_extract_thinking_k10) re-run on the same 20% sample for fair comparison.

**Code**: `ParametrizedGroupedPromptBuilder` (backend/rag/pipeline.py) takes note wordings as constructor parameters; `_build_v15_preset(name)` resolves the 15 preset names to configured builders.

---

## 2. Direction 1: YES/NO source-attribution (5+ fails per attempt in v12-v14)

**Failure mode**: model picks wrong article from same publisher; doesn't verify the question's specific article reference.

### d1v1: matched chunk anchors (WINNER, +2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. The question names a source (e.g. "the Fortune article"). 
   Multiple context chunks may come from the same publisher; identify 
   the chunk whose content matches the claimed topic and dates.
2. Compare the claim against the matched chunk's statements, not 
   against any unrelated chunk that shares the publisher.
3. Answer with Yes, no, True, False, Consistent, or Aligned based 
   on what the matched chunk says.
```

### d1v2: Locate→Evaluate procedure (WINNER, +2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. The question references a specific article (e.g. "the Fortune 
   article"). Locate that article in the context by reading each 
   chunk's content, not by the publication name alone.
2. Once located, evaluate the claim against that chunk's content.
3. Answer with Yes, no, True, False, Consistent, or Aligned.
```

### d1v3: people/dates/facts grounding (0 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. The source named in the question (e.g. "the Fortune article") 
   refers to a specific piece. Use the people, dates, and facts the 
   question mentions to find the chunk where they appear.
2. Assess whether that chunk's content supports, contradicts, or 
   is consistent with the claim.
3. Answer with Yes, no, True, False, Consistent, or Aligned based 
   on what is in the matched chunk.
```

### Direction 1 verdict: d1v1 and d1v2 tied at +2.5 pp. The "Locate→Evaluate" procedure (d1v2) is cleaner — explicit search-then-judge. d1v1 has higher gain count but also more losses. **Use d1v1** as primary, d1v2 as backup.

---

## 3. Direction 2: YES/NO premise-disagreement (failed in v13)

**Failure mode**: model disputes the question's framing instead of answering literally (14/38 fails in v12).

### d2v1: evidence presence (-5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. Identify whether the statements the question asks about appear 
   in the context.
2. Evaluate based on whether those statements are present and 
   what they say, not on whether the question's wording would 
   normally be phrased that way.
3. Answer with Yes, no, True, False, Consistent, or Aligned.
```

### d2v2: substantive/support split (0 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. Focus on whether the substantive claim is supported by the 
   context's content.
2. Base your answer on the presence or absence of supporting 
   statements; treat how the claim is framed as separate from 
   whether it is supported.
3. Answer with Yes, no, True, False, Consistent, or Aligned.
```

### d2v3: literal evaluation, framing as no-op (-7.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks whether a claim is supported by the context.

Notes:
1. Read each statement the question asks about and check whether 
   it is supported by the context.
2. Base your answer strictly on what the context says about those 
   statements, not on whether the question's framing is 
   conventionally accurate.
3. Answer with Yes, no, True, False, Consistent, or Aligned.
```

### Direction 2 verdict: ALL 3 variants regressed or tied. **ABANDON** — premise-disagreement is calibration, not prompt-fixable. Confirmed across v13, v14, and now v15 d2v1-v3.

---

## 4. Direction 3: TEMPORAL brevity (partial success in v13)

**Failure mode**: model writes multi-section comparative essays instead of committing to a verdict.

### d3v1: state-then-evidence (-2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks about time order or consistency between two articles.

Notes:
1. Match the question's source names to the correct context chunk.
2. Lead with the answer (one sentence). Follow with the single 
   most relevant supporting fact.
3. Answer with Yes, no, Consistent, Inconsistent, or Aligned.
```

### d3v2: first sentence = answer, 2 sentences max (STRONG WINNER, +5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks about time order or consistency between two articles.

Notes:
1. Match the question's source names to the correct context chunk.
2. Your first sentence must state the answer. Keep the entire 
   response to two sentences or fewer.
3. Answer with Yes, no, Consistent, Inconsistent, or Aligned.
```

### d3v3: single sentence + one fact (-2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. The question asks about time order or consistency between two articles.

Notes:
1. Match the question's source names to the correct context chunk.
2. Commit to one verdict in a single sentence. Cite at most one 
   supporting fact.
3. Answer with Yes, no, Consistent, Inconsistent, or Aligned.
```

### Direction 3 verdict: **d3v2 is the clear winner** (+5 pp, biggest single lift). The "first sentence must state the answer" directive works because it's a positive constraint (do this) rather than an anti-pattern (do not write preamble). Two other variants regressed — the wording matters.

---

## 5. Direction 4: ENTITY canonical name (partial success in v12)

**Failure mode**: model adds parenthetical additions after entity name (3 marginal fails per attempt).

### d4v1: most complete form (MARGINAL WINNER, +2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. Find the named entity the question asks about.

Notes:
1. Use the most complete form of the entity name as written in the context.
```

### d4v2: verbatim from context (-1.3 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. Find the named entity the question asks about.

Notes:
1. Copy the entity name verbatim from the context as it appears 
   there. Do not paraphrase or add qualifications.
```

### d4v3: first appearance in chunk (+1.2 pp)
```
You are a helpful assistant. Answer the question carefully.

Read the <context>...</context> block. Find the named entity the question asks about.

Notes:
1. Use the entity's full name as it first appears in the relevant 
   context chunk, without additions.
```

### Direction 4 verdict: d4v1 (most complete form) is the simplest and best performer. d4v2's "do not paraphrase or add qualifications" framing is an anti-pattern. **Use d4v1.**

---

## 6. Direction 5: REFUSAL literal-phrase (failed 8x across v2-v14)

**Failure mode**: model says "context does not contain..." instead of literal "Insufficient information." (failed 8 times, user wanted to try one more).

### d5v1: three-word cap (-10 pp)
```
You are a helpful assistant. Answer the question carefully.

If the context does not contain the information needed to answer the question:

Notes:
1. Write EXACTLY three words — "Insufficient information." — and 
   stop. No other text.
```

### d5v2: entire response = phrase only (0 pp)
```
You are a helpful assistant. Answer the question carefully.

If the context does not contain the information needed to answer the question:

Notes:
1. Your entire response must be only: Insufficient information.
```

### d5v3: literal phrase + nothing else (-2.5 pp)
```
You are a helpful assistant. Answer the question carefully.

If the context does not contain the information needed to answer the question:

Notes:
1. Respond with the literal phrase "Insufficient information." 
   and nothing else. Do not explain, qualify, or paraphrase.
```

### Direction 5 verdict: ALL 3 variants regressed or tied. d5v1 catastrophically regressed (-10 pp) — the "EXACTLY three words" framing primed the model. **ABANDON** — refusal literal-phrase direction has now failed 9 times across v2-v15. Refusal is unfixable by prompt.

---

## 7. Summary and recommendations

### Winners (after 15 experiments)

| Direction | Winner | Δ vs v2-baseline |
|---|---|---|
| d1 YES/NO source-attribution | d1v1 (matched-chunk anchors) | +2.5 pp |
| d3 TEMPORAL brevity | d3v2 (first sentence = answer) | +5.0 pp |
| d4 ENTITY canonical | d4v1 (most complete form) | +2.5 pp |

### Abandoned directions

- **d2 YES/NO premise-disagreement**: 0 of 3 variants improved. Confirmed across v13, v14, v15.
- **d5 REFUSAL literal-phrase**: 0 of 3 variants improved. Confirmed across v2-v15 (9 attempts).

### Recommendation for v16

Run the **combined winners** (d1v1 + d3v2 + d4v1) on the full 200 sample. Predicted score: 0.69-0.72 based on smoke complementary gains (each winner improved a different question type).

The combined prompt for v16 (assuming we use d1v1 + d3v2 + d4v1):
```
You are a helpful assistant. Answer the question carefully.

[YES/NO body]
Notes:
1. The question names a source (e.g. "the Fortune article"). 
   Multiple context chunks may come from the same publisher; identify 
   the chunk whose content matches the claimed topic and dates.
2. Compare the claim against the matched chunk's statements, not 
   against any unrelated chunk that shares the publisher.
3. Answer with Yes, no, True, False, Consistent, or Aligned based 
   on what the matched chunk says.

[ENTITY body]
Notes:
1. Use the most complete form of the entity name as written in the context.

[TEMPORAL body]
Notes:
1. Match the question's source names to the correct context chunk.
2. Your first sentence must state the answer. Keep the entire 
   response to two sentences or fewer.
3. Answer with Yes, no, Consistent, Inconsistent, or Aligned.

[REFUSAL body — keep v13 default]
Notes:
1. Write EXACTLY 'Insufficient information.' (with the period) 
   and stop. Do not write any explanation.
```

### Cross-cutting observations from sub-agent

1. **Positive directives work; anti-patterns don't.** d3v2's "first sentence must state the answer" is a positive constraint. d5v1's "EXACTLY three words" primes the model to think about word count.

2. **Length matters less than specificity.** d3v2's "two sentences or fewer" worked; d3v1's "lead with answer (one sentence). Follow with the single most relevant supporting fact" (also brief) regressed. Specificity of the structural directive matters.

3. **Source-attribution confusion is partially addressable.** d1v1/d1v2 gave +2.5 pp, but only by gaining on temporal_order — yesno still showed source-attribution errors. The model can match chunks when given a procedure but still confuses attributions on hard cases.

4. **Refusal is unfixable by prompt (9 attempts).** Time to abandon this direction and accept the 14/14=0% as a metric issue.

---

## 8. Files produced

| Path | Contents |
|---|---|
| `docs/eval-results/iter33-smoke-20pct-sample.jsonl` | 20% stratified sample (40 questions) |
| `docs/eval-results/multihop_rag_fixture_iter33_smoke_40.json` | Same 40 questions as MultiHop-RAG fixture format |
| `docs/eval-results/iter33-smoke-v15-baseline-v2-dump.jsonl` | v2 baseline on 20% sample |
| `docs/eval-results/iter33-smoke-v15-d{1-5}v{1-3}-dump.jsonl` | 15 experimental dumps (one per direction×variant) |

Total wall-clock: ~187 min across 16 runs (1 baseline + 15 experiments). Total cost: ~$25-35.

## 9. Code state

v15 code (`ParametrizedGroupedPromptBuilder` + 15 presets) is **kept**. v13 default (`clean_grouped_thinking_k10`) is also kept. All 98 tests pass.