# Chatbot Project — Iteration 10 Spec (Paraphrase Validation Gate Fix)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Bug-fix iteration: make the paraphrase generator's validation-gate success rate near 100%.

## Overview

Iteration 9 shipped the paraphrase-eval pipeline (`scripts/generate_paraphrases_hotpotqa.py` + `scripts/eval_hotpotqa.py --paraphrase-set`). The 1000-question run reported a 35% zero-coverage rate (116/334 questions got zero paraphrases), with retry success of 9.4%. The root cause is a two-part failure:

1. **Prompt structure**: the "do NOT include the answer" rule was buried in long system prompts and the tail of the user prompt. The model loses track of it on entity-as-answer questions.
2. **`temperature=0`** at both first attempt and retry: when the first attempt leaks the answer, the retry produces the *exact same output* (deterministic). The retry counter increments but nothing actually changes.

Iteration 10 fixes the generator so the validation-gate success rate is near 100%, without changing the validation gate itself (the gate is correctly enforcing a real constraint; the issue is the model not satisfying it).

## Problem

For question "Who is John Smith?" with gold answer "John Smith":
- The LLM produces "Tell me about John Smith" or "When was John Smith born?" — both are valid, fluent paraphrases.
- The 80% token-overlap gate fires because "John Smith" appears in both the question and the paraphrase.
- The LLM has no way to know "John Smith" is the gold answer to avoid, *unless* the prompt is structured clearly enough that the model actually obeys the instruction.
- At `temperature=0`, retries produce identical output → retry success 9.4%.

## Functional Requirements

### FR-35: Generator Prompt Hardening

| ID | Requirement |
|----|-------------|
| FR-35.1 | Each style's system prompt is restructured so the "do NOT include the answer" rule appears as the **first** sentence and is labeled as a "HARD RULE". The style-specific task instructions follow. |
| FR-35.2 | The system prompt begins with the line `You are a {style} paraphraser.` followed by an empty line and a "HARD RULE" block. |
| FR-35.3 | Each style's system prompt is shorter than the iter-9 version (target ≤ 200 chars); longer prompts dilute the rule's salience. |
| FR-35.4 | The user prompt keeps the gold-answer mention (so the model knows what to avoid) but formats it as the **first** line, before the paraphrasing task. |

### FR-36: Temperature Schedule

| ID | Requirement |
|----|-------------|
| FR-36.1 | First attempt: `temperature=0.3` (low variance, mostly deterministic; faster than 0 because retries have variance to escape). |
| FR-36.2 | First retry: `temperature=0.7` (medium variance; gives the model real variance to produce different output than the first attempt). |
| FR-36.3 | Second retry (only if the first retry also leaked): `temperature=1.0` (high variance; one last attempt before skipping). |
| FR-36.4 | Each retry attempt independently validates. If a retry succeeds, the success is logged. If all 3 attempts leak, the style is omitted (same as iter-9 skip-on-double-fail behavior, just with 3 attempts instead of 2). |

### FR-37: Retry Budget

| ID | Requirement |
|----|-------------|
| FR-37.1 | Each style gets up to **3 attempts total** (1 first + 2 retries) before being skipped. |
| FR-37.2 | The retry counts and outcomes are logged at INFO with `qid`, `style`, `attempt`, `temperature`, `accepted` fields. |

### FR-38: Concurrency Pacing (Rate-Limit Friendly)

| ID | Requirement |
|----|-------------|
| FR-38.1 | Between consecutive API calls within a single question's `gen_with_retry` flow, the implementation waits 5 seconds. This applies to *both* the first-attempt trio (so the 3 concurrent calls fire 5s apart) and subsequent retries within the same question. |
| FR-38.2 | Cross-question pacing: when moving from question N to question N+1, the implementation waits 5 seconds before starting question N+1's first-attempt trio. |
| FR-38.3 | Total wall-clock for the 1000-question eval generation scales accordingly: ~30 minutes (iter-9) → ~60 minutes (iter-10) for 334 effective questions. Acceptable: rate-limit hits drop from ~30% to ~5%. |

### FR-39: Backward Compatibility

| ID | Requirement |
|----|-------------|
| FR-39.1 | The validation gate (`backend.eval.paraphrases.validate_paraphrase`) is unchanged. The 80% threshold stays. |
| FR-39.2 | The output JSON schema (`{dataset_sha, schema_version, items: {qid: {paraphrases: {style: text}}}}`) is unchanged. |
| FR-39.3 | The eval pipeline (`scripts/eval_hotpotqa.py --paraphrase-set`) needs no changes — it consumes the same JSON shape. |
| FR-39.4 | Existing paraphrases JSON at `backend/storage/eval/hotpotqa/paraphrases/{dataset_sha}.json` is invalidated on the next generation run; users should run with `--force` to regenerate. |

## Non-Functional Requirements

### NFR-16: Success rate target

After the fix, the 1000-question stratified sample should produce:
- ≥ 95% of questions have all 3 styles accepted (up from 60.8%)
- ≥ 99% of questions have at least 1 style accepted (up from 65.3%)
- API calls per question: up to 9 in worst case (3 styles × 3 attempts). Typical: 3-4 (one attempt succeeds per style).

### NFR-17: Rate-limit reduction

With the 5-second pacing, the rate-limit (429) hit rate should drop below 5% (from ~30% in iter-9). Wall-clock for the 334-question effective sample should stay under 60 minutes.

### NFR-18: Test isolation

The generator tests in `scripts/tests/test_generate_paraphrases_hotpotqa.py` continue to use mocked `AsyncAnthropic` — no real API calls. Mock side-effects must reflect the new retry budget: a question whose all 3 styles leak at attempts 1-2 still gets attempts 3.

## Out of Scope (deferred to future iterations)

- Switching to a different validation gate strategy (entity-aware regex, sub-token matching, etc.). The 80% token-overlap gate stays.
- Switching to a different paraphrase model. `minimax-3` stays.
- Switching to a different embedding model. `all-MiniLM-L6-v2` stays (that change is in step 3 of the user's 3-step plan).
- Async batching across multiple questions. Iter-10 stays per-question parallelism.
- Adaptive temperature (raising temperature only when the previous attempt leaked the answer with high confidence). All 3 attempts get a fixed temperature for predictability.