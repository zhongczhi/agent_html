# Validation Gate Coverage Problem

**Date**: 2026-07-08
**Status**: Known issue, accepted for current iteration
**Affects**: `scripts/generate_paraphrases_hotpotqa.py` (output: paraphrase JSON)
**Discovered during**: 1000-question stratified eval run on HotpotQA dev_distractor

---

## Problem

35% of questions (116/334 in the latest run) produce **zero paraphrases** that pass the validation gate. The remaining 65% get at least one style; 61% get all three styles. This is not random noise — it clusters around a specific failure mode.

## Data (from the 1000-question run)

| Outcome | Count | % |
|---|---|---|
| All 3 styles accepted | 203 | 60.8% |
| 1-2 styles accepted | 15 | 4.5% |
| Zero styles (all 3 rejected) | 116 | 34.7% |
| **Total questions attempted** | **334** | **100%** |

Per-style coverage (out of 334):
- `lexical`: 205 (61.4%)
- `structural`: 205 (61.4%)
- `casual`: 217 (65.0%)

API call stats from the same run:
- 962 successful (200 OK) calls
- 646 rate-limited (429) calls — MiniMax endpoint throttling; not a correctness issue, just slows the run
- 171 "leaked answer; retrying" events
- 16 retries succeeded → **retry success rate: 9.4%**
- 151 styles permanently skipped after double-failure

## Root Cause

The validation gate (`backend.eval.paraphrases.validate_paraphrase`) rejects a paraphrase when ≥80% of the gold-answer tokens appear in the paraphrase (case-insensitive, punctuation-stripped). This works well for "type-of-question" answers (`"yes"`, `"1961"`, `"The Godfather"`) but fails systematically when the gold answer is the *subject entity* of the question.

**Failure-mode examples:**
- Q: "Who is John Smith?" / gold: "John Smith" — every reasonable paraphrase contains "John Smith"; gate rejects
- Q: "Tell me about Barack Obama" / gold: "Barack Obama" — same issue
- Q: "Was the film released in 1999?" / gold: "yes" — paraphrases don't usually contain "yes", passes
- Q: "When was X born?" / gold: "1968" — paraphrases don't contain "1968", passes

**Why retries fail:** `temperature=0` (deterministic) means the LLM produces the same output for the same prompt. When the LLM has internalized "this question is about John Smith", it keeps generating paraphrases that reference John Smith. Retries succeed only ~9% of the time.

## Resolution Options

### Option 1 — Accept it (current state, recommended for now)

**What:** Document the coverage rate honestly. Per-variant `n=` is already reported in the eval output.

**Pros:** Zero new code. Current numbers are defensible. The eval report can explicitly call out coverage.

**Cons:** 35% data loss is real. Some questions have no paraphrase robustness signal.

**When to use:** Default. Especially for benchmark-style reporting where consistency matters more than maximum coverage.

### Option 2 — Lower the validation threshold (80% → ~60%)

**What:** Edit `_LEAK_OVERLAP_THRESHOLD` in `backend/eval/paraphrases.py` from `0.80` to `0.60`.

**Effect:**
- 1-token answers (`"yes"`, `"1968"`): no change — even 1 token overlap is 100% > 60%
- 2-token answers (`"John Smith"`): currently rejects 2/2 tokens (100%); at 60% would accept 1/2 (50%); still rejects 2/2 (100%)
- 5-token answers (`"The Lord of the Rings"`): currently rejects 4+/5 (80%+); at 60% would accept up to 2/5 (40%)

**Pros:** Higher coverage. Cleaner for entity-as-answer cases.

**Cons:** Permits paraphrases that contain a meaningful chunk of the answer (e.g., "Smith" in "John Smith"). May make `answer_coverage@k` less meaningful if the answer leaks into the query.

**When to use:** If coverage is more important than leak-prevention. Risky — changes the semantics of `answer_coverage_at_k` (the answer might be in the query, so FAISS trivially retrieves paragraphs containing it).

### Option 3 — Post-generation answer-token stripping

**What:** After the LLM returns a paraphrase, run a string-replace to remove gold-answer tokens. Falls back to this only when a clean paraphrase isn't reachable on first attempt.

**Pros:** Maximizes coverage. Still rejects obvious verbatim leaks.

**Cons:** Produces ungrammatical paraphrases ("When was [ANSWER] born?" → awkward). Post-processing feels like papering over the model limitation rather than fixing it.

**When to use:** Last resort. Use only if Options 1 and 2 don't yield acceptable coverage.

## Recommendation

**Option 1 (accept)** for this iteration. Document the coverage rate in the eval report. The 35% loss is concentrated in questions where paraphrasing is inherently hard (entity-as-answer), and the per-variant breakdown with explicit `n=` gives the reader enough information to interpret the metrics correctly.

## How to Re-test

If a future iteration wants to try Option 2 or 3, the test path is:

```bash
# 1. Edit _LEAK_OVERLAP_THRESHOLD in backend/eval/paraphrases.py (Option 2)
#    OR add stripping logic in scripts/generate_paraphrases_hotpotqa.py (Option 3)

# 2. Clear the existing paraphrase JSON (force regeneration)
rm backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json

# 3. Re-generate
python scripts/generate_paraphrases_hotpotqa.py --subset 1000
# Watch for "leaked answer; retrying" counts in the log

# 4. Re-run eval
python scripts/eval_hotpotqa.py --subset 1000 --no-cache

# 5. Compare coverage and metrics
# - Coverage: count entries in the JSON
# - Per-variant n= in the eval output
# - ans_cov@k per variant
```

## Related Files

- `backend/eval/paraphrases.py` — contains `_LEAK_OVERLAP_THRESHOLD = 0.80` and `validate_paraphrase()`
- `scripts/generate_paraphrases_hotpotqa.py` — calls `validate_paraphrase()` twice per style (initial + retry); has no post-processing fallback
- `docs/superpowers/plans/2026-07-08-paraphrase-eval.md` — the original plan; the 80% threshold is specified in Task 2

## Related Memory

- `feedback_compound_muting.md` — earlier feedback on treating reviewer-flagged concerns as real signals (different topic but related principle: address underlying tension, don't paper over)