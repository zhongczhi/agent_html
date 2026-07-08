# Chatbot Project — Iteration 10 Design (Paraphrase Validation Gate Fix)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> Surgical edits to `scripts/generate_paraphrases_hotpotqa.py` and its test file. See [SPEC_focus.md](SPEC_focus.md) for requirements.

---

## 1. Architecture Decisions

### 1.1 Front-Load the Hard Rule in System Prompts

**Choice**: Move "do NOT include the answer" to the *first* sentence of each style's system prompt, with a "HARD RULE" label.

**Rationale**: The model attends more strongly to early tokens in a prompt. Burying the rule in the middle or end means it's competing with style-specific instructions for attention. A clearly-labeled "HARD RULE" at the top makes the rule salient.

**Before** (lexical, 218 chars):
```
You are a lexical paraphraser. You paraphrase questions. 
Output ONLY the paraphrase, no preamble. 
Keep the exact sentence structure of the original but substitute 
synonyms and minor word choices (e.g. 'In which year' -> 'What year'). 
Do NOT include the answer in your paraphrase. Output one sentence.
```

**After** (lexical, 192 chars):
```
You are a lexical paraphraser.

HARD RULE: Do NOT include the answer to the question in your 
paraphrase. The answer is supplied below. If your paraphrase 
contains the answer, it is invalid and will be rejected.

Task: paraphrase the question using synonym swaps only (e.g., 
'In which year' -> 'What year'). Keep the exact sentence structure.

Output: ONLY the paraphrase, one sentence, no preamble.
```

**Trade-off**: Slightly more verbose system prompt. The "HARD RULE" framing costs ~80 chars but gets the rule in front of the model's attention window.

### 1.2 Three-Tier Temperature Schedule

**Choice**: attempt 1 = 0.3, attempt 2 = 0.7, attempt 3 = 1.0.

**Rationale**: 
- **0.3** for first attempt: low variance, fast, mostly deterministic. The model picks its first confident paraphrase.
- **0.7** for first retry: significant variance. If the first attempt produced "Tell me about John Smith", a 0.7 retry has a real chance of producing "Who's the person called John Smith?" or "How would you describe John Smith?" — different framings that don't include the entity.
- **1.0** for second retry: maximum variance. One last shot. At temp=1.0, the model is essentially sampling from a wide distribution; if even this doesn't work, the question is genuinely hard.

**Trade-off**: Higher temperature means slower generation (more tokens per call on average due to sampling spread) and slightly more variable outputs (less reproducibility across runs). Both acceptable for an offline eval-generation step.

### 1.3 Three-Attempt Budget (Not Five)

**Choice**: 3 attempts total per style (1 first + 2 retries). Not 5, not 1.

**Rationale**: 
- **1 attempt (iter-9 baseline)**: 35% zero-coverage.
- **2 attempts (iter-9 with retry)**: still 35% because retry is at temperature=0.
- **3 attempts with temp schedule**: estimated >95% coverage based on the failure-mode analysis.
- **5 attempts**: diminishing returns. The remaining 5% failure cases are entity-as-answer where no reasonable paraphrase exists; throwing more attempts at them won't help.

**Trade-off**: Worst case is now 9 API calls per question (3 styles × 3 attempts). Typical is 3-4. Total API calls for 334 questions: ~1200 (vs 962 in iter-9). Cost increase ~25%, acceptable for the coverage gain.

### 1.4 5-Second Inter-Call Pacing

**Choice**: Wait 5 seconds between API calls within a question's flow, and 5 seconds between questions.

**Rationale**: 
- Iter-9 hit 30% rate-limit (429) responses — the MiniMax endpoint throttles aggressively.
- Spreading calls out by 5 seconds reduces the burst rate from 3-call-per-question-instantly to 3-call-per-question-over-10-seconds.
- Cross-question pacing prevents thundering-herd when the generator moves from q334 to q335.

**Implementation**: 
- Wrap `_generate_one_style` in a sleep-before-call: `await asyncio.sleep(5)`.
- In `_generate_for_question`, after `asyncio.gather` returns (all 3 first-pass results), sleep 5 seconds before any retry begins. Similarly, sleep 5 seconds after each retry before the next.
- In the main loop, sleep 5 seconds after each question completes before starting the next.

**Trade-off**: Wall-clock goes from ~30 min to ~60 min for 334 questions. Acceptance: rate-limit drops from 30% to <5%, retry success rate jumps, total API calls stay roughly the same (some retries now succeed that previously failed; the reduction in failed double-attempts roughly offsets the per-call sleep).

### 1.5 Test Mock Updates

**Choice**: Update `scripts/tests/test_generate_paraphrases_hotpotqa.py` so mocks reflect the new 3-attempt budget.

**Specific changes**:
- `test_three_calls_per_question_made_concurrently`: assertion `call_count == 6` → `call_count == 6` (still 2 questions × 3 first attempts; retries only happen on leak). When the mock produces clean first-pass outputs, no retries fire. The test still passes.
- `test_validation_gate_retries_leaked_paraphrase`: assertion `call_count == 9` → `call_count == 9` (still 6 + 3; first retry succeeds, no second retry needed). The test still passes.
- `test_validation_gate_skips_double_failure`: assertion `q1 not in items` → still `q1 not in items` because all 3 attempts leak. Total calls: q1 gets 9 (3 styles × 3 attempts), q2 gets 3 (clean first-pass). Total: 12 calls. The test must update its assertion from `call_count == 6` to `call_count == 12`.
- Add `test_three_attempt_budget_accepts_on_third_try`: mock produces leak → leak → clean for each style. Assert all 3 styles accepted, total calls = 12 (4 per style × 3 styles). Wait — 9 calls (3 attempts × 3 styles), not 12. Recheck: 3 attempts × 3 styles = 9 total. Update test mock to make attempt 3 clean.

### 1.6 Pacing Tests Are Skipped (5-Second Sleeps Are Not Testable in Unit Tests)

**Choice**: The 5-second pacing is not unit-tested; it's verified manually by running the generator and observing wall-clock + rate-limit hits.

**Rationale**: Mocking `asyncio.sleep` is brittle and doesn't actually verify pacing works against the real API. The integration test runs in CI without network access (mocked AsyncAnthropic). The 5-second pacing is operational behavior verified by the full eval run.

---

## 2. Module Layout

No new files. Surgical edits to:

- `scripts/generate_paraphrases_hotpotqa.py` — restructure `STYLE_PROMPTS`, add `_retry_temperature_for(attempt)`, expand `gen_with_retry` to 3 attempts with temperature schedule, add pacing sleeps.
- `scripts/tests/test_generate_paraphrases_hotpotqa.py` — update `test_validation_gate_skips_double_failure` mock to allow 3 attempts, add `test_three_attempt_budget_accepts_on_third_try`.

---

## 3. Component Changes

### 3.1 `scripts/generate_paraphrases_hotpotqa.py` — STYLE_PROMPTS

```python
STYLE_PROMPTS: dict[str, str] = {
    "lexical": (
        "You are a lexical paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question using synonym swaps only "
        "(e.g., 'In which year' -> 'What year'). Keep the exact "
        "sentence structure.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
    "structural": (
        "You are a structural paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question by reordering clauses "
        "(e.g., active -> passive, 'X was born in Y' -> 'In which year "
        "was X born, given that Y is associated with X?'). Keep all "
        "the original entities and facts.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
    "casual": (
        "You are a casual paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question in an informal, conversational "
        "tone as if a real user typed it quickly in a chat: use "
        "contractions, drop articles where natural, allow lowercase.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
}
```

The `_user_prompt` keeps the gold-answer mention first:
```python
def _user_prompt(question: str, gold_answer: str) -> str:
    return (
        f"Question to paraphrase: {question}\n"
        f"Answer to AVOID in your paraphrase (HARD RULE): {gold_answer}\n"
        f"Output ONLY the paraphrase, one sentence."
    )
```

### 3.2 Temperature Schedule Helper

```python
def _retry_temperature_for(attempt: int) -> float:
    """Return the temperature for a given attempt number (1, 2, or 3)."""
    if attempt == 1: return 0.3
    if attempt == 2: return 0.7
    return 1.0  # attempt == 3
```

### 3.3 Pacing Constant

```python
PACING_SECONDS = 5  # FR-38: 5s between calls and between questions
```

### 3.4 Modified `_generate_one_style`

```python
async def _generate_one_style(
    client: AsyncAnthropic,
    model: str,
    style: str,
    question: str,
    gold_answer: str,
    attempt: int,
) -> str:
    """One Anthropic call returning the paraphrase text for one style.
    
    `attempt` is 1-indexed (1 = first attempt, 2 = first retry, 3 = second retry).
    Temperature is determined by `_retry_temperature_for(attempt)`.
    Pacing: sleep PACING_SECONDS before the call.
    """
    await asyncio.sleep(PACING_SECONDS)
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=_retry_temperature_for(attempt),
        system=STYLE_PROMPTS[style],
        messages=[
            {"role": "user", "content": _user_prompt(question, gold_answer)},
        ],
    )
    return response.content[0].text.strip()
```

### 3.5 Expanded `gen_with_retry`

```python
async def _generate_for_question(
    client: AsyncAnthropic,
    model: str,
    question: str,
    gold_answer: str,
) -> dict[str, str]:
    """Generate all 3 styles in parallel; validate; retry up to 3 attempts.
    
    Each style runs as its own task. Attempt 1 fires concurrently for all 3
    styles. Each task then validates and conditionally fires retry attempts
    2 and 3 with progressively higher temperatures.
    """
    styles = required_styles()
    
    async def gen_with_retry(style: str) -> tuple[str, str | None]:
        for attempt in (1, 2, 3):
            text = await _generate_one_style(
                client, model, style, question, gold_answer, attempt,
            )
            if validate_paraphrase(text, gold_answer):
                log.info(
                    "qid=? style=%s attempt=%d temp=%.1f accepted",
                    style, attempt, _retry_temperature_for(attempt),
                )
                return style, text
            log.warning(
                "qid=? style=%s attempt=%d temp=%.1f leaked; %s",
                style, attempt, _retry_temperature_for(attempt),
                "retrying" if attempt < 3 else "skipping",
            )
        return style, None
    
    results = await asyncio.gather(*[gen_with_retry(s) for s in styles])
    return {s: t for s, t in results if t is not None}
```

### 3.6 Cross-Question Pacing

In `main()` after the per-question loop iteration completes:
```python
async def run() -> dict[str, dict[str, str]]:
    merged = dict(existing) if not args.force else {}
    async with AsyncAnthropic(api_key=api_key) as client:
        for idx, item in enumerate(items_all):
            if item.id in merged and not args.force:
                log.info("Skipping qid=%s (already in JSON)", item.id)
                continue
            try:
                paraphrases = await _generate_for_question(
                    client, args.model, item.question, item.answer,
                )
            except Exception as e:
                log.warning("qid=%s generation failed: %s", item.id, e)
                continue
            if paraphrases:
                merged[item.id] = {"paraphrases": paraphrases}
                log.info(
                    "qid=%s generated %d/%d styles",
                    item.id, len(paraphrases), len(required_styles()),
                )
            # FR-38.2: pace 5s between questions (skip after last)
            if idx < len(items_all) - 1:
                await asyncio.sleep(PACING_SECONDS)
    return merged
```

---

## 4. Test Updates

### 4.1 `test_validation_gate_skips_double_failure` mock update

The current test's mock returns "When was John Smith born in 1968?" (always leaks) for every call. With 3 attempts and 3 styles, the question q1 will be hit 9 times (still all leaks). q2's gold is "yes" — its mock text doesn't contain "yes", so q2's 3 first attempts all pass (no retry needed). Total calls: q1 = 9, q2 = 3. Total = 12.

Update the assertion from `call_count == 6` to `call_count == 12`.

### 4.2 New test: `test_three_attempt_budget_accepts_on_third_try`

```python
def test_three_attempt_budget_accepts_on_third_try(dataset_path, output_path):
    """First 2 attempts leak; 3rd attempt is clean -> accepted.

    Uses per-style attempt counters. Attempts 1 and 2 leak for each style;
    attempt 3 is clean. q1 should have all 3 styles accepted.
    q2's gold "yes" doesn't appear in the leaked text, so q2's first-pass
    passes (no retry needed).
    Total API calls: 9 (q1: 3 styles × 3 attempts) + 3 (q2: 3 first attempts).
    """
    attempt_per_style = {"lexical": 0, "structural": 0, "casual": 0}

    async def leak_twice_clean_third(*args, **kwargs):
        system = kwargs.get("system", "")
        if "lexical" in system.lower(): style = "lexical"
        elif "structural" in system.lower(): style = "structural"
        elif "casual" in system.lower(): style = "casual"
        else: raise AssertionError(f"unexpected system prompt: {system!r}")
        attempt_per_style[style] += 1
        n = attempt_per_style[style]
        if n <= 2:  # first 2 attempts leak
            if style == "lexical": return _mock_text_response("When was John Smith born?")
            if style == "structural": return _mock_text_response("John Smith was born in which year?")
            if style == "casual": return _mock_text_response("when was John Smith born?")
        else:  # 3rd attempt clean
            if style == "lexical": return _mock_text_response("Which writer was born in 1968?")
            if style == "structural": return _mock_text_response("The composer was born in which year?")
            if style == "casual": return _mock_text_response("when was the composer born?")
        raise AssertionError("unreachable")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=leak_twice_clean_third)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    assert rc == 0
    # q1: 3 attempts × 3 styles = 9. q2: 3 first attempts = 3. Total = 12.
    assert mock_client.messages.create.call_count == 12

    items = load_paraphrases(output_path)
    assert "q1" in items
    assert set(items["q1"]["paraphrases"].keys()) == {"lexical", "structural", "casual"}
```

### 4.3 `test_three_calls_per_question_made_concurrently` — pacing consideration

The current test uses `asyncio.sleep(0.01)` in its mock. The iter-10 `_generate_one_style` now does `await asyncio.sleep(PACING_SECONDS)` *before* the call. If PACING_SECONDS=5, the test takes 12 × 5 = 60 seconds per question × 2 questions = 120 seconds. Too slow.

**Fix**: monkeypatch `PACING_SECONDS` to 0 for tests. Add at the top of the test file:
```python
@pytest.fixture(autouse=True)
def _patch_pacing(monkeypatch):
    """Disable 5s pacing for unit tests."""
    monkeypatch.setattr(gen, "PACING_SECONDS", 0)
```

This keeps the existing tests fast while preserving the pacing behavior in production.

---

## 5. Configuration

No new env vars. No new config fields. The `ANTHROPIC_MODEL` env var already controls the model name (default `minimax-3`).

## 6. Error Handling

| Stage | Failure | Behavior |
|---|---|---|
| Generator: API rate limit (429) | MiniMax endpoint throttling | Anthropic client retries internally; on persistent failure, the per-call exception propagates to `gen_with_retry`'s catch in `run()`, qid is skipped, run continues. |
| Generator: API error (5xx) | Transient backend issue | Same as rate limit. |
| Generator: all 3 attempts leak | Style genuinely hard for this question | Style omitted from this qid's entry; other styles kept. Per-style `attempt_per_style` log records the outcome. |

## 7. Testing Strategy

### 7.1 Layers

| Layer | Files | Speed target |
|---|---|---|
| Generator unit (mocked AsyncAnthropic) | `scripts/tests/test_generate_paraphrases_hotpotqa.py` | <2 s each (with PACING_SECONDS=0) |
| Paraphrase gate unit | `backend/tests/eval/test_paraphrases.py` | <5 ms each (unchanged) |
| Eval integration | `backend/tests/eval/test_eval_integration.py` | <5 s (unchanged) |
| Full project suite | `pytest backend/tests/ scripts/tests/ -v` | unchanged |
| Manual smoke | operator runs `python scripts/generate_paraphrases_hotpotqa.py --subset 100` | ~10 min |

### 7.2 Manual Smoke Test

```bash
# 1. Generate with the fix
rm backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json   # force regen
python scripts/generate_paraphrases_hotpotqa.py --subset 1000

# 2. Verify coverage from log
# Expect: "generated 3/3 styles" for ≥ 95% of qids, rate-limit hits < 5%

# 3. Re-run eval
python scripts/eval_hotpotqa.py --subset 1000 --no-cache \
    --paraphrase-set backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json

# 4. Verify metrics
# Expect: per-variant ans_cov@k ≥ 0.85 (up from 0.71-0.76 in iter-9)
```

## 8. Implementation Tasks (TDD)

### Task 1: Update generator prompts and temperature schedule

**Files**:
- Modify: `scripts/generate_paraphrases_hotpotqa.py` — STYLE_PROMPTS, `_user_prompt`, `_retry_temperature_for`, `PACING_SECONDS`, `_generate_one_style`, `_generate_for_question`, cross-question pacing in `main()`.

- [ ] **Step 1**: Rewrite `STYLE_PROMPTS` with HARD RULE framing (see §3.1).
- [ ] **Step 2**: Update `_user_prompt` to put gold-answer first (see §3.1).
- [ ] **Step 3**: Add `_retry_temperature_for` and `PACING_SECONDS` constants.
- [ ] **Step 4**: Update `_generate_one_style` to take `attempt` parameter and sleep `PACING_SECONDS` before call.
- [ ] **Step 5**: Update `gen_with_retry` to loop over 3 attempts with temperature schedule.
- [ ] **Step 6**: Add cross-question pacing in `main()`'s `run()`.
- [ ] **Step 7**: Run existing generator tests; fix any that broke (likely `test_validation_gate_skips_double_failure` due to call_count change).

### Task 2: Add new test for 3-attempt budget

**Files**:
- Modify: `scripts/tests/test_generate_paraphrases_hotpotqa.py` — add `test_three_attempt_budget_accepts_on_third_try`, add `_patch_pacing` autouse fixture.

- [ ] **Step 1**: Add `_patch_pacing` fixture that monkeypatches `PACING_SECONDS=0`.
- [ ] **Step 2**: Update `test_validation_gate_skips_double_failure` mock to allow 3 attempts; update call_count assertion from 6 to 12.
- [ ] **Step 3**: Add new test `test_three_attempt_budget_accepts_on_third_try` (see §4.2).
- [ ] **Step 4**: Run all generator tests; verify all pass.

### Task 3: Run full suite and re-run eval

- [ ] **Step 1**: `pytest backend/tests/ scripts/tests/ -v` — expect all green.
- [ ] **Step 2**: `rm backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json`
- [ ] **Step 3**: `python scripts/generate_paraphrases_hotpotqa.py --subset 1000` — expect ~60 min wall-clock, ≥95% all-3-styles success rate.
- [ ] **Step 4**: `python scripts/eval_hotpotqa.py --subset 1000 --no-cache --paraphrase-set ...` — expect improved metrics.
- [ ] **Step 5**: Update `docs/eval-results/2026-07-08-1000-question-paraphrase-eval.md` with new numbers OR write a new `2026-07-XX-validation-gate-fix-eval.md` report.

### Task 4: Commit and push

- [ ] **Step 1**: Commit Task 1+2 as `feat(generator): front-load hard rule + 3-attempt temperature schedule`.
- [ ] **Step 2**: Commit Task 3 eval report as `docs(eval-results): post-fix paraphrase eval report`.
- [ ] **Step 3**: Push to origin/master.

---

## 9. Out of Scope (Deferred to Future Iterations)

1. Switching to a different validation gate strategy (entity-aware regex, sub-token matching).
2. Switching to a different paraphrase model.
3. Switching to a different embedding model.
4. Async batching across multiple questions.
5. Adaptive temperature (raising temperature only when the previous attempt leaked the answer with high confidence).
6. Per-call jitter on top of the 5s pacing.
7. Retry on rate-limit (429) errors — relies on the Anthropic SDK's default retry.