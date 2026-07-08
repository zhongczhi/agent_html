# Paraphrase Eval Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 3 styled paraphrases per HotpotQA question via 3 concurrent LLM calls per question, persist the result to disk, and extend the existing eval pipeline so retrieval is scored across `(original + 3 paraphrases)` per question with `answer_coverage@k`, `sf_recall@k`, and a `robustness@4` aggregate.

**Architecture:** Two scripts, separable by responsibility. `scripts/generate_paraphrases_hotpotqa.py` does the LLM work — 3 `asyncio.gather`'d calls per question, each with a style-specific system prompt; each paraphrase is token-overlap validated against the gold answer and regenerated once on failure. `scripts/eval_hotpotqa.py` reads the on-disk paraphrase JSON and runs the existing retrieval loop over `(question, variant)` tuples, aggregating per-variant + per-bucket metrics. The per-question FAISS cache already keys on `qid`, so the 4× variant queries per question reuse one cached index — no cache-layout change.

**Tech Stack:** `anthropic` (already in `requirements.txt` via `backend/chat/chain.py`) — `AsyncAnthropic` + `asyncio.gather`. No new Python dependencies. Pure-Python `json` / `re` / `pathlib` for storage and validation.

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `backend/eval/paraphrases.py` | Pure helpers: `validate_paraphrase`, `load_paraphrases`, `lookup`, `required_styles` |
| Create | `backend/tests/eval/test_paraphrases.py` | Unit tests for the helpers |
| Create | `scripts/generate_paraphrases_hotpotqa.py` | CLI: load HotpotQA → 3 concurrent LLM calls per qid → validate → write JSON |
| Create | `scripts/tests/test_generate_paraphrases_hotpotqa.py` | Integration test with mocked `AsyncAnthropic` |
| Modify | `backend/eval/metrics.py` | Add pure `answer_coverage_at_k(retrieved_texts, gold_answer)` |
| Modify | `backend/tests/eval/test_metrics.py` | Add tests for `answer_coverage_at_k` |
| Modify | `scripts/eval_hotpotqa.py` | Load paraphrases, iterate (item, variant), report new metrics |

Total: ~5 files created, ~3 files modified, ~310 lines added. No deletions. No renames.

---

## Storage Layout

**Paraphrase JSON** at `storage/eval/hotpotqa/paraphrases/{dataset_sha[:16]}.json`:

```json
{
  "dataset_sha": "abc123def456...",
  "schema_version": 1,
  "items": {
    "<qid>": {
      "paraphrases": {
        "lexical":    "What year was the composer born?",
        "structural": "The composer was born in which year?",
        "casual":     "when was the composer born?"
      }
    },
    "<qid2>": { "paraphrases": { "lexical": "...", "structural": "..." } }
  }
}
```

A qid may be missing one or more styles if both generation attempts failed validation — the eval handles missing styles by simply not scoring that variant for that qid. The aggregate `robustness@4` only counts qids where all 3 styles + the original succeeded together.

---

## Task 1: Add `answer_coverage_at_k` to `backend/eval/metrics.py`

**Files:**
- Modify: `backend/eval/metrics.py` (append a new pure function after `supporting_fact_metrics`).
- Modify: `backend/tests/eval/test_metrics.py` (append tests).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/eval/test_metrics.py`:

```python
from backend.eval.metrics import answer_coverage_at_k


def test_answer_coverage_empty_gold_vacuous():
    # Vacuous: an empty gold answer is treated as "already covered" so
    # we don't divide by zero and so degenerate items don't tank the mean.
    assert answer_coverage_at_k([], "") == 1.0
    assert answer_coverage_at_k(["anything"], "") == 1.0


def test_answer_coverage_empty_retrieved_zero():
    assert answer_coverage_at_k([], "yes") == 0.0


def test_answer_coverage_exact_match():
    assert answer_coverage_at_k(["The answer is yes."], "yes") == 1.0


def test_answer_coverage_case_insensitive():
    assert answer_coverage_at_k(["...YES..."], "yes") == 1.0


def test_answer_coverage_punctuation_stripped():
    # Gold has a comma; doc doesn't — should still match.
    assert answer_coverage_at_k(["Born in 1968"], "1968,") == 1.0


def test_answer_coverage_no_match():
    assert answer_coverage_at_k(["foo bar baz"], "qux") == 0.0


def test_answer_coverage_across_multiple_docs():
    # Gold answer is a multi-word phrase that spans two retrieved paragraphs.
    assert answer_coverage_at_k(
        ["The composer was", "born in 1968 in Berlin."],
        "1968 in Berlin",
    ) == 1.0


def test_answer_coverage_multiline_inside_one_doc():
    # Paragraphs retrieved from FAISS may contain embedded newlines from
    # the original Wikipedia source.
    assert answer_coverage_at_k(
        ["line one\nline two contains YES in it\nline three"],
        "yes",
    ) == 1.0
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
PYTHONPATH=. pytest backend/tests/eval/test_metrics.py -k answer_coverage -v
```

Expected: `ImportError` or `AttributeError: module 'backend.eval.metrics' has no attribute 'answer_coverage_at_k'` — i.e., all 8 new tests fail. Existing tests in the file still pass.

- [ ] **Step 3: Implement `answer_coverage_at_k`**

Append to `backend/eval/metrics.py`:

```python
import re

# Pre-compile at module load — the character class is constant and the
# regex runs once per query.
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_coverage(text: str) -> str:
    """Lowercase, strip non-word/non-space characters, collapse whitespace.
    'Yes,' -> 'yes'; '  multi  word ' -> 'multi word'."""
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def answer_coverage_at_k(retrieved_texts: list[str], gold_answer: str) -> float:
    """Returns 1.0 if the normalized gold_answer is a substring of any
    normalized retrieved text (paragraphs joined with newlines). Else 0.0.

    Vacuous: empty gold_answer -> 1.0 (matches the convention used by
    paragraph_recall_at_k). Empty retrieved -> 0.0.

    Why substring (not token-set): HotpotQA gold answers are typically
    1-3 words ('yes', '1968', 'The Godfather'). Substring containment after
    normalization is the simplest heuristic that handles all three forms.
    """
    gold = _normalize_for_coverage(gold_answer)
    if not gold:
        return 1.0
    if not retrieved_texts:
        return 0.0
    blob = "\n".join(_normalize_for_coverage(t) for t in retrieved_texts)
    return 1.0 if gold in blob else 0.0
```

- [ ] **Step 4: Run all tests in the file to confirm green**

```bash
PYTHONPATH=. pytest backend/tests/eval/test_metrics.py -v
```

Expected: all 12 tests pass (8 new + 4 existing `paragraph_recall_at_k` + the existing `supporting_fact_metrics` tests).

- [ ] **Step 5: Commit**

```bash
git add backend/eval/metrics.py backend/tests/eval/test_metrics.py
git commit -m "feat(eval): add answer_coverage_at_k metric"
```

---

## Task 2: Create `backend/eval/paraphrases.py` helpers

**Files:**
- Create: `backend/eval/paraphrases.py`.
- Create: `backend/tests/eval/test_paraphrases.py`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/eval/test_paraphrases.py`:

```python
import json
from pathlib import Path

import pytest

from backend.eval.paraphrases import (
    load_paraphrases,
    lookup,
    required_styles,
    validate_paraphrase,
)


# ---- validate_paraphrase ---------------------------------------------------

def test_validate_accepts_unrelated_paraphrase():
    # 'Paris' is not in the paraphrase; no leak.
    assert validate_paraphrase(
        "Which composer wrote the piece?",
        "Paris",
    ) is True


def test_validate_rejects_answer_leak_full():
    # Gold is 'Paris'; paraphrase contains 'Paris' verbatim -> reject.
    assert validate_paraphrase(
        "The composer lived in Paris, when?",
        "Paris",
    ) is False


def test_validate_rejects_answer_leak_partial_high_overlap():
    # Gold has 2 tokens ['john', 'smith']; paraphrase has both -> 100% overlap.
    assert validate_paraphrase(
        "Who is John Smith?",
        "John Smith",
    ) is False


def test_validate_threshold_is_80_percent():
    # Gold has 5 tokens; paraphrase contains 4 of them -> 80% overlap,
    # which is NOT strictly less than 80% -> rejected.
    # Token overlap test: gold=['a','b','c','d','e'],
    # paraphrase contains {'a','b','c','d','x'} -> 4/5 = 0.80 -> reject.
    assert validate_paraphrase(
        "x x x x a b c d",
        "a b c d e",
    ) is False


def test_validate_threshold_just_under_80_passes():
    # 3 of 5 = 60% overlap -> below threshold -> pass.
    assert validate_paraphrase(
        "x y z a b c",
        "a b c d e",
    ) is True


def test_validate_rejects_empty_paraphrase():
    assert validate_paraphrase("", "yes") is False


def test_validate_rejects_empty_gold():
    assert validate_paraphrase("anything", "") is False


# ---- load_paraphrases ------------------------------------------------------

def test_load_returns_items_dict(tmp_path: Path):
    p = tmp_path / "para.json"
    p.write_text(
        json.dumps(
            {
                "dataset_sha": "abc",
                "schema_version": 1,
                "items": {"q1": {"paraphrases": {"lexical": "x"}}},
            }
        ),
        encoding="utf-8",
    )
    out = load_paraphrases(p)
    assert out == {"q1": {"paraphrases": {"lexical": "x"}}}


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_paraphrases(tmp_path / "absent.json")


def test_load_corrupt_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_paraphrases(p)


def test_load_missing_items_key_returns_empty(tmp_path: Path):
    p = tmp_path / "noitems.json"
    p.write_text(json.dumps({"dataset_sha": "x"}), encoding="utf-8")
    assert load_paraphrases(p) == {}


# ---- lookup + required_styles ---------------------------------------------

def test_lookup_returns_paraphrases_when_present():
    data = {"q1": {"paraphrases": {"lexical": "L", "structural": "S", "casual": "C"}}}
    assert lookup(data, "q1") == {"lexical": "L", "structural": "S", "casual": "C"}


def test_lookup_returns_none_when_absent():
    assert lookup({"q1": {}}, "q2") is None


def test_required_styles_is_lexical_structural_casual():
    assert required_styles() == ("lexical", "structural", "casual")
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
PYTHONPATH=. pytest backend/tests/eval/test_paraphrases.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.eval.paraphrases'` — all tests fail at import.

- [ ] **Step 3: Implement the helpers**

Create `backend/eval/paraphrases.py`:

```python
"""Paraphrase load + validate helpers. Pure: no I/O at import time, no LLM.

These are used by both scripts/generate_paraphrases_hotpotqa.py (validation
gate at generation time) and scripts/eval_hotpotqa.py (loading the JSON at
eval time). Keeping them in backend.eval.* lets both scripts import them
without violating the FR-32 isolation rule against backend.chat.* imports.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REQUIRED_STYLES: tuple[str, ...] = ("lexical", "structural", "casual")

# Tokens from the gold answer that appear in the paraphrase at this fraction
# of the gold-answer tokens -> reject. 0.80 means ">=4 of 5 gold tokens in
# the paraphrase -> the LLM is leaking".
_LEAK_OVERLAP_THRESHOLD = 0.80

_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def required_styles() -> tuple[str, ...]:
    """The three styles the generator must produce per question."""
    return _REQUIRED_STYLES


def _normalize(text: str) -> str:
    """Lowercase, strip non-word/non-space characters, collapse whitespace."""
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def validate_paraphrase(paraphrase: str, gold_answer: str) -> bool:
    """Return True iff the paraphrase is acceptable.

    Reject if >= 80% of the gold-answer tokens appear in the paraphrase
    (case-insensitive, punctuation-stripped). Returns False for empty
    paraphrase or empty gold answer — both are degenerate.
    """
    if not paraphrase or not gold_answer:
        return False
    gold_tokens = set(_normalize(gold_answer).split())
    para_tokens = set(_normalize(paraphrase).split())
    if not gold_tokens or not para_tokens:
        return False
    overlap = len(gold_tokens & para_tokens) / len(gold_tokens)
    return overlap < _LEAK_OVERLAP_THRESHOLD


def load_paraphrases(path: Path) -> dict[str, dict[str, str]]:
    """Load the JSON at `path` and return its `items` dict.

    Shape: {qid: {"paraphrases": {style: text, ...}}, ...}

    Raises FileNotFoundError if path missing; json.JSONDecodeError if the
    file is not valid JSON. A JSON object with no `items` key returns {}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, dict):
        return {}
    return items


def lookup(paraphrases: dict[str, dict[str, str]], qid: str) -> dict[str, str] | None:
    """Return the {style: text} dict for `qid`, or None if absent."""
    entry = paraphrases.get(qid)
    if not isinstance(entry, dict):
        return None
    paras = entry.get("paraphrases")
    if not isinstance(paras, dict):
        return None
    return paras
```

- [ ] **Step 4: Run the new tests to confirm green**

```bash
PYTHONPATH=. pytest backend/tests/eval/test_paraphrases.py -v
```

Expected: all 14 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/paraphrases.py backend/tests/eval/test_paraphrases.py
git commit -m "feat(eval): add paraphrase load + validate helpers"
```

---

## Task 3: Create `scripts/generate_paraphrases_hotpotqa.py`

**Files:**
- Create: `scripts/generate_paraphrases_hotpotqa.py`.
- Create: `scripts/tests/test_generate_paraphrases_hotpotqa.py`.

- [ ] **Step 1: Write the failing integration test**

Create `scripts/tests/test_generate_paraphrases_hotpotqa.py`. The test mocks `anthropic.AsyncAnthropic` so no real API calls happen. It must verify:
1. Three concurrent LLM calls per question (`asyncio.gather`).
2. Each call uses a distinct system prompt corresponding to a style.
3. Validation gate rejects a leaking paraphrase and retries once.
4. Successful retry is accepted; double-failure is logged and skipped.
5. Idempotent: re-running with the same JSON in place skips already-paraphrased qids.
6. `--force` regenerates everything.
7. Output JSON has the expected schema (`dataset_sha`, `schema_version`, `items`).

```python
"""Integration tests for scripts/generate_paraphrases_hotpotqa.py.

Mocks AsyncAnthropic so no real API calls happen in CI. Verifies:
  - 3 concurrent LLM calls per question (one per style)
  - Distinct system prompts per style
  - Validation gate retries failures once, skips double-failures
  - Idempotent: skips qids already in the JSON
  - --force regenerates everything
  - Output schema: {dataset_sha, schema_version, items: {qid: {paraphrases: {...}}}}
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# scripts/ is on sys.path via conftest.py
import generate_paraphrases_hotpotqa as gen
from backend.eval.paraphrases import load_paraphrases


# --- fixtures ---------------------------------------------------------------

SAMPLE_DATASET = [
    {
        "_id": "q1",
        "question": "Which composer was born in 1968?",
        "answer": "John Smith",
        "type": "bridge",
        "level": "easy",
        "context": [["A", ["John Smith was born in 1968 in Berlin."]]],
        "supporting_facts": [["A", 0]],
    },
    {
        "_id": "q2",
        "question": "Was the film released in 1999?",
        "answer": "yes",
        "type": "comparison",
        "level": "easy",
        "context": [["B", ["The film premiered in 1999."]]],
        "supporting_facts": [["B", 0]],
    },
]


@pytest.fixture
def dataset_path(tmp_path):
    p = tmp_path / "hotpot.json"
    p.write_text(json.dumps(SAMPLE_DATASET), encoding="utf-8")
    return p


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "paraphrases.json"


def _mock_text_response(text: str) -> MagicMock:
    """Build a fake Anthropic Messages API response with one text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# --- tests ------------------------------------------------------------------

def test_three_calls_per_question_made_concurrently(dataset_path, output_path):
    """asyncio.gather dispatches 3 calls at once per question.

    We track the maximum number of in-flight calls. For 2 questions run
    sequentially, the maximum should be exactly 3 (the 3 styles for one
    question running concurrently via gather). If the implementation
    serializes the 3 styles, max_in_flight stays at 1.
    """
    state = {"active": 0, "max_active": 0}

    async def slow_create(*args, **kwargs):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        try:
            await asyncio.sleep(0.01)
            system = kwargs.get("system", "")
            if "lexical" in system.lower():
                return _mock_text_response("Which writer was born in 1968?")
            if "structural" in system.lower():
                return _mock_text_response("The composer was born in which year?")
            if "casual" in system.lower():
                return _mock_text_response("when was the composer born?")
            raise AssertionError(f"unexpected system prompt: {system!r}")
        finally:
            state["active"] -= 1

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=slow_create)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    assert rc == 0
    assert mock_client.messages.create.call_count == 6  # 2 questions × 3 styles

    # The 3 system prompts are distinct (one per style).
    system_prompts = {
        call.kwargs["system"]
        for call in mock_client.messages.create.call_args_list
    }
    assert len(system_prompts) == 3

    # Concurrency proof: at peak, all 3 styles for one question were in
    # flight simultaneously. If the impl serialized them, max_active would
    # be 1.
    assert state["max_active"] == 3, (
        f"expected max 3 concurrent calls per question, got {state['max_active']}"
    )


def test_validation_gate_retries_leaked_paraphrase(dataset_path, output_path):
    """First attempt leaks the answer; second attempt is clean -> accepted."""

    async def leaking_then_clean(*args, **kwargs):
        system = kwargs.get("system", "")
        call_count = getattr(leaking_then_clean, "_n", 0)
        leaking_then_clean._n = call_count + 1
        # Even-numbered calls leak; odd-numbered are clean.
        if call_count % 2 == 0:
            if "lexical" in system.lower():
                return _mock_text_response("When was John Smith born?")
            if "structural" in system.lower():
                return _mock_text_response("John Smith was born in which year?")
            if "casual" in system.lower():
                return _mock_text_response("when was John Smith born?")
        else:
            if "lexical" in system.lower():
                return _mock_text_response("Which writer was born in 1968?")
            if "structural" in system.lower():
                return _mock_text_response("The composer was born in which year?")
            if "casual" in system.lower():
                return _mock_text_response("when was the composer born?")
        raise AssertionError(f"unexpected system prompt: {system!r}")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=leaking_then_clean)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    assert rc == 0
    # 2 questions × 3 styles × 2 attempts = 12 calls (all leak first, all retry).
    assert mock_client.messages.create.call_count == 12

    items = load_paraphrases(output_path)
    assert "q1" in items
    # q1 has all 3 styles accepted after retry.
    assert set(items["q1"]["paraphrases"].keys()) == {"lexical", "structural", "casual"}


def test_validation_gate_skips_double_failure(dataset_path, output_path):
    """Both attempts leak -> that style is omitted from output, others kept."""

    async def always_leak(*args, **kwargs):
        # Always mention the gold answer.
        return _mock_text_response("When was John Smith born in 1968?")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=always_leak)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    assert rc == 0
    items = load_paraphrases(output_path)
    # q1 (gold='John Smith') -> all 3 styles leak -> no entry.
    assert "q1" not in items
    # q2 (gold='yes') -> 'yes' doesn't appear in any of the leaked outputs
    # because the leaked text says "John Smith" but not "yes" -> all 3 accepted.
    assert "q2" in items
    assert set(items["q2"]["paraphrases"].keys()) == {"lexical", "structural", "casual"}


def test_idempotent_skips_existing(dataset_path, output_path):
    """If output JSON has q1 already, q1 is not re-generated."""
    # Pre-populate the output file with q1 only.
    existing = {
        "dataset_sha": "abc",
        "schema_version": 1,
        "items": {
            "q1": {
                "paraphrases": {
                    "lexical": "L", "structural": "S", "casual": "C"
                }
            }
        },
    }
    output_path.write_text(json.dumps(existing), encoding="utf-8")

    call_log = []

    async def slow_create(*args, **kwargs):
        call_log.append(kwargs.get("system", ""))
        return _mock_text_response("anything clean")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=slow_create)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    assert rc == 0
    # Only q2 generated -> 3 calls (one per style).
    assert mock_client.messages.create.call_count == 3
    # Output preserves the pre-existing q1 entry AND adds q2.
    items = load_paraphrases(output_path)
    assert items["q1"]["paraphrases"] == {"lexical": "L", "structural": "S", "casual": "C"}
    assert "q2" in items


def test_force_regenerates_existing(dataset_path, output_path):
    """--force re-generates even when an entry exists."""
    existing = {
        "dataset_sha": "abc",
        "schema_version": 1,
        "items": {
            "q1": {"paraphrases": {"lexical": "OLD", "structural": "OLD", "casual": "OLD"}},
            "q2": {"paraphrases": {"lexical": "OLD", "structural": "OLD", "casual": "OLD"}},
        },
    }
    output_path.write_text(json.dumps(existing), encoding="utf-8")

    async def clean_create(*args, **kwargs):
        return _mock_text_response("anything clean and safe")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=clean_create)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        rc = gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
            "--force",
        ])

    assert rc == 0
    # Both qids regenerated -> 6 calls.
    assert mock_client.messages.create.call_count == 6


def test_output_schema(dataset_path, output_path):
    """Output JSON has dataset_sha, schema_version=1, items keyed by qid."""
    async def clean_create(*args, **kwargs):
        return _mock_text_response("clean paraphrase here")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=clean_create)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(gen, "AsyncAnthropic", return_value=mock_client):
        gen.main([
            "--fixture", str(dataset_path),
            "--output", str(output_path),
            "--model", "test-model",
        ])

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["dataset_sha"]  # non-empty hex string
    assert set(raw["items"].keys()) == {"q1", "q2"}
    for qid, entry in raw["items"].items():
        assert set(entry["paraphrases"].keys()) <= {"lexical", "structural", "casual"}
```

- [ ] **Step 2: Run the integration test to confirm it fails**

```bash
PYTHONPATH=. pytest scripts/tests/test_generate_paraphrases_hotpotqa.py -v
```

Expected: `ModuleNotFoundError: No module named 'generate_paraphrases_hotpotqa'` (no conftest setup needed because `scripts/tests/conftest.py` already puts `scripts/` on sys.path).

- [ ] **Step 3: Implement `scripts/generate_paraphrases_hotpotqa.py`**

Create `scripts/generate_paraphrases_hotpotqa.py`:

```python
"""Generate 3 styled paraphrases per HotpotQA question via 3 concurrent LLM
calls (one per style). Validates each paraphrase against the gold answer
(token-overlap gate, rejects >=80% overlap). Persists to disk as
{dataset_sha}.json under storage/eval/hotpotqa/paraphrases/. Idempotent:
re-running on an unchanged dataset skips qids already in the JSON.

Isolation: this script may import from backend.eval.* and from `anthropic`.
It does NOT import from backend.chat.* — that boundary is preserved for
scripts/eval_hotpotqa.py (the actual eval script), which must stay
LLM-call-free per FR-32.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# Bootstrap sys.path so `python scripts/generate_paraphrases_hotpotqa.py`
# finds the `backend` package from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from backend.eval.hotpotqa import dataset_sha, load as load_dataset, sample  # noqa: E402
from backend.eval.paraphrases import (  # noqa: E402
    load_paraphrases,
    required_styles,
    validate_paraphrase,
)

log = logging.getLogger("generate_paraphrases_hotpotqa")

REPO_ROOT = _REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"
PARAPHRASES_DIR = REPO_ROOT / "backend" / "storage" / "eval" / "hotpotqa" / "paraphrases"

# Style-specific system prompts. Each steers the LLM toward a distinct
# surface variation of the original question.
STYLE_PROMPTS: dict[str, str] = {
    "lexical": (
        "You paraphrase questions. Output ONLY the paraphrase, no preamble. "
        "Keep the exact sentence structure of the original but substitute "
        "synonyms and minor word choices (e.g. 'In which year' -> 'What year'). "
        "Do NOT include the answer in your paraphrase. Output one sentence."
    ),
    "structural": (
        "You paraphrase questions. Output ONLY the paraphrase, no preamble. "
        "Keep all the original entities and facts but reorder the clauses "
        "(e.g. active -> passive, 'X was born in Y' -> 'In which year was X "
        "born, given that Y is associated with X?'). Do NOT include the answer "
        "in your paraphrase. Output one sentence."
    ),
    "casual": (
        "You paraphrase questions. Output ONLY the paraphrase, no preamble. "
        "Make the question informal and conversational, as if a real user "
        "typed it quickly in a chat: use contractions, drop articles where "
        "natural, allow lowercase. Do NOT include the answer in your "
        "paraphrase. Output one sentence."
    ),
}


def _default_paraphrase_path(dataset_path: Path) -> Path:
    return PARAPHRASES_DIR / f"{dataset_sha(dataset_path)}.json"


def _user_prompt(question: str, gold_answer: str) -> str:
    # We DO include the gold answer in the prompt so the model knows what to
    # avoid — but the validation gate then rejects anything that leaks.
    # Without this, the model has no signal that "Paris" is the answer to
    # avoid using in "When was X born?" paraphrases.
    return (
        f"Original question: {question}\n"
        f"Do NOT include this answer in your paraphrase: {gold_answer}"
    )


async def _generate_one_style(
    client: AsyncAnthropic,
    model: str,
    style: str,
    question: str,
    gold_answer: str,
) -> str:
    """One Anthropic call returning the paraphrase text for one style."""
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        system=STYLE_PROMPTS[style],
        messages=[
            {"role": "user", "content": _user_prompt(question, gold_answer)},
        ],
    )
    # Response has one text block (we asked for one sentence).
    return response.content[0].text.strip()


async def _generate_for_question(
    client: AsyncAnthropic,
    model: str,
    question: str,
    gold_answer: str,
) -> dict[str, str]:
    """Generate all 3 styles in parallel; validate; retry failures once.

    Returns {style: text} for styles that passed validation (possibly fewer
    than 3 if some failed twice).
    """
    styles = required_styles()

    async def gen(style: str) -> tuple[str, str]:
        text = await _generate_one_style(client, model, style, question, gold_answer)
        return style, text

    first_pass = dict(await asyncio.gather(*[gen(s) for s in styles]))

    async def maybe_retry(style: str, text: str) -> tuple[str, str | None]:
        if validate_paraphrase(text, gold_answer):
            return style, text
        log.warning("qid=? style=%s leaked answer; retrying", style)
        retry_text = await _generate_one_style(
            client, model, style, question, gold_answer
        )
        if validate_paraphrase(retry_text, gold_answer):
            log.info("qid=? style=%s retry succeeded", style)
            return style, retry_text
        log.warning("qid=? style=%s failed validation twice; skipping", style)
        return style, None

    retried = await asyncio.gather(
        *[maybe_retry(s, t) for s, t in first_pass.items()]
    )
    return {s: t for s, t in retried if t is not None}


def _write_output(
    output_path: Path,
    dataset_sha_hex: str,
    items: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_sha": dataset_sha_hex,
        "schema_version": 1,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate HotpotQA paraphrase set via 3 concurrent LLM calls per question."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--subset",
        type=int,
        metavar="N",
        help="Stratified sample of N questions (mutually exclusive with --full)",
    )
    grp.add_argument(
        "--full",
        action="store_true",
        help="Use all questions (default)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Read dataset from PATH (test hook)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", "minimax-3"),
        help="Anthropic model name (default: minimax-3, override with --model or $ANTHROPIC_MODEL)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate paraphrases even for qids already in the output file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: storage/eval/hotpotqa/paraphrases/{dataset_sha}.json)",
    )
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    dataset_path = args.fixture or DEFAULT_DATASET
    if not dataset_path.exists():
        print(
            f"Dataset not found at {dataset_path}.\n"
            "Run scripts/ingest_hotpotqa.py first, or pass --fixture PATH.",
            file=sys.stderr,
        )
        return 1
    try:
        items_all = load_dataset(dataset_path)
    except json.JSONDecodeError as e:
        print(f"Dataset JSON is corrupt: {e}", file=sys.stderr)
        return 1
    if args.subset is not None:
        items_all = sample(items_all, args.subset)

    d_sha = dataset_sha(dataset_path)
    output_path = args.output or _default_paraphrase_path(dataset_path)

    existing: dict[str, dict[str, str]] = {}
    if output_path.exists() and not args.force:
        try:
            existing = load_paraphrases(output_path)
        except json.JSONDecodeError:
            log.warning("Existing paraphrase file at %s is corrupt; regenerating", output_path)
            existing = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY env var is not set. Set it before running.",
            file=sys.stderr,
        )
        return 1

    async def run() -> dict[str, dict[str, str]]:
        # Preserve existing entries (unless --force) and add new ones.
        merged = dict(existing) if not args.force else {}
        async with AsyncAnthropic(api_key=api_key) as client:
            for item in items_all:
                if item.id in merged and not args.force:
                    log.info("Skipping qid=%s (already in JSON)", item.id)
                    continue
                try:
                    paraphrases = await _generate_for_question(
                        client, args.model, item.question, item.answer
                    )
                except Exception as e:
                    log.warning("qid=%s generation failed: %s", item.id, e)
                    continue
                if paraphrases:
                    merged[item.id] = {"paraphrases": paraphrases}
                    log.info(
                        "qid=%s generated %d/%d styles",
                        item.id,
                        len(paraphrases),
                        len(required_styles()),
                    )
        return merged

    all_items = asyncio.run(run())
    _write_output(output_path, d_sha, all_items)
    log.info(
        "Wrote %d paraphrase entries to %s",
        len(all_items),
        output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the integration tests to confirm green**

```bash
PYTHONPATH=. pytest scripts/tests/test_generate_paraphrases_hotpotqa.py -v
```

Expected: all 6 integration tests pass. (The mocks stand in for the Anthropic API; no real calls are made.)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_paraphrases_hotpotqa.py scripts/tests/test_generate_paraphrases_hotpotqa.py
git commit -m "feat(scripts): generate HotpotQA paraphrases via 3 concurrent LLM calls"
```

---

## Task 4: Modify `scripts/eval_hotpotqa.py` for multi-variant eval

**Files:**
- Modify: `scripts/eval_hotpotqa.py`.

- [ ] **Step 1: Read the existing script and locate insertion points**

Open `scripts/eval_hotpotqa.py`. The current per-question loop lives at lines 90–112. The current argument parser is at lines 41–60. The current terminal-output block is at lines 119–132.

- [ ] **Step 2: Add `--paraphrase-set` argument + paraphrase loading**

Replace the argparse block (lines 41–60) with:

```python
    parser.add_argument("--k", type=int, default=4, help="Top-k to retrieve (default 4)")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force rebuild of every per-question index",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Read dataset from PATH (test hook)",
    )
    parser.add_argument(
        "--paraphrase-set",
        type=Path,
        default=None,
        help=(
            "Path to a paraphrase JSON file (default: "
            "storage/eval/hotpotqa/paraphrases/{dataset_sha}.json). "
            "If absent, eval runs in original-only mode with a WARNING."
        ),
    )
```

After `args = parser.parse_args(argv)` (around line 60), insert:

```python
    from backend.eval.paraphrases import load_paraphrases

    paraphrase_path = args.paraphrase_set or (
        REPO_ROOT
        / "backend" / "storage" / "eval" / "hotpotqa" / "paraphrases"
        / f"{d_sha}.json"
    )
    paraphrases: dict[str, dict[str, str]] = {}
    if paraphrase_path.exists():
        try:
            paraphrases = load_paraphrases(paraphrase_path)
            log.info(
                "Loaded %d paraphrase entries from %s",
                len(paraphrases),
                paraphrase_path,
            )
        except json.JSONDecodeError as e:
            log.warning(
                "Paraphrase set at %s is corrupt (%s); running original-only",
                paraphrase_path,
                e,
            )
    else:
        log.warning(
            "Paraphrase set not found at %s; running original-only mode",
            paraphrase_path,
        )
```

(Remove this insertion; use the same indentation style as the rest of the script — adjust as needed.)

- [ ] **Step 3: Replace the per-question loop with multi-variant loop**

Replace the loop body (lines 92–112) with:

```python
    per_q: list[dict] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()
    for item in items:
        try:
            index, hit = ev_cache.load_or_build(
                item, d_sha, embeddings, no_cache=args.no_cache
            )
            if hit:
                cache_hits += 1
            else:
                cache_builds += 1

            # Build the list of (question_text, variant_name) pairs.
            variants: list[tuple[str, str]] = [(item.question, "original")]
            para_entry = paraphrases.get(item.id)
            if para_entry:
                para_styles = para_entry.get("paraphrases", {})
                for style in ("lexical", "structural", "casual"):
                    if style in para_styles:
                        variants.append((para_styles[style], style))

            for question_text, variant_name in variants:
                docs = index.similarity_search(question_text, k=args.k)
                retrieved_titles = [d.metadata.get("title", "") for d in docs]
                retrieved_texts = [d.page_content for d in docs]
                gold_titles = hotpot.gold_paragraph_titles(item)
                pr = metrics.paragraph_recall_at_k(retrieved_titles, gold_titles)
                sp, sr, sf_f1, em = metrics.supporting_fact_metrics(
                    retrieved_titles, gold_titles
                )
                ac = metrics.answer_coverage_at_k(retrieved_texts, item.answer)
                per_q.append({
                    "qid": item.id,
                    "variant": variant_name,
                    "type": item.type,
                    "level": item.level,
                    "paragraph_recall": pr,
                    "sf_precision": sp,
                    "sf_recall": sr,
                    "sf_f1": sf_f1,
                    "sf_em": em,
                    "answer_coverage": ac,
                })
        except Exception as e:
            log.warning("qid=%s error: %s", item.id, e)
            errors += 1
```

- [ ] **Step 4: Replace the terminal-output block with the new multi-metric output**

Replace the output block (lines 114–132) with the block below.

**Backward compatibility note**: the existing integration test
`backend/tests/eval/test_eval_integration.py` greps for the literal
labels `paragraph_recall@4`, `sf_precision`, `sf_recall`, `sf_f1`,
`sf_em`, and the pattern `cache hits / builds\s+:\s+(\d+)\s+/\s+(\d+)`.
The new block keeps those exact labels and patterns as the headline
metrics, then adds the new per-variant / per-bucket / robustness blocks
below.

```python
    elapsed = time.monotonic() - t0

    def avg(predicate, key):
        relevant = [r[key] for r in per_q if predicate(r)]
        return (sum(relevant) / len(relevant)) if relevant else 0.0

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    label = "full" if args.subset is None else str(args.subset)
    print(f"\nHotpotQA Eval — subset={label}, k={args.k}, dataset_sha={d_sha}")
    if paraphrases:
        n_paraphrase_evals = sum(
            1 for r in per_q if r["variant"] != "original"
        )
        print(
            f"  paraphrase_set      : {paraphrase_path} "
            f"({len(paraphrases)} entries; {n_paraphrase_evals} paraphrase evaluations)"
        )
    else:
        print(f"  paraphrase_set      : (none — original-only mode)")

    # Headline (preserved labels for the existing integration test).
    print(f"  paragraph_recall@{args.k}  : {fmt(avg(lambda _: True, 'paragraph_recall'))}")
    print(f"  sf_precision        : {fmt(avg(lambda _: True, 'sf_precision'))}")
    print(f"  sf_recall           : {fmt(avg(lambda _: True, 'sf_recall'))}")
    print(f"  sf_f1               : {fmt(avg(lambda _: True, 'sf_f1'))}")
    print(f"  sf_em               : {fmt(avg(lambda _: True, 'sf_em'))}")

    # By variant (new)
    print("  -- by variant --")
    for variant in ("original", "lexical", "structural", "casual"):
        n = sum(1 for r in per_q if r["variant"] == variant)
        if n == 0:
            print(f"  {variant:<12} : (no data)")
            continue
        print(
            f"  {variant:<12} : "
            f"n={n:<4}  "
            f"ans_cov={fmt(avg(lambda r: r['variant'] == variant, 'answer_coverage'))}  "
            f"sf_recall={fmt(avg(lambda r: r['variant'] == variant, 'sf_recall'))}  "
            f"para_recall={fmt(avg(lambda r: r['variant'] == variant, 'paragraph_recall'))}"
        )

    # Aggregate (new)
    if per_q:
        print("  -- aggregate --")
        print(f"  mean_ans_cov@k    : {fmt(avg(lambda _: True, 'answer_coverage'))}")

        # Robustness@4: fraction of qids where all 4 variants had ans_cov=1.
        from collections import defaultdict
        by_qid: dict[str, dict[str, float]] = defaultdict(dict)
        for r in per_q:
            by_qid[r["qid"]][r["variant"]] = r["answer_coverage"]
        robust_count = sum(
            1
            for qid, vs in by_qid.items()
            if vs.get("original") == 1.0
            and vs.get("lexical") == 1.0
            and vs.get("structural") == 1.0
            and vs.get("casual") == 1.0
        )
        robust_total = sum(
            1
            for qid, vs in by_qid.items()
            if all(s in vs for s in ("original", "lexical", "structural", "casual"))
        )
        if robust_total:
            print(
                f"  robustness@4      : {robust_count / robust_total:.3f} "
                f"({robust_count}/{robust_total} qids with all 4 variants ans_cov=1)"
            )
        else:
            print("  robustness@4      : (no qid had all 4 variants)")

        # Per (type, level)
        print("  -- by (type, level) --")
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in per_q:
            buckets[(r["type"], r["level"])].append(r)
        for key in sorted(buckets):
            rows = buckets[key]
            n = len(rows)
            ac = sum(r["answer_coverage"] for r in rows) / n
            print(f"  {key[0]}/{key[1]:<8} : ans_cov={fmt(ac)}  (n={n})")

    # Footer (preserved labels for the existing integration test).
    print(
        f"  questions successfully evaluated : {len(per_q)} "
        f"(out of {len(items)} attempted)"
    )
    print(f"  cache hits / builds  : {cache_hits} / {cache_builds}")
    print(f"  errors               : {errors}")
    print(f"  elapsed              : {elapsed:.1f}s")

    if errors:
        log.warning(
            "%d questions errored (skipped, not counted in metrics above)", errors
        )
    return 0
```

- [ ] **Step 5: Run the full backend test suite as a regression check**

```bash
PYTHONPATH=. pytest backend/tests/ scripts/tests/ -q
```

Expected: existing 175 + new (12 metrics + 14 paraphrases + 6 generator) = 207 tests pass. No regressions.

- [ ] **Step 6: Smoke-test the modified eval script in original-only mode**

```bash
python scripts/eval_hotpotqa.py --subset 5 --no-cache --fixture /path/to/synthetic.json
```

(or any small `--fixture` JSON you already use in `backend/tests/eval/test_eval_integration.py`.)

Expected: prints the new output format with only the `original` row populated and `(none — original-only mode)` for paraphrase_set. Confirms the script didn't break the original-only path.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_hotpotqa.py
git commit -m "feat(eval): score retrieval across original + 3 paraphrases per question"
```

---

## Task 5: End-to-end manual verification on `--subset 100`

**Files:**
- Read: this plan's Storage Layout section.

- [ ] **Step 1: Generate paraphrases for a stratified 100-question sample**

```bash
python scripts/generate_paraphrases_hotpotqa.py --subset 100
```

Expected: ~300 LLM calls (100 questions × 3 styles, plus up to 100 retries if some leak). Wall-clock ~1–2 minutes depending on API latency. Output JSON written to `backend/storage/eval/hotpotqa/paraphrases/{dataset_sha}.json`.

If `ANTHROPIC_API_KEY` is not set, the script exits 1 with a clear error.

- [ ] **Step 2: Spot-check the generated paraphrases**

```bash
python -c "
import json
data = json.loads(open('backend/storage/eval/hotpotqa/paraphrases/{dataset_sha}.json').read())
items = data['items']
print('Total entries:', len(items))
qid = next(iter(items))
print('Example qid:', qid)
for style, text in items[qid]['paraphrases'].items():
    print(f'  [{style}]', text)
"
```

Expected: 100 entries (one per stratified-sampled question). At least one example has all 3 styles populated; each paraphrase is a single sentence and does NOT contain the gold answer string.

- [ ] **Step 3: Run eval on the same 100 questions**

```bash
python scripts/eval_hotpotqa.py --subset 100
```

Expected output shape:

```
HotpotQA Eval — subset=100, k=4, dataset_sha=...
  paraphrase_set      : .../paraphrases/{sha}.json (100 entries; 300 variant evaluations)
  -- by variant --
  original    : n=100 ans_cov=...  sf_recall=...  para_recall=...
  lexical     : n=98  ans_cov=...  sf_recall=...  para_recall=...
  structural  : n=97  ans_cov=...  sf_recall=...  para_recall=...
  casual      : n=95  ans_cov=...  sf_recall=...  para_recall=...
  -- aggregate --
  mean_ans_cov@k    : ...
  mean_sf_recall@k  : ...
  mean_para_recall  : ...
  robustness@4      : ... (NN/100 qids with all 4 variants ans_cov=1)
  -- by (type, level) --
  bridge/easy       : ans_cov=... (n=...)
  bridge/hard       : ans_cov=... (n=...)
  ...
  variant-evaluations   : ~390
  cache hits / builds  : ...
  errors               : 0 (or small)
  elapsed              : ...
```

Notes for verification:
- `n=` for each variant may be < 100 if some qids are missing styles (validation failures).
- `robustness@4` should be ≤ the lowest per-variant `ans_cov` (since it requires all 4 to be 1).
- Per-bucket breakdown should have entries for all 6 `(type, level)` pairs (stratified sampling guarantees this).
- If `errors > 0`, investigate the warnings before trusting the aggregate numbers.

- [ ] **Step 4: Run full test suite one more time**

```bash
PYTHONPATH=. pytest backend/tests/ scripts/tests/ -q
```

Expected: 207 passed, 0 failed.

- [ ] **Step 5: Final cleanup commit if any test fixtures were modified**

If `storage/conversations.json` or any test fixture was edited during verification, revert:

```bash
git checkout -- storage/
```

If anything new was added (e.g., a small fixture file in `backend/tests/eval/fixtures/`), commit it explicitly with a descriptive message.

If `git status --short` is clean, no commit needed for Task 5.

---

## Done

When all 5 tasks pass and the manual end-to-end smoke test produces sensible per-variant + per-bucket numbers:

- Paraphrase generation is a separate, idempotent CLI (`scripts/generate_paraphrases_hotpotqa.py`) with a validation gate.
- The eval pipeline runs across `(original + 3 paraphrases)` per question with `answer_coverage@k`, `sf_recall@k`, `paragraph_recall@k`, and a `robustness@4` aggregate.
- Per-question FAISS cache is unchanged (the 4× variant queries reuse one cached index).
- `scripts/eval_hotpotqa.py` still imports nothing from `backend.chat.*` — FR-32 isolation preserved.
- No new Python dependencies.
- All 207 tests green.

**Scaling to full 7405-question eval**: re-run step 1 without `--subset`. Wall-clock ~30 minutes for paraphrase generation (depending on API latency and retry rate) + a few minutes for eval (FAISS indices cached on second run).

---

## Risks recap (already raised in the brainstorming round)

- **LLM cost**: full-set generation = ~22k Anthropic calls + ~25k retries worst case. Configurable via `--model` (Haiku is cheaper; the default `minimax-3` is what chat uses).
- **Validation threshold (80%) is a heuristic, not a guarantee**. False negatives (clean paraphrase rejected) just shrink the per-variant `n=`. False positives (leaking paraphrase accepted) pollute the eval — for 100 questions this is rare enough to spot-check manually; for 7405 it warrants a follow-up audit.
- **Idempotency**: deleting a single qid's entry from the JSON re-generates just that qid; deleting the file re-generates everything. Use `--force` only when explicitly regenerating the whole set.
- **Per-question FAISS cache layout is unchanged**: 4× variant queries hit one cached index each, so wall-clock per question is roughly `4 × similarity_search_time` (sub-second).