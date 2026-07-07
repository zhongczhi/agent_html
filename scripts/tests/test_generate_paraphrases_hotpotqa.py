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