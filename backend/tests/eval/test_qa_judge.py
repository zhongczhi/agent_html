"""Tests for backend.eval.qa_judge. No real LLM calls."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.eval.qa_judge import ask_llm, build_qa_prompt


class FakeDoc:
    """Minimal stand-in for langchain_core.documents.Document in unit tests."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ---- build_qa_prompt ------------------------------------------------------

def test_build_qa_prompt_without_context_returns_user_only():
    msgs = build_qa_prompt("What year?", None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What year?"


def test_build_qa_prompt_with_empty_context_list_returns_user_only():
    """An empty list is treated as 'no context' (defensive)."""
    msgs = build_qa_prompt("What year?", [])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_build_qa_prompt_with_context_returns_system_and_user():
    docs = [
        FakeDoc("Born in 1968 in Berlin.", {"title": "John Smith"}),
        FakeDoc("A composer.", {"title": "John Smith Bio"}),
    ]
    msgs = build_qa_prompt("When was John Smith born?", docs)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_qa_prompt_user_message_includes_context_tags_and_question():
    docs = [
        FakeDoc("Born in 1968 in Berlin.", {"title": "John Smith"}),
    ]
    msgs = build_qa_prompt("When was John Smith born?", docs)
    user_content = msgs[1]["content"]
    assert "<context>" in user_content
    assert "</context>" in user_content
    assert "Born in 1968 in Berlin." in user_content
    assert "When was John Smith born?" in user_content
    assert "[John Smith]:" in user_content  # title prefix


def test_build_qa_prompt_system_prompt_is_rag_specific():
    docs = [FakeDoc("text", {"title": "T"})]
    msgs = build_qa_prompt("Q?", docs)
    # The system prompt should mention <context> grounding behavior.
    assert "<context>" in msgs[0]["content"]
    assert "grounding" in msgs[0]["content"].lower() or "prefer" in msgs[0]["content"].lower()


# ---- ask_llm --------------------------------------------------------------

def _mock_text_response(text: str) -> MagicMock:
    """Build a fake response with one text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return MagicMock(content=[block])


def _mock_thinking_then_text(thinking: str, text: str) -> MagicMock:
    """Build a fake response with a thinking block followed by a text block."""
    t = MagicMock()
    t.type = "thinking"
    t.thinking = thinking
    txt = MagicMock()
    txt.type = "text"
    txt.text = text
    return MagicMock(content=[t, txt])


def _mock_multiple_text_blocks(parts: list[str]) -> MagicMock:
    blocks = []
    for p in parts:
        b = MagicMock()
        b.type = "text"
        b.text = p
        blocks.append(b)
    return MagicMock(content=blocks)


def _mock_dict_response(text: str) -> MagicMock:
    """Build a fake response where blocks are plain dicts (older SDK shape)."""
    return MagicMock(content=[{"type": "text", "text": text}])


@pytest.mark.asyncio
async def test_ask_llm_returns_text():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_text_response("1968"))
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    assert out == "1968"


@pytest.mark.asyncio
async def test_ask_llm_skips_thinking_blocks():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_mock_thinking_then_text("internal reasoning...", "1968")
    )
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    assert out == "1968"
    assert "internal reasoning" not in out


@pytest.mark.asyncio
async def test_ask_llm_joins_multiple_text_blocks():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_mock_multiple_text_blocks(["The answer is", " 1968."])
    )
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    # Joined with newlines, trimmed.
    assert "1968" in out
    assert "The answer is" in out


@pytest.mark.asyncio
async def test_ask_llm_handles_dict_shaped_blocks():
    """Older Anthropic SDK versions return blocks as plain dicts."""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_dict_response("Berlin"))
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    assert out == "Berlin"


@pytest.mark.asyncio
async def test_ask_llm_passes_temperature_zero():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_text_response("ok"))
    await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 200


@pytest.mark.asyncio
async def test_ask_llm_passes_messages_unchanged():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_text_response("ok"))
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    await ask_llm(client, "minimax-3", msgs)
    assert client.messages.create.call_args.kwargs["messages"] == msgs