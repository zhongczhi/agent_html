import pytest
import asyncio
from unittest.mock import MagicMock
from backend.chat.service import ChatService

def test_generate_yields_thinking_dicts():
    """Test that generate yields thinking dicts before token dicts."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    # Simulate LLM chunk with reasoning block (NOT thinking - MiniMax uses 'reasoning')
    reasoning_chunk = MagicMock()
    reasoning_chunk.content = [
        {"type": "reasoning", "thinking": "User asks about Python..."},
        {"type": "reasoning", "thinking": "Let me think..."},
        {"type": "text", "text": "Python is a programming language."}
    ]

    # Create an async generator
    async def mock_stream():
        yield reasoning_chunk

    mock_chain.astream.return_value = mock_stream()

    async def consume():
        results = []
        async for item in service.generate("Hi", "conv-123"):
            results.append(item)
        return results

    results = asyncio.run(consume())

    # Should yield thinking dicts first
    assert any(isinstance(r, dict) and "thinking" in r for r in results), "Should yield thinking dict"
    # Thinking should come before tokens
    thinking_indices = [i for i, r in enumerate(results) if isinstance(r, dict) and "thinking" in r]
    token_indices = [i for i, r in enumerate(results) if isinstance(r, dict) and "token" in r]
    if thinking_indices and token_indices:
        assert max(thinking_indices) < min(token_indices), "Thinking should come before tokens"