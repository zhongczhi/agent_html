import pytest
import asyncio
from unittest.mock import MagicMock, patch
from backend.chat.service import ChatService
from backend.chat.stream_manager import STREAM_REGISTRY, clear_job

def test_generate_background_stores_thinking_in_job():
    """Test that generate_background stores thinking tokens in job."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    # Clear any existing job
    clear_job("test-conv-123")

    # Simulate LLM chunk with thinking block
    reasoning_chunk = MagicMock()
    reasoning_chunk.content = [
        {"type": "thinking", "thinking": "User asks about Python..."},
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "Python is a programming language."}
    ]

    # Create an async generator that yields the chunk
    async def mock_stream():
        yield reasoning_chunk

    mock_chain.astream.return_value = mock_stream()

    # Run the background task
    async def run():
        await service.generate_background("Hi", "test-conv-123")

    asyncio.run(run())

    # Check job has the thinking tokens
    from backend.chat.stream_manager import get_job
    job = get_job("test-conv-123")
    assert job is not None
    assert len(job.thinking_tokens) == 2
    assert job.get_full_thinking() == "User asks about Python...Let me think..."

    # Clean up
    clear_job("test-conv-123")


def test_generate_background_stores_tokens_in_job():
    """Test that generate_background stores text tokens in job."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    clear_job("test-conv-456")

    # Simulate LLM chunk with text only
    text_chunk = MagicMock()
    text_chunk.content = [
        {"type": "text", "text": "Python "},
        {"type": "text", "text": "is "},
        {"type": "text", "text": "great."}
    ]

    async def mock_stream():
        yield text_chunk

    mock_chain.astream.return_value = mock_stream()

    async def run():
        await service.generate_background("Tell me about Python", "test-conv-456")

    asyncio.run(run())

    from backend.chat.stream_manager import get_job
    job = get_job("test-conv-456")
    assert job is not None
    assert len(job.tokens) == 3
    assert job.get_full_content() == "Python is great."

    clear_job("test-conv-456")


def test_get_stream_status_returns_thinking_count():
    """Test get_stream_status includes thinking_count."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    clear_job("test-status")

    # Create a job with some tokens
    from backend.chat.stream_manager import get_or_create_job
    job = get_or_create_job("test-status", [])
    job.append_thinking("thinking 1")
    job.append_thinking("thinking 2")
    job.append_token("token 1")

    status = service.get_stream_status("test-status")
    assert status["thinking_count"] == 2
    assert status["tokens_count"] == 1
    assert status["thinking_pointer"] == 0
    assert status["pointer"] == 0

    clear_job("test-status")


def test_generate_background_handles_string_content():
    """Test that generate_background handles string content from chunk."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    clear_job("test-string-content")

    # Simulate LLM chunk with string content
    string_chunk = MagicMock()
    string_chunk.content = "Hello world"

    async def mock_stream():
        yield string_chunk

    mock_chain.astream.return_value = mock_stream()

    async def run():
        await service.generate_background("Hi", "test-string-content")

    asyncio.run(run())

    from backend.chat.stream_manager import get_job
    job = get_job("test-string-content")
    assert job is not None
    assert job.get_full_content() == "Hello world"

    clear_job("test-string-content")


@pytest.mark.asyncio
async def test_generate_background_uses_append_chunk():
    """ChatService should call job.append_chunk with type, not separate methods."""
    from backend.chat.service import ChatService
    from backend.chat.stream_manager import StreamJob, get_or_create_job, clear_job

    # Clear registry entry
    clear_job("test-append-chunk")

    mock_chain = MagicMock()

    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    # Simulate LLM yielding a thinking block then a text block
    thinking_block = MagicMock()
    thinking_block.content = [{"type": "thinking", "thinking": "Let me think..."}]
    text_block = MagicMock()
    text_block.content = [{"type": "text", "text": "Hello!"}]

    mock_chain.astream.return_value = AsyncIterator([thinking_block, text_block])

    service = ChatService(mock_chain)

    job = get_or_create_job("test-append-chunk", [])

    await service.generate_background("Hi", "test-append-chunk")

    # Verify chunks were added with correct types
    assert len(job.chunks) >= 2
    # Find thinking and token chunks
    thinking_chunks = [c for c in job.chunks if c["type"] == "thinking"]
    token_chunks = [c for c in job.chunks if c["type"] == "token"]
    assert len(thinking_chunks) >= 1, f"Expected at least 1 thinking chunk, got {len(thinking_chunks)}"
    assert len(token_chunks) >= 1, f"Expected at least 1 token chunk, got {len(token_chunks)}"

    clear_job("test-append-chunk")