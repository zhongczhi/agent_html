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

    # Check job has the thinking tokens via chunks
    from backend.chat.stream_manager import get_job
    job = get_job("test-conv-123")
    assert job is not None
    thinking_chunks = [c for c in job.chunks if c["type"] == "thinking"]
    assert len(thinking_chunks) == 2
    full_thinking = "".join(c["chunk"] for c in thinking_chunks)
    assert full_thinking == "User asks about Python...Let me think..."

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
    token_chunks = [c for c in job.chunks if c["type"] == "token"]
    assert len(token_chunks) == 3
    full_content = "".join(c["chunk"] for c in token_chunks)
    assert full_content == "Python is great."

    clear_job("test-conv-456")


def test_get_stream_status_returns_chunks_count():
    """Test get_stream_status returns chunks_count and status."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    clear_job("test-status")

    # Create a job with some chunks using the new API
    from backend.chat.stream_manager import get_or_create_job
    job = get_or_create_job("test-status", [])
    job.append_chunk("thinking", "thinking 1")
    job.append_chunk("thinking", "thinking 2")
    job.append_chunk("token", "token 1")

    status = service.get_stream_status("test-status")
    assert status["chunks_count"] == 3
    assert status["status"] == "pending"
    assert status["streaming"] == False
    assert status["is_complete"] == False

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
    token_chunks = [c for c in job.chunks if c["type"] == "token"]
    full_content = "".join(c["chunk"] for c in token_chunks)
    assert full_content == "Hello world"

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


@pytest.mark.asyncio
async def test_generate_background_aborts_on_cancellation():
    """If clear_job is called mid-stream, generate_background must NOT call
    file_storage.save_conversation (which would resurrect the deleted
    conversation)."""
    from backend.chat.service import ChatService
    from backend.chat.stream_manager import get_or_create_job, clear_job, get_job
    from backend.storage import file_storage

    conv_id = "test-cancel-mid-stream"
    clear_job(conv_id)

    # Pre-create the conversation so save_conversation would update-in-place
    # (this mirrors the "new conversation" path in stream_chat).
    file_storage.create_conversation(conv_id)
    file_storage.append_message(conv_id, "user", "Hi")

    mock_chain = MagicMock()

    class AsyncIterator:
        """Yields one chunk, then sleeps until cancelled, then would yield more."""
        def __init__(self):
            self.index = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.index == 0:
                self.index += 1
                chunk = MagicMock()
                chunk.content = [{"type": "text", "text": "Hello "}]
                return chunk
            # Sleep until cancelled
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
            # Would never reach here in normal flow; provide another chunk just in case.
            self.index += 1
            chunk = MagicMock()
            chunk.content = [{"type": "text", "text": "world"}]
            return chunk

    mock_chain.astream.return_value = AsyncIterator()

    service = ChatService(mock_chain)

    job = get_or_create_job(conv_id, [])

    task = asyncio.create_task(service.generate_background("Hi", conv_id))
    # Let the first chunk be processed
    await asyncio.sleep(0.1)
    # Simulate user deleting the conversation mid-stream
    clear_job(conv_id)
    # Cancel the LLM task too (the abort that's normally done by the frontend)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    # Wait long enough that any "I'm going to call save_conversation now" path
    # would have run.
    await asyncio.sleep(0.5)

    # Job was cleared by clear_job; verify it is no longer in the registry.
    assert get_job(conv_id) is None

    # CRITICAL: the conversation must NOT have been resurrected by save_conversation.
    # It should still be present in storage (we never asked delete_conversation
    # to remove it), but its messages list must NOT contain the assistant message
    # produced by the background task.
    conv = file_storage.get_conversation(conv_id)
    assert conv is not None
    roles = [m["role"] for m in conv["messages"]]
    assert "assistant" not in roles, (
        f"Resurrection bug: background task saved assistant message after cancellation. "
        f"Roles: {roles}"
    )

    # Cleanup
    file_storage.delete_conversation(conv_id)