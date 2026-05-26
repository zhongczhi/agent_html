import pytest
import asyncio


def test_stream_job_has_unified_chunks():
    """StreamJob should have chunks list and chunk_queue, not thinking/token separation."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")

    # Should have chunks attribute
    assert hasattr(job, 'chunks')
    assert job.chunks == []

    # Should have chunk_queue
    assert hasattr(job, 'chunk_queue')
    assert isinstance(job.chunk_queue, asyncio.Queue)

    # Should NOT have old thinking/token attributes
    assert not hasattr(job, 'thinking_queue')
    assert not hasattr(job, 'token_queue')
    assert not hasattr(job, 'thinking_tokens')
    assert not hasattr(job, 'sent_pointer')


def test_append_chunk_adds_to_chunks_and_queue():
    """append_chunk should add dict with chunk and type to both list and queue."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")

    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello")

    assert len(job.chunks) == 2
    assert job.chunks[0] == {"chunk": "First thought", "type": "thinking"}
    assert job.chunks[1] == {"chunk": "Hello", "type": "token"}

    # Queue should have same items
    assert job.chunk_queue.get_nowait() == {"chunk": "First thought", "type": "thinking"}
    assert job.chunk_queue.get_nowait() == {"chunk": "Hello", "type": "token"}