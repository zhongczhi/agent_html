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
    assert job.chunks[0]["chunk"] == "First thought"
    assert job.chunks[0]["type"] == "thinking"
    assert "message_id" in job.chunks[0]
    assert job.chunks[1]["chunk"] == "Hello"
    assert job.chunks[1]["type"] == "token"
    assert "message_id" in job.chunks[1]

    # Queue should have same items
    q0 = job.chunk_queue.get_nowait()
    assert q0["chunk"] == "First thought"
    assert q0["type"] == "thinking"
    assert "message_id" in q0
    q1 = job.chunk_queue.get_nowait()
    assert q1["chunk"] == "Hello"
    assert q1["type"] == "token"
    assert "message_id" in q1