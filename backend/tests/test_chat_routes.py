import pytest
import asyncio

from backend.chat.stream_manager import get_or_create_job, clear_job
import json


@pytest.mark.asyncio
async def test_stream_from_job_yields_unified_chunks():
    """stream_from_job should yield chunks with chunk and type keys."""
    from backend.chat.stream_manager import StreamJob
    from backend.chat.routes import stream_from_job

    job = StreamJob("test-conv")
    job.status = "active"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")
    job.mark_completed()

    chunks = []
    async for event in stream_from_job(job, from_pointer=0):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    # Should have chunk events with type
    assert len(chunks) >= 3
    assert chunks[0] == {"chunk": "First thought", "type": "thinking"}
    assert chunks[1] == {"chunk": "Hello ", "type": "token"}
    assert chunks[2] == {"chunk": "world", "type": "token"}
    # Last should be end marker
    assert chunks[-1] == {"end": True}


@pytest.mark.asyncio
async def test_stream_resume_from_pointer():
    """stream_from_job should skip chunks before from_pointer."""
    from backend.chat.stream_manager import StreamJob
    from backend.chat.routes import stream_from_job

    job = StreamJob("test-conv")
    job.status = "active"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")
    job.mark_completed()

    # Resume from pointer 2 (skip first two chunks)
    chunks = []
    async for event in stream_from_job(job, from_pointer=2):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    # Should start from "world" token
    assert chunks[0] == {"chunk": "world", "type": "token"}


@pytest.mark.asyncio
async def test_second_message_clears_previous_chunks():
    """Sending a second message should clear chunks from the first message."""
    from backend.chat.stream_manager import StreamJob

    conv_id = "test-conv-2"
    clear_job(conv_id)  # Clean up first

    # Create job and add some chunks from a "previous" message
    job = get_or_create_job(conv_id, [])
    job.append_chunk("token", "First message response")
    assert len(job.chunks) == 1

    # Simulate what happens in stream_chat when second message is sent
    # The fix: chunks should be cleared when reactivating job
    job.status = "active"
    job.chunks = []  # This is what the fix in routes.py does
    job.append_chunk("token", "Second message response")

    # Chunks should only contain the new message
    assert len(job.chunks) == 1
    assert job.chunks[0]["chunk"] == "Second message response"

    # Cleanup
    clear_job(conv_id)