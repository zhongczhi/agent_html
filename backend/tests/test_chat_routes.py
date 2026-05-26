import pytest
import asyncio
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