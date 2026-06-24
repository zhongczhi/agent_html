import pytest
import asyncio
import json

from backend.chat.stream_manager import clear_job, get_or_create_job
from backend.chat.routes import stream_from_active_job, stream_from_inactive_job


@pytest.mark.asyncio
async def test_stream_from_inactive_yields_unified_chunks():
    """stream_from_inactive_job should yield chunks with chunk, type, and message_id keys."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")
    job.status = "completed"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")

    chunks = []
    async for event in stream_from_inactive_job(job, from_pointer=0):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    assert len(chunks) == 4
    assert chunks[0]["chunk"] == "First thought"
    assert chunks[0]["type"] == "thinking"
    assert "message_id" in chunks[0]
    assert chunks[1]["chunk"] == "Hello "
    assert chunks[1]["type"] == "token"
    assert "message_id" in chunks[1]
    assert chunks[2]["chunk"] == "world"
    assert chunks[2]["type"] == "token"
    assert "message_id" in chunks[2]
    assert chunks[-1] == {"end": True}


@pytest.mark.asyncio
async def test_stream_resume_from_pointer():
    """stream_from_inactive_job should skip chunks before from_pointer."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")
    job.status = "completed"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")

    chunks = []
    async for event in stream_from_inactive_job(job, from_pointer=2):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    # Should start from "world" token
    assert chunks[0]["chunk"] == "world"
    assert chunks[0]["type"] == "token"
    assert "message_id" in chunks[0]
    assert chunks[-1] == {"end": True}


@pytest.mark.asyncio
async def test_stream_resume_at_boundary_yields_end_marker():
    """stream_from_active_job with from_pointer == len(chunks) must still wait for
    queued chunks / end marker instead of returning immediately.

    Regression test: the previous early-return condition treated
    from_pointer == len(chunks) > 0 as an error case, which broke resume
    during active streaming (the frontend's pointer always lands at
    len(job.chunks) because the user entry is in chunksCache only).
    """
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv-boundary")
    job.status = "active"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")
    from_pointer = len(job.chunks)
    assert from_pointer == 3

    job.mark_completed()

    events = []
    async for event in stream_from_active_job(job, from_pointer=from_pointer):
        if event.startswith("data: "):
            events.append(json.loads(event[6:]))

    assert events == [{"end": True}]


@pytest.mark.asyncio
async def test_stream_resume_at_boundary_streams_new_chunks():
    """At the boundary, new chunks arriving after the resume must still be
    yielded (not skipped by the early-return).
    """
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv-boundary-2")
    job.status = "active"
    job.append_chunk("token", "first")
    job.append_chunk("token", "second")
    from_pointer = len(job.chunks)

    async def append_later():
        await asyncio.sleep(0.05)
        job.append_chunk("token", "third")

    asyncio.create_task(append_later())
    asyncio.get_event_loop().call_later(0.2, job.mark_completed)

    events = []
    async for event in stream_from_active_job(job, from_pointer=from_pointer):
        if event.startswith("data: "):
            events.append(json.loads(event[6:]))

    token_events = [e for e in events if e.get("type") == "token"]
    assert any(e.get("chunk") == "third" for e in token_events)
    assert events[-1] == {"end": True}

    clear_job("test-conv-boundary-2")


@pytest.mark.asyncio
async def test_stream_resume_out_of_range_returns_empty():
    """from_pointer > len(chunks) is genuinely out of range -- generator
    must return immediately with no events. This guards the fix against
    over-relaxing the guard.
    """
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv-oor")
    job.status = "active"
    job.append_chunk("token", "a")
    job.append_chunk("token", "b")

    events = []
    async for event in stream_from_active_job(job, from_pointer=10):
        if event.startswith("data: "):
            events.append(json.loads(event[6:]))

    assert events == []


@pytest.mark.asyncio
async def test_second_message_clears_previous_chunks():
    """Sending a second message should clear chunks from the first message."""
    from backend.chat.stream_manager import StreamJob

    conv_id = "test-conv-2"
    clear_job(conv_id)

    job = get_or_create_job(conv_id, [])
    job.append_chunk("token", "First message response")
    assert len(job.chunks) == 1

    # Simulate what stream_chat does when reactivating the job.
    job.reset()
    job.append_chunk("token", "Second message response")

    assert len(job.chunks) == 1
    assert job.chunks[0]["chunk"] == "Second message response"

    clear_job(conv_id)
