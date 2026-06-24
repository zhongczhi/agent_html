import pytest
import asyncio

from backend.chat.stream_manager import (
    STREAM_REGISTRY,
    StreamJob,
    clear_job,
    consume_with_cleanup,
    get_or_create_job,
)


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


# --- consume_with_cleanup -----------------------------------------------------


@pytest.mark.asyncio
async def test_consume_with_cleanup_removes_job_after_full_consumption():
    """A resume that runs to completion removes the job from the registry —
    the cache has served its purpose."""
    conv_id = "test-cleanup-success"
    clear_job(conv_id)
    job = get_or_create_job(conv_id, [])
    job.append_chunk("token", "hello")
    job.mark_completed()

    async def gen():
        for chunk in job.chunks:
            yield f"data: {chunk}\n\n"
        yield "data: {\"end\": true}\n\n"

    assert conv_id in STREAM_REGISTRY
    consumed = [event async for event in consume_with_cleanup(gen(), conv_id)]
    assert len(consumed) == 2
    assert conv_id not in STREAM_REGISTRY, "job should be cleaned up after full consumption"


@pytest.mark.asyncio
async def test_consume_with_cleanup_keeps_job_when_no_events_yielded():
    """If the inner generator returns without yielding (e.g. from_pointer
    out of range), the job stays — the resume returned no data and the user
    may legitimately retry."""
    conv_id = "test-cleanup-empty"
    clear_job(conv_id)
    get_or_create_job(conv_id, [])

    async def gen():
        if False:
            yield "never"

    assert conv_id in STREAM_REGISTRY
    consumed = [event async for event in consume_with_cleanup(gen(), conv_id)]
    assert consumed == []
    assert conv_id in STREAM_REGISTRY, "job should be kept when nothing was yielded"


@pytest.mark.asyncio
async def test_consume_with_cleanup_keeps_job_on_cancellation():
    """If the consumer closes the stream mid-resume (the equivalent of the
    client dropping the SSE connection), the job stays so a future resume
    can continue."""
    conv_id = "test-cleanup-cancel"
    clear_job(conv_id)
    job = get_or_create_job(conv_id, [])
    job.append_chunk("token", "first")
    job.append_chunk("token", "second")
    job.mark_completed()

    async def gen():
        for chunk in job.chunks:
            yield f"data: {chunk}\n\n"
        yield "data: {\"end\": true}\n\n"

    assert conv_id in STREAM_REGISTRY

    # Simulate the client closing the connection after one event: explicitly
    # close the wrapper, which propagates GeneratorExit into the inner gen.
    wrapped = consume_with_cleanup(gen(), conv_id)
    await wrapped.__anext__()
    await wrapped.aclose()

    assert conv_id in STREAM_REGISTRY, "job should be kept on consumer cancellation"


@pytest.mark.asyncio
async def test_consume_with_cleanup_keeps_job_on_exception():
    """If the inner generator raises, the job stays — the delivery was
    incomplete and a future resume might salvage it."""
    conv_id = "test-cleanup-exception"
    clear_job(conv_id)
    get_or_create_job(conv_id, [])

    async def gen():
        yield "data: partial\n\n"
        raise RuntimeError("boom")

    assert conv_id in STREAM_REGISTRY
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in consume_with_cleanup(gen(), conv_id):
            pass

    assert conv_id in STREAM_REGISTRY, "job should be kept on exception"


@pytest.mark.asyncio
async def test_consume_with_cleanup_pop_is_idempotent():
    """If two concurrent resumes both finish, the second pop() is a no-op."""
    conv_id = "test-cleanup-concurrent"
    clear_job(conv_id)
    get_or_create_job(conv_id, [])

    async def gen():
        yield "data: x\n\n"

    # First consumption cleans up
    async for _ in consume_with_cleanup(gen(), conv_id):
        pass
    assert conv_id not in STREAM_REGISTRY

    # Second consumption: pop() must not raise even though the job is gone
    async def gen2():
        yield "data: y\n\n"

    async for _ in consume_with_cleanup(gen2(), conv_id):
        pass  # no exception
