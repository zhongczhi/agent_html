import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.chat.stream_manager import STREAM_REGISTRY, clear_job, get_or_create_job


@pytest.mark.asyncio
async def test_stream_status_has_status_and_chunks_count():
    """Test StreamStatusResponse carries status (string) and chunks_count."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/stream/status/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert "chunks_count" in data
        assert "status" in data
        # Booleans removed — status string is the single contract.
        assert "streaming" not in data
        assert "is_complete" not in data


@pytest.mark.asyncio
async def test_stream_status_returns_none_for_unknown_conversation():
    """Test stream status returns 'none' status for unknown conversation."""
    transport = ASGITransport(app=app)
    conv_id = "test-partial-status"

    clear_job(conv_id)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/chat/stream/status/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "none"


@pytest.mark.asyncio
async def test_resume_stream_endpoint_exists():
    """Test GET /api/chat/stream/{conversation_id} endpoint exists."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/stream/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_post_starts_background_task():
    """Test POST /api/chat/stream starts background task and returns stream."""
    transport = ASGITransport(app=app)
    conv_id = "test-bg-stream"

    clear_job(conv_id)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat/stream",
            json={"message": "hi", "conversation_id": conv_id}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    clear_job(conv_id)


@pytest.mark.asyncio
async def test_delete_clears_job():
    """Test DELETE /conversation/{id} clears the stream job."""
    from backend.chat.stream_manager import get_job, get_or_create_job

    transport = ASGITransport(app=app)
    conv_id = "test-delete-job"

    get_or_create_job(conv_id, [])

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(f"/api/chat/conversation/{conv_id}")
        assert response.status_code == 200

    assert get_job(conv_id) is None


@pytest.mark.asyncio
async def test_resume_route_cleans_up_completed_job():
    """End-to-end: a completed job that is fully resumed via the route is
    removed from the registry. This is the primary D1 mitigation."""
    import json
    from backend.chat.stream_manager import StreamJob

    transport = ASGITransport(app=app)
    conv_id = "test-resume-cleanup-route"
    clear_job(conv_id)

    job = StreamJob(conv_id)
    job.append_chunk("thinking", "reasoning")
    job.append_chunk("token", "the answer")
    job.mark_completed()
    STREAM_REGISTRY[conv_id] = job
    assert conv_id in STREAM_REGISTRY

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", f"/api/chat/stream/{conv_id}") as r:
            assert r.status_code == 200
            async for _ in r.aiter_lines():
                pass  # drain

    assert conv_id not in STREAM_REGISTRY, \
        "completed job should be cleaned up after a full resume via the route"


@pytest.mark.asyncio
async def test_initial_stream_does_not_clean_up():
    """The POST initial stream does not use the cleanup wrapper — the job
    stays in the registry so a later resume can still replay it."""
    transport = ASGITransport(app=app)
    conv_id = "test-initial-no-cleanup"
    clear_job(conv_id)

    async with AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        async with client.stream(
            "POST", "/api/chat/stream",
            json={"message": "hi", "conversation_id": conv_id},
        ) as r:
            assert r.status_code == 200
            async for _ in r.aiter_lines():
                pass  # drain

    assert conv_id in STREAM_REGISTRY, \
        "initial stream must not clean up; only resume does"
    clear_job(conv_id)
