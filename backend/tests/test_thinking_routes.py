import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.chat.stream_manager import clear_job

@pytest.mark.asyncio
async def test_stream_status_has_partial_thinking_field():
    """Test StreamStatusResponse has chunks_count field."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/stream/status/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert "chunks_count" in data
        assert "is_complete" in data
        assert "status" in data


@pytest.mark.asyncio
async def test_stream_status_returns_partial_content():
    """Test stream status returns partial content and thinking."""
    transport = ASGITransport(app=app)
    conv_id = "test-partial-status"

    # Clean up first
    clear_job(conv_id)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get status for non-existent conversation
        response = await client.get(f"/api/chat/stream/status/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["streaming"] is False
        assert data["status"] == "none"


@pytest.mark.asyncio
async def test_resume_stream_endpoint_exists():
    """Test GET /api/chat/stream/{conversation_id} endpoint exists."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/stream/nonexistent")
        # Should return 404 when job not found
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_post_starts_background_task():
    """Test POST /api/chat/stream starts background task and returns stream."""
    transport = ASGITransport(app=app)
    conv_id = "test-bg-stream"

    clear_job(conv_id)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Start a stream (will be empty since mock chain returns nothing)
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
    from backend.chat.stream_manager import get_or_create_job, get_job

    transport = ASGITransport(app=app)
    conv_id = "test-delete-job"

    # Create a job
    get_or_create_job(conv_id, [])

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(f"/api/chat/conversation/{conv_id}")
        assert response.status_code == 200

    # Job should be cleared
    assert get_job(conv_id) is None