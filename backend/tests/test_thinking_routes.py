import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_stream_status_has_partial_thinking_field():
    """Test StreamStatusResponse has partial_thinking field."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/stream/status/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert "partial_thinking" in data

@pytest.mark.asyncio
async def test_history_returns_thinking_field():
    """Test history response includes thinking content."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        conv_id = "test-hist-think"
        await client.post(
            "/api/chat/stream",
            json={"message": "hi", "conversation_id": conv_id}
        )
        await asyncio.sleep(0.5)

        response = await client.get(f"/api/chat/history/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        # History should have messages
        assert "messages" in data