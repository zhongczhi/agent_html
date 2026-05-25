import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from backend.chat.routes import router
from backend.chat.stream_manager import clear_job


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def clean_registry():
    # Clean up before and after each test
    clear_job("test-conv-1")
    clear_job("test-conv-2")
    clear_job("delete-me")
    clear_job("status-test")
    yield
    clear_job("test-conv-1")
    clear_job("test-conv-2")
    clear_job("delete-me")
    clear_job("status-test")


@pytest.mark.asyncio
async def test_get_conversations_list(client, clean_registry):
    response = await client.get("/api/chat/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert isinstance(data["conversations"], list)


@pytest.mark.asyncio
async def test_stream_status_endpoint_no_job(client, clean_registry):
    response = await client.get("/api/chat/stream/status/nonexistent")
    assert response.status_code == 200
    data = response.json()
    assert data["streaming"] == False
    assert data["status"] == "none"
    assert data["tokens_count"] == 0


@pytest.mark.asyncio
async def test_get_chat_history(client, clean_registry):
    response = await client.get("/api/chat/history/test-conv-1")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "test-conv-1"
    assert data["messages"] == []


@pytest.mark.asyncio
async def test_delete_conversation(client, clean_registry):
    # First create a conversation by posting to stream
    response = await client.post(
        "/api/chat/stream",
        json={"message": "test", "conversation_id": "delete-me"}
    )
    # Consume the stream to trigger storage
    async for _ in response.aiter_bytes():
        pass

    # Delete it
    response = await client.delete("/api/chat/conversation/delete-me")
    assert response.status_code == 200
    assert response.json()["deleted"] == True


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation(client, clean_registry):
    response = await client.delete("/api/chat/conversation/non-existent")
    assert response.status_code == 200
    assert response.json()["deleted"] == False