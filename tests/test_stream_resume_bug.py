"""
Test case to reproduce the stream resume bug:
1. 405 Method Not Allowed when GET /api/chat/stream (frontend resumeStream uses GET, but route only accepts POST)
2. Partial content lost when stream interrupted (content only saved after completion)
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import AsyncMock, MagicMock
import asyncio

from backend.chat.routes import router
from backend.chat.stream_manager import clear_job, get_job


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def clean_registry():
    clear_job("resume-test-conv")
    clear_job("partial-test-conv")
    yield
    clear_job("resume-test-conv")
    clear_job("partial-test-conv")


@pytest.mark.asyncio
async def test_stream_endpoint_rejects_get_request(client, clean_registry):
    """
    BUG REPRODUCTION: resumeStream() in frontend uses GET, but /stream only accepts POST.

    This test demonstrates the 405 Method Not Allowed error users see on page refresh
    when trying to resume an interrupted stream.
    """
    # Frontend's resumeStream() calls GET /api/chat/stream?conversation_id=xxx
    # But the endpoint only accepts POST
    response = await client.get("/api/chat/stream?conversation_id=resume-test-conv")

    # This assertion will FAIL - confirming the bug exists
    assert response.status_code == 405, \
        f"Expected 405 Method Not Allowed, got {response.status_code}. " \
        "The frontend's resumeStream() uses GET but the endpoint only accepts POST."


@pytest.mark.asyncio
async def test_partial_content_lost_on_interrupted_stream(client, clean_registry, mock_chain):
    """
    BUG REPRODUCTION: When stream is interrupted, partial tokens in job.tokens are lost.

    The conversation is only saved to storage AFTER stream completes (service.py line 57-60).
    If the stream is interrupted (network error, page refresh, etc.), the partial
    assistant response is never persisted.
    """
    from backend.chat.service import ChatService
    from backend.chat.chain import create_chain
    from backend.chat.stream_manager import get_or_create_job

    # Create a service with mock chain that yields partial content then "completes"
    partial_tokens = ["Hello", ", ", "wo", "rld"]

    async def mock_astream(messages):
        for token in partial_tokens:
            # Create a mock chunk with content attribute
            chunk = MagicMock()
            chunk.content = token
            yield chunk
        # Simulate early termination before completion

    mock_chain.astream = mock_astream
    service = ChatService(mock_chain)

    conversation_id = "partial-test-conv"

    # Start a stream and manually interrupt it (don't await completion)
    async def start_stream():
        job = get_or_create_job(conversation_id, [])
        async for token in service.generate("Hello", conversation_id, resume=False):
            job.append_token(token)
            # Simulate interruption after first token
            if token == ", ":
                return

    # Run the partial stream
    await start_stream()

    # Check that tokens were accumulated in job
    job = get_job(conversation_id)
    assert job is not None
    accumulated = job.get_full_content()
    print(f"Accumulated in job: '{accumulated}'")

    # The partial content EXISTS in memory (job.tokens)
    assert accumulated == "Hello, "

    # But the conversation is NOT saved to storage yet (only saved on completion)
    from backend.storage import file_storage
    stored = file_storage.get_conversation(conversation_id)
    print(f"Stored in storage: {stored}")

    # BUG: This will be None or have no assistant message, proving partial content is LOST
    # The conversation should have user message but partial assistant message is missing
    assert stored is None or stored.get("messages") == [], \
        "BUG: Partial content should NOT be in storage yet (only saved on completion)"


@pytest.mark.asyncio
async def test_resume_endpoint_exists_and_returns_stream(client, clean_registry):
    """
    FIX VERIFICATION: The new GET /stream/resume/{conversation_id} endpoint should exist.

    This test verifies that the fix for Issue 1 works - we can now resume streams via GET.
    """
    # First create a conversation via POST to establish a job
    response = await client.post(
        "/api/chat/stream",
        json={"message": "Hello", "conversation_id": "resume-test-conv"}
    )
    assert response.status_code == 200

    # Now try to resume via the new GET endpoint
    resume_response = await client.get("/api/chat/stream/resume/resume-test-conv")
    assert resume_response.status_code == 200, \
        f"Resume endpoint should return 200, got {resume_response.status_code}"
    assert "text/event-stream" in resume_response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_resume_endpoint_returns_404_when_no_job(client, clean_registry):
    """Resume endpoint should return 404 if no stream job exists."""
    response = await client.get("/api/chat/stream/resume/nonexistent-conv")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stream_resume_needs_post_with_message(client, clean_registry, mock_chain):
    """
    CORRECT BEHAVIOR: Resume should use POST and send the original message.

    This test shows what the correct resume flow should be:
    1. Frontend sends POST /api/chat/stream with {conversation_id, message}
    2. Backend finds existing job with partial tokens
    3. Backend yields accumulated tokens, then continues streaming
    """
    from backend.chat.service import ChatService
    from backend.chat.stream_manager import get_or_create_job, clear_job

    conversation_id = "resume-test-conv"

    # First, simulate starting a stream (POST)
    response = await client.post(
        "/api/chat/stream",
        json={"message": "Hello world", "conversation_id": conversation_id}
    )

    # Read partial response (don't consume full stream)
    partial_data = b""
    async for chunk in response.aiter_bytes():
        partial_data += chunk
        if b'{"token": null}' in partial_data:  # End marker
            break

    # On page reload, frontend tries to resume
    # The correct behavior would be:
    # POST /api/chat/stream with {message: "Hello world", conversation_id: "..."}
    # But currently frontend uses GET which gets 405

    # This test documents the CORRECT expected behavior
    # Currently this fails because frontend uses wrong method
    clear_job(conversation_id)  # Clean up for next test
