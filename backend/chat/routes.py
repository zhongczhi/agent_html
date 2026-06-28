import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from backend.chat.chain import create_chain
from backend.chat.service import ChatService
from backend.chat.stream_manager import clear_job, consume_with_cleanup, get_job, get_or_create_job
from backend.storage import file_storage

router = APIRouter(prefix="/api/chat", tags=["chat"])


_chat_service: ChatService | None = None
# Module-level reference to the long-lived RagService, set by main.py's
# lifespan when RAG is enabled. Read at chat-service-construction time so
# the Depends() reference in route decorators still resolves correctly.
_rag_service = None


def set_rag_service(rag) -> None:
    """Set the module-level rag service. Called from main.py at startup."""
    global _rag_service
    _rag_service = rag
    # If the chat service was already constructed (e.g., tests that build
    # the service before app startup), inject now so a freshly returned
    # service picks it up too. The standard lifespan order builds the rag
    # service BEFORE the first chat request, so this branch is the
    # unusual case.
    global _chat_service
    if _chat_service is not None and _chat_service.rag_service is None:
        _chat_service.rag_service = rag


def get_chat_service() -> ChatService:
    """Lazy-init singleton. Using Depends() keeps this testable and avoids
    constructing the LLM client at module-import time."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(create_chain(), rag_service=_rag_service)
    return _chat_service


class RetrievalConfig(BaseModel):
    library: bool = True
    uploads: bool = True
    top_k: int = 4


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    retrieval: RetrievalConfig | None = None


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    updated_at: str | None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class DeleteResponse(BaseModel):
    deleted: bool


class StreamStatusResponse(BaseModel):
    status: str  # "none" | "pending" | "active" | "completed" | "failed"
    chunks_count: int
    partial_content: str | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _serialize_chunk(chunk: dict) -> dict:
    return {"chunk": chunk["chunk"], "type": chunk["type"], "message_id": chunk["message_id"]}


async def _replay_cached_chunks(job, from_pointer: int) -> AsyncGenerator[str, None]:
    """Yield cached chunks [from_pointer:]. Caller handles the end marker / queue drain."""
    for chunk in job.chunks[from_pointer:]:
        yield _sse(_serialize_chunk(chunk))


async def stream_from_inactive_job(
    job,
    from_pointer: int = 0
) -> AsyncGenerator[str, None]:
    """Replay the full cached chunk history for a completed/failed/pending job and stop."""
    if from_pointer < 0 or from_pointer > len(job.chunks):
        return
    async for event in _replay_cached_chunks(job, from_pointer):
        yield event
    yield _sse({"end": True})


async def stream_from_active_job(
    job,
    from_pointer: int = 0
) -> AsyncGenerator[str, None]:
    """Replay cached chunks [from_pointer:], then drain queued chunks until the end marker.

    `from_pointer == len(job.chunks)` is a valid boundary — the slice is empty but
    we still need to wait for queued chunks and the end marker.
    """
    if from_pointer < 0 or from_pointer > len(job.chunks):
        return

    async for event in _replay_cached_chunks(job, from_pointer):
        yield event

    last = int(job.chunks[-1]["message_id"]) if job.chunks else 0

    while True:
        try:
            chunk = await asyncio.wait_for(job.chunk_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if job.status != "active":
                yield _sse({"end": True})
                return
            continue
        if chunk is None:
            yield _sse({"end": True})
            return
        if int(chunk["message_id"]) <= last:
            continue
        yield _sse(_serialize_chunk(chunk))


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Ensure the conversation exists in storage so it appears in the sidebar
    # immediately (and is updated in-place as the assistant response streams).
    file_storage.create_conversation(conversation_id)
    file_storage.append_message(conversation_id, "user", request.message)

    job = get_or_create_job(conversation_id, [])

    if job.status != "active":
        job.reset()

    asyncio.create_task(
        chat_service.generate_background(
            request.message, conversation_id, retrieval=request.retrieval,
        )
    )

    return StreamingResponse(
        stream_from_active_job(job),
        media_type="text/event-stream",
    )


@router.get("/stream/{conversation_id}")
async def stream_resume(
    conversation_id: str,
    from_pointer: int = Query(default=0),
):
    job = get_job(conversation_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No stream job found")

    if job.status == "active":
        gen = stream_from_active_job(job, from_pointer=from_pointer)
    else:
        gen = stream_from_inactive_job(job, from_pointer=from_pointer)
    # Only the resume path cleans up. The job exists to support resume; once
    # a resume has fully replayed the cached chunks, the cache is no longer
    # needed. Cancellation / early return / exception leaves the job in place.
    return StreamingResponse(
        consume_with_cleanup(gen, job.conversation_id),
        media_type="text/event-stream",
    )


@router.get("/stream/status/{conversation_id}", response_model=StreamStatusResponse)
async def get_stream_status(conversation_id: str):
    job = get_job(conversation_id)
    if job is None:
        return StreamStatusResponse(status="none", chunks_count=0)
    return StreamStatusResponse(
        status=job.status,
        chunks_count=len(job.chunks),
        partial_content="".join(c["chunk"] for c in job.chunks if c["type"] == "token") or None,
    )


@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id: str):
    history = file_storage.get_conversation(conversation_id)
    if history is None:
        return ChatHistoryResponse(conversation_id=conversation_id, messages=[])
    return ChatHistoryResponse(**history)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """List all conversations with metadata."""
    conversations = file_storage.get_conversation_list()
    return ConversationListResponse(conversations=conversations)


@router.delete("/conversation/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(conversation_id: str):
    """Delete a conversation and clear any active stream."""
    clear_job(conversation_id)
    deleted = file_storage.delete_conversation(conversation_id)
    return DeleteResponse(deleted=deleted)
