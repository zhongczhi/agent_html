import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.chat.chain import create_chain
from backend.chat.service import ChatService
from backend.chat.stream_manager import get_job, clear_job, get_or_create_job
from backend.storage import file_storage

router = APIRouter(prefix="/api/chat", tags=["chat"])
chain = create_chain()
chat_service = ChatService(chain)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


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
    streaming: bool
    status: str
    chunks_count: int
    is_complete: bool
    partial_content: str | None = None


async def stream_from_job(
    job,
    from_pointer: int = 0
) -> AsyncGenerator[str, None]:
    """Read chunks from StreamJob and yield SSE events.
    from_pointer is provided by frontend to resume from a specific position.
    Backend does NOT track sent_pointer.
    """
    # Error handling
    if from_pointer > len(job.chunks) or (from_pointer == len(job.chunks) and len(job.chunks) != 0):
        return

    # load strictly from chunks if inactive
    if job.status != "active":
        for chunk in job.chunks[from_pointer:]:
            yield f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type'], 'message_id': chunk['message_id']})}\n\n"
            print("ts: ", chunk['message_id'])
        yield f"data: {json.dumps({'end': True})}\n\n"
        return

    # first load from chunks when fetch from the queue
    for chunk in job.chunks[from_pointer:]:
        yield f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type'], 'message_id': chunk['message_id']})}\n\n"
        print("ts: ", chunk['message_id'], " ", f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type'], 'message_id': chunk['message_id']})}\n\n")
        print("ts: ", chunk['message_id'], " ", json.dumps({'chunk': chunk['chunk']}))
    if len(job.chunks) > 0:
        last = int(job.chunks[-1]['message_id'])
    else:
        last = 0

    # Stream from queue (new chunks being generated)
    while True:
        try:
            chunk = await asyncio.wait_for(job.chunk_queue.get(), timeout=0.5)
            if chunk is None:
                yield f"data: {json.dumps({'end': True})}\n\n"
                break
            if int(chunk['message_id']) <= last:
                continue
            yield f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type'], 'message_id': chunk['message_id']})}\n\n"
            print("ts: ", chunk['message_id'], " ", f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type'], 'message_id': chunk['message_id']})}\n\n")
            print("ts: ", chunk['message_id'], " ", json.dumps({'chunk': chunk['chunk']}))
        except asyncio.TimeoutError:
            if job.status != "active":
                yield f"data: {json.dumps({'end': True})}\n\n"
                for i, chunk in enumerate(job.chunks):
                    print(f"{i}: {chunk['message_id']}, len={len(chunk['chunk'])}") # check the data content
                break


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    conversation_id = request.conversation_id

    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    # Check if this is a new conversation (not yet in storage)
    existing_conv = file_storage.get_conversation(conversation_id)
    is_new_conversation = existing_conv is None

    job = get_or_create_job(conversation_id, [])

    # If job is not active, start background task
    if job.status != "active":
        job.status = "active"
        job.tokens = []
        job.thinking_tokens = []
        job.sent_pointer = 0
        job.thinking_sent_pointer = 0
        job.chunks = []  # Clear chunks from previous message

        # Append message to storage immediately for new conversations
        # so the conversation appears in the list with correct title
        if is_new_conversation:
            file_storage.append_message(conversation_id, "user", request.message)

        asyncio.create_task(
            chat_service.generate_background(request.message, conversation_id)
        )

    return StreamingResponse(
        stream_from_job(job),
        media_type="text/event-stream",
    )


@router.get("/stream/{conversation_id}")
async def stream_resume(
    conversation_id: str,
    from_pointer: int = Query(default=0)
):
    job = get_job(conversation_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No stream job found")

    return StreamingResponse(
        stream_from_job(job, from_pointer=from_pointer),
        media_type="text/event-stream",
    )


@router.get("/stream/status/{conversation_id}", response_model=StreamStatusResponse)
async def get_stream_status(conversation_id: str):
    job = get_job(conversation_id)
    if job is None:
        return StreamStatusResponse(
            streaming=False,
            status="none",
            chunks_count=0,
            is_complete=False,
        )
    return StreamStatusResponse(
        streaming=job.status == "active",
        status=job.status,
        chunks_count=len(job.chunks),
        is_complete=job.status == "completed",
        partial_content="".join(c["chunk"] for c in job.chunks if c["type"] == "token") if job.chunks else None
    )


@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id: str):
    history = chat_service.get_history(conversation_id)
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