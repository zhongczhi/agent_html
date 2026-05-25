import json
import logging
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.chat.chain import create_chain
from backend.chat.service import ChatService
from backend.chat.stream_manager import get_job, clear_job, get_or_create_job
from backend.storage import file_storage

logger = logging.getLogger(__name__)

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
    conversations: List[ConversationSummary]


class DeleteResponse(BaseModel):
    deleted: bool


class StreamStatusResponse(BaseModel):
    streaming: bool
    status: str
    tokens_count: int
    is_complete: bool
    partial_content: Optional[str] = None
    partial_thinking: Optional[str] = None


async def generate_stream(
    message: str,
    conversation_id: str | None = None,
    resume: bool = False
) -> AsyncGenerator[str, None]:
    job = get_job(conversation_id) if resume else get_or_create_job(conversation_id, [])
    thinking_complete = False  # Track when thinking phase ends
    has_seen_thinking = False  # Track if we've received any thinking content

    if not resume:
        async for event in chat_service.generate(message, conversation_id, resume=False):
            if isinstance(event, dict):
                if "thinking" in event:
                    yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                    has_seen_thinking = True
                elif "token" in event:
                    if not thinking_complete and has_seen_thinking:
                        # Only emit thinking_end if we've seen thinking and now receiving token
                        yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                        thinking_complete = True
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
            elif isinstance(event, str):
                # Legacy: plain string token
                if not thinking_complete:
                    yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                    thinking_complete = True
                yield f"data: {json.dumps({'token': event})}\n\n"
    else:
        # Resume: send existing tokens first
        if job and job.tokens:
            full_content = job.get_full_content()
            yield f"data: {json.dumps({'partial': full_content})}\n\n"
        # Send partial thinking if available
        if job and hasattr(job, 'get_full_thinking'):
            thinking = job.get_full_thinking()
            if thinking:
                yield f"data: {json.dumps({'partial_thinking': thinking})}\n\n"
                thinking_complete = True  # Already have thinking, mark complete

        if job and job.status == "active":
            async for event in chat_service.generate(message, conversation_id, resume=True):
                if isinstance(event, dict):
                    if "thinking" in event:
                        yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                    elif "token" in event:
                        if not thinking_complete:
                            yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                            thinking_complete = True
                        yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif isinstance(event, str):
                    if not thinking_complete:
                        yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                        thinking_complete = True
                    yield f"data: {json.dumps({'token': event})}\n\n"

    yield f"data: {json.dumps({'token': None})}\n\n"


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    conversation_id = request.conversation_id

    # Check if there's an active stream for this conversation
    existing_job = get_job(conversation_id) if conversation_id else None

    if existing_job and existing_job.status == "active":
        # Resume from existing stream
        return StreamingResponse(
            generate_stream(request.message, conversation_id, resume=True),
            media_type="text/event-stream",
        )

    # Save user message to storage BEFORE starting stream (so conversation appears in list immediately)
    if conversation_id:
        file_storage.append_message(conversation_id, "user", request.message)

    # Start new stream
    return StreamingResponse(
        generate_stream(request.message, conversation_id, resume=False),
        media_type="text/event-stream",
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


@router.get("/stream/resume/{conversation_id}")
async def resume_stream(conversation_id: str):
    """
    Resume a stream for an existing conversation.
    Returns SSE stream of the existing partial content plus any new tokens.
    """
    job = get_job(conversation_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No stream job found for this conversation")

    return StreamingResponse(
        generate_stream("", conversation_id, resume=True),
        media_type="text/event-stream",
    )


@router.get("/stream/status/{conversation_id}", response_model=StreamStatusResponse)
async def get_stream_status(conversation_id: str):
    """Check if a conversation has an active or completed stream."""
    job = get_job(conversation_id)
    if job is None:
        return StreamStatusResponse(
            streaming=False,
            status="none",
            tokens_count=0,
            is_complete=False,
            partial_content=None,
            partial_thinking=None
        )
    return StreamStatusResponse(
        streaming=job.status == "active",
        status=job.status,
        tokens_count=len(job.tokens),
        is_complete=job.status == "completed",
        partial_content=job.get_full_content() if job.tokens else None,
        partial_thinking=job.get_full_thinking() if hasattr(job, 'get_full_thinking') else None
    )