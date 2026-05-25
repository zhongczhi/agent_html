# backend/chat/routes.py
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.chat.chain import create_chain
from backend.chat.service import ChatService

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


async def generate_stream(message: str, conversation_id: str | None = None) -> AsyncGenerator[str, None]:
    full_response = []

    async for token in chat_service.generate(message, conversation_id):
        full_response.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"

    yield f"data: {json.dumps({'token': None})}\n\n"


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.message, request.conversation_id),
        media_type="text/event-stream",
    )


@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id: str):
    history = chat_service.get_history(conversation_id)
    if history is None:
        return ChatHistoryResponse(conversation_id=conversation_id, messages=[])
    return ChatHistoryResponse(**history)
