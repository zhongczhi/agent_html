import asyncio
import logging
import uuid
from typing import Optional

from backend.storage import file_storage
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, chain):
        self.chain = chain

    async def generate_background(
        self,
        message: str,
        conversation_id: str
    ) -> None:
        """
        Background task: fetches from LLM and stores in StreamJob.
        Does not yield to caller - runs independently.
        """
        job = get_or_create_job(conversation_id, [])

        # Load history and append user message (if not already last message)
        # This avoids duplicates when message was already appended in stream_chat for new conversations
        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []
        if not messages or messages[-1]["content"] != message:
            messages.append({"role": "user", "content": message})
        job.messages = messages

        try:
            async for chunk in self.chain.astream(messages):
                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk["content"]
                elif isinstance(chunk, str):
                    content = chunk

                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "thinking":
                            thinking_text = block.get("thinking", "")
                            job.append_chunk("thinking", thinking_text)
                        elif block.get("type") == "text":
                            token = block.get("text", "")
                            job.append_chunk("token", token)
                elif isinstance(content, str):
                    job.append_chunk("token", content)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            return

        job.mark_completed()

        # Save to history
        full_content = "".join(c["chunk"] for c in job.chunks if c["type"] == "token")
        full_thinking = "".join(c["chunk"] for c in job.chunks if c["type"] == "thinking")
        messages.append({
            "role": "assistant",
            "content": full_content,
            "thinking": full_thinking
        })
        file_storage.save_conversation(conversation_id, messages)

    def get_history(self, conversation_id: str) -> Optional[dict]:
        return file_storage.get_conversation(conversation_id)

    def get_stream_status(self, conversation_id: str) -> dict:
        job = get_job(conversation_id)
        if job is None:
            return {
                "streaming": False,
                "status": "none",
                "chunks_count": 0,
                "is_complete": False
            }
        return {
            "streaming": job.status == "active",
            "status": job.status,
            "chunks_count": len(job.chunks),
            "is_complete": job.status == "completed"
        }