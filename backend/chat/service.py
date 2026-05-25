import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from backend.storage import file_storage
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, chain):
        self.chain = chain

    async def generate(
        self, message: str, conversation_id: Optional[str] = None, resume: bool = False
    ) -> AsyncGenerator[dict, None]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []

        if not resume:
            messages.append({"role": "user", "content": message})

        job = get_or_create_job(conversation_id, messages)

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
                        # Handle thinking blocks from LLM
                        if block.get("type") == "thinking":
                            thinking_text = block.get("thinking", "")
                            job.append_thinking(thinking_text)
                            yield {"thinking": thinking_text}
                        elif block.get("type") == "text":
                            token = block.get("text", "")
                            job.append_token(token)
                            yield {"token": token}
                elif isinstance(content, str):
                    job.append_token(content)
                    yield {"token": content}

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            raise

        job.mark_completed()
        full_thinking = job.get_full_thinking() if hasattr(job, 'get_full_thinking') else None
        messages.append({
            "role": "assistant",
            "content": job.get_full_content(),
            "thinking": full_thinking
        })
        file_storage.save_conversation(conversation_id, messages)

    def get_history(self, conversation_id: str) -> Optional[dict]:
        return file_storage.get_conversation(conversation_id)

    def get_stream_status(self, conversation_id: str) -> dict:
        job = get_job(conversation_id)
        if job is None:
            return {"streaming": False, "status": "none", "tokens_count": 0, "is_complete": False}
        return {
            "streaming": job.status == "active",
            "status": job.status,
            "tokens_count": len(job.tokens),
            "is_complete": job.status == "completed"
        }


# Monkey-patch StreamJob to add thinking support
from backend.chat.stream_manager import StreamJob

def _get_full_thinking(self):
    return getattr(self, '_thinking_content', '')

StreamJob.get_full_thinking = _get_full_thinking

def append_thinking(self, thinking: str):
    current = getattr(self, '_thinking_content', '')
    self._thinking_content = current + thinking
    self.updated_at = datetime.now(timezone.utc)

StreamJob.append_thinking = append_thinking