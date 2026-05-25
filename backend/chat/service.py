import logging
import uuid
from typing import AsyncGenerator, Optional

from backend.storage import file_storage
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, chain):
        self.chain = chain

    async def generate(
        self, message: str, conversation_id: Optional[str] = None, resume: bool = False
    ) -> AsyncGenerator[str, None]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []

        # Only append user message if not resuming (user message already saved by routes.py for new streams)
        if not resume:
            messages.append({"role": "user", "content": message})

        # Get or create stream job for this conversation
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

                # Handle LangChain content blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            token = block.get("text", "")
                            job.append_token(token)
                            yield token
                elif isinstance(content, str):
                    job.append_token(content)
                    yield content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            raise

        # Mark completed and save to storage
        job.mark_completed()
        messages.append({"role": "assistant", "content": job.get_full_content()})
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