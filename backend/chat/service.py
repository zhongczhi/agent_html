# backend/chat/service.py
import logging
import uuid
from typing import AsyncGenerator, Optional

from backend.storage import file_storage

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, chain):
        self.chain = chain

    async def generate(
        self, message: str, conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []

        messages.append({"role": "user", "content": message})

        try:
            async for chunk in self.chain.astream(messages):
                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk["content"]
                elif isinstance(chunk, str):
                    content = chunk

                # Handle LangChain content blocks (list of text/image blocks)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield block.get("text", "")
                elif isinstance(content, str):
                    yield content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

        messages.append({"role": "assistant", "content": ""})
        file_storage.save_conversation(conversation_id, messages)

    def get_history(self, conversation_id: str) -> Optional[dict]:
        return file_storage.get_conversation(conversation_id)
