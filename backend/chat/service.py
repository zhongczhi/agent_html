import logging

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

        # The user message is already appended to storage by routes.stream_chat
        # (so the conversation appears in the sidebar with the correct title
        # during streaming). Load it from storage here — no need to append
        # again, since storage is the single source of truth.
        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []
        job.messages = messages

        try:
            async for chunk in self.chain.astream(messages):
                # If the user deleted the conversation mid-stream, stop early and
                # do NOT call mark_completed or save_conversation — that would
                # resurrect the deleted conversation in storage.
                if job.cancelled:
                    return

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
                            if thinking_text:
                                job.append_chunk("thinking", thinking_text)
                        elif block.get("type") == "text":
                            token = block.get("text", "")
                            if token:
                                job.append_chunk("token", token)
                elif isinstance(content, str) and content:
                    job.append_chunk("token", content)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            return

        # Defensive: if cancellation happened right as the LLM finished, do not save.
        if job.cancelled:
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

    def get_history(self, conversation_id: str) -> dict | None:
        return file_storage.get_conversation(conversation_id)
