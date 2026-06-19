import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional


class StreamJob:
    def __init__(
        self,
        conversation_id: str,
        messages: Optional[List[dict]] = None
    ):
        self.conversation_id = conversation_id
        self.status: Literal["pending", "active", "completed", "failed"] = "pending"
        self.chunks: List[dict] = []  # [{"chunk": "text", "type": "thinking|token", "message_id": "..."}]
        self.chunk_queue: asyncio.Queue = asyncio.Queue()
        self.messages: List[dict] = messages or []
        self.error: Optional[str] = None
        self.cancelled: bool = False  # Set by clear_job when the user deletes the conversation
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def append_chunk(self, chunk_type: str, text: str) -> None:
        """Add a chunk to both the chunks list and chunk_queue."""
        message_id = str(time.time_ns())
        chunk = {"chunk": text, "type": chunk_type, "message_id": message_id}
        self.chunks.append(chunk)
        self.chunk_queue.put_nowait(chunk)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        self.status = "completed"
        self.chunk_queue.put_nowait(None)  # End marker
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.chunk_queue.put_nowait(None)
        self.updated_at = datetime.now(timezone.utc)


STREAM_REGISTRY: Dict[str, StreamJob] = {}


def get_or_create_job(conversation_id: str, messages: List[dict]) -> StreamJob:
    if conversation_id in STREAM_REGISTRY:
        return STREAM_REGISTRY[conversation_id]
    job = StreamJob(conversation_id, messages)
    STREAM_REGISTRY[conversation_id] = job
    # Create empty conversation entry so it appears in conversation list immediately
    from backend.storage import file_storage
    file_storage.create_conversation(conversation_id)
    return job


def get_job(conversation_id: str) -> Optional[StreamJob]:
    return STREAM_REGISTRY.get(conversation_id)


def clear_job(conversation_id: str) -> None:
    if conversation_id in STREAM_REGISTRY:
        STREAM_REGISTRY[conversation_id].cancelled = True
        del STREAM_REGISTRY[conversation_id]