import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal


class StreamJob:
    def __init__(
        self,
        conversation_id: str,
        messages: List[dict] | None = None
    ):
        self.conversation_id = conversation_id
        self.status: Literal["pending", "active", "completed", "failed"] = "pending"
        self.chunks: List[dict] = []  # [{"chunk": "text", "type": "thinking|token", "message_id": "..."}]
        self.chunk_queue: asyncio.Queue = asyncio.Queue()
        self.messages: List[dict] = messages or []
        self.error: str | None = None
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

    def reset(self) -> None:
        """Reset job state to start a new message in the same conversation."""
        self.status = "active"
        self.chunks = []
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
    return job


def get_job(conversation_id: str) -> StreamJob | None:
    return STREAM_REGISTRY.get(conversation_id)


def clear_job(conversation_id: str) -> None:
    # Set cancelled on the job BEFORE removing it from the registry. The
    # background task holds a reference to the job and checks `cancelled`
    # to abort cleanly; setting the flag on the live object first ensures
    # the check sees it even if the task is between iterations.
    if conversation_id in STREAM_REGISTRY:
        STREAM_REGISTRY[conversation_id].cancelled = True
        del STREAM_REGISTRY[conversation_id]


async def consume_with_cleanup(gen, conversation_id: str):
    """Wrap a stream generator and remove the StreamJob from the registry
    after the generator is fully consumed. Cancellation, early return, or
    exception leaves the job in place so a future resume can continue.

    Used by the resume route only. The job exists to support resume; once a
    resume has delivered the full cached chunk history (including the end
    marker), there is nothing left to resume, so the cache can go.
    """
    completed = False
    any_event = False
    try:
        async for event in gen:
            any_event = True
            yield event
        completed = True
    finally:
        if completed and any_event:
            STREAM_REGISTRY.pop(conversation_id, None)
