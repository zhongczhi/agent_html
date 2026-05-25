from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional


class StreamJob:
    def __init__(
        self,
        conversation_id: str,
        messages: Optional[List[dict]] = None
    ):
        self.conversation_id = conversation_id
        self.status: Literal["active", "completed", "failed"] = "active"
        self.tokens: List[str] = []
        self.messages: List[dict] = messages or []
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def append_token(self, token: str) -> None:
        self.tokens.append(token)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        self.status = "completed"
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.now(timezone.utc)

    def get_full_content(self) -> str:
        return "".join(self.tokens)


STREAM_REGISTRY: Dict[str, StreamJob] = {}


def get_or_create_job(conversation_id: str, messages: List[dict]) -> StreamJob:
    if conversation_id in STREAM_REGISTRY:
        return STREAM_REGISTRY[conversation_id]
    job = StreamJob(conversation_id, messages)
    STREAM_REGISTRY[conversation_id] = job
    return job


def get_job(conversation_id: str) -> Optional[StreamJob]:
    return STREAM_REGISTRY.get(conversation_id)


def clear_job(conversation_id: str) -> None:
    if conversation_id in STREAM_REGISTRY:
        del STREAM_REGISTRY[conversation_id]