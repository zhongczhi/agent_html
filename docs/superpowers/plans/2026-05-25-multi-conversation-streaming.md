# Multi-Conversation & Streaming Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-conversation support with seamless streaming resume when switching conversations.

**Architecture:** Server-side StreamJob registry tracks active LLM streams per conversation. Background task accumulates tokens in memory. Clients resume from pointer position. Completed streams saved to conversations.json.

**Tech Stack:** FastAPI, SSE, Python asyncio, vanilla JS

---

## File Structure

```
backend/
├── main.py                      # No changes
├── config.py                    # No changes
├── chat/
│   ├── __init__.py             # No changes
│   ├── routes.py               # MODIFY: Add /conversations, /status, modify /stream
│   ├── chain.py                # No changes
│   ├── service.py             # MODIFY: Add StreamJob class, STREAM_REGISTRY, resume logic
│   └── stream_manager.py      # CREATE: StreamJob class and registry management
└── storage/
    ├── __init__.py             # No changes
    └── file_storage.py         # MODIFY: Add list_conversations, delete_conversation

frontend/
└── index.html                  # MODIFY: Add sidebar, conversation list, switch logic

tests/
├── conftest.py                 # MODIFY: Add fixtures for StreamJob
├── test_chat_service.py        # MODIFY: Add stream resume tests
└── test_stream_manager.py      # CREATE: Unit tests for StreamJob
```

---

## Data Structures

### StreamJob (in-memory, per server process)

```python
class StreamJob:
    conversation_id: str
    status: Literal["active", "completed", "failed"]
    tokens: List[str]           # All tokens generated (index = position)
    messages: List[dict]        # Full message history for resume
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

STREAM_REGISTRY: Dict[str, StreamJob] = {}  # conversation_id -> StreamJob
```

### conversations.json (updated)

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [...],
      "created_at": "ISO",
      "updated_at": "ISO"
    }
  }
}
```

---

## Task 1: Create StreamJob and StreamManager

**Files:**
- Create: `backend/chat/stream_manager.py`
- Modify: `backend/chat/service.py`
- Test: `tests/test_stream_manager.py`

- [ ] **Step 1: Write failing test for StreamJob**

```python
# tests/test_stream_manager.py
import pytest
from backend.chat.stream_manager import StreamJob, STREAM_REGISTRY, get_or_create_job, clear_job

def test_create_stream_job():
    job = StreamJob(conversation_id="test-123")
    assert job.conversation_id == "test-123"
    assert job.status == "active"
    assert job.tokens == []
    assert job.pointer == 0

def test_get_or_create_job_creates_new():
    clear_job("new-conv")
    job = get_or_create_job("new-conv", [{"role": "user", "content": "hi"}])
    assert job.conversation_id == "new-conv"
    assert job.status == "active"
    assert "new-conv" in STREAM_REGISTRY

def test_get_or_create_job_returns_existing():
    clear_job("existing-conv")
    job1 = get_or_create_job("existing-conv", [{"role": "user", "content": "hi"}])
    job2 = get_or_create_job("existing-conv", [{"role": "user", "content": "hi"}])
    assert job1 is job2  # Same instance

def test_job_status_transitions():
    job = StreamJob(conversation_id="status-test")
    assert job.status == "active"
    job.mark_completed()
    assert job.status == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_manager.py -v`
Expected: FAIL - module 'backend.chat.stream_manager' has no attribute 'StreamJob'

- [ ] **Step 3: Write StreamJob class**

```python
# backend/chat/stream_manager.py
from datetime import datetime
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
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def append_token(self, token: str) -> None:
        self.tokens.append(token)
        self.updated_at = datetime.utcnow()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.updated_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.utcnow()

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/chat/stream_manager.py tests/test_stream_manager.py
git commit -m "feat: add StreamJob class and stream manager"
```

---

## Task 2: Modify ChatService for Background Streaming

**Files:**
- Modify: `backend/chat/service.py`
- Test: `tests/test_chat_service.py`

- [ ] **Step 1: Write failing test for background streaming**

```python
# tests/test_chat_service.py - add new tests
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.chat.service import ChatService
from backend.chat.stream_manager import clear_job, get_job

@pytest.fixture
def mock_chain():
    chain = MagicMock()
    async def mock_aclose():
        pass
    chain.aclose = mock_aclose
    return chain

@pytest.fixture
def chat_service(mock_chain):
    return ChatService(mock_chain)

@pytest.mark.asyncio
async def test_generate_stores_tokens_in_job(chat_service, mock_chain):
    clear_job("stream-test")
    mock_chain.astream = MagicMock(return_value=iter(["H", "i", "!"]))

    tokens = []
    async for token in chat_service.generate("hello", "stream-test"):
        tokens.append(token)

    assert tokens == ["H", "i", "!"]
    job = get_job("stream-test")
    assert job.tokens == ["H", "i", "!"]
    assert job.status == "active"

@pytest.mark.asyncio
async def test_generate_marks_completed(chat_service, mock_chain):
    clear_job("complete-test")
    mock_chain.astream = MagicMock(return_value=iter(["done"]))

    async for _ in chat_service.generate("hello", "complete-test"):
        pass

    job = get_job("complete-test")
    assert job.status == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_service.py::test_generate_stores_tokens_in_job -v`
Expected: FAIL - job.tokens is empty

- [ ] **Step 3: Write implementation in ChatService**

```python
# backend/chat/service.py
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
        self, message: str, conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []

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
            return {"streaming": False, "status": "none", "tokens_count": 0}
        return {
            "streaming": job.status == "active",
            "status": job.status,
            "tokens_count": len(job.tokens),
            "is_complete": job.status == "completed"
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_service.py::test_generate_stores_tokens_in_job tests/test_chat_service.py::test_generate_marks_completed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/chat/service.py tests/test_chat_service.py
git commit -m "feat: integrate StreamJob tracking into ChatService"
```

---

## Task 3: Modify routes.py - Add New Endpoints and Resume Logic

**Files:**
- Modify: `backend/chat/routes.py`
- Test: `tests/test_chat_routes.py` (create)

- [ ] **Step 1: Write failing test for new endpoints**

```python
# tests/test_chat_routes.py
import pytest
from httpx import AsyncClient
from backend.chat.routes import router
from backend.chat.stream_manager import clear_job

@pytest.fixture
async def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_get_conversations_list(client):
    response = await client.get("/api/chat/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert isinstance(data["conversations"], list)

@pytest.mark.asyncio
async def test_delete_conversation(client):
    # First create a conversation
    response = await client.post(
        "/api/chat/stream",
        json={"message": "test", "conversation_id": "delete-me"}
    )
    # Consume the stream
    async for _ in response.aiter_bytes():
        pass

    # Delete it
    response = await client.delete("/api/chat/conversation/delete-me")
    assert response.status_code == 200
    assert response.json()["deleted"] == True

@pytest.mark.asyncio
async def test_stream_status_endpoint(client):
    clear_job("status-test")
    response = await client.get("/api/chat/stream/status/status-test")
    assert response.status_code == 200
    data = response.json()
    assert data["streaming"] == False
    assert data["status"] == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_routes.py -v`
Expected: FAIL - router has no /conversations, /conversation/{id}, /stream/status/{id} routes

- [ ] **Step 3: Write routes.py with new endpoints**

```python
# backend/chat/routes.py
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
    partial_content: str | None = None


async def generate_stream(
    message: str,
    conversation_id: str | None = None,
    resume: bool = False
) -> AsyncGenerator[str, None]:
    # For resume, get existing job
    job = get_job(conversation_id) if resume else get_or_create_job(conversation_id, [])

    if not resume:
        # Normal start: generate new
        async for token in chat_service.generate(message, conversation_id):
            yield f"data: {json.dumps({'token': token})}\n\n"
    else:
        # Resume: send existing tokens first, then continue if active
        if job and job.tokens:
            full_content = job.get_full_content()
            yield f"data: {json.dumps({'partial': full_content})}\n\n"

        if job and job.status == "active":
            # Continue streaming - generate new tokens from current position
            async for token in chat_service.generate(message, conversation_id):
                yield f"data: {json.dumps({'token': token})}\n\n"

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
    file_storage.delete_conversation(conversation_id)
    return DeleteResponse(deleted=True)


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
            partial_content=None
        )
    return StreamStatusResponse(
        streaming=job.status == "active",
        status=job.status,
        tokens_count=len(job.tokens),
        is_complete=job.status == "completed",
        partial_content=job.get_full_content() if job.tokens else None
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_routes.py -v`
Expected: PASS (may need adjustments based on actual file_storage API)

- [ ] **Step 5: Commit**

```bash
git add backend/chat/routes.py tests/test_chat_routes.py
git commit -m "feat: add conversation management endpoints and stream resume"
```

---

## Task 4: Modify file_storage.py - Add List and Delete

**Files:**
- Modify: `backend/storage/file_storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing test for new storage functions**

```python
# tests/test_storage.py - add new tests
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    # Override storage path
    from backend.storage import file_storage
    monkeypatch.setattr(file_storage, 'STORAGE_DIR', tmp_path / 'storage')
    monkeypatch.setattr(file_storage, 'CONVERSATIONS_FILE', tmp_path / 'storage' / 'conversations.json')
    file_storage._ensure_storage_dir()
    return tmp_path / 'storage'

def test_get_conversation_list_empty(temp_storage):
    from backend.storage import file_storage
    result = file_storage.get_conversation_list()
    assert result == []

def test_get_conversation_list_with_data(temp_storage):
    from backend.storage import file_storage
    file_storage.save_conversation("conv1", [{"role": "user", "content": "hello"}])
    file_storage.save_conversation("conv2", [{"role": "user", "content": "hi"}])
    result = file_storage.get_conversation_list()
    assert len(result) == 2
    ids = [c["conversation_id"] for c in result]
    assert "conv1" in ids
    assert "conv2" in ids

def test_delete_conversation(temp_storage):
    from backend.storage import file_storage
    file_storage.save_conversation("to-delete", [{"role": "user", "content": "hello"}])
    result = file_storage.delete_conversation("to-delete")
    assert result == True
    assert file_storage.get_conversation("to-delete") is None

def test_delete_conversation_not_exists(temp_storage):
    from backend.storage import file_storage
    result = file_storage.delete_conversation("non-existent")
    assert result == False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py::test_get_conversation_list_empty -v`
Expected: FAIL - get_conversation_list not defined

- [ ] **Step 3: Write new storage functions**

```python
# backend/storage/file_storage.py
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
CONVERSATIONS_FILE = STORAGE_DIR / "conversations.json"


def _ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_conversations() -> dict:
    _ensure_storage_dir()
    if not CONVERSATIONS_FILE.exists():
        return {"conversations": {}}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Failed to decode conversations.json, starting fresh")
        return {"conversations": {}}


def _save_conversations(data: dict) -> None:
    _ensure_storage_dir()
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_conversation(conversation_id: str) -> Optional[dict]:
    data = _load_conversations()
    conversations = data.get("conversations", {})
    return conversations.get(conversation_id)


def save_conversation(conversation_id: str, messages: list) -> None:
    data = _load_conversations()
    if "conversations" not in data:
        data["conversations"] = {}

    from datetime import datetime
    existing = data["conversations"].get(conversation_id)
    data["conversations"][conversation_id] = {
        "conversation_id": conversation_id,
        "messages": messages,
        "created_at": existing.get("created_at") if existing else datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    _save_conversations(data)


def append_message(conversation_id: str, role: str, content: str) -> list:
    data = _load_conversations()
    if "conversations" not in data:
        data["conversations"] = {}
    if conversation_id not in data["conversations"]:
        from datetime import datetime
        data["conversations"][conversation_id] = {
            "conversation_id": conversation_id,
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    data["conversations"][conversation_id]["messages"].append({"role": role, "content": content})
    data["conversations"][conversation_id]["updated_at"] = datetime.utcnow().isoformat()
    _save_conversations(data)
    return data["conversations"][conversation_id]["messages"]


def get_conversation_list() -> List[Dict[str, Any]]:
    """Get list of all conversations with metadata, sorted by updated_at desc."""
    data = _load_conversations()
    conversations = data.get("conversations", {})

    result = []
    for conv_id, conv_data in conversations.items():
        messages = conv_data.get("messages", [])
        first_msg = messages[0]["content"] if messages else ""
        title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg

        result.append({
            "conversation_id": conv_id,
            "title": title or "New conversation",
            "message_count": len(messages),
            "updated_at": conv_data.get("updated_at")
        })

    # Sort by updated_at descending
    result.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return result


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation. Returns True if deleted, False if not found."""
    data = _load_conversations()
    if "conversations" not in data:
        return False
    if conversation_id not in data["conversations"]:
        return False

    del data["conversations"][conversation_id]
    _save_conversations(data)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/storage/file_storage.py tests/test_storage.py
git commit -m "feat: add get_conversation_list and delete_conversation to storage"
```

---

## Task 5: Modify frontend/index.html - Add Sidebar and Conversation List

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Write the complete new frontend with sidebar**

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
        }
        .sidebar {
            width: 280px;
            background: #1a1a1a;
            color: white;
            display: flex;
            flex-direction: column;
            transition: margin-left 0.3s;
        }
        .sidebar.collapsed {
            margin-left: -280px;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .new-chat-btn {
            background: #4a90d9;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        .new-chat-btn:hover {
            background: #3a7bc8;
        }
        .toggle-sidebar {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
        }
        .conversation-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .conversation-item {
            padding: 12px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .conversation-item:hover {
            background: #333;
        }
        .conversation-item.active {
            background: #4a90d9;
        }
        .conversation-item .title {
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 14px;
        }
        .conversation-item .delete-btn {
            opacity: 0;
            background: none;
            border: none;
            color: #ff6b6b;
            cursor: pointer;
            padding: 4px 8px;
            font-size: 16px;
        }
        .conversation-item:hover .delete-btn {
            opacity: 1;
        }
        .conversation-item .streaming-badge {
            background: #f59e0b;
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 8px;
        }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
            display: flex;
            flex-direction: column;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-header {
            padding: 20px;
            background: #4a90d9;
            color: white;
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
        }
        .message.user {
            align-self: flex-end;
            background: #4a90d9;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: #e9e9e9;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        .chat-input-container {
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 12px;
        }
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 24px;
            font-size: 14px;
            outline: none;
        }
        .chat-input:focus {
            border-color: #4a90d9;
        }
        .send-button {
            padding: 12px 24px;
            background: #4a90d9;
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .send-button:hover {
            background: #3a7bc8;
        }
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #666;
            font-size: 14px;
        }
        .loading-dots {
            display: flex;
            gap: 4px;
        }
        .loading-dots span {
            width: 8px;
            height: 8px;
            background: #666;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" id="newChatBtn">+ New Chat</button>
            <button class="toggle-sidebar" id="toggleSidebar">&#9776;</button>
        </div>
        <div class="conversation-list" id="conversationList"></div>
    </div>

    <div class="main-content">
        <div class="chat-container">
            <div class="chat-header">
                <button class="toggle-sidebar" id="toggleSidebarMain">&#9776;</button>
                <span>Chatbot</span>
            </div>
            <div class="chat-messages" id="messages"></div>
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="messageInput" placeholder="Type your message..." autocomplete="off">
                <button class="send-button" id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const sidebar = document.getElementById('sidebar');
        const conversationList = document.getElementById('conversationList');
        const newChatBtn = document.getElementById('newChatBtn');
        const toggleSidebar = document.getElementById('toggleSidebar');
        const toggleSidebarMain = document.getElementById('toggleSidebarMain');

        let currentConversationId = localStorage.getItem('currentConversationId');
        let conversations = {};
        let isStreaming = false;
        let currentEventSource = null;

        // Initialize
        async function init() {
            await loadConversationList();
            if (currentConversationId) {
                await loadConversation(currentConversationId);
                await checkStreamStatus();
            }
        }

        // Load conversation list
        async function loadConversationList() {
            try {
                const response = await fetch('/api/chat/conversations');
                const data = await response.json();
                conversations = {};
                conversationList.innerHTML = '';

                for (const conv of data.conversations) {
                    conversations[conv.conversation_id] = conv;
                    addConversationToList(conv);
                }
            } catch (error) {
                console.error('Failed to load conversations:', error);
            }
        }

        // Add conversation to sidebar list
        function addConversationToList(conv) {
            const div = document.createElement('div');
            div.className = 'conversation-item' + (conv.conversation_id === currentConversationId ? ' active' : '');
            div.dataset.id = conv.conversation_id;

            const titleSpan = document.createElement('span');
            titleSpan.className = 'title';
            titleSpan.textContent = conv.title || 'New conversation';

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = '×';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteConversation(conv.conversation_id);
            };

            div.appendChild(titleSpan);
            div.appendChild(deleteBtn);
            div.onclick = () => switchConversation(conv.conversation_id);

            conversationList.appendChild(div);
        }

        // Switch to a conversation
        async function switchConversation(convId) {
            if (convId === currentConversationId) return;

            // Close current stream if any
            if (currentEventSource) {
                currentEventSource.close();
                currentEventSource = null;
            }

            currentConversationId = convId;
            localStorage.setItem('currentConversationId', convId);

            // Update active state in list
            document.querySelectorAll('.conversation-item').forEach(el => {
                el.classList.toggle('active', el.dataset.id === convId);
            });

            await loadConversation(convId);
            await checkStreamStatus();
        }

        // Load conversation history
        async function loadConversation(convId) {
            try {
                const response = await fetch(`/api/chat/history/${convId}`);
                const data = await response.json();

                messagesContainer.innerHTML = '';
                for (const msg of data.messages) {
                    addMessage(msg.role, msg.content);
                }
            } catch (error) {
                console.error('Failed to load conversation:', error);
            }
        }

        // Check stream status and resume if needed
        async function checkStreamStatus() {
            if (!currentConversationId) return;

            try {
                const response = await fetch(`/api/chat/stream/status/${currentConversationId}`);
                const status = await response.json();

                if (status.streaming) {
                    // Show streaming indicator
                    showStreamingBadge(true);
                    // Resume stream
                    await resumeStream();
                } else if (status.partial_content && !status.is_complete) {
                    // Partial content but not streaming - might need resume
                    // This is handled by the resume logic on send
                }
            } catch (error) {
                console.error('Failed to check stream status:', error);
            }
        }

        // Show/hide streaming badge on conversation item
        function showStreamingBadge(show) {
            const activeItem = document.querySelector(`.conversation-item[data-id="${currentConversationId}"]`);
            if (!activeItem) return;

            let badge = activeItem.querySelector('.streaming-badge');
            if (show && !badge) {
                badge = document.createElement('span');
                badge.className = 'streaming-badge';
                badge.textContent = 'Streaming';
                activeItem.appendChild(badge);
            } else if (!show && badge) {
                badge.remove();
            }
        }

        // Resume stream for current conversation
        async function resumeStream() {
            if (isStreaming || !currentConversationId) return;

            isStreaming = true;
            showLoading();
            showStreamingBadge(true);

            try {
                const response = await fetch(`/api/chat/stream?conversation_id=${currentConversationId}`);
                await processStreamResponse(response, true);
            } catch (error) {
                console.error('Stream error:', error);
                removeLoading();
                showStreamingBadge(false);
            }

            isStreaming = false;
            showStreamingBadge(false);
        }

        // Start new chat
        async function startNewChat() {
            if (currentEventSource) {
                currentEventSource.close();
                currentEventSource = null;
            }

            currentConversationId = null;
            localStorage.removeItem('currentConversationId');
            messagesContainer.innerHTML = '';
            document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
            messageInput.focus();
        }

        // Delete conversation
        async function deleteConversation(convId) {
            if (!confirm('Delete this conversation?')) return;

            try {
                await fetch(`/api/chat/conversation/${convId}`, { method: 'DELETE' });

                delete conversations[convId];
                const item = document.querySelector(`.conversation-item[data-id="${convId}"]`);
                if (item) item.remove();

                if (convId === currentConversationId) {
                    await startNewChat();
                }
            } catch (error) {
                console.error('Failed to delete conversation:', error);
            }
        }

        // Send message
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isStreaming) return;

            messageInput.value = '';
            sendButton.disabled = true;

            addMessage('user', message);
            showLoading();

            if (!currentConversationId) {
                // Create new conversation from first message
                currentConversationId = crypto.randomUUID();
                localStorage.setItem('currentConversationId', currentConversationId);
            }

            isStreaming = true;
            showStreamingBadge(true);

            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message,
                        conversation_id: currentConversationId
                    })
                });

                removeLoading();

                if (!response.ok) {
                    throw new Error('Failed to get response');
                }

                await processStreamResponse(response, false);

                // Refresh conversation list
                await loadConversationList();

            } catch (error) {
                removeLoading();
                addMessage('assistant', 'Sorry, an error occurred. Please try again.');
                console.error('Error:', error);
            }

            isStreaming = false;
            showStreamingBadge(false);
            sendButton.disabled = false;
            messageInput.focus();
        }

        // Process SSE stream response
        async function processStreamResponse(response, isResume) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = null;
            let partialReceived = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.partial && !partialReceived) {
                                // First message is partial content from resume
                                partialReceived = true;
                                assistantMessage = addMessage('assistant', data.partial);
                            } else if (data.token === null) {
                                // End of stream
                                break;
                            } else if (data.token) {
                                if (!assistantMessage) {
                                    assistantMessage = addMessage('assistant', '');
                                }
                                assistantMessage.textContent += data.token;
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            }
                        } catch (e) {
                            // Ignore parse errors for incomplete chunks
                        }
                    }
                }
            }
        }

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.textContent = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return messageDiv;
        }

        function showLoading() {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant loading';
            loadingDiv.id = 'loading';
            loadingDiv.innerHTML = '<span>Thinking</span><div class="loading-dots"><span></span><span></span><span></span></div>';
            messagesContainer.appendChild(loadingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeLoading() {
            const loading = document.getElementById('loading');
            if (loading) loading.remove();
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        newChatBtn.addEventListener('click', startNewChat);
        toggleSidebar.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
        toggleSidebarMain.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

        // Initialize
        init();
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify the file is complete**

Read the file and check it has all the features:
- Sidebar with conversation list
- New Chat button
- Delete conversation button
- Streaming badge
- Resume logic on conversation switch
- SSE stream handling with partial content

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add multi-conversation UI with sidebar and stream resume"
```

---

## Task 6: End-to-End Testing

**Files:**
- All files above

- [ ] **Step 1: Start the server**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- [ ] **Step 2: Open frontend and test**

1. Open http://localhost:8000
2. Send a message, verify it streams
3. Click "New Chat", send another message
4. Verify conversation list shows both
5. Send a message in Conv 1, quickly switch to Conv 2
6. Wait for stream to complete in Conv 1
7. Switch back to Conv 1 - verify complete message shown
8. Delete a conversation, verify it removes from list

- [ ] **Step 3: Run unit tests**

```bash
cd backend
pytest tests/ -v
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Multi-conversation support | Task 4, 5 |
| List conversations API | Task 3, 4 |
| Delete conversation API | Task 3, 4 |
| Stream status endpoint | Task 3, 2 |
| Stream resume logic | Task 3, 2 |
| Backend StreamJob registry | Task 1, 2 |
| Frontend sidebar + list | Task 5 |
| Frontend conversation switch | Task 5 |
| Frontend streaming indicator | Task 5 |
| Partial content replay on resume | Task 3, 5 |

---

## Type Consistency Check

| Item | Defined In | Used In |
|------|-----------|---------|
| `StreamJob` class | Task 1 | Task 2, 3 |
| `STREAM_REGISTRY` dict | Task 1 | Task 2, 3 |
| `get_or_create_job()` | Task 1 | Task 2, 3 |
| `get_job()` | Task 1 | Task 3 |
| `clear_job()` | Task 1 | Task 3 |
| `get_conversation_list()` | Task 4 | Task 5 |
| `delete_conversation()` | Task 4 | Task 5 |

All types are consistent across tasks.

---

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?