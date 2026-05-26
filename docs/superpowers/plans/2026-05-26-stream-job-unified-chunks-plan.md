# StreamJob Unified Chunks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor StreamJob to use a single unified chunk queue with typed chunks instead of separate thinking and token queues. Frontend tracks pointer via localStorage.

**Architecture:** Backend stores all chunks in a single list and queue, tagged with type ("thinking" or "token"). Frontend manages resume position via localStorage and sends `from_pointer` param on resume. No sent_pointer tracking on backend.

**Tech Stack:** Python (FastAPI, asyncio), JavaScript (plain HTML/JS), SSE streaming

---

## File Map

| File | Responsibility |
|------|----------------|
| `backend/chat/stream_manager.py` | StreamJob class - stores chunks, single queue |
| `backend/chat/service.py` | ChatService - calls append_chunk for each LLM output |
| `backend/chat/routes.py` | API routes - stream_from_job generator, StreamStatusResponse |
| `frontend/index.html` | UI - processStreamResponse, localStorage pointer tracking |
| `backend/tests/test_stream_manager.py` | Tests for StreamJob |
| `backend/tests/test_chat_service.py` | Tests for ChatService |
| `backend/tests/test_chat_routes.py` | Tests for API routes |

---

## Task 1: Update StreamJob class in stream_manager.py

**Files:**
- Modify: `backend/chat/stream_manager.py`
- Test: `backend/tests/test_stream_manager.py`

- [ ] **Step 1: Write failing test for new StreamJob structure**

```python
# backend/tests/test_stream_manager.py
import pytest
import asyncio

def test_stream_job_has_unified_chunks():
    """StreamJob should have chunks list and chunk_queue, not thinking/token separation."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")

    # Should have chunks attribute
    assert hasattr(job, 'chunks')
    assert job.chunks == []

    # Should have chunk_queue
    assert hasattr(job, 'chunk_queue')
    assert isinstance(job.chunk_queue, asyncio.Queue)

    # Should NOT have old thinking/token attributes
    assert not hasattr(job, 'thinking_queue')
    assert not hasattr(job, 'token_queue')
    assert not hasattr(job, 'thinking_tokens')
    assert not hasattr(job, 'sent_pointer')

def test_append_chunk_adds_to_chunks_and_queue():
    """append_chunk should add dict with chunk and type to both list and queue."""
    from backend.chat.stream_manager import StreamJob

    job = StreamJob("test-conv")

    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello")

    assert len(job.chunks) == 2
    assert job.chunks[0] == {"chunk": "First thought", "type": "thinking"}
    assert job.chunks[1] == {"chunk": "Hello", "type": "token"}

    # Queue should have same items
    assert job.chunk_queue.get_nowait() == {"chunk": "First thought", "type": "thinking"}
    assert job.chunk_queue.get_nowait() == {"chunk": "Hello", "type": "token"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_stream_manager.py::test_stream_job_has_unified_chunks -v`
Expected: FAIL - AttributeError: 'StreamJob' object has no attribute 'chunks'

- [ ] **Step 3: Implement new StreamJob structure**

```python
# backend/chat/stream_manager.py - StreamJob class
class StreamJob:
    def __init__(
        self,
        conversation_id: str,
        messages: Optional[List[dict]] = None
    ):
        self.conversation_id = conversation_id
        self.status: Literal["pending", "active", "completed", "failed"] = "pending"
        self.chunks: List[dict] = []  # [{"chunk": "text", "type": "thinking|token"}]
        self.chunk_queue: asyncio.Queue = asyncio.Queue()
        self.messages: List[dict] = messages or []
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def append_chunk(self, chunk_type: str, text: str) -> None:
        """Add a chunk to both the chunks list and chunk_queue."""
        chunk = {"chunk": text, "type": chunk_type}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_stream_manager.py::test_stream_job_has_unified_chunks tests/test_stream_manager.py::test_append_chunk_adds_to_chunks_and_queue -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/chat/stream_manager.py backend/tests/test_stream_manager.py
git commit -m "feat(stream_manager): use unified chunks list and queue"
```

---

## Task 2: Update ChatService to use append_chunk

**Files:**
- Modify: `backend/chat/service.py`
- Test: `backend/tests/test_chat_service.py`

- [ ] **Step 1: Write failing test for new ChatService.generate_background**

```python
# backend/tests/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_generate_background_uses_append_chunk():
    """ChatService should call job.append_chunk with type, not separate methods."""
    from backend.chat.service import ChatService
    from backend.chat.stream_manager import StreamJob, STREAM_REGISTRY

    # Clear registry
    STREAM_REGISTRY.clear()

    mock_chain = AsyncMock()
    # Simulate LLM yielding a thinking block then a text block
    thinking_block = MagicMock()
    thinking_block.content = [{"type": "thinking", "thinking": "Let me think..."}]
    text_block = MagicMock()
    text_block.content = [{"type": "text", "text": "Hello!"}]
    mock_chain.astream.return_value = AsyncIterator([thinking_block, text_block])

    service = ChatService(mock_chain)

    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    job = STREAM_REGISTRY.get_or_create_job("test-conv", [])

    await service.generate_background("Hi", "test-conv")

    # Verify chunks were added with correct types
    assert len(job.chunks) >= 2
    # Find thinking and token chunks
    thinking_chunks = [c for c in job.chunks if c["type"] == "thinking"]
    token_chunks = [c for c in job.chunks if c["type"] == "token"]
    assert len(thinking_chunks) >= 1
    assert len(token_chunks) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_chat_service.py::test_generate_background_uses_append_chunk -v`
Expected: FAIL - job doesn't have append_chunk method

- [ ] **Step 3: Update ChatService.generate_background**

```python
# backend/chat/service.py - generate_background method
async def generate_background(
    self,
    message: str,
    conversation_id: str
) -> None:
    job = get_or_create_job(conversation_id, [])

    history = file_storage.get_conversation(conversation_id)
    messages = history["messages"] if history else []
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_chat_service.py::test_generate_background_uses_append_chunk -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/chat/service.py backend/tests/test_chat_service.py
git commit -m "feat(service): use append_chunk for unified chunks"
```

---

## Task 3: Update routes.py stream_from_job and API

**Files:**
- Modify: `backend/chat/routes.py`
- Test: `backend/tests/test_chat_routes.py`

- [ ] **Step 1: Write failing test for unified stream format**

```python
# backend/tests/test_chat_routes.py
import pytest
import asyncio
import json

@pytest.mark.asyncio
async def test_stream_from_job_yields_unified_chunks():
    """stream_from_job should yield chunks with chunk and type keys."""
    from backend.chat.stream_manager import StreamJob
    from backend.chat.routes import stream_from_job

    job = StreamJob("test-conv")
    job.status = "active"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")
    job.mark_completed()

    chunks = []
    async for event in stream_from_job(job, from_pointer=0):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    # Should have chunk events with type
    assert len(chunks) >= 3
    assert chunks[0] == {"chunk": "First thought", "type": "thinking"}
    assert chunks[1] == {"chunk": "Hello ", "type": "token"}
    assert chunks[2] == {"chunk": "world", "type": "token"}
    # Last should be end marker
    assert chunks[-1] == {"end": True}

@pytest.mark.asyncio
async def test_stream_resume_from_pointer():
    """stream_from_job should skip chunks before from_pointer."""
    from backend.chat.stream_manager import StreamJob
    from backend.chat.routes import stream_from_job

    job = StreamJob("test-conv")
    job.status = "active"
    job.append_chunk("thinking", "First thought")
    job.append_chunk("token", "Hello ")
    job.append_chunk("token", "world")
    job.mark_completed()

    # Resume from pointer 2 (skip first two chunks)
    chunks = []
    async for event in stream_from_job(job, from_pointer=2):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            chunks.append(data)

    # Should start from "world" token
    assert chunks[0] == {"chunk": "world", "type": "token"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_chat_routes.py::test_stream_from_job_yields_unified_chunks -v`
Expected: FAIL - stream_from_job doesn't yield unified chunks

- [ ] **Step 3: Update stream_from_job function**

```python
# backend/chat/routes.py - stream_from_job function
async def stream_from_job(
    job,
    from_pointer: int = 0
) -> AsyncGenerator[str, None]:
    """Read chunks from StreamJob and yield SSE events.
    from_pointer is provided by frontend to resume from a specific position.
    Backend does NOT track sent_pointer.
    """
    # Yield accumulated chunks from from_pointer position
    if from_pointer < len(job.chunks):
        for chunk in job.chunks[from_pointer:]:
            yield f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type']})}\n\n"

    # Stream from queue (new chunks being generated)
    while True:
        try:
            chunk = await asyncio.wait_for(job.chunk_queue.get(), timeout=0.5)
            if chunk is None:
                yield f"data: {json.dumps({'end': True})}\n\n"
                break
            yield f"data: {json.dumps({'chunk': chunk['chunk'], 'type': chunk['type']})}\n\n"
        except asyncio.TimeoutError:
            if job.status != "active":
                break
```

- [ ] **Step 4: Update StreamStatusResponse**

```python
# backend/chat/routes.py - StreamStatusResponse
class StreamStatusResponse(BaseModel):
    streaming: bool
    status: str
    chunks_count: int
    is_complete: bool
    partial_content: str | None = None
```

- [ ] **Step 5: Update get_stream_status endpoint**

```python
# In get_stream_status function, replace old fields:
return StreamStatusResponse(
    streaming=job.status == "active",
    status=job.status,
    chunks_count=len(job.chunks),
    is_complete=job.status == "completed",
    partial_content="".join(c["chunk"] for c in job.chunks if c["type"] == "token") if job.chunks else None
)
```

- [ ] **Step 6: Update stream_resume route parameter**

```python
# Change from_token, from_thinking to single from_pointer
@router.get("/stream/{conversation_id}")
async def stream_resume(
    conversation_id: str,
    from_pointer: int = Query(default=0)
):
    # ... rest same
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_chat_routes.py::test_stream_from_job_yields_unified_chunks tests/test_chat_routes.py::test_stream_resume_from_pointer -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/chat/routes.py backend/tests/test_chat_routes.py
git commit -m "feat(routes): unify stream format with typed chunks"
```

---

## Task 4: Update frontend index.html

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Update localStorage key**

```javascript
// In frontend/index.html, update STORAGE_KEYS
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`
};
```

- [ ] **Step 2: Update processStreamResponse for unified chunks**

```javascript
// Replace the processStreamResponse function body:
async function processStreamResponse(response, isResume, existingMessage = null) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = existingMessage;
    let partialReceived = existingMessage !== null;
    let currentPointer = 0;

    // Get current pointer from localStorage
    if (currentConversationId) {
        currentPointer = parseInt(localStorage.getItem(STORAGE_KEYS.POINTER(currentConversationId)) || '0', 10);
    }

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.chunk) {
                        if (!assistantMessage) {
                            assistantMessage = addMessage('assistant', '');
                        }

                        if (data.type === 'thinking') {
                            // Update thinking section
                            const thinkingElement = assistantMessage.querySelector('.thinking-content');
                            if (thinkingElement) {
                                thinkingElement.textContent += data.chunk;
                                updateThinkingDisplay(assistantMessage);
                            }
                        } else if (data.type === 'token') {
                            // Update content section
                            const contentDiv = assistantMessage.querySelector('.message-content');
                            if (contentDiv) {
                                contentDiv.textContent += data.chunk;
                                contentDiv.innerHTML = marked.parse(contentDiv.textContent);
                            }
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            setupScrollbarAutoHide(assistantMessage);
                        }

                        // Cache chunk and increment pointer
                        if (currentConversationId) {
                            const chunksCache = JSON.parse(
                                localStorage.getItem(STORAGE_KEYS.CHUNKS(currentConversationId)) || '[]'
                            );
                            chunksCache.push(data);
                            localStorage.setItem(
                                STORAGE_KEYS.CHUNKS(currentConversationId),
                                JSON.stringify(chunksCache)
                            );
                            currentPointer++;
                            localStorage.setItem(
                                STORAGE_KEYS.POINTER(currentConversationId),
                                currentPointer.toString()
                            );
                        }
                    } else if (data.end) {
                        // End of stream - clear cache
                        if (currentConversationId) {
                            localStorage.removeItem(STORAGE_KEYS.CHUNKS(currentConversationId));
                            localStorage.removeItem(STORAGE_KEYS.POINTER(currentConversationId));
                        }
                        // Replace loading indicator with final content
                        if (assistantMessage) {
                            const contentDiv = assistantMessage.querySelector('.message-content');
                            if (contentDiv && contentDiv.classList.contains('loading')) {
                                contentDiv.classList.remove('loading');
                            }
                        }
                        break;
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }
        }
    }
}
```

- [ ] **Step 3: Update resumeStreamFromPosition**

```javascript
async function resumeStreamFromPosition(pointer) {
    if (isStreaming || !currentConversationId) return;

    isStreaming = true;
    showStreamingBadge(true);

    const existingMessages = messagesContainer.querySelectorAll('.message.assistant');
    const lastAssistantMessage = existingMessages.length > 0 ? existingMessages[existingMessages.length - 1] : null;

    try {
        // Call stream with single pointer parameter
        const url = `/api/chat/stream/${currentConversationId}?from_pointer=${pointer}`;
        const response = await fetch(url);
        await processStreamResponse(response, true, lastAssistantMessage);
    } catch (error) {
        console.error('Stream error:', error);
        showStreamingBadge(false);
    }

    isStreaming = false;
    showStreamingBadge(false);
}
```

- [ ] **Step 4: Update checkStreamStatus**

```javascript
async function checkStreamStatus() {
    if (!currentConversationId) return;

    try {
        const response = await fetch(`/api/chat/stream/status/${currentConversationId}`);
        const status = await response.json();

        if (status.streaming) {
            showStreamingBadge(true);

            // Get pointer from localStorage
            const pointer = parseInt(
                localStorage.getItem(STORAGE_KEYS.POINTER(currentConversationId)) || '0',
                10
            );

            await resumeStreamFromPosition(pointer);
        } else if (status.is_complete) {
            showStreamingBadge(false);
        }
    } catch (error) {
        console.error('Failed to check stream status:', error);
    }
}
```

- [ ] **Step 5: Update sendMessage - clear cache properly**

```javascript
// In sendMessage, update cache clearing:
localStorage.removeItem(STORAGE_KEYS.CHUNKS(currentConversationId));
localStorage.setItem(
    STORAGE_KEYS.POINTER(currentConversationId),
    '0'
);
```

- [ ] **Step 6: Update loadConversation to use chunks**

```javascript
// In loadConversation, update how assistant messages are built:
for (const msg of data.messages) {
    if (msg.role === 'assistant') {
        const messageDiv = addMessage(msg.role, msg.content || '');

        // If there's thinking content, add it
        if (msg.thinking) {
            const thinkingSection = messageDiv.querySelector('.thinking-section');
            const thinkingContent = messageDiv.querySelector('.thinking-content');
            if (thinkingContent) {
                thinkingContent.textContent = msg.thinking;
                updateThinkingDisplay(messageDiv);
            }
        }

        // Render markdown for content
        const contentDiv = messageDiv.querySelector('.message-content');
        if (contentDiv && contentDiv.textContent) {
            contentDiv.innerHTML = marked.parse(contentDiv.textContent);
        }
    } else {
        addMessage(msg.role, msg.content);
    }
}
```

- [ ] **Step 7: Test by running the app**

Run the server: `cd backend && uvicorn main:app --reload --port 8000`

Open browser and verify:
1. New chat sends message and receives thinking + token chunks
2. Thinking section displays correctly
3. Stream resume works after page refresh
4. No console errors

- [ ] **Step 8: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): use unified chunk format with localStorage pointer"
```

---

## Task 5: Run full test suite

**Files:**
- Run: All test files

- [ ] **Step 1: Run full backend test suite**

Run: `cd backend && pytest tests/ -v`

Expected: All tests pass. If any fail, investigate and fix.

- [ ] **Step 2: Commit final changes**

```bash
git add -A
git commit -m "feat: complete StreamJob unified chunks refactor"
```

---

## Self-Review Checklist

- [ ] Spec coverage: All spec items implemented
- [ ] No placeholders: All steps have actual code
- [ ] Type consistency: `chunk` and `type` keys used consistently throughout
- [ ] Tests exist for backend changes
- [ ] Frontend handles all three cases: `data.chunk` with type, `data.end`

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `backend/chat/stream_manager.py` | Remove thinking/token queues, add `chunks` list, `append_chunk` method |
| `backend/chat/service.py` | Call `append_chunk` instead of separate append methods |
| `backend/chat/routes.py` | `stream_from_job` yields unified chunks, single `from_pointer` param |
| `frontend/index.html` | Single-path chunk handling, pointer in localStorage |
| `backend/tests/test_stream_manager.py` | Tests for new StreamJob |
| `backend/tests/test_chat_service.py` | Tests for new ChatService |
| `backend/tests/test_chat_routes.py` | Tests for new API format |
