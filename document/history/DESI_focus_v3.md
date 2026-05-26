# Frontend Enhancement + Thinking Content + Stream Resume - Design

## Overview

This design document covers two major feature sets:
1. **Thinking Content Streaming & Display** — Backend extraction and frontend display of LLM thinking content, message block internal scrolling, empty state UX, input auto-expansion, and markdown rendering
2. **Stream Resume Refactor** — Refactor streaming logic so that LLM calls run in background tasks and tokens are stored in `StreamJob`. Frontend caches tokens in localStorage and tracks position for resume.

---

## Part I: Thinking Content Streaming & Display

### 1. Architecture Decisions

#### 1.1 Thinking Token Flow

**Design Choice:** Thinking blocks from LLM are streamed first, then text blocks.

**Rationale:** Thinking represents the model's internal reasoning process which logically precedes the final response. Streaming thinking first provides immediate feedback to users that the model is "thinking."

**Alternative considered:** Interleave thinking and text tokens — rejected because thinking is internal reasoning and should complete before response begins.

#### 1.2 Thinking Persistence

**Design Choice:** Store thinking content in message history alongside response content.

**Rationale:** Users may want to review the model's reasoning for past responses. Storage schema is backward-compatible (thinking field is optional).

**Storage Schema:**
```json
{
  "role": "assistant",
  "content": "Response text...",
  "thinking": "Internal reasoning..."  // optional
}
```

**Streaming ↔ Storage Transformation:**
- During streaming: thinking emitted as `{"chunk": "...", "type": "thinking"}` events, accumulated in `StreamJob.chunks`
- On stream complete: thinking extracted from chunks, stored as `thinking` field in message history
- On resume: `partial_content` event sends accumulated text; resume position tracked by single `from_pointer` parameter

#### 1.3 Markdown Library

**Design Choice:** marked.js via CDN.

**Rationale:** Lightweight (39KB), no build step required, widely used and maintained. Plain HTML/JS architecture makes CDN inclusion simple.

#### 1.4 Scrollbar Auto-Hide Implementation

**Design Choice:** CSS `::-webkit-scrollbar` hidden by default, shown via `.scrollbar-visible` class on wheel event, with 3s setTimeout.

**Rationale:** Native browser scrollbar styling is not possible cross-browser. Using CSS class toggle allows smooth transitions and proper timer management per-message-block.

**Code pattern:**
```javascript
messageBlock.addEventListener('wheel', function() {
  this.classList.add('scrollbar-visible');
  clearTimeout(this._hideTimer);
  this._hideTimer = setTimeout(() => {
    this.classList.remove('scrollbar-visible');
  }, 3000);
});
```

#### 1.5 Input Auto-Expand

**Design Choice:** CSS `field-sizing: content` with JS fallback for older browsers.

**Rationale:** `field-sizing` is a new CSS property (Chrome 123+, Firefox 129+) that handles this natively. For older browsers, JS fallback measures content height and adjusts rows attribute.

**Fallback approach:**
```javascript
function autoResizeInput(textarea) {
  const clone = textarea.cloneNode();
  clone.style.position = 'absolute';
  clone.style.visibility = 'hidden';
  document.body.appendChild(clone);
  const newHeight = clone.scrollHeight;
  document.body.removeChild(clone);
  textarea.style.height = newHeight + 'px';
}
```

#### 1.6 Empty State Centering

**Design Choice:** CSS flexbox with `.empty` class toggle on messages container.

**Rationale:** Pure CSS solution, no JS position calculations. When messages exist, remove `.empty` class and flexbox naturally flows input to bottom.

```css
.chat-messages.empty {
  justify-content: center;
  align-items: center;
}
.chat-messages:not(.empty) {
  justify-content: flex-start;
}
```

---

### 2. System Architecture

#### 2.1 Data Flow: New Message

```
User types message
       ↓
POST /api/chat/stream {message, conversation_id}
       ↓
LLM returns chunks with thinking + text blocks
       ↓
Backend: yield {"chunk": "...", "type": "thinking"} for each thinking chunk
Backend: yield {"chunk": "...", "type": "token"} for each text chunk
Backend: yield {"end": true} when done
       ↓
Frontend: receives thinking chunks → builds thinking section
Frontend: receives token chunks → append to message content (with markdown rendering)
Frontend: receives end → complete
```

#### 2.2 Data Flow: Resume Stream

```
Check stream status
       ↓
GET /api/chat/stream/status/{id}
       ↓
Response: {
  streaming: true,
  chunks_count: N,
  partial_content: "...",
  is_complete: false
}
       ↓
GET /api/chat/stream/{id}?from_pointer=N
       ↓
Frontend: reads cached chunks from localStorage, renders them
Backend sends remaining chunks (type + content)
Frontend continues streaming until {"end": true}
```

---

### 3. Frontend Structure

#### 3.1 HTML Structure

```html
<div class="chat-messages" id="messages">
  <!-- Empty: has .empty class, input centered -->
  <!-- With messages: .empty removed, input at bottom -->

  <div class="message user">...</div>

  <div class="message assistant">
    <div class="thinking-section">
      <div class="thinking-content">...</div>
      <button class="thinking-toggle">Show more</button>
    </div>
    <div class="message-content markdown-body"></div>
  </div>
</div>
```

#### 3.2 CSS Classes

| Class | Purpose |
|-------|---------|
| `.thinking-section` | Container for thinking content |
| `.thinking-content` | The actual thinking text |
| `.thinking-toggle` | Show more/less button |
| `.thinking-collapsed` | Applied when thinking is collapsed |
| `.scrollbar-visible` | Override scrollbar hiding |
| `.empty` | On messages container when no messages |

#### 3.3 JavaScript Functions

| Function | Responsibility |
|----------|---------------|
| `processStreamResponse()` | Parse SSE, handle thinking + token events |
| `addMessage()` | Create message element with proper structure |
| `updateThinkingDisplay()` | Handle thinking content and fold/unfold |
| `setupScrollbarAutoHide()` | Attach wheel listener to message blocks |
| `autoResizeInput()` | Expand textarea with content |
| `marked.parse()` | Render markdown |

---

### 4. Backend Implementation

#### 4.1 chain.py Changes

**No changes required.** The chain continues to yield content blocks. The service layer handles the extraction and routing of thinking vs token chunks.

#### 4.2 service.py Changes

The `generate_background` method extracts thinking and text blocks from LLM chunks and passes them to StreamJob via `append_chunk(chunk_type, text)`:
```python
if isinstance(content, list):
    for block in content:
        if block.get("type") == "thinking":
            job.append_chunk("thinking", block.get("thinking", ""))
        elif block.get("type") == "text":
            job.append_chunk("token", block.get("text", ""))
elif isinstance(content, str):
    job.append_chunk("token", content)
```

#### 4.3 routes.py Changes

The `stream_from_job` generator yields chunks with type information. A single unified `from_pointer` parameter tracks resume position across both thinking and text chunks.

---

### 5. Edge Cases

#### 5.1 No Thinking Block

Some responses may not include a thinking block (model behavior, especially on short/simple responses). Handle gracefully:
- If LLM returns no thinking block, immediately send `thinking_end` before text tokens
- Frontend renders only the text response without thinking section

#### 5.2 Stream Interruption Mid-Thinking

If stream is interrupted during thinking phase:
- StreamJob in memory retains accumulated thinking tokens
- On resume: status endpoint returns `partial_thinking` with accumulated thinking
- Frontend shows partial thinking with "..." continuation indicator
- Backend continues streaming from interruption point

#### 5.3 Stream Interruption Mid-Text

If stream is interrupted during text phase (after thinking_end):
- StreamJob retains partial text
- On resume: status endpoint returns `partial_content` (unchanged behavior)
- Thinking is already complete, only text resumes

#### 5.4 Very Long Thinking

If thinking exceeds 100KB:
- Still stream normally (no truncation in v1)
- Frontend may need to virtualize rendering for very long thinking
- Architecture supports storing large thinking content

#### 5.5 Thinking Field Size Limits

For extreme cases (>1MB thinking):
- Not explicitly limited in storage
- Consider adding size check at save time in future version
- Current design allows arbitrary size

#### 5.6 Markdown in Thinking

Thinking content is NOT rendered as markdown. It's displayed as plain text to avoid any injection risks.

#### 5.7 Partial Resume After Complete

If a conversation was fully completed (streaming=false, is_complete=true) but user sends another message:
- Backend starts new stream, no resume needed
- History already contains full thinking + content

---

## Part II: Stream Resume Refactor

### 6. Overall Function Calling Process

#### 6.1 Continuous Chatting

**Frontend Flow:**
```
sendMessage()
    ↓
POST /api/chat/stream {message, conversation_id}
    ↓
receive SSE stream
    ↓
for each chunk:
    display chunk (thinking or token based on type)
    append to localStorage[chunks_{conv_id}]
    increment pointer
    save pointer to localStorage[pointer_{conv_id}]
    ↓
on end=true:
    conversation complete
```

**Backend Flow:**
```
POST /api/chat/stream
    ↓
get_or_create_job(conversation_id)
    ↓
if new job:
    asyncio.create_task(chat_service.generate_background(...))
    ↓
return StreamingResponse(stream_from_job(job))
    ↓
stream_from_job(job):
    while True:
        chunk = await job.chunk_queue.get()
        if chunk is None: break
        yield SSE_event(chunk)
```

#### 6.2 Switch Away While Streaming

**Frontend Flow:**
```
switchConversation(new_conv_id)
    ↓
currentEventSource.close()
    ↓
Backend StreamJob remains active
Background task continues calling LLM
Chunks accumulate in job.chunks + job.chunk_queue
```

**Backend Flow:**
```
Background task (unchanged):
    async for chunk in chain.astream(messages):
        job.append_chunk(chunk_type, text)
    job.chunk_queue.put_nowait(None)  # End marker
    job.mark_completed()
```

#### 6.3 Refresh While Streaming

**Frontend Flow:**
```
init()
    ↓
checkStreamStatus() → GET /api/chat/stream/status/{id}
    ↓
status.streaming = true
    ↓
Read from localStorage[chunks_{conv_id}] → render cached
Read pointer from localStorage[pointer_{conv_id}]
    ↓
resumeStream() → GET /api/chat/stream/{id}?from_pointer=N
    ↓
Receive remaining chunks via SSE
Append to displayed content
```

**Backend Flow:**
```
GET /api/chat/stream/{conv_id}?from_pointer=N
    ↓
job = get_job(conv_id)
    ↓
stream_from_job(job, from_pointer=N):
    current_len = len(job.chunks)
    if N < current_len:
        # Send accumulated chunks as partial
        for chunk in job.chunks[N:]:
            yield SSE_event(chunk)
    ↓
    # Continue reading from queue for new chunks
    while True:
        chunk = await job.chunk_queue.get()
        if chunk is None: break
        yield SSE_event(chunk)
```

#### 6.4 Resume After Switch/Refresh (While Still Streaming)

**Frontend Flow:**
```
Same as 6.3 Refresh Flow
```

**Backend Flow:**
```
Same as 6.3 Backend Flow
```

---

### 7. Backend Implementation

#### 7.1 StreamJob (stream_manager.py)

```python
import asyncio
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

#### 7.2 Service (service.py)

```python
class ChatService:
    async def generate_background(self, message: str, conversation_id: str) -> None:
        job = get_or_create_job(conversation_id, [])
        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []
        messages.append({"role": "user", "content": message})
        job.messages = messages

        try:
            async for chunk in self.chain.astream(messages):
                content = getattr(chunk, "content", chunk.get("content") if isinstance(chunk, dict) else chunk if isinstance(chunk, str) else None)
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "thinking":
                            job.append_chunk("thinking", block.get("thinking", ""))
                        elif block.get("type") == "text":
                            job.append_chunk("token", block.get("text", ""))
                elif isinstance(content, str):
                    job.append_chunk("token", content)
        except Exception as e:
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
```

#### 7.3 Routes (routes.py)

```python
class StreamStatusResponse(BaseModel):
    streaming: bool
    status: str
    chunks_count: int
    is_complete: bool
    partial_content: str | None = None

async def stream_from_job(job, from_pointer: int = 0) -> AsyncGenerator[str, None]:
    """Read chunks from StreamJob and yield SSE events."""
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

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    job = get_or_create_job(conversation_id, [])

    if job.status != "active":
        job.status = "active"
        job.chunks = []
        asyncio.create_task(chat_service.generate_background(request.message, conversation_id))

    return StreamingResponse(stream_from_job(job), media_type="text/event-stream")

@router.get("/stream/{conversation_id}")
async def stream_resume(conversation_id: str, from_pointer: int = Query(default=0)):
    job = get_job(conversation_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No stream job found")
    return StreamingResponse(stream_from_job(job, from_pointer=from_pointer), media_type="text/event-stream")

@router.get("/stream/status/{conversation_id}", response_model=StreamStatusResponse)
async def get_stream_status(conversation_id: str):
    job = get_job(conversation_id)
    if job is None:
        return StreamStatusResponse(streaming=False, status="none", chunks_count=0, is_complete=False)
    return StreamStatusResponse(
        streaming=job.status == "active",
        status=job.status,
        chunks_count=len(job.chunks),
        is_complete=job.status == "completed",
        partial_content="".join(c["chunk"] for c in job.chunks if c["type"] == "token") if job.chunks else None
    )
```

---

### 8. Frontend Implementation

#### 8.1 localStorage Keys (Constants)

```javascript
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`,
    STREAMING: (convId) => `streaming_${convId}`
};
```

#### 8.2 Init Flow

```javascript
async function init() {
    await loadConversationList();

    if (currentConversationId) {
        // Check if stream is still active
        status = await checkStreamStatus();
        if (status === true) return;  // Streaming resumed
        // Load displayed content from history
        await loadConversation(currentConversationId);
    } else {
        messagesContainer.classList.add('empty');
    }
}
```

#### 8.3 checkStreamStatus

```javascript
async function checkStreamStatus() {
    if (!currentConversationId) return;

    try {
        const response = await fetch(`/api/chat/stream/status/${currentConversationId}`);
        const status = await response.json();

        if (status.streaming) {
            showStreamingBadge(true);
            const pointer = parseInt(
                localStorage.getItem(STORAGE_KEYS.POINTER(currentConversationId)) || '0', 10
            );
            return await resumeStreamFromPosition(pointer);
        } else if (status.is_complete) {
            showStreamingBadge(false);
        }
    } catch (error) {
        console.error('Failed to check stream status:', error);
    }
}
```

#### 8.4 resumeStream Function

```javascript
async function resumeStreamFromPosition(pointer) {
    if (!currentConversationId) return;

    // First, render cached chunks from localStorage
    const { assistantMessage: cachedAssistant, rawContent: cachedRawContent } = renderCachedChunks(currentConversationId);

    setStreamingForConv(currentConversationId, true);
    showStreamingBadge(true);

    try {
        const url = `/api/chat/stream/${currentConversationId}?from_pointer=${pointer}`;
        const response = await fetch(url);
        await processStreamResponse(response, true, cachedAssistant, cachedRawContent);
    } catch (error) {
        console.error('Stream error:', error);
        showStreamingBadge(false);
    }

    setStreamingForConv(currentConversationId, false);
    showStreamingBadge(false);
    return true;
}
```

#### 8.5 processStreamResponse

```javascript
async function processStreamResponse(response, isResume, existingMessage = null, existingRawContent = '') {
    const convId = currentConversationId;
    if (!convId) return;

    if (!isResume && !isStreamingForConv(convId)) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = existingMessage;
    let partialReceived = existingMessage !== null;
    let currentPointer = 0;
    let rawContent = existingRawContent;

    currentPointer = parseInt(localStorage.getItem(STORAGE_KEYS.POINTER(convId)) || '0', 10);

    while (true) {
        if (convId !== currentConversationId) return;

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
                            const thinkingElement = assistantMessage.querySelector('.thinking-content');
                            if (thinkingElement) {
                                thinkingElement.textContent += data.chunk;
                                updateThinkingDisplay(assistantMessage);
                            }
                        } else if (data.type === 'token') {
                            const contentDiv = assistantMessage.querySelector('.message-content');
                            if (contentDiv) {
                                contentDiv.classList.remove('loading');
                                rawContent += data.chunk;
                                contentDiv.innerHTML = marked.parse(rawContent);
                            }
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            setupScrollbarAutoHide(assistantMessage);
                        }

                        const chunksCache = JSON.parse(
                            localStorage.getItem(STORAGE_KEYS.CHUNKS(convId)) || '[]'
                        );
                        chunksCache.push(data);
                        localStorage.setItem(STORAGE_KEYS.CHUNKS(convId), JSON.stringify(chunksCache));
                        currentPointer++;
                        localStorage.setItem(STORAGE_KEYS.POINTER(convId), currentPointer.toString());
                    } else if (data.end) {
                        localStorage.removeItem(STORAGE_KEYS.POINTER(convId));
                        setStreamingForConv(convId, false);
                        if (assistantMessage) {
                            const contentDiv = assistantMessage.querySelector('.message-content');
                            if (contentDiv) {
                                contentDiv.classList.remove('loading');
                                contentDiv.innerHTML = marked.parse(rawContent.trim());
                            }
                        }
                        return;
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }
        }
    }
}
```

#### 8.6 sendMessage

```javascript
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    if (isStreamingForConv(currentConversationId)) return;

    messageInput.value = '';
    sendButton.disabled = true;

    addMessage('user', message);
    const assistantMessage = addAssistantPlaceholder();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    if (!currentConversationId) {
        currentConversationId = crypto.randomUUID();
        localStorage.setItem('currentConversationId', currentConversationId);
    }

    // Cache user message
    const userChunksCache = JSON.parse(
        localStorage.getItem(STORAGE_KEYS.CHUNKS(currentConversationId)) || '[]'
    );
    userChunksCache.push({ type: 'user', content: message });
    localStorage.setItem(STORAGE_KEYS.CHUNKS(currentConversationId), JSON.stringify(userChunksCache));

    localStorage.setItem(STORAGE_KEYS.POINTER(currentConversationId), '0');
    setStreamingForConv(currentConversationId, true);
    showStreamingBadge(true);

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, conversation_id: currentConversationId })
        });

        if (!response.ok) throw new Error('Failed to get response');

        await loadConversationList();
        await processStreamResponse(response, false, assistantMessage);
    } catch (error) {
        if (assistantMessage && assistantMessage.parentNode) assistantMessage.remove();
        addMessage('assistant', 'Sorry, an error occurred. Please try again.');
        console.error('Error:', error);
    }

    setStreamingForConv(currentConversationId, false);
    showStreamingBadge(false);
    sendButton.disabled = false;
    messageInput.focus();
}
```

---

### 9. Error Handling

#### 9.1 Backend

| Scenario | Handling |
|----------|----------|
| LLM API error | `job.mark_failed()`, queue puts None |
| Queue timeout | Check `job.status` in loop, exit if not active |
| Job not found | Return 404 |

#### 9.2 Frontend

| Scenario | Handling |
|----------|----------|
| Stream fetch fails | Show error message, retry button |
| localStorage full | Fall back to memory-only caching |
| Parse error | Ignore malformed SSE lines |

---

### 10. Cleanup

#### 10.1 When conversation is deleted

```python
@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    clear_job(conversation_id)  # Remove from STREAM_REGISTRY
    deleted = file_storage.delete_conversation(conversation_id)
    return {"deleted": deleted}
```

#### 10.2 After stream completes

Frontend clears `pointer_*` localStorage on `end: true`. Backend keeps StreamJob in registry for status queries until conversation is deleted.

---

## 11. Testing Checklist

### Thinking Content & Display

#### Backend
- [x] Thinking blocks are extracted and yielded as `{"chunk": "...", "type": "thinking"}`
- [x] Token blocks yielded as `{"chunk": "...", "type": "token"}`
- [x] `end: true` event sent when stream completes
- [x] History API returns thinking field
- [x] Resume sends accumulated chunks

#### Frontend
- [x] Thinking displayed above response with "Show more" toggle when >3 lines
- [x] Message blocks scroll internally
- [x] Scrollbar auto-hides after 3s on wheel
- [x] Empty state input is centered
- [x] Input expands with content (field-sizing CSS + JS fallback)
- [x] Markdown renders correctly using marked.js
- [x] Resume works with cached chunks from localStorage

### Stream Resume

#### Backend
- [x] Background task stores chunks in StreamJob
- [x] Queue delivers chunks to /stream readers
- [x] `from_pointer` parameter skips already-sent chunks
- [x] Status API returns `chunks_count`
- [x] Job cleanup on conversation delete

#### Frontend
- [x] Chunks cached to localStorage on each receive
- [x] Pointer tracked and updated on each chunk
- [x] Init checks stream status before loading history
- [x] resumeStream passes `from_pointer` correctly
- [x] Pointer cleared on stream complete
- [x] Resume renders cached chunks first, then continues streaming

---

## 12. Files Summary

### Modify
| File | Changes |
|------|---------|
| `backend/chat/stream_manager.py` | Unified chunks list + chunk_queue (vs separate token/thinking) |
| `backend/chat/service.py` | Uses `append_chunk(chunk_type, text)` for both thinking and tokens |
| `backend/chat/routes.py` | Single `from_pointer` param, unified chunk events, `end: true` sentinel |
| `frontend/index.html` | Unified chunk handling, cached chunks rendering, streaming state per conversation |

### No Change
| File | Reason |
|------|--------|
| `backend/chat/chain.py` | Yields content blocks, no changes needed |
| `backend/storage/file_storage.py` | JSON supports new thinking field |
| `backend/main.py` | No routing changes |
| `backend/config.py` | No config needed |