# Chatbot Project - Detailed Design

## 1. Architecture Decisions

### 1.1 Communication Protocol: SSE

**Choice:** Server-Sent Events (SSE) for streaming responses.

**Rationale:**
- Unidirectional (server → client) fits chat streaming use case
- FastAPI native `StreamingResponse` support
- Frontend uses Fetch API with `ReadableStream`
- Simpler than WebSocket for this use case

### 1.2 LLM Integration: LangChain LCEL

**Choice:** LangChain with LangChain Expression Language (LCEL).

**Rationale:**
- LCEL enables clean streaming chain composition
- Built-in support for Anthropic models via MiniMax endpoint
- Easy to swap models without changing application logic
- Environment variables: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`

### 1.3 Storage: JSON Files

**Choice:** File-based JSON storage (`storage/conversations.json`).

**Rationale:**
- Zero infrastructure setup
- Easy to inspect and debug
- Migration path to PostgreSQL defined for future
- Per-conversation structure allows future sharding

### 1.4 Frontend: Plain HTML/JS

**Choice:** No framework, single `index.html`.

**Rationale:**
- Zero build step
- Easy to understand and modify
- Served directly by FastAPI
- Native `ReadableStream` for SSE handling

### 1.5 Configuration: Environment Variables

**Choice:** `pydantic-settings` with `.env` file.

**Rationale:**
- Standard Python pattern
- Type validation on startup
- Secrets kept out of codebase

### 1.6 Thinking Content Streaming

**Choice:** Thinking blocks from LLM are streamed first, then text blocks.

**Rationale:** Thinking represents the model's internal reasoning process which logically precedes the final response. Streaming thinking first provides immediate feedback to users that the model is "thinking."

### 1.7 Thinking Persistence

**Choice:** Store thinking content in message history alongside response content.

**Rationale:** Users may want to review the model's reasoning for past responses. Storage schema is backward-compatible (thinking field is optional).

### 1.8 Markdown Library

**Choice:** marked.js via CDN.

**Rationale:** Lightweight (39KB), no build step required, widely used and maintained. Plain HTML/JS architecture makes CDN inclusion simple.

### 1.9 Stream Resume Architecture

**Choice:** LLM calls run in background tasks; tokens stored in `StreamJob`. Frontend caches tokens in localStorage and tracks position for resume.

**Rationale:** Allows seamless continuation when user switches tabs/refreshes during streaming. Background task continues generating while frontend reconnects.

---

## 2. System Architecture

```
┌─────────────┐     SSE/HTTP      ┌─────────────────┐
│   Frontend  │ ◄───────────────► │   FastAPI       │
│  (Plain     │                   │   Backend       │
│   HTML/JS)  │                   │   (LangChain)   │
└─────────────┘                   └────────┬────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                       │
                    │              ┌───────────────┐        ┌───────────────┐
                    │              │  chat domain  │        │ storage domain│
                    │              └───────────────┘        └───────────────┘
                    └──────────────────────────────────────────────────────┘
```

---

## 3. Backend Structure

```
backend/
├── main.py                 # FastAPI app entry, serves frontend + mounts routers
├── config.py               # Pydantic Settings from env vars
├── chat/
│   ├── routes.py           # /api/chat/* endpoints
│   ├── chain.py            # LangChain LCEL chain definition
│   ├── service.py          # ChatService orchestration
│   └── stream_manager.py   # StreamJob tracking + STREAM_REGISTRY
└── storage/
    └── file_storage.py     # JSON file read/write operations
```

### Domain Pattern

Each domain has three files:
- `routes.py` — HTTP interface (endpoints)
- `chain.py` — Business logic composition (LCEL)
- `service.py` — Orchestration and state management

### Stream Registry

In-memory registry (`STREAM_REGISTRY`) tracks active streams:
- `conversation_id` → `StreamJob` mapping
- Enables stream resume on conversation switch/refresh

---

## 4. API Protocol

### SSE Format (Unified Chunk Format)

```
data: {"chunk": "Hello", "type": "thinking"}\n\n
data: {"chunk": "!", "type": "token"}\n\n
data: {"end": true}\n\n
```

| Event | Format | Description |
|-------|--------|-------------|
| Thinking | `data: {"chunk": "...", "type": "thinking"}` | Single thinking token |
| Token | `data: {"chunk": "...", "type": "token"}` | Single text token |
| End | `data: {"end": true}` | Stream complete |

### Request/Response Flow

**New Stream:**
1. Parse request body (`message`, `conversation_id`)
2. Generate UUID for new conversation if needed
3. Retrieve existing history or start fresh
4. Append user message to history
5. Start background task for LLM streaming
6. Stream thinking chunks first, then token chunks via SSE
7. Save assistant response (with thinking) on completion

**Resume Stream:**
1. Check `STREAM_REGISTRY` for active job
2. Send accumulated chunks from `from_pointer` position
3. Continue streaming from current queue position

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/stream` | Start new chat stream |
| GET | `/api/chat/stream/{conversation_id}?from_pointer=N` | Resume stream from position N |
| GET | `/api/chat/stream/status/{conversation_id}` | Get stream status |
| DELETE | `/api/chat/conversation/{conversation_id}` | Delete conversation + cleanup |

---

## 5. Data Storage

### conversations.json

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "thinking": "..."}
      ],
      "created_at": "ISO8601",
      "updated_at": "ISO8601"
    }
  }
}
```

Note: `thinking` field is optional. History API returns thinking field for assistant messages.

### Storage Operations

| Function | Behavior |
|----------|----------|
| `get_conversation(id)` | Retrieve conversation or `None` |
| `save_conversation(id, messages)` | Persist entire messages array |
| `append_message(id, role, content)` | Append single message |
| `list_conversations()` | Return sorted list (updated_at desc) |
| `delete_conversation(id)` | Remove conversation, clear stream |

**Error Handling:** Invalid JSON returns empty dict with warning log.

---

## 6. Frontend Implementation

### Responsibilities

- Manage conversation list UI and state
- Store `currentConversationId` in `localStorage`
- Cache chunks to localStorage during streaming
- On page load: check stream status, resume if needed
- Handle SSE stream parsing and display

### Page Load Flow

```
1. Read currentConversationId from localStorage
2. GET /api/chat/stream/status/{id}
3. If streaming=true → GET /stream/{id}?from_pointer=N to resume
4. If streaming=false → GET /api/chat/history/{id}
```

### localStorage Keys

| Key | Purpose |
|-----|---------|
| `chunks_{conv_id}` | Cached chunks for resume |
| `pointer_{conv_id}` | Current position in stream |
| `streaming_{conv_id}` | Active streaming state per conversation |

### Conversation Switch Flow

```
1. User clicks conversation in sidebar
2. If currently streaming → SSE connection closes
3. Server continues streaming (STREAM_REGISTRY intact)
4. Chunks accumulate in StreamJob
5. Load clicked conversation's history
6. If it was streaming → resume via GET /stream/{id}?from_pointer=N
```

### UI Styling

- Centered chat container (max 800px)
- User messages: right-aligned, blue bubble
- Assistant messages: left-aligned, gray bubble
- Thinking section: displayed above response, collapsible
- Loading indicator: "Thinking..." with animated dots

### Frontend Structure

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

### CSS Classes

| Class | Purpose |
|-------|---------|
| `.thinking-section` | Container for thinking content |
| `.thinking-content` | The actual thinking text |
| `.thinking-toggle` | Show more/less button |
| `.thinking-collapsed` | Applied when thinking is collapsed |
| `.scrollbar-visible` | Override scrollbar hiding |
| `.empty` | On messages container when no messages |
| `.markdown-body` | Rendered markdown content |

### JavaScript Functions

| Function | Responsibility |
|----------|---------------|
| `processStreamResponse()` | Parse SSE, handle thinking + token events |
| `addMessage()` | Create message element with proper structure |
| `updateThinkingDisplay()` | Handle thinking content and fold/unfold |
| `setupScrollbarAutoHide()` | Attach wheel listener to message blocks |
| `autoResizeInput()` | Expand textarea with content |
| `marked.parse()` | Render markdown |
| `resumeStreamFromPosition()` | Resume stream from localStorage cache |

---

## 7. StreamJob Architecture

### StreamJob (stream_manager.py)

```python
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

### Background Task Flow

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

---

## 8. Edge Cases

### 8.1 No Thinking Block

Some responses may not include a thinking block (model behavior, especially on short/simple responses). Handle gracefully:
- If LLM returns no thinking block, immediately send text tokens only
- Frontend renders only the text response without thinking section

### 8.2 Stream Interruption Mid-Thinking

If stream is interrupted during thinking phase:
- StreamJob in memory retains accumulated thinking tokens
- On resume: status endpoint returns streaming state
- Frontend shows partial thinking with "..." continuation indicator
- Backend continues streaming from interruption point

### 8.3 Stream Interruption Mid-Text

If stream is interrupted during text phase (after thinking):
- StreamJob retains partial text
- On resume: status endpoint returns partial_content
- Thinking is already complete, only text resumes

### 8.4 Very Long Thinking

If thinking exceeds 100KB:
- Still stream normally (no truncation in v1)
- Frontend may need to virtualize rendering for very long thinking
- Architecture supports storing large thinking content

### 8.5 Thinking Field Size Limits

For extreme cases (>1MB thinking):
- Not explicitly limited in storage
- Consider adding size check at save time in future version
- Current design allows arbitrary size

### 8.6 Markdown in Thinking

Thinking content is NOT rendered as markdown. It's displayed as plain text to avoid any injection risks.

### 8.7 Partial Resume After Complete

If a conversation was fully completed (streaming=false, is_complete=true) but user sends another message:
- Backend starts new stream, no resume needed
- History already contains full thinking + content

### 8.8 Refresh While Streaming

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

---

## 9. Error Handling

### Backend

| Scenario | Handling |
|----------|----------|
| LLM API error | `job.mark_failed()`, queue puts None |
| Queue timeout | Check `job.status` in loop, exit if not active |
| Job not found | Return 404 |

### Frontend

| Scenario | Handling |
|----------|----------|
| Stream fetch fails | Show error message, retry button |
| localStorage full | Fall back to memory-only caching |
| Parse error | Ignore malformed SSE lines |

---

## 10. Cleanup

### When conversation is deleted

```python
@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    clear_job(conversation_id)  # Remove from STREAM_REGISTRY
    deleted = file_storage.delete_conversation(conversation_id)
    return {"deleted": deleted}
```

### After stream completes

Frontend clears `pointer_*` localStorage on `end: true`. Backend keeps StreamJob in registry for status queries until conversation is deleted.

---

## 11. Testing Strategy

### Test Approach

- Mock LLM calls via `langchain.anthropic.ChatModel`
- Use temporary directories for storage tests
- HTTP tests via `httpx.AsyncClient` against FastAPI TestClient

### Test Coverage

| File | What is Tested |
|------|----------------|
| `test_chat_service.py` | `ChatService.generate()` with mocked LLM |
| `test_storage.py` | JSON read/write roundtrip |
| `test_chat_routes.py` | HTTP endpoint responses |
| `test_stream_manager.py` | StreamJob state transitions |

### Test Dependencies

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

### Testing Checklist

#### Thinking Content & Display

**Backend:**
- [x] Thinking blocks are extracted and yielded as `{"chunk": "...", "type": "thinking"}`
- [x] Token blocks yielded as `{"chunk": "...", "type": "token"}`
- [x] `end: true` event sent when stream completes
- [x] History API returns thinking field
- [x] Resume sends accumulated chunks

**Frontend:**
- [x] Thinking displayed above response with "Show more" toggle when >3 lines
- [x] Message blocks scroll internally
- [x] Scrollbar auto-hides after 3s on wheel
- [x] Empty state input is centered
- [x] Input expands with content (field-sizing CSS + JS fallback)
- [x] Markdown renders correctly using marked.js
- [x] Resume works with cached chunks from localStorage

#### Stream Resume

**Backend:**
- [x] Background task stores chunks in StreamJob
- [x] Queue delivers chunks to /stream readers
- [x] `from_pointer` parameter skips already-sent chunks
- [x] Status API returns `chunks_count`
- [x] Job cleanup on conversation delete

**Frontend:**
- [x] Chunks cached to localStorage on each receive
- [x] Pointer tracked and updated on each chunk
- [x] Init checks stream status before loading history
- [x] resumeStream passes `from_pointer` correctly
- [x] Pointer cleared on stream complete
- [x] Resume renders cached chunks first, then continues streaming

---

## 12. Future Extension Points

### Adding RAG

1. Create `backend/rag/` domain
2. Add document loader in `rag/loader.py`
3. Add vector store in `rag/vectorstore.py`
4. Integrate into `chat/chain.py` as RAG chain
5. No changes to `chat/routes.py` — same interface

### Adding Authentication

1. Create `backend/auth/` domain
2. Add FastAPI dependency `get_current_user`
3. Apply via `Depends(get_current_user)` on routes

### PostgreSQL Migration

1. Replace `storage/file_storage.py` with `storage/db_storage.py`
2. Update `chat/service.py` to use new storage
3. Keep interface the same

---

## 13. File Inventory

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry, serves frontend + mounts routers |
| `backend/config.py` | Pydantic Settings from environment variables |
| `backend/chat/routes.py` | `/api/chat/*` endpoints, stream resume logic |
| `backend/chat/chain.py` | LangChain LCEL chain definition |
| `backend/chat/service.py` | ChatService (background generation, thinking extraction) |
| `backend/chat/stream_manager.py` | StreamJob + STREAM_REGISTRY |
| `backend/storage/file_storage.py` | JSON file operations |
| `frontend/index.html` | Chat UI with SSE streaming, thinking display, localStorage cache |
| `tests/conftest.py` | Pytest fixtures (mock LLM, temp storage) |
| `tests/test_chat_*.py` | Chat domain tests |
| `tests/test_storage.py` | Storage layer tests |
| `tests/test_stream_manager.py` | Stream registry tests |

### Modified Files (Recent Features)

| File | Changes |
|------|---------|
| `backend/chat/stream_manager.py` | Unified chunks list + chunk_queue (vs separate token/thinking) |
| `backend/chat/service.py` | Uses `append_chunk(chunk_type, text)` for both thinking and tokens |
| `backend/chat/routes.py` | Single `from_pointer` param, unified chunk events, `end: true` sentinel |
| `frontend/index.html` | Unified chunk handling, cached chunks rendering, streaming state per conversation |

---

## 14. Environment Configuration

```env
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1
ANTHROPIC_API_KEY=your-api-key-here
```

### Runtime Dependencies

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
langchain>=0.1.0
langchain-anthropic>=0.1.0
pydantic>=2.0
pydantic-settings>=2.0
python-multipart>=0.0.6
```
