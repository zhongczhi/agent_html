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

### SSE Format

```
data: {"token": "Hello"}\n\n
data: {"token": "!"}\n\n
data: {"token": null}\n\n
```

| Event | Format | Description |
|-------|--------|-------------|
| Token | `data: {"token": "..."}` | Single token |
| Partial | `data: {"partial": "..."}` | Already-streamed content on resume |
| End | `data: {"token": null}` | Stream complete |

### Request/Response Flow

**New Stream:**
1. Parse request body (`message`, `conversation_id`)
2. Generate UUID for new conversation if needed
3. Retrieve existing history or start fresh
4. Append user message to history
5. Stream LLM tokens via SSE
6. Save assistant response on completion

**Resume Stream:**
1. Check `STREAM_REGISTRY` for active job
2. Send `{"partial": "already streamed..."}` to catch up client
3. Continue streaming from current position

---

## 5. Data Storage

### conversations.json

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [
        {"role": "user"|"assistant", "content": "..."}
      ],
      "created_at": "ISO8601",
      "updated_at": "ISO8601"
    }
  }
}
```

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
- On page load: check stream status, resume if needed
- Handle SSE stream parsing and display

### Page Load Flow

```
1. Read currentConversationId from localStorage
2. If exists, GET /api/chat/stream/status/{id}
3. If streaming=true → POST /stream to resume
4. If streaming=false → GET /api/chat/history/{id}
```

### Conversation Switch Flow

```
1. User clicks conversation in sidebar
2. If currently streaming → SSE connection closes
3. Server continues streaming (STREAM_REGISTRY intact)
4. Load clicked conversation's history
5. If it was streaming → resume via POST /stream
```

### UI Styling

- Centered chat container (max 800px)
- User messages: right-aligned, blue bubble
- Assistant messages: left-aligned, gray bubble
- Loading indicator: "Thinking..." with animated dots

---

## 7. Testing Strategy

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

---

## 8. Future Extension Points

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

## 9. File Inventory

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry, serves frontend + mounts routers |
| `backend/config.py` | Pydantic Settings from environment variables |
| `backend/chat/routes.py` | `/api/chat/*` endpoints |
| `backend/chat/chain.py` | LangChain LCEL chain definition |
| `backend/chat/service.py` | ChatService (orchestrates chain + storage) |
| `backend/chat/stream_manager.py` | StreamJob + STREAM_REGISTRY |
| `backend/storage/file_storage.py` | JSON file operations |
| `frontend/index.html` | Chat UI with SSE streaming JS |
| `tests/conftest.py` | Pytest fixtures (mock LLM, temp storage) |
| `tests/test_chat_*.py` | Chat domain tests |
| `tests/test_storage.py` | Storage layer tests |
| `tests/test_stream_manager.py` | Stream registry tests |

---

## 10. Environment Configuration

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
