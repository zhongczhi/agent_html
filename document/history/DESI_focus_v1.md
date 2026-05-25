# Chatbot Project - Detailed Design

## Decisions

### 1. Communication Protocol

- **SSE (Server-Sent Events)** for streaming responses
- FastAPI native `StreamingResponse` support
- Frontend uses Fetch API with `ReadableStream`

### 2. Authentication

- No authentication for initial version
- Easy to add API Key or JWT later via FastAPI dependencies

### 3. Conversation History Storage

- File-based JSON storage (`storage/conversations.json`)
- Per-conversation JSON files for future scalability
- Future migration path to PostgreSQL

### 4. LLM Integration

- **LangChain with LCEL** (LangChain Expression Language)
- Model: Anthropic Claude via MiniMax endpoint
- Environment variables:
  - `ANTHROPIC_BASE_URL`
  - `ANTHROPIC_API_KEY`

### 5. Frontend

- Plain HTML/JS (no framework)
- Zero build step
- Served by FastAPI as static file
- Streaming via native `ReadableStream` API

### 6. Backend Structure

- **Domain-driven modular structure**
- Separate domains: `chat`, `storage`
- Clean separation of routes, chains, services

### 7. Backend Server

- Uvicorn directly for development
- Single port serving both API and frontend

### 8. Configuration

- Environment variables via `pydantic-settings`
- `config.py` for typed settings with defaults
- `.env` file for secrets

### 9. Development Serve

- FastAPI serves frontend static files
- Single origin, no CORS issues
- Routes:
  - `/` → frontend `index.html`
  - `/api/chat/stream` → SSE streaming endpoint

### 10. Logging

- Python standard `logging` module
- Uvicorn default logging (INFO level)

## Key Patterns

### SSE Streaming

`POST /api/chat/stream` returns `StreamingResponse` with SSE format.

### LangChain LCEL

Chat logic in `chat/chain.py` using LangChain Expression Language.

### Domain Structure

Each domain has `routes.py`, `chain.py`, `service.py`.

### Conversation History

File-based JSON in `storage/conversations.json`.

## File Descriptions

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry point, serves frontend + API |
| `backend/config.py` | Pydantic settings from environment variables |
| `backend/chat/routes.py` | `/api/chat/stream` SSE streaming endpoint |
| `backend/chat/chain.py` | LangChain LCEL chain definition |
| `backend/chat/service.py` | Chat business logic |
| `backend/storage/file_storage.py` | JSON file read/write operations |
| `frontend/index.html` | Chat UI with streaming JavaScript |
| `tests/conftest.py` | Shared pytest fixtures (mock LLM, temp storage dir) |
| `tests/test_chat_service.py` | Unit tests for `ChatService` |
| `tests/test_chat_chain.py` | Unit tests for LangChain chain construction |
| `tests/test_storage.py` | Unit tests for file-based storage |

---

## Functional Requirements

### FR-1: Chat Streaming API

- **Endpoint:** `POST /api/chat/stream`
- **Behavior:** Accepts a JSON payload with `message` (string) and optional `conversation_id` (string), returns SSE stream of tokens
- **SSE Format:** `data: {"token": "..."}\n\n` for tokens, `data: {"token": null}\n\n` to signal end
- **Response Flow:**
  1. Parse request body for `message` and `conversation_id`
  2. Generate new `conversation_id` via UUID if not provided
  3. Retrieve existing conversation history or start fresh
  4. Append user message to history
  5. Stream LLM response tokens in real-time via SSE
  6. Save assistant response to conversation history on completion

### FR-2: Conversation History API

- **Endpoint:** `GET /api/chat/history/{conversation_id}`
- **Behavior:** Returns JSON with `conversation_id` and `messages` array
- **Response Format:**
  ```json
  {
    "conversation_id": "uuid-string",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
  ```
- **Edge Case:** Returns empty messages array for unknown `conversation_id`

### FR-3: Conversation History Storage

- **Storage Location:** `storage/conversations.json`
- **Data Structure:**
  ```json
  {
    "conversation_id": {
      "conversation_id": "uuid",
      "messages": [{"role": "user"|"assistant", "content": "..."}]
    }
  }
  ```
- **Operations:**
  - `get_conversation(conversation_id)` - Retrieve conversation or `None`
  - `save_conversation(conversation_id, messages)` - Persist entire messages array
  - `append_message(conversation_id, role, content)` - Append single message
- **Storage Directory:** Auto-created at `storage/` if missing
- **Error Handling:** Invalid JSON returns empty dict, logging warning

### FR-4: Frontend Chat UI

- **Technology:** Plain HTML/JS (no framework), single `index.html`
- **Features:**
  - Text input for messages with Enter key or button click submission
  - Display area for conversation history (user/assistant messages)
  - Loading indicator ("Thinking..." with animated dots) during streaming
  - Real-time token streaming as SSE responses arrive
  - Persistent `conversation_id` maintained across messages in session
- **Styling:** Centered chat container (max 800px), message bubbles (user right-aligned blue, assistant left-aligned gray), rounded input field
- **Error Handling:** Shows "Sorry, an error occurred" message on fetch failure

---

## Non-Functional Requirements

### NFR-1: Configuration

- **Method:** Environment variables via `pydantic-settings`
- **Variables:**
  | Variable | Default | Description |
  |----------|---------|-------------|
  | `ANTHROPIC_BASE_URL` | `https://api.minimax.chat/v1` | LLM API base URL |
  | `ANTHROPIC_API_KEY` | `""` | API key for authentication |
- **Settings File:** `.env` file support (gitignored)

### NFR-2: LLM Integration

- **Provider:** Anthropic Claude via MiniMax endpoint
- **Model:** `minimax-2.7-highspeed`
- **Max Tokens:** 4096
- **Message Format:** Convert `{role, content}` dicts to LangChain `HumanMessage` objects

### NFR-3: Backend Structure

- **Framework:** FastAPI with Uvicorn
- **Architecture:** Domain-driven modular structure
  ```
  backend/
  ├── main.py           # FastAPI app entry, serves frontend + mounts routers
  ├── config.py         # Pydantic Settings class
  ├── chat/
  │   ├── routes.py     # /api/chat endpoints (stream, history)
  │   ├── chain.py      # LangChain LCEL chain (messages → LLM)
  │   └── service.py    # ChatService (orchestrates chain + storage)
  └── storage/
      └── file_storage.py  # JSON file operations
  ```
- **Domain Pattern:** Each domain has `routes.py`, `chain.py`, `service.py`

### NFR-4: Logging

- **Module:** Python standard `logging`
- **Level:** INFO (via `logging.basicConfig`)
- **Error Cases:** `chat/service.py` logs LLM generation errors

### NFR-5: Frontend Serving

- **Root Route:** `GET /` serves `frontend/index.html`
- **Static Files:** Mounted at `/static` if `frontend/index.html` exists
- **Single Origin:** No CORS issues

---

## Testing Requirements

### TR-1: Unit Tests

- **Location:** `tests/` directory
- **Test Files:**
  - `test_chat_service.py` - Tests ChatService.generate() with mocked LLM
  - `test_chat_chain.py` - Tests LCEL chain construction and invocation
  - `test_storage.py` - Tests JSON read/write roundtrip
- **Fixtures:** `conftest.py` provides mock LLM, temp storage directory
- **Mocking:** Mock `ChatModel` from LangChain to avoid real API calls
- **Test Client:** `httpx.AsyncClient` against FastAPI TestClient

### TR-2: Test Dependencies

- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `httpx>=0.25.0`

## Testing Approach

Tests use **mocking** for LLM calls — no real API calls during tests.

**Key testing principles:**

- Mock `ChatModel` from LangChain to return controlled responses
- Use temporary directory for storage tests (cleanup after)
- Test streaming behavior via `AsyncIterator` mocking
- HTTP tests via `httpx.AsyncClient` against FastAPI TestClient

**Test coverage:**

- `test_chat_service.py`: Mock LLM, test `ChatService.generate()` returns expected response
- `test_chat_chain.py`: Test LCEL chain construction and invocation
- `test_storage.py`: Test JSON read/write roundtrip for conversation history

---

## Implementation Details for Future Extensions

### Adding RAG

1. Create `backend/rag/` domain
2. Add document loader in `rag/loader.py`
3. Add vector store in `rag/vectorstore.py`
4. Integrate into `chat/chain.py` as a LangChain RAG chain
5. No changes to `chat/routes.py` — same interface

### Adding File Upload

1. Create `backend/files/` domain
2. Add upload route in `files/routes.py`
3. Store files in `storage/uploads/`
4. Link files to conversations via `storage/file_storage.py`

### Adding Memory Persistence

1. Replace `storage/file_storage.py` with `storage/db_storage.py` using PostgreSQL
2. Update `chat/service.py` to use new storage
3. Add `chat/memory.py` for conversation memory buffer

### Adding Authentication

1. Add `backend/auth/` domain
2. Create FastAPI dependency `get_current_user`
3. Apply to routes via `Depends(get_current_user)`

## Verification Steps

1. **Start backend:**
   ```bash
   pip install -r requirements.txt
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Open frontend:**
   Navigate to `http://localhost:8000` in browser

3. **Test chat:**
   - Type a message, verify streaming response appears token by token
   - Check `storage/conversations.json` for saved history

4. **Verify streaming:**
   - Response should stream in real-time, not wait for full generation

5. **Run tests:**
   ```bash
   cd backend
   pip install pytest pytest-asyncio httpx
   pytest tests/ -v
   ```

---

## Out of Scope (Future)

- RAG (Retrieval-Augmented Generation)
- File upload
- Authentication / API keys
- PostgreSQL storage
- Conversation memory buffer
