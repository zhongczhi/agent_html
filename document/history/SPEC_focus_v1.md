# Chatbot Project Specification

## Context

Building a modular, extensible chatbot application from scratch. Initial version: users type questions, chatbot generates answers via streaming. Future additions: RAG, file uploads, conversation memory persistence. Architecture priority: clear, modular, easy to add new features, scalable.

## Architecture

```
┌─────────────┐     SSE/HTTP      ┌─────────────────┐
│   Frontend  │ ◄───────────────► │   FastAPI       │
│  (Plain     │                   │   Backend       │
│   HTML/JS)  │                   │   (LangChain)   │
└─────────────┘                   └────────┬────────┘
                                           │
┌──────────────────────────────────────────┼──────────────────────────────────────────┐
│                                          │                                           │
│              ┌───────────────┐          │          ┌───────────────┐                │
│              │  chat domain  │          │          │ storage domain │               │
│              └───────────────┘          │          └───────────────┘                │
└─────────────────────────────────────────┘
```

- **Frontend:** Plain HTML/JS (no framework), served by FastAPI at `/`
- **Backend:** Python + FastAPI + LangChain
- **LLM:** Anthropic Claude via MiniMax endpoint
- **Storage:** JSON files (future: PostgreSQL)

## Project Structure

```
project/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, serves frontend + API
│   ├── config.py               # Typed settings from env vars
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── routes.py           # /api/chat/stream endpoint
│   │   ├── chain.py            # LangChain LCEL chain definition
│   │   └── service.py          # Chat business logic
│   └── storage/
│       ├── __init__.py
│       └── file_storage.py     # JSON file read/write
├── frontend/
│   └── index.html              # Chat UI with streaming JS
├── .env                        # Secrets (gitignored)
├── .env.example                # Template for .env
├── requirements.txt            # Python dependencies
├── tests/                      # Test module
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_chat_service.py    # Chat service unit tests
│   └── test_storage.py          # Storage layer tests
└── SPEC.md                     # This file
```

## API Endpoints

### `POST /api/chat/stream`

Stream chat responses as SSE.

**Request:**
```json
{
  "message": "Hello, who are you?",
  "conversation_id": "uuid-string"
}
```

**Response:** SSE stream
```
data: {"token": "Hello"}
data: {"token": "!"}
data: {"token": null}
```

### `GET /api/chat/history/{conversation_id}`

Retrieve conversation history.

**Response:**
```json
{
  "conversation_id": "uuid-string",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

## Environment Variables

```env
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1
ANTHROPIC_API_KEY=your-api-key-here
```

## Dependencies

```
# runtime
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
langchain>=0.1.0
langchain-anthropic>=0.1.0
pydantic>=2.0
pydantic-settings>=2.0
python-multipart>=0.0.6

# dev (tests)
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

## Future Extensibility

- **RAG:** Create `backend/rag/` domain, integrate into `chat/chain.py`
- **File upload:** Create `backend/files/` domain
- **Auth:** Create `backend/auth/` domain with FastAPI dependency
- **PostgreSQL:** Replace JSON storage with database
