# Chatbot Project - Requirements Specification

## Overview

A modular chatbot application that provides streaming AI responses with conversation history persistence. Users type questions and receive real-time streaming answers from an LLM. The system supports multiple conversations, persists history across sessions, and displays LLM thinking content with stream resume capability.

**Core Goal:** Deliver a chat interface with real-time streaming responses, persistent conversation history, and LLM thinking content display.

---

## Functional Requirements

### FR-1: Chat Streaming

| ID | Requirement |
|----|-------------|
| FR-1.1 | Users can send a text message to the chatbot |
| FR-1.2 | The chatbot streams AI response tokens in real-time as they are generated |
| FR-1.3 | Streaming completes with an end signal (`end: true`) |
| FR-1.4 | Users can initiate a new conversation when no `conversation_id` is provided |
| FR-1.5 | Thinking content streams before text tokens |
| FR-1.6 | Streaming supports resume from a given position via `from_pointer` |

### FR-2: Conversation History

| ID | Requirement |
|----|-------------|
| FR-2.1 | Each conversation is identified by a unique `conversation_id` (UUID) |
| FR-2.2 | User and assistant messages are stored with role and content |
| FR-2.3 | Users can retrieve the full message history of any conversation |
| FR-2.4 | Unknown conversation IDs return an empty messages array |
| FR-2.5 | Assistant messages may include a `thinking` field with internal reasoning |

### FR-3: Multi-Conversation Support

| ID | Requirement |
|----|-------------|
| FR-3.1 | Users can create new conversations at any time |
| FR-3.2 | Users can switch between existing conversations |
| FR-3.3 | Conversation list shows title (first message preview) and last updated time |

### FR-4: Frontend Chat Interface

| ID | Requirement |
|----|-------------|
| FR-4.1 | Display a text input for composing messages |
| FR-4.2 | Submit messages via Enter key or button click |
| FR-4.3 | Display conversation history with user/assistant message distinction |
| FR-4.4 | Show loading indicator ("Thinking...") during streaming |
| FR-4.5 | Display error message on fetch failure |
| FR-4.6 | Maintain `conversation_id` across messages in a session |
| FR-4.7 | Display thinking content in collapsible section (Show more/less when >3 lines) |
| FR-4.8 | Message blocks scroll internally with max-height 400px |
| FR-4.9 | Empty state: input centered vertically and horizontally |
| FR-4.10 | Input box auto-expands (min 5 lines, max ~50% viewport) |
| FR-4.11 | Markdown rendering for assistant text responses |

### FR-5: Stream Resume

| ID | Requirement |
|----|-------------|
| FR-5.1 | Chunks cached to localStorage during streaming |
| FR-5.2 | Pointer tracks current position for resume |
| FR-5.3 | Page refresh resumes from cached position |
| FR-5.4 | Backend continues LLM generation when frontend disconnects |

---

## Non-Functional Requirements

### NFR-1: Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_BASE_URL` | `https://api.minimax.chat/v1` | LLM API base URL |
| `ANTHROPIC_API_KEY` | `""` | API key for authentication |

### NFR-2: Performance

- Streaming latency: tokens delivered as generated, not batched
- Model: `minimax-2.7-highspeed`
- Max tokens per response: 4096

### NFR-3: Reliability

- Invalid JSON in storage file returns empty dict with warning log
- LLM generation errors are logged
- Storage directory auto-created if missing
- StreamJob continues in background when EventSource disconnects

### NFR-4: Security

- No authentication for initial version
- API key stored in environment variables, not in code

---

## Interface Requirements

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/stream` | Send message, receive streaming response |
| `GET` | `/api/chat/history/{conversation_id}` | Get conversation history |
| `GET` | `/api/chat/conversations` | List all conversations |
| `DELETE` | `/api/chat/conversation/{conversation_id}` | Delete a conversation |
| `GET` | `/api/chat/stream/status/{conversation_id}` | Check stream status |
| `GET` | `/api/chat/stream/{conversation_id}` | Resume stream from pointer |

### POST /api/chat/stream

**Request:**
```json
{
  "message": "Hello, who are you?",
  "conversation_id": "uuid-string"
}
```

**Response:** SSE stream
```
data: {"chunk": "thinking content...", "type": "thinking"}
data: {"chunk": "Hello", "type": "token"}
data: {"chunk": "!", "type": "token"}
data: {"end": true}
```

**Behavior:**
- No existing StreamJob → create new + start background task
- `status == "active"` → return existing stream
- `status != "active"` → start new stream

### GET /api/chat/stream/{conversation_id}

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `from_pointer` | `int` | `0` | Start position for chunks |

**Response:** SSE stream resuming from `from_pointer`

### GET /api/chat/history/{conversation_id}

**Response:**
```json
{
  "conversation_id": "uuid-string",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!", "thinking": "Internal reasoning..."}
  ]
}
```

### GET /api/chat/conversations

**Response:**
```json
{
  "conversations": [
    {
      "conversation_id": "uuid",
      "title": "First 50 chars of first message...",
      "message_count": 5,
      "updated_at": "ISO"
    }
  ]
}
```

### GET /api/chat/stream/status/{conversation_id}

**Response:**
```json
{
  "streaming": true,
  "status": "active",
  "chunks_count": 70,
  "is_complete": false,
  "partial_content": "Hello, I'm..."
}
```

### DELETE /api/chat/conversation/{conversation_id}

**Response:**
```json
{ "deleted": true }
```

---

## Data Requirements

### Conversation Storage Schema

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [
        {"role": "user"|"assistant", "content": "...", "thinking": "..."}
      ],
      "created_at": "ISO",
      "updated_at": "ISO"
    }
  }
}
```

Note: `thinking` field is optional and only present on assistant messages when the LLM provided internal reasoning.

### Stream Status Schema

```json
{
  "streaming": "boolean",
  "status": "active|completed|failed",
  "chunks_count": "integer",
  "is_complete": "boolean",
  "partial_content": "string",
  "error": "string|null"
}
```

### StreamJob Schema (Backend)

```json
{
  "chunks": [{"chunk": "string", "type": "thinking|token"}],
  "chunk_queue": "asyncio.Queue",
  "messages": [{"role": "string", "content": "string"}],
  "status": "pending|active|completed|failed",
  "error": "string|null"
}
```

### localStorage Schema (Frontend)

| Key | Content |
|-----|---------|
| `chunks_{conv_id}` | JSON array of chunks `[{"chunk": str, "type": "thinking|token"}]` |
| `pointer_{conv_id}` | Integer position for resume |
| `streaming_{conv_id}` | Boolean flag for active streaming |

---

## Frontend Requirements

### Thinking Display

- Location: Inside assistant message block, above the text response
- ≤3 lines: show fully, no toggle button
- >3 lines: show first 3 lines with "Show more" → expand with "Show less"

### Message Block Scrolling

- max-height: 400px with overflow-y auto
- Custom scrollbar auto-hides (hidden by default, shown on wheel, hides after 3s)

### Empty State

- No messages: input vertically and horizontally centered
- First message sent: input moves to bottom
- Smooth CSS flexbox transition

### Input Box

- Minimum: 5 lines, Maximum: ~50% viewport
- CSS `field-sizing: Content` with JS fallback

### Markdown Rendering

- Use marked.js via CDN for text responses only
- Thinking content: plain text only

---

## Dependencies

### Frontend

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

---

## Out of Scope (Future)

- RAG (Retrieval-Augmented Generation)
- File upload
- Authentication / API key management
- PostgreSQL storage (current: JSON file)
- Conversation memory buffer
- Cross-browser tab synchronization
- Conversation search/filter
- Stream backpressure handling
