# Multi-Conversation & Streaming Persistence - Detailed Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  • Manages conversation list UI                              │
│  • Stores currentConversationId in localStorage              │
│  • On load: checks stream status, resumes if needed          │
└─────────────────────┬───────────────────────────────────────┘
                      │ SSE / HTTP
┌─────────────────────▼───────────────────────────────────────┐
│                     Backend                                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              STREAM_REGISTRY (in-memory)            │    │
│  │  conversation_id → StreamJob                        │    │
│  │    - status: active | completed | failed            │    │
│  │    - tokens: List[str]                              │    │
│  │    - messages: List[dict]                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         conversations.json (persistent)              │    │
│  │  conversations: { id → { messages, created_at } }   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### StreamJob (in-memory)

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | `str` | Unique identifier |
| `status` | `Literal["active", "completed", "failed"]` | Current state |
| `tokens` | `List[str]` | All tokens generated so far |
| `messages` | `List[dict]` | Full history for resume |
| `error` | `Optional[str]` | Error message if failed |
| `created_at` | `datetime` | Job creation time |
| `updated_at` | `datetime` | Last update time |

### conversations.json (persistent)

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [{"role": "user"|"assistant", "content": "..."}],
      "created_at": "ISO",
      "updated_at": "ISO"
    }
  }
}
```

---

## API Endpoints

### `POST /api/chat/stream`

**Request:**
```json
{ "message": "Hello", "conversation_id": "uuid-1" }
```

**Behavior:**
1. Check `STREAM_REGISTRY` for active job with this `conversation_id`
2. If active job exists → **resume** from existing tokens
3. If no active job → **start new** stream

**Response:** SSE stream
```
data: {"partial": "already streamed..."}\n\n  (if resuming)
data: {"token": "..."}\n\n
...
data: {"token": null}\n\n
```

### `GET /api/chat/stream/status/{conversation_id}`

Returns stream status for a conversation.

```json
{
  "streaming": true,
  "status": "active",
  "tokens_count": 42,
  "is_complete": false,
  "partial_content": "Hello, I'm..."
}
```

### `GET /api/chat/conversations`

Returns all conversations sorted by `updated_at` descending.

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

### `DELETE /api/chat/conversation/{conversation_id}`

Deletes conversation and clears any active stream.

```json
{ "deleted": true }
```

### `GET /api/chat/history/{conversation_id}`

Returns conversation history.

```json
{
  "conversation_id": "uuid",
  "messages": [{"role": "user"|"assistant", "content": "..."}]
}
```

---

## Stream Flow

### New Stream Start

```
1. Client POST /api/chat/stream {message, conversation_id}
2. No active job in STREAM_REGISTRY
3. Create new StreamJob(status="active")
4. LLM generates tokens → job.append_token() for each
5. Tokens sent to client via SSE
6. On completion: job.mark_completed(), save to conversations.json
```

### Stream Resume (on switch-back or refresh)

```
1. Client POST /api/chat/stream {message, conversation_id}
2. Active job found in STREAM_REGISTRY
3. Send existing tokens via {"partial": "full_content"}
4. Continue streaming new tokens from current position
5. Completion flow same as above
```

### Stream Status Check

```
1. Client GET /api/chat/stream/status/{id}
2. Return job status, tokens_count, partial_content
3. Client decides: resume (if active) or load history (if completed)
```

---

## Frontend Logic

### Page Load

```
1. Read currentConversationId from localStorage
2. If exists, GET /api/chat/stream/status/{id}
3. If streaming=true: call POST /stream to resume
4. If streaming=false: load history via GET /api/chat/history/{id}
```

### Conversation Switch

```
1. User clicks conversation in sidebar
2. If currently streaming, SSE connection closes (disconnect)
3. Server continues streaming in background (STREAM_REGISTRY intact)
4. Load clicked conversation's history
5. If it was streaming, resume via POST /stream
```

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `backend/chat/stream_manager.py` | CREATE | StreamJob class, STREAM_REGISTRY, helper functions |
| `backend/chat/service.py` | MODIFY | Integrate StreamJob tracking during generate() |
| `backend/chat/routes.py` | MODIFY | Add /conversations, /stream/status/{id}, resume logic |
| `backend/storage/file_storage.py` | MODIFY | Add get_conversation_list(), delete_conversation() |
| `frontend/index.html` | MODIFY | Add sidebar, conversation list, switch/resume logic |

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Resume but stream already completed | Status shows `streaming: false`, frontend loads history |
| Resume but conversation deleted | Return 404, frontend starts fresh |
| Server restart during stream | Stream lost (acceptable for v1) |
| Page refresh with active stream | Frontend checks status, resumes from current position |
| Multiple tabs same conversation | First tab to resume gets stream; others get completed state |
