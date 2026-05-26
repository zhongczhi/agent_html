# Frontend Enhancement + Thinking Content + Stream Resume - Specification

## Overview

Two major feature sets:

1. **Thinking Content Streaming & Display** — Backend extraction and frontend display of LLM thinking content, message block scrolling, empty state UX, input auto-expansion, and markdown rendering
2. **Stream Resume Refactor** — LLM calls run in background tasks, tokens stored in `StreamJob`, frontend caches to localStorage with position tracking for resume

---

## Part I: Thinking Content Streaming & Display

### 1. Backend: Thinking Content Streaming

#### 1.1 LLM Response Structure

The MiniMax LLM returns content blocks of two types:

- `type: "thinking"` — internal reasoning, content in `thinking` field
- `type: "text"` — final response, content in `text` field

#### 1.2 SSE Protocol (Extended)

**Stream sequence:** thinking chunks → token chunks → end marker

**Event types:**

| Event | Format | Description |
|-------|--------|-------------|
| `chunk` | `data: {"chunk": "...", "type": "thinking"|"token"}` | Single token (thinking or text) |
| `end` | `data: {"end": true}` | Stream complete |

#### 1.3 Storage Schema Change

Messages with thinking support new optional field:

```json
{"role": "assistant", "content": "Full response text...", "thinking": "Internal reasoning..."}
```

#### 1.4 Files to Modify

| File | Change |
|------|--------|
| `backend/chat/chain.py` | Extract thinking blocks, yield thinking tokens separately before text tokens |
| `backend/chat/service.py` | Pass through thinking tokens in async generator |
| `backend/chat/routes.py` | Read from queue, yield chunks as SSE |

#### 1.5 API Changes

| Endpoint | Change |
|----------|--------|
| `POST /api/chat/stream` | Emit thinking chunks, then text chunks, then end marker |
| `GET /api/chat/history/{id}` | Returns messages with optional `thinking` field |
| `GET /api/chat/stream/status/{id}` | Returns `chunks_count` and `partial_content` |
| `GET /api/chat/stream/{id}` | Accepts `from_pointer` param for resume |

### 2. Frontend: Thinking Display

**Location:** Inside assistant message block, above the text response

**Fold/Unfold Behavior:**

- ≤3 lines: show fully, hide toggle button
- >3 lines: show first 3 lines with "Show more" → expand fully with "Show less"

### 3. Message Block Internal Scrolling

- max-height: 400px with overflow-y auto
- Custom scrollbar auto-hides (hidden by default, shown on wheel, hides after 3s)

### 4. Empty State: Centered Input

- No messages: input vertically and horizontally centered
- First message sent: input moves to bottom
- Smooth CSS flexbox transition

### 5. Input Box Auto-Expand

- Minimum: 5 lines, Maximum: ~50% viewport
- CSS `field-sizing: Content` with JS fallback

### 6. Markdown Rendering

- Use marked.js via CDN for text responses only
- Thinking content: plain text only

---

## Part II: Stream Resume Refactor

### 7. StreamJob Enhancement

| Field | Type | Purpose |
|-------|------|---------|
| `chunks` | `List[dict]` | Accumulated chunks `[{"chunk": str, "type": "thinking\|token"}]` |
| `chunk_queue` | `asyncio.Queue` | Queue for async chunk iteration |
| `messages` | `List[dict]` | Conversation history |
| `status` | `Literal["pending", "active", "completed", "failed"]` | Job status |
| `error` | `str \| None` | Error message if failed |

**Task Separation:**

- **Background Task**: Calls LLM, stores chunks in StreamJob, puts chunks in queue
- **`/stream` Endpoint**: Reads from queue, yields SSE, supports `from_pointer`

### 8. Frontend localStorage

| Key | Content |
|-----|---------|
| `chunks_{conv_id}` | JSON array of chunks `[{"chunk": str, "type": "thinking\|token"}]` |
| `pointer_{conv_id}` | Integer position for resume |
| `streaming_{conv_id}` | Boolean flag for active streaming |

**Init Flow:**

1. Check localStorage for cached conversation
2. If cached + streaming: display cached, call `/stream/{conv_id}?from_pointer=pointer`
3. If cached + complete: call `/history`
4. If no cache: call `/history` normally

### 9. Four Scenarios

| Scenario | Behavior |
|----------|----------|
| **Continuous Chatting** | User sends → backend creates StreamJob + background task → SSE to frontend → localStorage cache |
| **Switch Away** | Frontend closes EventSource → StreamJob stays active → background task continues |
| **Refresh While Streaming** | Page reload → check status → read localStorage → call `/stream/{id}?from_pointer=N` |
| **Resume After Disconnect** | Same as refresh, or if stream completed → call `/history` instead |

### 10. API Changes

#### POST /api/chat/stream

```json
{"message": "Hello", "conversation_id": "uuid"}
```

- No StreamJob → create new + start background task
- `status == "active"` → return existing stream
- `status != "active"` → start new stream

#### GET /api/chat/stream/{conversation_id}

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `from_pointer` | `int` | `0` | Start position for chunks (unified for both thinking and token) |

#### GET /api/chat/stream/status/{conversation_id}

```json
{
  "streaming": true,
  "status": "active",
  "chunks_count": 70,
  "is_complete": false,
  "partial_content": "..."
}
```

#### GET /api/chat/history/{conversation_id}

Returns full message history with `thinking` field.

### 11. Edge Cases

| Scenario | Handling |
|----------|----------|
| localStorage missing but StreamJob exists | Call `/stream/{id}?from_pointer=0` |
| localStorage exists but StreamJob is gone | Call `/history` |
| StreamJob completed before resume | Call `/history` instead of `/stream` |
| Multiple concurrent /stream calls | Return same stream (job already active) |
| Very long streams | localStorage updates per chunk (no debouncing) |

### 12. Summary of Changes

#### Backend Files

| File | Change |
|------|--------|
| `backend/chat/chain.py` | Extract thinking blocks from LLM output, yield separately |
| `backend/chat/service.py` | Pass through thinking tokens; Background task, queue writing |
| `backend/chat/routes.py` | Read from queue, from_pointer param, trigger background task |
| `backend/chat/stream_manager.py` | Contains StreamJob class with chunks/chunk_queue, registry management |

#### Frontend Files

| File | Change |
|------|--------|
| `frontend/index.html` | Thinking display, CSS + JS, localStorage caching, pointer tracking, init flow |

#### No Changes

- `backend/storage/file_storage.py` — JSON schema supports new fields
- `backend/config.py` — No config changes
- `backend/main.py` — No changes

### 13. Dependencies

#### Frontend

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

### 14. Acceptance Criteria

#### Thinking Content & Display

- [x] Thinking chunks stream before text chunks in SSE
- [x] Thinking displayed in collapsible section (Show more/less when >3 lines)
- [x] Message blocks scroll internally, not expand the page
- [x] Scrollbar auto-hides after 3s on message blocks
- [x] Empty state has centered input
- [x] Input box expands with content (min 5 lines)
- [x] Markdown rendering works in assistant messages
- [x] History API returns thinking content
- [x] Resume works with partial thinking

#### Stream Resume

- [x] Continuous chatting works as before
- [x] Switch away preserves stream (backend continues)
- [x] Refresh while streaming resumes from cached position
- [x] Resume after switch shows partial content first, then new chunks
- [x] Chunks are persisted to localStorage on each receive
- [x] Pointer tracks position correctly for resume
- [x] Thinking chunks follow same caching/pointer pattern
- [x] History API returns complete messages with thinking
- [x] StreamJob cleanup on conversation delete