# StreamJob Unified Chunks Refactor

## Overview

Refactor StreamJob to use a single unified queue with typed chunks instead of separate thinking and token queues.

## Problem

Current implementation uses two separate queues (`thinking_queue`, `token_queue`) with separate sent pointers, resulting in complex frontend logic to track two streams and handle `thinking_end` markers.

## Solution

Use a single `chunk_queue` with chunks tagged by type. Pointer tracking is done entirely by frontend via localStorage - backend only stores chunks.

```json
{"chunk": "text", "type": "thinking"}
{"chunk": "text", "type": "token"}
{"end": true}
```

---

## Backend Changes

### 1. `backend/chat/stream_manager.py` - StreamJob

**Remove:**
- `thinking_queue: asyncio.Queue`
- `thinking_tokens: List[str]`
- `thinking_sent_pointer: int`
- `_thinking_content: str`
- `sent_pointer: int`
- `append_token()`
- `append_thinking()`
- `get_full_thinking()`
- `stream_tokens()`
- `stream_thinking()`

**Add:**
- `chunks: List[dict]` - accumulated chunks with type
- `chunk_queue: asyncio.Queue` - unified queue

**Add method:**
- `append_chunk(chunk_type: str, text: str)` - adds to `chunks` list and `chunk_queue`

**Update methods:**
- `mark_completed()` - puts `None` in single queue
- `mark_failed()` - puts `None` in single queue

**Removed properties:**
- No `sent_pointer` on backend - frontend tracks pointer

### 2. `backend/chat/service.py` - ChatService.generate_background

**Before:**
```python
if block.get("type") == "thinking":
    job.append_thinking(thinking_text)
    job.thinking_queue.put_nowait(thinking_text)
elif block.get("type") == "text":
    job.append_token(token)
    job.token_queue.put_nowait(token)
```

**After:**
```python
if block.get("type") == "thinking":
    job.append_chunk("thinking", thinking_text)
elif block.get("type") == "text":
    job.append_chunk("token", token)
```

### 3. `backend/chat/routes.py` - stream_from_job

**Before:** Complex dual-queue handling with `thinking_sent_end`, `token_sent_end` flags

**After:** Simple single-queue loop. No pointer tracking on backend.

```python
async def stream_from_job(job, from_pointer: int = 0) -> AsyncGenerator[str, None]:
    """
    Read tokens from StreamJob and yield SSE events.
    from_pointer is provided by frontend to resume from a specific position.
    Backend does NOT track sent_pointer - it only stores chunks.
    """
    # Send accumulated chunks from from_pointer position
    if from_pointer < len(job.chunks):
        for c in job.chunks[from_pointer:]:
            yield f"data: {json.dumps({'chunk': c['chunk'], 'type': c['type']})}\n\n"

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

**API Change:** `/stream/{conversation_id}?from_pointer=N` (single param instead of `from_token` + `from_thinking`)

### 4. `backend/chat/routes.py` - StreamStatusResponse

**Before:**
```python
class StreamStatusResponse(BaseModel):
    thinking_count: int
    thinking_pointer: int
    partial_thinking: str | None
```

**After:**
```python
class StreamStatusResponse(BaseModel):
    chunks_count: int  # total chunks in job.chunks
    # pointer is tracked by frontend in localStorage
```

---

## Frontend Changes

### `frontend/index.html`

**Pointer tracking:**
- `pointer_{convId}` in localStorage stores the resume position (integer)
- Frontend reads this on page load / stream resume
- Frontend increments this as it receives and processes each chunk

**processStreamResponse:**

Remove:
- `thinkingContent` tracking variable
- `currentThinkingPointer` variable
- `data.partial_thinking` handling
- `data.thinking_end` handling
- `STORAGE_KEYS.THINKING` and `STORAGE_KEYS.POINTER.thinking_pointer`

Simplify to single-path:
```javascript
if (data.chunk) {
    if (data.type === "thinking") {
        thinkingElement.textContent += data.chunk;
        updateThinkingDisplay(messageDiv);
    } else if (data.type === "token") {
        contentDiv.textContent += data.chunk;
        contentDiv.innerHTML = marked.parse(contentDiv.textContent);
    }
    // Increment pointer and cache chunk
    currentPointer++;
    localStorage.setItem(STORAGE_KEYS.POINTER(currentConversationId), currentPointer.toString());
} else if (data.end) {
    // Clear cache, complete
}
```

**localStorage keys:**
- `chunks_{convId}` - array of `{chunk, type}` objects (optional cache, can also re-fetch from backend)
- `pointer_{convId}` - single integer pointer (frontend-only tracking)

---

## File Summary

| File | Changes |
|------|---------|
| `backend/chat/stream_manager.py` | Remove thinking queue, add unified chunk queue. No sent_pointer |
| `backend/chat/service.py` | Use `append_chunk` instead of separate append methods |
| `backend/chat/routes.py` | Simplify stream generator, no pointer tracking |
| `frontend/index.html` | Single-path chunk handling, pointer tracked in localStorage |

---

## Compatibility

- Old API format no longer supported after this change
- Frontend and backend must be deployed together
