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
- User-facing environment variables: `ANTHROPIC_BASE_URL` (consumed by `pydantic-settings`) and `ANTHROPIC_API_KEY`. Internally `chain.py` also sets the legacy `ANTHROPIC_API_BASE` env var because `langchain-anthropic` reads that name.

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

### 1.10 Two-Cache Frontend Architecture

**Choice:** Maintain two separate localStorage caches — `history_{convId}` for full message lists and `chunks_{convId}` for in-flight streaming chunks.

**Rationale:** History cache stores the "done" state (complete messages, updated on stream end via append); chunks cache stores the "in-flight" state (individual tokens, updated per chunk). Different update patterns and different consumers (`loadConversation` vs. `processStreamResponse`). Separation makes invalidation explicit and avoids mixing concerns. The history cache is loaded cache-first by `loadConversation`, with a backend fetch on miss and a stale-chunks-cache clear on success.

### 1.11 New Conversation Sidebar Visibility

**Choice:** When a brand-new conversation is created, the backend appends the user message to storage synchronously in `stream_chat` (before the background LLM task starts).

**Rationale:** Without this, the new conversation only appears in the sidebar after the LLM finishes streaming (5-30+ seconds later) and lacks a user-message-derived title. The synchronous append also relies on `get_or_create_job` calling `file_storage.create_conversation(conversation_id)` to ensure an empty entry exists. The background task has an idempotency check to avoid duplicating the user message.

### 1.12 SSE Event Boundary Handling

**Choice:** The frontend SSE parser accumulates chunks in an `sseBuffer` variable, splits on `\n\n` per SSE spec, and retains the trailing incomplete event in the buffer for the next chunk.

**Rationale:** A single network read can split events across chunks. Without buffering, JSON payloads straddling a boundary would produce parse errors or dropped data. Buffering by the actual `\n\n` boundary (not chunk boundary) correctly handles any chunk size.

### 1.13 Streaming Markdown Parser State Preservation

**Choice:** The `streaming-markdown` parser/renderer is hoisted to function scope in `processStreamResponse` and created once per stream.

**Rationale:** The library is stateful — `parser_write` accumulates tokens so multi-line constructs (tables, fenced code blocks, math blocks, lists, blockquotes) render correctly when their tokens arrive in multiple chunks. Hoisting prevents the parser from being recreated on every chunk batch. `parser_end` is called exactly once on stream completion (in the `data.end` handler). On resume, the parser created during `renderCachedChunks` is reused by `processStreamResponse` so multi-line constructs spanning the cache-replay → live-stream boundary render correctly.

### 1.14 Stream Resume Error Handling

**Choice:** In `resumeStreamFromPosition`'s catch block, drop any error-type branching. Log the error and return false; do not touch the streaming flag or badge.

**Rationale:** The streaming flag has exactly two responsibilities — `init` / `checkStreamStatus` decides whether to attempt a resume on page load, and `loadConversationList` derives the sidebar "Streaming" badge. For (1), a transient fetch failure must leave the flag set so the next refresh can retry. The flag should only be cleared on `data.end` or on explicit user action. A fetch killed by browser navigation (refresh, close tab, back button) is rejected with `TypeError: network error` rather than `AbortError`; the two cannot be reliably distinguished, so both are treated as transient.

### 1.15 init Fallback Gate

**Choice:** In `init`, only fall through to `loadConversation` when the streaming flag was unset at the start of the decision.

**Rationale:** `checkStreamStatus` calls `resumeStreamFromPosition`, which always renders cached chunks into the DOM before issuing the network fetch. Whatever happens afterwards, the partial content stays in the DOM. Calling `loadConversation` after a resume attempt would `messagesContainer.innerHTML = ''` and re-render only what is in the history cache (missing the in-progress assistant message), wiping the partial content.

### 1.16 Custom Confirmation Modal

**Choice:** A single reusable modal element appended to `<body>` with a single `showConfirmModal({title, message, confirmText, cancelText, danger})` helper that returns a Promise resolving to `true`/`false`.

**Rationale:** Browser-native `confirm()` has a generic OS look that does not match the page's dark-theme + cyan/purple-accent styling. The helper uses the same theme tokens as the rest of the page (`var(--bg-secondary)`, `var(--border-color)`, etc.) and is invoked by both single-item and batch deletion.

### 1.17 Batch Delete Selection Mode

**Choice:** Module-level state (`selectionMode: boolean`, `selectedConvIds: Set`) toggled via `enterSelectionMode()` / `exitSelectionMode()`. The sidebar header has two layouts (normal vs selection), switched in a single `renderSidebarHeader()` function that reads `selectionMode`. `+ New Chat` is hidden in selection mode; the per-item `×` is also hidden so the user uses the batch Delete button exclusively.

**Rationale:** A `Set` makes add/remove/lookup O(1) and avoids duplicate selections. Two layout branches in one renderer keeps the DOM state in sync without separate header functions. Hiding `+ New Chat` and per-item `×` in selection mode prevents the user from inadvertently starting a new chat or single-deleting an item while in "I'm about to delete these" mode.

### 1.18 StreamJob Cancellation Flag

**Choice:** `StreamJob` carries a `cancelled: bool` flag. `clear_job` sets the flag before removing the job from `STREAM_REGISTRY`. `generate_background` checks the flag mid-loop and immediately before `save_conversation`, bailing out without `mark_completed` or save if set.

**Rationale:** `generate_background` is a fire-and-forget `asyncio.create_task`. Deleting the conversation only removes the `StreamJob` and the entry from `conversations.json` — it does not stop the background task. When the task finishes, `file_storage.save_conversation` is create-or-update and silently re-creates the deleted entry (the "resurrection" bug). A `cancelled` flag stops the background task at its next natural checkpoint without taking the heavier step of cancelling the asyncio task itself (which would abort the LLM read but waste server-side LLM work).

### 1.19 Smart Auto-Scroll Pin State

**Choice:** During streaming, capture the user's pinned-to-bottom state BEFORE each chunk's DOM update, and only force `scrollTop = scrollHeight` when the captured state is true.

**Rationale:** A naive post-update check breaks because a single chunk can add more than 50px of height, causing the post-update check to incorrectly report "not pinned" even though the user never scrolled. Capturing pre-update reflects the user's true intent at the moment the chunk arrived.

### 1.20 Selection-Mode Send Guard

**Choice:** An early-return guard at the top of `sendMessage()` checks `selectionMode` and returns immediately if true. No UI changes — the textarea and send button remain enabled-looking; the user is expected to exit selection mode via Cancel if they want to send.

**Rationale:** Both the Send button click handler and the `messageInput` keydown (Enter) handler route through `sendMessage()`. Guarding `sendMessage()` blocks both paths with one line. Visual disablement is a separate UX choice and is intentionally omitted to keep the change minimal.

### 1.21 Backend Stream Resume Boundary Case

**Choice:** The `stream_from_job` `from_pointer` guard treats `from_pointer == len(chunks)` as a valid boundary (all current chunks already sent), not as an out-of-range error. Only negative or strictly-greater-than pointers are rejected.

**Rationale:** The frontend's pointer always lands at exactly `len(job.chunks)` during active streaming (because the user-entered message is only in `chunksCache`, not yet committed to history). Treating this as an error would cause the resume to return immediately and drop the live stream.

### 1.22 LLM Model Configuration

**Choice:** The backend LLM is configured as `minimax-3` with `max_tokens=16000` and `thinking={"type": "enabled", "budget_tokens": 10000}`.

**Rationale:** The larger output budget (4× the previous 4096) supports long code blocks and multi-paragraph answers. Explicit extended thinking with a 10k budget reserves reasoning capacity; visible answer has ~6k tokens. Frontend already supports `thinking` blocks via the unified chunk format.

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
2. Generate UUID for new conversation if `conversation_id` is None
3. Call `get_or_create_job(conversation_id, [])` — creates a `StreamJob` and ensures an empty conversation entry exists via `file_storage.create_conversation(conversation_id)`
4. If the conversation is new (`existing_conv is None`), append the user message to storage synchronously in `stream_chat` before the background task starts (so the conversation appears in the sidebar with the correct title while the LLM is still generating)
5. Start background task for LLM streaming
6. Stream thinking chunks first, then token chunks via SSE
7. Save assistant response (with thinking) on completion (with idempotency check to skip duplicate user message append)

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
| `create_conversation(id)` | Create empty conversation entry; called by `get_or_create_job` so a brand-new conversation appears in the list before the first message arrives |
| `get_conversation_list()` | Return sorted list (updated_at desc) |
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

The frontend maintains two separate caches per conversation: a **chunks cache** for in-flight streaming state and a **history cache** for the full message list (see Section 1.10).

| Key | Purpose |
|-----|---------|
| `chunks_{conv_id}` | Cached chunks for resume |
| `pointer_{conv_id}` | Current position in stream |
| `streaming_{conv_id}` | Active streaming state per conversation |
| `history_{conv_id}` | Full message list for fast load on conversation switch/refresh |

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

- Centered chat container (max 1100px)
- User messages: right-aligned, blue bubble
- Assistant messages: left-aligned, gray bubble
- Thinking section: displayed above response, collapsible
- Loading indicator: "Thinking" with animated dots

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
    <div class="message-content"></div>
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
| `.message-content` | Rendered markdown content (assistant body) |

### JavaScript Functions

| Function | Responsibility |
|----------|---------------|
| `processStreamResponse()` | Parse SSE via `sseBuffer` accumulator; handle thinking + token events; reuse parser/renderer across chunks; accept `existingRenderer` / `existingParser` and return `{renderer, parser}` for parser handoff on resume |
| `renderContent()` | Lazy-create parser/renderer; write streaming-markdown tokens; apply LaTeX |
| `addMessage()` | Create message element with proper structure |
| `updateThinkingDisplay()` | Handle thinking content and fold/unfold |
| `setupScrollbarAutoHide()` | Attach wheel listener to message blocks |
| `autoResizeInput()` | Expand textarea with content |
| `resumeStreamFromPosition()` | Resume stream from localStorage cache; thread parser/renderer to `processStreamResponse` for handoff |
| `loadConversation()` | Cache-first load: read `history_{convId}`; fetch from backend on miss and store; clear stale `chunks_{convId}` on success |
| `sendMessage()` | Append user message to UI, then to history cache; trigger stream and `loadConversationList()` |
| `deleteConversation()` | Remove conversation + all related caches (history + chunks + pointer + streaming) |
| `loadConversationList()` | Derive streaming badge from `isStreamingForConv()` on every render (not just once at send time) |
| `startNewChat()` | Abort in-flight stream via `currentAbortController`; clear current conversation; reset input + send button; refocus input |
| `getHistoryCache()` / `setHistoryCache()` / `appendToHistoryCache()` / `clearHistoryCache()` / `clearChunkCache()` | History cache helpers |
| `renderMessagesFromCache()` | Re-render message list from cached or fetched messages array |
| `isStreamingForConv()` | Check whether the streaming flag is set in localStorage for a given conversation |

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
        self.chunks: List[dict] = []  # [{"chunk": "text", "type": "thinking|token", "message_id": str}] — message_id is for debug tracing; frontend does not depend on it
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

### 8.9 Two-Cache Architecture

The frontend maintains separate history and chunks caches with different update patterns and consumers. Edge cases:

- **Empty cache on first load**: `getHistoryCache()` returns null; `loadConversation` falls through to backend and populates the cache.
- **Cache exists but conversation was deleted on another tab**: Local cache remains; backend returns empty array on next load. Cross-tab invalidation is out of scope.
- **Resume streaming after refresh**: `checkStreamStatus` → `resumeStreamFromPosition` continues to use `chunks_{convId}` and `pointer_{convId}`. After resume, the new assistant message is appended to `history_{convId}`.
- **Switching conversations mid-stream**: Only the streaming conversation's chunks/pointer are updated. Other conversations' history cache is untouched. On switch back, `loadConversation` reads from cache.
- **Stale chunks cleared on history load**: `loadConversation` calls `clearChunkCache(convId)` after a successful backend fetch so a stale in-flight chunk cache cannot replay on top of fresh history.

### 8.10 SSE Event Boundary Handling

The frontend SSE parser uses an `sseBuffer` accumulator to handle events that straddle chunk boundaries.

- **Chunk ends mid-event**: Incomplete event kept in buffer, combined with next chunk.
- **Multiple events in one chunk**: All complete events processed; only the trailing partial is buffered.
- **Stream ends mid-event**: Partial is silently dropped (per SSE spec — incomplete events are dropped).
- **Empty `\n\n` at end of chunk**: Produces empty string in events list; ignored by the `event.startsWith('data: ')` guard.
- **Conversation switch mid-stream**: `convId !== currentConversationId` returns early, dropping partial work cleanly.

### 8.11 Parser State Across Chunks and Resume

The streaming markdown parser is hoisted to function scope so it persists across chunks within a single stream, and survives the cache-replay → live-stream boundary on resume.

- **Markdown table split across chunks**: Parser accumulates all rows; table renders correctly when complete.
- **Fenced code block split across chunks**: Parser accumulates opening fence, content, closing fence.
- **Multi-line math block split across chunks**: `applyLaTeX` finds `<equation-block>` tags after parser finalizes.
- **Conversation switch during stream**: `processStreamResponse` returns early; parser is discarded with the message element. On resume, a new parser is created (or the one from cache replay is reused).
- **Cache replay from localStorage**: `renderCachedChunks` reuses the same `renderContent` path; the parser is created once and reused for all cached chunks, then handed off to `processStreamResponse`.
- **Empty content chunk**: `renderContent` early-returns; parser state untouched.
- **No thinking block**: `smd.parser_write` is still called on every token chunk; render path is unchanged.

### 8.12 New Chat UX

- **No in-flight stream**: `currentAbortController` is null; abort block is a no-op.
- **Stream in-flight when new chat clicked**: Stream aborts; new chat is immediate; previous stream's `data.end` handler is a no-op because `currentConversationId` has changed.
- **Sidebar re-render mid-stream**: Streaming badge derived from `isStreamingForConv()`; appears correctly on the active conversation item.
- **Sidebar re-render post-stream**: `isStreamingForConv` returns false; no badge.
- **Rapid double-click on New Chat**: First call aborts; second call sees null controller; both safely no-op.

### 8.13 Backend Stream Resume Boundary Case

- **`from_pointer == 0`, no chunks yet**: Valid; queue loop runs until chunks arrive or end marker.
- **`from_pointer == len(chunks)`, stream still active**: Valid boundary (previously bug); slice is empty, queue loop streams new chunks.
- **`from_pointer == len(chunks)`, stream completed**: Valid; slice is empty, queue loop terminates immediately on `status != "active"`.
- **`from_pointer > len(chunks)`**: Reject (returns nothing).
- **`from_pointer < 0`**: Reject (defensive — frontends should never send negative).

### 8.14 Repeated Refresh During Streaming

- **First refresh during streaming**: Streaming flag is `'true'` → `checkStreamStatus` → `resumeStreamFromPosition` renders cached chunks + processes live stream. DOM intact.
- **Second (or Nth) refresh during streaming**: Same as first. Streaming flag survives (Part 14 / 1.14). `init` doesn't fall back (Part 15 / 1.15). DOM shows cached chunks + live tail.
- **Stream completes naturally between refreshes**: `data.end` handler clears flag and chunks cache. Next refresh sees flag `'false'` → falls back to `loadConversation` → renders full history.
- **Refresh after stream completes**: Flag `'false'` → `loadConversation` → history cache.
- **New conversation in a fresh tab**: Flag `'true'` immediately after send → resume path. If stream is fast and already done, the resume gets the end marker on first read; flag cleared; UI correct.
- **User switches conversation mid-stream**: `switchConversation` aborts `currentAbortController` → fetch throws `AbortError` → catch block logs and returns false → next refresh of the original conversation can resume from chunk cache. Streaming flag preserved.
- **User clicks "New Chat" mid-stream**: `startNewChat` aborts controller + clears `currentConversationId`. No resume possible for the previous conversation from this tab.
- **Genuine stream failure (backend 404, etc.)**: Fetch rejects with non-Abort error. Streaming flag preserved → next refresh retries.

### 8.15 Confirmation Modal

- **User presses Enter while modal is open**: Confirm button gets a click. For `danger: true`, focus is on cancel so Enter cancels instead.
- **User double-clicks confirm**: First click resolves the Promise; modal hides immediately. Second click is a no-op.
- **Page refresh while modal is open**: Modal element is gone with the page. No Promise resolution (caller never sees the result).
- **Multiple concurrent confirmations**: Not supported. Caller awaits the Promise before issuing another.

### 8.16 Batch Delete Selection Mode

- **Enter selection mode with 0 conversations**: Selection mode renders, but list is empty; Delete button disabled.
- **Enter selection mode with 1 conversation**: User can select and batch-delete that one item (effectively a single delete).
- **Delete the active conversation**: After deletion, `startNewChat()` clears current conversation and resets input.
- **Delete all conversations**: Sidebar becomes empty; selection mode exits.
- **Switch conversation while in selection mode**: The row-click handler routes to selection toggle (not switch) — switching is disabled in selection mode.
- **Refresh page while in selection mode**: `selectionMode` is module-level, not persisted. On reload, normal state is restored.
- **Backend DELETE fails for one item**: Other deletions proceed; failed item is logged but UI does not block.

### 8.17 Streaming-Conversation Resurrection

- **User deletes right as LLM finishes**: The pre-save `if job.cancelled: return` catches it.
- **User deletes before any chunks are received**: The mid-loop check triggers on the very first chunk iteration.
- **`clear_job` is called multiple times**: Idempotent — flag is set once; subsequent calls are no-ops on the registry.
- **Normal stream completion (no delete)**: `cancelled` stays `False`; the existing `mark_completed` + `save_conversation` path runs unchanged.
- **LLM errors mid-stream (`generate_background` except branch)**: `mark_failed` runs as before; `save_conversation` is not called by this code path so cancellation is irrelevant.

### 8.18 Smart Auto-Scroll Pin State

- **User at bottom, chunk arrives with content > 50px**: `wasPinnedToBottom = true` (captured before), scroll restored to new bottom.
- **User scrolled up to read earlier content, chunk arrives**: `wasPinnedToBottom = false` (captured before), scroll position preserved.
- **User scrolls back to bottom manually, next chunk arrives**: `wasPinnedToBottom = true` again, scroll pinned.
- **Page is shorter than clientHeight (no scroll possible)**: `scrollHeight - clientHeight <= 0`, always pinned; scroll is a no-op but harmless.
- **Resize of the messages container (window resize) mid-stream**: The helper re-evaluates on every chunk, so the next chunk corrects any drift.
- **Multiple rapid chunks**: Each captures its own pinned state; works correctly even at high token rates.

### 8.19 Selection-Mode Send Guard

- **User in selection mode, clicks Send button**: `sendMessage()` returns immediately — no message, no state change.
- **User in selection mode, presses Enter in textarea**: Same — `sendMessage()` returns.
- **User in selection mode, types text and tries to send**: Text stays in the input; no send.
- **User exits selection mode (Cancel), then sends**: Guard is a no-op; normal flow.
- **`sendMessage()` called programmatically while in selection mode**: Returns immediately — future-proofs against any other call path.

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
| `test_chat_service.py` | `ChatService.generate_background()` with mocked LLM |
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

#### Repeated Refresh During Streaming

**Frontend:**
- [x] Streaming flag survives transient fetch errors (no longer cleared on `error.name !== 'AbortError'`)
- [x] `init` does not fall back to `loadConversation` when the streaming flag is set
- [x] 2nd, 3rd, 5th refresh during streaming all show the partial assistant message

#### Confirmation Modal

**Frontend:**
- [x] `confirm()` is gone — replaced by `showConfirmModal`
- [x] Modal is centered both horizontally and vertically
- [x] Backdrop click and Escape close (cancel)
- [x] Cancel and Confirm buttons work
- [x] `danger: true` produces a red confirm button
- [x] Focus moves to cancel when `danger: true` (Enter doesn't accidentally delete)

#### Batch Delete

**Frontend:**
- [x] Sidebar-header shows "Batch Delete" instead of `≡`
- [x] Selection mode shows checkboxes; clicking row toggles
- [x] Header count updates as items are toggled
- [x] Delete button disabled when count is 0
- [x] Cancel exits selection mode
- [x] Delete opens modal; confirm deletes all; cancel keeps all
- [x] Active conversation deletion → auto new chat
- [x] Single-item `×` delete still works and uses the new modal
- [x] Sidebar collapse via chat-header `≡` still works

#### Streaming-Conversation Resurrection Fix

**Backend:**
- [x] `test_generate_background_aborts_on_cancellation` passes — proves the resurrection bug is fixed
- [x] Normal stream completion still saves the assistant message to history
- [x] Deleting a conversation that has no in-flight LLM task still works

#### Smart Auto-Scroll During Streaming

**Frontend:**
- [x] Streaming with no user interaction: scroll stays pinned to the bottom throughout
- [x] User scrolls up during streaming: scroll position stays where the user put it
- [x] User scrolls back to the bottom: next chunk re-pins the scroll
- [x] Refresh-during-streaming (regression check): cached-chunks replay still scrolls to the bottom

#### Selection-Mode Send Guard

**Frontend:**
- [x] Click Send in selection mode → no message sent, input value preserved, selection intact
- [x] Press Enter in selection mode → no message sent, input value preserved, selection intact
- [x] Exit selection mode (Cancel) → Send and Enter both work normally
- [x] Deleting the active conversation via single `×` still ends in a fresh empty chat (no regression)
- [x] Deleting the active conversation via batch-delete (selected) still ends in a fresh empty chat (no regression)

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
| `backend/tests/conftest.py` | Pytest fixtures (`temp_storage_dir`, `mock_chain`) |
| `backend/tests/test_chat_routes.py` | `/api/chat/stream`, `/api/chat/stream/{id}`, `/api/chat/stream/status/{id}` HTTP tests (including from_pointer boundary regression tests) |
| `backend/tests/test_thinking_routes.py` | HTTP tests for thinking-aware endpoints (status partial_content, resume 404, post starts background task, delete clears job) |
| `backend/tests/test_chat_service.py` | `ChatService.generate_background()` with mocked LLM (thinking + token extraction, string content, append_chunk, cancellation, failure handling) |
| `backend/tests/test_storage.py` | Storage layer tests (JSON read/write, list sorting, title truncation, delete, invalid JSON) |
| `backend/tests/test_stream_manager.py` | StreamJob state transitions, unified chunks list + chunk_queue |

### Modified Files (Recent Features)

| File | Changes |
|------|---------|
| `backend/chat/stream_manager.py` | Unified chunks list + chunk_queue (vs separate token/thinking) |
| `backend/chat/service.py` | Uses `append_chunk(chunk_type, text)` for both thinking and tokens |
| `backend/chat/routes.py` | Single `from_pointer` param, unified chunk events, `end: true` sentinel; `stream_chat` synchronously appends user message for new conversations; `stream_from_job` `from_pointer` guard relaxed so boundary case is valid |
| `backend/chat/chain.py` | LLM configured as `minimax-3` with `max_tokens=16000` and extended thinking (`budget_tokens=10000`) |
| `backend/chat/stream_manager.py` | `get_or_create_job` calls `file_storage.create_conversation(conversation_id)` to ensure an empty entry exists in the conversations list |
| `backend/tests/test_chat_routes.py` | Added regression tests for `stream_from_job` boundary behavior (boundary yields end marker, boundary streams new chunks, out-of-range returns empty) |
| `frontend/index.html` | Unified chunk handling, cached chunks rendering, streaming state per conversation; history cache (`history_{convId}`) with helpers; cache-first `loadConversation`; SSE event buffering; streaming markdown parser state preservation across chunks and across the cache-replay → live-stream resume boundary; streaming badge derived per-render; `startNewChat` aborts in-flight stream and resets UI state; stream-resume catch block drops AbortError branching so the streaming flag survives transient fetch errors; `init` gates the `loadConversation` fallback on the streaming flag; themed `showConfirmModal` component (replaces `confirm()`); batch-delete selection mode (sidebar header has two layouts, per-item `×` hidden in selection mode, batch delete via the new modal); smart auto-scroll that captures pinned-to-bottom state before the DOM update; `sendMessage()` early-return guard when `selectionMode === true` |
| `backend/chat/stream_manager.py` | `StreamJob.cancelled` flag; `clear_job` sets the flag before removing from registry |
| `backend/chat/service.py` | `generate_background` checks `job.cancelled` in the chunk loop and before `save_conversation`; bails out (no `mark_completed`, no save) if set |
| `backend/tests/test_chat_service.py` | Regression test `test_generate_background_aborts_on_cancellation` — proves the resurrection bug is fixed |

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
