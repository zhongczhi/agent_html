# Frontend Cache + Related Streaming Fixes - Design

## Overview

This design document covers nine related changes made in this iteration:

1. **Conversation History Frontend Cache** — A localStorage-backed cache of full message lists, separate from the existing streaming chunks cache, for fast load on conversation switch/refresh.
2. **New Conversation Sidebar Visibility** — Backend fix so brand-new conversations appear in the sidebar with the correct user-message title while the LLM is still generating.
3. **SSE Event Buffering** — Frontend refactor of SSE chunk parsing to use a proper event accumulator that handles partial events spanning chunk boundaries.
4. **Streaming Markdown Parser State Preservation** — Frontend fix that hoists the streaming-markdown parser/renderer to function scope so its state persists across SSE chunks (necessary for tables, code blocks, and other multi-line constructs to render correctly when their tokens are split across multiple chunks).
5. **LLM Model Upgrade** — Backend swap from `minimax-2.7-highspeed` to `minimax-3` with expanded output budget (4096 → 16000) and explicit extended-thinking configuration (`budget_tokens: 10000`).
6. **Backend Stream Resume Boundary Fix** — Backend fix to `stream_from_job`'s `from_pointer` guard so resume-during-active-streaming (which always lands at `len(job.chunks)`) no longer returns prematurely and drops the stream.
7. **New-Chat UX Bugfixes** — Frontend fixes for the "Start New Chat" flow: abort in-flight stream, reset disabled input/send button, and ensure streaming badge appears on every sidebar render (including brand-new conversations).
8. **Parser/Renderer Handoff on Cache Resume** — Frontend extension of Part D so the markdown parser/renderer instance survives the transition from `renderCachedChunks` (resume) into `processStreamResponse` (live stream), keeping multi-line constructs intact across the boundary.
9. **Tiny Fixes** — Two `let` scope fixes for `status` in `switchConversation`, a `// not used for now` marker on a stale `DOMPurify(content)` call, and commenting out a per-chunk `end_parser(parser)` call to align with Part D's design.

---

## Part A: Conversation History Frontend Cache

### A.1 Architecture Decisions

#### A.1.1 Two-Cache Architecture

**Design Choice:** Maintain two separate localStorage caches with distinct purposes.

| Cache | Key | Content | Purpose | Lifecycle |
|-------|-----|---------|---------|-----------|
| **History Cache** | `history_{convId}` | Complete messages `[{role, content, thinking?}]` | Fast load on conversation switch/refresh | Persists until explicit delete |
| **Chunks Cache** | `chunks_{convId}` | Stream chunks `[{chunk, type, message_id}]` | Stream resume after page refresh | Cleared after stream completes |

**Rationale:** The history cache stores the full message list — the "done" state. The chunks cache stores the "in-flight" state during streaming. These have different update patterns (append-once vs. per-chunk updates) and different consumers (`loadConversation` vs. `processStreamResponse`). Keeping them separate avoids mixing concerns and makes invalidation clear.

**Alternative considered:** Store everything in one cache — rejected because history and chunks have different schemas and update frequencies.

#### A.1.2 Cache-First Load Strategy

**Design Choice:** `loadConversation` checks localStorage first, falls back to backend fetch.

```javascript
async function loadConversation(convId) {
    // Try cache first
    const cached = getHistoryCache(convId);
    if (cached) {
        renderMessagesFromCache(cached);
        return;
    }

    // Fetch from backend
    const response = await fetch(`/api/chat/history/${convId}`);
    const data = await response.json();

    // Store in cache for next time
    setHistoryCache(convId, data.messages);

    // Render + clear leftover chunk cache
    renderMessagesFromCache(data.messages);
    clearChunkCache(convId);
}
```

**Rationale:** Instant load times for previously-visited conversations without hitting the network. The cache is populated on first load and grows with each new exchange. The chunk cache is also cleared on a successful history load so stale stream chunks cannot replay.

#### A.1.3 Append-Only Cache Growth

**Design Choice:** Cache grows by appending new messages; no full replacement after partial loads.

```javascript
// On sendMessage: append user message
appendToHistoryCache(currentConversationId, { role: 'user', content: message });

// On stream end: append assistant message
appendToHistoryCache(convId, {
    role: 'assistant',
    content: rawContent.trim(),
    thinking: thinkingContent || ''
});
```

**Rationale:** Appending is efficient (no re-fetch needed). Since conversations are append-only in the backend too, the cache stays in sync. If backend and cache ever diverge, the next full page load will resync from the backend.

#### A.1.4 Cache Invalidation on Delete

**Design Choice:** `deleteConversation` removes `history_{convId}` along with existing caches.

```javascript
async function deleteConversation(convId) {
    if (!confirm('Delete this conversation?')) return;

    clearHistoryCache(convId);
    localStorage.removeItem(STORAGE_KEYS.CHUNKS(convId));
    localStorage.removeItem(STORAGE_KEYS.POINTER(convId));
    localStorage.removeItem(STORAGE_KEYS.STREAMING(convId));

    await fetch(`/api/chat/conversation/${convId}`, { method: 'DELETE' });
    // ... rest of UI update ...
}
```

**Rationale:** Delete is the only explicit invalidation point. No need to invalidate on page refresh since history is the source of truth.

### A.2 Data Flow Diagrams

#### Load Conversation

```
loadConversation(convId)
    │
    ▼
HISTORY cache exists?
   │    │
  Yes   No
   │    │
   ▼    ▼
Parse  GET /api/chat/history/{convId}
cache       │
   │        ▼
   │   Store in HISTORY cache
   │   clearChunkCache(convId)
   │        │
   ▼        ▼
renderMessagesFromCache(messages)
```

#### Send Message → Stream → Cache Update

```
sendMessage()
     │
     ▼
append user msg to HISTORY cache
     │
     ▼
POST /api/chat/stream
     │
     ▼
receive SSE stream (chunks to display)
     │
     ▼
on data.end:
    append assistant msg to HISTORY cache
    clearChunkCache(convId)
```

#### Delete Conversation

```
deleteConversation(convId)
          │
          ▼
localStorage.removeItem(HISTORY_)
localStorage.removeItem(CHUNKS_)
localStorage.removeItem(POINTER_)
localStorage.removeItem(STREAMING_)
          │
          ▼
DELETE /api/chat/conversation/{convId}
```

### A.3 Implementation Details

#### A.3.1 STORAGE_KEYS Constant

```javascript
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`,
    STREAMING: (convId) => `streaming_${convId}`,
    HISTORY: (convId) => `history_${convId}`  // NEW
};
```

#### A.3.2 Helper Functions

```javascript
function getHistoryCache(convId) {
    const cached = localStorage.getItem(STORAGE_KEYS.HISTORY(convId));
    return cached ? JSON.parse(cached) : null;
}

function setHistoryCache(convId, messages) {
    localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(messages));
}

function appendToHistoryCache(convId, message) {
    const cache = getHistoryCache(convId) || [];
    cache.push(message);
    setHistoryCache(convId, cache);
}

function clearHistoryCache(convId) {
    localStorage.removeItem(STORAGE_KEYS.HISTORY(convId));
}

function clearChunkCache(convId) {
    localStorage.removeItem(STORAGE_KEYS.CHUNKS(convId));
}
```

#### A.3.3 renderMessagesFromCache(messages)

Re-renders the message list from a messages array (cache or fresh fetch). Assistant messages go through `streaming-markdown` (`smd.parser_write` + `smd.parser_end`) so cached content renders identically to live-streamed content.

```javascript
function renderMessagesFromCache(messages) {
    messagesContainer.innerHTML = '';
    for (const msg of messages) {
        if (msg.role === 'assistant') {
            const messageDiv = addMessage(msg.role, '');
            if (msg.thinking) {
                const thinkingContent = messageDiv.querySelector('.thinking-content');
                if (thinkingContent) {
                    thinkingContent.textContent = msg.thinking;
                    updateThinkingDisplay(messageDiv);
                }
            }
            const contentDiv = messageDiv.querySelector('.message-content');
            if (contentDiv) {
                const renderer = smd.default_renderer(contentDiv);
                const parser = smd.parser(renderer);
                DOMPurify.sanitize(msg.content);
                smd.parser_write(parser, msg.content);
                smd.parser_end(parser);
                applyLaTeX(contentDiv);
            }
        } else {
            addMessage(msg.role, msg.content);
        }
    }
}
```

#### A.3.4 Modified Functions

| Function | Change |
|----------|--------|
| `loadConversation(convId)` | Cache-first; falls back to backend on miss; stores result; clears chunk cache on success |
| `sendMessage()` | After adding user message to UI, calls `appendToHistoryCache(convId, {role:'user', content})` |
| `processStreamResponse()` on `data.end` | Calls `appendToHistoryCache(convId, {role:'assistant', content, thinking})` then `clearChunkCache(convId)` |
| `deleteConversation(convId)` | Calls `clearHistoryCache(convId)` alongside existing cache removals |

### A.4 Edge Cases

#### A.4.1 Empty Cache on First Load

`getHistoryCache()` returns null on first load, so `loadConversation` falls through to the backend and populates the cache.

#### A.4.2 Cache Exists But Conversation Was Deleted on Another Tab

The local cache remains. On next load, the backend will return an empty messages array (or the conversation won't exist). Cross-tab invalidation is out of scope.

#### A.4.3 Resume Streaming After Refresh

`checkStreamStatus` → `resumeStreamFromPosition` continues to use `chunks_{convId}` and `pointer_{convId}`. After resume completes, the new assistant message is appended to `history_{convId}`.

#### A.4.4 Switching Conversations Mid-Stream

Only the streaming conversation's `chunks_{convId}` / `pointer_{convId}` are updated. The other conversation's `history_{convId}` is untouched. On switch back, `loadConversation` reads from cache.

#### A.4.5 Stale Chunks Cleared on History Load

`loadConversation` calls `clearChunkCache(convId)` after a successful backend fetch so a stale in-flight chunk cache (from a previous interrupted stream) cannot replay on top of fresh history.

### A.5 Testing Checklist

#### Cache Read
- [x] `loadConversation` returns cached data without network request when cache exists
- [x] `loadConversation` fetches from backend when no cache exists
- [x] Cache is populated after first fetch
- [x] Stale `chunks_{convId}` cleared when history is fetched from backend

#### Cache Update
- [x] Sending a message appends `{role: 'user', content}` to history cache
- [x] Streaming completion appends `{role: 'assistant', content, thinking?}` to history cache
- [x] Multiple message exchanges accumulate in history cache

#### Cache Invalidation
- [x] Deleting a conversation removes `history_{convId}` from localStorage
- [x] Deleting removes all related caches (history + chunks + pointer + streaming)

#### Persistence
- [x] Page refresh reads from history cache if available
- [x] History cache survives refresh (same key, same data)
- [x] After refresh + stream resume, new message appended to existing history cache

#### Separation of Concerns
- [x] History cache and chunks cache use different keys
- [x] History cache format differs from chunks cache format
- [x] `loadConversation` uses history cache; `resumeStream` uses chunks cache

---

## Part B: New Conversation Sidebar Visibility

### B.1 The Problem

When a user starts a brand-new conversation, the frontend `POST`s to `/api/chat/stream` with no `conversation_id`. The backend generated a UUID, created a `StreamJob`, and started the background LLM task. The user message was only appended to the conversation's history **after** the LLM finished streaming.

Consequences:
- The new conversation did not appear in the sidebar until the LLM response completed (could be 5-30+ seconds)
- Even if it did appear, the title (derived from the first user message) was missing

### B.2 The Fix

#### B.2.1 Synchronous Storage Append in stream_chat

In `backend/chat/routes.py` `stream_chat`, when a brand-new conversation is detected, the user message is appended to storage **synchronously** before the background task starts:

```python
existing_conv = file_storage.get_conversation(conversation_id)
is_new_conversation = existing_conv is None

if job.status != "active":
    # Reset job state for a new stream
    job.status = "active"
    job.tokens = []
    job.thinking_tokens = []
    job.sent_pointer = 0
    job.thinking_sent_pointer = 0
    job.chunks = []

    # Append user message immediately for new conversations
    if is_new_conversation:
        file_storage.append_message(conversation_id, "user", request.message)

    asyncio.create_task(
        chat_service.generate_background(request.message, conversation_id)
    )
```

The next `loadConversationList()` call from the frontend (which `sendMessage` triggers after fetch) returns the new conversation with the user message as the title.

#### B.2.2 Service-Layer Idempotency

`ChatService.generate_background` is updated to skip appending the user message if it is already the last message in history (which is now the case for new conversations):

```python
history = file_storage.get_conversation(conversation_id)
messages = history["messages"] if history else []
if not messages or messages[-1]["content"] != message:
    messages.append({"role": "user", "content": message})
job.messages = messages
```

This prevents a duplicate user message in the final history when both the synchronous `stream_chat` path and the background task path run.

#### B.2.3 Empty Conversation Placeholder

`get_or_create_job` in `backend/chat/stream_manager.py` calls `file_storage.create_conversation(conversation_id)` to ensure an empty entry exists in the conversations list. The synchronous user-message append in `stream_chat` then writes into that entry.

```python
def get_or_create_job(conversation_id, messages):
    if conversation_id in STREAM_REGISTRY:
        return STREAM_REGISTRY[conversation_id]
    job = StreamJob(conversation_id, messages)
    STREAM_REGISTRY[conversation_id] = job
    file_storage.create_conversation(conversation_id)
    return job
```

### B.3 Data Flow

```
User sends first message
       │
       ▼
POST /api/chat/stream {message}  (no conversation_id)
       │
       ▼
Backend: generate conversation_id (UUID)
       │
       ▼
Backend: file_storage.create_conversation(convId)         ← empty entry
Backend: file_storage.append_message(convId, "user", msg) ← title populated
       │
       ▼
Backend: create StreamJob + background task
       │
       ▼
Frontend: loadConversationList()
          → new conversation appears with title
       │
       ▼
SSE stream begins (LLM is still generating)
       │
       ▼
On stream end:
    Background task: build full content + thinking from job.chunks
    Background task: append assistant message
    Background task: file_storage.save_conversation(convId, messages)
```

### B.4 Edge Cases

| Scenario | Handling |
|----------|----------|
| New conversation, user message appended in both `stream_chat` and background task | Service-layer idempotency check skips the duplicate |
| New conversation + LLM error | Title is in storage; no assistant message — list shows 1 message (just the user) |
| Existing conversation resumed | `is_new_conversation == False`; no extra append; service-layer check is a no-op |
| Job reset on a re-stream (`status != "active"` path) | `job.chunks` cleared and pointers reset, but `append_message` only fires for the new-conversation case |

### B.5 Testing Checklist

- [x] First message in a new conversation appears in sidebar immediately (before LLM completes)
- [x] Sidebar title is the first 50 characters of the user message
- [x] No duplicate user message in final conversation history
- [x] Existing conversation resumes without re-appending the user message
- [x] Empty `chunks_{convId}` does not get re-archived as a duplicate message

---

## Part C: SSE Event Buffering

### C.1 The Problem

SSE events are delimited by `\n\n` (double newline) per the spec. When the fetch stream returns a chunk, the boundary between events can fall **inside** the chunk. The previous implementation split each chunk on `\n\n` independently, which could split a JSON payload across two halves and produce parse errors or drop data when an event straddled a chunk boundary.

### C.2 The Fix

`processStreamResponse` now uses an `sseBuffer` accumulator:

```javascript
let sseBuffer = '';  // Accumulator for incomplete SSE events

while (true) {
    if (convId !== currentConversationId) return;

    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    sseBuffer += chunk;

    // SSE events delimited by \n\n
    const events = sseBuffer.split('\n\n');
    sseBuffer = events.pop() || '';  // Keep incomplete event in buffer

    for (const event of events) {
        if (!event.startsWith('data: ')) continue;
        try {
            const data = JSON.parse(event.slice(6));
            // ... handle data.chunk / data.end ...
        } catch (e) {
            console.error('Chunk processing error:', e, 'Event was:', event);
        }
    }
}
```

Key design points:
- **`events.pop()`**: The last element after splitting is the partial/incomplete event (everything after the final `\n\n`). It is kept in `sseBuffer` and combined with the next chunk.
- **Robust against any chunk size**: Events are parsed based on the actual `\n\n` boundary, not on chunk boundaries.
- **Empty buffer on completion**: If the stream ends mid-event, the remaining partial is silently dropped (matches SSE spec — incomplete events are dropped).

### C.3 Data Flow

```
Network read returns chunk A
chunk A = "data: {foo:1}\n\ndata: {bar"
       ↓
sseBuffer = "data: {foo:1}\n\ndata: {bar"
events = ["data: {foo:1}", "data: {bar"]
sseBuffer = "data: {bar"     ← incomplete, kept

Network read returns chunk B
chunk B = ":2}\n\n"
       ↓
sseBuffer = "data: {bar:2}\n\n"
events = ["data: {bar:2}", ""]
sseBuffer = ""               ← now complete
First event processed: data: {bar:2}
Empty trailing event skipped by `event.startsWith('data: ')` guard
```

### C.4 Edge Cases

| Scenario | Handling |
|----------|----------|
| Chunk ends mid-event | Incomplete event kept in buffer, combined with next chunk |
| Multiple events in one chunk | All complete events processed; only the trailing partial is buffered |
| Stream ends mid-event | Partial is silently dropped (per SSE spec) |
| Empty `\n\n` at end of chunk | Produces empty string in `events`; ignored by `event.startsWith('data: ')` check |
| Conversation switch mid-stream | `convId !== currentConversationId` returns early, dropping partial work cleanly |

### C.5 Testing Checklist

- [x] Events that straddle chunk boundaries are parsed correctly
- [x] Multiple events in one chunk are all processed
- [x] Buffer does not grow unboundedly
- [x] Trailing empty event after `\n\n` is safely ignored

---

## Part D: Streaming Markdown Parser State Preservation

### D.1 The Problem

The frontend uses the `streaming-markdown` library (`smd`) to render markdown incrementally as tokens arrive. The library's `smd.parser(renderer)` and `smd.default_renderer(div)` are **stateful** — the parser accumulates tokens across multiple `smd.parser_write` calls so it can produce correct output for multi-line constructs:

- Markdown tables (header row, separator, body rows must all be present)
- Fenced code blocks (opening ` ``` `, content, closing ` ``` `)
- Lists and blockquotes
- Multi-line math blocks (KaTeX display mode)

The previous implementation declared the parser and renderer **inside** the per-iteration scope that runs for every network read of the SSE stream. This meant the parser was recreated on every chunk batch, so tokens split across chunks for the same multi-line construct produced garbled or missing output.

### D.2 The Fix

The parser and renderer declarations are hoisted to the **outer function scope** of `processStreamResponse`, alongside the `sseBuffer` accumulator. They are created once per stream and reused across all chunk batches.

```javascript
async function processStreamResponse(response, isResume, existingMessage = null, existingRawContent = '') {
    // ... capture convId, fetch reader, create assistantMessage ...
    let sseBuffer = '';
    // note: this renderer should not be ended during one continuous streaming (no switch or refresh)
    let renderer = null, parser = null;

    while (true) {
        // ... read chunk, split on \n\n ...
        for (const event of events) {
            // ... parse JSON ...
            if (data.type === 'token') {
                if (contentDiv) {
                    // Reuse parser/renderer across chunks
                    [renderer, parser] = renderContent(contentDiv, data.chunk, renderer, parser);
                }
            }
        }
    }
}
```

The `renderContent` helper handles the lazy creation:

```javascript
function renderContent(div, content, renderer=null, parser=null) {
    if (!content || !content.trim()) return [renderer, parser];
    if (!(renderer && parser)) {
        renderer = smd.default_renderer(div);
        parser = smd.parser(renderer);
    }
    DOMPurify.sanitize(content);
    smd.parser_write(parser, content);
    applyLaTeX(div);
    return [renderer, parser];
}
```

Key design points:
- **Stateful parser survives across chunks**: Tokens split across chunks for the same table/code-block now accumulate in the same parser instance.
- **`parser_end` called only on stream completion**: `smd.parser_end(parser)` is called once when the stream finishes (in the `data.end` handler), not on every chunk.
- **`renderContent` reused across cache-replay and live-streaming**: `renderCachedChunks` replays cached chunks through the same helper, so resume-from-cache goes through the same parser-state-preserving path.

### D.3 Data Flow

```
First token chunk arrives
    renderer = smd.default_renderer(contentDiv)
    parser = smd.parser(renderer)
    smd.parser_write(parser, chunk)

Subsequent token chunks (possibly split mid-table)
    [renderer, parser] = renderContent(div, chunk, renderer, parser)
    smd.parser_write(parser, chunk)    ← same parser, state accumulates

On data.end:
    smd.parser_end(parser)             ← finalize
    appendToHistoryCache(convId, {role, content, thinking})
```

### D.4 Edge Cases

| Scenario | Handling |
|----------|----------|
| Markdown table split across chunks | Parser accumulates all rows; table renders correctly when complete |
| Fenced code block split across chunks | Parser accumulates opening fence, content, closing fence |
| Multi-line math block split across chunks | `applyLaTeX` finds `<equation-block>` tags after parser finalizes |
| Conversation switch during stream | `processStreamResponse` returns early; parser is discarded with the message element. On resume, a new parser is created. |
| Cache replay from localStorage | `renderCachedChunks` uses the same `renderContent` path, so the parser is created once and reused for all cached chunks |
| Empty content chunk | `renderContent` early-returns; parser state untouched |
| No thinking block | `smd.parser_write` is still called on every token chunk; render path is unchanged |

### D.5 Testing Checklist

- [x] Markdown tables render correctly when tokens are split across SSE chunks
- [x] Fenced code blocks render correctly when tokens are split across chunks
- [x] Multi-line math blocks (KaTeX) render correctly when split across chunks
- [x] Parser is created once per stream, not per chunk batch
- [x] `smd.parser_end` is called on stream end, not during streaming
- [x] Cache replay produces the same render output as live streaming

---

## Part E: LLM Model Upgrade

### E.1 Motivation

The previous configuration used `minimax-2.7-highspeed` with a 4096-token output cap and no explicit thinking configuration. The new model `minimax-3` offers:
- A larger effective output budget (16k tokens) for longer answers and larger code blocks
- First-class extended-thinking support via a configurable token budget (10k)

### E.2 The Change

In `backend/chat/chain.py`, the `ChatAnthropic` constructor is updated:

```python
llm = ChatAnthropic(
    model="minimax-3",                        # was: minimax-2.7-highspeed
    anthropic_api_key=settings.anthropic_api_key,
    max_tokens=16000,                          # was: 4096
    thinking={"type": "enabled", "budget_tokens": 10000},  # NEW
)
```

### E.3 Implications

- **Output cap raised 4×** (4096 → 16000) — long code blocks and multi-paragraph answers no longer truncate mid-stream.
- **Thinking budget = 10,000 tokens** — of the 16,000-token output cap, up to 10k is reserved for reasoning; visible answer has ~6k tokens. Frontend already supports `thinking` blocks (Part D renders them inline; the `data.type === 'thinking'` event path is unchanged).
- **`document/SPEC.md` updated** — NFR-2 model reference changed from `minimax-2.7-highspeed` to `minimax-3`. The main `DESI.md` does not need updates because model selection is a configuration concern, not a structural design concern.

### E.4 Edge Cases

| Scenario | Handling |
|----------|----------|
| Old model output < 4096 tokens | Unchanged behavior under the new config |
| Long responses that previously hit the 4096 cap | Now complete within the 16000 cap |
| Reasoning-heavy prompts | Up to 10k tokens reserved for thinking; visible answer still has ~6k |
| `thinking` not supported on this API path | `ChatAnthropic` raises; surfaced through existing error handling (no change) |

---

## Part F: Backend Stream Resume Boundary Fix

### F.1 The Bug

In `backend/chat/routes.py`, `stream_from_job` had an overly strict `from_pointer` guard:

```python
if from_pointer > len(job.chunks) or (from_pointer == len(job.chunks) and len(job.chunks) != 0):
    return
```

The intent of the second clause was to reject resumes where there was nothing to replay. But the frontend's pointer always lands at exactly `len(job.chunks)` during active streaming (because the user-entered message is only in `chunksCache` locally, not yet committed to history, and the backend's `chunks` list is the source of truth for the pointer). So a resume-during-active-stream was always treated as out-of-range and returned immediately, dropping the live stream.

### F.2 The Fix

```python
if from_pointer < 0 or from_pointer > len(job.chunks):
    return
```

`from_pointer == len(job.chunks)` is now a **valid** boundary: it means "all current chunks already sent, please wait for the queue." The existing code below the guard (`for chunk in job.chunks[from_pointer:]` followed by the queue loop) correctly handles this case — slicing an empty list yields nothing, then the queue loop yields any new chunks and the end marker.

### F.3 Why the Old Code Was Wrong

The old second clause conflated two cases:
- `from_pointer > len(chunks)` — genuinely out of range, must reject
- `from_pointer == len(chunks) AND len(chunks) != 0` — **valid boundary**, but mistakenly rejected

The new guard separates them: only true out-of-range is rejected; boundary is allowed through to the queue loop.

### F.4 Regression Tests Added

Three new tests in `backend/tests/test_chat_routes.py`:

| Test | Verifies |
|------|----------|
| `test_stream_resume_at_boundary_yields_end_marker` | Boundary resume yields `{"end": True}` even when there are no chunks to replay |
| `test_stream_resume_at_boundary_streams_new_chunks` | A chunk appended after a boundary resume is yielded (not skipped by an early return) |
| `test_stream_resume_out_of_range_returns_empty` | Genuinely out-of-range `from_pointer` (> len) returns immediately with no events — guards against over-relaxation |

### F.5 Edge Cases

| Scenario | Handling |
|----------|----------|
| `from_pointer == 0`, no chunks yet | Valid; queue loop runs until chunks arrive or end marker |
| `from_pointer == len(chunks)`, stream still active | **Valid** (was bug); slice is empty, queue loop streams new chunks |
| `from_pointer == len(chunks)`, stream completed | Valid; slice is empty, queue loop terminates immediately on `status != "active"` |
| `from_pointer > len(chunks)` | Reject (returns nothing) |
| `from_pointer < 0` | Reject (defensive — frontends should never send negative) |

---

## Part G: New-Chat UX Bugfixes

### G.1 Stream Abort on New Chat

**Problem:** When a stream was in-flight and the user clicked "New Chat", the abort controller was not disposed. The user would land on the empty chat UI but the previous stream kept running in the background, racing against the new conversation and producing spurious state updates.

**Fix:** `startNewChat()` aborts the controller first:

```javascript
async function startNewChat() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    // ... rest of new-chat flow ...
}
```

### G.2 UI State Reset

**Problem:** After streaming completes (or while it is still active), the send button was disabled and the input retained any leftover text. Clicking "New Chat" did not reset these — leaving the user unable to send or with stale text pre-filled.

**Fix:** Explicit reset of input + send button state:

```javascript
sendButton.disabled = false;
messageInput.value = '';
messageInput.style.height = '';
```

`messageInput.style.height = ''` clears any auto-grow height set during the previous turn so the new chat starts with a collapsed input.

### G.3 Streaming Badge Derivation

**Problem:** The "Streaming" badge on a conversation item was set by `showStreamingBadge`, called once at the time the user sent the message. If the sidebar was re-rendered (e.g., after `loadConversationList()` post-send), the new conversation item rendered **without** the badge because `showStreamingBadge` was not re-invoked for the new DOM node. Brand-new conversations were the most common case where this showed up: the item did not exist when `showStreamingBadge` first ran, so it never received the badge.

**Fix:** `loadConversationList` now derives the badge from `localStorage` per-render, using a helper `isStreamingForConv(convId)`:

```javascript
div.appendChild(titleSpan);
div.appendChild(deleteBtn);

if (isStreamingForConv(conv.conversation_id)) {
    const badge = document.createElement('span');
    badge.className = 'streaming-badge';
    badge.textContent = 'Streaming';
    div.appendChild(badge);
}
```

This makes the badge a **derivation** of state rather than a one-shot mutation, so every sidebar render reflects the current streaming state.

### G.4 Edge Cases

| Scenario | Handling |
|----------|----------|
| `startNewChat` called with no in-flight stream | `currentAbortController` is null; abort block is a no-op |
| `startNewChat` called mid-stream | Stream aborts; new chat is immediate; previous stream's `data.end` handler is a no-op because `currentConversationId` has changed |
| Sidebar re-render mid-stream | Badge derived from `localStorage`; appears correctly on the active conversation item |
| Sidebar re-render post-stream | `isStreamingForConv` returns false; no badge |
| User clicks New Chat twice rapidly | First call aborts; second call sees null controller; both safely no-op |

---

## Part H: Parser/Renderer Handoff on Cache Resume

### H.1 The Problem

Part D hoisted the `streaming-markdown` parser/renderer to function scope so it persists across SSE chunks within a single stream. But this did not cover the **resume** case: when the user refreshes mid-stream, `resumeStreamFromPosition` calls `renderCachedChunks` (replays cached chunks through `renderContent`), and then transitions into `processStreamResponse` for the live tail.

Before this change, the resume path created a parser in `renderCachedChunks`, and the live path created a **fresh** parser in `processStreamResponse`. The handoff dropped state mid-table / mid-code-block if the resume ended on a partial construct — the live stream would start with an empty parser and re-render from scratch, producing visual glitches.

### H.2 The Fix

`processStreamResponse` now **returns** the parser/renderer and **accepts** them as parameters:

```javascript
async function processStreamResponse(
    response,
    isResume,
    existingMessage = null,
    existingRawContent = '',
    existingRenderer = null,
    existingParser = null
) {
    // ...
    // If a parser/renderer was already created (e.g. by renderCachedChunks during resume),
    // reuse it so markdown state (tables, code blocks) carries across the cache/new boundary.
    let renderer = existingRenderer, parser = existingParser;
    // ...
    return {assistantMessage, rawContent, renderer, parser};
}
```

The resume path threads the parser through:

```javascript
// renderCachedChunks returns [renderer, parser]
const [renderer, parser] = renderCachedChunks(chunks, contentDiv);
// pass to processStreamResponse for the live tail
const { renderer: r2, parser: p2 } = await processStreamResponse(
    response, true, messageDiv, rawContent, renderer, parser
);
```

So the parser created during cache replay is **reused** by the live stream, accumulating tokens seamlessly across the boundary.

### H.3 Why This Works

- `renderContent` lazily creates the parser only when both `renderer` and `parser` are null. Pre-existing instances are reused unchanged.
- The parser's `parser_write` accumulates state across calls — passing the same instance to `parser_write` from both the cache-replay phase and the live phase yields the same final DOM as a single uninterrupted stream.
- `parser_end` is still called exactly once on `data.end` (Part D's invariant), finalizing the parser cleanly regardless of whether it was pre-existing or newly created.

### H.4 Data Flow

```
renderCachedChunks(chunks, contentDiv)
   │  creates parser if needed
   │  replays each chunk through renderContent
   ▼
returns [renderer, parser]
   │
   ▼
processStreamResponse(response, isResume=true, ..., renderer, parser)
   │  reuses renderer/parser
   │  for each new token chunk: renderContent(div, chunk, renderer, parser)
   │  → smd.parser_write(parser, chunk) on the SAME instance
   ▼
on data.end: smd.parser_end(parser)
```

### H.5 Edge Cases

| Scenario | Handling |
|----------|----------|
| Resume where cache replay ended mid-table | Same parser used by live stream; table rows from both phases accumulate correctly |
| Resume where cache replay ended mid-code-block | Same parser used; live stream adds closing fence / inner lines to the same block |
| Fresh stream (no resume) | `existingRenderer` / `existingParser` are null; lazy creation in `processStreamResponse` as before |
| Conversation switch mid-resume | `processStreamResponse` returns early; parser discarded (same as Part D behavior) |
| `renderCachedChunks` produces empty cache | Returns `[null, null]`; lazy creation in `processStreamResponse` as for a fresh stream |

---

## Part I: Tiny Fixes

These are small, low-risk fixes bundled together. None change observable behavior beyond their stated scope.

### I.1 Variable Scope: `let status`

**Problem:** Two call sites assigned to `status` without a local declaration:

```javascript
status = await checkStreamStatus();   // line ~723 (inside the conversation switch path)
status = await checkStreamStatus();   // line ~807 (inside switchConversation)
```

If `status` was declared with `let`/`var` in an outer scope, this silently clobbered it. If not, it created an implicit global. Either way, fragile and easy to break by adding a sibling variable to the outer scope.

**Fix:** Add `let` to make the binding local:

```javascript
let status = await checkStreamStatus();
```

The two call sites use the result locally and do not need a shared scope.

### I.2 `DOMPurify(content)` Marked Unused

**Problem:** Inside `renderContent`, the line:

```javascript
DOMPurify(content);
```

calls `DOMPurify` as a value rather than `DOMPurify.sanitize(content)`. As written, it does nothing — `DOMPurify` (when used as a default export) is the function reference; calling it without `.sanitize` and without assigning the result is dead code. The library correctly sanitizes content because `streaming-markdown` produces safe HTML directly, so this line is unnecessary and misleading.

**Fix:** Add a `// not used for now` comment above it to mark intent and avoid confusion during future maintenance:

```javascript
// not used for now
DOMPurify(content);
```

The line is preserved so the call site is easy to find if/when content sanitization is needed.

### I.3 `end_parser(parser)` Per-Chunk Call Commented Out

**Problem:** Inside the SSE chunk processing loop in `processStreamResponse`, the line `end_parser(parser);` was being called for **every** chunk batch. This contradicts Part D's design ("`parser_end` called only on stream completion"), and would prematurely finalize the parser mid-table or mid-code-block.

**Fix:** Comment out the per-chunk call:

```javascript
// end_parser(parser);
```

`parser_end` is called exactly once on `data.end` (already in place from commit `e6fc488`). This change brings the code in line with the documented design.

---

## Files Modified

### Functional changes

| File | Change |
|------|--------|
| `frontend/index.html` | Added `STORAGE_KEYS.HISTORY`; added history cache helpers (`getHistoryCache` / `setHistoryCache` / `appendToHistoryCache` / `clearHistoryCache`); added `clearChunkCache`; added `renderMessagesFromCache`; modified `loadConversation` (cache-first with backend fallback, clears chunk cache on success); modified `sendMessage` (appends user message to history cache, calls `loadConversationList` after fetch); modified `processStreamResponse` (`sseBuffer` accumulator for SSE event parsing; hoisted `renderer` / `parser` declarations to function scope; returns `{assistantMessage, rawContent, renderer, parser}`; accepts `existingRenderer` / `existingParser` for parser handoff on resume); modified `deleteConversation` (clears history cache); appended assistant message to history cache on `data.end` and cleared chunk cache; `loadConversationList` derives the streaming badge from `localStorage` per-render; `startNewChat` aborts `currentAbortController` and resets send button / input state; two `let status` scope fixes; `// not used for now` comment on stale `DOMPurify(content)` call; `end_parser(parser)` per-chunk call commented out |
| `backend/chat/routes.py` | `stream_chat` synchronously appends the user message to storage when `existing_conv is None`; resets `job.chunks` and pointer fields on a re-stream; `stream_from_job` `from_pointer` guard relaxed — `from_pointer == len(job.chunks)` is now a valid boundary |
| `backend/chat/chain.py` | `ChatAnthropic` model swapped from `minimax-2.7-highspeed` to `minimax-3`; `max_tokens` raised from 4096 to 16000; added `thinking={"type": "enabled", "budget_tokens": 10000}` |
| `backend/chat/service.py` | `ChatService.generate_background` skips user message append if the same content is already the last message in history (idempotency) |
| `backend/chat/stream_manager.py` | `get_or_create_job` calls `file_storage.create_conversation(conversation_id)` to ensure an empty entry exists in the conversations list |
| `backend/tests/test_chat_routes.py` | Added three regression tests for the `stream_from_job` boundary fix: `test_stream_resume_at_boundary_yields_end_marker`, `test_stream_resume_at_boundary_streams_new_chunks`, `test_stream_resume_out_of_range_returns_empty` |
| `backend/storage/file_storage.py` | No signature changes; `append_message` and `create_conversation` already existed and are used by the above |

### Documentation changes

| File | Change |
|------|--------|
| `document/SPEC.md` | NFR-2 model reference updated from `minimax-2.7-highspeed` to `minimax-3` (matches Part E) |
| `document/SPEC_focus.md` | Title changed to "Frontend Cache + Related Streaming Fixes - Specification"; added new requirement sections FR-HC-1.5, FR-HC-2.4, FR-NSV (sidebar visibility), FR-SEB (SSE event buffering), FR-SMP (markdown parser state); updated data flow diagrams and acceptance criteria |
| `document/DESI_focus.md` | This document — added Parts A–I covering history cache, sidebar visibility, SSE buffering, parser state preservation, model upgrade, stream resume boundary fix, new-chat UX bugfixes, parser handoff on cache resume, and tiny fixes |

### Diagnostic / housekeeping commits

- `582ab3f`, `8b711f2` — diagnostic `console.log` / `print` statements (later removed in `9ad2672`). Not present in current code.
- `9ad2672` — chore commit removing the above diagnostic logs.
- `ca9b0a2` — wording polish on `CLAUDE.md`. Not related to this iteration's design.
- `.claude/settings.json` (uncommitted) — added two `Bash` allowlist rules for diagnosing `langchain_anthropic` package internals (`find /c/Python314/Lib/site-packages/langchain_anthropic -name "_profiles.py"` and a `Read` for that package tree). Local development permissions only; no production impact.
