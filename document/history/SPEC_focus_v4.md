# Frontend Cache + Related Streaming Fixes - Specification

## Overview

This iteration delivers nine related changes:

1. **Conversation History Frontend Cache** — A localStorage cache of full conversation message lists, separate from the streaming chunks cache. Enables fast load on conversation switch/refresh without a backend round-trip.
2. **New Conversation Sidebar Visibility** — New conversations appear in the sidebar with the user message title while the LLM is still generating. Previously the conversation only appeared after the LLM response completed.
3. **SSE Event Buffering** — Frontend SSE parser correctly handles events that straddle chunk boundaries. Foundation for reliable streaming markdown rendering.
4. **Streaming Markdown Parser State Preservation** — Streaming markdown with multi-line constructs (tables, code blocks, math blocks) renders correctly when the tokens for a single construct are split across multiple SSE chunks.
5. **LLM Model Upgrade** — Backend LLM configuration is updated to use a new model with expanded output budget and explicit extended-thinking support.
6. **Backend Stream Resume Boundary Fix** — Resuming a stream at the exact-chunk-count boundary now streams queued chunks and the end marker instead of returning immediately and dropping the live stream.
7. **New-Chat UX Bugfixes** — Starting a new chat aborts any in-flight stream and resets the input/send button state so the user can immediately send in the new conversation.
8. **Streaming Badge Derivation** — The streaming badge on sidebar conversation items is derived from localStorage state on every sidebar render, so brand-new conversations display the badge correctly.
9. **Parser Handoff on Cache Resume** — The markdown parser state survives the transition from cache replay to live streaming during a resume, so multi-line constructs span the resume boundary correctly.

---

## Functional Requirements

### FR-HC-1: Conversation History Cache

| ID | Requirement |
|----|-------------|
| FR-HC-1.1 | Conversation message lists are cached in localStorage on first load |
| FR-HC-1.2 | `loadConversation` reads from cache if available, otherwise fetches from backend |
| FR-HC-1.3 | Cache persists across page refreshes |
| FR-HC-1.4 | Cache is invalidated when user deletes a conversation |
| FR-HC-1.5 | When a backend fetch succeeds, any stale `chunks_{conv_id}` is cleared so a leftover in-flight chunk cache cannot replay on top of fresh history |

### FR-HC-2: Cache Update Strategy

| ID | Requirement |
|----|-------------|
| FR-HC-2.1 | When user sends a message, append `{role: 'user', content}` to history cache |
| FR-HC-2.2 | When streaming completes, append `{role: 'assistant', content, thinking?}` to history cache |
| FR-HC-2.3 | Cache grows via append only — no full replacement after new exchanges |
| FR-HC-2.4 | When streaming completes, the chunks cache (`chunks_{conv_id}`) is cleared |

### FR-NSV: New Conversation Sidebar Visibility

| ID | Requirement |
|----|-------------|
| FR-NSV-1 | A brand-new conversation appears in the sidebar immediately after the user sends the first message — before the LLM response is generated |
| FR-NSV-2 | The sidebar title is derived from the user message (first 50 characters) |
| FR-NSV-3 | The user message is appended to backend storage synchronously when the stream is initiated, so the conversation is visible in `GET /api/chat/conversations` right away |
| FR-NSV-4 | The background LLM task does not duplicate the user message in conversation history |

### FR-SEB: SSE Event Boundary Handling

| ID | Requirement |
|----|-------------|
| FR-SEB-1 | The frontend SSE parser correctly handles events whose payload straddles two network chunks |
| FR-SEB-2 | Partial events at the end of a chunk are buffered and combined with the next chunk |
| FR-SEB-3 | Multiple complete events in one chunk are all processed |

### FR-SMP: Streaming Markdown Parser State

| ID | Requirement |
|----|-------------|
| FR-SMP-1 | Markdown with multi-line constructs (tables, fenced code blocks, math blocks, lists, blockquotes) renders correctly when the tokens for a single construct arrive in multiple SSE chunks |
| FR-SMP-2 | The streaming markdown parser is created once per stream and reused across chunks so its state accumulates correctly |
| FR-SMP-3 | The parser is finalized (`parser_end`) on stream completion, not during streaming |
| FR-SMP-4 | Markdown parser state persists across the transition from cache replay to live streaming during a resume, so multi-line constructs spanning the resume boundary render correctly |

### FR-MOD: Model Configuration

| ID | Requirement |
|----|-------------|
| FR-MOD-1 | The backend LLM is configured to use the `minimax-3` model |
| FR-MOD-2 | The maximum output tokens per response is 16000 |
| FR-MOD-3 | Extended thinking is enabled with a budget of 10000 tokens |
| FR-MOD-4 | The `document/SPEC.md` NFR-2 section records the `minimax-3` model name |

### FR-SR: Stream Resume Boundary

| ID | Requirement |
|----|-------------|
| FR-SR-1 | Resuming a stream with `from_pointer` equal to the current chunk count (the boundary case) yields queued chunks and the end marker instead of returning immediately |
| FR-SR-2 | Resuming a stream with a genuinely out-of-range `from_pointer` (greater than the current chunk count) returns immediately with no events |
| FR-SR-3 | A chunk appended to a stream after a boundary resume is yielded to the new resume request |

### FR-NC: New Chat Behavior

| ID | Requirement |
|----|-------------|
| FR-NC-1 | Starting a new chat aborts any in-flight stream for the current conversation before clearing UI state |
| FR-NC-2 | After starting a new chat, the input field is empty, the send button is enabled, and the input is focused for immediate typing |
| FR-NC-3 | The message input's auto-grow height is reset to default on new chat |

### FR-SB: Streaming Badge

| ID | Requirement |
|----|-------------|
| FR-SB-1 | The streaming badge is shown on a sidebar conversation item whenever that conversation is actively streaming |
| FR-SB-2 | The streaming badge is derived from localStorage state on every sidebar render (not just once when the message is sent), so brand-new conversations display the badge correctly on their first appearance in the sidebar |
| FR-SB-3 | The streaming badge is removed from a sidebar item once streaming completes for that conversation |

---

## Interface Requirements

### localStorage Schema (Frontend)

| Key | Content |
|-----|---------|
| `history_{conv_id}` | JSON array of messages `[{"role": "user"\|"assistant", "content": "...", "thinking": "..."?}]` (new) |
| `chunks_{conv_id}` | JSON array of stream chunks `[{"chunk": str, "type": "thinking\|token", "message_id": str}]` (existing) |
| `pointer_{conv_id}` | Integer position for resume (existing) |
| `streaming_{conv_id}` | Boolean flag for active streaming (existing) |

Note: `chunks_{conv_id}` caches streaming chunks (individual tokens during SSE). `history_{conv_id}` caches the full message list (complete messages). These are separate caches with different purposes and formats.

### Backend Storage Schema (unchanged)

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "thinking": "..."}
      ],
      "created_at": "ISO",
      "updated_at": "ISO"
    }
  }
}
```

### API Endpoints (unchanged)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/stream` | Start new chat stream |
| GET | `/api/chat/stream/{conversation_id}?from_pointer=N` | Resume stream from position N |
| GET | `/api/chat/stream/status/{conversation_id}` | Get stream status |
| GET | `/api/chat/history/{conversation_id}` | Get conversation history |
| GET | `/api/chat/conversations` | List all conversations |
| DELETE | `/api/chat/conversation/{conversation_id}` | Delete conversation + cleanup |

### SSE Event Format (unchanged)

```
data: {"chunk": "...", "type": "thinking|token", "message_id": "..."}
```

SSE events are delimited by `\n\n` (double newline).

---

## Data Flow

### Load Conversation (Cache-First)

```
loadConversation(convId)
├── localStorage['history_' + convId] exists?
│   ├── Yes → parse JSON, render directly
│   └── No  → fetch /api/chat/history/{convId}
│             → store in localStorage['history_' + convId]
│             → clear stale localStorage['chunks_' + convId]
│             → render
```

### Send Message → Cache Update + Sidebar Update

```
sendMessage()
├── user sends first message
│   → POST /api/chat/stream (no conversation_id)
│   → backend creates conversation, appends user message synchronously
│   → loadConversationList() picks up new conversation with title
│   → append {role: 'user', content} to history cache
│   → SSE stream begins (LLM still generating)
│
└── on stream end:
    → append {role: 'assistant', content, thinking?} to history cache
    → clear localStorage['chunks_' + convId]
```

### Delete Conversation → Cache Invalidation

```
deleteConversation(convId)
→ remove localStorage['history_' + convId]
→ remove localStorage['chunks_' + convId]
→ remove localStorage['pointer_' + convId]
→ remove localStorage['streaming_' + convId]
→ DELETE /api/chat/conversation/{convId}
```

### SSE Event Buffering

```
Network read returns chunk
       ↓
Append to sseBuffer
       ↓
sseBuffer.split('\n\n') → events
       ↓
events.pop() → sseBuffer (incomplete event kept)
       ↓
Process each complete event in events
```

### Streaming Markdown Parser Lifecycle

```
First token chunk:
    create parser + renderer

Subsequent token chunks:
    parser_write(parser, chunk)   ← same parser, state accumulates

On data.end:
    parser_end(parser)            ← finalize
```

### New Chat Flow

```
User clicks "New Chat"
       ↓
Abort in-flight stream (if any)
       ↓
Clear current conversation state (id, messages, active markers)
       ↓
Reset input field + send button state
       ↓
Focus input for immediate typing
```

### Resume Parser Handoff

```
resumeStreamFromPosition(convId)
       ↓
renderCachedChunks(chunks, contentDiv)  ← creates parser if needed
       ↓
returns [renderer, parser]
       ↓
processStreamResponse(response, isResume=true, ..., renderer, parser)
       ↓
parser reuses the same instance across cache replay and live streaming
       ↓
on data.end: parser_end (called exactly once)
```

---

## Frontend Changes

### New localStorage Key Generator

```javascript
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`,
    STREAMING: (convId) => `streaming_${convId}`,
    HISTORY: (convId) => `history_${convId}`  // NEW
};
```

### New Helper Functions

| Function | Purpose |
|----------|---------|
| `getHistoryCache(convId)` | Read history cache for a conversation |
| `setHistoryCache(convId, messages)` | Write history cache |
| `appendToHistoryCache(convId, message)` | Append a single message to history cache |
| `clearHistoryCache(convId)` | Remove history cache |
| `clearChunkCache(convId)` | Remove chunks cache |
| `renderMessagesFromCache(messages)` | Render messages from a cached or fetched messages array |
| `isStreamingForConv(convId)` | Check whether the streaming flag is set in localStorage for the given conversation |

### Modified Functions

| Function | Change |
|----------|--------|
| `loadConversation(convId)` | Check history cache before fetching; store fetched result in cache; clear stale chunks cache on backend fetch |
| `sendMessage()` | After adding the user message to UI, append to history cache; calls `loadConversationList()` after fetch to pick up the new conversation |
| `processStreamResponse()` on `data.end` | Append assistant message to history cache; clear chunks cache |
| `deleteConversation(convId)` | Remove `history_{convId}` from localStorage |
| `processStreamResponse()` | Use `sseBuffer` accumulator for SSE event parsing; hoist `renderer` / `parser` declarations to function scope so parser state persists across chunks |
| `processStreamResponse()` | Accept `existingRenderer` / `existingParser` parameters and return `{renderer, parser}` so the parser survives the cache-replay → live-stream transition during a resume |
| `loadConversationList()` | Derive the streaming badge from `isStreamingForConv(convId)` on every sidebar render so brand-new conversations display the badge correctly |
| `startNewChat()` | Abort `currentAbortController` to cancel any in-flight stream; reset `sendButton.disabled`, `messageInput.value`, and `messageInput.style.height`; refocus input |

### Unchanged Functions

- `init()` — calls `loadConversation`, which is now cache-first
- `switchConversation()` — calls `loadConversation`, which is now cache-first
- `checkStreamStatus()` — continues to use `chunks_{conv_id}` for stream resume
- `resumeStreamFromPosition()` — continues to use `chunks_{conv_id}` + `pointer_{conv_id}` (and threads the parser/renderer pair to `processStreamResponse` for parser handoff)

---

## Backend Changes

### Modified Functions

| Function | Change |
|----------|--------|
| `stream_chat` (`backend/chat/routes.py`) | When `existing_conv is None`, immediately call `file_storage.append_message(conversation_id, "user", request.message)` so the conversation appears in the sidebar with the correct title while the LLM is still generating |
| `stream_from_job` (`backend/chat/routes.py`) | Relax the `from_pointer` guard so the boundary case (`from_pointer == len(chunks)`) is valid and yields queued chunks / end marker; only genuinely out-of-range pointers are rejected |
| `create_chain` (`backend/chat/chain.py`) | Configure the backend LLM with the `minimax-3` model, 16000 max output tokens, and extended thinking enabled with a 10000-token budget |
| `ChatService.generate_background` (`backend/chat/service.py`) | Skip user message append if the same content is already the last message in history (idempotency) |
| `get_or_create_job` (`backend/chat/stream_manager.py`) | Calls `file_storage.create_conversation(conversation_id)` to ensure an empty entry exists in the conversations list |

### Unchanged

- `backend/storage/file_storage.py` — `append_message` and `create_conversation` already existed; no signature changes
- `backend/main.py`, `backend/config.py` — no changes

---

## Files to Modify

| File | Change |
|------|--------|
| `frontend/index.html` | Add `HISTORY` to `STORAGE_KEYS`; add history cache helpers; add `clearChunkCache`; add `renderMessagesFromCache`; add `isStreamingForConv`; modify `loadConversation` (cache-first, clear chunk cache on success); modify `sendMessage` to update history cache and call `loadConversationList`; modify `deleteConversation` to clear history cache; modify `processStreamResponse` to append assistant message to history cache on `data.end`; add `sseBuffer` event accumulator; hoist `renderer` / `parser` declarations to function scope; accept `existingRenderer` / `existingParser` and return `{renderer, parser}` for parser handoff on resume; derive streaming badge from `isStreamingForConv` in `loadConversationList`; abort stream and reset UI state in `startNewChat` |
| `backend/chat/chain.py` | Configure `minimax-3` model with 16000 max tokens and 10000 thinking budget |
| `backend/chat/routes.py` | Modify `stream_chat` to append user message to storage immediately for new conversations; reset job state on re-stream; relax `stream_from_job` `from_pointer` guard so the boundary case is valid |
| `backend/chat/service.py` | Modify `generate_background` to skip duplicate user message append |
| `backend/chat/stream_manager.py` | `get_or_create_job` now calls `file_storage.create_conversation` |
| `backend/tests/test_chat_routes.py` | Add regression tests for `stream_from_job` boundary behavior (boundary yields end marker, boundary streams new chunks, out-of-range returns empty) |
| `document/SPEC.md` | Update NFR-2 model reference to `minimax-3` |
| `backend/storage/file_storage.py` | No changes — `append_message` and `create_conversation` already existed |

---

## Acceptance Criteria

### History Cache (FR-HC)

- [ ] `loadConversation` returns cached data without network request when cache exists
- [ ] `loadConversation` fetches from backend and populates cache when no cache exists
- [ ] Sending a message appends user message to history cache immediately
- [ ] Streaming completion appends assistant message to history cache
- [ ] Streaming completion clears the chunks cache
- [ ] Deleting a conversation removes `history_{convId}` from localStorage
- [ ] Page refresh reads from history cache if available
- [ ] History cache and chunks cache use separate localStorage keys and serve different purposes
- [ ] A successful backend history fetch clears any stale `chunks_{convId}` so a leftover in-flight chunk cache cannot replay

### New Conversation Sidebar Visibility (FR-NSV)

- [ ] First message in a new conversation appears in the sidebar immediately (before LLM completes)
- [ ] Sidebar title is the first 50 characters of the user message
- [ ] No duplicate user message in final conversation history
- [ ] Existing conversation resumes without re-appending the user message

### SSE Event Buffering (FR-SEB)

- [ ] Events that straddle chunk boundaries are parsed correctly
- [ ] Multiple events in one chunk are all processed
- [ ] Buffer does not grow unboundedly

### Streaming Markdown Parser State (FR-SMP)

- [ ] Markdown tables render correctly when tokens are split across SSE chunks
- [ ] Fenced code blocks render correctly when tokens are split across chunks
- [ ] Multi-line math blocks (KaTeX) render correctly when split across chunks
- [ ] Parser is created once per stream, not per chunk batch
- [ ] `smd.parser_end` is called on stream end, not during streaming
- [ ] Multi-line constructs spanning the cache-replay → live-stream resume boundary render correctly

### Model Configuration (FR-MOD)

- [ ] Backend LLM is configured with the `minimax-3` model
- [ ] Max output tokens per response is 16000
- [ ] Extended thinking is enabled with a 10000-token budget
- [ ] `document/SPEC.md` NFR-2 references `minimax-3`

### Stream Resume Boundary (FR-SR)

- [ ] Resume at the boundary (`from_pointer == len(chunks)`) yields the end marker
- [ ] Resume at the boundary streams new chunks arriving after the resume request
- [ ] Genuinely out-of-range `from_pointer` (> len) returns immediately with no events

### New Chat Behavior (FR-NC)

- [ ] Starting a new chat aborts the in-flight stream (no background requests continue)
- [ ] After new chat, the input is empty, the send button is enabled, and the input is focused
- [ ] Input height is reset to default on new chat

### Streaming Badge (FR-SB)

- [ ] Streaming badge appears on a sidebar item whenever that conversation is actively streaming
- [ ] Streaming badge appears for brand-new conversations on the first sidebar render after send
- [ ] Streaming badge is removed from a sidebar item once streaming completes for that conversation

---

## Code Quality Fixes

The following internal fixes are bundled with this iteration; they do not change observable behavior but are documented for completeness. Implementation details are in DESI_focus.md Part I.

- **Variable scope hardening in `switchConversation`** — Two call sites that previously assigned to a bare `status` identifier now declare a local binding, eliminating the risk of clobbering an outer variable or creating an implicit global.
- **Dead sanitization call acknowledged** — A stale sanitization call that did nothing useful (called without its sanitization function and without assigning the result) is preserved with an explicit unused marker for future reference.
- **Per-chunk parser finalization removed** — A parser-finalization call that ran inside the SSE processing loop on every chunk has been removed so that parser finalization occurs exactly once on stream completion (per FR-SMP-3).

---

## Out of Scope

- Backend changes beyond those described
- Cache expiration / TTL
- Cache size limits
- Preloading caches for multiple conversations
- Cross-tab cache invalidation
- Diagnostic prints (already removed)
