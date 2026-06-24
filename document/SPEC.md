# Chatbot Project - Requirements Specification

## Overview

A modular chatbot application that provides streaming AI responses with conversation history persistence. Users type questions and receive real-time streaming answers from an LLM. The system supports multiple conversations, persists history across sessions, and displays LLM thinking content with stream resume capability.

**Core Goal:** Deliver a chat interface with real-time streaming responses, persistent conversation history, and LLM thinking content display.

**Recent Improvements:** The system now includes a frontend conversation history cache for fast load, immediate sidebar visibility for new conversations, robust SSE event boundary handling, streaming markdown parser state preservation, an updated LLM model configuration with extended thinking, refined stream resume boundary semantics, a streamlined new-chat UX, stream resume that survives any number of mid-stream refreshes, a themed confirmation modal that replaces the browser-native dialog, batch delete for conversations, a fix for the streaming-conversation resurrection bug, smart auto-scroll that respects manual scrolling during streaming, a guard that blocks sending messages while in batch-deletion selection mode, code-quality cleanup across the codebase (frontend split into `index.html` + `static/styles.css` + `static/app.js`, all `localStorage` access consolidated into a `cache.js` module, `StreamStatusResponse` simplified to a single `status` string, FastAPI `Depends` for the chat service, typed `convert_messages` using `HumanMessage`/`AIMessage`), multi-turn thinking continuity (prior assistant `thinking` is fed back to the LLM as a content block so turns 2+ emit `thinking` chunks), stream-registry memory cleanup (a `StreamJob` is removed from `STREAM_REGISTRY` after a successful resume), file-storage concurrency safety (a per-process `threading.Lock` plus an atomic write helper so concurrent writers don't lose updates and a crash mid-write leaves the previous file fully intact), and explicit `TODO(U2 — known issue)` comments at the two `DOMPurify` no-op call sites in the frontend.

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

### FR-6: Frontend Conversation History Cache

| ID | Requirement |
|----|-------------|
| FR-6.1 | Conversation message lists are cached in localStorage on first load |
| FR-6.2 | `loadConversation` reads from cache if available, otherwise fetches from backend |
| FR-6.3 | Cache persists across page refreshes |
| FR-6.4 | Cache is invalidated when user deletes a conversation |
| FR-6.5 | When a backend fetch succeeds, any stale `chunks_{conv_id}` is cleared so a leftover in-flight chunk cache cannot replay on top of fresh history |
| FR-6.6 | When user sends a message, append `{role: 'user', content}` to history cache |
| FR-6.7 | When streaming completes, append `{role: 'assistant', content, thinking?}` to history cache |
| FR-6.8 | Cache grows via append only — no full replacement after new exchanges |
| FR-6.9 | When streaming completes, the chunks cache (`chunks_{conv_id}`) is cleared |

### FR-7: New Conversation Sidebar Visibility

| ID | Requirement |
|----|-------------|
| FR-7.1 | A brand-new conversation appears in the sidebar immediately after the user sends the first message — before the LLM response is generated |
| FR-7.2 | The sidebar title is derived from the user message (first 50 characters) |
| FR-7.3 | The user message is appended to backend storage synchronously when the stream is initiated, so the conversation is visible in `GET /api/chat/conversations` right away |
| FR-7.4 | The background LLM task does not duplicate the user message in conversation history |

### FR-8: SSE Event Boundary Handling

| ID | Requirement |
|----|-------------|
| FR-8.1 | The frontend SSE parser correctly handles events whose payload straddles two network chunks |
| FR-8.2 | Partial events at the end of a chunk are buffered and combined with the next chunk |
| FR-8.3 | Multiple complete events in one chunk are all processed |

### FR-9: Streaming Markdown Parser State

| ID | Requirement |
|----|-------------|
| FR-9.1 | Markdown with multi-line constructs (tables, fenced code blocks, math blocks, lists, blockquotes) renders correctly when the tokens for a single construct arrive in multiple SSE chunks |
| FR-9.2 | The streaming markdown parser is created once per stream and reused across chunks so its state accumulates correctly |
| FR-9.3 | The parser is finalized (`parser_end`) on stream completion, not during streaming |
| FR-9.4 | Markdown parser state persists across the transition from cache replay to live streaming during a resume, so multi-line constructs spanning the resume boundary render correctly |

### FR-10: Model Configuration

| ID | Requirement |
|----|-------------|
| FR-10.1 | The backend LLM is configured to use the `minimax-3` model |
| FR-10.2 | The maximum output tokens per response is 16000 |
| FR-10.3 | Extended thinking is enabled with a budget of 10000 tokens |
| FR-10.4 | The model name is recorded in the main SPEC document NFR-2 |

### FR-11: Stream Resume Boundary

| ID | Requirement |
|----|-------------|
| FR-11.1 | Resuming a stream with `from_pointer` equal to the current chunk count (the boundary case) yields queued chunks and the end marker instead of returning immediately |
| FR-11.2 | Resuming a stream with a genuinely out-of-range `from_pointer` (greater than the current chunk count) returns immediately with no events |
| FR-11.3 | A chunk appended to a stream after a boundary resume is yielded to the new resume request |

### FR-12: New Chat UX

| ID | Requirement |
|----|-------------|
| FR-12.1 | Starting a new chat aborts any in-flight stream for the current conversation before clearing UI state |
| FR-12.2 | After starting a new chat, the input field is empty, the send button is enabled, and the input is focused for immediate typing |
| FR-12.3 | The message input's auto-grow height is reset to default on new chat |

### FR-13: Streaming Badge

| ID | Requirement |
|----|-------------|
| FR-13.1 | The streaming badge is shown on a sidebar conversation item whenever that conversation is actively streaming |
| FR-13.2 | The streaming badge is derived from localStorage state on every sidebar render (not just once when the message is sent), so brand-new conversations display the badge correctly on their first appearance in the sidebar |
| FR-13.3 | The streaming badge is removed from a sidebar item once streaming completes for that conversation |

### FR-14: Stream Resume Survives Repeated Refresh

| ID | Requirement |
|----|-------------|
| FR-14.1 | On any number of refreshes during the streaming phase of a conversation, the partial assistant message (every token and thinking chunk stored in the chunk cache) is rendered before live streaming resumes |
| FR-14.2 | Live streaming continues to the same assistant message DOM node after the partial message is rendered |
| FR-14.3 | The streaming badge appears on the sidebar item for the active conversation on every refresh during streaming |
| FR-14.4 | The fix does not regress first-refresh-during-streaming, post-completion refresh, or new-message flows |

### FR-15: Custom Confirmation Modal

| ID | Requirement |
|----|-------------|
| FR-15.1 | A reusable confirmation modal component is provided (function `showConfirmModal({title, message, confirmText, cancelText, danger})` returns a Promise resolving to `true`/`false`) |
| FR-15.2 | The modal is rendered with a semi-transparent backdrop covering the whole viewport and is centered horizontally and vertically |
| FR-15.3 | The modal uses page-theme colors: `var(--bg-secondary)` background, `var(--border-color)` border, `var(--text-primary)` text, `var(--text-secondary)` muted text |
| FR-15.4 | The modal has a title, a message, a Cancel button (secondary), and a Confirm button (primary, red/danger when `danger: true`) |
| FR-15.5 | Clicking the backdrop, pressing Escape, or clicking Cancel closes the modal and resolves the Promise with `false` |
| FR-15.6 | Clicking Confirm closes the modal and resolves the Promise with `true` |
| FR-15.7 | All delete confirmations (single, batch) use this modal instead of `confirm()` — no browser-native `confirm()` calls remain in the codebase |

### FR-16: Batch Delete

| ID | Requirement |
|----|-------------|
| FR-16.1 | The sidebar-header `≡` button is replaced with a "Batch Delete" button (icon: trash) |
| FR-16.2 | Clicking "Batch Delete" enters selection mode |
| FR-16.3 | In selection mode, each conversation item shows a leading checkbox; the per-item `×` delete button is hidden |
| FR-16.4 | Clicking a conversation's checkbox toggles its selection state |
| FR-16.5 | The sidebar header in selection mode shows: a count, a Delete button, and a Cancel button — replacing the normal header content |
| FR-16.6 | The Delete button is disabled when zero items are selected; enabled and shows the count when one or more are selected |
| FR-16.7 | Clicking Cancel exits selection mode without deleting anything |
| FR-16.8 | Clicking Delete opens the themed confirmation modal with the count and item titles |
| FR-16.9 | On confirmation, all selected conversations are deleted: frontend caches cleared, backend DELETE called per item |
| FR-16.10 | After batch deletion completes, selection mode exits and the sidebar list refreshes |
| FR-16.11 | If the currently-active conversation was among the deleted ones, the user is switched to a new empty chat |
| FR-16.12 | The chat-header `≡` button remains and still collapses/expands the sidebar |

### FR-17: Streaming-Conversation Resurrection Fix

| ID | Requirement |
|----|-------------|
| FR-17.1 | When the user deletes a conversation while the LLM is still generating, the conversation must NOT reappear in `GET /api/chat/conversations` after the LLM finishes |
| FR-17.2 | The fix does not affect the normal stream-completion path (no delete): the assistant message is still appended to storage on `data.end` |
| FR-17.3 | The fix does not affect conversations that have no in-flight background task at delete time |
| FR-17.4 | The fix is contained to the backend (no frontend changes required) |

### FR-18: Smart Auto-Scroll During Streaming

| ID | Requirement |
|----|-------------|
| FR-18.1 | During streaming, the messages container auto-scrolls to the bottom on each chunk **only if** the user was already at (or within 50px of) the bottom before the chunk was applied |
| FR-18.2 | The "pinned" state is captured BEFORE the DOM update for the chunk, since the per-chunk content height can exceed 50px and would cause a post-update check to incorrectly report "not pinned" |
| FR-18.3 | Scrolling back to the bottom during streaming re-pins the scroll on the next chunk |
| FR-18.4 | Other scroll sites (cached-chunks replay on resume, sendMessage placeholder, addMessage) are unchanged — they happen once per action and aren't per-chunk auto-scrolls |

### FR-19: Block Sending Messages in Selection Mode

| ID | Requirement |
|----|-------------|
| FR-19.1 | When `sendMessage()` is invoked while `selectionMode === true`, the function returns immediately without mutating any state (selection, active conversation, messages, input value) |
| FR-19.2 | The Send button click handler routes through `sendMessage()` and is therefore blocked by FR-19.1 |
| FR-19.3 | The `messageInput` `keydown` handler (Enter key, when Shift is not held) routes through `sendMessage()` and is therefore blocked by FR-19.1 |
| FR-19.4 | After exiting selection mode (Cancel button), Send and Enter both work normally — no regression |

### FR-20: Multi-Turn Thinking Continuity

| ID | Requirement |
|----|-------------|
| FR-20.1 | On every turn of a multi-turn conversation, the assistant response includes at least one `thinking` content chunk |
| FR-20.2 | The previous assistant's `thinking` content is preserved when fed back to the LLM as a content block, so the chain of reasoning can continue across the conversation |
| FR-20.3 | Prior assistant messages still have only their visible `content` available to the LLM as the assistant's "output" |
| FR-20.4 | The change is contained to message conversion; no other code paths are affected |

### FR-21: Stream Registry Memory Cleanup

| ID | Requirement |
|----|-------------|
| FR-21.1 | A `StreamJob` is removed from `STREAM_REGISTRY` after a **successful** resume call has delivered the full cached chunk history (including the `end` marker) |
| FR-21.2 | A resume that is interrupted (client disconnect, browser navigation), returns early without yielding the `end` marker, or raises an exception leaves the job in place — a future resume must be able to continue |
| FR-21.3 | The cleanup applies only to the **resume** route (`GET /api/chat/stream/{conversation_id}`). The **initial** stream (`POST /api/chat/stream`) does not trigger cleanup, so a later resume is still possible if the initial stream is interrupted |
| FR-21.4 | A `StreamJob` for which the user never calls resume still leaks for the lifetime of the process (no TTL-based sweep) — a known limitation, not addressed in this iteration |

### FR-22: File Storage Concurrency Safety

| ID | Requirement |
|----|-------------|
| FR-22.1 | Two threads writing to `conversations.json` concurrently (e.g., two `append_message` calls) must both be reflected in the final file — no lost updates |
| FR-22.2 | A process crash (or any failure) during a write must leave the previous on-disk file fully intact — readers see either the fully-old or fully-new file, never partial |
| FR-22.3 | A failed write must clean up any leftover `.tmp` file in `STORAGE_DIR` so the directory does not accumulate stale partial writes |
| FR-22.4 | The behavior is observable to existing callers as "still produces a valid JSON file at `CONVERSATIONS_FILE`" — no signature change |
| FR-22.5 | The lock is per-process; a multi-worker deployment (`uvicorn --workers N`) is not addressed and would need a file-level lock — out of scope |

### FR-23: Frontend Cache Module

| ID | Requirement |
|----|-------------|
| FR-23.1 | All `localStorage` access in the frontend is encapsulated in [`frontend/static/cache.js`](../frontend/static/cache.js) |
| FR-23.2 | The module exposes typed accessors for each cache: history (get/set/append/clear), chunks (get/set/append/clear), consumed (get/set/clear), streaming (is/get/set/clear), and currentConversationId (get/set) |
| FR-23.3 | The five keys and their JSON-vs-string encoding remain identical to the pre-refactor behavior |
| FR-23.4 | `app.js` contains no `localStorage`, no `STORAGE_KEYS`, no `JSON.parse` / `JSON.stringify` related to cached state, and no helper functions for localStorage access. The only `JSON.*` calls that remain are for the `fetch` request body and SSE event parsing, which are not localStorage-related |
| FR-23.5 | The five caches keep their existing roles: `history` is the per-conversation message history (for instant load), `chunks` is the per-conversation streaming chunks (for page-refresh-during-stream resume), `consumed` is the per-conversation resume pointer, `streaming` is the per-conversation "is this conversation being streamed" flag, `currentConversationId` is the global "which conversation is open" key |

### FR-24: Frontend Asset Path

| ID | Requirement |
|----|-------------|
| FR-24.1 | The URL path `/static/...` references an actual `static` directory in the project tree |
| FR-24.2 | `index.html` is served at `/` |
| FR-24.3 | The four frontend files (`index.html`, `styles.css`, `app.js`, `cache.js`) are served by FastAPI's `StaticFiles` mount at `/static` (for the three under `frontend/static/`) and the explicit route at `/` (for `index.html`) |

---

## Non-Functional Requirements

### NFR-1: Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_BASE_URL` | `https://api.minimax.chat/v1` | LLM API base URL |
| `ANTHROPIC_API_KEY` | `""` | API key for authentication |

### NFR-2: Performance

- Streaming latency: tokens delivered as generated, not batched
- Model: `minimax-3`
- Max output tokens per response: 16000
- Extended thinking is enabled with a 10000-token budget (visible answer has ~6000 tokens after reasoning)

### NFR-3: Reliability

- Invalid JSON in storage file: the file is renamed to `conversations.json.corrupt` and the application starts fresh (with a warning log)
- LLM generation errors are logged
- Storage directory auto-created if missing
- StreamJob continues in background when EventSource disconnects
- Concurrent writes to `conversations.json` within a process are serialized by a `threading.Lock` so the read-modify-write cycle never interleaves
- A process crash during a write leaves the previous on-disk file fully intact (atomic write via `<path>.tmp` + `os.replace`)

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
  "status": "none|pending|active|completed|failed",
  "chunks_count": "integer",
  "partial_content": "string|null"
}
```

The `status` string is the single contract: `"none"` when no `StreamJob` exists, `"active"` while the LLM is producing, `"completed"` when it finished normally, `"failed"` if the LLM errored, `"pending"` if a job exists but has not started streaming yet.

### StreamJob Schema (Backend)

```json
{
  "chunks": [{"chunk": "string", "type": "thinking|token", "message_id": "string"}],
  "chunk_queue": "asyncio.Queue",
  "messages": [{"role": "string", "content": "string"}],
  "status": "pending|active|completed|failed",
  "error": "string|null",
  "cancelled": "boolean"
}
```

`cancelled` is set by `clear_job` to signal the background task to stop cleanly.

### localStorage Schema (Frontend)

All `localStorage` access is encapsulated in [`frontend/static/cache.js`](../frontend/static/cache.js). The module exposes typed accessors; `app.js` does not touch `localStorage` directly. The five keys are:

| Key | Content |
|-----|---------|
| `chunks_{conv_id}` | JSON array of stream chunks `[{"chunk": str, "type": "thinking\|token", "message_id": str?}]` — `message_id` is emitted by the backend for debug tracing; the frontend does not depend on or consume it |
| `consumed_{conv_id}` | Integer position for resume (renamed from `pointer_{conv_id}`) |
| `streaming_{conv_id}` | `"true"` or `"false"` flag for active streaming |
| `history_{conv_id}` | JSON array of complete messages `[{role, content, thinking?}]` for fast load on switch/refresh |
| `currentConversationId` | UUID of the currently open conversation (global, not per-conversation) |
| `history_{conv_id}` | JSON array of complete messages `[{"role": "user"\|"assistant", "content": "...", "thinking": "..."?}]` |

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

- Use `streaming-markdown` (CDN) for streamed token rendering, preserving parser state across resume
- `marked.js` is not loaded — the streaming parser handles final-message rendering as well
- Thinking content: plain text only

---

## Dependencies

### Frontend (CDN scripts in `index.html`)

- `katex` — LaTeX math rendering
- `dompurify` — loaded but currently unused at two no-op call sites; see `TODO(U2 — known issue)` comments in `frontend/static/app.js`
- `streaming-markdown` — incremental markdown parser

Styles live in `frontend/static/styles.css`; app logic in `frontend/static/app.js`; `localStorage` access in `frontend/static/cache.js`. `index.html` lives at `frontend/index.html` and is served at `/` by FastAPI.

---

## Out of Scope (Future)

- RAG (Retrieval-Augmented Generation)
- File upload
- Authentication / API key management
- PostgreSQL storage (current: JSON file)
- Conversation memory buffer
- Cross-browser tab synchronization (cross-tab cache invalidation included)
- Conversation search/filter
- Stream backpressure handling
- Cache expiration / TTL
- Cache size limits
- Preloading caches for multiple conversations
