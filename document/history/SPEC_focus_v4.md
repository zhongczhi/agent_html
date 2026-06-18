# Conversation History Frontend Cache - Specification

## Overview

Add a localStorage-based cache for full conversation message lists. This is separate from the existing streaming chunks cache. The history cache persists across page refreshes, is invalidated only on conversation delete, and grows via append on new messages.

---

## Functional Requirements

### FR-HC-1: Conversation History Cache

| ID | Requirement |
|----|-------------|
| FR-HC-1.1 | Conversation message lists are cached in localStorage on first load |
| FR-HC-1.2 | `loadConversation` reads from cache if available, otherwise fetches from backend |
| FR-HC-1.3 | Cache persists across page refreshes |
| FR-HC-1.4 | Cache is invalidated when user deletes a conversation |

### FR-HC-2: Cache Update Strategy

| ID | Requirement |
|----|-------------|
| FR-HC-2.1 | When user sends a message, append `{role: 'user', content}` to history cache |
| FR-HC-2.2 | When streaming completes, append `{role: 'assistant', content, thinking?}` to history cache |
| FR-HC-2.3 | Cache grows via append only — no full replacement after new exchanges |

---

## Interface Requirements

### localStorage Schema (Frontend)

| Key | Content |
|-----|---------|
| `history_{conv_id}` | JSON array of messages `[{"role": "user"\|"assistant", "content": "...", "thinking": "..."?}]` |
| `chunks_{conv_id}` | JSON array of stream chunks `[{"chunk": str, "type": "thinking\|token"}]` (existing) |
| `pointer_{conv_id}` | Integer position for resume (existing) |
| `streaming_{conv_id}` | Boolean flag for active streaming (existing) |

Note: `chunks_{conv_id}` caches streaming chunks (individual tokens during SSE). `history_{conv_id}` caches the full message list (complete messages after API response). These are separate caches with different purposes and formats.

---

## Data Flow

### Load Conversation (Cache-First)

```
loadConversation(convId)
├── localStorage['history_' + convId] exists?
│   ├── Yes → parse JSON, render directly
│   └── No  → fetch /api/chat/history/{convId}
│             → store in localStorage['history_' + convId]
│             → render
```

### Send Message → Cache Update

```
sendMessage()
├── user sends message
│   → append {role: 'user', content} to history cache
│   → POST /api/chat/stream
│   → receive SSE stream
│   → on stream end:
│       → append {role: 'assistant', content, thinking?} to history cache
```

### Delete Conversation → Cache Invalidation

```
deleteConversation(convId)
→ remove localStorage['history_' + convId]
→ remove localStorage['chunks_' + convId]      (existing)
→ remove localStorage['pointer_' + convId]     (existing)
→ remove localStorage['streaming_' + convId]  (existing)
```

### Page Refresh

```
init()
├── loadConversationList()
├── currentConversationId exists?
│   ├── Yes → checkStreamStatus()
│   │         ├── streaming → resume stream (existing behavior)
│   │         └── not streaming → loadConversation() → reads from cache if exists
│   └── No → empty state
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

### Modified Functions

| Function | Change |
|----------|--------|
| `loadConversation(convId)` | Check history cache before fetching; store fetched result in cache |
| `sendMessage()` | After user sends, append to history cache; after stream ends, append assistant message |
| `deleteConversation(convId)` | Remove `history_{convId}` from localStorage |
| `switchConversation(convId)` | No change — calls `loadConversation` which is cache-first |
| `init()` | No change — calls `loadConversation` which is cache-first |

### Unchanged Functions

- `checkStreamStatus()` — continues to use `chunks_{convId}` for stream resume
- `processStreamResponse()` — continues to cache individual chunks in `chunks_{convId}`
- `resumeStreamFromPosition()` — continues to use `chunks_{convId}` + `pointer_{convId}`

---

## Files to Modify

| File | Change |
|------|--------|
| `frontend/index.html` | Add `HISTORY` to STORAGE_KEYS; modify `loadConversation` for cache-first; modify `sendMessage` to update history cache; modify `deleteConversation` to clear history cache |

---

## Acceptance Criteria

- [ ] `loadConversation` returns cached data without network request when cache exists
- [ ] `loadConversation` fetches from backend and populates cache when no cache exists
- [ ] Sending a message appends user message to history cache immediately
- [ ] Streaming completion appends assistant message to history cache
- [ ] Deleting a conversation removes `history_{convId}` from localStorage
- [ ] Page refresh reads from history cache if available
- [ ] History cache and chunks cache use separate localStorage keys and serve different purposes

---

## Out of Scope

- Backend changes (this is frontend-only)
- Cache expiration / TTL
- Cache size limits
- Preloading caches for multiple conversations
