# Conversation History Frontend Cache - Design

## Overview

Add a localStorage-based cache for full conversation message lists, separate from the existing streaming chunks cache.

---

## Architecture Decisions

### 1. Two-Cache Architecture

**Design Choice:** Maintain two separate localStorage caches with distinct purposes.

| Cache | Key | Content | Purpose | Lifecycle |
|-------|-----|---------|---------|-----------|
| **History Cache** | `history_{convId}` | Complete messages `[{role, content, thinking?}]` | Fast load on conversation switch/refresh | Persists until explicit delete |
| **Chunks Cache** | `chunks_{convId}` | Stream chunks `[{chunk, type}]` | Stream resume after page refresh | Cleared after stream completes |

**Rationale:** The history cache stores the full message list — the "done" state. The chunks cache stores the "in-flight" state during streaming. These have different update patterns (append-once vs. per-chunk updates) and different consumers (loadConversation vs. processStreamResponse). Keeping them separate avoids mixing concerns and makes invalidation clear.

**Alternative considered:** Store everything in one cache — rejected because history and chunks have different schemas and update frequencies.

### 2. Cache-First Load Strategy

**Design Choice:** `loadConversation` checks localStorage first, falls back to backend fetch.

```javascript
async function loadConversation(convId) {
    // Try cache first
    const cached = localStorage.getItem(STORAGE_KEYS.HISTORY(convId));
    if (cached) {
        const messages = JSON.parse(cached);
        renderMessages(messages);
        return;
    }

    // Fetch from backend
    const response = await fetch(`/api/chat/history/${convId}`);
    const data = await response.json();

    // Store in cache for next time
    localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(data.messages));

    renderMessages(data.messages);
}
```

**Rationale:** This gives instant load times for previously-visited conversations without hitting the network. The cache is populated on first load and grows with each new exchange.

### 3. Append-Only Cache Growth

**Design Choice:** Cache grows by appending new messages; no full replacement after partial loads.

```javascript
// On sendMessage: append user message
function appendToHistoryCache(convId, message) {
    const cache = JSON.parse(localStorage.getItem(STORAGE_KEYS.HISTORY(convId)) || '[]');
    cache.push(message);
    localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(cache));
}

// On stream end: append assistant message
function appendToHistoryCache(convId, message) {
    const cache = JSON.parse(localStorage.getItem(STORAGE_KEYS.HISTORY(convId)) || '[]');
    cache.push(message);
    localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(cache));
}
```

**Rationale:** Appending is efficient (no re-fetch needed). Since conversations are append-only in the backend too, the cache stays in sync. If backend and cache ever diverge, the next full page load will resync from the backend.

### 4. Cache Invalidation on Delete

**Design Choice:** `deleteConversation` removes `history_{convId}` along with existing caches.

```javascript
async function deleteConversation(convId) {
    // Remove all caches
    localStorage.removeItem(STORAGE_KEYS.HISTORY(convId));
    localStorage.removeItem(STORAGE_KEYS.CHUNKS(convId));
    localStorage.removeItem(STORAGE_KEYS.POINTER(convId));
    localStorage.removeItem(STORAGE_KEYS.STREAMING(convId));

    // Delete from backend
    await fetch(`/api/chat/conversation/${convId}`, { method: 'DELETE' });

    // ... rest of UI update
}
```

**Rationale:** Delete is the only explicit invalidation point. No need to invalidate on page refresh since history is the source of truth.

---

## Data Flow Diagrams

### Load Conversation

```
┌─────────────────────────────────┐
│ loadConversation(convId)        │
└─────────────┬───────────────────┘
              │
              ▼
    ┌─────────────────┐
    │ HISTORY cache   │
    │ exists?         │
    └────┬───┬───────┘
         │   │
       Yes   No
         │   │
         ▼   ▼
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌────────┐        ┌──────────────┐
│ Parse  │        │ GET /api/    │
│ cache  │        │ chat/history │
└───┬────┘        │ /{convId}    │
    │             └──────┬───────┘
    │                    │
    │                    ▼
    │            ┌──────────────┐
    │            │ Store in     │
    │            │ HISTORY cache│
    │            └──────┬───────┘
    │                   │
    ▼                   ▼
┌─────────────────────────┐
│ renderMessages(messages) │
└─────────────────────────┘
```

### Send Message → Stream → Cache Update

```
sendMessage()
     │
     ▼
┌─────────────────────┐
│ append user msg to  │
│ HISTORY cache       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ POST /api/chat/     │
│ stream              │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ receive SSE stream  │
│ (chunks to display) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ on data.end:       │
│ append assistant    │
│ msg to HISTORY cache│
└─────────────────────┘
```

### Delete Conversation

```
deleteConversation(convId)
          │
          ▼
┌─────────────────────────────────────┐
│ localStorage.removeItem(HISTORY_)   │
│ localStorage.removeItem(CHUNKS_)    │
│ localStorage.removeItem(POINTER_)  │
│ localStorage.removeItem(STREAMING_)│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────┐
│ DELETE /api/chat/   │
│ conversation/{convId}│
└─────────────────────┘
```

---

## Implementation Details

### STORAGE_KEYS Constant

Add `HISTORY` key to existing `STORAGE_KEYS` object:

```javascript
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`,
    STREAMING: (convId) => `streaming_${convId}`,
    HISTORY: (convId) => `history_${convId}`  // NEW
};
```

### Helper Functions

```javascript
// Get history cache for a conversation
function getHistoryCache(convId) {
    const cached = localStorage.getItem(STORAGE_KEYS.HISTORY(convId));
    return cached ? JSON.parse(cached) : null;
}

// Set history cache for a conversation
function setHistoryCache(convId, messages) {
    localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(messages));
}

// Append a message to history cache
function appendToHistoryCache(convId, message) {
    const cache = getHistoryCache(convId) || [];
    cache.push(message);
    setHistoryCache(convId, cache);
}

// Clear history cache for a conversation
function clearHistoryCache(convId) {
    localStorage.removeItem(STORAGE_KEYS.HISTORY(convId));
}
```

### loadConversation(convId) — Modified

```javascript
async function loadConversation(convId) {
    try {
        // Check cache first
        const cached = getHistoryCache(convId);
        if (cached) {
            // Render from cache — no fetch needed
            renderMessagesFromCache(cached);
            return;
        }

        // Fetch from backend
        const response = await fetch(`/api/chat/history/${convId}`);
        const data = await response.json();

        // Store in cache
        setHistoryCache(convId, data.messages);

        // Render
        renderMessagesFromCache(data.messages);
    } catch (error) {
        console.error('Failed to load conversation:', error);
    }
}

// Render messages from cache (or fetched data)
function renderMessagesFromCache(messages) {
    messagesContainer.innerHTML = '';

    for (const msg of messages) {
        if (msg.role === 'assistant') {
            const messageDiv = addMessage(msg.role, '');
            const thinkingContent = messageDiv.querySelector('.thinking-content');
            if (thinkingContent && msg.thinking) {
                thinkingContent.textContent = msg.thinking;
                updateThinkingDisplay(messageDiv);
            }
            const contentDiv = messageDiv.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.innerHTML = renderMarkdown(msg.content || '');
            }
        } else {
            addMessage(msg.role, msg.content);
        }
    }
}
```

### sendMessage() — Modified (cache update)

In `sendMessage()`, after adding the user message to UI:

```javascript
// Cache user message in history cache
appendToHistoryCache(currentConversationId, { role: 'user', content: message });
```

When streaming completes (in `processStreamResponse` on `data.end`):

```javascript
// Append assistant message to history cache
appendToHistoryCache(convId, {
    role: 'assistant',
    content: rawContent.trim(),
    thinking: thinkingContent || undefined
});
```

### deleteConversation(convId) — Modified (cache cleanup)

```javascript
async function deleteConversation(convId) {
    if (!confirm('Delete this conversation?')) return;

    try {
        // Clear all caches including history
        clearHistoryCache(convId);
        localStorage.removeItem(STORAGE_KEYS.CHUNKS(convId));
        localStorage.removeItem(STORAGE_KEYS.POINTER(convId));
        localStorage.removeItem(STORAGE_KEYS.STREAMING(convId));

        await fetch(`/api/chat/conversation/${convId}`, { method: 'DELETE' });

        delete conversations[convId];
        const item = document.querySelector(`.conversation-item[data-id="${convId}"]`);
        if (item) item.remove();

        if (convId === currentConversationId) {
            await startNewChat();
        }
    } catch (error) {
        console.error('Failed to delete conversation:', error);
    }
}
```

---

## Edge Cases

### Empty Cache on First Load

If `history_{convId}` doesn't exist (first time loading a conversation), `getHistoryCache()` returns null, so `loadConversation` falls through to fetch from backend and populates the cache.

### Cache Exists But Conversation Was Deleted on Another Tab

If the user deletes a conversation in another tab, the local cache remains. On next load, the backend will return an empty messages array (or the conversation won't exist). The current design does NOT handle this cross-tab invalidation — it's out of scope.

### Resume Streaming After Refresh

The stream resume flow (`checkStreamStatus` → `resumeStreamFromPosition`) continues to use `chunks_{convId}` and `pointer_{convId}` as before. It does NOT interact with `history_{convId}`. After resume completes, the new assistant message is appended to `history_{convId}` via `appendToHistoryCache`.

### Switching Conversations Mid-Stream

When user switches conversations while a stream is active in another conversation, only `chunks_{convId}` and `pointer_{convId}` are updated for the streaming conversation. The other conversation's `history_{convId}` is untouched. When user switches back, `loadConversation` reads from cache.

---

## Testing Checklist

### Cache Read
- [ ] `loadConversation` returns cached data without network request when cache exists
- [ ] `loadConversation` fetches from backend when no cache exists
- [ ] Cache is populated after first fetch

### Cache Update
- [ ] Sending a message appends `{role: 'user', content}` to history cache
- [ ] Streaming completion appends `{role: 'assistant', content, thinking?}` to history cache
- [ ] Multiple message exchanges accumulate in history cache

### Cache Invalidation
- [ ] Deleting a conversation removes `history_{convId}` from localStorage
- [ ] Deleting removes all related caches (history + chunks + pointer + streaming)

### Persistence
- [ ] Page refresh reads from history cache if available
- [ ] History cache survives refresh (same key, same data)
- [ ] After refresh + stream resume, new message appended to existing history cache

### Separation of Concerns
- [ ] History cache and chunks cache use different keys
- [ ] History cache format differs from chunks cache format
- [ ] `loadConversation` uses history cache; `resumeStream` uses chunks cache

---

## Files Modified

### `frontend/index.html`

| Change | Description |
|--------|-------------|
| `STORAGE_KEYS.HISTORY` | Add `history_{convId}` key generator |
| `getHistoryCache()` | New helper — read history cache |
| `setHistoryCache()` | New helper — write history cache |
| `appendToHistoryCache()` | New helper — append single message |
| `clearHistoryCache()` | New helper — remove history cache |
| `renderMessagesFromCache()` | New helper — render from cached messages |
| `loadConversation()` | Modify — cache-first with fallback |
| `sendMessage()` | Modify — call `appendToHistoryCache` for user + assistant |
| `deleteConversation()` | Modify — call `clearHistoryCache` |
| `processStreamResponse()` | Modify — append assistant message to history cache on `data.end` |
