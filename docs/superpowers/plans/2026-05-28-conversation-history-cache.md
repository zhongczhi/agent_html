# Conversation History Frontend Cache - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add localStorage-based cache for conversation history, separate from streaming chunks cache.

**Architecture:** Cache-first load strategy: `loadConversation` reads from `history_{convId}` cache if present, otherwise fetches from backend. Cache grows by appending on send/stream-end. Invalidated only on delete.

**Tech Stack:** Vanilla JS, localStorage (no new dependencies)

---

## File Map

- **Modify:** `frontend/index.html` — single file, all changes

---

## Task 1: Add HISTORY to STORAGE_KEYS

**Files:** `frontend/index.html:603-607`

- [ ] **Step 1: Add HISTORY key to STORAGE_KEYS constant**

Locate the `STORAGE_KEYS` constant around line 603 and add `HISTORY`:

```javascript
const STORAGE_KEYS = {
    CHUNKS: (convId) => `chunks_${convId}`,
    POINTER: (convId) => `pointer_${convId}`,
    STREAMING: (convId) => `streaming_${convId}`,
    HISTORY: (convId) => `history_${convId}`  // NEW
};
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add HISTORY localStorage key for conversation cache"
```

---

## Task 2: Add history cache helper functions

**Files:** `frontend/index.html` — insert after `isStreamingForConv` helper (around line 673)

- [ ] **Step 1: Add four helper functions after `setStreamingForConv`**

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

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add history cache helper functions"
```

---

## Task 3: Add renderMessagesFromCache helper

**Files:** `frontend/index.html` — insert after `loadConversation` function (around line 795)

- [ ] **Step 1: Add `renderMessagesFromCache` after `loadConversation`**

```javascript
// Render messages from cache (or fetched data)
function renderMessagesFromCache(messages) {
    messagesContainer.innerHTML = '';

    for (const msg of messages) {
        if (msg.role === 'assistant') {
            const messageDiv = addMessage(msg.role, '');

            // If there's thinking content, add it
            if (msg.thinking) {
                const thinkingContent = messageDiv.querySelector('.thinking-content');
                if (thinkingContent) {
                    thinkingContent.textContent = msg.thinking;
                    updateThinkingDisplay(messageDiv);
                }
            }

            // Render markdown for content
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

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add renderMessagesFromCache helper"
```

---

## Task 4: Modify loadConversation for cache-first

**Files:** `frontend/index.html:754-795` — replace existing `loadConversation`

- [ ] **Step 1: Replace loadConversation with cache-first version**

Replace the existing `loadConversation` function body (lines 754-795) with:

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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): make loadConversation cache-first with fallback to backend"
```

---

## Task 5: Modify sendMessage to append user message to history cache

**Files:** `frontend/index.html:948-1022` — in `sendMessage`, after adding user message to UI

- [ ] **Step 1: Add cache append after user message is added**

Locate in `sendMessage()` after this line:
```javascript
addMessage('user', message);
```

Add after it:
```javascript
// Cache user message in history cache
appendToHistoryCache(currentConversationId, { role: 'user', content: message });
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): append user message to history cache on send"
```

---

## Task 6: Modify processStreamResponse to append assistant message on stream end

**Files:** `frontend/index.html:1024-1152` — in `processStreamResponse`, on `data.end`

- [ ] **Step 1: Find the `data.end` handler and add history cache append**

Locate the `data.end` block in `processStreamResponse` (around lines 1112-1145). After the final cleanup and before `return`, add:

```javascript
// Append assistant message to history cache
appendToHistoryCache(convId, {
    role: 'assistant',
    content: rawContent.trim(),
    thinking: thinkingContent || undefined
});
```

The `thinkingContent` variable should be extracted from the assistant message's `.thinking-content` element at that point. Make sure the variable is accessible — it should be defined in the outer scope of the `processStreamResponse` function as `let thinkingContent = '';` and updated in the thinking block.

- [ ] **Step 2: Ensure thinkingContent variable exists in processStreamResponse**

Check the function beginning — if `let thinkingContent = '';` doesn't exist, add it alongside the other state variables (`rawContent`, `assistantMessage`, etc.).

- [ ] **Step 3: Update thinkingContent in the thinking block**

In the `data.type === 'thinking'` block (around lines 1069-1075), ensure `thinkingContent` is being accumulated:
```javascript
} else if (data.type === 'thinking') {
    const thinkingElement = assistantMessage.querySelector('.thinking-content');
    if (thinkingElement) {
        thinkingElement.textContent += data.chunk;
        thinkingContent += data.chunk;  // Add this line
        updateThinkingDisplay(assistantMessage);
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): append assistant message to history cache on stream end"
```

---

## Task 7: Modify deleteConversation to clear history cache

**Files:** `frontend/index.html:929-946` — in `deleteConversation`

- [ ] **Step 1: Add clearHistoryCache before the DELETE request**

Find the `deleteConversation` function. Before the `fetch` call, add the history cache clear:

```javascript
// Clear all caches including history
clearHistoryCache(convId);
localStorage.removeItem(STORAGE_KEYS.CHUNKS(convId));
localStorage.removeItem(STORAGE_KEYS.POINTER(convId));
localStorage.removeItem(STORAGE_KEYS.STREAMING(convId));
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): clear history cache on conversation delete"
```

---

## Task 8: Verify and run tests

- [ ] **Step 1: Run existing tests**

```bash
cd backend && pytest tests/ -v
```

- [ ] **Step 2: Test manually in browser**

1. Open the chat app
2. Send a message — verify history cache is created in localStorage
3. Refresh the page — verify conversation loads from cache without network request
4. Send another message — verify history cache grows (two messages)
5. Delete conversation — verify history cache is removed

---

## Summary of Commits

| # | Message |
|---|---------|
| 1 | feat(frontend): add HISTORY localStorage key for conversation cache |
| 2 | feat(frontend): add history cache helper functions |
| 3 | feat(frontend): add renderMessagesFromCache helper |
| 4 | feat(frontend): make loadConversation cache-first with fallback to backend |
| 5 | feat(frontend): append user message to history cache on send |
| 6 | feat(frontend): append assistant message to history cache on stream end |
| 7 | feat(frontend): clear history cache on conversation delete |
