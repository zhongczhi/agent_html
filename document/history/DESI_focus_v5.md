# Stream Resume on Second Refresh + Batch Delete + Confirmation Modal + Resurrection Fix + Scroll-Pin - Design

This iteration bundles six changes:

1. **Stream Resume on Second Refresh** (Parts A–B) — bug fix so that any number of refreshes during streaming preserve the partial assistant message.
2. **Custom Confirmation Modal** (Part C) — replaces browser-native `confirm()` with a themed modal centered on the page.
3. **Batch Delete** (Part D) — replaces the sidebar-header ≡ button with a batch-delete entry point; selection mode with checkboxes; batch confirm via the new modal.
4. **Streaming-Conversation Resurrection Fix** (Part E) — backend fix so a conversation deleted mid-stream stays deleted when its background LLM task finishes.
5. **Smart Auto-Scroll During Streaming** (Part F) — pin to the newest line by default, but stop forcing the scroll if the user scrolled up.
6. **Block Sending Messages in Selection Mode** (Part G) — add an early-return guard at the top of `sendMessage()` so the Send button and Enter key are both blocked while the user is selecting conversations for batch deletion.

### Reproduction

1. User sends a message; LLM starts streaming.
2. User refreshes the page once. Partial assistant message is rendered from the chunk cache; streaming continues. ✓
3. User refreshes the page a second time while still streaming. Partial assistant message is gone. ✗

### Root Cause

Two compounding issues in `frontend/index.html`:

**Issue 1 — `resumeStreamFromPosition` catch block clears the streaming flag on transient errors.** In Chromium, a fetch aborted by page navigation throws `TypeError: network error` — not `AbortError` like an explicit `AbortController.abort()` does. The catch block checked `error.name === 'AbortError'`, fell through to the `else` branch on a navigation-abort, and called `setStreamingForConv(currentConversationId, false)`. This wiped the streaming flag in `localStorage` from the dying JS context.

**Issue 2 — `init` unconditionally falls back to `loadConversation` whenever `status !== true`.** `loadConversation` calls `renderMessagesFromCache`, which does `messagesContainer.innerHTML = ''` and re-renders from the history cache. The history cache only contains `[user]` until `data.end` has been processed, so the partial assistant message rendered earlier by `renderCachedChunks` is wiped.

### Sequence

```
REFRESH 1 page load
  └─ checkStreamStatus → resumeStreamFromPosition
       ├─ renderCachedChunks → DOM has [user, assistant(partial)]
       ├─ fetch stream → 200 OK, chunks arrive, DOM updates
       └─ await processStreamResponse...   ← BLOCKED here

[User clicks refresh] (REFRESH 2)

REFRESH 1's fetch is aborted by page navigation
  └─ Chromium rejects with TypeError: network error (NOT AbortError)
  └─ resumeStreamFromPosition catch block runs:
       ├─ error.name !== 'AbortError' → else branch
       ├─ setStreamingForConv(convId, false)   ← streaming flag wiped
       └─ return false

REFRESH 2 page load
  └─ init → checkStreamStatus
       └─ getStreamingForConv === 'false' → early return (no resume attempted)
  └─ init → loadConversation
       └─ renderMessagesFromCache(historyCache = [user])
            └─ messagesContainer.innerHTML = ''   ← partial assistant wiped
            └─ DOM now has [user only]
```

---

## Part A: Preserve Streaming Flag on Fetch Errors

### A.1 The Fix

In `resumeStreamFromPosition`'s catch block, drop the `AbortError`-specific branch entirely. Log the error and return false; do not touch the streaming flag or the badge.

```javascript
} catch (error) {
    console.error('Stream error:', error);
    return false;
}
```

### A.2 Why This Is Safe

The streaming flag has exactly two responsibilities:

1. `init` / `checkStreamStatus` decides whether to attempt a resume on page load.
2. `loadConversationList` derives the sidebar "Streaming" badge.

For (1), we *want* a transient fetch failure to leave the flag set so the next refresh can retry. The flag should only be cleared on `data.end` (the normal completion path) or on explicit user action (`startNewChat`, `deleteConversation`).

For (2), the badge accurately reflects "this conversation has an in-flight stream we know about." Leaving the badge set after a transient failure is honest UX — the backend is still generating, and the user can refresh to retry.

### A.3 Why We Can't Distinguish Page-Abort from Other Errors

`AbortController.abort()` throws `DOMException` with `name === 'AbortError'`. But a fetch killed by browser navigation (refresh, close tab, back button) is rejected with a generic `TypeError: network error`. There is no reliable way to tell the two apart in user code, so the conservative choice is to treat both as transient: log and return, but do not mutate the resume state.

---

## Part B: Don't Fall Back to `loadConversation` After Resume Attempt

### B.1 The Fix

In `init`, gate the `loadConversation` fallback on the streaming flag being set at the start of the decision:

```javascript
async function init() {
    await loadConversationList();

    if (currentConversationId) {
        if (isStreamingForConv(currentConversationId)) {
            await checkStreamStatus();
            return;
        }
        await loadConversation(currentConversationId);
    } else {
        messagesContainer.classList.add('empty');
    }
}
```

### B.2 Why This Works

`checkStreamStatus` calls `resumeStreamFromPosition` which **always** renders cached chunks into the DOM before issuing the network fetch:

1. `renderMessagesFromCache(historyCache)` — renders any complete messages from history (the user message, possibly earlier exchanges).
2. `renderCachedChunks(convId)` — creates the assistant message and applies any cached in-flight chunks to it.
3. Connects to the stream and processes new chunks.

After step 2 the DOM already contains the partial assistant message. Whatever happens in step 3 (success, fetch error, conversation switch), the partial content stays in the DOM. `init` must not call `loadConversation` afterwards, because `loadConversation` would `messagesContainer.innerHTML = ''` and re-render only what is in history cache (which is missing the in-progress assistant message until `data.end` is processed).

The fallback is only correct when `resumeStreamFromPosition` was never called — i.e., the streaming flag was `'false'` to begin with. In that case the conversation is not actively streaming and `loadConversation`'s history-cache render is the right thing to show.

### B.3 Interaction With Part A

The two changes are complementary:

- Part A alone: the streaming flag survives a transient error, so the next refresh's `init` sees it `'true'` and calls `checkStreamStatus`, which renders cached chunks. ✓
- Part B alone: even if a previous refresh's catch block had cleared the flag, the current refresh sees the flag `'true'` at its start and renders cached chunks. ✓
- Both: any refresh during streaming renders cached chunks and never falls back to `loadConversation`, regardless of what previous refreshes did to the flag.

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| First refresh during streaming | Streaming flag `'true'` → `checkStreamStatus` → `resumeStreamFromPosition` renders cached chunks + processes live stream. DOM intact. |
| Second (or Nth) refresh during streaming | Same as first. Streaming flag survives (Part A). `init` doesn't fall back (Part B). DOM shows cached chunks + live tail. |
| Stream completes naturally between refreshes | `data.end` handler clears flag and chunks cache. Next refresh sees flag `'false'` → falls back to `loadConversation` → renders full history. No regression. |
| Refresh after stream completes | Flag `'false'` → `loadConversation` → history cache. No regression. |
| New conversation in a fresh tab | Flag `'true'` immediately after send → resume path. If stream is fast and already done, the resume gets the end marker on first read; flag cleared; UI correct. |
| User switches conversation mid-stream | `switchConversation` aborts `currentAbortController` → fetch throws `AbortError` → catch block logs and returns false → next refresh of the original conversation can resume from chunk cache. Streaming flag preserved. |
| User clicks "New Chat" mid-stream | `startNewChat` aborts controller + clears `currentConversationId` + removes `currentConversationId` from `localStorage`. No resume possible for the previous conversation from this tab; if the user navigates back to it via the sidebar, fresh state. Streaming flag on the previous conversation remains `'true'` in `localStorage` but its UI is gone. (Existing behavior — not changed.) |
| Genuine stream failure (backend 404, etc.) | Fetch rejects with non-Abort error. Streaming flag preserved → next refresh retries. If retry also fails, the user sees the cached partial content indefinitely until they refresh again or backend recovers. Acceptable: the alternative (current behavior) is silently deleting the partial content. |

---

## Files Modified

| File | Change |
|------|--------|
| `frontend/index.html` | `init` (Part B): gate `loadConversation` fallback on `isStreamingForConv`; `resumeStreamFromPosition` catch block (Part A): drop the `AbortError` branch, only log + return false |
| `document/SPEC_focus.md` | New — minimal bug spec |
| `document/DESI_focus.md` | New — this document |

No backend, schema, or API changes.

---

## Testing

### Automated (Playwright reproduction script — not committed)

- Before fix: 2nd refresh during streaming → `domAssistant: 0` (bug reproduced).
- After fix: 2nd refresh → `domAssistant: 1`, partial content visible.
- After fix: 5 consecutive refreshes during streaming → all show partial content; idempotent.
- After fix: stream completion mid-refresh → final state has full assistant message in history cache.

### Backend tests

- `pytest backend/tests/` → 18/18 pass. Frontend-only change; backend untouched.

### Manual regression checklist

- [ ] Send a message → response streams → works.
- [ ] Refresh during streaming once → partial content visible → works.
- [ ] Refresh during streaming twice → partial content still visible → fixed.
- [ ] Refresh 5 times during streaming → partial content visible on every refresh → fixed.
- [ ] Refresh after streaming completes → full conversation from history cache → no regression.
- [ ] Send a new message in an existing conversation → works, no leftover partial content from previous turn.
- [ ] Delete a conversation mid-stream → conversation disappears from sidebar, stream cancelled.
- [ ] Switch conversation mid-stream → switch succeeds, original conversation can be resumed later.

---

## Out of Scope (Part 1 / Bug Fix)

- Backend changes (none required).
- New localStorage keys or schema.
- New API endpoints.
- Refactoring of `resumeStreamFromPosition` beyond the two-line catch-block change.
- Distinguishing page-navigation aborts from other fetch failures at the JS level (not possible with current browser APIs).
- Cross-tab synchronization.
- Cache expiration / TTL.

---

## Part C: Custom Confirmation Modal

### C.1 Component Contract

`showConfirmModal({ title, message, confirmText, cancelText, danger }) → Promise<boolean>`

- `title`: string (required) — modal heading.
- `message`: string (required) — body text (may contain newlines).
- `confirmText`: string (default `'Confirm'`) — label of the confirm button.
- `cancelText`: string (default `'Cancel'`) — label of the cancel button.
- `danger`: boolean (default `false`) — when `true`, the confirm button uses the danger color scheme (red accent).
- Returns a Promise resolving to `true` if the user confirms, `false` if they cancel (backdrop click, Escape, or cancel button).

### C.2 DOM Structure

A single reusable modal element appended to `<body>` and shown/hidden:

```html
<div class="modal-backdrop" hidden>
    <div class="modal-panel" role="dialog" aria-modal="true">
        <h3 class="modal-title"></h3>
        <p class="modal-message"></p>
        <div class="modal-actions">
            <button class="modal-cancel"></button>
            <button class="modal-confirm"></button>
        </div>
    </div>
</div>
```

Only one modal exists at a time. Calling `showConfirmModal` while one is open is not expected (single confirmation flow).

### C.3 Styling

CSS uses existing theme tokens:

- Backdrop: `position: fixed; inset: 0; background: rgba(0, 0, 0, 0.6); display: flex; align-items: center; justify-content: center; z-index: 1000`.
- Panel: `background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; min-width: 320px; max-width: 480px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);`.
- Title: `color: var(--text-primary); font-size: 18px; font-weight: 600; margin-bottom: 12px`.
- Message: `color: var(--text-secondary); font-size: 14px; line-height: 1.5; margin-bottom: 24px; white-space: pre-line`.
- Cancel button: secondary style (border, muted text).
- Confirm button: `background: var(--accent-cyan); color: var(--bg-primary)` for primary; `background: #ef4444; color: white` for `danger: true`.

### C.4 Behavior

- Backdrop click → cancel (resolve `false`).
- Escape key → cancel (resolve `false`).
- Cancel button click → cancel (resolve `false`).
- Confirm button click → confirm (resolve `true`).
- Focus is moved to the confirm button when the modal opens (or to the cancel button when `danger: true`, so accidental Enter doesn't delete).
- After resolution, the modal is hidden and event listeners on the backdrop and Escape are removed.

### C.5 Edge Cases

| Scenario | Handling |
|----------|----------|
| User presses Enter while modal is open | Default form-submit behavior avoided; confirm button gets a click. For `danger: true`, focus is on cancel so Enter cancels instead. |
| User double-clicks confirm | First click resolves the Promise; modal hides immediately. Second click is a no-op (modal is `hidden`). |
| Page refresh while modal is open | Modal element is gone with the page. No Promise resolution (caller never sees the result). This is acceptable — caller should not assume confirmation persists across refreshes. |
| Multiple concurrent confirmations | Not supported. Caller awaits the Promise before issuing another. |

### C.6 Where It's Wired

- `deleteConversation(convId)` — single-item delete. `danger: true`, message: `"This will permanently delete this conversation."`.
- `confirmBatchDelete(selectedIds)` — batch delete. `danger: true`, message lists titles (or count if more than 5).
- No other call sites for now. The helper is exposed as a generic component for future use.

---

## Part D: Batch Delete

### D.1 UI States

The sidebar has two states: **normal** and **selection**.

**Normal state** (current header):
```
+ New Chat            [🗑 Batch Delete]
```
The ≡ button is replaced by a "Batch Delete" button with a trash icon.

**Selection state**:
```
Delete (3)                  [Cancel]
```
When N > 0, button label is `Delete (N)`. When N === 0, button is disabled and labeled `Delete (0)` or just `Delete`.

### D.2 Conversation Item — Normal

Current (preserved):
```
[Title text]                                    ×
[Streaming badge if streaming]
```

### D.3 Conversation Item — Selection Mode

```
[☐] [Title text]
[☑] [Title text]            ← selected
```

- A leading checkbox appears before the title.
- Clicking the checkbox toggles selection.
- Clicking anywhere else on the row (outside the checkbox) also toggles selection for ergonomic large-target selection.
- The `×` per-item delete button is **hidden** in selection mode (only batch delete is allowed).
- Selected items get a subtle highlight (e.g., `background: var(--bg-tertiary)`).

### D.4 State Management

A new module-level variable:

```javascript
let selectionMode = false;
let selectedConvIds = new Set();
```

- Entering selection mode: `selectionMode = true; selectedConvIds = new Set();` then re-render the sidebar list with checkboxes.
- Exiting selection mode: `selectionMode = false; selectedConvIds.clear();` then re-render without checkboxes.
- Toggling selection: `selectedConvIds.has(id) ? selectedConvIds.delete(id) : selectedConvIds.add(id);` then update header (count + button enabled state) without full re-render (or full re-render is fine — simpler).
- The sidebar re-render path (`loadConversationList` → `addConversationToList`) needs to know whether we're in selection mode and whether each item is selected.

### D.5 Re-render Strategy

`addConversationToList(conv)` becomes selection-aware:

```javascript
function addConversationToList(conv) {
    const div = document.createElement('div');
    div.className = 'conversation-item';
    if (conv.conversation_id === currentConversationId) div.classList.add('active');
    if (selectionMode && selectedConvIds.has(conv.conversation_id)) div.classList.add('selected');

    if (selectionMode) {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'select-checkbox';
        checkbox.checked = selectedConvIds.has(conv.conversation_id);
        checkbox.addEventListener('click', (e) => e.stopPropagation());
        checkbox.addEventListener('change', () => toggleSelection(conv.conversation_id));
        div.appendChild(checkbox);
    }

    const titleSpan = document.createElement('span');
    titleSpan.className = 'title';
    titleSpan.textContent = conv.title || 'New conversation';
    div.appendChild(titleSpan);

    if (!selectionMode) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = '×';
        deleteBtn.onclick = (e) => { e.stopPropagation(); deleteConversation(conv.conversation_id); };
        div.appendChild(deleteBtn);
    }

    if (!selectionMode && isStreamingForConv(conv.conversation_id)) {
        const badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'Streaming';
        div.appendChild(badge);
    }

    div.onclick = () => {
        if (selectionMode) {
            toggleSelection(conv.conversation_id);
        } else {
            switchConversation(conv.conversation_id);
        }
    };

    conversationList.appendChild(div);
}
```

### D.6 Header Update Strategy

Two header layouts (toggled in `setSelectionMode`):

- Normal: `+ New Chat` button + `Batch Delete` button.
- Selection: `Delete (N)` button (disabled when N===0) + `Cancel` button.

Rather than maintaining two separate DOM trees, the implementation can use a single header with conditional content via a function `renderSidebarHeader()` that builds the inner HTML based on `selectionMode` and `selectedConvIds.size`.

### D.7 Batch Delete Flow

```
User clicks "Batch Delete"
   │
   ▼
setSelectionMode(true)
   ├─ selectionMode = true
   ├─ selectedConvIds = new Set()
   └─ renderSidebarList()  + renderSidebarHeader()
       │
       ▼
User toggles checkboxes / clicks rows
   │
   ▼
toggleSelection(id)
   ├─ add or remove id from selectedConvIds
   └─ renderSidebarHeader()  (count + button enabled)
       │
       ▼
User clicks "Delete (N)"
   │
   ▼
confirmBatchDelete(Array.from(selectedConvIds))
   ├─ showConfirmModal({title, message, danger: true})
   │     ├─ User cancels → return
   │     └─ User confirms → proceed
   ├─ For each id in selectedIds:
   │     ├─ DELETE /api/chat/conversation/{id}
   │     ├─ clearHistoryCache(id)
   │     ├─ removeItem chunks/pointer/streaming
   │     └─ delete conversations[id]
   ├─ If currentConversationId in selectedIds → startNewChat()
   ├─ setSelectionMode(false)
   └─ await loadConversationList()  (refresh sidebar)
```

### D.8 Edge Cases

| Scenario | Handling |
|----------|----------|
| Enter selection mode with 0 conversations | Selection mode renders, but list is empty; Delete button disabled |
| Enter selection mode with 1 conversation | User can select and batch-delete that one item (effectively a single delete) |
| Delete the active conversation | After deletion, `startNewChat()` clears current conversation and resets input |
| Delete all conversations | Sidebar becomes empty; selection mode exits |
| Click "New Chat" while in selection mode | Selection mode exits; `startNewChat` proceeds |
| Switch conversation while in selection mode | The row-click handler routes to selection toggle (not switch) — switching is disabled in selection mode |
| Refresh page while in selection mode | `selectionMode` is module-level, not persisted. On reload, normal state is restored. Acceptable. |
| Backend DELETE fails for one item | Other deletions proceed; failed item is logged but UI does not block. A toast or silent failure is acceptable; spec does not require per-item error UX. |

### D.9 Reused Code

- `deleteConversation(convId)` → single delete. After Part C, it uses the new modal.
- `startNewChat()` → reused as-is when the active conversation is deleted.
- `loadConversationList()` → reused as-is to refresh the sidebar after deletion.

### D.10 Testing Checklist

- [ ] Sidebar-header shows "Batch Delete" instead of ≡
- [ ] Clicking "Batch Delete" enters selection mode (checkboxes appear, header changes)
- [ ] Selecting 0 items keeps Delete button disabled
- [ ] Selecting 1+ items enables Delete button with the correct count
- [ ] Cancel exits selection mode without any deletion
- [ ] Delete opens the themed modal listing the count and titles
- [ ] Confirming deletes all selected; backend + localStorage cleared; sidebar refreshes
- [ ] If active conversation is deleted, a new empty chat is started
- [ ] Single-item delete (via `×`) still works and uses the themed modal
- [ ] Sidebar collapse/expand via the chat-header ≡ button still works

---

## Files Modified (combined)

| File | Change |
|------|--------|
| `frontend/index.html` | Part A (Bug fix in `resumeStreamFromPosition` catch + `init` fallback gate); Part C (new modal CSS + `showConfirmModal` helper); Part D (selection mode state + header + per-item rendering + batch delete flow + `deleteConversation` rewired to modal); Part G (early-return guard in `sendMessage()` when `selectionMode === true`) |
| `backend/chat/stream_manager.py` | Part E: `StreamJob.cancelled` flag; `clear_job` sets the flag before removing from registry |
| `backend/chat/service.py` | Part E: `generate_background` checks `job.cancelled` in the chunk loop and before `save_conversation`; bails out (no `mark_completed`, no save) if set |
| `backend/tests/test_chat_service.py` | Part E: regression test `test_generate_background_aborts_on_cancellation` — proves the resurrection bug is fixed |
| `document/SPEC_focus.md` | Bundles Parts 1–3 + Part 4 (resurrection fix) + Part 6 (block new chat in selection mode) |
| `document/DESI_focus.md` | Bundles Parts A–G |

---

## Part E: Fix for Streaming-Conversation Resurrection

### E.1 The Bug

Discovered while testing the batch-delete feature: if a user deletes a conversation while the background LLM task is still generating, the conversation **resurrects in backend storage** after the LLM finishes.

Reproduction:
1. Send a long prompt that triggers extended thinking.
2. Wait until several chunks have been received (so the LLM task is actively running).
3. Click `×` on the conversation, confirm in the modal. UI looks fine.
4. Wait ~60 seconds for the LLM to finish.
5. `GET /api/chat/conversations` returns the deleted conversation again — full user message + assistant response.

### E.2 Why It Happens

1. `backend/chat/routes.py:delete_conversation` calls `clear_job(conversation_id)` (removes the `StreamJob` from the in-memory registry) and `file_storage.delete_conversation(conversation_id)` (removes from `conversations.json`).
2. But `backend/chat/service.py:generate_background` was started as a fire-and-forget `asyncio.create_task` from `stream_chat`. Nothing stops it. It continues consuming from `self.chain.astream(...)`, calling `job.append_chunk(...)` on the orphaned `StreamJob` instance.
3. When the LLM finishes, `generate_background` runs:
   ```python
   job.mark_completed()
   file_storage.save_conversation(conversation_id, messages)
   ```
4. `file_storage.save_conversation` is "create-or-update": if the key isn't present, it creates the entry. The deleted conversation comes back to life with full message history.

### E.3 The Fix

Three small backend changes:

**`StreamJob.cancelled: bool = False`** — new flag, defaulting to `False`. Initialized in `__init__`.

**`clear_job`** — sets `job.cancelled = True` on the job before removing it from `STREAM_REGISTRY`:

```python
def clear_job(conversation_id: str) -> None:
    if conversation_id in STREAM_REGISTRY:
        STREAM_REGISTRY[conversation_id].cancelled = True
        del STREAM_REGISTRY[conversation_id]
```

**`generate_background`** — checks `job.cancelled` in two places:

```python
async for chunk in self.chain.astream(messages):
    # If the user deleted the conversation mid-stream, stop early and
    # do NOT call mark_completed or save_conversation — that would
    # resurrect the deleted conversation in storage.
    if job.cancelled:
        return
    # ... process chunk ...

# Defensive: if cancellation happened right as the LLM finished, do not save.
if job.cancelled:
    return

job.mark_completed()
# ... save ...
```

### E.4 Why Both Checks

- The mid-loop check stops processing new chunks early (saves a few lines of work).
- The pre-save check is defensive: covers the race where the LLM's final chunk arrives between the loop check and `save_conversation`, or where the LLM was already past its last chunk when `clear_job` ran. Either way, we never call `save_conversation` on a cancelled job.

### E.5 Why We Don't Cancel the asyncio Task

We considered calling `task.cancel()` on the background task itself. We didn't because:
- The task holds the LLM stream (`self.chain.astream(...)`) open. Cancelling the task aborts the read but the LLM may continue producing tokens server-side that go nowhere.
- `job.cancelled` is simpler, race-free, and the cleanup happens at the next natural checkpoint (chunk arrival or end of stream).
- The user's already-deleted conversation stays deleted; any further LLM work is wasted but harmless.

### E.6 Test Coverage

`backend/tests/test_chat_service.py::test_generate_background_aborts_on_cancellation`:
1. Pre-create a conversation with one user message (so `save_conversation` would be a real update, not a create).
2. Start a mocked `astream` that yields one chunk, then blocks on `asyncio.sleep(30)`.
3. Start `generate_background` as an asyncio task.
4. After the first chunk is processed, call `clear_job(conv_id)` and `task.cancel()`.
5. Assert: no assistant message appears in the conversation's stored messages.

### E.7 Edge Cases

| Scenario | Handling |
|----------|----------|
| User deletes right as LLM finishes (between last chunk and `save_conversation`) | The pre-save `if job.cancelled: return` catches it |
| User deletes before any chunks are received | The mid-loop check triggers on the very first chunk iteration |
| `clear_job` is called multiple times | Idempotent — flag is set once; subsequent calls are no-ops on the registry |
| Normal stream completion (no delete) | `cancelled` stays `False`; the existing `mark_completed` + `save_conversation` path runs unchanged |
| LLM errors mid-stream (`generate_background` except branch) | `mark_failed` runs as before; `save_conversation` is not called by this code path so cancellation is irrelevant |

---

## Part F: Smart Auto-Scroll During Streaming

### F.1 The Problem

`processStreamResponse` ran `messagesContainer.scrollTop = messagesContainer.scrollHeight` on every token chunk, unconditionally forcing the scroll to the bottom. If the user wanted to scroll up to re-read an earlier sentence while the response was still streaming, the next chunk would yank them back to the bottom.

### F.2 The Fix

Only auto-scroll if the user was already at (or within 50px of) the bottom BEFORE the chunk's DOM update. If they had scrolled up, leave their position alone. They re-pin by scrolling back down.

### F.3 The Subtle Bit — When to Capture the "Pinned" State

A naive implementation:

```javascript
// WRONG — checked after DOM update
[renderer, parser] = renderContent(contentDiv, data.chunk, renderer, parser);
if (isScrolledToBottom()) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
```

breaks because a single chunk can add more than 50px of height (a long paragraph, a code block, a table row). After the DOM update, `scrollHeight - scrollTop - clientHeight` equals the height of the new content — easily > 50 — and `isScrolledToBottom` returns false. The scroll stops being pinned even though the user never touched it.

The fix:

```javascript
// CORRECT — captured before DOM update
const wasPinnedToBottom = isScrolledToBottom();
// ... DOM update ...
[renderer, parser] = renderContent(contentDiv, data.chunk, renderer, parser);
if (wasPinnedToBottom) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
```

Now `wasPinnedToBottom` reflects the user's true intent at the moment the chunk arrived, before the chunk reshaped the scroll geometry.

### F.4 The Helper

```javascript
function isScrolledToBottom(threshold = 50) {
    return messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight <= threshold;
}
```

50px is a small tolerance — close enough to feel "at the bottom" without being so strict that a single pixel of mouse-wheel jitter un-pins the scroll.

### F.5 Scope — Only the Per-Chunk Scroll

Other scroll sites in the codebase are not modified:

| Line | Context | Why unchanged |
|------|---------|---------------|
| `renderCachedChunks` end | Resume replay of cached chunks | One-shot, not per-chunk. The user just refreshed the page and wants to see the latest content. |
| `sendMessage` after `addAssistantPlaceholder` | New message | One-shot per message. The user just sent a message and wants to see the assistant's response. |
| `addMessage` end | Any message add | One-shot per message. |

### F.6 Edge Cases

| Scenario | Handling |
|----------|----------|
| User at bottom, chunk arrives with content > 50px | `wasPinnedToBottom = true` (captured before), scroll restored to new bottom |
| User scrolled up to read earlier content, chunk arrives | `wasPinnedToBottom = false` (captured before), scroll position preserved |
| User scrolls back to bottom manually, next chunk arrives | `wasPinnedToBottom = true` again, scroll pinned |
| Page is shorter than clientHeight (no scroll possible) | `scrollHeight - clientHeight <= 0`, always pinned; scroll is a no-op but harmless |
| Resize of the messages container (window resize) mid-stream | The helper re-evaluates on every chunk, so the next chunk corrects any drift |
| Multiple rapid chunks | Each captures its own pinned state; works correctly even at high token rates |

### F.7 Testing

Playwright verification:

1. Send a prompt that generates a long, scrollable response.
2. Assert scroll is pinned to bottom (distance ≈ 0).
3. Programmatically scroll the messages container to the top.
4. Wait for more chunks to arrive.
5. Assert scrollTop is still at the top — scroll was NOT forced back.
6. Programmatically scroll back to the bottom.
7. Wait for more chunks.
8. Assert scroll is pinned to bottom again — re-pinned.

---

## Part G: Block Sending Messages in Selection Mode

### G.1 The Goal

While the user is in batch-deletion selection mode, both paths to sending a message — clicking the Send button and pressing Enter in the textarea — must be blocked. The user's pending batch-delete selection must not be disturbed.

### G.2 The Fix

Add an early-return guard at the top of `sendMessage()` in [frontend/index.html:1500](frontend/index.html#L1500):

```javascript
async function sendMessage() {
    // Sending messages is not allowed while in batch deletion selection
    // mode — bail without disturbing the user's pending selection.
    // Covers both the Send button click and Enter keypress (both route
    // through sendMessage()).
    if (selectionMode) return;

    const message = messageInput.value.trim();
    // ... rest unchanged
}
```

### G.3 Why One Guard Covers Both Interactions

Both user-facing paths to send converge on `sendMessage()`:

| Path | Handler | Calls |
|------|---------|-------|
| Send button click | `sendButton.addEventListener('click', sendMessage)` | `sendMessage()` |
| Enter in textarea | `messageInput.addEventListener('keydown', ...)` on Enter | `sendMessage()` |

Guarding `sendMessage()` blocks both with one line. There is no other path that sends a message.

### G.4 Why a Functional Block Only, Not Visual Disable

The user-facing requirement is "sending is not allowed." Visual disablement of the input or send button (e.g., `messageInput.disabled = true`, `sendButton.disabled = true`) is a separate UX choice. The user asked for the functional block only:

- The textarea remains enabled-looking so the user can still type (any text is just inert).
- The Send button remains enabled-looking; clicks and Enter are simply no-ops.
- The user is expected to exit selection mode via the Cancel button if they want to send.

This keeps the change minimal. If visual disablement is desired later, it's a one-line addition per element.

### G.5 What Is NOT Changed

- `startNewChat()` retains its existing defensive block (silently exits selection mode). No change there.
- The `+ New Chat` button is still hidden in selection mode (header is replaced by `Cancel` / `Delete (N)`).
- Switching conversations by clicking sidebar items is still routed to `toggleSelection` while in selection mode (existing Part D behavior).

### G.6 Edge Cases

| Scenario | Handling |
|----------|----------|
| User in selection mode, clicks Send button | `sendMessage()` returns immediately — no message, no state change |
| User in selection mode, presses Enter in textarea | Same — `sendMessage()` returns |
| User in selection mode, types text and clicks Send | Same — text stays in input, no send |
| User in selection mode, types text and presses Enter | Same — text stays in input, no send |
| User exits selection mode (Cancel), then sends | Guard is a no-op; normal flow |
| `sendMessage()` called programmatically while in selection mode | Returns immediately — future-proofs against any other call path |

### G.7 Files Modified

| File | Change |
|------|--------|
| `frontend/index.html` | Add `if (selectionMode) return;` guard at the top of `sendMessage()` |

### G.8 Testing

Manual checklist:

- [ ] Click "Batch Delete" → selection mode enters.
- [ ] Click Send → no message sent; input value preserved; selection intact.
- [ ] Type into textarea and press Enter → no message sent; text stays in input; selection intact.
- [ ] Click Cancel → selection mode exits; input still contains any text typed; Send and Enter work normally.
- [ ] Send a message after exiting selection mode → works as before; no regression.
- [ ] Deleting the active conversation via single `×` still ends in a fresh empty chat (existing behavior).
- [ ] Deleting the active conversation via batch-delete (selected) still ends in a fresh empty chat (existing behavior).

No backend, schema, or API changes. Frontend-only.

All assertions pass.

---

## Combined Testing Checklist

### Bug fix (Part 1)
- [ ] Same as Part A/B testing checklist above

### Confirmation modal (Part 2)
- [ ] `confirm()` is gone — replaced by `showConfirmModal`
- [ ] Modal is centered both horizontally and vertically
- [ ] Backdrop click and Escape close (cancel)
- [ ] Cancel and Confirm buttons work
- [ ] `danger: true` produces a red confirm button
- [ ] Focus moves to cancel when `danger: true` (Enter doesn't accidentally delete)

### Batch delete (Part 3)
- [ ] Sidebar-header shows "Batch Delete" instead of ≡
- [ ] Selection mode shows checkboxes; clicking row toggles
- [ ] Header count updates as items are toggled
- [ ] Delete button disabled when count is 0
- [ ] Cancel exits selection mode
- [ ] Delete opens modal; confirm deletes all; cancel keeps all
- [ ] Active conversation deletion → auto new chat
- [ ] Single-item `×` delete still works and uses the new modal
- [ ] Sidebar collapse via chat-header ≡ still works

### Block sending in selection mode (Part 6)
- [ ] Click Send in selection mode → no message sent, input value preserved, selection intact
- [ ] Press Enter in selection mode → no message sent, input value preserved, selection intact
- [ ] Exit selection mode (Cancel) → Send and Enter both work normally
- [ ] Deleting the active conversation via single `×` still ends in a fresh empty chat (no regression on `startNewChat()` defensive block)
- [ ] Deleting the active conversation via batch-delete (selected) still ends in a fresh empty chat (no regression)

### Regression
- [ ] All 18 backend tests pass
- [ ] Stream-resume fix (Part 1) still works under the new sidebar code