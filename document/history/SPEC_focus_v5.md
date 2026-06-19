# Stream Resume on Second Refresh + Batch Delete + Confirmation Modal + Resurrection Fix + Scroll-Pin - Specification

This iteration bundles six changes:

1. Fix for the second-refresh-during-streaming bug.
2. Replace the browser-native `confirm()` dialog with a custom themed modal centered on the page.
3. Replace the existing ≡ sidebar-collapse button with a batch-delete button that lets the user select and delete multiple conversations at once.
4. Fix for the streaming-conversation-resurrection bug discovered while testing batch delete: deleting a conversation mid-stream causes it to come back when the background LLM task finishes.
5. Smart auto-scroll during streaming: pin to the newest line by default, but stop forcing the scroll if the user has scrolled up to read earlier content. Re-pin when they scroll back to the bottom.
6. Block starting a new conversation while in batch-deletion selection mode.

---

## Part 1: Stream Resume on Second Refresh

### Bug

Refreshing the page a second time during the streaming phase of a conversation causes the in-flight streaming message to disappear.

Reproduction:
1. Send a message in a conversation; LLM starts streaming.
2. Refresh the page once — partial assistant message is restored, streaming continues. ✓
3. Refresh the page a second time while still streaming — partial assistant message is gone. ✗

### Goal

Refreshing the page any number of times during streaming must always show the partial assistant message and continue streaming until completion.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1.1 | On any refresh during streaming, the partial assistant message (every token and thinking chunk stored in the chunk cache) is rendered before live streaming resumes |
| FR-1.2 | Live streaming continues to the same assistant message DOM node after the partial message is rendered |
| FR-1.3 | The streaming badge appears on the sidebar item for the active conversation on every refresh during streaming |
| FR-1.4 | The fix does not regress first-refresh-during-streaming, post-completion refresh, or new-message flows |

### Acceptance Criteria

- [ ] Reproduction steps above show the partial assistant message after the second refresh
- [ ] Partial message content matches what would be visible without the refresh at that moment
- [ ] Streaming continues after the second refresh and completes with `data.end` exactly once
- [ ] Refreshing 3, 5, or more times during streaming produces the same visible state at each moment as a single refresh at that moment
- [ ] No duplicate assistant messages accumulate in the DOM across multiple refreshes
- [ ] Existing tests still pass; first-refresh and post-completion-refresh behaviors are unchanged

---

## Part 2: Custom Confirmation Modal

### Current Behavior (Bug)

`deleteConversation` calls `confirm('Delete this conversation?')`. The browser-native dialog has a generic OS look that does not match the page's dark-theme + cyan/purple-accent styling.

### Goal

Replace the browser-native `confirm()` with a themed modal that:
- Is centered horizontally and vertically on the page.
- Matches the page's visual theme (dark background, accent colors).
- Has explicit Cancel and Confirm buttons.
- Closes on backdrop click or Escape key.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-2.1 | A reusable confirmation modal component is provided (function `showConfirmModal({title, message, confirmText, cancelText, danger})` returns a Promise resolving to `true`/`false`) |
| FR-2.2 | The modal is rendered with a semi-transparent backdrop covering the whole viewport |
| FR-2.3 | The modal panel is centered horizontally and vertically in the viewport using fixed positioning |
| FR-2.4 | The modal uses page-theme colors: `var(--bg-secondary)` background, `var(--border-color)` border, `var(--text-primary)` text, `var(--text-secondary)` muted text |
| FR-2.5 | The modal has a title (prominent), a message (smaller), a Cancel button (secondary), and a Confirm button (primary, red/danger when `danger: true`) |
| FR-2.6 | Clicking the backdrop, pressing Escape, or clicking Cancel closes the modal and resolves the Promise with `false` |
| FR-2.7 | Clicking Confirm closes the modal and resolves the Promise with `true` |
| FR-2.8 | All delete confirmations (single, batch) use this modal instead of `confirm()` |

### Acceptance Criteria

- [ ] No browser-native `confirm()` calls remain in the codebase
- [ ] Modal is centered both horizontally and vertically
- [ ] Modal colors match the page theme
- [ ] Backdrop click and Escape close the modal (cancel)
- [ ] Cancel and Confirm buttons work correctly
- [ ] Modal can be invoked with different title/message/button text per use

---

## Part 3: Batch Delete

### Current Behavior

The sidebar header has a ≡ button that collapses/expands the sidebar. Each conversation item has a `×` button (visible on hover) that opens a native `confirm()` and deletes one conversation.

### Goal

Replace the sidebar-header ≡ button with a batch-delete button. Clicking it enters a "selection mode" where the user can check multiple conversations and delete them all via the themed modal.

The ≡ button in the chat header (`toggleSidebarMain`) is preserved for sidebar collapse/expand.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-3.1 | The sidebar-header ≡ button is replaced with a "Batch Delete" button (icon: trash, label: "Batch Delete") |
| FR-3.2 | Clicking "Batch Delete" enters selection mode |
| FR-3.3 | In selection mode, each conversation item shows a leading checkbox |
| FR-3.4 | Clicking a conversation's checkbox toggles its selection state |
| FR-3.5 | The sidebar header in selection mode shows: a count ("N selected" or "Delete (N)" label), a Delete button, and a Cancel button — replacing the normal header content |
| FR-3.6 | The Delete button is disabled when zero items are selected; enabled and shows the count when one or more are selected |
| FR-3.7 | Clicking Cancel exits selection mode without deleting anything |
| FR-3.8 | Clicking Delete opens the themed confirmation modal with the count and item titles |
| FR-3.9 | On confirmation, all selected conversations are deleted: frontend caches cleared, backend DELETE called per item |
| FR-3.10 | After batch deletion completes, selection mode exits and the sidebar list refreshes |
| FR-3.11 | If the currently-active conversation was among the deleted ones, the user is switched to a new empty chat |
| FR-3.12 | The chat-header ≡ button remains and still collapses/expands the sidebar |
| FR-3.13 | The per-item `×` delete button remains (single-item delete uses the same themed modal) |

### Acceptance Criteria

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
- [ ] Existing tests still pass; no regressions on stream-resume fix or other features

---

## Part 4: Streaming-Conversation Resurrection Fix

### Bug

Deleting a conversation while the background LLM task is still generating causes the conversation to **resurrect in backend storage** after the LLM finishes. The user sees the deleted conversation reappear in the sidebar with the full user message + the LLM-generated assistant message.

### Root Cause

`generate_background` is started as a fire-and-forget `asyncio.create_task`. Deleting the conversation only removes the `StreamJob` from the in-memory registry and the entry from `conversations.json` — it does not stop the background task. When the task finishes, it calls `file_storage.save_conversation(...)`, which is "create-or-update" and silently re-creates the deleted entry.

### Goal

A deleted conversation stays deleted, even if its background LLM task is still running at the time of deletion.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-4.1 | When the user deletes a conversation while the LLM is still generating, the conversation must NOT reappear in `GET /api/chat/conversations` after the LLM finishes |
| FR-4.2 | The fix does not affect the normal stream-completion path (no delete): the assistant message is still appended to storage on `data.end` |
| FR-4.3 | The fix does not affect conversations that have no in-flight background task at delete time |
| FR-4.4 | The fix is contained to the backend (no frontend changes required) |

### Acceptance Criteria

- [ ] Reproduction (delete mid-stream, wait for LLM to finish) shows 0 conversations afterward — no resurrection
- [ ] Normal stream completion still saves the assistant message to history
- [ ] Deleting a conversation that has no in-flight LLM task still works
- [ ] Backend regression test `test_generate_background_aborts_on_cancellation` passes
- [ ] All existing backend tests still pass

---

## Part 5: Smart Auto-Scroll During Streaming

### Current Behavior (Problem)

While the LLM is streaming tokens into the assistant message, every new chunk calls `messagesContainer.scrollTop = messagesContainer.scrollHeight`, which forces the scroll position to the bottom. If the user scrolls up to read earlier content, the next chunk yanks them back to the bottom.

### Goal

- By default, the auto-scroll follows the latest content (preserves existing behavior).
- If the user manually scrolls up during streaming, leave their scroll position alone. The next chunk does NOT force them back to the bottom.
- When the user scrolls back to the bottom, the next chunk re-pins the scroll.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-5.1 | During streaming, the messages container auto-scrolls to the bottom on each chunk **only if** the user was already at (or within 50px of) the bottom before the chunk was applied |
| FR-5.2 | The "pinned" state is captured BEFORE the DOM update for the chunk, since the per-chunk content height can exceed 50px and would cause a post-update check to incorrectly report "not pinned" |
| FR-5.3 | Scrolling back to the bottom during streaming re-pins the scroll on the next chunk |
| FR-5.4 | Other scroll sites (cached-chunks replay on resume, sendMessage placeholder, addMessage) are unchanged — they happen once per action and aren't per-chunk auto-scrolls |

### Acceptance Criteria

- [ ] Streaming with no user interaction: scroll stays pinned to the bottom throughout (regression of the existing behavior)
- [ ] User scrolls up during streaming: scroll position stays where the user put it; new chunks do NOT force them back to the bottom
- [ ] User scrolls back to the bottom: next chunk re-pins the scroll
- [ ] Refresh-during-streaming (Part 1 fix): no regression — the cached-chunks replay still scrolls to the bottom

---

## Part 6: Block Sending Messages in Selection Mode

### Goal

While the user is in batch-deletion selection mode, sending a message is not allowed. Both interaction paths to sending must be blocked:
- Clicking the Send button.
- Pressing Enter in the message textarea.

The user's pending batch-delete selection must not be disturbed.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-6.1 | When `sendMessage()` is invoked while `selectionMode === true`, the function returns immediately without mutating any state (selection, active conversation, messages, input value) |
| FR-6.2 | The Send button click handler routes through `sendMessage()` and is therefore blocked by FR-6.1 |
| FR-6.3 | The `messageInput` `keydown` handler (Enter key, when Shift is not held) routes through `sendMessage()` and is therefore blocked by FR-6.1 |
| FR-6.4 | The existing defensive block in `startNewChat()` (which silently exits selection mode) is preserved as-is |

### Acceptance Criteria

- [ ] With the user in selection mode (after clicking "Batch Delete"), clicking the Send button does nothing — no message sent, no state change, input value preserved
- [ ] With the user in selection mode, pressing Enter in the textarea does nothing — same as above
- [ ] Selection, active conversation, and message content are preserved when send is blocked
- [ ] After exiting selection mode (Cancel button), Send and Enter both work normally — no regression
- [ ] Deleting the active conversation via single-delete or via batch-delete still ends in a fresh empty chat — no regression on the existing `startNewChat()` defensive block