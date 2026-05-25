# Frontend Enhancement + Thinking Content - Specification

## Overview

Add thinking content display to the chat interface, enhance message block scrolling behavior, improve empty state UX, and add input auto-expansion.

---

## 1. Backend: Thinking Content Streaming

### 1.1 LLM Response Structure

The MiniMax LLM returns content blocks of two types:
- `type: "thinking"` — internal reasoning, content in `thinking` field
- `type: "text"` — final response, content in `text` field

### 1.2 SSE Protocol (Extended)

**New event types:**

| Event | Format | Description |
|-------|--------|-------------|
| `thinking` | `data: {"thinking": "..."}` | Single thinking token |
| `thinking_end` | `data: {"thinking_end": true}` | Thinking phase complete |
| `token` | `data: {"token": "..."}` | Single response token (existing) |
| `partial` | `data: {"partial": "..."}` | Already-streamed content on resume (unchanged) |

**Stream sequence:**
```
data: {"thinking": "The user asks about..."}
data: {"thinking": "Let me think..."}
data: {"thinking_end": true}
data: {"token": "Python is..."}
data: {"token": null}
```

### 1.3 Storage Schema Change

**Before:**
```json
{"role": "assistant", "content": "Full response text..."}
```

**After:**
```json
{"role": "assistant", "content": "Full response text...", "thinking": "Internal reasoning..."}
```

### 1.4 Files to Modify

- `backend/chat/chain.py` — Extract thinking blocks, yield thinking tokens separately before text tokens
- `backend/chat/service.py` — Pass through thinking tokens in async generator
- `backend/storage/file_storage.py` — No schema change needed (JSON storage of dict)
- `backend/chat/routes.py` — May need adjustment for thinking_end event emission

### 1.5 API Changes

| Endpoint | Change |
|----------|--------|
| `POST /api/chat/stream` | Emit thinking tokens, then thinking_end, then text tokens |
| `GET /api/chat/history/{id}` | Returns messages with optional `thinking` field |
| `GET /api/chat/stream/status/{id}` | Add `partial_thinking` field |
| `POST /api/chat/stream/resume/{id}` | Send `{"partial": "...", "partial_thinking": "..."}` |

---

## 2. Frontend: Thinking Display

### 2.1 Thinking Section

**Location:** Inside assistant message block, above the text response

**Structure:**
```html
<div class="message assistant">
  <div class="thinking-section">
    <div class="thinking-content">...</div>
    <button class="thinking-toggle">Show more</button>
  </div>
  <div class="message-content markdown-body">...</div>
</div>
```

### 2.2 Fold/Unfold Behavior

- If thinking content ≤ 3 lines: show fully, hide toggle button
- If thinking content > 3 lines:
  - Default: show first 3 lines with "Show more" button
  - On "Show more": expand fully, button changes to "Show less"
  - On "Show less": collapse to 3 lines

### 2.3 Line Counting

Line counting is based on rendered display, not raw character count.

**Algorithm:**
1. Render thinking content into a temporary hidden div with identical styling
2. Measure the rendered height
3. Divide by line-height to get approximate line count
4. If ≥ 4 lines (accounting for partial lines), enable collapse

**Implementation:**
```javascript
function countLines(element) {
  const style = window.getComputedStyle(element);
  const lineHeight = parseFloat(style.lineHeight);
  const height = element.scrollHeight;
  return Math.round(height / lineHeight);
}

function shouldCollapseThinking(thinkingElement) {
  const clone = thinkingElement.cloneNode();
  clone.style.position = 'absolute';
  clone.style.visibility = 'hidden';
  clone.style.width = thinkingElement.offsetWidth + 'px';
  clone.style.maxHeight = 'none';
  document.body.appendChild(clone);
  const lines = countLines(clone);
  document.body.removeChild(clone);
  return lines > 3;
}
```

**Threshold:** 3 lines displayed when collapsed. If content renders to 4 or more lines, collapse is enabled.

---

## 3. Message Block Internal Scrolling

### 3.1 CSS

```css
.message {
  max-height: 400px; /* or some reasonable max */
  overflow-y: auto;
  /* Custom scrollbar auto-hide */
  scrollbar-width: none; /* Firefox */
  -ms-overflow-style: none;  /* IE/Edge */
}

.message::-webkit-scrollbar {
  display: none;
}
```

### 3.2 Scrollbar Auto-Hide

**Trigger:** `wheel` event on the message element

**Behavior:**
1. On wheel event within a `.message` block: show scrollbar, start 3s timer
2. After 3s of no wheel events on that block: hide scrollbar
3. Scrollbar remains visible if user is actively scrolling
4. Only applies when content overflows (scrollbar hidden by default)

**Implementation approach:**
- Add `wheel` event listener to each message block
- Toggle a class `.scrollbar-visible` that overrides scrollbar display
- Use `setTimeout` for 3s timer, clear on new wheel event

---

## 4. Empty State: Centered Input

### 4.1 Behavior

- When `messagesContainer` has no messages: input is vertically and horizontally centered
- When first message is sent: input moves to bottom position
- Transition is smooth (CSS flexbox)

### 4.2 CSS Structure

```css
.chat-messages {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-messages.empty {
  justify-content: center;
  align-items: center;
}

.chat-messages:not(.empty) {
  justify-content: flex-start;
}
```

### 4.3 Scrollbar Auto-Hide

The empty state input box does NOT have scrollbar auto-hide (it has no scrollable content). This requirement is N/A for empty state.

---

## 5. Input Box Auto-Expand

### 5.1 Behavior

- Minimum height: 5 lines
- Auto-expands as user types
- Maximum height: ~50% of viewport (to still show some context)

### 5.2 Implementation

Use CSS `field-sizing: content` for modern browsers, with fallback:

```css
.chat-input {
  field-sizing: content; /* Chrome 123+, Firefox 129+ */
  min-height: 120px; /* ~5 lines */
  max-height: 50vh;
  overflow-y: auto;
  resize: none; /* Let it size naturally */
}
```

**Fallback JS (older browsers):**
- Listen to `input` event
- Clone input to temp div, measure height
- Adjust `rows` attribute

---

## 6. Markdown Rendering

### 6.1 Library

**marked.js** — included via CDN

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

### 6.2 Rendering

- Only render text response through `marked.parse()`
- Thinking content: plain text, no markdown processing
- Code blocks, lists, bold, italic, links, tables all supported

### 6.3 Sanitization

marked.js outputs HTML. For safety, configure marked to not allow raw HTML (default safe mode).

---

## 7. Summary of Changes

### Backend Files
| File | Change |
|------|--------|
| `backend/chat/chain.py` | Extract thinking blocks from LLM output, yield separately |
| `backend/chat/service.py` | Pass through thinking tokens |
| `backend/chat/routes.py` | Ensure thinking_end event emitted |

### Frontend Files
| File | Change |
|------|--------|
| `frontend/index.html` | All frontend changes (CSS + JS) |

### No Changes
- `backend/storage/file_storage.py` — JSON schema supports new fields
- `backend/config.py` — No config changes
- `backend/main.py` — No changes

---

## 8. Dependencies

### Backend
None (already using langchain)

### Frontend
```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

---

## 9. Acceptance Criteria

1. [ ] Thinking tokens stream before text tokens in SSE
2. [ ] Thinking displayed in collapsible section (Show more/less when >3 lines)
3. [ ] Message blocks scroll internally, not expand the page
4. [ ] Scrollbar auto-hides after 3s on message blocks
5. [ ] Empty state has centered input
6. [ ] Input box expands with content (min 5 lines)
7. [ ] Markdown rendering works in assistant messages
8. [ ] History API returns thinking content
9. [ ] Resume works with partial thinking
