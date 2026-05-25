# Frontend Enhancement + Thinking Content - Design

## 1. Architecture Decisions

### 1.1 Thinking Token Flow

**Design Choice:** Thinking blocks from LLM are streamed first, then text blocks.

**Rationale:** Thinking represents the model's internal reasoning process which logically precedes the final response. Streaming thinking first provides immediate feedback to users that the model is "thinking."

**Alternative considered:** Interleave thinking and text tokens — rejected because thinking is internal reasoning and should complete before response begins.

### 1.2 Thinking Persistence

**Design Choice:** Store thinking content in message history alongside response content.

**Rationale:** Users may want to review the model's reasoning for past responses. Storage schema is backward-compatible (thinking field is optional).

**Schema:**
```json
{
  "role": "assistant",
  "content": "Response text...",
  "thinking": "Internal reasoning..."  // optional
}
```

### 1.3 Markdown Library

**Design Choice:** marked.js via CDN.

**Rationale:** Lightweight (39KB), no build step required, widely used and maintained. Plain HTML/JS architecture makes CDN inclusion simple.

### 1.4 Scrollbar Auto-Hide Implementation

**Design Choice:** CSS `::-webkit-scrollbar` hidden by default, shown via `.scrollbar-visible` class on wheel event, with 3s setTimeout.

**Rationale:** Native browser scrollbar styling is not possible cross-browser. Using CSS class toggle allows smooth transitions and proper timer management per-message-block.

**Code pattern:**
```javascript
messageBlock.addEventListener('wheel', function() {
  this.classList.add('scrollbar-visible');
  clearTimeout(this._hideTimer);
  this._hideTimer = setTimeout(() => {
    this.classList.remove('scrollbar-visible');
  }, 3000);
});
```

### 1.5 Input Auto-Expand

**Design Choice:** CSS `field-sizing: content` with JS fallback for older browsers.

**Rationale:** `field-sizing` is a new CSS property (Chrome 123+, Firefox 129+) that handles this natively. For older browsers, JS fallback measures content height and adjusts rows attribute.

**Fallback approach:**
```javascript
function autoResizeInput(textarea) {
  const clone = textarea.cloneNode();
  clone.style.position = 'absolute';
  clone.style.visibility = 'hidden';
  document.body.appendChild(clone);
  const newHeight = clone.scrollHeight;
  document.body.removeChild(clone);
  textarea.style.height = newHeight + 'px';
}
```

### 1.6 Empty State Centering

**Design Choice:** CSS flexbox with `.empty` class toggle on messages container.

**Rationale:** Pure CSS solution, no JS position calculations. When messages exist, remove `.empty` class and flexbox naturally flows input to bottom.

```css
.chat-messages.empty {
  justify-content: center;
  align-items: center;
}
.chat-messages:not(.empty) {
  justify-content: flex-start;
}
```

---

## 2. System Architecture

### 2.1 Data Flow: New Message

```
User types message
       ↓
POST /api/chat/stream {message, conversation_id}
       ↓
LLM returns chunks with thinking + text blocks
       ↓
Backend: yield {"thinking": "..."} for each thinking chunk
Backend: yield {"thinking_end": true} when thinking done
Backend: yield {"token": "..."} for each text chunk
Backend: yield {"token": null} when done
       ↓
Frontend: receives thinking tokens → builds thinking section
Frontend: receives thinking_end → finalize thinking display
Frontend: receives text tokens → append to message
Frontend: receives token null → complete
```

### 2.2 Data Flow: Resume Stream

```
Check stream status
       ↓
GET /api/chat/stream/status/{id}
       ↓
Response: {
  streaming: true,
  partial_content: "...",
  partial_thinking: "...",
  tokens_count: N,
  is_complete: false
}
       ↓
POST /api/chat/stream/resume/{id}
       ↓
Frontend: displays partial_thinking first, then partial_content
Backend sends remaining thinking tokens
Backend sends thinking_end
Backend continues streaming text tokens
```

---

## 3. Frontend Structure

### 3.1 HTML Structure

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
    <div class="message-content markdown-body"></div>
  </div>
</div>
```

### 3.2 CSS Classes

| Class | Purpose |
|-------|---------|
| `.thinking-section` | Container for thinking content |
| `.thinking-content` | The actual thinking text |
| `.thinking-toggle` | Show more/less button |
| `.thinking-collapsed` | Applied when thinking is collapsed |
| `.scrollbar-visible` | Override scrollbar hiding |
| `.empty` | On messages container when no messages |

### 3.3 JavaScript Functions

| Function | Responsibility |
|----------|---------------|
| `processStreamResponse()` | Parse SSE, handle thinking + token events |
| `addMessage()` | Create message element with proper structure |
| `updateThinkingDisplay()` | Handle thinking content and fold/unfold |
| `setupScrollbarAutoHide()` | Attach wheel listener to message blocks |
| `autoResizeInput()` | Expand textarea with content |
| `marked.parse()` | Render markdown |

---

## 4. Backend Implementation

### 4.1 chain.py Changes

**Current (simplified):**
```python
async for chunk in self.chain.astream(messages):
    content = chunk.content
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "text":
                yield block.get("text", "")
```

**New:**
```python
async for chunk in self.chain.astream(messages):
    content = chunk.content
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "thinking":
                yield {"thinking": block.get("thinking", "")}
            elif block.get("type") == "text":
                yield {"token": block.get("text", "")}
    elif isinstance(content, str):
        yield {"token": content}
```

### 4.2 service.py Changes

The async generator yields dicts now (`{"thinking": "..."}` or `{"token": "..."}`) instead of plain strings. Routes.py needs to handle this format.

### 4.3 routes.py Changes

Emit `thinking_end` event after all thinking tokens sent, before text tokens begin.

---

## 5. Edge Cases

### 5.1 No Thinking Block

Some responses may not include a thinking block (model behavior, especially on short/simple responses). Handle gracefully:
- If LLM returns no thinking block, immediately send `thinking_end` before text tokens
- Frontend renders only the text response without thinking section

### 5.2 Stream Interruption Mid-Thinking

If stream is interrupted during thinking phase:
- StreamJob in memory retains accumulated thinking tokens
- On resume: status endpoint returns `partial_thinking` with accumulated thinking
- Frontend shows partial thinking with "..." continuation indicator
- Backend continues streaming from interruption point

### 5.3 Stream Interruption Mid-Text

If stream is interrupted during text phase (after thinking_end):
- StreamJob retains partial text
- On resume: status endpoint returns `partial_content` (unchanged behavior)
- Thinking is already complete, only text resumes

### 5.4 Very Long Thinking

If thinking exceeds 100KB:
- Still stream normally (no truncation in v1)
- Frontend may need to virtualize rendering for very long thinking
- Architecture supports storing large thinking content

### 5.5 Thinking Field Size Limits

For extreme cases (>1MB thinking):
- Not explicitly limited in storage
- Consider adding size check at save time in future version
- Current design allows arbitrary size

### 5.6 Markdown in Thinking

Thinking content is NOT rendered as markdown. It's displayed as plain text to avoid any injection risks.

### 5.7 Partial Resume After Complete

If a conversation was fully completed (streaming=false, is_complete=true) but user sends another message:
- Backend starts new stream, no resume needed
- History already contains full thinking + content

---

## 6. Testing Checklist

### Backend
- [ ] Thinking blocks are extracted and yielded before text blocks
- [ ] thinking_end event is sent after all thinking tokens
- [ ] History API returns thinking field
- [ ] Resume sends partial_thinking

### Frontend
- [ ] Thinking displayed above response
- [ ] Show more/less works when >3 lines
- [ ] Message blocks scroll internally
- [ ] Scrollbar auto-hides after 3s
- [ ] Empty state input is centered
- [ ] Input expands with content
- [ ] Markdown renders correctly
- [ ] Resume works with partial thinking

---

## 7. Files Summary

### Create
None

### Modify
| File | Changes |
|------|---------|
| `backend/chat/chain.py` | Extract thinking blocks, yield as dicts |
| `backend/chat/service.py` | Pass through thinking dicts |
| `backend/chat/routes.py` | Emit thinking_end, handle dict yields |
| `frontend/index.html` | All frontend changes |

### No Change
| File | Reason |
|------|--------|
| `backend/storage/file_storage.py` | JSON supports new fields |
| `backend/main.py` | No routing changes |
| `backend/config.py` | No config needed |
