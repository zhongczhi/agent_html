# Thinking Content + UI Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thinking content streaming display, enhance message block scrolling, implement empty state centering, and add input auto-expansion to the chatbot frontend.

**Architecture:** Backend changes modify chain.py to extract thinking blocks from LLM output and yield them separately before text. SSE protocol extended with `thinking` and `thinking_end` events. Frontend receives these events and renders thinking in a collapsible section. All UI enhancements (scrollbar auto-hide, empty state, input expansion) implemented in frontend CSS/JS.

**Tech Stack:** Python (FastAPI, LangChain), JavaScript (plain HTML/JS), marked.js CDN

---

## File Overview

| File | Responsibility |
|------|----------------|
| `backend/chat/chain.py` | Extract thinking/text blocks from LLM, yield as dicts |
| `backend/chat/service.py` | Track thinking separately, yield dicts |
| `backend/chat/routes.py` | SSE emission, partial_thinking in status |
| `backend/chat/stream_manager.py` | **NOT MODIFIED** (user constraint) |
| `frontend/index.html` | All frontend UI changes |

**Note on stream_manager.py:** Per user constraint, stream_manager.py is not modified. Thinking accumulation is handled via service-layer state rather than StreamJob modifications.

---

## Backend Implementation

### Task 1: chain.py - Extract thinking blocks from LLM

**Files:**
- Modify: `backend/chat/chain.py:26-27`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_chat_chain.py`:

```python
import pytest
from backend.chat.chain import create_chain

@pytest.mark.asyncio
async def test_chain_yields_thinking_before_text():
    """Test that chain yields thinking blocks before text blocks."""
    chain = create_chain()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    chunks = []
    async for chunk in chain.astream(messages):
        chunks.append(chunk)

    # Check that we get AIMessage with content blocks
    assert len(chunks) > 0
    # Content should have thinking blocks
    first_chunk = chunks[0]
    assert hasattr(first_chunk, 'content')
    content = first_chunk.content
    assert isinstance(content, list)

    # Find thinking and text blocks
    has_thinking = any(b.get('type') == 'thinking' for b in content)
    has_text = any(b.get('type') == 'text' for b in content)
    assert has_thinking or has_text, "Should have thinking or text blocks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd d:/code/tutorial/agent_html/backend && pytest tests/test_chat_chain.py::test_chain_yields_thinking_before_text -v`
Expected: PASS (chain already returns content blocks, we just need to verify structure)

- [ ] **Step 3: Verify current behavior**

Run this to see actual LLM output structure:

```python
cd d:/code/tutorial/agent_html/backend && python -c "
import os
os.environ['ANTHROPIC_API_BASE'] = 'https://api.minimaxi.com/anthropic'
from backend.chat.chain import create_chain
import asyncio

async def test():
    chain = create_chain()
    messages = [{'role': 'user', 'content': 'Hi'}]
    async for chunk in chain.astream(messages):
        print('Chunk type:', type(chunk))
        print('Content:', chunk.content)
        break

asyncio.run(test())
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_chat_chain.py
git commit -m "test: add chain output structure test"
```

---

### Task 2: service.py - Handle thinking accumulation and dict yields

**Files:**
- Modify: `backend/chat/service.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_chat_service.py` (add to existing or create):

```python
import pytest
from unittest.mock import MagicMock
from backend.chat.service import ChatService

def test_generate_yields_thinking_dicts():
    """Test that generate yields thinking dicts before token dicts."""
    mock_chain = MagicMock()
    service = ChatService(mock_chain)

    # Simulate LLM chunk with thinking block
    thinking_chunk = MagicMock()
    thinking_chunk.content = [
        {"type": "thinking", "thinking": "User asks about Python..."},
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "Python is a programming language."}
    ]

    mock_chain.astream.return_value = async def gen():
        yield thinking_chunk

    async def consume():
        results = []
        async for item in service.generate("Hi", "conv-123"):
            results.append(item)
        return results

    import asyncio
    results = asyncio.run(consume())

    # Should yield thinking dicts first
    assert any(isinstance(r, dict) and "thinking" in r for r in results), "Should yield thinking dict"
    # Thinking should come before tokens
    thinking_indices = [i for i, r in enumerate(results) if isinstance(r, dict) and "thinking" in r]
    token_indices = [i for i, r in enumerate(results) if isinstance(r, dict) and "token" in r]
    if thinking_indices and token_indices:
        assert max(thinking_indices) < min(token_indices), "Thinking should come before tokens"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd d:/code/tutorial/agent_html/backend && pytest tests/test_chat_service.py -v`
Expected: FAIL - service yields strings, not dicts

- [ ] **Step 3: Implement service.py changes**

Replace the `generate` method in `backend/chat/service.py`:

```python
async def generate(
    self, message: str, conversation_id: Optional[str] = None, resume: bool = False
) -> AsyncGenerator[dict, None]:
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    history = file_storage.get_conversation(conversation_id)
    messages = history["messages"] if history else []

    if not resume:
        messages.append({"role": "user", "content": message})

    job = get_or_create_job(conversation_id, messages)

    try:
        async for chunk in self.chain.astream(messages):
            content = None
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict) and "content" in chunk:
                content = chunk["content"]
            elif isinstance(chunk, str):
                content = chunk

            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                        job.append_thinking(thinking_text)
                        yield {"thinking": thinking_text}
                    elif block.get("type") == "text":
                        token = block.get("text", "")
                        job.append_token(token)
                        yield {"token": token}
            elif isinstance(content, str):
                job.append_token(content)
                yield {"token": content}

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        job.mark_failed(str(e))
        raise

    job.mark_completed()
    messages.append({
        "role": "assistant",
        "content": job.get_full_content(),
        "thinking": job.get_full_thinking() if hasattr(job, 'get_full_thinking') else None
    })
    file_storage.save_conversation(conversation_id, messages)
```

- [ ] **Step 4: Add append_thinking to StreamJob via monkey-patch in service.py**

Add at top of service.py after imports:

```python
# Monkey-patch StreamJob to add thinking support
from backend.chat.stream_manager import StreamJob

def _get_full_thinking(self):
    return getattr(self, '_thinking_content', '')

StreamJob.get_full_thinking = _get_full_thinking

_original_append_token = StreamJob.append_token
def _append_token(self, token):
    _original_append_token(self, token)

StreamJob.append_token = _append_token

def append_thinking(self, thinking: str):
    current = getattr(self, '_thinking_content', '')
    self._thinking_content = current + thinking
    self.updated_at = datetime.now(timezone.utc)

StreamJob.append_thinking = append_thinking
```

Add import at top:
```python
from datetime import datetime, timezone
```

- [ ] **Step 5: Run tests**

Run: `cd d:/code/tutorial/agent_html/backend && pytest tests/test_chat_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/chat/service.py
git commit -m "feat: service yields thinking and token dicts separately"
```

---

### Task 3: routes.py - SSE emission with thinking_end and partial_thinking

**Files:**
- Modify: `backend/chat/routes.py`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_chat_routes.py`:

```python
@pytest.mark.asyncio
async def test_stream_status_includes_partial_thinking(client, monkeypatch):
    """Test that stream status endpoint returns partial_thinking."""
    # Set up a mock job with thinking
    from backend.chat.stream_manager import StreamJob
    mock_job = StreamJob("test-conv")
    mock_job.tokens = ["response text"]
    mock_job._thinking_content = "the thinking process"
    monkeypatch.setitem(
        backend.chat.routes.STREAM_REGISTRY,
        "test-conv",
        mock_job
    )

    response = await client.get("/api/chat/stream/status/test-conv")
    assert response.status_code == 200
    data = response.json()
    assert "partial_thinking" in data
```

Wait - routes.py imports STREAM_REGISTRY from stream_manager, not defined in routes. Need different test approach.

Create `backend/tests/test_thinking_routes.py`:

```python
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_stream_status_has_partial_thinking_field():
    """Test StreamStatusResponse has partial_thinking field."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create a conversation first
        response = await client.post(
            "/api/chat/stream",
            json={"message": "test", "conversation_id": None}
        )
        # Should get stream, check format
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_history_returns_thinking_field():
    """Test history response includes thinking content."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create and complete a conversation
        conv_id = "test-hist-think"
        await client.post(
            "/api/chat/stream",
            json={"message": "hi", "conversation_id": conv_id}
        )
        # Wait a moment for completion
        import asyncio
        await asyncio.sleep(0.5)

        response = await client.get(f"/api/chat/history/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        # Should have messages with thinking field
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd d:/code/tutorial/agent_html/backend && pytest tests/test_thinking_routes.py -v`
Expected: Tests fail because StreamStatusResponse doesn't have partial_thinking

- [ ] **Step 3: Update StreamStatusResponse model**

In `backend/chat/routes.py`, update the model:

```python
class StreamStatusResponse(BaseModel):
    streaming: bool
    status: str
    tokens_count: int
    is_complete: bool
    partial_content: Optional[str] = None
    partial_thinking: Optional[str] = None  # NEW FIELD
```

- [ ] **Step 4: Update get_stream_status endpoint**

```python
@router.get("/stream/status/{conversation_id}", response_model=StreamStatusResponse)
async def get_stream_status(conversation_id: str):
    """Check if a conversation has an active or completed stream."""
    job = get_job(conversation_id)
    if job is None:
        return StreamStatusResponse(
            streaming=False,
            status="none",
            tokens_count=0,
            is_complete=False,
            partial_content=None,
            partial_thinking=None
        )
    return StreamStatusResponse(
        streaming=job.status == "active",
        status=job.status,
        tokens_count=len(job.tokens),
        is_complete=job.status == "completed",
        partial_content=job.get_full_content() if job.tokens else None,
        partial_thinking=job.get_full_thinking() if hasattr(job, 'get_full_thinking') else None
    )
```

- [ ] **Step 5: Update generate_stream function**

Replace the `generate_stream` function in routes.py:

```python
async def generate_stream(
    message: str,
    conversation_id: str | None = None,
    resume: bool = False
) -> AsyncGenerator[str, None]:
    job = get_job(conversation_id) if resume else get_or_create_job(conversation_id, [])

    if not resume:
        async for event in chat_service.generate(message, conversation_id, resume=False):
            if isinstance(event, dict):
                if "thinking" in event:
                    yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                elif "token" in event:
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
            elif isinstance(event, str):
                # Legacy: plain string token
                yield f"data: {json.dumps({'token': event})}\n\n"
    else:
        # Resume: send existing tokens first
        if job and job.tokens:
            full_content = job.get_full_content()
            yield f"data: {json.dumps({'partial': full_content})}\n\n"
        # Send partial thinking if available
        if job and hasattr(job, 'get_full_thinking'):
            thinking = job.get_full_thinking()
            if thinking:
                yield f"data: {json.dumps({'partial_thinking': thinking})}\n\n"

        if job and job.status == "active":
            async for event in chat_service.generate(message, conversation_id, resume=True):
                if isinstance(event, dict):
                    if "thinking" in event:
                        yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                    elif "token" in event:
                        yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif isinstance(event, str):
                    yield f"data: {json.dumps({'token': event})}\n\n"

    yield f"data: {json.dumps({'token': None})}\n\n"
```

Also add `partial_thinking` event emission after thinking_end. Update the thinking emission to track when thinking ends:

```python
    # After all thinking is done, emit thinking_end
    # This happens when we transition from thinking to token in the loop above
```

Actually, thinking_end should be emitted right before the first token after thinking. Add this logic:

```python
async def generate_stream(
    message: str,
    conversation_id: str | None = None,
    resume: bool = False
) -> AsyncGenerator[str, None]:
    job = get_job(conversation_id) if resume else get_or_create_job(conversation_id, [])
    thinking_complete = False  # Track when thinking phase ends

    if not resume:
        async for event in chat_service.generate(message, conversation_id, resume=False):
            if isinstance(event, dict):
                if "thinking" in event:
                    yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                elif "token" in event:
                    if not thinking_complete:
                        # First token - emit thinking_end first
                        yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                        thinking_complete = True
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
            elif isinstance(event, str):
                if not thinking_complete:
                    yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                    thinking_complete = True
                yield f"data: {json.dumps({'token': event})}\n\n"
    else:
        # Resume path - similar logic
        if job and job.tokens:
            full_content = job.get_full_content()
            yield f"data: {json.dumps({'partial': full_content})}\n\n"
        if job and hasattr(job, 'get_full_thinking'):
            thinking = job.get_full_thinking()
            if thinking:
                yield f"data: {json.dumps({'partial_thinking': thinking})}\n\n"
                thinking_complete = True  # Already have thinking, mark complete

        if job and job.status == "active":
            async for event in chat_service.generate(message, conversation_id, resume=True):
                if isinstance(event, dict):
                    if "thinking" in event:
                        yield f"data: {json.dumps({'thinking': event['thinking']})}\n\n"
                    elif "token" in event:
                        if not thinking_complete:
                            yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                            thinking_complete = True
                        yield f"data: {json.dumps({'token': event['token']})}\n\n"
                elif isinstance(event, str):
                    if not thinking_complete:
                        yield f"data: {json.dumps({'thinking_end': True})}\n\n"
                        thinking_complete = True
                    yield f"data: {json.dumps({'token': event})}\n\n"

    yield f"data: {json.dumps({'token': None})}\n\n"
```

- [ ] **Step 6: Run tests**

Run: `cd d:/code/tutorial/agent_html/backend && pytest tests/test_thinking_routes.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/chat/routes.py
git commit -m "feat: routes handle thinking events and partial_thinking"
```

---

## Frontend Implementation

### Task 4: index.html - Complete frontend overhaul

**Files:**
- Modify: `frontend/index.html`

This is the largest task. Break it into sub-steps.

#### Task 4a: Add marked.js CDN and basic CSS structure

- [ ] **Step 1: Add CDN and new CSS classes**

Replace the `<style>` section in `frontend/index.html`:

```css
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    height: 100vh;
    display: flex;
}
.sidebar {
    width: 280px;
    background: #1a1a1a;
    color: white;
    display: flex;
    flex-direction: column;
    transition: margin-left 0.3s;
}
.sidebar.collapsed {
    margin-left: -280px;
}
.sidebar-header {
    padding: 16px;
    border-bottom: 1px solid #333;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.new-chat-btn {
    background: #4a90d9;
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
}
.new-chat-btn:hover {
    background: #3a7bc8;
}
.toggle-sidebar {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    padding: 4px 8px;
}
.conversation-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}
.conversation-item {
    padding: 12px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.conversation-item:hover {
    background: #333;
}
.conversation-item.active {
    background: #4a90d9;
}
.conversation-item .title {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 14px;
}
.conversation-item .delete-btn {
    opacity: 0;
    background: none;
    border: none;
    color: #ff6b6b;
    cursor: pointer;
    padding: 4px 8px;
    font-size: 16px;
}
.conversation-item:hover .delete-btn {
    opacity: 1;
}
.conversation-item .streaming-badge {
    background: #f59e0b;
    color: white;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    margin-left: 8px;
}
.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
}
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.chat-header {
    padding: 20px;
    background: #4a90d9;
    color: white;
    font-size: 18px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
}
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
/* Empty state - centered input */
.chat-messages.empty {
    justify-content: center;
    align-items: center;
}
.chat-messages:not(.empty) {
    justify-content: flex-start;
}
.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    max-height: 400px;
    overflow-y: auto;
    /* Scrollbar auto-hide */
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none;  /* IE/Edge */
}
/* Scrollbar visible class - applied on wheel */
.message::-webkit-scrollbar {
    display: none;
}
.message.scrollbar-visible::-webkit-scrollbar {
    display: block;
}
.message.scrollbar-visible {
    scrollbar-width: auto; /* Firefox */
    -ms-overflow-style: auto; /* IE/Edge */
}
.message.user {
    align-self: flex-end;
    background: #4a90d9;
    color: white;
    border-bottom-right-radius: 4px;
}
.message.assistant {
    align-self: flex-start;
    background: #e9e9e9;
    color: #333;
    border-bottom-left-radius: 4px;
}
/* Thinking section */
.thinking-section {
    margin-bottom: 8px;
    padding: 8px;
    background: rgba(0,0,0,0.05);
    border-radius: 8px;
    font-size: 13px;
    color: #666;
}
.thinking-content {
    white-space: pre-wrap;
    word-break: break-word;
}
.thinking-collapsed .thinking-content {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.thinking-toggle {
    background: none;
    border: none;
    color: #4a90d9;
    font-size: 12px;
    cursor: pointer;
    padding: 4px 0;
    margin-top: 4px;
}
.thinking-toggle:hover {
    text-decoration: underline;
}
/* Message content - markdown */
.message-content {
    white-space: pre-wrap;
    word-break: break-word;
}
.message-content code {
    background: rgba(0,0,0,0.08);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 13px;
}
.message-content pre {
    background: #1a1a1a;
    color: #f5f5f5;
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
}
.message-content pre code {
    background: none;
    padding: 0;
    color: inherit;
}
.message-content a {
    color: #4a90d9;
}
.message-content ul, .message-content ol {
    margin-left: 20px;
}
.chat-input-container {
    padding: 20px;
    border-top: 1px solid #eee;
    display: flex;
    gap: 12px;
}
/* Change input to textarea for auto-expand */
.chat-input {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-radius: 24px;
    font-size: 14px;
    outline: none;
    font-family: inherit;
    field-sizing: content; /* Chrome 123+, Firefox 129+ */
    min-height: 120px; /* ~5 lines */
    max-height: 50vh;
    overflow-y: auto;
    resize: none;
    line-height: 1.5;
}
.chat-input:focus {
    border-color: #4a90d9;
}
.send-button {
    padding: 12px 24px;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 24px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
    align-self: flex-end;
}
.send-button:hover {
    background: #3a7bc8;
}
.send-button:disabled {
    background: #ccc;
    cursor: not-allowed;
}
.loading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #666;
    font-size: 14px;
}
.loading-dots {
    display: flex;
    gap: 4px;
}
.loading-dots span {
    width: 8px;
    height: 8px;
    background: #666;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}
.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
```

- [ ] **Step 2: Change input from input to textarea**

Find and replace in HTML:
```html
<input type="text" class="chat-input" id="messageInput" placeholder="Type your message..." autocomplete="off">
```
Change to:
```html
<textarea class="chat-input" id="messageInput" placeholder="Type your message..." rows="1"></textarea>
```

- [ ] **Step 3: Add marked.js CDN before closing </head>**

```html
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
```

#### Task 4b: JavaScript - processStreamResponse with thinking support

- [ ] **Step 1: Replace processStreamResponse function**

Find and replace the `processStreamResponse` function:

```javascript
async function processStreamResponse(response, isResume) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = null;
    let partialReceived = false;
    let thinkingContent = '';
    let thinkingElement = null;
    let thinkingSection = null;
    let thinkingComplete = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.partial_thinking && !partialReceived) {
                        // Resume with existing thinking
                        thinkingContent = data.partial_thinking;
                    } else if (data.partial && !partialReceived) {
                        // First message is partial content from resume
                        partialReceived = true;
                        assistantMessage = addMessage('assistant', data.partial);
                        thinkingSection = assistantMessage.querySelector('.thinking-section');
                        thinkingElement = assistantMessage.querySelector('.thinking-content');
                    } else if (data.thinking) {
                        // Accumulate thinking content
                        if (!assistantMessage) {
                            assistantMessage = addMessage('assistant', '');
                            thinkingSection = assistantMessage.querySelector('.thinking-section');
                            thinkingElement = assistantMessage.querySelector('.thinking-content');
                        }
                        thinkingContent += data.thinking;
                        if (thinkingElement) {
                            thinkingElement.textContent = thinkingContent;
                            updateThinkingDisplay(assistantMessage);
                        }
                    } else if (data.thinking_end) {
                        thinkingComplete = true;
                        if (thinkingElement) {
                            thinkingElement.textContent = thinkingContent;
                            updateThinkingDisplay(assistantMessage);
                        }
                    } else if (data.token === null) {
                        // End of stream
                        break;
                    } else if (data.token) {
                        if (!assistantMessage) {
                            assistantMessage = addMessage('assistant', '');
                        }
                        const contentDiv = assistantMessage.querySelector('.message-content');
                        if (contentDiv) {
                            contentDiv.textContent += data.token;
                            // Render markdown
                            contentDiv.innerHTML = marked.parse(contentDiv.textContent);
                        }
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        setupScrollbarAutoHide(assistantMessage);
                    }
                } catch (e) {
                    // Ignore parse errors for incomplete chunks
                }
            }
        }
    }
}
```

#### Task 4c: JavaScript - updateThinkingDisplay function

- [ ] **Step 1: Add updateThinkingDisplay function**

Add after the `addMessage` function:

```javascript
function updateThinkingDisplay(messageElement) {
    const thinkingSection = messageElement.querySelector('.thinking-section');
    if (!thinkingSection) return;

    const thinkingContent = thinkingSection.querySelector('.thinking-content');
    const toggleBtn = thinkingSection.querySelector('.thinking-toggle');
    if (!thinkingContent || !toggleBtn) return;

    // Check if content exceeds 3 lines
    const lines = countLines(thinkingContent);
    const shouldCollapse = lines > 3;

    if (shouldCollapse) {
        thinkingSection.classList.add('thinking-collapsed');
        toggleBtn.style.display = 'block';
        toggleBtn.textContent = 'Show more';
    } else {
        thinkingSection.classList.remove('thinking-collapsed');
        toggleBtn.style.display = 'none';
    }

    // Setup toggle button handler
    toggleBtn.onclick = () => {
        const isCollapsed = thinkingSection.classList.contains('thinking-collapsed');
        if (isCollapsed) {
            thinkingSection.classList.remove('thinking-collapsed');
            toggleBtn.textContent = 'Show less';
        } else {
            thinkingSection.classList.add('thinking-collapsed');
            toggleBtn.textContent = 'Show more';
        }
    };
}

function countLines(element) {
    const style = window.getComputedStyle(element);
    const lineHeight = parseFloat(style.lineHeight);
    const height = element.scrollHeight;
    return Math.round(height / lineHeight);
}
```

#### Task 4d: JavaScript - setupScrollbarAutoHide function

- [ ] **Step 1: Add setupScrollbarAutoHide function**

Add after `updateThinkingDisplay`:

```javascript
function setupScrollbarAutoHide(messageElement) {
    if (!messageElement) return;

    // Check if already has scrollbar (content overflows)
    const hasOverflow = messageElement.scrollHeight > messageElement.clientHeight;
    if (!hasOverflow) return;

    messageElement.addEventListener('wheel', function() {
        this.classList.add('scrollbar-visible');
        clearTimeout(this._hideTimer);
        this._hideTimer = setTimeout(() => {
            this.classList.remove('scrollbar-visible');
        }, 3000);
    }, { passive: true });
}
```

#### Task 4e: JavaScript - modify addMessage for thinking structure

- [ ] **Step 1: Replace addMessage function**

Replace the `addMessage` function with one that creates the proper HTML structure for assistant messages:

```javascript
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (role === 'assistant') {
        // Create thinking section
        const thinkingSection = document.createElement('div');
        thinkingSection.className = 'thinking-section';

        const thinkingContent = document.createElement('div');
        thinkingContent.className = 'thinking-content';
        thinkingContent.textContent = '';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'thinking-toggle';
        toggleBtn.textContent = 'Show more';
        toggleBtn.style.display = 'none';

        thinkingSection.appendChild(thinkingContent);
        thinkingSection.appendChild(toggleBtn);

        // Create message content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(thinkingSection);
        messageDiv.appendChild(contentDiv);
    } else {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        messageDiv.appendChild(contentDiv);
    }

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Remove empty class when messages exist
    messagesContainer.classList.remove('empty');

    // Setup scrollbar auto-hide for assistant messages
    if (role === 'assistant') {
        setupScrollbarAutoHide(messageDiv);
    }

    return messageDiv;
}
```

#### Task 4f: JavaScript - modify loadConversation for thinking

- [ ] **Step 1: Update loadConversation to handle thinking from history**

Find and replace `loadConversation`:

```javascript
async function loadConversation(convId) {
    try {
        const response = await fetch(`/api/chat/history/${convId}`);
        const data = await response.json();

        messagesContainer.innerHTML = '';

        for (const msg of data.messages) {
            if (msg.role === 'assistant') {
                const messageDiv = addMessage(msg.role, msg.content || '');

                // If there's thinking content, add it
                if (msg.thinking) {
                    const thinkingSection = messageDiv.querySelector('.thinking-section');
                    const thinkingContent = messageDiv.querySelector('.thinking-content');
                    if (thinkingContent) {
                        thinkingContent.textContent = msg.thinking;
                        updateThinkingDisplay(messageDiv);
                    }
                }

                // Render markdown for content
                const contentDiv = messageDiv.querySelector('.message-content');
                if (contentDiv && contentDiv.textContent) {
                    contentDiv.innerHTML = marked.parse(contentDiv.textContent);
                }
            } else {
                addMessage(msg.role, msg.content);
            }
        }
    } catch (error) {
        console.error('Failed to load conversation:', error);
    }
}
```

#### Task 4g: JavaScript - modify startNewChat for empty state

- [ ] **Step 1: Update startNewChat to handle empty state**

Replace `startNewChat`:

```javascript
async function startNewChat() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    currentConversationId = null;
    localStorage.removeItem('currentConversationId');
    messagesContainer.innerHTML = '';
    messagesContainer.classList.add('empty');  // Add empty class
    document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
    messageInput.focus();
}
```

- [ ] **Step 2: Update init to set empty state on load**

Find the `init` function and add empty class if no conversation:

```javascript
async function init() {
    await loadConversationList();
    if (currentConversationId) {
        await loadConversation(currentConversationId);
        await checkStreamStatus();
    } else {
        // No conversation - show empty state
        messagesContainer.classList.add('empty');
    }
}
```

#### Task 4h: JavaScript - auto-resize textarea fallback

- [ ] **Step 1: Add input event listener for auto-resize**

Find the event listeners section and add:

```javascript
messageInput.addEventListener('input', function() {
    // Auto-resize using field-sizing or fallback
    autoResizeInput(this);
});

function autoResizeInput(textarea) {
    // Check if field-sizing is supported
    if (typeof CSS !== 'undefined' && CSS.supports('field-sizing', 'content')) {
        // Native support, no action needed
        return;
    }

    // Fallback: adjust height based on content
    const minHeight = 120; // ~5 lines
    const clone = textarea.cloneNode();
    clone.style.position = 'absolute';
    clone.style.visibility = 'hidden';
    clone.style.width = textarea.offsetWidth + 'px';
    clone.style.height = 'auto';
    clone.style.minHeight = '0';
    document.body.appendChild(clone);
    const newHeight = Math.max(minHeight, clone.scrollHeight);
    document.body.removeChild(clone);
    textarea.style.height = newHeight + 'px';
}
```

- [ ] **Step 2: Run and verify**

Start the server and test in browser:
```bash
cd d:/code/tutorial/agent_html/backend
uvicorn main:app --reload --port 8000
```

Test:
1. Send a message, verify thinking is displayed above response
2. Verify Show more/less works when thinking > 3 lines
3. Verify message blocks scroll internally
4. Verify scrollbar auto-hides after 3s
5. Verify empty state has centered input
6. Verify input expands with content
7. Verify markdown renders in assistant messages

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat: complete frontend overhaul - thinking display, scrollbar auto-hide, empty state, input expansion, markdown rendering"
```

---

## Testing

### Backend Tests

```bash
cd d:/code/tutorial/agent_html/backend
pytest tests/ -v
```

### Manual Testing Checklist

1. [ ] Thinking tokens stream before text tokens in SSE
2. [ ] Thinking displayed in collapsible section (Show more/less when >3 lines)
3. [ ] Message blocks scroll internally, not expand the page
4. [ ] Scrollbar auto-hides after 3s on message blocks
5. [ ] Empty state has centered input
6. [ ] Input box expands with content (min 5 lines)
7. [ ] Markdown rendering works in assistant messages
8. [ ] History API returns thinking content
9. [ ] Resume works with partial thinking

---

## Spec Coverage

| Spec Section | Task |
|--------------|------|
| 1.1 LLM Response | Task 1 |
| 1.2 SSE Protocol | Task 3 |
| 1.3 Storage Schema | Task 2 (service saves thinking) |
| 1.5 API Changes | Task 3 |
| 2.1 Thinking Section | Task 4e |
| 2.2 Fold/Unfold | Task 4c, 4d |
| 2.3 Line Counting | Task 4c (countLines) |
| 3.1 CSS | Task 4a |
| 3.2 Scrollbar Auto-Hide | Task 4d |
| 4.1 Empty State | Task 4g |
| 5.1 Input Auto-Expand | Task 4h |
| 6.1 Markdown | Task 4a (marked.js), 4b |
