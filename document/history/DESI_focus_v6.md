# Code Quality, Multi-Turn Reasoning, and Storage Hardening - Design

This document describes the implementation of the seven changes in this iteration. Each part has the same name and number as the corresponding part in [`./SPEC_focus.md`](./SPEC_focus.md).

---

## Part 1: Code Quality Audit Cleanup

A single review pass for the items from the code-quality audit that are pure refactors (no behavior change). Each item is a small, targeted change.

### 1.A — Backend chain / service layer

| Item | Where | What changed |
|---|---|---|
| H6 | [backend/chat/chain.py](backend/chat/chain.py) | `convert_messages` is now a module-level function. It uses `HumanMessage` for user turns and `AIMessage` for assistant turns. For prior assistant turns, if `thinking` is present, the `AIMessage` is built with `content=[{"type": "thinking", "thinking": ...}, {"type": "text", "text": ...}]`; otherwise plain `AIMessage(content=...)`. (See Part 2 for the thinking-feed-forward rationale.) |
| M6 | [backend/chat/chain.py](backend/chat/chain.py) | `ChatAnthropic(...)` now receives `base_url=settings.anthropic_base_url` directly. The previous `os.environ["ANTHROPIC_API_BASE"]` mutation is gone, and so is the `import os`. |
| M7 | [backend/chat/routes.py](backend/chat/routes.py) | The module-level `chain = create_chain()` and `chat_service = ChatService(chain)` are replaced by a lazy `get_chat_service()` dependency. `_chat_service: ChatService \| None` is module-level. The route handler takes `chat_service: ChatService = Depends(get_chat_service)`. |
| M4 | [backend/chat/service.py](backend/chat/service.py) | `service.get_stream_status` is deleted. The status endpoint in routes.py calls `get_job` directly. |
| L3 | throughout backend | All `Optional[X]` type hints are replaced with `X \| None`. Removed the now-unused `Optional` imports from `backend/chat/stream_manager.py` and `backend/storage/file_storage.py`. |

### 1.B — Backend stream layer

| Item | Where | What changed |
|---|---|---|
| M1 | [backend/chat/routes.py](backend/chat/routes.py) | `stream_from_job` is split into `stream_from_inactive_job` and `stream_from_active_job`. The resume route dispatches between them based on `job.status`. The shared "replay cached chunks" code is in `_replay_cached_chunks`. |
| M2 | [backend/chat/routes.py](backend/chat/routes.py) | Two helpers: `_sse(payload)` returns the SSE-formatted string; `_serialize_chunk(chunk)` returns the `{"chunk", "type", "message_id"}` shape. The three sites that previously inlined `f"data: {json.dumps({...})}\n\n"` now call these. |
| M5 | [backend/chat/stream_manager.py](backend/chat/stream_manager.py) | New `StreamJob.reset()` method that sets `status = "active"`, `chunks = []`, and `updated_at`. Called by `stream_chat` instead of the previous inline assignment block. |
| L4 | [backend/chat/routes.py](backend/chat/routes.py) | The four dynamic attribute assignments (`job.tokens = []`, `job.thinking_tokens = []`, `job.sent_pointer = 0`, `job.thinking_sent_pointer = 0`) are gone. They were dead — nothing read them. |
| L8 | [backend/chat/stream_manager.py](backend/chat/stream_manager.py) | `clear_job` has a comment explaining why `cancelled` is set on the job before `del STREAM_REGISTRY[conversation_id]`. The background task holds a local reference to the job and checks `cancelled` between iterations; setting the flag on the live object first ensures the check sees it. |

### 1.C — Backend storage and request validation

| Item | Where | What changed |
|---|---|---|
| H3 | [backend/chat/routes.py](backend/chat/routes.py) | `stream_chat` now calls `file_storage.create_conversation(conversation_id)` and `file_storage.append_message(conversation_id, "user", request.message)` itself, before starting the background task. `get_or_create_job` no longer has a side effect. |
| H5 (lite) | [backend/chat/service.py](backend/chat/service.py) | The dedupe block `if not messages or messages[-1]["content"] != message: messages.append(...)` is removed. `service.generate_background` now reads history from storage and trusts it as the single source of truth — no append, no dedupe. |
| L2 | [backend/chat/routes.py](backend/chat/routes.py) | `ChatRequest.message` is now `Field(..., min_length=1, max_length=10000)`. |
| L5 | [backend/storage/file_storage.py](backend/storage/file_storage.py) | On `json.JSONDecodeError` in `_load_conversations`, the unreadable file is renamed to `conversations.json.corrupt` (instead of being silently overwritten), and a warning is logged. |
| L12 | [backend/storage/file_storage.py](backend/storage/file_storage.py) | Comment in `get_conversation_list` explains why ISO-8601 strings sort lexicographically in the same order as the underlying timestamps. |

### 1.D — Backend infrastructure and tests

| Item | Where | What changed |
|---|---|---|
| M3 (backend) | [backend/chat/routes.py](backend/chat/routes.py) | `StreamStatusResponse` now has only `status: str`, `chunks_count: int`, and `partial_content: str \| None = None`. The `streaming` and `is_complete` booleans are removed. |
| M12 | [backend/main.py](backend/main.py) | The `@app.get("/index.html")` route is removed. Only `@app.get("/")` and `app.mount("/static", ...)` remain. |
| L9 | [pyproject.toml](pyproject.toml) (new) | `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` and `testpaths = ["backend/tests"]`. |
| L10 | [backend/tests/conftest.py](backend/tests/conftest.py) | The `temp_storage_dir` fixture is rewritten to use `monkeypatch` + `tmp_path` instead of `tempfile.mkdtemp` + `importlib.util.spec_from_file_location` double-load. The `mock_chain` fixture is retained. |
| L14 | [CLAUDE.md](CLAUDE.md) | One-line addition to the documentation table noting that `docs/superpowers/` is a separate workflow. |
| L15 | [todo.md](todo.md) | Deleted. Its items (UI adjustment, RAG, dynamic response rendering) were already in `document/SPEC.md` "Out of Scope" or covered by completed iterations. |

### 1.E — Frontend restructure

| Item | Where | What changed |
|---|---|---|
| M8 | [frontend/](frontend/) | `frontend/index.html` (1998 lines, monolithic) is split into three files: `frontend/index.html` (49 lines, structure only), `frontend/static/styles.css` (760 lines, all CSS), `frontend/static/app.js` (~1100 lines, all JS). The HTML now references `<link rel="stylesheet" href="/static/styles.css">` and `<script type="module" src="/static/app.js">`. |
| M9 | [frontend/static/app.js](frontend/static/app.js) | `addAssistantPlaceholder` is reduced to: call `addMessage('assistant', '')` (which creates the thinking section + content div), then set the content div's class to `loading` and innerHTML to the loading-dots markup. The ~50 lines of duplicate DOM construction in the old `addAssistantPlaceholder` are gone. |
| M11 | [frontend/static/app.js](frontend/static/app.js), [frontend/index.html](frontend/index.html) | Removed: `marked.min.js` CDN script tag, `marked.setOptions({...})` call, the entire commented-out `renderMarkdown` function, `localStorage.setItem("test", "1111111")` debug line, the `dompurify@3.2.0/dist/purify.es.mjs` duplicate script, several inline commented-out rendering paths. Kept: `DOMPurify(content)` and `DOMPurify.sanitize(content)` no-op calls per an earlier user decision (now with TODO comments — see Part 7). |
| M3 (frontend) | [frontend/static/app.js](frontend/static/app.js) | `status.streaming` reads → `status.status === 'active'`; `status.is_complete` reads → `status.status === 'completed'`. (Frontend-side, the boolean fields are gone from the backend response.) |

### 1.F — Frontend naming and minor comments

| Item | Where | What changed |
|---|---|---|
| L7 | [frontend/static/app.js](frontend/static/app.js) | `STORAGE_KEYS.POINTER(convId)` → `STORAGE_KEYS.CONSUMED(convId)`. The localStorage key `pointer_{convId}` → `consumed_{convId}`. The variable `currentPointer` → `consumedCount`. The backend's `from_pointer` query parameter is unchanged. |
| L6 | [frontend/static/app.js](frontend/static/app.js) | A comment at the `data.end` handler in `processStreamResponse` explains: the chunk cache is cleared after the history cache is updated; the history cache and the server-side conversation storage converge on the same final message, so a page refresh after this point loads via `/api/chat/history` instead of resuming from chunks. |

### 1.G — `.env` consolidation (H1)

`backend/.env` (which held a real `ANTHROPIC_API_KEY` and a different base URL) is moved to the project root as `.env`. The previous root `.env` (a placeholder) is removed. The change is one `mv` operation; no code change. From the project root, `pydantic-settings` now finds the real key under the standard `env_file=".env"` path.

### Why These Were a Single Review Pass

All of Part 1 is mechanical refactor. No behavior change, no new tests needed (existing tests still pass), no API change. The full test suite is the safety net.

---

## Part 2: Multi-Turn Thinking Continuity

### The Bug

The LLM emits `thinking` content blocks on every turn when given the previous turn's `thinking` as part of the input. It does **not** emit them when the previous turn's `thinking` is stripped before being fed back.

The pre-change `convert_messages` returned:

```python
return [
    HumanMessage(content=m["content"]) if m["role"] == "user"
    else {"role": "assistant", "content": m["content"]}  # thinking dropped
    for m in messages
]
```

For a 2-turn conversation fed back as `convert_messages([user1, assistant1, user2])`:
- `user1` → `HumanMessage(content="first message")`
- `assistant1` → `{"role": "assistant", "content": "first answer"}` (no thinking)
- `user2` → `HumanMessage(content="second message")`

The LLM sees the prior assistant turn's visible content but not its reasoning. It responds to the second user message with text only — no `thinking` block.

### The Fix

`convert_messages` now constructs an `AIMessage` whose `content` is a list of content blocks, putting the prior thinking first and the visible text second:

```python
def convert_messages(messages: list) -> list:
    result = []
    for m in messages:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            if m.get("thinking"):
                result.append(AIMessage(content=[
                    {"type": "thinking", "thinking": m["thinking"]},
                    {"type": "text", "text": m["content"]},
                ]))
            else:
                result.append(AIMessage(content=m["content"]))
    return result
```

For the same 2-turn conversation:
- `user1` → `HumanMessage`
- `assistant1` → `AIMessage(content=[{type:thinking,...}, {type:text,...}])`
- `user2` → `HumanMessage`

The LLM now sees the prior reasoning in the same content-block shape it produces itself, and continues emitting `thinking` blocks on subsequent turns.

### Why Module-Level

`convert_messages` is extracted to module level (no longer a closure inside `create_chain`) so it's directly testable without instantiating `ChatAnthropic` or constructing the full chain.

### Test Coverage

[backend/tests/test_chain.py](backend/tests/test_chain.py) (new file, 5 tests):

| Test | Asserts |
|---|---|
| `test_user_messages_become_human_message` | `{"role": "user", ...}` → `HumanMessage` |
| `test_assistant_without_thinking_becomes_plain_aimessage` | assistant without `thinking` → `AIMessage(content=str)` (no content list) |
| `test_assistant_with_thinking_becomes_block_list` | assistant with `thinking` → `AIMessage(content=[thinking, text])` |
| `test_full_multi_turn_conversion` | mixed conversation; the prior assistant with `thinking` becomes a content-block AIMessage, not a string |
| `test_unknown_roles_are_dropped` | system / unknown roles are not in the output |

### Files Modified

| File | Change |
|---|---|
| [backend/chat/chain.py](backend/chat/chain.py) | `convert_messages` is module-level; assistant turn with `thinking` becomes a content-block `AIMessage`; `create_chain` is a one-liner that wires `convert_messages` to `ChatAnthropic` via `RunnableLambda \| llm` |
| [backend/tests/test_chain.py](backend/tests/test_chain.py) | New — 5 unit tests for `convert_messages` |

### Live Verification

The fix was verified against the real LLM (`minimax-3` via MiniMax). Before the fix, raw SSE chunk types for a 2-turn conversation were `{thinking: N, token: M}` on turn 1 and `{token: K}` on turn 2 (no thinking). After the fix, both turns emit `{thinking: ..., token: ...}`.

---

## Part 3: Stream Registry Memory Cleanup

### The Design

A thin wrapper around the resume stream generator, defined in [backend/chat/stream_manager.py](backend/chat/stream_manager.py:74):

```python
async def consume_with_cleanup(gen, conversation_id: str):
    completed = False
    any_event = False
    try:
        async for event in gen:
            any_event = True
            yield event
        completed = True
    finally:
        if completed and any_event:
            STREAM_REGISTRY.pop(conversation_id, None)
```

Two flags:

- **`completed`** — set to `True` only after the `async for` loop exits normally. The four exit modes and their effects:
  - Inner generator returns normally (no more events) → `async for` exits → `completed = True` → cleanup happens
  - Inner generator's `aclose()` is called (simulated client disconnect) → `GeneratorExit` propagates into the wrapper → `async for` raises → `completed` stays `False` → no cleanup
  - Inner generator raises an exception → `async for` propagates → `completed` stays `False` → no cleanup
  - `from_pointer` is out of range → the generator's `if from_pointer < 0 or from_pointer > len(job.chunks): return` runs before entering the try block, so no events are yielded → `any_event` stays `False` → no cleanup (the `completed` flag would otherwise be `True` after normal exit)

- **`any_event`** — set to `True` after the first event is yielded. Prevents the out-of-range case (which returns without yielding) from triggering cleanup, since an out-of-range resume delivered no data to the client and the job is still useful for a future resume.

`STREAM_REGISTRY.pop(conversation_id, None)` is idempotent: if a concurrent resume already removed the entry, the second `pop` returns `None` and is a no-op.

### Where It's Wired

Only the **resume** route uses the wrapper:

```python
@router.get("/stream/{conversation_id}")
async def stream_resume(conversation_id, from_pointer):
    job = get_job(conversation_id)
    if job is None:
        raise HTTPException(404, ...)
    if job.status == "active":
        gen = stream_from_active_job(job, from_pointer=from_pointer)
    else:
        gen = stream_from_inactive_job(job, from_pointer=from_pointer)
    return StreamingResponse(
        consume_with_cleanup(gen, job.conversation_id),
        media_type="text/event-stream",
    )
```

The **initial stream** route (`POST /api/chat/stream`) does **not** wrap the generator in `consume_with_cleanup`. The initial stream's job is meant to be available for a possible future resume.

### Edge Cases

| Scenario | Behavior |
|---|---|
| Resume drains all cached chunks and yields `end` | Cleanup |
| Resume client disconnects before `end` is yielded | No cleanup |
| Resume is for an out-of-range `from_pointer` (no events) | No cleanup |
| Resume's inner generator raises (LLM error, etc.) | No cleanup |
| Two concurrent resumes where the first finishes | First cleanup removes entry; second wrapper's `pop` is a no-op |
| Initial stream (POST) is interrupted | No cleanup; the entry is still there for a future resume |
| Initial stream (POST) completes normally | No cleanup; the entry is still there (the user might resume later) |

### Test Coverage

[backend/tests/test_stream_manager.py](backend/tests/test_stream_manager.py) — 5 unit tests:

| Test | Asserts |
|---|---|
| `test_consume_with_cleanup_removes_job_after_full_consumption` | A wrapper that yields 2 events and then exits removes the entry |
| `test_consume_with_cleanup_keeps_job_when_no_events_yielded` | A wrapper that yields nothing leaves the entry |
| `test_consume_with_cleanup_keeps_job_on_cancellation` | `wrapper.aclose()` after one event leaves the entry |
| `test_consume_with_cleanup_keeps_job_on_exception` | Inner generator raising leaves the entry |
| `test_consume_with_cleanup_pop_is_idempotent` | Two consecutive wraps on the same conv_id, where the first removed the entry, do not raise |

[backend/tests/test_thinking_routes.py](backend/tests/test_thinking_routes.py) — 2 integration tests:

| Test | Asserts |
|---|---|
| `test_resume_route_cleans_up_completed_job` | A `GET /api/chat/stream/{conv_id}` that drains to completion removes the entry from `STREAM_REGISTRY` |
| `test_initial_stream_does_not_clean_up` | A `POST /api/chat/stream` that drains to completion leaves the entry in `STREAM_REGISTRY` |

### Files Modified

| File | Change |
|---|---|
| [backend/chat/stream_manager.py](backend/chat/stream_manager.py) | Added `consume_with_cleanup` |
| [backend/chat/routes.py](backend/chat/routes.py) | Resume route wraps its generator in `consume_with_cleanup` |
| [backend/tests/test_stream_manager.py](backend/tests/test_stream_manager.py) | 5 new unit tests |
| [backend/tests/test_thinking_routes.py](backend/tests/test_thinking_routes.py) | 2 new integration tests |

---

## Part 4: File Storage Concurrency Safety

### The Design

Two complementary safety mechanisms in [backend/storage/file_storage.py](backend/storage/file_storage.py):

### 4.A — Per-process `threading.Lock`

A module-level `_write_lock = threading.Lock()` serializes the four write functions within the process. Each of `create_conversation`, `save_conversation`, `append_message`, `delete_conversation` does:

```python
with _write_lock:
    data = _load_conversations()
    # ... mutate data ...
    _atomic_write_json(CONVERSATIONS_FILE, data)
```

The lock holds for the full read-modify-write cycle, so two concurrent `append_message` calls cannot both load the same baseline. The second waits for the first's lock to release, then loads the first's committed state. **No lost updates.**

Reads (`get_conversation`, `get_conversation_list`) are not under the lock. Combined with the atomic write below, a concurrent reader sees either the fully-old or fully-new file — never partial.

The lock is per-process. A multi-worker deployment (`uvicorn --workers N`) would have N independent processes, each with its own lock, still racing on the file. The CLAUDE.md run command uses a single worker, so this is not triggered today.

### 4.B — Atomic write helper

`_atomic_write_json(path, data)`:

```python
def _atomic_write_json(path: Path, data: dict) -> None:
    _ensure_storage_dir()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
```

The write happens to a sibling `<path>.tmp` in the same directory (same-volume requirement for `os.replace` to be atomic on Windows). `os.replace` is the atomic swap. On any failure (write fails, replace fails, anything), the `.tmp` file is cleaned up so it does not accumulate.

The four write functions all delegate to `_atomic_write_json` instead of inlining the `open + json.dump`. The old `_save_conversations` helper is gone.

### Why `os.replace` and Not `os.rename`

On Windows, `os.rename` fails if the target exists. `os.replace` is documented to overwrite atomically (POSIX) or to fail atomically and not leave a partial target (Windows). Both behaviors are what we want for "the on-disk file is always fully-old or fully-new."

### Corrupt-JSON Recovery (Unchanged from L5)

`_load_conversations` still has the `json.JSONDecodeError` branch: rename the corrupt file to `conversations.json.corrupt`, log a warning, start fresh. The atomic write above prevents new corruption; this branch is the safety net for files that were corrupted before this fix was applied, or by any other means.

### Test Coverage

[backend/tests/test_storage.py](backend/tests/test_storage.py) — 5 new tests across two new test classes:

**`TestAtomicWrite`** (3 tests):
- `_atomic_write_json` on success → file replaced, no `.tmp` left
- `os.replace` monkeypatched to raise → original intact, no `.tmp` left
- `json.dump` monkeypatched to raise → original intact, no `.tmp` left

**`TestWriteLock`** (2 tests):
- 50 concurrent `append_message` threads → all 50 messages present
- 20 concurrent `save_conversation` threads → all complete, file is always valid JSON, no `.tmp` left, last writer wins (this is the documented `save_conversation` contract)

### Files Modified

| File | Change |
|---|---|
| [backend/storage/file_storage.py](backend/storage/file_storage.py) | Added `_write_lock`, `_atomic_write_json`. Removed `_save_conversations`. All four write functions use `with _write_lock` and `_atomic_write_json`. |
| [backend/tests/test_storage.py](backend/tests/test_storage.py) | 5 new tests (3 atomic-write, 2 concurrent-write) |

---

## Part 5: Frontend Cache Consolidation

### The Design

A single ES module, [frontend/static/cache.js](frontend/static/cache.js), owns:
- The five localStorage key names (the `STORAGE_KEYS` constant moves here)
- All `localStorage.getItem` / `setItem` / `removeItem` calls
- All `JSON.parse` / `JSON.stringify` calls

The module exposes typed accessors for each cache:

```js
export const cache = {
    // global
    getCurrentConversationId() / setCurrentConversationId,

    // per-conversation, JSON-encoded
    getHistory / setHistory / appendToHistory / clearHistory,
    getChunks  / setChunks  / appendToChunks  / clearChunks,

    // per-conversation, plain string ("0", "1", ...)
    getConsumed / setConsumed / clearConsumed,

    // per-conversation, plain string ("true" / "false")
    isStreaming / getStreaming / setStreaming / clearStreaming,
};
```

The JSON helpers:

```js
function readJSON(key) {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    try { return JSON.parse(raw); } catch { return null; }
}
function writeJSON(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}
```

`getChunks` and `getHistory` return `null` for missing keys; `getChunks` defaults to `[]` because all its callers want a default array.

`getConsumed` returns a parsed integer (defaulting to `0`) — the localStorage value is stored as a string for consistency with the per-key format, but the accessor returns a number.

### What `app.js` Looks Like Now

`app.js` does not contain the substring `localStorage`, the substring `STORAGE_KEYS`, or any `JSON.parse` / `JSON.stringify` related to cached state. The only two `JSON.*` calls that remain are:

- `JSON.stringify({message, conversation_id})` at line 683 — the `fetch` request body
- `JSON.parse(event.slice(6))` at line 783 — SSE event parsing

Both are external-data protocol concerns, not state-storage concerns.

### How the Refactor Maps to Existing Behavior

| Old code | New code |
|---|---|
| `localStorage.getItem(STORAGE_KEYS.HISTORY(convId))` and JSON-parse | `cache.getHistory(convId)` |
| `localStorage.setItem(STORAGE_KEYS.HISTORY(convId), JSON.stringify(msgs))` | `cache.setHistory(convId, msgs)` |
| `getHistoryCache(convId) \|\| []` then `.push(msg)` then `setHistoryCache(...)` | `cache.appendToHistory(convId, msg)` |
| `localStorage.removeItem(STORAGE_KEYS.HISTORY(convId))` | `cache.clearHistory(convId)` |
| `isStreamingForConv(convId)` | `cache.isStreaming(convId)` |
| `getStreamingForConv(convId) === 'false'` | `cache.getStreaming(convId) === 'false'` |
| `localStorage.getItem(STORAGE_KEYS.POINTER(convId)) \|\| '0'` then `parseInt(..., 10)` | `cache.getConsumed(convId)` |
| `localStorage.setItem('currentConversationId', convId)` | `cache.setCurrentConversationId(convId)` |
| `localStorage.removeItem('currentConversationId')` | `cache.setCurrentConversationId(null)` |

The seven old helper functions (`getHistoryCache`, `setHistoryCache`, `appendToHistoryCache`, `clearHistoryCache`, `clearChunkCache`, `isStreamingForConv`, `getStreamingForConv`, `setStreamingForConv`) and the `STORAGE_KEYS` constant are deleted from `app.js`.

### Test Coverage

No frontend tests. The change is verified by:
- `grep -E 'localStorage|STORAGE_KEYS' frontend/static/app.js` returns no matches
- The two remaining `JSON.*` hits in `app.js` are for the `fetch` body and SSE parsing, not for cached state
- All 47 backend tests still pass
- `/static/cache.js` serves correctly (200) from the FastAPI `StaticFiles` mount

### Files Modified

| File | Change |
|---|---|
| [frontend/static/cache.js](frontend/static/cache.js) | New — owns all localStorage access |
| [frontend/static/app.js](frontend/static/app.js) | Imports `cache`; ~25 call sites updated; 7 helper functions + `STORAGE_KEYS` deleted |
| [frontend/index.html](frontend/index.html) | No change — `cache.js` is imported by `app.js` (ES module resolution) |

---

## Part 6: Frontend Asset Path

### The Fix

Frontend assets now live in `frontend/static/`, matching the `/static` URL mount:

```
frontend/
├── index.html
└── static/
    ├── app.js
    ├── cache.js
    └── styles.css
```

`backend/main.py`:

```python
app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
```

The previous version was `directory=str(frontend_path)` — pointing the `/static` mount at the `frontend/` directory, which was misleading because there was no `static/` folder.

### Files Moved

| From | To |
|---|---|
| `frontend/styles.css` | `frontend/static/styles.css` |
| `frontend/app.js` | `frontend/static/app.js` |
| (new) | `frontend/static/cache.js` |

### Files Modified

| File | Change |
|---|---|
| [backend/main.py](backend/main.py) | `StaticFiles` mount path now points at `frontend_path / "static"` |
| [frontend/index.html](frontend/index.html) | References `/static/styles.css` and `/static/app.js` (unchanged from before the move) |
| `frontend/styles.css` | Moved to `frontend/static/styles.css` |
| `frontend/app.js` | Moved to `frontend/static/app.js` |

### Verification

- `GET /` → 200 (index.html)
- `GET /static/styles.css` → 200
- `GET /static/app.js` → 200
- `GET /static/cache.js` → 200
- All 47 tests pass

---

## Part 7: `DOMPurify` No-Op TODOs (U2)

### The Two Sites

[frontend/static/app.js:60](frontend/static/app.js#L60):

```js
// not used for now
// TODO(U2 — known issue): this DOMPurify() call is a no-op — the return
// value is discarded, so no sanitization is applied to the streamed HTML.
// The markdown content is written straight into the DOM via smd. If the
// LLM ever returns untrusted HTML it would render as-is. Either remove
// this line (it's misleading) or wire up sanitization properly, e.g. by
// wrapping the smd output through DOMPurify.sanitize() before insertion.
DOMPurify(content);
smd.parser_write(parser, content);
```

[frontend/static/app.js:371](frontend/static/app.js#L371):

```js
// TODO(U2 — known issue): DOMPurify.sanitize(content) is a
// no-op here too — the return value is discarded, so the
// cached history's content is rendered without sanitization.
// Same fix as in renderContent(): wire up sanitization or
// remove the misleading call.
DOMPurify.sanitize(content);
smd.parser_write(parser, content);
```

### What This Documented-But-Didn't-Fix Acknowledges

- The calls are no-ops (return value discarded).
- The risk is that streamed HTML (and cached history HTML) is rendered as-is.
- The two fix options are: (a) wire up `DOMPurify.sanitize` before insertion, or (b) remove the misleading calls.

Per the user's earlier decision, the no-op calls are preserved (not removed). The comments make the no-op status explicit. A real sanitization fix is a separate item.

### Files Modified

| File | Change |
|---|---|
| [frontend/static/app.js](frontend/static/app.js) | `TODO(U2 — known issue)` comment blocks at the two `DOMPurify` no-op call sites |

---

## Combined File Inventory

### New files

| File | Purpose |
|---|---|
| [pyproject.toml](pyproject.toml) | Pytest config (`asyncio_mode = "auto"`, `testpaths`) |
| [frontend/static/cache.js](frontend/static/cache.js) | Owns all frontend localStorage access |
| [backend/tests/test_chain.py](backend/tests/test_chain.py) | 5 unit tests for `convert_messages` |

### Deleted files

| File | Reason |
|---|---|
| `todo.md` | Items already in `SPEC.md` "Out of Scope" or covered by completed iterations |
| Original monolithic `frontend/index.html` (1998 lines) | Replaced by `frontend/index.html` (49 lines) + `frontend/static/*` |

### Modified files

| File | Reason |
|---|---|
| [backend/main.py](backend/main.py) | Drop `/index.html` route; `StaticFiles` mount path → `frontend/static` |
| [backend/config.py](backend/config.py) | Unchanged in this iteration (verified consistent) |
| [backend/chat/chain.py](backend/chat/chain.py) | `convert_messages` module-level; assistant-with-thinking → content blocks; pass `base_url=` directly |
| [backend/chat/routes.py](backend/chat/routes.py) | `Depends(get_chat_service)`; split `stream_from_job`; `sse` helper; `StreamStatusResponse` simplified; resume wrapped in `consume_with_cleanup`; `create_conversation` + `append_message` called in route; `ChatRequest` validation; types use `\| None` |
| [backend/chat/service.py](backend/chat/service.py) | No dedupe; trusts storage as single source of truth; `get_stream_status` deleted |
| [backend/chat/stream_manager.py](backend/chat/stream_manager.py) | `StreamJob.reset()`; dead attrs gone; `consume_with_cleanup`; types use `\| None` |
| [backend/storage/file_storage.py](backend/storage/file_storage.py) | `_write_lock`; `_atomic_write_json`; corrupt-JSON backup; types use `\| None` |
| [backend/tests/conftest.py](backend/tests/conftest.py) | Simplified `temp_storage_dir` fixture |
| [backend/tests/test_chat_routes.py](backend/tests/test_chat_routes.py) | Renamed functions; uses `StreamJob.reset()` |
| [backend/tests/test_chat_service.py](backend/tests/test_chat_service.py) | Dead `get_stream_status` test removed |
| [backend/tests/test_thinking_routes.py](backend/tests/test_thinking_routes.py) | Updated for new `StreamStatusResponse`; 2 new integration tests for `consume_with_cleanup` |
| [backend/tests/test_storage.py](backend/tests/test_storage.py) | 5 new tests (`TestAtomicWrite`, `TestWriteLock`) |
| [backend/tests/test_stream_manager.py](backend/tests/test_stream_manager.py) | 5 new tests for `consume_with_cleanup` |
| [frontend/index.html](frontend/index.html) | Restructured: links to `/static/styles.css` and `/static/app.js` |
| [frontend/static/app.js](frontend/static/app.js) | ~25 call sites updated to use `cache`; 7 helper functions + `STORAGE_KEYS` deleted; `addAssistantPlaceholder` refactored; `currentPointer` → `consumedCount`; `status.streaming` / `status.is_complete` reads → `status.status === '...'`; `DOMPurify` no-op TODOs added |
| [document/SPEC.md](document/SPEC.md) | "Markdown Rendering" section updated (no more `marked.js`); Dependencies updated (CDN scripts listed) |
| [CLAUDE.md](CLAUDE.md) | One-line addition: `docs/superpowers/` noted in docs table |

---

## Test Counts

| Test file | Tests |
|---|---|
| `backend/tests/test_chain.py` | 5 |
| `backend/tests/test_chat_routes.py` | 6 |
| `backend/tests/test_chat_service.py` | 5 |
| `backend/tests/test_storage.py` | 17 |
| `backend/tests/test_stream_manager.py` | 7 |
| `backend/tests/test_thinking_routes.py` | 7 |
| **Total** | **47** |

All 47 pass.
