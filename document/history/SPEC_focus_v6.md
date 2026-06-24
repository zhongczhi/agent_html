# Code Quality, Multi-Turn Reasoning, and Storage Hardening - Specification

This iteration bundles the changes that resulted from a code-quality audit, plus three behavioral fixes that came out of exercising the audit's findings:

1. **Code quality audit cleanup** (Part 1) — a batch of small refactors across the backend and frontend, with no behavior change. See [`./DESI_focus.md`](./DESI_focus.md) Part 1 for the full item-by-item list.
2. **Multi-turn thinking continuity** (Part 2) — fix: the second and later turns in a conversation were produced without any `thinking` content. Cause: prior assistant `thinking` was stripped before being fed back to the LLM. Fix: feed it back.
3. **Stream registry memory cleanup** (Part 3) — fix: `STREAM_REGISTRY` kept completed `StreamJob` objects forever. Fix: remove the entry from the registry after a resume call drains the cached chunks end-to-end.
4. **File storage concurrency safety** (Part 4) — fix: storage writers were neither serialized (lost-update hazard) nor crash-safe (a SIGKILL mid-write left a partial JSON on disk). Fix: a per-process `threading.Lock` plus an atomic write helper.
5. **Frontend cache consolidation** (Part 5) — refactor: every `localStorage` access in the frontend was scattered across `app.js` (raw `getItem` / `setItem` / `JSON.parse` / `JSON.stringify` calls). Fix: encapsulate all of it behind a single `cache` module.
6. **Frontend asset path** (Part 6) — fix: the `index.html` URL path `/static/...` did not match the actual folder `frontend/`. Fix: move the assets into a `frontend/static/` subdirectory and serve from there.
7. **`DOMPurify` no-op TODOs** (Part 7) — documentation: two `DOMPurify(content)` / `DOMPurify.sanitize(content)` call sites in `app.js` are no-ops (return value discarded). They were preserved per an earlier user decision, but the misleading silent failure now carries an explicit `TODO(U2 — known issue)` comment at each call site.

---

## Part 1: Code Quality Audit Cleanup

### Background

A full-codebase audit identified a long list of small issues: dead code, redundant state, scattered `localStorage` access, mixed type-hint styles, and tight coupling between modules. Some of the items in the audit are addressed in this iteration (Parts 2–7) or in the upcoming D-series items (Parts 3–5). The remaining items are pure refactors with no behavior change and are bundled here for a single review pass. The exact list of changes is in [`./DESI_focus.md`](./DESI_focus.md) Part 1.

### Functional Requirements

Each item in the audit list is implemented as a refactor: same behavior, less code, no API change. Because the requirements are "do the same thing in less code," the functional requirement is:

| ID | Requirement |
|----|-------------|
| FR-1.1 | All backend tests in `backend/tests/` continue to pass without modification |
| FR-1.2 | The 1998-line monolithic `frontend/index.html` is no longer present; frontend assets live in `frontend/static/` as separate files |
| FR-1.3 | All frontend localStorage access goes through a single module; no `localStorage` references remain in `app.js` |
| FR-1.4 | The `STREAM_REGISTRY` module exposes `get_or_create_job`, `get_job`, `clear_job`, and `consume_with_cleanup` only; the dead `tokens` / `thinking_tokens` / `sent_pointer` / `thinking_sent_pointer` dynamic attributes that routes.py was assigning to `job` are gone |
| FR-1.5 | The `StreamStatusResponse` carries only `status` (string), `chunks_count` (int), and `partial_content` (string \| null) — no `streaming` or `is_complete` booleans |
| FR-1.6 | The frontend's "pointer" concept is renamed from `pointer` to `consumed` throughout (localStorage key, variable, comments) |

### Acceptance Criteria

- [ ] `pytest backend/tests/ -v` → 47 / 47 pass after the refactor
- [ ] `grep localStorage frontend/static/app.js` returns no matches
- [ ] `grep -E 'tokens|thinking_tokens|sent_pointer|thinking_sent_pointer' backend/chat/` returns no matches
- [ ] `grep -E '"streaming"|"is_complete"' backend/chat/routes.py` returns no matches (only `streaming` flag in `cache.js`, which is a different file)

---

## Part 2: Multi-Turn Thinking Continuity

### Bug

In a multi-turn conversation, the **first** assistant turn includes the model's `thinking` content (rendered in the `thinking-section` of the assistant message), but every **subsequent** turn in the same conversation produces no `thinking` content at all — the model's reasoning section is empty for turns 2 and beyond. The user-visible symptom: the thinking box for the first turn is populated, but the thinking box for every later turn is empty.

### Root Cause (in the code, not the LLM)

The LLM *does* emit `thinking` content blocks on every turn when given the previous turn's `thinking` as part of the input. It does *not* emit them when the previous turn's `thinking` is stripped before being fed back.

The previous conversion of history to messages:

```python
return [
    HumanMessage(content=m["content"]) if m["role"] == "user"
    else {"role": "assistant", "content": m["content"]}
    for m in messages
]
```

fed back only the visible `content`, dropping `thinking` for every prior assistant message. The result: the LLM sees only the visible conversation and emits `thinking` only on the first turn.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-2.1 | On every turn of a multi-turn conversation, the assistant response includes at least one `thinking` content chunk (model-dependent, but should match the first turn's behavior) |
| FR-2.2 | The previous assistant's `thinking` content is preserved when fed back to the LLM, so the chain of reasoning can continue across the conversation |
| FR-2.3 | Prior assistant messages still have only their visible `content` available to the LLM as the assistant's "output" (so the model sees a coherent conversation history) |
| FR-2.4 | The change is contained to message conversion; no other code paths are affected |

### Acceptance Criteria

- [ ] With the real LLM, turn 2 of a conversation produces a non-empty `thinking` field in storage (verified manually: the storage `messages[3].thinking` for a 2-turn conversation is non-empty)
- [ ] Turn 3+ also produces non-empty `thinking` (same condition)
- [ ] `pytest backend/tests/test_chain.py` → 5 / 5 pass (covers the new `convert_messages` behavior, including: assistant-without-thinking → plain AIMessage; assistant-with-thinking → AIMessage with content block list `[thinking, text]`; multi-turn scenario; unknown roles dropped)
- [ ] All 47 existing tests continue to pass

---

## Part 3: Stream Registry Memory Cleanup

### Problem

`STREAM_REGISTRY: Dict[str, StreamJob]` in [backend/chat/stream_manager.py](backend/chat/stream_manager.py) was never garbage-collected. Every `StreamJob` that was ever created lived in the registry for the lifetime of the process. `clear_job` (called by `DELETE /api/chat/conversation/{id}`) was the only thing that removed entries.

The reason entries had to be kept around was to support **stream resume**: if the user switches away or refreshes during a stream, the frontend can re-attach to the in-memory `StreamJob` and read the remaining cached chunks via `from_pointer`. But the registry did not distinguish between "needed for resume" and "no longer needed" — every job, including fully-completed ones that no resume would ever want, lived forever.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-3.1 | A `StreamJob` is removed from `STREAM_REGISTRY` after a **successful** resume call has delivered the full cached chunk history (including the `end` marker) |
| FR-3.2 | A resume that is interrupted (client disconnect, browser navigation, `aclose()` on the wrapper), returns early without yielding the `end` marker, or raises an exception leaves the job in place — a future resume must be able to continue |
| FR-3.3 | The cleanup is applied only to the **resume** route (`GET /api/chat/stream/{conversation_id}`). The **initial** stream (`POST /api/chat/stream`) does not trigger cleanup, so a later resume is still possible if the initial stream is interrupted |
| FR-3.4 | The first resume that completes wins the deletion; a second concurrent resume whose `pop` finds the registry entry already gone is a no-op (no error) |

### Acceptance Criteria

- [ ] After a `GET /api/chat/stream/{conv_id}` call drains the cached chunks and yields `end`, the entry is no longer present in `STREAM_REGISTRY` (integration test in `test_thinking_routes.py::test_resume_route_cleans_up_completed_job`)
- [ ] After a `POST /api/chat/stream` call drains the initial stream to completion, the entry **is still** present in `STREAM_REGISTRY` (integration test in `test_thinking_routes.py::test_initial_stream_does_not_clean_up`)
- [ ] A resume whose client disconnects before the `end` marker is received leaves the entry in place (unit test in `test_stream_manager.py::test_consume_with_cleanup_keeps_job_on_cancellation` via `gen.aclose()`)
- [ ] A resume whose inner generator returns without yielding any events (e.g., `from_pointer` out of range) leaves the entry in place (unit test in `test_stream_manager.py::test_consume_with_cleanup_keeps_job_when_no_events_yielded`)
- [ ] A resume whose inner generator raises leaves the entry in place (unit test in `test_stream_manager.py::test_consume_with_cleanup_keeps_job_on_exception`)
- [ ] Two concurrent resumes where the first finishes do not raise on the second's `pop` (unit test in `test_stream_manager.py::test_consume_with_cleanup_pop_is_idempotent`)
- [ ] All 47 tests pass

### Known Limitation (Out of Scope)

A `StreamJob` for which the user never calls resume (e.g., the initial stream completes normally and the user closes the tab) still leaks for the lifetime of the process. A TTL-based sweep is a separate item.

---

## Part 4: File Storage Concurrency Safety

### Problem

Two real hazards in [backend/storage/file_storage.py](backend/storage/file_storage.py):

1. **Lost updates.** Every write function did `load → modify → save` against a single JSON file. If two requests did this concurrently within the same process (e.g., two simultaneous `append_message` calls), each one would load the same baseline, append its own change, and write back. The second writer's save would overwrite the first writer's change. With FastAPI's async handlers, both `await` calls yield to the event loop and can interleave at the `await file_storage.append_message(...)` boundary.
2. **Crash corruption.** A write was a single `open(..., 'w').write(...)` call. A SIGKILL, OOM, or power loss mid-write would leave a half-written file on disk. The previous layer of safety (`json.JSONDecodeError → rename to *.corrupt → start fresh`) was a recovery net, not a prevention.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-4.1 | Two threads writing to `conversations.json` concurrently (e.g., two `append_message` calls) must both be reflected in the final file — no lost updates |
| FR-4.2 | A process crash (or any failure) during a write must leave the previous on-disk file fully intact — readers see either the fully-old or fully-new file, never partial |
| FR-4.3 | A failed write must clean up any leftover `.tmp` file in `STORAGE_DIR` so the directory does not accumulate stale partial writes |
| FR-4.4 | The behavior is observable to existing callers (`save_conversation`, `append_message`, `create_conversation`, `delete_conversation`) only as "still produces a valid JSON file at `CONVERSATIONS_FILE`" — no signature change |

### Acceptance Criteria

- [ ] 50 concurrent `append_message` calls on the same conversation all leave their message in storage (unit test in `test_storage.py::test_concurrent_appends_preserve_all_messages`)
- [ ] 20 concurrent `save_conversation` calls all complete without deadlock, the file is always valid JSON, and no `.tmp` file is left behind (unit test in `test_storage.py::test_concurrent_saves_are_serialized_with_no_corruption`)
- [ ] When `os.replace` is monkeypatched to raise mid-swap, the original file is fully intact afterward and no `.tmp` is left (unit test in `test_storage.py::test_atomic_write_leaves_original_when_replace_fails`)
- [ ] When `json.dump` is monkeypatched to raise mid-write, the original file is fully intact and no `.tmp` is left (unit test in `test_storage.py::test_atomic_write_cleans_tmp_when_write_fails`)
- [ ] All 47 tests pass

### Known Limitation (Out of Scope)

The lock is per-process. A multi-worker deployment (`uvicorn --workers N`) would still have N independent processes racing on the file. For that, a file-level lock (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows) is needed. The current `CLAUDE.md` run command uses a single worker, so this is not triggered today.

---

## Part 5: Frontend Cache Consolidation

### Problem

[frontend/static/app.js](frontend/static/app.js) had ~15 raw `localStorage` calls and ~8 `JSON.parse` / `JSON.stringify` calls scattered across 6+ call sites. The key names were constructed inline via a `STORAGE_KEYS` constant. There were 7 helper functions (`getHistoryCache`, `setHistoryCache`, `appendToHistoryCache`, `clearHistoryCache`, `clearChunkCache`, `isStreamingForConv`, `getStreamingForConv`, `setStreamingForConv`) that were each a one- or two-liner around the same pattern.

This made the file harder to read, easier to get wrong (e.g., forgetting to `JSON.parse` on a read, or `JSON.stringify` on a write), and gave no single place to change the storage format or migrate to a different backing store.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-5.1 | All `localStorage` access in the frontend is encapsulated in a single module: [frontend/static/cache.js](frontend/static/cache.js) |
| FR-5.2 | The module exposes typed accessors for each cache: history (get/set/append/clear), chunks (get/set/append/clear), consumed (get/set/clear), streaming (is/get/set/clear), and currentConversationId (get/set) |
| FR-5.3 | The five keys (`chunks_`, `consumed_`, `streaming_`, `history_`, `currentConversationId`) and their JSON-vs-string encoding remain identical to the pre-refactor behavior |
| FR-5.4 | `app.js` contains no `localStorage`, no `STORAGE_KEYS`, no `JSON.parse` / `JSON.stringify` related to cached state, and no helper functions for localStorage access. The only `JSON.*` calls that remain are for the `fetch` request body (`JSON.stringify({...})`) and SSE event parsing (`JSON.parse(event.slice(6))`), which are not localStorage-related |
| FR-5.5 | The five caches keep their existing roles: `history` is the per-conversation message history (for instant load), `chunks` is the per-conversation streaming chunks (for page-refresh-during-stream resume), `consumed` is the per-conversation resume pointer, `streaming` is the per-conversation "is this conversation being streamed" flag, `currentConversationId` is the global "which conversation is open" key |

### Acceptance Criteria

- [ ] `grep -E 'localStorage|STORAGE_KEYS|JSON\.(parse|stringify)' frontend/static/app.js` returns no matches for `localStorage` and `STORAGE_KEYS`. The two remaining `JSON.*` hits are `JSON.stringify({message, conversation_id})` (line 683, the fetch body) and `JSON.parse(event.slice(6))` (line 783, SSE event parsing) — both legitimate
- [ ] All 7 deleted helper functions are gone from `app.js`
- [ ] All 47 backend tests pass
- [ ] `/static/cache.js` serves correctly (200) from FastAPI's `StaticFiles` mount

---

## Part 6: Frontend Asset Path

### Problem

After Part 1's frontend split, `index.html` referenced `/static/styles.css` and `/static/app.js`, but the actual files lived in `frontend/` — there was no `static/` folder. FastAPI's `StaticFiles` mount at `/static` was pointing at `frontend/`, so the URL worked, but the mismatch was misleading.

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-6.1 | The URL path `/static/...` references an actual `static` directory in the project tree |
| FR-6.2 | `index.html` is served at `/` as before |
| FR-6.3 | All three frontend files (`index.html`, `styles.css`, `app.js`, plus the new `cache.js`) continue to be served correctly by FastAPI |

### Acceptance Criteria

- [ ] `frontend/static/` exists and contains `app.js`, `cache.js`, `styles.css`
- [ ] `frontend/index.html` exists and is served at `/`
- [ ] `app.mount("/static", StaticFiles(directory=str(frontend_path / "static")))` in [backend/main.py](backend/main.py) is the only mount
- [ ] All 47 tests pass
- [ ] Smoke: `GET /` → 200, `GET /static/styles.css` → 200, `GET /static/app.js` → 200, `GET /static/cache.js` → 200

---

## Part 7: `DOMPurify` No-Op TODOs (U2)

### Background

In [frontend/static/app.js:60](frontend/static/app.js#L60) and [frontend/static/app.js:371](frontend/static/app.js#L371), the code calls `DOMPurify(content);` and `DOMPurify.sanitize(content);` respectively. **Both are no-ops** — the return value is discarded, so no sanitization actually happens. The content is then written to the DOM via `smd.parser_write(parser, content)` without any sanitization.

These calls were preserved per an earlier user decision (the original audit recommended removing them as misleading dead code; the user preferred to keep them). To make the no-op status explicit rather than hidden, a `TODO(U2 — known issue)` comment is now added at each call site describing:

1. The call is a no-op
2. The risk (untrusted HTML renders as-is)
3. The two fix options (wire up `DOMPurify.sanitize` before insertion, or remove the misleading call)

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-7.1 | Each of the two `DOMPurify` no-op call sites has a `TODO(U2 — known issue)` comment block above it describing the issue, the risk, and the fix options |
| FR-7.2 | No code change — the calls are still no-ops, but now the no-op status is documented inline |

### Acceptance Criteria

- [ ] `grep -n "TODO(U2" frontend/static/app.js` returns exactly 2 matches (one at line ~60, one at line ~371)
- [ ] All 47 tests pass

---

## Out of Scope

Items that are **not** addressed in this iteration:

- **Real API key in `.env` (U1)** — the project root `.env` still contains a real `ANTHROPIC_API_KEY`. Out of scope; user is aware.
- **TTL-based eviction of `StreamJob` (Part 3 limitation)** — never-resumed jobs still leak.
- **Multi-worker file lock (Part 4 limitation)** — only relevant if `uvicorn --workers N` is used.
- **Cross-tab cache synchronization** — not affected by any of these changes.
- **Wiring up `DOMPurify` sanitization** (Part 7) — the no-ops are now commented but still no-ops. A real fix is a separate change.

---

## Testing

All 47 backend tests in `backend/tests/` pass after the implementation. There are no frontend tests in this project; correctness of the frontend refactor is verified by grep and by the smoke check that all four static paths return 200.
