# Chatbot Project - Detailed Design

## 1. Architecture Decisions

### 1.1 Communication Protocol: SSE

**Choice:** Server-Sent Events (SSE) for streaming responses.

**Rationale:**
- Unidirectional (server → client) fits chat streaming use case
- FastAPI native `StreamingResponse` support
- Frontend uses Fetch API with `ReadableStream`
- Simpler than WebSocket for this use case

### 1.2 LLM Integration: LangChain LCEL

**Choice:** LangChain with LangChain Expression Language (LCEL).

**Rationale:**
- LCEL enables clean streaming chain composition
- Built-in support for Anthropic models via MiniMax endpoint
- Easy to swap models without changing application logic
- `ChatAnthropic` receives `base_url=settings.anthropic_base_url` directly via its constructor argument. No `os.environ` mutation is needed.

### 1.3 Storage: JSON Files

**Choice:** File-based JSON storage (`storage/conversations.json`).

**Rationale:**
- Zero infrastructure setup
- Easy to inspect and debug
- Migration path to PostgreSQL defined for future
- Per-conversation structure allows future sharding

### 1.4 Frontend: Plain HTML/JS

**Choice:** No framework, single `index.html`.

**Rationale:**
- Zero build step
- Easy to understand and modify
- Served directly by FastAPI
- Native `ReadableStream` for SSE handling

### 1.5 Configuration: Environment Variables

**Choice:** `pydantic-settings` with `.env` file.

**Rationale:**
- Standard Python pattern
- Type validation on startup
- Secrets kept out of codebase

### 1.6 Thinking Content Streaming

**Choice:** Thinking blocks from LLM are streamed first, then text blocks.

**Rationale:** Thinking represents the model's internal reasoning process which logically precedes the final response. Streaming thinking first provides immediate feedback to users that the model is "thinking."

### 1.7 Thinking Persistence

**Choice:** Store thinking content in message history alongside response content.

**Rationale:** Users may want to review the model's reasoning for past responses. Storage schema is backward-compatible (thinking field is optional).

### 1.8 Markdown Library

**Choice:** `streaming-markdown` via CDN. `marked.js` is no longer loaded.

**Rationale:** The streaming parser (`smd`) handles both incremental token rendering during streaming and full-message rendering for cached/replayed history. It is stateful (a `parser_write` accumulates tokens across chunks), so multi-line constructs (tables, fenced code blocks, math blocks, lists) render correctly when their tokens arrive in multiple chunks or span the cache-replay → live-stream resume boundary. The earlier `marked.js` (used for final-message rendering) is now redundant.

### 1.9 Stream Resume Architecture

**Choice:** LLM calls run in background tasks; tokens stored in `StreamJob`. Frontend caches tokens in localStorage and tracks position for resume.

**Rationale:** Allows seamless continuation when user switches tabs/refreshes during streaming. Background task continues generating while frontend reconnects.

### 1.10 Two-Cache Frontend Architecture

**Choice:** Maintain two separate localStorage caches — `history_{convId}` for full message lists and `chunks_{convId}` for in-flight streaming chunks. All `localStorage` access is encapsulated in [`frontend/static/cache.js`](../frontend/static/cache.js) (see 1.26).

**Rationale:** History cache stores the "done" state (complete messages, updated on stream end via append); chunks cache stores the "in-flight" state (individual tokens, updated per chunk). Different update patterns and different consumers (`loadConversation` vs. `processStreamResponse`). Separation makes invalidation explicit and avoids mixing concerns. The history cache is loaded cache-first by `loadConversation`, with a backend fetch on miss and a stale-chunks-cache clear on success.

### 1.11 New Conversation Sidebar Visibility

**Choice:** When a brand-new conversation is created, `stream_chat` (the POST `/api/chat/stream` route) calls `file_storage.create_conversation(conversation_id)` and `file_storage.append_message(conversation_id, "user", request.message)` itself, before starting the background LLM task. `get_or_create_job` no longer has any storage side effect.

**Rationale:** Without this, the new conversation only appears in the sidebar after the LLM finishes streaming (5-30+ seconds later) and lacks a user-message-derived title. Pulling the create+append into the route keeps the storage side effect co-located with the user-visible "new conversation" action and makes `get_or_create_job` a pure in-memory operation. The background task trusts the storage contents as the single source of truth (no idempotency dedupe is needed).

### 1.12 SSE Event Boundary Handling

**Choice:** The frontend SSE parser accumulates chunks in an `sseBuffer` variable, splits on `\n\n` per SSE spec, and retains the trailing incomplete event in the buffer for the next chunk.

**Rationale:** A single network read can split events across chunks. Without buffering, JSON payloads straddling a boundary would produce parse errors or dropped data. Buffering by the actual `\n\n` boundary (not chunk boundary) correctly handles any chunk size.

### 1.13 Streaming Markdown Parser State Preservation

**Choice:** The `streaming-markdown` parser/renderer is hoisted to function scope in `processStreamResponse` and created once per stream.

**Rationale:** The library is stateful — `parser_write` accumulates tokens so multi-line constructs (tables, fenced code blocks, math blocks, lists, blockquotes) render correctly when their tokens arrive in multiple chunks. Hoisting prevents the parser from being recreated on every chunk batch. `parser_end` is called exactly once on stream completion (in the `data.end` handler). On resume, the parser created during `renderCachedChunks` is reused by `processStreamResponse` so multi-line constructs spanning the cache-replay → live-stream boundary render correctly.

### 1.14 Stream Resume Error Handling

**Choice:** In `resumeStreamFromPosition`'s catch block, drop any error-type branching. Log the error and return false; do not touch the streaming flag or badge.

**Rationale:** The streaming flag has exactly two responsibilities — `init` / `checkStreamStatus` decides whether to attempt a resume on page load, and `loadConversationList` derives the sidebar "Streaming" badge. For (1), a transient fetch failure must leave the flag set so the next refresh can retry. The flag should only be cleared on `data.end` or on explicit user action. A fetch killed by browser navigation (refresh, close tab, back button) is rejected with `TypeError: network error` rather than `AbortError`; the two cannot be reliably distinguished, so both are treated as transient.

### 1.15 init Fallback Gate

**Choice:** In `init`, only fall through to `loadConversation` when the streaming flag was unset at the start of the decision.

**Rationale:** `checkStreamStatus` calls `resumeStreamFromPosition`, which always renders cached chunks into the DOM before issuing the network fetch. Whatever happens afterwards, the partial content stays in the DOM. Calling `loadConversation` after a resume attempt would `messagesContainer.innerHTML = ''` and re-render only what is in the history cache (missing the in-progress assistant message), wiping the partial content.

### 1.16 Custom Confirmation Modal

**Choice:** A single reusable modal element appended to `<body>` with a single `showConfirmModal({title, message, confirmText, cancelText, danger})` helper that returns a Promise resolving to `true`/`false`.

**Rationale:** Browser-native `confirm()` has a generic OS look that does not match the page's dark-theme + cyan/purple-accent styling. The helper uses the same theme tokens as the rest of the page (`var(--bg-secondary)`, `var(--border-color)`, etc.) and is invoked by both single-item and batch deletion.

### 1.17 Batch Delete Selection Mode

**Choice:** Module-level state (`selectionMode: boolean`, `selectedConvIds: Set`) toggled via `enterSelectionMode()` / `exitSelectionMode()`. The sidebar header has two layouts (normal vs selection), switched in a single `renderSidebarHeader()` function that reads `selectionMode`. `+ New Chat` is hidden in selection mode; the per-item `×` is also hidden so the user uses the batch Delete button exclusively.

**Rationale:** A `Set` makes add/remove/lookup O(1) and avoids duplicate selections. Two layout branches in one renderer keeps the DOM state in sync without separate header functions. Hiding `+ New Chat` and per-item `×` in selection mode prevents the user from inadvertently starting a new chat or single-deleting an item while in "I'm about to delete these" mode.

### 1.18 StreamJob Cancellation Flag

**Choice:** `StreamJob` carries a `cancelled: bool` flag. `clear_job` sets the flag before removing the job from `STREAM_REGISTRY`. `generate_background` checks the flag mid-loop and immediately before `save_conversation`, bailing out without `mark_completed` or save if set.

**Rationale:** `generate_background` is a fire-and-forget `asyncio.create_task`. Deleting the conversation only removes the `StreamJob` and the entry from `conversations.json` — it does not stop the background task. When the task finishes, `file_storage.save_conversation` is create-or-update and silently re-creates the deleted entry (the "resurrection" bug). A `cancelled` flag stops the background task at its next natural checkpoint without taking the heavier step of cancelling the asyncio task itself (which would abort the LLM read but waste server-side LLM work).

### 1.19 Smart Auto-Scroll Pin State

**Choice:** During streaming, capture the user's pinned-to-bottom state BEFORE each chunk's DOM update, and only force `scrollTop = scrollHeight` when the captured state is true.

**Rationale:** A naive post-update check breaks because a single chunk can add more than 50px of height, causing the post-update check to incorrectly report "not pinned" even though the user never scrolled. Capturing pre-update reflects the user's true intent at the moment the chunk arrived.

### 1.20 Selection-Mode Send Guard

**Choice:** An early-return guard at the top of `sendMessage()` checks `selectionMode` and returns immediately if true. No UI changes — the textarea and send button remain enabled-looking; the user is expected to exit selection mode via Cancel if they want to send.

**Rationale:** Both the Send button click handler and the `messageInput` keydown (Enter) handler route through `sendMessage()`. Guarding `sendMessage()` blocks both paths with one line. Visual disablement is a separate UX choice and is intentionally omitted to keep the change minimal.

### 1.21 Backend Stream Resume Boundary Case

**Choice:** The `stream_from_job` `from_pointer` guard treats `from_pointer == len(chunks)` as a valid boundary (all current chunks already sent), not as an out-of-range error. Only negative or strictly-greater-than pointers are rejected.

**Rationale:** The frontend's pointer always lands at exactly `len(job.chunks)` during active streaming (because the user-entered message is only in `chunksCache`, not yet committed to history). Treating this as an error would cause the resume to return immediately and drop the live stream.

### 1.22 LLM Model Configuration

**Choice:** The backend LLM is configured as `minimax-3` with `max_tokens=16000` and `thinking={"type": "enabled", "budget_tokens": 10000}`.

**Rationale:** The larger output budget (4× the previous 4096) supports long code blocks and multi-paragraph answers. Explicit extended thinking with a 10k budget reserves reasoning capacity; visible answer has ~6k tokens. Frontend already supports `thinking` blocks via the unified chunk format.

### 1.23 Multi-Turn Thinking Continuity

**Choice:** `convert_messages` (module-level in `chat/chain.py`) constructs an `AIMessage` whose `content` is a list of content blocks when a prior assistant turn has a `thinking` field. The list is `[{"type": "thinking", "thinking": ...}, {"type": "text", "text": ...}]`. When the prior turn has no `thinking` field, the `AIMessage` is a plain `AIMessage(content=...)`.

**Rationale:** The LLM emits `thinking` content blocks on every turn when given the previous turn's `thinking` as part of the input. It does not emit them when the previous turn's `thinking` is stripped before being fed back. The pre-change code dropped the prior `thinking` (returning only the visible `content`), which produced empty thinking sections for turns 2+. Feeding the prior `thinking` back as a content block alongside the visible text restores continuous reasoning. Prior messages without `thinking` (e.g., loaded from older storage) become a plain `AIMessage` so the LLM still sees a coherent conversation history.

### 1.24 Stream Registry Memory Cleanup

**Choice:** A thin async generator wrapper `consume_with_cleanup(gen, conversation_id)` in `chat/stream_manager.py` is applied only to the resume route's `StreamingResponse`. Two flags inside the wrapper:
- `completed` — set to `True` only after the `async for` loop exits normally.
- `any_event` — set to `True` after the first event is yielded.

The wrapper's `finally` block calls `STREAM_REGISTRY.pop(conversation_id, None)` if and only if both flags are `True`. The initial stream (`POST /api/chat/stream`) does NOT use the wrapper, so the entry is still available for a possible later resume.

**Rationale:** The `StreamJob` exists in the registry to support resume. Once a resume has delivered the full cached chunk history (including `end`), there is nothing left to resume. The two-flag pattern ensures that:
- An interrupted resume (`aclose()`, exception, or out-of-range `from_pointer` returning no events) leaves the entry in place.
- A normal completion removes the entry.
- Two concurrent resumes where the first finishes are safe: the second wrapper's `pop` is a no-op (idempotent).
- The initial stream is not affected — its job stays in the registry for a potential future resume.

### 1.25 File Storage Concurrency Safety

**Choice:** `chat/storage/file_storage.py` uses two complementary safety mechanisms:
- A module-level `_write_lock = threading.Lock()` serializes the four write functions (`create_conversation`, `save_conversation`, `append_message`, `delete_conversation`). The lock holds for the full read-modify-write cycle.
- An `_atomic_write_json(path, data)` helper writes JSON to `<path>.tmp` in the same directory, then `os.replace()` swaps it into place. On any failure, the `.tmp` file is cleaned up.

**Rationale:** Two distinct hazards:
- **Lost updates** (per-process). Two concurrent `append_message` calls would each read the same baseline, append their own change, and write back; the second overwrites the first. The lock prevents this.
- **Crash corruption.** A `SIGKILL`/`OOM`/power-loss mid-write would leave a partial JSON on disk. `os.replace` is atomic on POSIX and on Windows when source and destination are on the same volume (the `.tmp` is in `STORAGE_DIR`, so this holds). A crash before the `replace` leaves the original file fully intact; after the `replace`, the new file is in place.

Reads (`get_conversation`, `get_conversation_list`) are not under the lock. Combined with the atomic write, a concurrent reader sees either the fully-old or fully-new file — never partial. The lock is per-process; a multi-worker deployment (`uvicorn --workers N`) would need a file-level lock (out of scope).

### 1.26 Frontend Cache Module

**Choice:** All `localStorage` access in the frontend is encapsulated in a single ES module, [`frontend/static/cache.js`](../frontend/static/cache.js). The module owns:
- The five localStorage key names (`chunks_`, `consumed_`, `streaming_`, `history_`, `currentConversationId`).
- All `localStorage.getItem` / `setItem` / `removeItem` calls.
- All `JSON.parse` / `JSON.stringify` calls (for the JSON-encoded caches: `chunks`, `history`).

The module exposes typed accessors: `getHistory` / `setHistory` / `appendToHistory` / `clearHistory`, `getChunks` / `setChunks` / `appendToChunks` / `clearChunks`, `getConsumed` / `setConsumed` / `clearConsumed`, `isStreaming` / `getStreaming` / `setStreaming` / `clearStreaming`, `getCurrentConversationId` / `setCurrentConversationId`. `app.js` imports the module and calls these accessors; no `localStorage` reference, no `STORAGE_KEYS`, no `JSON.parse` / `JSON.stringify` for cached state, and no helper functions remain in `app.js`.

**Rationale:** The pre-refactor `app.js` had ~15 raw `localStorage` calls and ~8 `JSON.parse` / `JSON.stringify` calls scattered across 6+ call sites, plus 7 helper functions and a `STORAGE_KEYS` constant. This made the file harder to read, easy to get wrong (forgetting `JSON.parse` on a read, or `JSON.stringify` on a write), and gave no single place to migrate to a different backing store (e.g., IndexedDB) later. The cache module also makes the storage format discoverable from one file. The two remaining `JSON.*` calls in `app.js` are for the `fetch` request body and SSE event parsing, which are external-data protocol concerns, not state-storage concerns.

### 1.27 Frontend Asset Path

**Choice:** Frontend assets live in `frontend/static/`:
```
frontend/
├── index.html
└── static/
    ├── app.js
    ├── cache.js
    └── styles.css
```

FastAPI's `StaticFiles` mount is at `/static` and points at `frontend_path / "static"`. The explicit `@app.get("/")` route serves `frontend/index.html`. There is no separate `/index.html` route.

**Rationale:** The pre-change code mounted `/static` on `frontend/` (with no actual `static/` subdirectory). The URL path `/static/...` was misleading because there was no matching folder. Moving the assets into `frontend/static/` makes the URL and the directory match. `index.html` stays at `frontend/index.html` because it is the application entry point, not a static asset.

### 1.28 Plugin Activation via Env Flag

**Choice:** `RAG_ENABLED` environment variable (default `false`) controls whether the RAG module is constructed at all.

**Rationale:**
- Hard-off behavior is provable: when `false`, no `rag/` imports execute in the request hot path and `delete_conversation` has no rag callback wired in.
- Avoids runtime "is RAG enabled?" branches scattered through the codebase — instead the code is constructed conditionally at startup.
- A test (`test_no_rag_path_is_byte_identical_to_today`) asserts that calling `ChatService.generate_background` without `retrieval` produces exactly the same `job.chunks` as the iteration-6 version.

**Trade-off:** Per-request toggling of "is RAG even installed" is impossible; once enabled, it's enabled until restart. For a v1 exploration project this is acceptable.

### 1.29 Retrieval-Augmented Chat Without Chain Modification

**Choice:** Retrieval happens in `ChatService.generate_background` as a pre-processing step before the LLM call. The chain implementation (`backend/chat/chain.py`) is **not modified**.

**Rationale:**
- The chain becomes "pure LLM call" — `RunnableLambda(convert_messages) | llm`. Its job is well-defined and stable.
- Retrieval results are pushed to `job.chunks` via `job.append_chunk("sources", ...)` before the LLM call, so SSE replays them in order: sources → token → token → ... → done.
- The plugin-off property becomes literally true: the chain source file is byte-identical to iteration 6.
- If retrieval happens *inside* the LCEL chain, the retrieved `context` field is consumed by `augment_messages` and discarded. The route handler (which reads from `job.chunks`, not from the chain's intermediate state) never sees it. This was the load-bearing finding during the brainstorming self-review.

**Trade-off:** Retrieval runs synchronously before the LLM call, adding ~50-200 ms to the first-token latency. Acceptable for v1; could move to a streaming side-channel later if needed.

### 1.30 Two-Scope Index Architecture

**Choice:** Two separate FAISS indexes — `library_index/` and `uploads_index/` — merged at query time via a custom `ScopedRetriever`.

**Rationale:**
- Library changes are admin operations (rare, read-mostly); uploads change every chat session (mutable, per-conversation). Mixing them in one index forces admin rebuilds to walk the user namespace and makes conversation deletion surgically remove chunks from a shared file.
- Two indexes are easy to reason about and easy to operate on independently.
- Library index can be reloaded read-only at startup; uploads index supports `add_documents` mid-session.
- Adding a third scope later ("web cache", "agent memory") is one more FAISS index + one more entry in `ScopedRetriever.retrievers`. No chain changes, no chat-service changes.

**Trade-off:** ~30 lines of glue code for the merge-and-filter step. Acceptable for the clarity gained.

### 1.31 Conversation-Scoped Filter with Explicit Targeting

**Choice:** `ScopedRetriever` takes `list[tuple[BaseRetriever, bool]]` where the bool flags whether to apply the `conversation_id` filter. Library retrievers get `False`; upload retrievers get `True`.

**Rationale:**
- Explicit > implicit. The earlier design filtered all hits and relied on the magic of "library hits don't have `conversation_id` so they survive". Fragile and hard to reason about.
- Each tuple's bool is documented at the call site (in `RagService.make_scoped_retriever`), making the security/correctness property reviewable: "library chunks are never filtered out, upload chunks are always filtered to the current conversation".

**Trade-off:** Slightly more verbose than a single `filter_key/filter_value` parameter pair.

### 1.32 Pluggable Embeddings

**Choice:** `EMBEDDING_BACKEND` env var (`sentence-transformers` or `minimax`) selects the embeddings implementation. Default is `sentence-transformers` with model `all-MiniLM-L6-v2`.

**Rationale:**
- Local sentence-transformers works offline, has no per-call API cost, and is fast (~5 ms per query). Adds ~80 MB pip dep but eliminates a network round-trip per chunk during ingestion.
- MiniMax alternative (subject to verifying the endpoint exposes `/embeddings`) allows using the same vendor as the chat model. Useful for A/B comparing embedding quality.
- The factory pattern in `backend/rag/embeddings.py` returns a `langchain_core.embeddings.Embeddings` instance — both backends satisfy the same interface, so the rest of the system doesn't know which is active.

**Trade-off:** Two backends to test. Mitigated by injecting a `FakeEmbeddings` in tests.

### 1.33 Pluggable Vector Store: FAISS Default

**Choice:** `backend/rag/vector_store.py` wraps LangChain's `FAISS` class. Swapping to Chroma means rewriting this file; nothing else moves.

**Rationale:**
- FAISS via `langchain_community.vectorstores` provides `add_documents`, `save_local`, `load_local`, `as_retriever` out of the box.
- On-disk persistence matches the project's "no external services" theme (same as the JSON conversation storage).
- No FAISS server process; index lives in files.
- LangChain's `FAISS.save_local` writes a sidecar `index.pkl` containing the docstore — this is what enables fast rebuild-on-delete without re-embedding.

**Trade-off:** Pickle sidecar limits index size to ~50k chunks before memory pressure becomes a concern. Documented as future work: replace with a SQLite/DuckDB-backed docstore when needed.

### 1.34 Deletion via Docstore Rebuild

**Choice:** When a conversation is deleted, the uploads index is rebuilt from the surviving docstore entries (no re-embedding).

**Rationale:**
- LangChain's `FAISS.save_local` writes vectors + docstore together. Rebuild reads the docstore, filters by `conversation_id`, creates a new `FAISS.from_documents(...)`, and atomically rebinds `self.uploads_index = new_index`.
- Rebuild cost is dominated by walking the docstore dict (~50 ms for 1000 chunks). No API calls, no file re-reading.
- For 10k chunks: ~400 ms. For 100k chunks: ~4 s. Acceptable for v1.
- Escape hatch documented: tombstone set + periodic compaction when rebuild cost becomes a bottleneck.

**Trade-off:** Synchronous in the request handler — the user waits for the rebuild before the delete returns. Mitigated by the fast path (sub-100 ms for normal-sized indexes).

### 1.35 Hook Wiring via Callback Parameter

**Choice:** `file_storage.delete_conversation` accepts an optional `on_delete: Callable[[str], None]` parameter. `main.py` wires `rag_service.purge_uploads` as the callback at startup.

**Rationale:**
- Explicit over implicit. The storage layer declares its extensibility via the `on_delete` parameter on `delete_conversation`; the application layer wires the actual callback at startup (via `functools.partial`), so the storage module itself is never modified.
- Default `None` keeps old callers unaffected.
- Exceptions in the callback are caught and logged but don't fail the delete — the JSON state remains consistent.
- Testable: tests pass a fake callback to verify the wiring without needing a real `RagService`.

**Trade-off:** Tiny signature change to an existing function. Internal API only.

### 1.36 Sources Event via Job.Append Chunk

**Choice:** New SSE event type `sources` fired once before the first token, populated by `job.append_chunk("sources", json.dumps({...}))` in `ChatService.generate_background`.

**Rationale:**
- Reuses the existing background-task + job.chunks pattern. No new streaming mechanism.
- Order is preserved by `job.append_chunk`'s append-only semantics: SSE replays sources before tokens.
- Vanilla chain never emits sources — natural divergence between modes.
- Frontend toggle is local state; server doesn't know or care.

**Trade-off:** Sources are formatted as a single JSON blob in a chunk string. Alternative would be a structured SSE event with `event: sources\ndata: {...}` headers; chosen approach uses the existing chunk format for simplicity.

### 1.37 Size-Based Upload Routing

**Choice:** Upload is **not** a RAG-only feature. Both columns expose an Upload button beside the Send button. Files are routed to one of two pipelines based on a server-side threshold (`RAG_INLINE_CONTEXT_THRESHOLD_BYTES`, default 8192 bytes), applied to the raw upload size:

- **Small files** (≤ threshold): the upload endpoint reads the file content, returns `{filename, content, mode: "inline"}` to the client. The client includes this content in the next `/api/chat/stream` request via the `uploaded_files` field. `ChatService` injects each file's content as a single system message before the user message. No FAISS, no chunking, no retrieval. The file is also added to a per-conversation pending-files list, so subsequent turns in the same conversation see the same context without re-sending.
- **Large files** (> threshold): the existing FAISS path — save to `storage/uploads/<conversation_id>/`, embed, index, retrieve per-turn via ScopedRetriever.

**Rationale:**
- A small `.md` or `.txt` file (a paragraph or two) doesn't justify the embedding + indexing round-trip. Reading the whole file inline is faster, has no embedding cost, and the LLM can see all of it at once.
- Per-conversation pending-files list keeps the feature stateful: the user uploads once, then can send multiple follow-up turns without re-uploading.
- The threshold is server-side and authoritative — the upload endpoint reads it and routes to inline vs. FAISS. The threshold value is also exposed via `/api/rag/stats` so the client can observe (not enforce) the same boundary. Routing decisions live in the server's response (`mode: "inline" | "indexed"`); the frontend does not duplicate the comparison.
- Both columns use the same upload logic. The semantic difference: the vanilla column's `uploaded_files` content is the **only** context it sees (no library, no FAISS), so vanilla stays a "pure LLM grounded on the file" comparison arm. The RAG column sees library + uploads + the new file's content — a fully-grounded answer.
- Threshold applies to **raw upload size**, not post-extraction text. PDFs and HTML always go through FAISS (raw sizes are typically megabytes). `.md`/`.txt` under 8 KB take the inline path. This matches the user's intuition: "small text file" vs. "anything larger".

**Trade-off:**
- The pending-files list is in-memory state. A server restart clears it. If the user uploaded a small file and then the server restarts, the file's content is lost. This is acceptable for v1 because small files are small enough that re-uploading is cheap. (Future work: persist pending files to disk or to a separate `storage/inline_uploads/` keyed by conversation_id.)
- The threshold is a single number. A more sophisticated policy could route based on file type, post-extraction text size, or token count. Single number keeps the v1 simple.

### 1.38 Frontend Side-by-Side Comparison UI

**Choice:** The chat UI is a side-by-side comparison layout: one shared input at the bottom, two message columns above (Vanilla on the left, RAG on the right). The two columns share a base UUID; vanilla maps to `<base>-0`, RAG maps to `<base>-1`. One Send click fires two parallel `/api/chat/stream` POSTs — vanilla with `retrieval: null`, RAG with `retrieval: {library, uploads, top_k}` — and each column streams its SSE response into its own column. The two columns are otherwise independent: separate histories, separate abort controllers, separate cache state.

**Rationale:**
- Direct visual comparison on the same question is the core evaluation use case. A side-by-side layout answers "did RAG actually improve this answer?" without making the user retype and re-send.
- The backend already supports per-request `retrieval` toggling (`RetrievalConfig` on `ChatRequest`). The frontend fan-out is a thin client-side loop: two `fetch` calls with the same `message` and different `retrieval` / `conversation_id`. No backend changes required.
- Conversation IDs `<base>-0` and `<base>-1` keep the existing storage model: the two columns are normal conversations in `conversations.json`, so the sidebar, CRUD, and deletion work without modification.
- When `RAG_ENABLED=false`, the RAG column is hidden, the vanilla column takes full width, and only one POST fires per Send. The layout degrades cleanly to iteration-6 behavior.

**Trade-off:** The frontend is more complex than a single-pane UI: two parallel SSE readers, two `AbortController`s, per-column resume-from-cache, per-column badge/state. To keep this manageable, the existing `processStreamResponse` is parameterized by a per-column context object (assistant message element, cache keys, abort controller) and invoked twice in parallel from `sendMessage`.

---

## 2. System Architecture

```
┌─────────────┐     SSE/HTTP      ┌─────────────────┐
│   Frontend  │ ◄───────────────► │   FastAPI       │
│  (Plain     │                   │   Backend       │
│   HTML/JS)  │                   │   (LangChain)   │
└─────────────┘                   └────────┬────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                       │
                    │              ┌───────────────┐        ┌───────────────┐
                    │              │  chat domain  │        │ storage domain│
                    │              └───────────────┘        └───────────────┘
                    │              ┌───────────────┐
                    │              │   rag domain  │
                    │              └───────────────┘
                    └──────────────────────────────────────────────────────┘
```

---

## 3. Backend Structure

```
backend/
├── main.py                 # FastAPI app entry, serves frontend + mounts routers
├── config.py               # Pydantic Settings from env vars
├── chat/
│   ├── routes.py           # /api/chat/* endpoints
│   ├── chain.py            # LangChain LCEL chain definition
│   ├── service.py          # ChatService orchestration
│   └── stream_manager.py   # StreamJob tracking + STREAM_REGISTRY
├── rag/                    # NEW (optional, gated by RAG_ENABLED)
│   ├── __init__.py
│   ├── config.py           # RAG-specific Pydantic settings
│   ├── service.py          # RagService facade — ingest, purge, reindex, retriever, stats
│   ├── retriever.py        # ScopedRetriever (merge + explicit metadata filter)
│   ├── vector_store.py     # FAISS load/save/rebuild helpers
│   ├── embeddings.py       # Embeddings factory: sentence-transformers or MiniMax
│   ├── splitter.py         # Text splitter factory + chunk metadata assembly
│   └── routes.py           # /api/rag/* endpoints (upload, library reindex, stats)
└── storage/
    └── file_storage.py     # JSON file read/write operations
```

### Domain Pattern

Each domain has three files:
- `routes.py` — HTTP interface (endpoints)
- `chain.py` — Business logic composition (LCEL)
- `service.py` — Orchestration and state management

### Stream Registry

In-memory registry (`STREAM_REGISTRY`) tracks active streams:
- `conversation_id` → `StreamJob` mapping
- Enables stream resume on conversation switch/refresh

---

## 4. API Protocol

### SSE Format (Unified Chunk Format)

```
data: {"chunk": "Hello", "type": "thinking"}\n\n
data: {"chunk": "!", "type": "token"}\n\n
data: {"end": true}\n\n
```

| Event | Format | Description |
|-------|--------|-------------|
| Thinking | `data: {"chunk": "...", "type": "thinking"}` | Single thinking token |
| Token | `data: {"chunk": "...", "type": "token"}` | Single text token |
| End | `data: {"end": true}` | Stream complete |

### Request/Response Flow

**New Stream (POST /api/chat/stream):**
1. `routes.stream_chat` validates the `ChatRequest` (message 1-10000 chars)
2. `file_storage.create_conversation(conversation_id)` ensures an empty entry exists
3. `file_storage.append_message(conversation_id, "user", request.message)` stores the user message synchronously (so the conversation appears in the sidebar with the correct title while the LLM is still generating)
4. `get_or_create_job(conversation_id, [])` returns the existing `StreamJob` or creates a new one (pure in-memory; no storage side effect)
5. If the job is not already `active`, `job.reset()` clears `chunks` and sets `status = "active"`
6. `asyncio.create_task(chat_service.generate_background(...))` starts the background LLM task
7. The route returns a `StreamingResponse` wrapping `stream_from_active_job(job)` — the initial stream. **No cleanup wrapper** is applied; the job stays in `STREAM_REGISTRY` for a possible future resume.

**Resume Stream (GET /api/chat/stream/{conversation_id}):**
1. `get_job(conversation_id)` returns the existing job; 404 if none
2. If `job.status == "active"`, use `stream_from_active_job(job, from_pointer)`; otherwise `stream_from_inactive_job(job, from_pointer)`
3. The route returns a `StreamingResponse` wrapping `consume_with_cleanup(gen, conversation_id)` — the entry is removed from `STREAM_REGISTRY` after the resume delivers the full cached chunk history (including `end`)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/stream` | Start new chat stream |
| GET | `/api/chat/stream/{conversation_id}?from_pointer=N` | Resume stream from position N; removes the `StreamJob` from `STREAM_REGISTRY` on full completion |
| GET | `/api/chat/stream/status/{conversation_id}` | Get stream status (single `status` string) |
| GET | `/api/chat/history/{conversation_id}` | Get conversation history |
| GET | `/api/chat/conversations` | List all conversations |
| DELETE | `/api/chat/conversation/{conversation_id}` | Delete conversation + clear stream job |
| POST | `/api/rag/upload` | Upload a file; small files (≤ `RAG_INLINE_CONTEXT_THRESHOLD_BYTES`) come back inline, large files are FAISS-indexed |
| POST | `/api/rag/library/reindex` | Rebuild the global library FAISS index from `storage/library/` |
| GET | `/api/rag/stats` | RAG stats (chunk counts, conversations, threshold) — 503 when `RAG_ENABLED=false` |

---

## 5. Data Storage

### conversations.json

```json
{
  "conversations": {
    "uuid-1": {
      "conversation_id": "uuid-1",
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "thinking": "..."}
      ],
      "created_at": "ISO8601",
      "updated_at": "ISO8601"
    }
  }
}
```

Note: `thinking` field is optional. History API returns thinking field for assistant messages.

### Storage Operations

| Function | Behavior |
|----------|----------|
| `get_conversation(id)` | Retrieve conversation or `None` (no lock) |
| `save_conversation(id, messages)` | Acquire `_write_lock`; load → replace messages → atomic write |
| `append_message(id, role, content)` | Acquire `_write_lock`; load → append → atomic write; return the messages list |
| `create_conversation(id)` | Acquire `_write_lock`; load → ensure entry exists → atomic write (no-op if already present) |
| `get_conversation_list()` | Return sorted list (updated_at desc; no lock) |
| `delete_conversation(id)` | Acquire `_write_lock`; load → remove entry → atomic write |

The four write functions (`create_conversation`, `save_conversation`, `append_message`, `delete_conversation`) all run inside a per-process `threading.Lock` and write through `_atomic_write_json`, which writes to `<path>.tmp` and `os.replace`s into place. Reads are lock-free; combined with the atomic write, a reader sees either the fully-old or fully-new file. **Error Handling:** Invalid JSON is moved aside as `conversations.json.corrupt` and the application starts fresh with a warning log (the previous behavior of silently overwriting the file is gone).

---

## 6. Frontend Implementation

### Responsibilities

- Manage conversation list UI and state
- Store `currentConversationId` in `localStorage`
- Cache chunks to localStorage during streaming
- On page load: check stream status, resume if needed
- Handle SSE stream parsing and display

### Page Load Flow

```
1. Read currentConversationId from localStorage
2. GET /api/chat/stream/status/{id}
3. If streaming=true → GET /stream/{id}?from_pointer=N to resume
4. If streaming=false → GET /api/chat/history/{id}
```

### localStorage Keys

The frontend maintains two separate caches per conversation: a **chunks cache** for in-flight streaming state and a **history cache** for the full message list (see Section 1.10). All `localStorage` access is via the `cache` module — see Section 1.26.

| Key | Purpose |
|-----|---------|
| `chunks_{conv_id}` | Cached chunks for resume |
| `consumed_{conv_id}` | Current position in stream (renamed from `pointer_{conv_id}`) |
| `streaming_{conv_id}` | Active streaming state per conversation (`"true"` / `"false"`) |
| `history_{conv_id}` | Full message list for fast load on conversation switch/refresh |
| `currentConversationId` | UUID of the currently open conversation (global, not per-conversation) |

### Conversation Switch Flow

```
1. User clicks conversation in sidebar
2. If currently streaming → SSE connection closes
3. Server continues streaming (STREAM_REGISTRY intact)
4. Chunks accumulate in StreamJob
5. Load clicked conversation's history
6. If it was streaming → resume via GET /stream/{id}?from_pointer=N
```

### UI Styling

- Centered chat container (max 1100px)
- User messages: right-aligned, blue bubble
- Assistant messages: left-aligned, gray bubble
- Thinking section: displayed above response, collapsible
- Loading indicator: "Thinking" with animated dots

### Frontend Structure

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
    <div class="message-content"></div>
  </div>
</div>
```

### CSS Classes

| Class | Purpose |
|-------|---------|
| `.thinking-section` | Container for thinking content |
| `.thinking-content` | The actual thinking text |
| `.thinking-toggle` | Show more/less button |
| `.thinking-collapsed` | Applied when thinking is collapsed |
| `.scrollbar-visible` | Override scrollbar hiding |
| `.empty` | On messages container when no messages |
| `.message-content` | Rendered markdown content (assistant body) |

### JavaScript Functions

| Function | Responsibility |
|----------|---------------|
| `processStreamResponse()` | Parse SSE via `sseBuffer` accumulator; handle thinking + token events; reuse parser/renderer across chunks; accept `existingRenderer` / `existingParser` and return `{renderer, parser}` for parser handoff on resume |
| `renderContent()` | Lazy-create parser/renderer; write streaming-markdown tokens; apply LaTeX |
| `addMessage()` | Create message element with proper structure |
| `addAssistantPlaceholder()` | Thin wrapper: calls `addMessage('assistant', '')` and sets the content div's class to `loading` with the loading-dots markup |
| `updateThinkingDisplay()` | Handle thinking content and fold/unfold |
| `setupScrollbarAutoHide()` | Attach wheel listener to message blocks |
| `autoResizeInput()` | Expand textarea with content |
| `resumeStreamFromPosition()` | Resume stream from cache; thread parser/renderer to `processStreamResponse` for handoff |
| `loadConversation()` | Cache-first load: read `history_{convId}`; fetch from backend on miss and store; clear stale `chunks_{convId}` on success |
| `sendMessage()` | Append user message to UI, then to history cache; trigger stream and `loadConversationList()` |
| `deleteConversation()` | Remove conversation + all related caches (history + chunks + consumed + streaming) |
| `loadConversationList()` | Derive streaming badge from `cache.isStreaming(convId)` on every render (not just once at send time) |
| `startNewChat()` | Abort in-flight stream via `currentAbortController`; clear current conversation; reset input + send button; refocus input |
| `renderMessagesFromCache()` | Re-render message list from cached or fetched messages array |
| `showStreamingBadge()` | Show/hide the streaming badge on the active sidebar item |

All `localStorage` reads/writes are delegated to the `cache` module imported from `frontend/static/cache.js`. The seven old localStorage helper functions (`getHistoryCache`, `setHistoryCache`, `appendToHistoryCache`, `clearHistoryCache`, `clearChunkCache`, `isStreamingForConv`, `getStreamingForConv`, `setStreamingForConv`) and the `STORAGE_KEYS` constant are gone from `app.js`.

---

## 7. StreamJob Architecture

### StreamJob (stream_manager.py)

```python
class StreamJob:
    def __init__(self, conversation_id, messages=None):
        self.conversation_id = conversation_id
        self.status: Literal["pending", "active", "completed", "failed"] = "pending"
        self.chunks: List[dict] = []  # [{"chunk": text, "type": "thinking|token", "message_id": str}]
        self.chunk_queue: asyncio.Queue = asyncio.Queue()
        self.messages: List[dict] = messages or []
        self.error: str | None = None
        self.cancelled: bool = False  # Set by clear_job when the user deletes the conversation
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def append_chunk(self, chunk_type, text):
        message_id = str(time.time_ns())
        chunk = {"chunk": text, "type": chunk_type, "message_id": message_id}
        self.chunks.append(chunk)
        self.chunk_queue.put_nowait(chunk)
        self.updated_at = datetime.now(timezone.utc)

    def reset(self):
        """Reset job state to start a new message in the same conversation."""
        self.status = "active"
        self.chunks = []
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self):
        self.status = "completed"
        self.chunk_queue.put_nowait(None)  # End marker
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error):
        self.status = "failed"
        self.error = error
        self.chunk_queue.put_nowait(None)
        self.updated_at = datetime.now(timezone.utc)
```

### Module-Level Functions

- `get_or_create_job(conversation_id, messages) -> StreamJob`: returns the existing job or creates a new one (pure in-memory; no storage side effect).
- `get_job(conversation_id) -> StreamJob | None`: returns the existing job or `None`.
- `clear_job(conversation_id) -> None`: sets `cancelled = True` on the live job, then removes it from `STREAM_REGISTRY`. The background task holds a local reference to the job and checks `cancelled` between iterations; setting the flag on the live object first ensures the check sees it (see 1.18 for the resurrection-bug rationale).
- `consume_with_cleanup(gen, conversation_id)`: see Section 1.24. Wraps a stream generator and removes the entry from `STREAM_REGISTRY` after a successful resume.

### Background Task Flow

```
POST /api/chat/stream
    ↓
file_storage.create_conversation(conversation_id)
file_storage.append_message(conversation_id, "user", request.message)
    ↓
get_or_create_job(conversation_id, [])
    ↓
if job.status != "active":
    job.reset()
    ↓
asyncio.create_task(chat_service.generate_background(...))
    ↓
return StreamingResponse(stream_from_active_job(job))
    ↓
stream_from_active_job(job):
    async for chunk in _replay_cached_chunks(job, 0): yield
    while True:
        chunk = await job.chunk_queue.get()
        if chunk is None: yield end; return
        yield SSE_event(chunk)
```

```
GET /api/chat/stream/{conversation_id}?from_pointer=N
    ↓
job = get_job(conversation_id); 404 if None
    ↓
if job.status == "active":
    gen = stream_from_active_job(job, from_pointer)
else:
    gen = stream_from_inactive_job(job, from_pointer)
    ↓
return StreamingResponse(consume_with_cleanup(gen, conversation_id))
    ↓
consume_with_cleanup removes the entry from STREAM_REGISTRY
after the resume has fully delivered the cached chunk history.
```

---

## 8. Edge Cases

### 8.1 No Thinking Block

Some responses may not include a thinking block (model behavior, especially on short/simple responses). Handle gracefully:
- If LLM returns no thinking block, immediately send text tokens only
- Frontend renders only the text response without thinking section

### 8.2 Stream Interruption Mid-Thinking

If stream is interrupted during thinking phase:
- StreamJob in memory retains accumulated thinking tokens
- On resume: status endpoint returns streaming state
- Frontend shows partial thinking with "..." continuation indicator
- Backend continues streaming from interruption point

### 8.3 Stream Interruption Mid-Text

If stream is interrupted during text phase (after thinking):
- StreamJob retains partial text
- On resume: status endpoint returns partial_content
- Thinking is already complete, only text resumes

### 8.4 Very Long Thinking

If thinking exceeds 100KB:
- Still stream normally (no truncation in v1)
- Frontend may need to virtualize rendering for very long thinking
- Architecture supports storing large thinking content

### 8.5 Thinking Field Size Limits

For extreme cases (>1MB thinking):
- Not explicitly limited in storage
- Consider adding size check at save time in future version
- Current design allows arbitrary size

### 8.6 Markdown in Thinking

Thinking content is NOT rendered as markdown. It's displayed as plain text to avoid any injection risks.

### 8.7 Partial Resume After Complete

If a conversation was fully completed (`status == "completed"`) but the user sends another message:
- Backend starts a new stream (the old `StreamJob` is reset via `job.reset()`)
- The completion path used the cleanup wrapper on a resume, so the entry may already be gone from `STREAM_REGISTRY`; the route calls `get_or_create_job` which creates a fresh one
- History already contains full thinking + content from the previous turn

### 8.8 Refresh While Streaming

```
init()
    ↓
Read currentConversationId from localStorage (cache module)
    ↓
checkStreamStatus() → GET /api/chat/stream/status/{id}
    ↓
status.status === 'active' (or cache.isStreaming(currentConversationId) === true)
    ↓
Read cache.getConsumed(currentConversationId) → N
    ↓
resumeStreamFromPosition(N)
    ↓
Read cache.getHistory(convId) → renderMessagesFromCache
Read cache.getChunks(convId) → renderCachedChunks (into a fresh assistant message)
    ↓
fetch GET /api/chat/stream/{id}?from_pointer=N  (wrapped in consume_with_cleanup)
    ↓
Receive remaining chunks via SSE
    ↓
On end: cache.appendToHistory, cache.clearChunks, cache.clearConsumed, cache.setStreaming(false)
        (consume_with_cleanup removes the StreamJob from STREAM_REGISTRY)
```

### 8.9 Two-Cache Architecture

The frontend maintains separate history and chunks caches with different update patterns and consumers. All access goes through the `cache` module (see 1.26). Edge cases:

- **Empty cache on first load**: `cache.getHistory(convId)` returns `null`; `loadConversation` falls through to backend and populates the cache.
- **Cache exists but conversation was deleted on another tab**: Local cache remains; backend returns empty array on next load. Cross-tab invalidation is out of scope.
- **Resume streaming after refresh**: `checkStreamStatus` → `resumeStreamFromPosition` continues to use the `chunks_` and `consumed_` keys. After resume, the new assistant message is appended to the `history_` key.
- **Switching conversations mid-stream**: Only the streaming conversation's chunks/consumed are updated. Other conversations' history cache is untouched. On switch back, `loadConversation` reads from cache.
- **Stale chunks cleared on history load**: `loadConversation` calls `cache.clearChunks(convId)` after a successful backend fetch so a stale in-flight chunk cache cannot replay on top of fresh history.

### 8.10 SSE Event Boundary Handling

The frontend SSE parser uses an `sseBuffer` accumulator to handle events that straddle chunk boundaries.

- **Chunk ends mid-event**: Incomplete event kept in buffer, combined with next chunk.
- **Multiple events in one chunk**: All complete events processed; only the trailing partial is buffered.
- **Stream ends mid-event**: Partial is silently dropped (per SSE spec — incomplete events are dropped).
- **Empty `\n\n` at end of chunk**: Produces empty string in events list; ignored by the `event.startsWith('data: ')` guard.
- **Conversation switch mid-stream**: `convId !== currentConversationId` returns early, dropping partial work cleanly.

### 8.11 Parser State Across Chunks and Resume

The streaming markdown parser is hoisted to function scope so it persists across chunks within a single stream, and survives the cache-replay → live-stream boundary on resume.

- **Markdown table split across chunks**: Parser accumulates all rows; table renders correctly when complete.
- **Fenced code block split across chunks**: Parser accumulates opening fence, content, closing fence.
- **Multi-line math block split across chunks**: `applyLaTeX` finds `<equation-block>` tags after parser finalizes.
- **Conversation switch during stream**: `processStreamResponse` returns early; parser is discarded with the message element. On resume, a new parser is created (or the one from cache replay is reused).
- **Cache replay from localStorage**: `renderCachedChunks` reuses the same `renderContent` path; the parser is created once and reused for all cached chunks, then handed off to `processStreamResponse`.
- **Empty content chunk**: `renderContent` early-returns; parser state untouched.
- **No thinking block**: `smd.parser_write` is still called on every token chunk; render path is unchanged.

### 8.12 New Chat UX

- **No in-flight stream**: `currentAbortController` is null; abort block is a no-op.
- **Stream in-flight when new chat clicked**: Stream aborts; new chat is immediate; previous stream's `data.end` handler is a no-op because `currentConversationId` has changed.
- **Sidebar re-render mid-stream**: Streaming badge derived from `isStreamingForConv()`; appears correctly on the active conversation item.
- **Sidebar re-render post-stream**: `isStreamingForConv` returns false; no badge.
- **Rapid double-click on New Chat**: First call aborts; second call sees null controller; both safely no-op.

### 8.13 Backend Stream Resume Boundary Case

- **`from_pointer == 0`, no chunks yet**: Valid; queue loop runs until chunks arrive or end marker.
- **`from_pointer == len(chunks)`, stream still active**: Valid boundary (previously bug); slice is empty, queue loop streams new chunks.
- **`from_pointer == len(chunks)`, stream completed**: Valid; slice is empty, queue loop terminates immediately on `status != "active"`.
- **`from_pointer > len(chunks)`**: Reject (returns nothing).
- **`from_pointer < 0`**: Reject (defensive — frontends should never send negative).

### 8.14 Repeated Refresh During Streaming

- **First refresh during streaming**: Streaming flag is `'true'` → `checkStreamStatus` → `resumeStreamFromPosition` renders cached chunks + processes live stream. DOM intact.
- **Second (or Nth) refresh during streaming**: Same as first. Streaming flag survives (Part 14 / 1.14). `init` doesn't fall back (Part 15 / 1.15). DOM shows cached chunks + live tail.
- **Stream completes naturally between refreshes**: `data.end` handler clears flag and chunks cache. Next refresh sees flag `'false'` → falls back to `loadConversation` → renders full history.
- **Refresh after stream completes**: Flag `'false'` → `loadConversation` → history cache.
- **New conversation in a fresh tab**: Flag `'true'` immediately after send → resume path. If stream is fast and already done, the resume gets the end marker on first read; flag cleared; UI correct.
- **User switches conversation mid-stream**: `switchConversation` aborts `currentAbortController` → fetch throws `AbortError` → catch block logs and returns false → next refresh of the original conversation can resume from chunk cache. Streaming flag preserved.
- **User clicks "New Chat" mid-stream**: `startNewChat` aborts controller + clears `currentConversationId`. No resume possible for the previous conversation from this tab.
- **Genuine stream failure (backend 404, etc.)**: Fetch rejects with non-Abort error. Streaming flag preserved → next refresh retries.

### 8.15 Confirmation Modal

- **User presses Enter while modal is open**: Confirm button gets a click. For `danger: true`, focus is on cancel so Enter cancels instead.
- **User double-clicks confirm**: First click resolves the Promise; modal hides immediately. Second click is a no-op.
- **Page refresh while modal is open**: Modal element is gone with the page. No Promise resolution (caller never sees the result).
- **Multiple concurrent confirmations**: Not supported. Caller awaits the Promise before issuing another.

### 8.16 Batch Delete Selection Mode

- **Enter selection mode with 0 conversations**: Selection mode renders, but list is empty; Delete button disabled.
- **Enter selection mode with 1 conversation**: User can select and batch-delete that one item (effectively a single delete).
- **Delete the active conversation**: After deletion, `startNewChat()` clears current conversation and resets input.
- **Delete all conversations**: Sidebar becomes empty; selection mode exits.
- **Switch conversation while in selection mode**: The row-click handler routes to selection toggle (not switch) — switching is disabled in selection mode.
- **Refresh page while in selection mode**: `selectionMode` is module-level, not persisted. On reload, normal state is restored.
- **Backend DELETE fails for one item**: Other deletions proceed; failed item is logged but UI does not block.

### 8.17 Streaming-Conversation Resurrection

- **User deletes right as LLM finishes**: The pre-save `if job.cancelled: return` catches it.
- **User deletes before any chunks are received**: The mid-loop check triggers on the very first chunk iteration.
- **`clear_job` is called multiple times**: Idempotent — flag is set once; subsequent calls are no-ops on the registry.
- **Normal stream completion (no delete)**: `cancelled` stays `False`; the existing `mark_completed` + `save_conversation` path runs unchanged.
- **LLM errors mid-stream (`generate_background` except branch)**: `mark_failed` runs as before; `save_conversation` is not called by this code path so cancellation is irrelevant.

### 8.18 Smart Auto-Scroll Pin State

- **User at bottom, chunk arrives with content > 50px**: `wasPinnedToBottom = true` (captured before), scroll restored to new bottom.
- **User scrolled up to read earlier content, chunk arrives**: `wasPinnedToBottom = false` (captured before), scroll position preserved.
- **User scrolls back to bottom manually, next chunk arrives**: `wasPinnedToBottom = true` again, scroll pinned.
- **Page is shorter than clientHeight (no scroll possible)**: `scrollHeight - clientHeight <= 0`, always pinned; scroll is a no-op but harmless.
- **Resize of the messages container (window resize) mid-stream**: The helper re-evaluates on every chunk, so the next chunk corrects any drift.
- **Multiple rapid chunks**: Each captures its own pinned state; works correctly even at high token rates.

### 8.19 Selection-Mode Send Guard

- **User in selection mode, clicks Send button**: `sendMessage()` returns immediately — no message, no state change.
- **User in selection mode, presses Enter in textarea**: Same — `sendMessage()` returns.
- **User in selection mode, types text and tries to send**: Text stays in the input; no send.
- **User exits selection mode (Cancel), then sends**: Guard is a no-op; normal flow.
- **`sendMessage()` called programmatically while in selection mode**: Returns immediately — future-proofs against any other call path.

### 8.20 Multi-Turn Thinking Continuity

- **First turn of a new conversation**: No prior assistant; no `thinking` to feed back. `convert_messages` returns only `HumanMessage`s. LLM emits `thinking` + tokens normally.
- **Second turn**: `convert_messages` reads the saved first assistant message (with `thinking`), constructs `AIMessage(content=[{type:thinking,...}, {type:text,...}])`, and prepends it before the new `HumanMessage`. LLM sees its own prior reasoning and continues to emit `thinking` on the second turn.
- **Prior assistant without `thinking`** (e.g., older storage): `convert_messages` returns a plain `AIMessage(content=str)` so the LLM still sees a coherent conversation history; no list, no content blocks.
- **Unknown roles** (e.g., `system`): silently dropped by `convert_messages`. The LLM never sees them.

### 8.21 Stream Registry Memory Cleanup

- **Resume drains all cached chunks and yields `end`**: `consume_with_cleanup` removes the entry from `STREAM_REGISTRY`.
- **Resume client disconnects before `end` is yielded** (e.g., `wrapper.aclose()` from a closed SSE connection): the wrapper's `GeneratorExit` propagates, `completed` stays `False`, the entry is kept.
- **Resume is for an out-of-range `from_pointer`**: the inner generator returns without yielding; `any_event` stays `False`, the entry is kept.
- **Resume's inner generator raises** (e.g., LLM error): exception propagates through the wrapper, `completed` stays `False`, the entry is kept.
- **Two concurrent resumes where the first finishes**: the first `pop` removes the entry; the second `pop` is a no-op (`STREAM_REGISTRY.pop(..., None)` returns `None`).
- **Initial stream (POST) is interrupted**: no `consume_with_cleanup` is applied; the entry stays so a later resume is possible.
- **Initial stream (POST) completes normally**: no `consume_with_cleanup` is applied; the entry stays. (A `StreamJob` for which the user never resumes leaks for the lifetime of the process — a known limitation, addressed by a separate TTL-based sweep item.)

### 8.22 File Storage Concurrency Safety

- **Two concurrent `append_message` calls on the same conversation**: the first acquires `_write_lock` and runs to completion; the second waits on the lock, then loads the first's committed state and appends. **No lost updates.**
- **Two concurrent `save_conversation` calls**: serialized by the lock. The last writer wins for the messages field (this is the documented contract of `save_conversation` — it replaces, not merges). The on-disk file is always fully valid JSON.
- **A process crash (SIGKILL, OOM, power loss) during a write**: the `.tmp` file is in the same directory and the `os.replace` is atomic; a crash before the replace leaves the original file fully intact; a crash after the replace means the new file is in place. No partial state.
- **A failure during the write** (e.g., `json.dump` raises): the `except` in `_atomic_write_json` removes the `.tmp` file and re-raises. The original `conversations.json` is untouched.
- **A failure during the replace** (e.g., `os.replace` raises): same — `.tmp` is removed, original untouched, exception re-raised.
- **Corrupt JSON at load time** (e.g., from a file that was corrupted before this fix was applied): renamed to `conversations.json.corrupt`; the application starts fresh with a warning log.
- **Multi-worker deployment (`uvicorn --workers N`)**: out of scope. The per-process lock would not serialize across processes. A file-level lock (`fcntl.flock` / `msvcrt.locking`) would be needed.

---

## 9. Error Handling

### Backend

| Scenario | Handling |
|----------|----------|
| LLM API error | `job.mark_failed()`, queue puts None |
| Queue timeout | Check `job.status` in loop, exit if not active |
| Job not found | Return 404 |

### Frontend

| Scenario | Handling |
|----------|----------|
| Stream fetch fails | Show error message, retry button |
| Parse error | Ignore malformed SSE lines |

---

## 10. Cleanup

### When conversation is deleted

```python
@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    clear_job(conversation_id)  # Set cancelled=True, then remove from STREAM_REGISTRY
    deleted = file_storage.delete_conversation(conversation_id)
    return {"deleted": deleted}
```

### After a successful resume

`consume_with_cleanup` (applied only to the resume route) removes the entry from `STREAM_REGISTRY` once the resume has delivered the full cached chunk history (including `end`).

### After a successful initial stream

The initial stream (`POST /api/chat/stream`) does not have `consume_with_cleanup` applied. The `StreamJob` stays in `STREAM_REGISTRY` until either (a) the user calls resume and the cleanup fires, or (b) the user deletes the conversation and `clear_job` removes it. A `StreamJob` for which neither ever happens leaks for the lifetime of the process — a known limitation, not addressed in this iteration.

### Frontend cleanup on `data.end`

On `data.end`, `processStreamResponse` calls `cache.appendToHistory(...)` (to add the assistant message to the history cache), `cache.clearChunks(convId)`, and `cache.setStreaming(convId, false)`. The `consumed_` key was already cleared earlier in the stream (in the per-chunk path, `cache.setConsumed(convId, consumedCount)` updates it; on `data.end`, no further consumed cleanup is needed since the next page load will use the history cache).

---

## 11. Testing Strategy

### Test Approach

- Mock LLM calls via `langchain.anthropic.ChatModel`
- Use temporary directories for storage tests
- HTTP tests via `httpx.AsyncClient` against FastAPI TestClient

### Test Coverage

| File | What is Tested |
|------|----------------|
| `test_chat_service.py` | `ChatService.generate_background()` with mocked LLM (thinking + token extraction, string content, append_chunk, cancellation, failure handling) |
| `test_storage.py` | JSON read/write roundtrip; list sorting; title truncation; delete; invalid-JSON recovery; `TestAtomicWrite` (replace-fails-original-intact, dump-fails-tmp-cleaned, success-no-tmp-left); `TestWriteLock` (50 concurrent appends, 20 concurrent saves serialized) |
| `test_chat_routes.py` | `stream_from_inactive_job` / `stream_from_active_job`; pointer / boundary / out-of-range; `job.reset()` |
| `test_stream_manager.py` | `StreamJob` state transitions; unified chunks list + chunk_queue; 5 `consume_with_cleanup` tests (full consumption, no events, cancellation, exception, idempotent pop) |
| `test_thinking_routes.py` | HTTP tests for thinking-aware endpoints (status partial_content, resume 404, post starts background task, delete clears job); 2 integration tests for `consume_with_cleanup` (resume route cleans up, initial stream does not) |
| `test_chain.py` | `convert_messages` shape: user → `HumanMessage`; assistant without `thinking` → plain `AIMessage`; assistant with `thinking` → `AIMessage(content=[thinking, text])`; multi-turn scenario; unknown roles dropped |

### Test Dependencies

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

### Testing Checklist

#### Thinking Content & Display

**Backend:**
- [x] Thinking blocks are extracted and yielded as `{"chunk": "...", "type": "thinking"}`
- [x] Token blocks yielded as `{"chunk": "...", "type": "token"}`
- [x] `end: true` event sent when stream completes
- [x] History API returns thinking field
- [x] Resume sends accumulated chunks

**Frontend:**
- [x] Thinking displayed above response with "Show more" toggle when >3 lines
- [x] Message blocks scroll internally
- [x] Scrollbar auto-hides after 3s on wheel
- [x] Empty state input is centered
- [x] Input expands with content (field-sizing CSS + JS fallback)
- [x] Markdown renders correctly using marked.js
- [x] Resume works with cached chunks from localStorage

#### Stream Resume

**Backend:**
- [x] Background task stores chunks in StreamJob
- [x] Queue delivers chunks to /stream readers
- [x] `from_pointer` parameter skips already-sent chunks
- [x] Status API returns `chunks_count`
- [x] Job cleanup on conversation delete

**Frontend:**
- [x] Chunks cached to localStorage on each receive
- [x] Pointer tracked and updated on each chunk
- [x] Init checks stream status before loading history
- [x] resumeStream passes `from_pointer` correctly
- [x] Pointer cleared on stream complete
- [x] Resume renders cached chunks first, then continues streaming

#### Repeated Refresh During Streaming

**Frontend:**
- [x] Streaming flag survives transient fetch errors (no longer cleared on `error.name !== 'AbortError'`)
- [x] `init` does not fall back to `loadConversation` when the streaming flag is set
- [x] 2nd, 3rd, 5th refresh during streaming all show the partial assistant message

#### Confirmation Modal

**Frontend:**
- [x] `confirm()` is gone — replaced by `showConfirmModal`
- [x] Modal is centered both horizontally and vertically
- [x] Backdrop click and Escape close (cancel)
- [x] Cancel and Confirm buttons work
- [x] `danger: true` produces a red confirm button
- [x] Focus moves to cancel when `danger: true` (Enter doesn't accidentally delete)

#### Batch Delete

**Frontend:**
- [x] Sidebar-header shows "Batch Delete" instead of `≡`
- [x] Selection mode shows checkboxes; clicking row toggles
- [x] Header count updates as items are toggled
- [x] Delete button disabled when count is 0
- [x] Cancel exits selection mode
- [x] Delete opens modal; confirm deletes all; cancel keeps all
- [x] Active conversation deletion → auto new chat
- [x] Single-item `×` delete still works and uses the new modal
- [x] Sidebar collapse via chat-header `≡` still works

#### Streaming-Conversation Resurrection Fix

**Backend:**
- [x] `test_generate_background_aborts_on_cancellation` passes — proves the resurrection bug is fixed
- [x] Normal stream completion still saves the assistant message to history
- [x] Deleting a conversation that has no in-flight LLM task still works

#### Smart Auto-Scroll During Streaming

**Frontend:**
- [x] Streaming with no user interaction: scroll stays pinned to the bottom throughout
- [x] User scrolls up during streaming: scroll position stays where the user put it
- [x] User scrolls back to the bottom: next chunk re-pins the scroll
- [x] Refresh-during-streaming (regression check): cached-chunks replay still scrolls to the bottom

#### Selection-Mode Send Guard

**Frontend:**
- [x] Click Send in selection mode → no message sent, input value preserved, selection intact
- [x] Press Enter in selection mode → no message sent, input value preserved, selection intact
- [x] Exit selection mode (Cancel) → Send and Enter both work normally
- [x] Deleting the active conversation via single `×` still ends in a fresh empty chat (no regression)
- [x] Deleting the active conversation via batch-delete (selected) still ends in a fresh empty chat (no regression)

#### Multi-Turn Thinking Continuity

**Backend (`convert_messages`):**
- [x] User message → `HumanMessage(content=...)`
- [x] Assistant without `thinking` → `AIMessage(content=str)`
- [x] Assistant with `thinking` → `AIMessage(content=[{type:thinking,...}, {type:text,...}])`
- [x] Multi-turn: prior assistant with `thinking` becomes a content-block `AIMessage`; the chain stays coherent
- [x] Unknown roles (e.g., `system`) are dropped

**Live (real LLM):**
- [x] 2-turn conversation: turn 2 stores a non-empty `thinking` field
- [x] 3-turn conversation: turn 3 stores a non-empty `thinking` field

#### Stream Registry Memory Cleanup

- [x] Resume via `GET /api/chat/stream/{id}` drains to completion → entry removed from `STREAM_REGISTRY`
- [x] `POST /api/chat/stream` (initial stream) drains to completion → entry **stays** in `STREAM_REGISTRY`
- [x] Resume whose `wrapper.aclose()` is called (client disconnect simulation) → entry stays
- [x] Resume with an out-of-range `from_pointer` (no events) → entry stays
- [x] Resume whose inner generator raises → entry stays
- [x] Two concurrent resumes where the first finishes → no error on the second's `pop`

#### File Storage Concurrency Safety

- [x] 50 concurrent `append_message` threads → all 50 messages present in the final file
- [x] 20 concurrent `save_conversation` threads → all complete; file is always valid JSON; no `.tmp` left
- [x] `os.replace` fails mid-swap → original file fully intact; no `.tmp` left
- [x] `json.dump` fails mid-write → original file fully intact; no `.tmp` left

#### Frontend Cache Consolidation

- [x] `app.js` contains no `localStorage`, no `STORAGE_KEYS`, no `JSON.parse` / `JSON.stringify` for cached state
- [x] The only two remaining `JSON.*` calls in `app.js` are the `fetch` request body and SSE event parsing (both legitimate)
- [x] `cache.js` is served correctly at `/static/cache.js` (200)
- [x] All 47 backend tests still pass

---

## 12. Future Extension Points

### Adding Authentication

1. Create `backend/auth/` domain
2. Add FastAPI dependency `get_current_user`
3. Apply via `Depends(get_current_user)` on routes

### Adding Additional RAG Scopes

1. Add a new FAISS index to `backend/rag/service.py` (one more `load_or_init` call)
2. Add `(index.as_retriever(...), should_filter)` to `make_scoped_retriever`
3. Add a bool field to `RetrievalConfig`

### PostgreSQL Migration

1. Replace `storage/file_storage.py` with `storage/db_storage.py`
2. Update `chat/service.py` to use new storage
3. Keep interface the same

---

## 13. File Inventory

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry, serves frontend, mounts `/static` from `frontend/static/`, includes chat router; lifespan builds `RagService` and wires `purge_uploads` + `clear_pending_inline_files` into `delete_conversation` when `RAG_ENABLED=true` |
| `backend/config.py` | Pydantic Settings from environment variables |
| `backend/chat/routes.py` | `/api/chat/*` endpoints; `stream_from_active_job` / `stream_from_inactive_job`; `_sse` and `_serialize_chunk` helpers; `consume_with_cleanup` is applied to the resume route's `StreamingResponse`; `stream_chat` calls `file_storage.create_conversation` + `file_storage.append_message` synchronously; `ChatRequest` gains optional `retrieval: RetrievalConfig` and `uploaded_files: list[UploadedFile]` |
| `backend/chat/chain.py` | `convert_messages` is module-level: `HumanMessage` for user turns; `AIMessage` for assistant turns (plain string when no `thinking`, content-block list when there is). `create_chain` wires it to `ChatAnthropic` via `RunnableLambda \| llm`. Passes `base_url=settings.anthropic_base_url` directly. Unchanged in this iteration (RAG happens in `service.py`, not the chain). |
| `backend/chat/service.py` | `ChatService.generate_background` reads history from `file_storage` as the single source of truth; trusts it, no dedupe. Checks `job.cancelled` in the loop and before `save_conversation`. Optional `rag_service` and `_pending_inline_files` dict for retrieval-augmented and inline-upload paths. |
| `backend/chat/stream_manager.py` | `StreamJob` (with `cancelled` flag and `reset()` method); `STREAM_REGISTRY`; `get_or_create_job` / `get_job` / `clear_job`; `consume_with_cleanup` |
| `backend/rag/__init__.py` | Public surface: `RagService`, `ScopedRetriever` |
| `backend/rag/config.py` | `RagSettings` (independent of `backend/config.py`) reading `RAG_*` env vars |
| `backend/rag/service.py` | `RagService` facade: `ingest_file`, `purge_uploads`, `reindex_library`, `make_scoped_retriever`, `persist_all`, `stats` |
| `backend/rag/retriever.py` | `ScopedRetriever` (`BaseRetriever` subclass) merging multiple per-scope retrievers with explicit conversation_id filtering |
| `backend/rag/vector_store.py` | FAISS load/save/rebuild helpers: `load_or_init`, `save`, `rebuild_filtered` |
| `backend/rag/embeddings.py` | Embeddings factory: `make_embeddings(backend)` returns a `langchain_core.embeddings.Embeddings` instance |
| `backend/rag/splitter.py` | `make_splitter(chunk_size, chunk_overlap)` returning a `RecursiveCharacterTextSplitter` |
| `backend/rag/routes.py` | FastAPI routes: `POST /api/rag/upload` (size-routed), `POST /api/rag/library/reindex`, `GET /api/rag/stats` — each gated by `RAG_ENABLED` (503 otherwise) |
| `backend/storage/file_storage.py` | Per-process `_write_lock` (threading); `_atomic_write_json` helper (tmp + `os.replace`); corrupt-JSON-aside recovery; the four write functions are wrapped in the lock and use the atomic helper; `delete_conversation` accepts optional `on_delete` callback |
| `frontend/index.html` | Two-column comparison layout (Vanilla + RAG), shared input, upload + send buttons, CDN scripts (katex, dompurify, streaming-markdown), modal markup |
| `frontend/static/styles.css` | All UI styling (theme tokens, layout, animations, responsive, modal, two-column) |
| `frontend/static/app.js` | Per-column SSE stream processing, two-column fan-out, abort + resume per column, upload + sources rendering, modal flow. Imports `cache` from `./cache.js` for all `localStorage` access. No raw `localStorage` or `JSON.parse` / `JSON.stringify` of cached state. |
| `frontend/static/cache.js` | The single owner of `localStorage` access: typed accessors for `chunks`, `history`, `consumed`, `streaming`, `currentConversationId`, plus `getBaseConversationId` / `setBaseConversationId` for the paired-column base UUID |
| `pyproject.toml` | Pytest config: `asyncio_mode = "auto"`, `testpaths = ["backend/tests"]` |
| `backend/tests/conftest.py` | Pytest fixtures: `temp_storage_dir` (uses `monkeypatch` + `tmp_path` to redirect `STORAGE_DIR` / `CONVERSATIONS_FILE`); `mock_chain`; RAG fixtures for `FakeEmbeddings` + temp FAISS dirs |
| `backend/tests/test_chat_routes.py` | `stream_from_inactive_job` / `stream_from_active_job`; pointer / boundary / out-of-range; `job.reset()` |
| `backend/tests/test_chat_service.py` | `ChatService.generate_background()` with mocked LLM (thinking + tokens, string content, append_chunk, cancellation, retrieval pre-processing) |
| `backend/tests/test_storage.py` | `TestStorage`, `TestConversationList`, `TestDeleteConversation`, `TestAtomicWrite` (3 tests), `TestWriteLock` (2 tests), `TestOnDeleteCallback` |
| `backend/tests/test_stream_manager.py` | `StreamJob` state transitions; unified chunks list + chunk_queue; 5 `consume_with_cleanup` tests |
| `backend/tests/test_thinking_routes.py` | HTTP tests for status, resume 404, post starts background task, delete clears job; 2 integration tests for `consume_with_cleanup` (resume route cleans up, initial stream does not) |
| `backend/tests/test_chain.py` | `convert_messages` shape: user → `HumanMessage`; assistant without `thinking` → plain `AIMessage`; assistant with `thinking` → content-block `AIMessage`; multi-turn scenario; unknown roles dropped |
| `backend/tests/rag/test_retriever.py` | `ScopedRetriever` merging, metadata filter, per-scope cap (no merged-result cap) |
| `backend/tests/rag/test_service.py` | `RagService` ingest / purge / reindex using real FAISS in `tmp_path` + `FakeEmbeddings` |
| `backend/tests/rag/test_embeddings_factory.py` | `make_embeddings` returns the correct LangChain `Embeddings` subclass per backend |
| `backend/tests/rag/test_splitter.py` | Chunk size, overlap, metadata assembly |
| `backend/tests/rag/test_chain_integration.py` | `ChatService` with a fake retriever: sources pushed, augmentation correct; byte-identical non-RAG path |
| `backend/tests/rag/test_routes.py` | Upload + reindex + stats via `TestClient` (incl. size-based routing) |

---

## 14. Environment Configuration

```env
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1
ANTHROPIC_API_KEY=your-api-key-here
RAG_ENABLED=false
EMBEDDING_BACKEND=sentence-transformers
RAG_LIBRARY_DIR=storage/library
RAG_UPLOADS_DIR=storage/uploads
RAG_INDEX_DIR=storage/rag
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4
RAG_INLINE_CONTEXT_THRESHOLD_BYTES=8192
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
```

### Runtime Dependencies

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
langchain>=0.1.0
langchain-anthropic>=0.1.0
langchain-community>=0.0.20
pydantic>=2.0
pydantic-settings>=2.0
python-multipart>=0.0.6
faiss-cpu
sentence-transformers
pypdf
```

---

## 15. RAG Module

This section consolidates the implementation-level details for the RAG plugin introduced by the architecture decisions in Section 1.28-1.38. Section 1 covers the *why*; this section covers the *what* — module layout, component skeletons, request flow, and operational concerns. The full brainstorming artifact is in `docs/superpowers/specs/2026-06-26-rag-module-design.md`.

### 15.1 Module Layout

```
backend/rag/
├── __init__.py          # public surface: RagService, ScopedRetriever
├── config.py            # RagSettings (RAG_ENABLED, EMBEDDING_BACKEND, paths, chunk params)
├── service.py           # RagService — ingest, purge, reindex, make_scoped_retriever, stats
├── retriever.py         # ScopedRetriever (merge + explicit metadata filter)
├── vector_store.py      # FAISS load/save/rebuild helpers
├── embeddings.py        # Embeddings factory: sentence-transformers or MiniMax
├── splitter.py          # Text splitter factory + chunk metadata assembly
└── routes.py            # /api/rag/* endpoints (upload, library reindex, stats)
```

The `rag/` domain is independent of `backend/config.py` — `rag/config.py` has its own `RagSettings` so the global config has zero RAG awareness when `RAG_ENABLED=false`.

### 15.2 Component Skeletons

**`ScopedRetriever`** — a `BaseRetriever` subclass holding `retrievers: list[tuple[BaseRetriever, bool]]` and `conversation_id: str`. Per-scope cap is applied by the underlying retrievers via `search_kwargs={"k": k}`; `ScopedRetriever` does **not** cap the merged result (so top-K hits from each scope accumulate). Convention: library chunks have metadata `source="library"` and no `conversation_id` field; upload chunks have both. Library retrievers get `should_filter=False`; upload retrievers get `should_filter=True`.

```python
class ScopedRetriever(BaseRetriever):
    retrievers: list[tuple[BaseRetriever, bool]]
    conversation_id: str

    def _get_relevant_documents(self, query, *, run_manager):
        hits = []
        for r, should_filter in self.retrievers:
            r_hits = r.invoke(query)
            if should_filter:
                r_hits = [d for d in r_hits
                          if d.metadata.get("conversation_id") == self.conversation_id]
            hits.extend(r_hits)
        return hits
```

**`RagService`** — the only module other domains import. Holds the long-lived `library_index`, `uploads_index`, `embeddings`, `splitter`, and the per-conversation uploads directory. Index paths are tagged with the embedding backend name (e.g. `storage/rag/library_index.sentence-transformers/`) so switching `EMBEDDING_BACKEND` produces a fresh path and forces an explicit reindex — preventing the silent-failure mode where a stale index built with a different model is loaded.

`RagService` exposes: `ingest_file(conversation_id, file_path)` (FAISS pipeline, returns chunk IDs), `reindex_library()` (rebuild from `RAG_LIBRARY_DIR`), `purge_uploads(conversation_id)` (delete files + rebuild uploads index without re-embedding), `make_scoped_retriever(conversation_id, top_k)`, `persist_all()`, `stats()`.

**`vector_store.py`** — three helpers:
- `load_or_init(path, embeddings)` returns a `FAISS.load_local` if present, else creates an empty `FAISS` with a single placeholder doc (avoids the `from_documents([])` error and keeps the next `add_documents` call working).
- `save(index, path)` writes `index.faiss` + the docstore sidecar.
- `rebuild_filtered(index, embeddings, keep)` walks `index.docstore._dict`, filters by `keep(doc)`, returns a fresh `FAISS.from_documents(surviving, embeddings)` (or `load_or_init` for the empty case). This is the load-bearing trick that makes conversation deletion O(chunks), not O(re-embedding).

### 15.3 Modified ChatService Flow

`ChatService.__init__` accepts an optional `rag_service: RagService | None` and holds a per-process `dict[str, list[dict]]` of pending inline files keyed by `conversation_id`. `generate_background(message, conversation_id, retrieval=None, uploaded_files=None)` runs two pre-processing blocks before `self.chain.astream(messages)`:

1. **Inline-files block** (when `uploaded_files` is non-empty): merge the new uploads into the pending list for that conversation; emit a `sources` event listing each file's `{filename, scope: "upload", excerpt: first 300 chars}`; insert a system message before the last user message of the form `"Use this uploaded file as context:\n\n[filename]:\n<content>"`.
2. **RAG block** (when `retrieval is not None and self.rag_service is not None and pending is empty`): build a `ScopedRetriever` via `rag_service.make_scoped_retriever(conversation_id, retrieval.top_k)`; invoke it on the latest user message; if hits are returned, emit a `sources` event and insert a `"Use this retrieved context:\n<chunks>"` system message before the last user message.

The two blocks are **mutually exclusive per turn**: inline files take precedence over FAISS retrieval. If retrieval raises, the LLM call proceeds with the un-augmented messages and the error is logged.

`clear_pending_inline_files(conversation_id)` is wired into `delete_conversation` alongside `rag_service.purge_uploads` (see 15.4).

### 15.4 Startup Wiring

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if RagSettings().enabled:
        rag = RagService.from_settings()
        app.state.rag = rag
        original_delete = file_storage.delete_conversation
        file_storage.delete_conversation = partial(
            original_delete, on_delete=rag.purge_uploads
        )
        app.include_router(rag_router)
    yield
    if hasattr(app.state, "rag"):
        app.state.rag.persist_all()
```

Both `rag.purge_uploads` and `chat_service.clear_pending_inline_files` are chained into a single `partial` callback (the storage layer only takes one `on_delete`), so a single `delete_conversation(id)` call cleans up both FAISS uploads and in-memory pending files for that conversation.

### 15.5 Error Handling

| Failure | Behavior |
|---|---|
| `RAG_ENABLED=false`, client sends `retrieval` | Server ignores field, runs vanilla. Debug log. |
| Uploaded file unreadable / too large | 400, file not saved. |
| Embedding API down | 500, file kept on disk, index untouched. User retries. |
| FAISS save fails after `add_documents` | 500, critical log. In-memory state consistent, disk stale until next reindex. |
| Retrieval fails mid-turn | Stream error chunk; LLM call proceeds with un-augmented messages. |
| `delete_conversation` succeeds, `purge_uploads` fails | JSON state consistent; orphan chunks invisible at query time. Warning logged. |
| Library reindex encounters unreadable file | File skipped, error in response `errors` list. Run continues. |
| `/api/rag/*` endpoint hit when `RAG_ENABLED=false` | 503. |

### 15.6 Testing Strategy

The plugin-off property is asserted by `test_no_rag_path_is_byte_identical_to_today`: when `retrieval=None`, `ChatService.generate_background` produces exactly the same `job.chunks` as the iteration-6 version (no `sources` chunk ever appended; LLM called with the original messages).

Layers:
- **Unit** (`test_retriever.py`, `test_splitter.py`, `test_embeddings_factory.py`): <10 ms each.
- **Service** (`test_service.py`): real FAISS in `tmp_path` + `FakeEmbeddings`, <500 ms each.
- **Chain integration** (`test_chain_integration.py`): `ChatService` with a fake retriever, <1 s each.
- **Route** (`test_routes.py`): `TestClient` against FastAPI app with a test `RagService`, <1 s each.

Frontend verification is manual for v1: page loads with both panels empty, type-and-send fans out to two streams, sources appear per panel, uploads route by size, refresh-during-stream resumes per column, and deletion cleans up both columns and on-disk uploads.

### 15.7 Future Work (post-v1, documented but out of scope)

1. **Tombstone-based delete** — mark chunks deleted in a set, exclude at query time, periodic compaction. Replaces full-rebuild when indexes grow.
2. **SQLite/DuckDB docstore** — replaces the FAISS pickle sidecar when chunk counts exceed ~50k.
3. **Cross-encoder re-ranking** — improves retrieval precision.
4. **Multi-process locking** — `fcntl`/`msvcrt` file lock for FAISS index (and `conversations.json`).
5. **Background ingestion** — long uploads don't block the request; client polls for completion.
6. **Additional scopes** — "web cache", "agent memory", "tool results". Pattern: add FAISS index + entry in `ScopedRetriever.retrievers` + bool flag in `RetrievalConfig`.
7. **Per-conversation FAISS index** — if `uploads` grows large enough that one index becomes a hotspot.
8. **Auth on `/api/rag/library/reindex`** — when the app leaves local-only.
9. **Chunk deduplication** — manifest-based skip-if-unchanged during library reindex.
10. **Persist pending inline files** to disk (e.g., `storage/inline_uploads/`) so a server restart doesn't lose small-upload context.

---

## 16. HotpotQA Library Ingest + Retrieval Eval Pipeline (Iteration 9)

The eval pipeline sits orthogonally next to chat. Chat core, the chat-time library reindex path, the chat-time RAG chain, and the frontend are unchanged. Two CLI scripts (`scripts/ingest_hotpotqa.py`, `scripts/eval_hotpotqa.py`) share a small pure-Python package at `backend/eval/`. The chat-time library uses HotpotQA files as if they were user-uploaded documents; the eval pipeline reads the dataset JSON directly and builds its own transient per-question FAISS indices.

### 16.1 Architecture Decisions

**16.1.1 One Markdown File Per Question (Not Per Paragraph)**: Library ingest writes one `.md` per question with each paragraph as an H1 section in the same file. The existing `MarkdownTextSplitter` (in `backend/rag/splitter.py`) splits at H1 boundaries during library reindex, so each H1 section becomes one chunk automatically with `header_path` set to the paragraph title. ~7,405 files instead of ~74k — git, IDEs, and `find` stay snappy. Trade-off: slight index inflation from duplicate paragraph text (~5–15% estimated) due to shared Wikipedia paragraphs across questions.

**16.1.2 Frontmatter-Only Metadata (No Gold Leakage)**: Each library file's frontmatter contains `question_id`, `question_type`, `question_level`, `source` — never `question`, `answer`, or `supporting_facts`. If the question text or gold answer landed in a chunk, a chat-time retrieval against that chunk would let the model see the ground-truth answer in its own context — a trivial form of contamination. Chat UX doesn't need the question or answer in the library; it needs the paragraphs.

**16.1.3 Separate CLI Scripts (No Shared Library Code Beyond Pure Primitives)**: Two scripts share: the downloaded JSON at `scripts/.cache/hotpot_dev_distractor_v1.json`, and a small set of helpers in `backend/eval/` (`hotpotqa.py`, `metrics.py`, `cache.py`). The eval pipeline is forbidden from importing anything under `backend/chat/`. The import surface it may use: `backend.rag.embeddings` (the sentence-transformers factory), `backend.rag.vector_store` (`load_or_init` / `save`), and `backend.eval.*`. A grep guard verifies this at review time.

**16.1.4 Paragraph-Level Document in the Eval Index (Not MarkdownTextSplitter)**: `backend/eval/cache.py::_build_index` constructs `Document(page_content=paragraph_text, metadata={...})` for each paragraph in `item.context` — no splitter is invoked. The chat pipeline's `MarkdownTextSplitter` will produce paragraph-level chunks for these files only if each H1 section is short enough relative to `rag_chunk_size`; decoupling the eval's retrieval granularity from the chat pipeline's chunking decisions keeps the metric stable across reindex-config changes.

**16.1.5 SHA-Keyed On-Disk Cache for Per-Question Indices**: `backend/eval/cache.py::load_or_build` keys the cache directory by `dataset_sha[:16]` (sha256 of the dataset JSON). Each question gets its own subdirectory `cache/{dataset_sha[:16]}/{qid}/` containing FAISS's native `index.faiss` + `index.pkl`. Re-downloading the JSON (e.g., if HotpotQA updates the dataset) busts all caches atomically.

**16.1.6 Dataset Auto-Download + .cache Stash**: `scripts/ingest_hotpotqa.py` is the canonical way to acquire the dataset. It downloads once, stashes at `scripts/.cache/hotpot_dev_distractor_v1.json`. `scripts/eval_hotpotqa.py` reads from the stash (or from `--fixture PATH` for tests).

**16.1.7 CC BY-SA 4.0 Attribution in Two Places**: Attribution lives at the file level (`storage/library/hotpotqa/README.md` — written by ingest) and at the run level (eval script prints attribution to stdout before the metric block). The CC BY-SA 4.0 license is a requirement of using the dataset.

**16.1.8 Stratified Sample With Deterministic Seed**: `--subset N` samples `min(ceil(N / 6), len(bucket))` items from each of the 6 `(type, level)` buckets using `random.Random(42)`. The sampled set is concatenated and shuffled with the same RNG.

**16.1.9 Exit Codes: Partial Errors Are Non-Fatal**: `--subset N` or `--full` runs always exit 0 unless setup fails. Per-question retrieval errors are logged at WARNING and counted in the `errors` field.

**16.1.10 Paraphrase Generator Concurrency**: `scripts/generate_paraphrases_hotpotqa.py` runs 3 concurrent LLM calls per question (one per style) via `asyncio.gather`. Each style's task fires its first attempt, validates, and conditionally retries. The 3 tasks run concurrently so wall-clock per question ≈ 2× a single API call's latency (first + possible retry) rather than 3×.

### 16.2 Module Layout

```
backend/eval/
├── __init__.py
├── hotpotqa.py         # HotpotQaItem dataclass, load(), dataset_sha(), gold_paragraph_titles(), sample()
├── metrics.py          # paragraph_recall_at_k(), supporting_fact_metrics(), answer_coverage_at_k()
├── cache.py            # load_or_build(), EVAL_CACHE_ROOT, _build_index()
└── paraphrases.py      # validate_paraphrase(), load_paraphrases(), lookup(), required_styles()

backend/tests/eval/
├── __init__.py
├── fixtures/
│   ├── tiny_hotpot.json           # 3-question fixture for hotpotqa.py tests
│   └── integration_hotpot.json    # 5-question fixture for the eval_integration test
├── test_metrics.py     # pure-function unit tests
├── test_hotpotqa.py    # loader + sha + gold_paragraph_titles + sample
├── test_cache.py       # per-question FAISS cache + corruption recovery
├── test_paraphrases.py # validate / load / lookup / required_styles
├── test_eval_integration.py   # subprocess-driven end-to-end with synthetic JSON
└── test_answer_coverage.py    # answer_coverage_at_k pure-function tests

scripts/
├── ingest_hotpotqa.py        # CLI: download + write library files
├── eval_hotpotqa.py          # CLI: run the eval pipeline
└── generate_paraphrases_hotpotqa.py  # CLI: produce 3 styled paraphrases per question

storage/library/hotpotqa/
└── README.md           # generated by ingest; one-line license notice

storage/eval/hotpotqa/
├── cache/{dataset_sha[:16]}/{qid}/  # FAISS indices (gitignored)
└── paraphrases/{dataset_sha}.json   # paraphrase entries keyed by qid (gitignored)

scripts/.cache/
└── hotpot_dev_distractor_v1.json  # generated by ingest; gitignored
```

### 16.3 Configuration

No new env vars. No new config fields. Reuses existing `EMBEDDING_BACKEND` and the sentence-transformers model wired up in `backend/rag/embeddings.py`. The paraphrase generator uses `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` (default `minimax-3`).

### 16.4 Error Handling

| Stage | Failure | Behavior |
|---|---|---|
| Ingest: download | Network error | One retry after 5s; second failure exits 1 with download URL. |
| Ingest: per-question write | Permission error / disk full | Log path, continue with remaining. Exit non-zero if any file failed. |
| Ingest: whole-file `JSONDecodeError` | JSON corrupt | Exit 1 with "fix the file or re-download" hint. |
| Ingest: per-question schema error | Missing fields | Log WARNING with qid, skip, continue. Final summary lists skipped IDs. |
| Eval: dataset missing | Path doesn't exist | Print expected path + download instructions, exit 1. |
| Eval: dataset corrupt (`JSONDecodeError`) | Parse fails | Print exception tail, exit 1. |
| Eval: embedding model load | sentence-transformers not installed | Fail fast with `pip install -r requirements.txt` hint, exit 1. |
| Eval: per-question cache corrupted | `load_local` raises | `shutil.rmtree(cache_path, ignore_errors=True)`, rebuild, WARNING log, continue. |
| Eval: per-question retrieval | Embedding call raises (transient) | Log WARNING, count as errored, skip rest of run unaffected. |
| Paraphrase: validation gate reject | Token overlap ≥80% | One retry; on second rejection that style is omitted. Other styles kept. |
| Paraphrase: API rate limit (429) | MiniMax endpoint throttling | Backoff + retry handled by Anthropic client. Run continues. |

### 16.5 Testing Strategy

Layers:
- **Metrics unit** (`test_metrics.py`, `test_answer_coverage.py`) — pure tests, no fixtures, <5 ms each.
- **Paraphrase unit** (`test_paraphrases.py`) — pure validate/load/lookup tests, <5 ms each.
- **Loader unit** (`test_hotpotqa.py`) — uses `tiny_hotpot.json`, <50 ms each.
- **Cache unit** (`test_cache.py`) — uses `FakeEmbeddings` and tmp dir, <500 ms each.
- **Generator integration** (`scripts/tests/test_generate_paraphrases_hotpotqa.py`) — mocks `AsyncAnthropic`, verifies concurrency, retry, idempotence, schema. <1 s each.
- **Eval integration** (`test_eval_integration.py`) — invokes `scripts/eval_hotpotqa.py` via subprocess with synthetic JSON. Asserts exit 0, all metric labels, cache hit/build split. <5 s.

Manual smoke test (full scale):
```bash
python scripts/ingest_hotpotqa.py --full   # ~minutes
python scripts/eval_hotpotqa.py --full --k 4   # minutes cold, seconds warm
python scripts/generate_paraphrases_hotpotqa.py --subset 100   # ~minutes
python scripts/eval_hotpotqa.py --subset 100 --paraphrase-set <JSON> --k 4
```

### 16.6 Future Work (post-iter-9, documented but out of scope)

1. **LLM-based answer evaluation** (`answer_em`, `answer_f1`) — would require calling `minimax-3` per question, doubling eval cost and adding API-key dependencies at eval-time.
2. **`/api/eval/` route** — CLI only by design; UI integration deferred.
3. **JSON output (`--json-out`)** — easy add later if downstream tooling wants to consume eval results.
4. **Per-type / per-level breakdown in default output** — already implemented for paraphrase-eval pipeline (per `(type, level)` bucket); base eval output still uses aggregate only.
5. **HotpotQA `fullwiki` setting** — requires a separate 5M+ Wikipedia paragraph corpus ingestion pipeline.
6. **Multi-process or distributed evaluation.**
7. **Incremental cache invalidation beyond dataset SHA change.**
8. **CI hookup for the eval script.**
9. **Embedding-recipe sweeps** (automatically try multiple `EMBEDDING_BACKEND` values).
10. **Cross-encoder re-ranking on top of FAISS results.**
11. **Sentence-level supporting-fact metrics** (would require LLM-based extraction).
12. **Surface-attribution in the library sidebar UI** when files with `source=hotpotqa` are present (would require frontend changes).
13. **Multi-pass retrieval** (Hop 2 using Hop-1 results to refine the query) — interesting follow-up for true multi-hop performance, but requires the fullwiki pipeline.
14. **Smarter validation gate** for paraphrase generation (entity-aware prompt, non-zero retry temperature, larger retry budget) — see iteration 10.
