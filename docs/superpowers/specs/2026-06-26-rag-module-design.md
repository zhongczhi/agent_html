# RAG Module — Design Spec

**Date**: 2026-06-26
**Status**: Draft, pending user review
**Iteration goal**: Add a Retrieval-Augmented Generation module as a selective, isolated plugin. The existing chat must run unmodified when RAG is disabled, and gain retrieval capability when enabled. The start implementation ships a side-by-side comparison UI for evaluating RAG impact.

---

## 1. Goals & Non-Goals

### Goals

1. **Plugin isolation**: `RAG_ENABLED=false` produces a chat system that is functionally identical to today — same chain object, same behavior, same tests passing.
2. **Two retrieval scopes**: a global library (admin-seeded) and per-conversation uploads (user-supplied), merged at query time.
3. **Pluggable internals**: vector store, embedding model, and splitter each have an interface; FAISS + sentence-transformers + RecursiveCharacterTextSplitter ship as defaults.
4. **Visible evidence**: the user can see which chunks were retrieved (collapsible sources panel) to make the comparison UI educational.
5. **Conversation-scoped uploads**: a conversation only retrieves its own uploads; cross-conversation leakage is impossible.
6. **Conversation deletion cleans up uploads**: when a conversation is deleted, its uploaded files and index chunks are removed.
7. **Easy to extend**: adding a third retrieval scope later is a small additive change.

### Non-Goals (v1)

- No multi-process locking. Single-process FastAPI; same assumption as `conversations.json`.
- No cross-encoder re-ranking. Pure vector retrieval.
- No chunk deduplication across reindex.
- No per-conversation FAISS indexes — single `uploads_index` with metadata filtering.
- No admin UI for library management. Library is seeded from `storage/library/` on disk; reindex is an HTTP endpoint.
- No auth on any rag endpoint. Local exploration project.
- No document-level permissions beyond `conversation_id`.
- No background tasks / queues — ingestion is in-request.
- No Playwright/JS test framework for frontend. Manual UI verification.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                   │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │  Chat Panel (vanilla)│    │ Chat Panel (RAG)     │  shared input box  │
│  │  conv: <uuid>-0      │    │  conv: <uuid>-1      │                   │
│  └──────────┬───────────┘    └──────────┬───────────┘                   │
└─────────────┼───────────────────────────┼──────────────────────────────┘
              │ POST /api/chat/stream       │ POST /api/chat/stream
              │ retrieval=null              │ retrieval={library:true,uploads:true,top_k:4}
              ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          backend/chat                                   │
│                                                                         │
│   routes.py  ─►  ChatService  ─►  chain.astream(messages)  ─►  job.chunks│
│                       │                                                 │
│                       │ when retrieval is set and RAG is enabled:        │
│                       ▼                                                 │
│                  ┌─────────────────────────────────────┐                 │
│                  │ 1. retriever.with_conversation(id) │                 │
│                  │ 2. hits = retriever.invoke(msg)    │                 │
│                  │ 3. job.append_chunk("sources", …)  │                 │
│                  │ 4. messages = augment(messages, …) │                 │
│                  └─────────────────────────────────────┘                 │
│                       │                                                 │
│                       ▼                                                 │
│                  chain.astream(messages)   ← today's chain, unchanged   │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          backend/rag  (NEW)                             │
│                                                                         │
│   RagService                                                            │
│      ├─► ingest_file(conversation_id, path)   ─► upload path            │
│      ├─► purge_uploads(conversation_id)         ─► delete path          │
│      ├─► reindex_library()                      ─► library path         │
│      └─► get_retriever() → ScopedRetriever      ─► chat path            │
│                       │                                                 │
│                       ▼                                                 │
│             ScopedRetriever (merge + explicit metadata filter)          │
│                       │                                                 │
│            ┌──────────┴──────────┐                                      │
│            ▼                     ▼                                      │
│     FAISS library_index     FAISS uploads_index                          │
│     (storage/rag/library/)  (storage/rag/uploads/)                      │
│                                                                         │
│   Embeddings: sentence-transformers  OR  MiniMax endpoint                │
│   Splitter: RecursiveCharacterTextSplitter (configurable)               │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              storage/  (existing, untouched)
                          conversations.json + uploads/<conv_id>/
```

**Key properties:**

- **Plugin-off = literally identical chain.** `RAG_ENABLED=false` → `create_chain()` returns today's `RunnableLambda(convert_messages) | llm`. The chain source file is unchanged.
- **Chat depends on RagService only for the retriever interface.** `BaseRetriever` comes from `langchain_core.retrievers`, not from `rag/`. So `chat/chain.py` does not import anything from `rag/`.
- **Two write paths, one read path.** Library admin updates → `RagService.reindex_library()`. User uploads → `RagService.ingest_file()`. Both mutate indexes in place. Chat → `RagService.get_retriever()` returns the same singleton `ScopedRetriever` configured per-request with the conversation id.

---

## 3. Data Flow — Four Paths

### 3.1 Library Seeding

**Trigger**: App startup OR admin `POST /api/rag/library/reindex`.

```
1. Walk storage/library/ recursively (extensions: .md, .txt, .pdf, .html)
2. For each file:
     text = read_file(path)              # PDF via pypdf; others UTF-8
     chunks = splitter.split_text(text)
     for chunk in chunks:
         chunk.metadata = {
             "source":   "library",
             "filename": relative_path,
             "chunk_id": sha256(text)[:16],
         }
3. library_index = FAISS.from_documents(all_chunks, embeddings)
4. library_index.save_local("storage/rag/library_index/")
```

**Properties**:
- Idempotent: re-running with unchanged files produces the same chunks (deterministic sha256).
- Failure isolation: unreadable files are logged and skipped; doesn't abort the run.

### 3.2 User Upload

**Trigger**: User drops a file into the RAG panel → `POST /api/rag/upload` with `conversation_id` and the file.

```
1. Save file to  storage/uploads/<conversation_id>/<filename>
2. text = read_file(path)
3. chunks = splitter.split_text(text)
4. for chunk in chunks:
       chunk.metadata = {
           "source":         "upload",
           "conversation_id": conversation_id,
           "filename":        filename,
           "chunk_id":        sha256(text)[:16],
       }
5. uploads_index.add_documents(chunks)
6. uploads_index.save_local("storage/rag/uploads_index/")
```

**Properties**:
- File saved before index mutation: a crash mid-embedding leaves a recoverable artifact.
- Single-process concurrency assumption (matches `conversations.json`).
- Failure: embedding API failure → file kept on disk, index untouched, return 500. User can retry.

### 3.3 Chat (the streaming path)

**Trigger**: `POST /api/chat/stream` with `retrieval: RetrievalConfig`.

```
1. routes.stream_chat() appends user message to storage, kicks off background task
2. ChatService.generate_background(message, conversation_id, retrieval):
     a. messages = file_storage.get_conversation(conversation_id)["messages"]
     b. IF retrieval is not None AND self.rag_retriever is not None:
          scoped = self.rag_retriever.with_conversation(conversation_id)
          hits = scoped.invoke(message)
          job.append_chunk("sources", json.dumps({
              "sources": [
                  {"filename": h.metadata.get("filename"),
                   "excerpt": h.page_content[:300],
                   "scope":   h.metadata.get("source")}    # "library" | "upload"
                  for h in hits
              ]
          }))
          context_str = "\n\n".join(
              f"[{h.metadata.get('filename')}]: {h.page_content}" for h in hits
          )
          messages = messages[:-1] + [
              {"role": "system", "content": f"Use this retrieved context:\n{context_str}"},
              messages[-1],
          ]
     c. async for chunk in self.chain.astream(messages):
          # existing logic — push tokens to job.chunks
3. SSE replays job.chunks → client sees: sources → token → token → ... → done
```

**Properties**:
- Retrieval happens once per turn, before the first token.
- Retrieval query is the latest user message only.
- When retrieval is null, the service path is byte-identical to today.

### 3.4 Conversation Deletion

**Trigger**: existing `delete_conversation()` call (UI button or storage API).

```
1. existing: drop from conversations.json
2. NEW: rag_service.purge_uploads(conversation_id)
     a. shutil.rmtree(storage/uploads/<conversation_id>/)
     b. surviving = {cid: doc for cid, doc in uploads_index.docstore._dict.items()
                     if doc.metadata.get("conversation_id") != conversation_id}
     c. new_uploads_index = FAISS.from_documents(list(surviving.values()), embeddings)
     d. self.uploads_index = new_uploads_index   ← atomic Python rebind
     e. self.uploads_index.save_local("storage/rag/uploads_index/")
```

**Properties**:
- JSON delete happens first; if index rebuild fails, the conversation is still gone from the user's view (orphan chunks are invisible at query time since no conversation_id matches).
- Library index untouched by conversation deletion — correct by design.
- Rebuild cost is dominated by walking the docstore dict (~50 ms for 1000 chunks). No re-embedding needed.

**Hook wiring** (chosen over monkey-patching for explicitness):

```python
# backend/storage/file_storage.py — extended signature
def delete_conversation(
    conversation_id: str,
    on_delete: Callable[[str], None] | None = None,
) -> bool:
    # ... existing logic ...
    if on_delete:
        try:
            on_delete(conversation_id)
        except Exception:
            logger.exception("on_delete hook failed for %s", conversation_id)
    return True

# backend/main.py — wired at startup
delete_with_purge = partial(file_storage.delete_conversation,
                            on_delete=rag_service.purge_uploads)
```

---

## 4. Module Layout

### 4.1 New files

```
backend/rag/
├── __init__.py          # public surface: RagService, ScopedRetriever
├── config.py            # RAG settings (RAG_ENABLED, EMBEDDING_BACKEND, chunk size, paths)
├── service.py           # RagService facade: ingest, purge, reindex, get_retriever
├── retriever.py         # ScopedRetriever (merge + explicit metadata filter)
├── vector_store.py      # FAISS load/save/rebuild helpers
├── embeddings.py        # Embeddings factory: sentence-transformers OR MiniMax
├── splitter.py          # Text splitter factory + chunk metadata assembly
└── routes.py            # FastAPI routes: upload, library reindex, rag stats
```

**Per-file responsibilities:**

- **`config.py`**: Reads `RAG_*` env vars into a Pydantic settings object. Independent of the global `config.py` so RAG is fully opt-in.
- **`service.py`**: The only thing other domains import. Holds the long-lived `uploads_index`, `library_index`, `embeddings`, and `splitter` state.
- **`retriever.py`**: `ScopedRetriever` (a `BaseRetriever` subclass). Isolated so the merge/filter logic is unit-testable in isolation.
- **`vector_store.py`**: Wraps FAISS operations (`load_local`, `save_local`, `add_documents`, `from_documents`, `rebuild`). Swapping for Chroma means rewriting this file; nothing else moves.
- **`embeddings.py`**: One factory: `make_embeddings(backend: str) -> Embeddings`. Default returns `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`. Alternative returns a custom wrapper calling MiniMax's `/embeddings` endpoint (subject to verifying the endpoint is exposed).
- **`splitter.py`**: `make_splitter(chunk_size=800, chunk_overlap=200)` returns a `RecursiveCharacterTextSplitter`.
- **`routes.py`**: Three endpoints (see Section 5).

### 4.2 Modified files

| File | Change |
|---|---|
| [backend/chat/chain.py](backend/chat/chain.py) | **No changes.** `create_chain()` returns today's chain. |
| [backend/chat/service.py](backend/chat/service.py) | `__init__` accepts `rag_retriever`. `generate_background` adds the retrieval+augmentation block when `retrieval` is set. |
| [backend/chat/routes.py](backend/chat/routes.py) | `ChatRequest` gains optional `retrieval: RetrievalConfig \| None`. `stream_chat` passes `retrieval` to `generate_background`. |
| [backend/storage/file_storage.py](backend/storage/file_storage.py) | `delete_conversation` accepts `on_delete: Callable \| None`. |
| [backend/main.py](backend/main.py) | Lifespan: if `RAG_ENABLED`, build `RagService`, wire `delete_conversation` with `purge_uploads`, mount rag routes. |
| [requirements.txt](requirements.txt) | Add `faiss-cpu`, `sentence-transformers`, `pypdf`. |
| [frontend/index.html](frontend/index.html) | Two-panel comparison UI with shared input, upload button, sources toggle. |

### 4.3 Wiring at startup

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.RAG_ENABLED:
        rag = RagService.from_settings()
        app.state.rag = rag
        # Patch delete_conversation to also purge uploads
        original_delete = file_storage.delete_conversation
        file_storage.delete_conversation = partial(
            original_delete, on_delete=rag.purge_uploads
        )
    yield
    if hasattr(app.state, "rag"):
        app.state.rag.persist_all()
```

---

## 5. API Surface

### 5.1 ChatRequest changes

```python
class RetrievalConfig(BaseModel):
    library: bool = True         # include global library chunks
    uploads: bool = True         # include this conversation's uploaded chunks
    top_k: int = 4               # chunks per scope

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    retrieval: RetrievalConfig | None = None   # NEW, optional, default None
```

Resolution rules:

| `retrieval` | `RAG_ENABLED` | Chain path |
|---|---|---|
| `None` | true or false | vanilla (no retrieval) |
| set | false | vanilla (server logs debug, ignores field) |
| set | true | RAG-enabled path with the requested scopes |

### 5.2 SSE event extension — `sources`

The existing stream emits `token`, `thinking`, and `done` events. We add one new event type, fired **before** the first token:

```
event: sources
data: {"sources": [
  {"filename": "guide.pdf", "excerpt": "RAG stands for...", "scope": "library"},
  {"filename": "notes.md",  "excerpt": "Yesterday we...",   "scope": "upload"}
]}

event: token
data: {"text": "RAG"}

event: token
data: {"text": " stands"}
...

event: done
```

- Vanilla chain never emits `sources` → FE renders nothing.
- RAG chain emits `sources` once, then tokens.
- FE toggle decides whether to render the sources block.

### 5.3 New endpoints (`backend/rag/routes.py`)

```
POST /api/rag/upload
  Content-Type: multipart/form-data
  Body: conversation_id (form field) + file (form field)
  → 200 {filename, chunks_added, chunk_ids: [...]}
  → 400 if file unreadable / too large
  → 503 if RAG disabled
  → 500 if embedding/indexing fails (file kept on disk for retry)

POST /api/rag/library/reindex
  → 200 {files_processed, chunks_added, errors: [...]}
  → 503 if RAG disabled
  (No auth in v1. OpenAPI docstring marks this as "internal endpoint".)

GET /api/rag/stats
  → 200 {
      enabled: bool,
      embedding_backend: str,
      library_chunks: int,
      uploads_chunks: int,
      uploads_conversations: list[str],
    }
  → 503 if RAG disabled
```

All three are mounted under `/api/rag` and registered in `main.py` only when `RAG_ENABLED=true`.

---

## 6. Frontend — Comparison UI

```
┌────────────────────────────────────────────────────────────┐
│  Compare Chat                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Ask anything:  [___________________________________] │ │
│  │                                          [Send →]    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌────────────────────────┐  ┌────────────────────────┐  │
│  │ Vanilla                │  │ RAG                    │  │
│  │                        │  │  [☑] Show sources      │  │
│  │  user: hello           │  │  [📎 Upload]           │  │
│  │  asst: hi there        │  │  user: hello           │  │
│  │                        │  │  sources: 3 chunks     │  │
│  │                        │  │  asst: hi there...     │  │
│  └────────────────────────┘  └────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**Conversation ID convention**:
- Both panels share a random base UUID generated once per page load.
- Vanilla panel uses `<random>-0`.
- RAG panel uses `<random>-1`.
- They are easy to identify as a pair in the conversation list and easy to delete together.

**Send behavior**:
- Shared input + Send button.
- Two concurrent POSTs to `/api/chat/stream` — one per panel — with the same `message` text but different `conversation_id` and different `retrieval`:
  - Vanilla panel: `conversation_id: <uuid>-0, retrieval: null`
  - RAG panel: `conversation_id: <uuid>-1, retrieval: {library: true, uploads: true, top_k: 4}`
- Each panel maintains its own scroll, history, and SSE stream.

**Upload**:
- File picker in the RAG panel → `POST /api/rag/upload` with the RAG panel's `conversation_id`.
- After 200, optionally refresh `GET /api/rag/stats` to update the "X docs indexed" indicator.

**Sources toggle**:
- Per-panel checkbox; default ON.
- When OFF, the panel renders no sources block even if the event arrives.
- State is local to the panel — server doesn't know or care.

---

## 7. Key Components

### 7.1 ScopedRetriever

```python
class ScopedRetriever(BaseRetriever):
    """Wraps multiple retrievers, filtering specific ones by metadata."""
    retrievers: list[tuple[BaseRetriever, bool]]   # (retriever, should_filter_by_conv)
    conversation_id: str
    top_k: int = 4

    def _get_relevant_documents(self, query, *, run_manager):
        hits = []
        for r, should_filter in self.retrievers:
            r_hits = r.invoke(query)
            if should_filter:
                r_hits = [d for d in r_hits
                          if d.metadata.get("conversation_id") == self.conversation_id]
            hits.extend(r_hits)
        return hits[:self.top_k]

    def with_conversation(self, conversation_id: str) -> "ScopedRetriever":
        return ScopedRetriever(
            retrievers=self.retrievers,
            conversation_id=conversation_id,
            top_k=self.top_k,
        )
```

The bool flag makes filter application explicit. Library retrievers get `False`; upload retrievers get `True`. No magic.

### 7.2 RagService

```python
class RagService:
    def __init__(self, embeddings, splitter, library_dir, uploads_dir, rag_dir):
        self.embeddings = embeddings
        self.splitter = splitter
        self.library_dir = library_dir
        self.uploads_dir = uploads_dir
        self.rag_dir = rag_dir
        self.library_index = self._load_or_init_library_index()
        self.uploads_index = self._load_or_init_uploads_index()

    @classmethod
    def from_settings(cls) -> "RagService": ...

    def ingest_file(self, conversation_id: str, file_path: Path) -> list[str]: ...
    def purge_uploads(self, conversation_id: str) -> None: ...
    def reindex_library(self) -> dict: ...
    def get_retriever(self) -> ScopedRetriever:
        # Returns a template ScopedRetriever. Callers must invoke .with_conversation(id)
        # before .invoke(query). The conversation_id field is required by the dataclass
        # shape but is always overwritten by with_conversation at the call site.
        return ScopedRetriever(
            retrievers=[
                (self.library_index.as_retriever(search_kwargs={"k": self.top_k}), False),
                (self.uploads_index.as_retriever(search_kwargs={"k": self.top_k}), True),
            ],
            conversation_id="",
            top_k=self.top_k,
        )
    def persist_all(self) -> None: ...
    def stats(self) -> dict: ...
```

The retriever factory is called once at startup; per-request scoping happens via `.with_conversation(id)` which returns a copy with the conversation field set.

---

## 8. Error Handling

| Failure | Behavior |
|---|---|
| `RAG_ENABLED=false`, client sends `retrieval` | Server ignores, runs vanilla. Logs debug line. |
| File too large / unreadable on upload | 400, file not saved. |
| Embedding API down (MiniMax) | 500, file kept on disk, index untouched. User retries. |
| FAISS save fails after `add_documents` succeeded | 500, log critical. In-memory state consistent but disk stale until next reindex. |
| Retrieval fails mid-chat | Stream error chunk; LLM call proceeds with the original (un-augmented) messages. |
| `delete_conversation` succeeds but `purge_uploads` fails | JSON state consistent; orphan chunks invisible at query time. Logged warning. |
| Library reindex encounters unreadable file | File skipped, error added to response `errors` list. Run continues. |

---

## 9. Testing Strategy

### 9.1 Layers

| Layer | Files | Speed |
|---|---|---|
| **Unit** | `test_retriever.py`, `test_splitter.py`, `test_embeddings_factory.py` | <10 ms each |
| **Service** | `test_service.py` — real FAISS in `tmp_path` with `FakeEmbeddings` | <500 ms each |
| **Chain integration** | `test_chain_integration.py` — ChatService with mocked retriever | <1 s each |
| **Route** | `test_routes.py` — `TestClient` against FastAPI app with test `RagService` | <1 s each |

### 9.2 Test cases (must-have)

```python
# ScopedRetriever
def test_merges_two_scopes()             # lib + upl → both present in result
def test_filters_uploads_by_conv_id()    # other conversation's chunks excluded
def test_library_not_filtered()          # library chunks pass through unfiltered
def test_top_k_caps_merged_results()     # N hits → top_k returned

# RagService
def test_ingest_file_adds_chunks_with_metadata()
def test_purge_uploads_removes_only_target_conversation()
def test_reindex_library_is_idempotent()

# ChatService with RAG
def test_rag_path_pushes_sources_before_tokens()
def test_no_rag_path_is_byte_identical_to_today()   # ← THE plugin-off guarantee
def test_retrieval_query_uses_latest_user_message()
def test_augmentation_inserts_system_message_before_last_user()

# Routes
def test_upload_endpoint_indexes_file()
def test_upload_endpoint_503_when_rag_disabled()
def test_stats_endpoint_returns_counts()
def test_library_reindex_returns_processed_count()
```

The test `test_no_rag_path_is_byte_identical_to_today` is the load-bearing assertion for the plugin property: it verifies that calling `generate_background` without `retrieval` produces exactly the same output (job.chunks) as the pre-RAG version.

### 9.3 What we are NOT testing in v1

- Multi-process concurrency.
- Real MiniMax embedding API calls.
- Index corruption recovery beyond logging.
- FAISS index size > 100k chunks (the sidecar pattern is the escape hatch).
- PDF parsing edge cases (basic happy path only).
- Frontend (manual verification per Section 6).

---

## 10. Configuration

New env vars (all optional; module is inert when not set):

```env
RAG_ENABLED=false                       # master switch; default false
EMBEDDING_BACKEND=sentence-transformers # or "minimax"
RAG_LIBRARY_DIR=storage/library          # admin-seeded corpus
RAG_UPLOADS_DIR=storage/uploads          # per-conversation uploads
RAG_INDEX_DIR=storage/rag                # FAISS index files live here
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1    # existing, reused for MiniMax embeddings
ANTHROPIC_API_KEY=...                              # existing, reused for MiniMax embeddings
```

**Notes:**
- `RAG_ENABLED=false` is the default — this means the system runs as today, with zero rag-related imports executed.
- `EMBEDDING_BACKEND=minimax` requires verification that the MiniMax endpoint exposes an `/embeddings` route. If it doesn't, only `sentence-transformers` is available.
- New pip dependencies: `faiss-cpu`, `sentence-transformers`, `pypdf`.

---

## 11. Future Work (post-v1)

Documented but explicitly out of scope:

1. **Tombstone-based delete** — mark chunks deleted in a set, exclude at query time, periodic compaction. Replaces full-rebuild on conversation delete when indexes grow large.
2. **SQLite/DuckDB docstore** — replaces the FAISS pickle sidecar when chunk counts exceed ~50k.
3. **Cross-encoder re-ranking** — improves retrieval precision at the cost of an extra model.
4. **Multi-process locking** — `fcntl`/`msvcrt` file lock for the FAISS index, matching what would be needed for `conversations.json` too.
5. **Background ingestion** — long uploads don't block the request; client polls for completion.
6. **Additional scopes** — "web cache", "agent memory", "tool results". Pattern: add a new FAISS index + a new entry in `ScopedRetriever.retrievers` + a new bool flag in `RetrievalConfig`.
7. **Per-conversation FAISS index** — if "uploads" grows large enough that one index becomes a hotspot, switch to one index per conversation.
8. **Auth on `/api/rag/library/reindex`** — when this app leaves the local-only stage.
9. **Chunk deduplication** — manifest-based skip-if-unchanged during library reindex.

---

## 12. Open Decisions Resolved

These were decided during brainstorming and are recorded here so the spec is self-contained:

| Decision | Choice |
|---|---|
| Document corpus | Both file upload (per-conversation) AND global library |
| Vector store | Pluggable interface + FAISS default |
| Embeddings | Pluggable; sentence-transformers default + MiniMax alternative |
| Activation model | Env flag + per-request `retrieval` config |
| Compare UI | Synced input, independent histories |
| Sources visibility | Per-panel toggle, default ON |
| Scope merging | Two separate indexes + `ScopedRetriever` with explicit filter targeting |
| Delete strategy | Rebuild from docstore (fast for v1); tombstone escape hatch documented |
| Chain integration | Service-layer pre-processing; chain itself untouched |
| Hook for delete | Callback parameter on `delete_conversation` (no monkey-patching) |
| Conversation IDs | `<random-uuid>-0` (vanilla) and `<random-uuid>-1` (RAG) |