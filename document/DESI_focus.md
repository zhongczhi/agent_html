# Chatbot Project — Iteration 7 Design (RAG Module)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> The architecture decisions and module-level design for the RAG module. See [SPEC_focus.md](SPEC_focus.md) for requirements and [docs/superpowers/specs/2026-06-26-rag-module-design.md](../docs/superpowers/specs/2026-06-26-rag-module-design.md) for the full brainstorming artifact.

## 1. Architecture Decisions

### 1.1 Plugin Activation via Env Flag

**Choice:** `RAG_ENABLED` environment variable (default `false`) controls whether the RAG module is constructed at all.

**Rationale:**
- Hard-off behavior is provable: when `false`, no `rag/` imports execute in the request hot path and `delete_conversation` has no rag callback wired in.
- Avoids runtime "is RAG enabled?" branches scattered through the codebase — instead the code is constructed conditionally at startup.
- A test (`test_no_rag_path_is_byte_identical_to_today`) asserts that calling `ChatService.generate_background` without `retrieval` produces exactly the same `job.chunks` as the iteration-6 version.

**Trade-off:** Per-request toggling of "is RAG even installed" is impossible; once enabled, it's enabled until restart. For a v1 exploration project this is acceptable.

### 1.2 Retrieval-Augmented Chat Without Chain Modification

**Choice:** Retrieval happens in `ChatService.generate_background` as a pre-processing step before the LLM call. The chain implementation (`backend/chat/chain.py`) is **not modified**.

**Rationale:**
- The chain becomes "pure LLM call" — `RunnableLambda(convert_messages) | llm`. Its job is well-defined and stable.
- Retrieval results are pushed to `job.chunks` via `job.append_chunk("sources", ...)` before the LLM call, so SSE replays them in order: sources → token → token → ... → done.
- The plugin-off property becomes literally true: the chain source file is byte-identical to iteration 6.
- If retrieval happens *inside* the LCEL chain, the retrieved `context` field is consumed by `augment_messages` and discarded. The route handler (which reads from `job.chunks`, not from the chain's intermediate state) never sees it. This was the load-bearing finding during the brainstorming self-review.

**Trade-off:** Retrieval runs synchronously before the LLM call, adding ~50-200 ms to the first-token latency. Acceptable for v1; could move to a streaming side-channel later if needed.

### 1.3 Two-Scope Index Architecture

**Choice:** Two separate FAISS indexes — `library_index/` and `uploads_index/` — merged at query time via a custom `ScopedRetriever`.

**Rationale:**
- Library changes are admin operations (rare, read-mostly); uploads change every chat session (mutable, per-conversation). Mixing them in one index forces admin rebuilds to walk the user namespace and makes conversation deletion surgically remove chunks from a shared file.
- Two indexes are easy to reason about and easy to operate on independently.
- Library index can be reloaded read-only at startup; uploads index supports `add_documents` mid-session.
- Adding a third scope later ("web cache", "agent memory") is one more FAISS index + one more entry in `ScopedRetriever.retrievers`. No chain changes, no chat-service changes.

**Trade-off:** ~30 lines of glue code for the merge-and-filter step. Acceptable for the clarity gained.

### 1.4 Conversation-Scoped Filter with Explicit Targeting

**Choice:** `ScopedRetriever` takes `list[tuple[BaseRetriever, bool]]` where the bool flags whether to apply the `conversation_id` filter. Library retrievers get `False`; upload retrievers get `True`.

**Rationale:**
- Explicit > implicit. The earlier design filtered all hits and relied on the magic of "library hits don't have `conversation_id` so they survive". Fragile and hard to reason about.
- Each tuple's bool is documented at the call site (in `RagService.make_scoped_retriever`), making the security/correctness property reviewable: "library chunks are never filtered out, upload chunks are always filtered to the current conversation".

**Trade-off:** Slightly more verbose than a single `filter_key/filter_value` parameter pair.

### 1.5 Pluggable Embeddings: Sentence-Transformers Default, MiniMax Alternative

**Choice:** `EMBEDDING_BACKEND` env var (`sentence-transformers` or `minimax`) selects the embeddings implementation. Default is `sentence-transformers` with model `all-MiniLM-L6-v2`.

**Rationale:**
- Local sentence-transformers works offline, has no per-call API cost, and is fast (~5 ms per query). Adds ~80 MB pip dep but eliminates a network round-trip per chunk during ingestion.
- MiniMax alternative (subject to verifying the endpoint exposes `/embeddings`) allows using the same vendor as the chat model. Useful for A/B comparing embedding quality.
- The factory pattern in `backend/rag/embeddings.py` returns a `langchain_core.embeddings.Embeddings` instance — both backends satisfy the same interface, so the rest of the system doesn't know which is active.

**Trade-off:** Two backends to test. Mitigated by injecting a `FakeEmbeddings` in tests.

### 1.6 Pluggable Vector Store: FAISS Default

**Choice:** `backend/rag/vector_store.py` wraps LangChain's `FAISS` class. Swapping to Chroma means rewriting this file; nothing else moves.

**Rationale:**
- FAISS via `langchain_community.vectorstores` provides `add_documents`, `save_local`, `load_local`, `as_retriever` out of the box.
- On-disk persistence matches the project's "no external services" theme (same as the JSON conversation storage).
- No FAISS server process; index lives in files.
- LangChain's `FAISS.save_local` writes a sidecar `index.pkl` containing the docstore — this is what enables fast rebuild-on-delete without re-embedding.

**Trade-off:** Pickle sidecar limits index size to ~50k chunks before memory pressure becomes a concern. Documented as future work: replace with a SQLite/DuckDB-backed docstore when needed.

### 1.7 Deletion via Docstore Rebuild

**Choice:** When a conversation is deleted, the uploads index is rebuilt from the surviving docstore entries (no re-embedding).

**Rationale:**
- LangChain's `FAISS.save_local` writes vectors + docstore together. Rebuild reads the docstore, filters by `conversation_id`, creates a new `FAISS.from_documents(...)`, and atomically rebinds `self.uploads_index = new_index`.
- Rebuild cost is dominated by walking the docstore dict (~50 ms for 1000 chunks). No API calls, no file re-reading.
- For 10k chunks: ~400 ms. For 100k chunks: ~4 s. Acceptable for v1.
- Escape hatch documented: tombstone set + periodic compaction when rebuild cost becomes a bottleneck.

**Trade-off:** Synchronous in the request handler — the user waits for the rebuild before the delete returns. Mitigated by the fast path (sub-100 ms for normal-sized indexes).

### 1.8 Hook Wiring via Callback Parameter

**Choice:** `file_storage.delete_conversation` accepts an optional `on_delete: Callable[[str], None]` parameter. `main.py` wires `rag_service.purge_uploads` as the callback at startup.

**Rationale:**
- Explicit over implicit. The storage layer declares its extensibility via the `on_delete` parameter on `delete_conversation`; the application layer wires the actual callback at startup (via `functools.partial`), so the storage module itself is never modified.
- Default `None` keeps old callers unaffected.
- Exceptions in the callback are caught and logged but don't fail the delete — the JSON state remains consistent.
- Testable: tests pass a fake callback to verify the wiring without needing a real `RagService`.

**Trade-off:** Tiny signature change to an existing function. Internal API only.

### 1.9 Sources Event via Job.Append Chunk

**Choice:** New SSE event type `sources` fired once before the first token, populated by `job.append_chunk("sources", json.dumps({...}))` in `ChatService.generate_background`.

**Rationale:**
- Reuses the existing background-task + job.chunks pattern. No new streaming mechanism.
- Order is preserved by `job.append_chunk`'s append-only semantics: SSE replays sources before tokens.
- Vanilla chain never emits sources — natural divergence between modes.
- Frontend toggle is local state; server doesn't know or care.

**Trade-off:** Sources are formatted as a single JSON blob in a chunk string. Alternative would be a structured SSE event with `event: sources\ndata: {...}` headers; chosen approach uses the existing chunk format for simplicity.

### 1.10 Size-Based Upload Routing

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

### 1.11 Frontend Side-by-Side Comparison UI

**Choice:** The chat UI is a side-by-side comparison layout: one shared input at the bottom, two message columns above (Vanilla on the left, RAG on the right). The two columns share a base UUID; vanilla maps to `<base>-0`, RAG maps to `<base>-1`. One Send click fires two parallel `/api/chat/stream` POSTs — vanilla with `retrieval: null`, RAG with `retrieval: {library, uploads, top_k}` — and each column streams its SSE response into its own column. The two columns are otherwise independent: separate histories, separate abort controllers, separate cache state.

**Rationale:**
- Direct visual comparison on the same question is the core evaluation use case. A side-by-side layout answers "did RAG actually improve this answer?" without making the user retype and re-send.
- The backend already supports per-request `retrieval` toggling (`RetrievalConfig` on `ChatRequest`). The frontend fan-out is a thin client-side loop: two `fetch` calls with the same `message` and different `retrieval` / `conversation_id`. No backend changes required.
- Conversation IDs `<base>-0` and `<base>-1` keep the existing storage model: the two columns are normal conversations in `conversations.json`, so the sidebar, CRUD, and deletion work without modification.
- When `RAG_ENABLED=false`, the RAG column is hidden, the vanilla column takes full width, and only one POST fires per Send. The layout degrades cleanly to iteration-6 behavior.

**Trade-off:** The frontend is more complex than a single-pane UI: two parallel SSE readers, two `AbortController`s, per-column resume-from-cache, per-column badge/state. To keep this manageable, the existing `processStreamResponse` is parameterized by a per-column context object (assistant message element, cache keys, abort controller) and invoked twice in parallel from `sendMessage`.

---

## 2. Module Layout

### 2.1 New Files

```
backend/rag/
├── __init__.py          # public surface: RagService, ScopedRetriever
├── config.py            # RAG-specific settings (RAG_ENABLED, EMBEDDING_BACKEND, paths, chunk params)
├── service.py           # RagService facade — ingest, purge, reindex, make_scoped_retriever, stats
├── retriever.py         # ScopedRetriever (merge + explicit metadata filter)
├── vector_store.py      # FAISS load/save/rebuild helpers
├── embeddings.py        # Embeddings factory: sentence-transformers OR MiniMax
├── splitter.py          # Text splitter factory + chunk metadata assembly
└── routes.py            # FastAPI routes: upload, library reindex, rag stats
```

### 2.2 Per-File Responsibilities

**`backend/rag/config.py`** — Pydantic settings reading `RAG_*` env vars. Independent of `backend/config.py` so the global config has no RAG awareness when `RAG_ENABLED=false`.

**`backend/rag/service.py`** — The only module other domains import. Holds the long-lived `library_index`, `uploads_index`, `embeddings`, `splitter`, and the per-conversation uploads directory path.

**`backend/rag/retriever.py`** — `ScopedRetriever`, a `BaseRetriever` subclass. Has method `_get_relevant_documents` (the actual retrieval). Per-scope cap is applied by the underlying retrievers via `search_kwargs={"k": k}`; ScopedRetriever does NOT cap the merged result.

**`backend/rag/vector_store.py`** — `load_or_init(path, embeddings)` returns a FAISS instance. `save(index, path)` persists it. `rebuild_filtered(index, embeddings, keep_predicate)` rebuilds from surviving entries — must handle the empty case by creating a single-placeholder index so the next `add_documents` call works. Swapping for Chroma means rewriting this file.

**`backend/rag/embeddings.py`** — `make_embeddings(backend: str) -> Embeddings`. Default returns `HuggingFaceEmbeddings(model_name=settings.sentence_transformers_model)`. Alternative returns a custom wrapper calling MiniMax's `/embeddings` endpoint.

**`backend/rag/splitter.py`** — `make_splitter(chunk_size, chunk_overlap) -> RecursiveCharacterTextSplitter`. Single factory function.

**`backend/rag/routes.py`** — Three endpoints, mounted under `/api/rag`. Each checks `RAG_ENABLED` and returns 503 if disabled.

### 2.3 Modified Files

| File | Change |
|---|---|
| `backend/chat/chain.py` | **No changes.** `create_chain()` returns today's chain, unmodified. |
| `backend/chat/service.py` | `ChatService.__init__` accepts optional `rag_service: RagService \| None` and a `pending_inline_files: dict[str, list[dict]]` keyed by conversation_id. `generate_background(message, conversation_id, retrieval=None, uploaded_files=None)` does two pre-processing blocks: (1) when `uploaded_files` is non-empty, inject each file's content as a system message and merge the list into `pending_inline_files` for that conversation (subsequent turns in the same conversation re-use it without re-sending); (2) when `retrieval is not None and self.rag_service is not None and not uploaded_files`, do the existing FAISS retrieval path. The two paths are mutually exclusive per turn: inline files take precedence over retrieval. When the conversation is deleted (via the `on_delete` callback), the entry for that conversation is removed from `pending_inline_files`. |
| `backend/chat/routes.py` | `ChatRequest` gains optional `retrieval: RetrievalConfig \| None` and optional `uploaded_files: list[UploadedFile] \| None`. Each `UploadedFile` has `filename: str` and `content: str`. `stream_chat` passes both to `generate_background`. |
| `backend/rag/config.py` | `RagSettings` gains `inline_context_threshold_bytes: int = 8192` (env `RAG_INLINE_CONTEXT_THRESHOLD_BYTES`). |
| `backend/rag/service.py` | `stats()` includes `inline_context_threshold_bytes` in its payload. `purge_uploads(conversation_id)` clears only the FAISS uploads for that conversation; the pending inline files are cleared separately by `chat_service.clear_pending_inline_files`. The two cleanup calls are chained together at the application layer by `main.py`'s startup wiring (see below). |
| `backend/rag/routes.py` | `POST /api/rag/upload` checks raw upload size against `RagSettings.inline_context_threshold_bytes`. If ≤ threshold: read the file content, return `{filename, content, mode: "inline", bytes: N}`. If > threshold: save to disk + FAISS ingest as before, return `{filename, chunks_added, chunk_ids, mode: "indexed", bytes: N}`. |
| `backend/storage/file_storage.py` | `delete_conversation` signature gains `on_delete: Callable[[str], None] \| None = None`. Calls it after the JSON delete succeeds, with exception swallowing. |
| `backend/main.py` | Lifespan: if `RAG_ENABLED`, build `RagService.from_settings()`, mount rag routes. Wire **two** callbacks on `delete_conversation`: `rag.purge_uploads` (FAISS cleanup) AND `chat_service.clear_pending_inline_files` (in-memory list cleanup). Pass both via a single `partial` chain. |
| `backend/chat/routes.py` (chat service singleton) | `set_rag_service` is updated to also wire `chat_service.clear_pending_inline_files` into the same callback. The `get_chat_service` factory passes `clear_pending_inline_files` as a method reference. |
| `requirements.txt` | Add `faiss-cpu`, `sentence-transformers`, `pypdf`. |
| `frontend/index.html` | Replace the single `chat-messages` pane with a two-column layout: a "Vanilla" column and a "RAG" column side-by-side, each with its own header (channel title only) and message list, plus a single shared input box, **Upload button** (both columns), and Send button below. The RAG column is hidden when `RAG_ENABLED=false`; in that case the Upload button is hidden too (only the Send button remains). |
| `frontend/static/app.js` | Maintain a `baseConversationId` in `cache.js`; the vanilla column maps to `<base>-0`, the RAG column maps to `<base>-1`. `sendMessage` fans out to two parallel `fetch('/api/chat/stream', ...)` calls (one per column) and pipes each SSE stream into its column. `processStreamResponse` is parameterized by a per-column context (target message element, cache key, abort controller) and is invoked twice in parallel. The `sources` chunk type renders a Sources block in **both** columns when the column's request sent `uploaded_files` (the inline path also emits a sources event so the user sees which file was used). Upload button: each column has its own `pendingInlineFiles` array (in-memory). Click → `POST /api/rag/upload` with multipart; server returns `{mode: "inline" \| "indexed"}`. If `inline`: append `{filename, content}` to the column's `pendingInlineFiles`; the next `sendToColumn` includes them in the `uploaded_files` field of the stream request. If `indexed`: no further client action needed (FAISS retrievable on next turn via the existing `retrieval` field). Threshold is read once from `/api/rag/stats` at init and cached. |
| `frontend/static/cache.js` | Add `getBaseConversationId / setBaseConversationId` for the base UUID. Per-column streaming flags, chunk caches, and consumed pointers are keyed by the column's full conversation_id. Pending inline files are NOT persisted — they're in-memory only (a page reload drops them; re-upload is cheap for small files). |

### 2.4 New Tests

```
backend/tests/rag/
├── test_retriever.py              # ScopedRetriever merging, metadata filter, per-scope cap (no merged-result cap)
├── test_service.py                # RagService ingest, purge, reindex — temp FAISS dirs + FakeEmbeddings
├── test_embeddings_factory.py     # returns correct LangChain Embeddings subclass per backend
├── test_splitter.py               # chunk size, overlap, metadata assembly
├── test_chain_integration.py      # ChatService with fake retriever — sources pushed, augmentation correct
└── test_routes.py                 # upload + reindex + stats via TestClient
```

---

## 3. Component Skeletons

### 3.1 ScopedRetriever

```python
from langchain_core.retrievers import BaseRetriever

class ScopedRetriever(BaseRetriever):
    """Wraps multiple retrievers, filtering specific ones by metadata.

    Per-scope cap is applied by the underlying retrievers (via
    search_kwargs={"k": k}) — ScopedRetriever does NOT cap the merged
    result, so the LLM can see top_k hits from each scope (e.g., 4
    library + 4 uploads = 8 total when both scopes are enabled).

    Convention: library chunks are tagged with metadata.source == "library"
    and have NO "conversation_id" field. Upload chunks have both. The
    filter `metadata.get("conversation_id") == self.conversation_id` is
    False for library chunks, so they pass through unfiltered under the
    should_filter=False branch. Do not add a "conversation_id" field to
    library chunks; the asymmetric metadata is the mechanism that makes
    the "library is global, uploads are per-conversation" property work.
    """
    retrievers: list[tuple[BaseRetriever, bool]]   # (retriever, should_filter_by_conv)
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

### 3.2 RagService

```python
import json, logging, shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from backend.rag.config import RagSettings
from backend.rag.retriever import ScopedRetriever
from backend.rag.vector_store import load_or_init, save, rebuild_filtered
from backend.rag.splitter import make_splitter
from backend.rag.embeddings import make_embeddings

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, settings: RagSettings):
        self.settings = settings
        self.embeddings: Embeddings = make_embeddings(settings.embedding_backend)
        self.splitter = make_splitter(settings.chunk_size, settings.chunk_overlap)

        # Anchor all paths to backend/rag/ — same pattern as backend/storage/file_storage.py
        # which uses Path(__file__).parent.parent.parent / "storage". This way uvicorn
        # can be started from any directory and paths still resolve correctly.
        backend_root = Path(__file__).parent.parent
        self.library_dir = (backend_root / settings.library_dir).resolve()
        self.uploads_dir = (backend_root / settings.uploads_dir).resolve()
        self.rag_dir = (backend_root / settings.rag_dir).resolve()

        # Index files are tagged with the embedding backend name. This prevents the
        # silent-failure mode where switching EMBEDDING_BACKEND loads a stale index
        # built with a different embedding model. After a backend change, the tagged
        # path is new and load_or_init starts fresh; reindex/upload is required.
        self.backend_tag = settings.embedding_backend
        self.library_index = load_or_init(
            self._index_path("library_index"), self.embeddings
        )
        self.uploads_index = load_or_init(
            self._index_path("uploads_index"), self.embeddings
        )

    def _index_path(self, name: str) -> Path:
        return self.rag_dir / f"{name}.{self.backend_tag}"

    @classmethod
    def from_settings(cls) -> "RagService":
        return cls(RagSettings())

    # ── Write paths ──────────────────────────────────────────────────

    def ingest_file(self, conversation_id: str, file_path: Path) -> list[str]:
        conv_uploads = self.uploads_dir / conversation_id
        conv_uploads.mkdir(parents=True, exist_ok=True)
        dest = conv_uploads / file_path.name
        if file_path.resolve() != dest.resolve():
            shutil.copy2(file_path, dest)

        text = _read_text(dest)
        chunks = self.splitter.split_text(text)
        docs = []
        for chunk_text in chunks:
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": "upload",
                    "conversation_id": conversation_id,
                    "filename": file_path.name,
                    "chunk_id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                },
            ))
        # The uploads_index may hold a placeholder doc when first initialized
        # (see vector_store.load_or_init). FAISS.add_documents does not handle
        # this cleanly, so rebuild against the existing docstore first to drop
        # the placeholder, then add the new chunks.
        self.uploads_index = rebuild_filtered(
            self.uploads_index, self.embeddings, keep=lambda d: True,
        )
        self.uploads_index.add_documents(docs)
        save(self.uploads_index, self._index_path("uploads_index"))
        return [d.metadata["chunk_id"] for d in docs]

    def reindex_library(self) -> dict:
        files = _walk_library(self.library_dir)
        errors = []
        all_docs = []
        for path in files:
            try:
                text = _read_text(path)
                chunks = self.splitter.split_text(text)
                for chunk_text in chunks:
                    all_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": "library",
                            "filename": str(path.relative_to(self.library_dir)),
                            "chunk_id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                        },
                    ))
            except Exception as e:
                errors.append(f"{path}: {e}")

        self.library_index = FAISS.from_documents(all_docs, self.embeddings)
        save(self.library_index, self._index_path("library_index"))
        return {"files_processed": len(files), "chunks_added": len(all_docs), "errors": errors}

    def purge_uploads(self, conversation_id: str) -> None:
        # 1. Remove files
        conv_uploads = self.uploads_dir / conversation_id
        if conv_uploads.exists():
            shutil.rmtree(conv_uploads)
        # 2. Rebuild index from surviving chunks
        self.uploads_index = rebuild_filtered(
            self.uploads_index,
            self.embeddings,
            keep=lambda doc: doc.metadata.get("conversation_id") != conversation_id,
        )
        save(self.uploads_index, self._index_path("uploads_index"))

    # ── Read path ────────────────────────────────────────────────────

    def make_scoped_retriever(self, conversation_id: str, top_k: int) -> ScopedRetriever:
        return ScopedRetriever(
            retrievers=[
                (self.library_index.as_retriever(search_kwargs={"k": top_k}), False),
                (self.uploads_index.as_retriever(search_kwargs={"k": top_k}), True),
            ],
            conversation_id=conversation_id,
        )

    # ── Misc ─────────────────────────────────────────────────────────

    def persist_all(self) -> None:
        save(self.library_index, self._index_path("library_index"))
        save(self.uploads_index, self._index_path("uploads_index"))

    def stats(self) -> dict:
        # NOTE: `docstore._dict` is a leading-underscore (private) attribute. Stable
        # in practice but technically internal to langchain_community.vectorstores.faiss.
        # If LangChain exposes a public iteration method later, switch to that.
        library_dict = self.library_index.docstore._dict
        uploads_dict = self.uploads_index.docstore._dict
        return {
            "enabled": True,
            "embedding_backend": self.settings.embedding_backend,
            "library_chunks": len(library_dict),
            "uploads_chunks": len(uploads_dict),
            "uploads_conversations": sorted({
                d.metadata["conversation_id"]
                for d in uploads_dict.values()
                if d.metadata.get("conversation_id")
            }),
        }
```

### 3.3 Modified ChatService

```python
class ChatService:
    def __init__(self, chain, rag_service: RagService | None = None):
        self.chain = chain
        self.rag_service = rag_service
        self._pending_inline_files: dict[str, list[dict]] = {}

    def clear_pending_inline_files(self, conversation_id: str) -> None:
        self._pending_inline_files.pop(conversation_id, None)

    async def generate_background(
        self,
        message: str,
        conversation_id: str,
        retrieval: RetrievalConfig | None = None,
        uploaded_files: list[UploadedFile] | None = None,
    ) -> None:
        job = get_or_create_job(conversation_id, [])

        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []
        job.messages = messages

        # Merge this turn's uploaded_files into the per-conversation pending list.
        if uploaded_files:
            self._pending_inline_files.setdefault(conversation_id, []).extend(
                {"filename": uf.filename, "content": uf.content}
                for uf in uploaded_files
            )

        pending = self._pending_inline_files.get(conversation_id, [])

        # ── Inline-files pre-processing block ────────────────────────
        # Takes precedence over RAG retrieval: a small file uploaded this turn
        # (or any earlier turn) means the response should be grounded on that
        # file directly, not augmented with unrelated FAISS hits.
        if pending:
            sources_event = {
                "sources": [
                    {
                        "filename": f["filename"],
                        "excerpt": f["content"][:300],
                        "scope": "upload",
                    }
                    for f in pending
                ]
            }
            job.append_chunk("sources", json.dumps(sources_event))
            context_str = "\n\n".join(
                f"[{f['filename']}]:\n{f['content']}" for f in pending
            )
            messages = messages[:-1] + [
                {"role": "system", "content": f"Use this uploaded file as context:\n\n{context_str}"},
                messages[-1],
            ]
        # ── RAG pre-processing block ────────────────────────────────
        elif retrieval is not None and self.rag_service is not None:
            try:
                scoped = self.rag_service.make_scoped_retriever(
                    conversation_id, retrieval.top_k
                )
                hits = scoped.invoke(message)
                if hits:
                    sources_event = {
                        "sources": [
                            {
                                "filename": h.metadata.get("filename"),
                                "excerpt": h.page_content[:300],
                                "scope": h.metadata.get("source"),
                            }
                            for h in hits
                        ]
                    }
                    job.append_chunk("sources", json.dumps(sources_event))
                    context_str = "\n\n".join(
                        f"[{h.metadata.get('filename')}]: {h.page_content}" for h in hits
                    )
                    messages = messages[:-1] + [
                        {"role": "system", "content": f"Use this retrieved context:\n{context_str}"},
                        messages[-1],
                    ]
            except Exception as e:
                logger.exception("Retrieval failed; continuing without context: %s", e)
        # ────────────────────────────────────────────────────────────

        try:
            async for chunk in self.chain.astream(messages):
                if job.cancelled:
                    return
                # ... existing chunk handling unchanged ...
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            return

        if job.cancelled:
            return

        job.mark_completed()
        # ... existing save logic unchanged ...
```

### 3.4 Startup Wiring

```python
# backend/main.py
from functools import partial
from backend.rag.service import RagService
from backend.rag.config import RagSettings
from backend.rag.routes import router as rag_router

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

### 3.5 `vector_store.py` helpers (sketch)

```python
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

def load_or_init(path: Path, embeddings: Embeddings) -> FAISS:
    """Load an existing FAISS index from path, or create an empty one if absent.
    An empty FAISS holds a single placeholder doc; this avoids the
    `from_documents([])` error and keeps the next `add_documents` call working."""
    if (path / "index.faiss").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    placeholder = Document(page_content="", metadata={"_placeholder": True})
    return FAISS.from_documents([placeholder], embeddings)

def save(index: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))

def rebuild_filtered(
    index: FAISS, embeddings: Embeddings, keep
) -> FAISS:
    """Rebuild index keeping only docs where keep(doc) is True. Handles the
    empty case by creating a placeholder index (see load_or_init)."""
    surviving = [doc for doc in index.docstore._dict.values() if keep(doc)]
    surviving = [d for d in surviving if not d.metadata.get("_placeholder")]
    if not surviving:
        return load_or_init(
            Path(index.index_dir_or_path) if hasattr(index, "index_dir_or_path") else Path("/dev/null"),
            embeddings,
        )
    return FAISS.from_documents(surviving, embeddings)
```

---

## 4. Configuration

New env vars (all optional; module is inert when `RAG_ENABLED` is unset/false):

```env
RAG_ENABLED=false                       # master switch; default false
EMBEDDING_BACKEND=sentence-transformers # or "minimax"
RAG_LIBRARY_DIR=storage/library          # admin-seeded corpus root
RAG_UPLOADS_DIR=storage/uploads          # per-conversation uploads root
RAG_INDEX_DIR=storage/rag                # FAISS index files live here
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4
SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1    # existing, reused for MiniMax embeddings
ANTHROPIC_API_KEY=...                              # existing, reused for MiniMax embeddings
```

**Notes:**
- `RAG_ENABLED=false` (default) means the system runs identically to iteration 6.
- `EMBEDDING_BACKEND=minimax` requires verification that MiniMax exposes an `/embeddings` route; if not, only `sentence-transformers` is selectable.
- New pip deps: `faiss-cpu`, `sentence-transformers`, `pypdf`.

---

## 5. Error Handling

| Failure | Behavior |
|---|---|
| `RAG_ENABLED=false`, client sends `retrieval` | Server ignores field, runs vanilla. Debug log. |
| Uploaded file unreadable / too large | 400, file not saved. |
| Embedding API down (MiniMax) | 500, file kept on disk, index untouched. User retries. |
| FAISS save fails after `add_documents` | 500, critical log. In-memory state consistent, disk stale until next reindex. |
| Retrieval fails mid-turn | Stream error chunk; LLM call proceeds with un-augmented messages. |
| `delete_conversation` succeeds, `purge_uploads` fails | JSON state consistent; orphan chunks invisible at query time. Warning logged. |
| Library reindex encounters unreadable file | File skipped, error in response `errors` list. Run continues. |

---

## 6. Testing Strategy

### 6.1 Layers

| Layer | Files | Speed target |
|---|---|---|
| Unit | `test_retriever.py`, `test_splitter.py`, `test_embeddings_factory.py` | <10 ms each |
| Service | `test_service.py` (real FAISS in `tmp_path` + `FakeEmbeddings`) | <500 ms each |
| Chain integration | `test_chain_integration.py` (ChatService with mock retriever) | <1 s each |
| Route | `test_routes.py` (`TestClient` against FastAPI app with test `RagService`) | <1 s each |

### 6.2 Load-bearing Test

```python
def test_no_rag_path_is_byte_identical_to_today():
    """The plugin-off guarantee: when retrieval is None, the service produces
    exactly the same job.chunks as iteration 6. This is the assertion that
    makes 'RAG is a selective plugin' testable, not just claimable."""
    service = ChatService(chain=FakeListLLM(responses=["x"]))   # no rag_service
    job = StreamJob("c1")
    await service.generate_background("hi", "c1", retrieval=None)
    chunks = job.chunks
    # No 'sources' chunk was ever appended
    assert all(c["type"] != "sources" for c in chunks)
    # The LLM was called with the original messages, un-augmented
    assert chain.last_input == [{"role": "user", "content": "hi"}]
```

### 6.3 Frontend Verification (manual, v1)

1. Page loads → both panels render empty.
2. Type "What is RAG?" → Send → both panels stream responses.
3. Vanilla answer is general; RAG answer references the library doc (if seeded).
4. Upload `doc.txt` to RAG panel → re-ask → answer references `doc.txt`.
5. Toggle "Show sources" off → sources block disappears; response still streams.
6. Refresh page → conversation list shows one row per pair (the base UUID; the `-0`/`-1` suffix is not displayed in the sidebar).
7. Delete both → both disappear; uploads dir cleaned.

---

## 7. Future Work (post-v1, documented but out of scope)

1. Tombstone-based delete — mark chunks deleted in a set, exclude at query time, periodic compaction. Replaces full-rebuild when indexes grow.
2. SQLite/DuckDB docstore — replaces the FAISS pickle sidecar when chunk counts exceed ~50k.
3. Cross-encoder re-ranking — improves retrieval precision.
4. Multi-process locking — `fcntl`/`msvcrt` file lock for FAISS index (and `conversations.json`).
5. Background ingestion — long uploads don't block the request; client polls for completion.
6. Additional scopes — "web cache", "agent memory", "tool results". Pattern: add FAISS index + entry in `ScopedRetriever.retrievers` + bool flag in `RetrievalConfig`.
7. Per-conversation FAISS index — if `uploads` grows large enough that one index becomes a hotspot.
8. Auth on `/api/rag/library/reindex` — when the app leaves local-only.
9. Chunk deduplication — manifest-based skip-if-unchanged during library reindex.