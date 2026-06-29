# Chatbot Project — Iteration 7 Spec (RAG Module)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Covers the new RAG (Retrieval-Augmented Generation) capability and the side-by-side comparison UI for evaluating it.

## Overview

This iteration adds a **Retrieval-Augmented Generation (RAG) module** to the chatbot as a **selective plugin**. The current chat system must continue to run **unmodified when RAG is disabled**; when enabled, chats can be augmented with content retrieved from a global document library and/or per-conversation uploaded files.

**Core Goal:** Add a pluggable RAG capability that integrates seamlessly with the existing chat flow, exposes a side-by-side comparison UI for evaluating retrieval impact, and keeps the chain implementation untouched when the feature is off.

**Iteration 7 Highlights:**
- New `backend/rag/` domain with pluggable embeddings (sentence-transformers / MiniMax) and vector store (FAISS default) backends.
- Two retrieval scopes merged at query time: a global **library** (admin-seeded from `storage/library/`) and per-conversation **uploads** (user-supplied via the UI).
- `RAG_ENABLED` env flag isolates the feature completely. With it off, the chat runs identically to iteration 6.
- New `RetrievalConfig` field on `ChatRequest` lets clients opt into library and/or uploads on a per-request basis.
- New `sources` SSE event fires before the first token, listing the chunks retrieved for the current turn.
- New `/api/rag/upload`, `/api/rag/library/reindex`, and `/api/rag/stats` endpoints.
- New two-panel comparison UI: shared input box, vanilla panel (left) and RAG panel (right), with independent histories and per-panel sources toggle.
- Conversation IDs follow the pattern `<random-uuid>-0` (vanilla panel) and `<random-uuid>-1` (RAG panel) so pairs are easy to identify and delete together.

---

## Functional Requirements

### FR-10: RAG Activation

| ID | Requirement |
|----|-------------|
| FR-10.1 | The RAG module is controlled by a `RAG_ENABLED` environment variable, defaulting to `false`. |
| FR-10.2 | When `RAG_ENABLED=false`, the chat system runs exactly as in iteration 6 — no rag imports executed in the request hot path, no behavior change. |
| FR-10.3 | `ChatRequest` accepts an optional `retrieval: RetrievalConfig` field. When the field is absent or `null`, the vanilla chain runs. |
| FR-10.4 | When `RAG_ENABLED=true` and `retrieval` is set on a request, the chat response is augmented with retrieved context using the requested scopes. |
| FR-10.5 | When `RAG_ENABLED=false` and a client sends `retrieval` anyway, the server ignores the field and runs the vanilla chain (logs at debug level). |
| FR-10.6 | `RetrievalConfig` has three fields: `library: bool = True`, `uploads: bool = True`, `top_k: int = 4`. |

### FR-11: Global Document Library

| ID | Requirement |
|----|-------------|
| FR-11.1 | The system reads documents from a configurable directory (default `storage/library/`). |
| FR-11.2 | Supported file types: `.md`, `.txt`, `.pdf`, `.html`. |
| FR-11.3 | Documents are chunked, embedded, and stored in a FAISS index on disk at `storage/rag/library_index.<embedding_backend>/`. The `<embedding_backend>` tag is appended to the directory name so that switching `EMBEDDING_BACKEND` does not silently load an index built with a different embedding model — after a backend change, the tagged path is new and a reindex is required to repopulate it. |
| FR-11.4 | `POST /api/rag/library/reindex` rebuilds the library index from the directory contents. |
| FR-11.5 | Reindex is idempotent: running it twice with unchanged files produces the same chunks. |
| FR-11.6 | Unreadable files are reported in the response's `errors` list and do not abort the run. |
| FR-11.7 | The library index is loaded once at startup and persisted to disk after every reindex. |

### FR-12: Per-Conversation File Upload (size-based routing)

The upload feature is available from **both** columns (Vanilla and RAG). Each column has its own upload button beside the Send button. Files are routed to one of two pipelines based on their **raw size** compared to a configurable threshold (`RAG_INLINE_CONTEXT_THRESHOLD_BYTES`, default 8192 bytes):

- **Small files** (≤ threshold): file text content is sent inline with the next chat request and injected as a system message into the LLM call. No FAISS ingestion, no chunking, no retrieval. The file content is bound to that conversation and survives across turns until the conversation is deleted.
- **Large files** (> threshold): embedded and indexed in FAISS as today; retrieved per-turn via the existing ScopedRetriever path.

The threshold applies to **raw upload size**, not post-extraction text size. PDFs and HTML always go through FAISS (they're typically > threshold in raw form); `.md`/`.txt` files under the threshold use the inline path.

| ID | Requirement |
|----|-------------|
| FR-12.1 | Both columns expose an Upload button in their column header. The button accepts a single file at a time. Only files with extensions in the allowlist `{".md", ".txt", ".pdf", ".html"}` are accepted. The frontend's `<input type="file" accept="...">` attribute hints this to the browser's file picker. The server enforces the allowlist and rejects any other extension with HTTP 400. |
| FR-12.2 | **Inline path (small file):** the upload endpoint reads the file content server-side and returns `{filename, content, mode: "inline"}` to the client. The client includes this content in the next `/api/chat/stream` request via the `uploaded_files` field. No file is written to disk. |
| FR-12.3 | **FAISS path (large file):** the upload endpoint saves the file to `storage/uploads/<conversation_id>/` and ingests it into the uploads FAISS index. Returns `{filename, chunks_added, chunk_ids, mode: "indexed"}` to the client. |
| FR-12.4 | The threshold comparison uses raw upload size in bytes, not post-extraction text size. `.md`/`.txt` ≤ 8 KB typically take the inline path; `.pdf`/`.html` always take the indexed path. |
| FR-12.5 | The `RAG_INLINE_CONTEXT_THRESHOLD_BYTES` env var (default 8192) controls the threshold. The threshold value is exposed via `GET /api/rag/stats` so the frontend can apply the same boundary the server uses. |
| FR-12.6 | When the client sends `uploaded_files` with a chat request, the `ChatService` injects each file's content as a single system message before the user message: `"Use this uploaded file as context:\n\n[filename]:\n<content>"`. The files are appended to a server-side pending-files list keyed by `conversation_id`, so subsequent turns in the same conversation also see them without re-sending. |
| FR-12.7 | The pending-files list is cleared when the conversation is deleted (via the same `on_delete` callback that clears FAISS uploads). |
| FR-12.8 | `ChatRequest` gains an optional `uploaded_files: list[UploadedFile]` field. Each `UploadedFile` has `filename: str` and `content: str`. When this field is non-empty, the RAG retriever step is **skipped** for that turn — the inline context takes its place. The vanilla and RAG columns both use this path for small files. |
| FR-12.9 | When `uploaded_files` is non-empty, the server emits a `sources` SSE event listing `{filename, scope: "upload", excerpt: first 300 chars}` for each file before the first token. This makes it visible in the RAG column that uploaded files were used (and the vanilla column renders the same sources block). |
| FR-12.10 | FAISS path persistence: each indexed chunk is tagged with `conversation_id`, `filename`, `chunk_id`, and `source="upload"` metadata. The uploads index is persisted at `storage/rag/uploads_index.<backend_tag>/` after every upload. |
| FR-12.11 | FAISS path error handling: if embedding/indexing fails, the file is kept on disk; the index is untouched; the response is 500. The user can retry. |
| FR-12.12 | The upload endpoint calls `file_storage.create_conversation(conversation_id)` first (idempotent). This guarantees the conversation is visible in the sidebar immediately after upload, even if the user hasn't sent any message yet. |

### FR-13: Retrieval-Augmented Chat

| ID | Requirement |
|----|-------------|
| FR-13.1 | When `retrieval` is set, the server retrieves chunks using the **latest user message** as the query. |
| FR-13.2 | Retrieved chunks from the library scope are not filtered; library chunks pass through to all conversations. |
| FR-13.3 | Retrieved chunks from the uploads scope are filtered by `conversation_id`. A conversation cannot retrieve another conversation's uploads. |
| FR-13.4 | The number of chunks returned is at most `top_k` per scope (default 4). |
| FR-13.5 | Retrieved chunks are formatted into a system message inserted just before the last user message, in the form `"Use this retrieved context:\n[filename]: text..."` (single newline between the header and the chunks; chunks are separated by `\n\n`). |
| FR-13.6 | Retrieval happens once per turn, before the first token of the LLM response streams. |
| FR-13.7 | If retrieval fails mid-turn, the chat continues with the original (un-augmented) messages and the user receives an error chunk via SSE. |

### FR-14: Sources Visibility

| ID | Requirement |
|----|-------------|
| FR-14.1 | Before the first token, the server emits a `sources` SSE event listing the chunks used in the augmented response. |
| FR-14.2 | Each source has `filename`, `excerpt` (first 300 chars of `page_content`), and `scope` (`"library"` or `"upload"`). |
| FR-14.3 | The vanilla chain never emits a `sources` event. |
| FR-14.4 | The RAG panel has a "Show sources" checkbox (default ON). When OFF, the panel renders no sources block even if the event arrives. |
| FR-14.5 | Sources state is local to the RAG panel — the server does not know or care whether the user is rendering them. |

### FR-15: Side-by-Side Comparison UI

The chat UI is replaced with a side-by-side comparison layout: a single shared input at the bottom, and two message columns above — **Vanilla** (left) and **RAG** (right). One Send click triggers two parallel POSTs to `/api/chat/stream`, one per column, with `retrieval: null` for vanilla and `retrieval: {library: true, uploads: true, top_k: 4}` for RAG. The two columns stream their responses independently and simultaneously, so the user sees both answers at once and can directly compare them.

| ID | Requirement |
|----|-------------|
| FR-15.1 | The chat container renders two message columns side-by-side: a "Vanilla" panel (left) and a "RAG" panel (right). Each column has its own scrollable message list and its own header label. |
| FR-15.2 | A single shared input box and Send button sit below both columns. Typing a message and pressing Send triggers two parallel `/api/chat/stream` POSTs — one per column — using the same `message` text and two distinct `conversation_id`s. |
| FR-15.3 | The two columns map to two distinct conversation IDs sharing a base UUID: `<base>-0` (vanilla) and `<base>-1` (RAG). The base is generated once and persisted in localStorage so the pair survives page reloads. |
| FR-15.4 | The vanilla POST sends `retrieval: null`. The RAG POST sends `retrieval: {library: true, uploads: true, top_k: 4}`. Each column streams its SSE response into its own column only — vanilla's tokens never appear in the RAG column, and vice versa. |
| FR-15.5 | Each column renders the `sources` SSE event (when present) as a "Sources" block above its assistant message, before the first token. The vanilla chain only emits a `sources` event when the user has uploaded files to it (FR-12.9); the RAG column emits sources from FAISS retrieval and also from inline uploads. |
| FR-15.6 | Both columns expose an Upload button beside the Send button (FR-12.1). The vanilla column's upload uses the inline path (FR-12.6/12.8) — the file content becomes system context but no retrieval happens, so the vanilla stream stays a pure LLM response grounded on the file. The RAG column's upload may use either path depending on file size. |
| FR-15.7 | Both sub-conversations live as normal entries in `conversations.json`, but the sidebar shows a single row per pair (identified by the base UUID; the `-0`/`-1` suffix is not displayed). Clicking a pair row loads both columns with that pair's sub-conversations (`<base>-0` into the vanilla column, `<base>-1` into the RAG column). The columns are not loaded independently — a pair is the unit of selection. |
| FR-15.8 | On page load, both columns render empty (no prior history) or load the histories for `<base>-0` and `<base>-1` if they exist. The shared input is empty. |
| FR-15.9 | When `RAG_ENABLED=false`, the RAG column is hidden. The vanilla column takes the full width. The user can chat normally with no RAG option shown. The shared input is still present, but only one POST fires per Send. |
| FR-15.10 | Cancelling (or a stream error in) one column must not cancel the other column's stream. Each column manages its own abort controller and resume-from-cache state. |
| FR-15.11 | The columns share a single "Send" affordance. The Send button is disabled while EITHER column is streaming. Re-enabled when BOTH columns have completed (or aborted). |

### FR-16: Upload Lifecycle

| ID | Requirement |
|----|-------------|
| FR-16.1 | When a conversation is deleted, all uploaded files for that conversation are removed from disk. |
| FR-16.2 | When a conversation is deleted, all index chunks tagged with that conversation's `conversation_id` are removed from the uploads index. |
| FR-16.3 | Library chunks are untouched by conversation deletion. |
| FR-16.4 | If the index rebuild on delete fails, the conversation is still removed from the user's view; orphan chunks remain invisible because no conversation_id matches them. |
| FR-16.5 | The storage layer's `delete_conversation` accepts an optional `on_delete: Callable[[str], None]` parameter, so the deletion flow is extended via the signature rather than by patching the storage layer's internals. The callback itself is wired at application startup. |

### FR-17: RAG Stats Endpoint

| ID | Requirement |
|----|-------------|
| FR-17.1 | `GET /api/rag/stats` returns `{enabled, embedding_backend, library_chunks, uploads_chunks, uploads_conversations, inline_context_threshold_bytes}`. |
| FR-17.2 | Returns 503 when `RAG_ENABLED=false`. |

### FR-18: RAG Disabled Behavior

| ID | Requirement |
|----|-------------|
| FR-18.1 | When `RAG_ENABLED=false`, the `/api/rag/*` endpoints return 503. |
| FR-18.2 | When `RAG_ENABLED=false`, `delete_conversation` does not invoke any rag cleanup hook (the hook is only wired at startup when RAG is enabled). |
| FR-18.3 | When `RAG_ENABLED=false`, all existing iteration-6 tests pass without modification. |