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
| FR-11.3 | Documents are chunked, embedded, and stored in a FAISS index on disk at `storage/rag/library_index/`. |
| FR-11.4 | `POST /api/rag/library/reindex` rebuilds the library index from the directory contents. |
| FR-11.5 | Reindex is idempotent: running it twice with unchanged files produces the same chunks. |
| FR-11.6 | Unreadable files are reported in the response's `errors` list and do not abort the run. |
| FR-11.7 | The library index is loaded once at startup and persisted to disk after every reindex. |

### FR-12: Per-Conversation File Upload

| ID | Requirement |
|----|-------------|
| FR-12.1 | The RAG chat panel exposes an upload button that accepts a single file at a time. |
| FR-12.2 | Uploaded files are saved to `storage/uploads/<conversation_id>/` before indexing. |
| FR-12.3 | Each chunk is tagged with `conversation_id`, `filename`, `chunk_id`, and `source="upload"` metadata. |
| FR-12.4 | Upload returns `{filename, chunks_added, chunk_ids}` to the client. |
| FR-12.5 | If embedding/indexing fails, the file is kept on disk; the index is untouched; the response is 500. The user can retry. |
| FR-12.6 | The uploads index is persisted to disk at `storage/rag/uploads_index.<backend_tag>/` after every upload. |
| FR-12.7 | The upload endpoint calls `file_storage.create_conversation(conversation_id)` first (idempotent). This guarantees the conversation is visible in the sidebar immediately after upload, even if the user hasn't sent any message yet. |

### FR-13: Retrieval-Augmented Chat

| ID | Requirement |
|----|-------------|
| FR-13.1 | When `retrieval` is set, the server retrieves chunks using the **latest user message** as the query. |
| FR-13.2 | Retrieved chunks from the library scope are not filtered; library chunks pass through to all conversations. |
| FR-13.3 | Retrieved chunks from the uploads scope are filtered by `conversation_id`. A conversation cannot retrieve another conversation's uploads. |
| FR-13.4 | The number of chunks returned is at most `top_k` per scope (default 4). |
| FR-13.5 | Retrieved chunks are formatted into a system message inserted just before the last user message, in the form `"Use this retrieved context:\n\n[filename]: text..."`. |
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

### FR-15: Comparison UI

| ID | Requirement |
|----|-------------|
| FR-15.1 | The page shows two chat panels side by side: a vanilla panel and a RAG panel. |
| FR-15.2 | A single shared input box and Send button at the top of the page. |
| FR-15.3 | On Send, both panels POST to `/api/chat/stream` with the same `message` text but different `conversation_id` and `retrieval` settings. |
| FR-15.4 | Each panel maintains its own conversation history, scroll position, and SSE stream. |
| FR-15.5 | Each panel generates its own `conversation_id` on page load: vanilla = `<random>-0`, RAG = `<random>-1`, where `<random>` is a shared UUID base. |
| FR-15.6 | The pair is easy to identify in the conversation list and easy to delete together. |
| FR-15.7 | Each panel can independently load its history from `GET /api/chat/history/<conversation_id>` on mount. |

### FR-16: Upload Lifecycle

| ID | Requirement |
|----|-------------|
| FR-16.1 | When a conversation is deleted, all uploaded files for that conversation are removed from disk. |
| FR-16.2 | When a conversation is deleted, all index chunks tagged with that conversation's `conversation_id` are removed from the uploads index. |
| FR-16.3 | Library chunks are untouched by conversation deletion. |
| FR-16.4 | If the index rebuild on delete fails, the conversation is still removed from the user's view; orphan chunks remain invisible because no conversation_id matches them. |
| FR-16.5 | The deletion flow is wired through a callback parameter on `file_storage.delete_conversation`; no monkey-patching. |

### FR-17: RAG Stats Endpoint

| ID | Requirement |
|----|-------------|
| FR-17.1 | `GET /api/rag/stats` returns `{enabled, embedding_backend, library_chunks, uploads_chunks, uploads_conversations}`. |
| FR-17.2 | Returns 503 when `RAG_ENABLED=false`. |
| FR-17.3 | Used by the frontend to render an optional "X docs indexed" indicator. |

### FR-18: RAG Disabled Behavior

| ID | Requirement |
|----|-------------|
| FR-18.1 | When `RAG_ENABLED=false`, the `/api/rag/*` endpoints return 503. |
| FR-18.2 | When `RAG_ENABLED=false`, `delete_conversation` does not invoke any rag cleanup hook (the hook is only wired at startup when RAG is enabled). |
| FR-18.3 | When `RAG_ENABLED=false`, all existing iteration-6 tests pass without modification. |