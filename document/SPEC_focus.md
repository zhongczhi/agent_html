# Chatbot Project — Iteration 8 Spec (Multi-Format RAG, Library Mgmt, Show Sources)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Covers the multi-format document loader pipeline, library management API + UI, and the show-sources toggle.

## Overview

Iteration 7 shipped the RAG module (two-scope FAISS retrieval, per-conversation file upload with size-based routing, side-by-side compare UI). Iteration 8 closes three real gaps in that module:

1. **Multi-format ingestion**: the loader pipeline currently handles 4 formats (`.md`, `.txt`, `.pdf`, `.html`) and has quality problems — HTML read raw (no tag stripping), PDF loses page numbers, Markdown splits mid-header. Iteration 8 replaces the single `_read_text` function with a `LoaderRegistry` mapping extension → loader function, adds `.docx` and `.csv`, fixes HTML/PDF/MD quality, and outputs a uniform chunk Document with rich per-format metadata.
2. **Library management**: today, populating `storage/library/` requires dropping files on disk + calling `POST /api/rag/library/reindex`. Iteration 8 adds a library upload endpoint + sidebar tab so the library is operable from the UI.
3. **Show sources toggle**: FR-14.4 (RAG panel "Show sources" checkbox) is in the iter-7 spec but missing from the frontend. Iteration 8 adds it.

The chain implementation (`backend/chat/chain.py`) is unchanged; the plugin-off property is preserved.

**Iteration 8 Highlights:**
- New `backend/rag/loaders/` package — one module per format, registered via decorator.
- File allowlist extended from 4 to 6 formats: `.md`, `.txt`, `.pdf`, `.html`, `.docx`, `.csv`.
- Every chunk Document carries `format`, `source_type`, `filename`, `chunk_id`, plus optional per-format metadata (`page_number`, `header_path`, `title`, `row_number`, etc.).
- `MarkdownTextSplitter` for `.md` (was: char-based `RecursiveCharacterTextSplitter`).
- HTML loaded via BeautifulSoup; `<script>` / `<style>` stripped; `<title>` captured as metadata.
- PDF loaded page-by-page; `page_number` and `total_pages` captured per chunk.
- Library endpoints: `POST /api/rag/library/upload`, `GET /api/rag/library/files`, `DELETE /api/rag/library/file/{filename}`.
- Auto-reindex on library upload + delete (user-approved resolution during planning).
- Sidebar gains `[Conversations]` / `[Library]` tabs. Library tab lists files with size + modified-at, exposes Upload / Reindex / per-file delete, and shows a stats footer.
- Show-sources checkbox in RAG column header (default ON). Hides the sources block without affecting retrieval or the SSE event.
- `chunk_id` formula changes from `sha256(chunk_text)` to `sha256(f"{path.name}:{chunk_text}")` so identical chunks from different files get distinct IDs.

---

## Functional Requirements

### FR-25: Pluggable Document Loader Registry

| ID | Requirement |
|----|-------------|
| FR-25.1 | A `LoaderRegistry` in `backend/rag/loaders/` maps file extension → loader function. Each loader yields `RawDocument(text, metadata)` tuples where `text` is one extractable unit (whole file for `.md`/`.txt`/`.html`, one page for `.pdf`, one paragraph for `.docx`, one row for `.csv`). |
| FR-25.2 | `ALLOWED_EXTENSIONS` is derived from `REGISTRY.keys()`. Adding a new format requires exactly two edits: a new loader module + a registration entry. No changes to routes, service, or splitter. |
| FR-25.3 | Supported formats: `.md`, `.txt`, `.pdf`, `.html`, `.docx`, `.csv`. The `POST /api/rag/upload` (per-conversation) and `POST /api/rag/library/upload` (new in FR-26) endpoints reject any other extension with HTTP 400 before any IO. |
| FR-25.4 | Every chunk Document carries these guaranteed metadata fields: `source_type` (`"library"` or `"upload"`), `filename`, `chunk_id`, `format` (the file extension), and where applicable `source` (iter-7 compat key with the same value as `source_type`). |
| FR-25.5 | Per-format metadata is propagated to each chunk: PDFs carry `page_number` and `total_pages`; Markdown carries `header_path` (e.g., `"Intro / Setup"`); HTML carries `title`; CSV carries `row_number` and `headers`. |
| FR-25.6 | Empty RawDocuments (e.g., a PDF page with no extractable text, an empty CSV row) are filtered out at the splitter level — no empty Documents are emitted into FAISS. |
| FR-25.7 | If a loader raises (corrupt DOCX, malformed CSV), the error is captured in the parent's `errors` list and the run continues with remaining files. |
| FR-25.8 | Calling `split_into_documents` with an extension not in `REGISTRY` raises `UnsupportedFormatError` at the boundary; the caller surfaces it as HTTP 400 / error to the user. |

### FR-26: Per-Format Splitter

| ID | Requirement |
|----|-------------|
| FR-26.1 | `pick_splitter(extension)` returns `MarkdownTextSplitter` for `.md` and `RecursiveCharacterTextSplitter` for all other supported formats. Both splitters are configured by `RagSettings.rag_chunk_size` and `RagSettings.rag_chunk_overlap`. |
| FR-26.2 | `split_into_documents(path, source_type, conversation_id, chunk_size, chunk_overlap)` is the single entry point used by `RagService.ingest_file` and `RagService.reindex_library`. It dispatches to the registered loader, runs the format-appropriate splitter on each `RawDocument`, and yields `Document(page_content, metadata)` with propagated per-format metadata + iter-7 compat fields. |
| FR-26.3 | `chunk_id` is computed as `hashlib.sha256(f"{path.name}:{chunk_text}").hexdigest()[:16]`. Two chunks from different files with identical text get distinct `chunk_id`s. |
| FR-26.4 | Markdown chunks carry `header_path` metadata derived by walking the original file's header hierarchy at the chunk's offset. The breadcrumb is the chain of headers (H1, H2, H3, …) most-recent-before the chunk, joined by `" / "`. Chunks before any header have `header_path = ""`. |

### FR-27: Library Management API

| ID | Requirement |
|----|-------------|
| FR-27.1 | `POST /api/rag/library/upload` accepts a single file (multipart). Allowed extensions match FR-25.3. Saves to `storage/library/<filename>`. Atomic write via `tmp + os.replace`. Returns `{filename, size, saved: true}` on success. |
| FR-27.2 | `POST /api/rag/library/upload` returns HTTP 409 when `<filename>` already exists in the library; the user must delete first. |
| FR-27.3 | `POST /api/rag/library/upload` returns HTTP 400 when `<filename>` contains `/`, `\`, or starts with `.` (path-traversal / dotfile guard). |
| FR-27.4 | `GET /api/rag/library/files` returns `{files: [{filename, size, modified_at}]}`, sorted alphabetically. Only allowlisted extensions are listed. |
| FR-27.5 | `DELETE /api/rag/library/file/{filename}` removes the file and triggers reindex. Returns HTTP 200 `{deleted: true, filename}` on success, HTTP 404 when not found, HTTP 400 on invalid filename or non-allowlisted extension. |
| FR-27.6 | **Auto-reindex on upload**: after a successful library upload, `RagService.reindex_library()` runs automatically so the uploaded file is queryable immediately. No manual reindex step is required for the typical flow. |
| FR-27.7 | **Auto-reindex on delete**: after a successful library file delete, `RagService.reindex_library()` runs automatically so the FAISS index reflects the on-disk state. |
| FR-27.8 | **Manual reindex**: `POST /api/rag/library/reindex` (existing from iter-7) remains available as a force-refresh / recovery action. |
| FR-27.9 | `RagService.__init__` calls `self.library_dir.mkdir(parents=True, exist_ok=True)` so the directory exists from first startup regardless of whether anyone has uploaded anything. |
| FR-27.10 | `save_library_file` uses an atomic write (tmp file + `os.replace`) — a crash mid-write leaves the previous file fully intact. |

### FR-28: Library Sidebar Tab

| ID | Requirement |
|----|-------------|
| FR-28.1 | The sidebar gains a tab strip above the existing conversation list: `[Conversations]` and `[Library]` buttons. |
| FR-28.2 | When `RAG_ENABLED=false`, the Library tab is hidden — the feature is gated on RAG being enabled. |
| FR-28.3 | The active tab persists in `localStorage` (`currentSidebarTab`, default `"conversations"`) and is restored on page load. Tab switch does not affect the active conversation or any in-flight chat stream. |
| FR-28.4 | The Library tab renders: header with `[Upload]` and `[Reindex]` buttons; a file list (name, size, last-modified) with a per-row `×` delete button; a stats footer `"N chunks from M files"`; an empty state `"No files in library. Click Upload to add one."`. |
| FR-28.5 | Clicking `[Upload]` triggers an `<input type="file" accept=".md,.txt,.pdf,.html,.docx,.csv">`; on file selection, the file is POSTed to `/api/rag/library/upload`. On success: refresh the file list. On error: inline error message. |
| FR-28.6 | Clicking a row's `×` opens the iter-7 themed confirmation modal; on confirm, `DELETE /api/rag/library/file/{filename}` is called and the list refreshes. |
| FR-28.7 | Clicking `[Reindex]` calls `POST /api/rag/library/reindex` and refreshes both the file list and the stats footer. |
| FR-28.8 | `GET /api/rag/stats` response gains a `library_files: int` field so the footer can render without a second request. |

### FR-29: Show Sources Toggle

| ID | Requirement |
|----|-------------|
| FR-29.1 | The RAG column header gains a `<label class="show-sources-toggle">` containing `<input type="checkbox" id="showSourcesToggle" checked> Show sources</label>`, positioned between the column title and the Upload button. |
| FR-29.2 | Default state is ON. State persists in `localStorage` (`showSources`, default `"true"`). Restored on page load. |
| FR-29.3 | When OFF, `renderSourcesBlock` early-returns — sources blocks are not added to the DOM. Retrieval and the SSE event itself are unaffected. |
| FR-29.4 | The toggle change handler calls `applySourcesVisibility()` which walks every `.sources-block` element in the RAG column and sets `style.display` to `''` or `'none'` based on the new state. This re-shows previously-rendered blocks when toggled back ON. |
| FR-29.5 | The vanilla column does not show the toggle. The vanilla column never emits sources (unless the user has uploaded a file to that column — iter-7 inline-file path emits a sources block; per FR-29.3, that block is also gated by the toggle if the toggle is OFF, even in the vanilla column). |

---

## Non-Functional Requirements

### NFR-8: Loader Isolation

- A failing loader (corrupt DOCX, malformed CSV) must not abort the entire reindex. The error is captured in the response `errors` list and the remaining files are still ingested.

### NFR-9: Atomic Library Writes

- A library file write is atomic via `tmp + os.replace`. A crash mid-write leaves the previous file fully intact (no partial files, no corrupted state).

### NFR-10: Backward Compatibility

- All iter-7 tests must continue to pass unchanged.
- The iter-7 chunk metadata `source` (value: `"library"` or `"upload"`) is preserved alongside the new explicit `source_type`. Code reading chunks should use `source_type` going forward; `source` is retained for iter-7 frontend compatibility.
- A mixed-state FAISS index (iter-7 chunk_ids + iter-8 chunk_ids) is safe — chunk_id is metadata, not a FAISS vector ID.

### NFR-11: Format Allowlist Single Source

- `ALLOWED_EXTENSIONS` is derived from `REGISTRY.keys()`. There is one allowlist, defined by the loader modules. Routes, frontend `accept` attribute, and tests all derive from this single source.

---

## Out of Scope (deferred to future iterations)

- PPTX, XLSX, RTF, EPUB (the loader registry is ready to absorb these later)
- Auto-reindex debouncing (current implementation runs synchronously after each upload)
- Per-file incremental reindex (current: whole-index)
- Versioned library (undo, diff, history)
- Markdown link-target extraction, code-block language detection
- OCR for images
- Web URL ingestion
- Inline-files persistence across server restart (already deferred in iter-7)
- Cross-encoder reranking, per-conversation FAISS indexes, chunk deduplication