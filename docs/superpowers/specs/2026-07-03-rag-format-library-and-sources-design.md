# RAG Module — Iteration 8 Design Spec

**Date**: 2026-07-03
**Status**: Draft, pending user review
**Iteration goal**: Extend the iter-7 RAG module in three places: (1) a pluggable, per-format document loader registry that fixes HTML/PDF/Markdown quality issues and adds DOCX + CSV; (2) a library-management API + sidebar tab so admins can populate `storage/library/` from inside the app instead of dropping files on disk; (3) a per-column "Show sources" checkbox in the RAG panel so users can compare answers without source noise.

This iteration extends the existing iter-7 RAG code; the chat core remains unmodified.

---

## 1. Goals & Non-Goals

### Goals

1. **Loader registry**: replace the single `_read_text(path)` function in `backend/rag/splitter.py` with a `LoaderRegistry` mapping extension → loader function. Each loader yields `RawDocument(text, metadata)` tuples, supporting text-bearing units (PDF pages, CSV rows, DOCX paragraphs, MD sections).
2. **Add DOCX and CSV** to the file allowlist (existing: `.md`, `.txt`, `.pdf`, `.html`).
3. **Fix HTML**: use BeautifulSoup to strip tags and extract `<title>` metadata — today HTML is read raw, polluting chunks with `<html>`, `<div>`, `<script>` content.
4. **Fix PDF**: preserve page numbers as `page_number` metadata on each chunk. Today all pages are concatenated with `\n`, losing pagination.
5. **Fix Markdown**: use `MarkdownTextSplitter` so splits respect header / code-block / list boundaries. Today character-based splitting produces mid-header chunks.
6. **Uniform output format**: every chunk Document carries `source_type`, `filename`, `chunk_id`, `format`, and optional format-specific metadata (`page_number`, `header_path`, `title`, `row_number`). Downstream code reads these without format-specific branches.
7. **Library management API**: `POST /api/rag/library/upload`, `GET /api/rag/library/files`, `DELETE /api/rag/library/file/{filename}`. `storage/library/` is auto-created at RagService startup.
8. **Library sidebar tab**: `[Conversations] [Library]` tabs in the sidebar. Library tab shows files with size + modified-at, plus Upload / Reindex buttons and stats footer.
9. **Show sources toggle**: per-RAG-column checkbox (default ON). Local state in localStorage. Hides the sources block without affecting retrieval or the SSE event itself.

### Non-Goals (v1)

- No PPTX / XLSX / RTF / EPUB. The 6 supported formats cover the common case; the loader registry pattern means adding more is additive later.
- No auto-reindex after library upload. Admin clicks Reindex explicitly. Trade-off: `library_chunks` count is stale until reindex runs.
- No versioned library (no undo, no diff, no per-file reindex). Reindex is whole-index.
- No inline-files persistence across server restart (iter-7 DESI already deferred this).
- No markdown-aware metadata propagation beyond `header_path` (e.g. link target extraction, code-block language detection).
- No Playwright/JS test framework. Manual UI verification (consistent with iter-7).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                   │
│                                                                         │
│  Sidebar: [Conversations] [Library]                                      │
│       │                            │                                    │
│       ▼                            ▼                                    │
│  Conversation list          Library tab: list of files,                 │
│  (existing iter-7 UI)       Upload / Reindex / Delete buttons,          │
│                             stats footer (N chunks from M files)        │
│                                                                         │
│  Compare grid (existing):                                               │
│   Vanilla column  |  RAG column                                         │
│                    │   [Show sources ☑]   ← NEW                        │
└─────────────────────────────────────────────────────────────────────────┘
              │ POST /api/chat/stream       │ POST /api/chat/stream
              ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          backend/chat                                   │
│                                                                         │
│   ChatService.generate_background(…) — UNCHANGED core                   │
│      ├─ inline-file branch (FR-12.6)                                    │
│      └─ RAG branch → RagService.make_scoped_retriever(…)                │
│                       │                                                 │
│                       ▼                                                 │
│             ScopedRetriever → hits with new metadata                     │
└─────────────────────────────────────────────────────────────────────────┬─┘
                                                                          │
                                  ┌───────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          backend/rag  (EXTENDED)                        │
│                                                                         │
│   RagService                                                            │
│      ├─► ingest_file(conv_id, path)                                     │
│      │       └─► split_into_documents(path, "upload", conv_id, …)       │
│      ├─► reindex_library()                                              │
│      │       └─► for each file: split_into_documents(…, "library", …)   │
│      ├─► make_scoped_retriever(conv_id, top_k)                          │
│      ├─► library_files()                  ─► new                         │
│      ├─► save_library_file(filename, bytes)  ─► new                     │
│      ├─► delete_library_file(filename)        ─► new                     │
│      └─► stats()                                                      │
│                                                                         │
│   loaders/                            splitter.py                       │
│      ├─ text.py  (txt, md)            ├─ pick_splitter(ext)            │
│      ├─ pdf.py                        └─ split_into_documents(…)        │
│      ├─ html.py                                                          │
│      ├─ doc.py                                                           │
│      └─ csv.py                                                           │
│         │                                                                │
│         └─► REGISTRY[ext] → loader(path, source) → Iterator[RawDocument]│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Loader registry

```
backend/rag/loaders/
├── __init__.py          # RawDocument, REGISTRY, load() dispatcher, UnsupportedFormatError
├── text.py              # .txt, .md — utf-8 read; md wraps with header_path detection
├── pdf.py               # .pdf — pypdf page-by-page, page_number metadata
├── html.py              # .html — BeautifulSoup get_text(); captures <title>
├── docx.py              # .docx — python-docx paragraphs; style hints as metadata
└── csv.py               # .csv — csv.reader rows; row_number + first-row headers
```

Each loader exports a single function with signature:

```python
def load(path: Path, source: str) -> Iterator[RawDocument]: ...
```

### Splitting pipeline

`backend/rag/splitter.py` becomes:

- `pick_splitter(extension: str, chunk_size: int, chunk_overlap: int) -> TextSplitter` — picks `MarkdownTextSplitter` for `.md`, `RecursiveCharacterTextSplitter` otherwise.
- `split_into_documents(path, source_type, conversation_id, chunk_size, chunk_overlap) -> Iterator[Document]` — dispatches to the loader, then per `RawDocument` runs the splitter and emits chunks with propagated metadata.

`RagService.ingest_file` and `RagService.reindex_library` shrink to call `split_into_documents`.

---

## 3. Component Details

### 3.1 `RawDocument` and `REGISTRY`

```python
# backend/rag/loaders/__init__.py
from dataclasses import dataclass, field

@dataclass
class RawDocument:
    text: str
    metadata: dict = field(default_factory=dict)

class UnsupportedFormatError(Exception):
    pass

REGISTRY: dict[str, "LoaderFn"] = {}  # populated by loader modules

def load(path: Path, source: str) -> Iterator[RawDocument]:
    ext = path.suffix.lower()
    loader = REGISTRY.get(ext)
    if loader is None:
        raise UnsupportedFormatError(ext)
    yield from loader(path, source)
```

### 3.2 Per-format loaders

**`backend/rag/loaders/text.py`** — `.txt` and `.md`:

```python
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _detect_md_header_path(text: str, offset: int) -> str:
    """Walk back from `offset` to find the most recent markdown header.
    Used by the splitter to attach `header_path` to each chunk."""
    # Header regex: ^#{1,6}\s+(.+)$
    # Walk from offset upward; collect latest of each level
    ...

def load(path: Path, source: str):
    text = _read_text(path)
    if path.suffix.lower() == ".md":
        # Single RawDocument for markdown; splitter preserves structure via MarkdownTextSplitter
        yield RawDocument(text=text, metadata={"format": ".md"})
    else:
        yield RawDocument(text=text, metadata={"format": ".txt"})
```

(MD header_path is computed by the splitter after chunks exist, not by the loader — see §3.3.)

**`backend/rag/loaders/pdf.py`** — `.pdf`:

```python
from pypdf import PdfReader

def load(path: Path, source: str):
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield RawDocument(
            text=text,
            metadata={
                "format": ".pdf",
                "page_number": i,
                "total_pages": total_pages,
            },
        )
```

**`backend/rag/loaders/html.py`** — `.html`:

```python
from bs4 import BeautifulSoup

def load(path: Path, source: str):
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    # Strip script/style
    for tag in soup(["script", "style"]):
        tag.decompose()
    title = (soup.title.string if soup.title else None) or ""
    yield RawDocument(
        text=soup.get_text(separator="\n", strip=True),
        metadata={"format": ".html", "title": title.strip()},
    )
```

**`backend/rag/loaders/docx.py`** — `.docx`:

```python
from docx import Document as DocxDocument

def load(path: Path, source: str):
    doc = DocxDocument(str(path))
    for i, para in enumerate(doc.paragraphs, start=1):
        text = para.text
        if not text.strip():
            continue
        style = para.style.name if para.style else ""
        yield RawDocument(
            text=text,
            metadata={
                "format": ".docx",
                "paragraph_number": i,
                "style": style,
            },
        )
```

**`backend/rag/loaders/csv.py`** — `.csv`:

```python
import csv

def load(path: Path, source: str):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return
    headers = rows[0]
    for i, row in enumerate(rows[1:], start=1):
        # Format each row as "header1: value1\nheader2: value2\n..."
        cells = []
        for h, v in zip(headers, row):
            h = h.strip() or "(unnamed)"
            cells.append(f"{h}: {v.strip()}")
        yield RawDocument(
            text="\n".join(cells),
            metadata={
                "format": ".csv",
                "row_number": i,
                "headers": headers,
            },
        )
```

### 3.3 Refactored splitter

```python
# backend/rag/splitter.py
import hashlib
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from backend.rag.loaders import RawDocument, UnsupportedFormatError, load as registry_load


def pick_splitter(extension: str, chunk_size: int, chunk_overlap: int) -> TextSplitter:
    if extension == ".md":
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _md_header_path(text: str, chunk_offset: int) -> str:
    """Return the markdown header path at chunk_offset, e.g. '## Section / ### Sub'.
    Walk backward from chunk_offset collecting the most recent header at each level.
    Returns "" if no header precedes the chunk.
    """
    import re
    headers: dict[int, str] = {}  # level -> text
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", text[:chunk_offset], re.MULTILINE):
        level = len(m.group(1))
        headers[level] = m.group(2).strip()
        # Drop deeper levels so we keep a single breadcrumb path
        for deeper in list(headers.keys()):
            if deeper > level:
                headers.pop(deeper)
    if not headers:
        return ""
    # Render as a breadcrumb: H1 > H2 > H3 in ascending level order
    parts = [headers[k] for k in sorted(headers)]
    return " / ".join(parts)


def split_into_documents(
    path: Path,
    source_type: str,        # "library" | "upload"
    conversation_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[Document]:
    ext = path.suffix.lower()
    splitter = pick_splitter(ext, chunk_size, chunk_overlap)
    raw_text_for_md = None  # Only set for .md so header_path can be computed
    full_text = ""
    if ext == ".md":
        full_text = path.read_text(encoding="utf-8")

    cursor = 0
    for raw in registry_load(path, source_type):
        if not raw.text.strip():
            continue
        chunks = splitter.split_text(raw.text)
        for chunk_text in chunks:
            meta = dict(raw.metadata)  # copy per-format metadata
            meta["source"] = source_type              # iter-7 compat
            meta["source_type"] = source_type          # new explicit name
            meta["filename"] = path.name
            meta["format"] = ext
            if conversation_id is not None:
                meta["conversation_id"] = conversation_id
            if ext == ".md":
                # Find this chunk's offset in full_text and derive header path
                idx = full_text.find(chunk_text[:80], cursor)
                if idx >= 0:
                    meta["header_path"] = _md_header_path(full_text, idx)
                    cursor = idx + len(chunk_text)
            meta["chunk_id"] = hashlib.sha256(
                f"{path.name}:{chunk_text}".encode()
            ).hexdigest()[:16]
            yield Document(page_content=chunk_text, metadata=meta)
```

**Compatibility note**: `chunk_id` formula changes from `sha256(chunk_text)` to `sha256(f"{path.name}:{chunk_text}")`. Two chunks from different files with identical text now have distinct `chunk_id`s. Reindexing produces new IDs but that's expected — IDs are content-addressed within a file.

### 3.4 `RagService` changes

`ingest_file` and `reindex_library` shrink to:

```python
def ingest_file(self, conversation_id: str, file_path: Path) -> list[str]:
    conv_uploads = self.uploads_dir / conversation_id
    conv_uploads.mkdir(parents=True, exist_ok=True)
    dest = conv_uploads / file_path.name
    if file_path.resolve() != dest.resolve():
        shutil.copy2(file_path, dest)

    docs = list(split_into_documents(
        dest, source_type="upload", conversation_id=conversation_id,
        chunk_size=self.settings.rag_chunk_size,
        chunk_overlap=self.settings.rag_chunk_overlap,
    ))
    self.uploads_index = rebuild_filtered(self.uploads_index, self.embeddings, keep=lambda d: True)
    self.uploads_index.add_documents(docs)
    save(self.uploads_index, self._index_path("uploads_index"))
    return [d.metadata["chunk_id"] for d in docs]

def reindex_library(self) -> dict:
    files = _walk_library(self.library_dir)
    errors: list[str] = []
    all_docs: list[Document] = []
    for path in files:
        try:
            all_docs.extend(split_into_documents(
                path, source_type="library", conversation_id=None,
                chunk_size=self.settings.rag_chunk_size,
                chunk_overlap=self.settings.rag_chunk_overlap,
            ))
        except UnsupportedFormatError:
            # _walk_library already filters by extension; this is defensive
            continue
        except Exception as e:
            errors.append(f"{path}: {e}")

    if not all_docs:
        self.library_index = FAISS.from_documents([_placeholder_doc()], self.embeddings)
    else:
        self.library_index = FAISS.from_documents(all_docs, self.embeddings)
    save(self.library_index, self._index_path("library_index"))
    return {"files_processed": len(files), "chunks_added": len(all_docs), "errors": errors}
```

`__init__` adds `self.library_dir.mkdir(parents=True, exist_ok=True)` so the directory exists from first startup.

### 3.5 New methods on `RagService`

```python
def list_library_files(self) -> list[dict]:
    """Return metadata for every file in the library directory matching the
    allowlist. Sorted alphabetically."""
    if not self.library_dir.exists():
        return []
    files = []
    for p in sorted(self.library_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        stat = p.stat()
        files.append({
            "filename": p.name,
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
        })
    return files

def save_library_file(self, filename: str, content: bytes) -> Path:
    """Write `content` to storage/library/<filename>. Returns the saved path.
    Caller is responsible for validating filename / extension."""
    self.library_dir.mkdir(parents=True, exist_ok=True)
    dest = self.library_dir / filename
    dest.write_bytes(content)
    return dest

def delete_library_file(self, filename: str) -> bool:
    """Delete <filename> from library, run reindex. Returns True if the file existed."""
    target = self.library_dir / filename
    if not target.exists() or not target.is_file():
        return False
    target.unlink()
    self.reindex_library()
    return True
```

`ALLOWED_EXTENSIONS` lives in `backend/rag/routes.py` (already exists for the upload endpoint) and is imported by the loader registry. Loaders register their extension in `REGISTRY`; `ALLOWED_EXTENSIONS` is the union of keys in `REGISTRY`.

### 3.6 New routes

```python
# backend/rag/routes.py

@router.get("/library/files")
def library_files():
    svc = _service_or_503()
    return {"files": svc.list_library_files()}

@router.post("/library/upload")
def library_upload(file: UploadFile = File(...)):
    svc = _service_or_503()
    suffix = _check_extension(file.filename)
    if (svc.library_dir / file.filename).exists():
        raise HTTPException(409, detail=f"'{file.filename}' is already in the library")
    try:
        content = file.file.read()
        path = svc.save_library_file(file.filename, content)
    except OSError as e:
        raise HTTPException(500, detail=f"Could not save file: {e}")
    return {"filename": file.filename, "size": path.stat().st_size, "saved": True}

@router.delete("/library/file/{filename}")
def library_file_delete(filename: str):
    svc = _service_or_503()
    # Path-traversal and dotfile guards
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, detail="Invalid filename")
    _check_extension(filename)
    if not svc.delete_library_file(filename):
        raise HTTPException(404, detail="Not found")
    return {"deleted": True, "filename": filename}
```

`POST /api/rag/library/reindex` stays unchanged.

---

## 4. Frontend Changes

### 4.1 Sidebar tabs (B1)

`index.html` sidebar gains a tab strip above the conversation list:

```html
<div class="sidebar">
    <div class="sidebar-tabs">
        <button class="sidebar-tab active" data-tab="conversations">Conversations</button>
        <button class="sidebar-tab" data-tab="library">Library</button>
    </div>
    <div class="sidebar-header" id="sidebarHeader"></div>
    <div class="conversation-list" id="conversationList"></div>
    <div class="library-view" id="libraryView" hidden>
        <!-- populated by renderLibraryView() -->
    </div>
</div>
```

The Library tab is only rendered when `cache.getRagEnabled() === true` (otherwise `display: none` — the RAG feature isn't on, library isn't useful).

### 4.2 Library tab contents

`renderLibraryView()` fetches `/api/rag/library/files` and `/api/rag/stats`, then renders:

```
┌─ Library ──────────────────────────────┐
│ [Upload] [Reindex]                     │
├─────────────────────────────────────────┤
│ 📄 intro.md          12 KB   2 days ago  ×│
│ 📄 reference.pdf     2.1 MB  1 hour ago  ×│
│ 📄 notes.docx        8 KB    just now    ×│
├─────────────────────────────────────────┤
│ 17 chunks from 3 files                  │
└─────────────────────────────────────────┘
```

Empty state: `No files in library. Click Upload to add one.`

Upload: `<input type="file" accept=".md,.txt,.pdf,.html,.docx,.csv">` + `POST /api/rag/library/upload` (multipart). On success: refresh list. On error: inline error message under the upload button.

Delete: per-row `×` button → `confirm` (themed modal, iter-7) → `DELETE /api/rag/library/file/{filename}` → refresh list + stats.

Reindex: button → `POST /api/rag/library/reindex` → on success, refresh stats. Errors render inline.

Tab switch is local state — does not affect the active conversation. Switching to Library does not abort an in-flight chat stream.

### 4.3 Show sources toggle (B2)

RAG column header (`index.html`):

```html
<section class="column rag-column" id="ragColumn" data-channel="rag">
    <div class="column-header">
        <span class="column-title">RAG</span>
        <label class="show-sources-toggle">
            <input type="checkbox" id="showSourcesToggle" checked>
            Show sources
        </label>
        <label class="upload-btn" title="Upload a file to the RAG conversation ...">
            Upload
            <input type="file" class="upload-input" data-column="rag" accept=".md,.txt,.pdf,.html,.docx,.csv">
        </label>
    </div>
    <div class="column-messages" id="ragMessages"></div>
</section>
```

`app.js` — `renderSourcesBlock` becomes a no-op when `cache.getShowSources() === false`:

```javascript
function renderSourcesBlock(assistantMessageEl, sources) {
    if (!assistantMessageEl || !sources || sources.length === 0) return;
    if (!cache.getShowSources()) return;  // ← NEW
    // ... existing render unchanged ...
}
```

Toggle change handler:

```javascript
showSourcesToggle.addEventListener('change', (e) => {
    cache.setShowSources(e.target.checked);
    // Re-render the RAG column's last assistant message if it has a sources block
    rerenderLastSourcesBlock();
});
```

`cache.js` adds:

```javascript
getShowSources() {
    const raw = readString('showSources');
    return raw === null ? true : raw === 'true';   // default ON
},
setShowSources(value) {
    writeString('showSources', value ? 'true' : 'false');
},
```

---

## 5. Configuration

`requirements.txt` gains:

```
python-docx>=1.0.0
beautifulsoup4>=4.12.0
```

Both are pure-Python, no native deps. CSV parsing uses stdlib (`csv` module). PDF parsing uses the existing `pypdf`. Markdown splitting uses the existing `langchain-text-splitters` (`MarkdownTextSplitter` is in the same package as `RecursiveCharacterTextSplitter`).

No new env vars. The `RAG_*` settings (`rag_chunk_size`, `rag_chunk_overlap`) continue to govern chunking uniformly across formats.

`.env.example` is updated to include a documented `RAG_*` block — currently it lists only `ANTHROPIC_*`, which hides the feature from new users.

---

## 6. Error Handling

| Failure | Behavior |
|---|---|
| Library upload: extension not in allowlist | HTTP 400 (same path as upload-to-conversation) |
| Library upload: filename already exists | HTTP 409 "delete first" |
| Library upload: filename contains `/`, `\`, or starts with `.` | HTTP 400 (path-traversal guard on delete and upload) |
| Library upload: disk write fails | HTTP 500, library dir unchanged |
| Library delete: file not found | HTTP 404 |
| Library delete: reindex fails after delete | HTTP 500, file gone; next manual reindex recovers. Critical log. |
| Loader for an extension raises (corrupt DOCX, malformed CSV) | `reindex_library` / `ingest_file` reports in `errors`, continues. Returns the error list. |
| Loader returns empty text (PDF blank page, CSV empty row) | Skipped — no empty Document emitted. |
| `split_into_documents` called with extension not in `REGISTRY` | `UnsupportedFormatError` raised; caller surfaces as 400 / error. |
| HTML: malformed markup | BeautifulSoup is permissive; degrades gracefully. |
| Frontend library upload: backend returns non-2xx | Inline error in the library tab; file not added to list. |
| Show-sources toggle: localStorage corrupted (non-bool string) | Treated as `false` defensively; renderSourcesBlock no-ops. User re-toggle restores. |

**Atomicity principle**: a per-file library upload is atomic — write-or-not. Reindex is all-or-nothing at the FAISS index level; if any file fails to parse, the others still ingest, and the index is rebuilt from the parsed set.

---

## 7. Testing

### Unit tests — loaders (`backend/tests/rag/test_loaders.py`, NEW)

Fixtures in `backend/tests/rag/fixtures/`:
- `sample.md` — H1 + H2 + fenced code block + numbered list
- `sample.txt` — plain text
- `sample.pdf` — 3 pages, page 2 sparse
- `sample.html` — `<title>Sample</title>`, `<script>`, `<style>`, nested `<div>`s
- `sample.docx` — 2 paragraphs, one with `Heading 2` style
- `sample.csv` — 3 rows + header

Tests:
- For each loader: yields correct `RawDocument` count; text non-empty; metadata fields populated.
- Empty PDF: yields no RawDocument.
- CSV with quoted fields: cells parse correctly.
- HTML with `<script>`/`<style>`: content stripped.
- DOCX with empty paragraphs: filtered out.

### Splitter tests (`backend/tests/rag/test_splitter.py`, EXTEND existing)

- `pick_splitter(".md", …)` returns `MarkdownTextSplitter`; everything else returns `RecursiveCharacterTextSplitter`.
- `split_into_documents` propagates `RawDocument` metadata to each chunk.
- Empty `RawDocument` filtered out.
- `UnsupportedFormatError` raised for `.xyz`.
- MD: chunks contain `header_path` matching the section.
- PDF: each chunk's `page_number` is present and monotonic.

### Service tests (`backend/tests/rag/test_service.py`, EXTEND existing)

- `RagService()` constructor auto-creates `library_dir`.
- `ingest_file` with each new format produces Documents with correct `format` and `source`.
- `reindex_library` with mixed-format fixtures produces correct chunks.
- `list_library_files` returns sorted metadata; ignores non-allowlist files.
- `save_library_file` writes to disk and returns the path.
- `delete_library_file` removes the file and triggers reindex; returns False on missing.

### Route tests (`backend/tests/rag/test_routes.py`, EXTEND existing)

- `POST /library/upload`: 200 happy path; 400 bad ext; 409 duplicate; 400 path-traversal.
- `GET /library/files`: 200 with correct shape, sorted alphabetically.
- `DELETE /library/file/{name}`: 200 happy path; 404 missing; 400 bad name; 400 bad ext.
- All endpoints 503 when RAG disabled.

### Integration tests (`backend/tests/test_chat_rag_integration.py`, EXTEND)

- Upload a DOCX to a conversation; verify retrieval produces chunks with `format = ".docx"`.
- Upload a CSV; verify `row_number` metadata is present on each chunk.
- Sources SSE event from a PDF upload includes `page_number` per source.

### Iter-7 compatibility

- All iter-7 tests must continue to pass unchanged.
- `metadata["source"]` is preserved on every chunk (was the iter-7 key for filter logic).
- `metadata["source_type"]` is the new explicit name; both coexist.
- `metadata["chunk_id"]` formula changes (see §3.3); existing tests that don't assert specific chunk_ids are unaffected. Tests that do are updated.

### Frontend — manual v1

1. Sidebar tab switch works; Library tab shows "no files" empty state when empty.
2. Upload `notes.md` to library → file appears with correct size.
3. Click Reindex → stats footer updates from "0 chunks" to "N chunks".
4. Open a fresh chat, ask a question whose answer is in the library doc → RAG column answer references it.
5. Delete a library file → list refreshes; chunk count drops on next reindex.
6. Toggle "Show sources" off in RAG column → sources blocks disappear; toggle back on → re-appear (and the toggle persists across page reload).
7. Upload `bad.xyz` → inline error; file not added.
8. Library tab hidden when `RAG_ENABLED=false`.

---

## 8. Module Layout

### New files

```
backend/rag/loaders/
├── __init__.py
├── text.py
├── pdf.py
├── html.py
├── docx.py
└── csv.py

backend/tests/rag/fixtures/
├── sample.md
├── sample.txt
├── sample.pdf
├── sample.html
├── sample.docx
└── sample.csv

backend/tests/rag/test_loaders.py    (new)
```

### Modified files

| File | Change |
|---|---|
| `backend/rag/loaders/__init__.py` | NEW — RawDocument, REGISTRY, load() dispatcher, UnsupportedFormatError |
| `backend/rag/loaders/text.py` | NEW — txt/md loader |
| `backend/rag/loaders/pdf.py` | NEW — pdf loader (moves page-by-page logic out of `_read_text`) |
| `backend/rag/loaders/html.py` | NEW — html loader with BeautifulSoup |
| `backend/rag/loaders/docx.py` | NEW — docx loader |
| `backend/rag/loaders/csv.py` | NEW — csv loader |
| `backend/rag/splitter.py` | `pick_splitter` + `split_into_documents`. `_read_text` and `_walk_library` move to loaders / routes respectively. |
| `backend/rag/service.py` | `ingest_file` and `reindex_library` use `split_into_documents`. `__init__` auto-creates `library_dir`. New `list_library_files`, `save_library_file`, `delete_library_file`. |
| `backend/rag/routes.py` | `ALLOWED_EXTENSIONS` becomes `REGISTRY.keys()`. Three new endpoints. `_check_extension` reused. |
| `requirements.txt` | Add `python-docx`, `beautifulsoup4`. |
| `.env.example` | Document `RAG_*` env vars (cosmetic — values stay defaults). |
| `frontend/index.html` | Sidebar tab strip + library-view container; RAG column header gains "Show sources" checkbox; upload accept attribute gains `.docx,.csv`. |
| `frontend/static/app.js` | `renderLibraryView()`, upload/reindex/delete handlers; `renderSourcesBlock` checks `cache.getShowSources()`; toggle change handler. |
| `frontend/static/cache.js` | `getShowSources` / `setShowSources`. |

### Unchanged files

- `backend/chat/chain.py` — same as iter-7.
- `backend/chat/routes.py` — `RetrievalConfig` and `UploadedFile` unchanged.
- `backend/chat/service.py` — `generate_background` unchanged. Sources event payload gains optional `format` field but it's additive.
- `backend/main.py` — startup wiring unchanged. RAG routes already mounted under `/api/rag`; the new endpoints register on the same router.

---

## 9. Migration & Rollback

- New pip deps (`python-docx`, `beautifulsoup4`) are pure-Python, no native bindings. `pip install -r requirements.txt` is sufficient.
- Existing iter-7 indexes load unchanged. The new metadata fields are additive; the FAISS docstore is read back via `docstore._dict` and missing fields are simply absent (downstream code reads with `.get()`).
- If a deploy needs to roll back: revert `requirements.txt` first, then the code. Old loader code (`_read_text`) can stay as `_legacy_read_text` for one release as a fallback.
- The `chunk_id` formula change is the only semantic change in the data layer. Since iter-7 tests don't assert specific chunk_id values (they assert presence and count), they continue to pass.

---

## 10. Out of Scope (deferred)

- PPTX, XLSX, RTF, EPUB (loader registry is ready to absorb these later)
- Auto-reindex after library upload
- Per-file reindex (incremental)
- Versioned library (undo, diff)
- Inline-files persistence across server restart
- Cross-encoder reranking
- Background ingestion
- Per-conversation FAISS indexes
- Chunk deduplication during reindex
- Markdown link-target extraction, code-block language detection
- OCR for images
- Web URL ingestion

---

## 11. Open Questions

None at design time. Library reindex semantics (whole-index, not incremental) were decided in §3.4 / §6; auto-reindex-after-upload was explicitly out-of-scope per §1.