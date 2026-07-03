# Chatbot Project — Iteration 8 Design (Multi-Format RAG, Library Mgmt, Show Sources)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> The architecture decisions and module-level design for iteration 8. See [SPEC_focus.md](SPEC_focus.md) for requirements and [docs/superpowers/specs/2026-07-03-rag-format-library-and-sources-design.md](../docs/superpowers/specs/2026-07-03-rag-format-library-and-sources-design.md) for the full brainstorming artifact.

## 1. Architecture Decisions

### 1.1 Loader Registry over Switch-on-Extension

**Choice**: `backend/rag/loaders/` is a package; each format lives in its own module exposing a `load(path, source) -> Iterator[RawDocument]` function. Loaders self-register via a `@register(".ext")` decorator at import time. `ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())` is the single source of truth for the allowlist.

**Rationale**:
- Adding `.docx` / `.csv` / future formats = new module + import in `loaders/__init__.py`. No edits to `routes.py`, `service.py`, or `splitter.py`.
- The dispatcher is a 4-line `load(path, source)` function with a clean `UnsupportedFormatError` boundary.
- Per-format parsing code (pypdf, BeautifulSoup, python-docx, csv.reader) is contained in one file each, easy to test in isolation.

**Trade-off**: One more directory level in the import graph. Mitigated by the consistent module API (`load(path, source) -> Iterator[RawDocument]`).

### 1.2 Format-Aware Splitter Selection

**Choice**: `pick_splitter(extension)` returns `MarkdownTextSplitter` for `.md` and `RecursiveCharacterTextSplitter` for everything else.

**Rationale**:
- Markdown has structure (headers, code blocks, lists) that should be preserved across chunk boundaries. `MarkdownTextSplitter` understands `# / ## / ###` and fenced code blocks, so splits don't slice mid-header or mid-code-block.
- All other formats (text, PDF page text, HTML stripped text, DOCX paragraph, CSV row) are linear character sequences — `RecursiveCharacterTextSplitter`'s paragraph/sentence/word fallback is correct.
- Both splitters take the same `chunk_size` / `chunk_overlap` knobs from `RagSettings`, so chunk-size policy stays uniform across formats.

**Trade-off**: One-line per-format branch in `pick_splitter`. If a future format needs custom chunking (e.g., CSV by-row rather than character-split), the loader can return one RawDocument per row and the splitter won't split it further because each row is small. Already handled — no special case needed.

### 1.3 RawDocument with Per-Unit Metadata

**Choice**: Each `RawDocument` carries `text: str` + `metadata: dict`. Loaders attach format-specific keys (`page_number`, `header_path`, `title`, `row_number`, etc.). The splitter propagates these into each chunk Document.

**Rationale**:
- "Uniform output format" is a hard requirement (FR-25.4). Every chunk has the same metadata shape, regardless of source format.
- The propagation is mechanical: `meta = dict(raw.metadata); meta.update({source, source_type, filename, format, chunk_id, conversation_id?})`. No format-specific branches in the splitter.
- Downstream code (retriever, sources SSE event, frontend sources block) reads these via `metadata.get(...)` and works on any format.

**Trade-off**: The metadata dict grows as formats are added. For 6 formats with ~2 keys each, this is small. If metadata explodes (10+ formats with complex nested structures), consider a typed `ChunkMetadata` dataclass — but that's premature.

### 1.4 Markdown `header_path` via First-Occurrence Find

**Choice**: For each Markdown chunk, the splitter finds the first occurrence of `chunk_text[:80]` in the original file text (no cursor — single find per chunk), then walks back from that offset to find the most recent header at each level. Joins the headers with `" / "`.

**Rationale**:
- A cursor-based approach (advance after each chunk) is fragile — it loses position if `chunk_text[:80]` is found at the wrong offset due to repeated phrases, and never advances on a miss.
- First-occurrence find is correct (the chunk came from the file, so its text exists somewhere) and O(n) per chunk, which is fine for files under 1 MB.
- The walk-back produces a breadcrumb like `"# Intro / ## Setup / ### Install"` for any chunk below that header hierarchy. Chunks before any header get `header_path = ""`.

**Trade-off**: For a 1 MB markdown file with 100 chunks, the splitter does 100 O(n) finds + 100 header-walks. ~200 ms total. Acceptable for the library ingestion path (one-time, not on the hot retrieval path).

### 1.5 `chunk_id` Formula Change (with Iter-7 Compat)

**Choice**: `chunk_id = sha256(f"{path.name}:{chunk_text}")` instead of `sha256(chunk_text)`.

**Rationale**:
- Two chunks from different files with identical text should be distinguishable (e.g., `intro.md` and `reference.md` both containing `"FAQ"`). Old formula collapses them.
- The new formula is content-addressed within a file. Reindexing the same file produces the same chunk_ids (idempotency preserved).

**Trade-off**: Iter-7 tests that assert specific `chunk_id` values must be updated. The plugin-off guarantee (no RAG in vanilla path) is unaffected — `chunk_id` is only set during ingestion, never during retrieval.

### 1.6 Auto-Reindex on Library Upload (User-Approved Resolution)

**Choice**: After a successful `save_library_file`, `RagService.reindex_library()` runs automatically. Same for `delete_library_file`. The typical upload flow is one-step: file is on disk AND in FAISS, queryable immediately.

**Rationale**:
- User feedback during planning: "Auto-reindex on upload" chosen over "dirty flag + manual reindex" and "debounced auto-reindex".
- Removes the UX footgun where a user uploads a file, asks a question, gets no retrieval, doesn't know they need to click Reindex.
- For small libraries (the typical v1 case, 10-100 files), full reindex after each upload takes <500 ms. Acceptable.
- The manual `POST /api/rag/library/reindex` endpoint stays as a force-refresh / recovery action.

**Trade-off**: Slow if the library grows large. The library index is rebuilt from scratch on every upload — for 1000 files / 10k chunks, this is ~4 seconds. Mitigations for future work: incremental reindex (only re-embed the new file's chunks and add to the existing index), or debounced auto-reindex.

### 1.7 Atomic Library File Write via `tmp + os.replace`

**Choice**: `save_library_file` writes to `<dest>.tmp.<rand>` first, then `os.replace(tmp, dest)`. The tmp file is cleaned up on failure.

**Rationale**:
- Same pattern as `backend/storage/file_storage.py`. Reused convention; no new code paths.
- `os.replace` is atomic on POSIX and Windows (within the same filesystem). A crash mid-write leaves the previous `dest` file fully intact.
- Without this, a power loss between `dest.write_bytes(content)`'s first and last byte leaves a half-written file that subsequent reindex would parse as garbage.

**Trade-off**: Doubles disk IO for uploads (write tmp, then rename). For typical small library files (KB to low MB), negligible.

### 1.8 Library Tab in Sidebar (Not Separate Page)

**Choice**: The Library management UI lives as a second tab inside the existing sidebar — `[Conversations]` / `[Library]` buttons at the top of the sidebar. Switching tabs is a local UI state change; the active conversation and any in-flight stream are unaffected.

**Rationale**:
- Sidebar is already the "list of conversations" container. Library is the "list of documents" — same UI shape, same place.
- A separate route (`/library`) would require new routing, new top-level nav, and would feel like a separate app. The tab approach is one-click and stays in-context.
- Tab choice persists in `localStorage` (`currentSidebarTab`) so reload returns the user to where they were.

**Trade-off**: Sidebar gets taller (two views stacked). Mitigated by the active tab being the only one rendered — `libraryView` is `hidden` when Conversations is active, and vice versa.

### 1.9 Show-Sources Toggle via CSS Visibility (Not Re-Render)

**Choice**: When the toggle is OFF, `renderSourcesBlock` early-returns so blocks aren't added to the DOM. The toggle change handler calls `applySourcesVisibility()` which walks every `.sources-block` element and toggles `style.display`. When the user toggles back ON, pre-existing blocks reappear.

**Rationale**:
- Re-rendering from cache is fragile: the cached chunks have the raw JSON, but rendering requires the same DOM template; keeping two code paths (initial render + re-render) invites drift.
- CSS `display: none` is instant, correct, and works for any number of pre-existing blocks.
- When OFF + new message streams in: `renderSourcesBlock` no-ops, so no block is added. Toggle back ON: the new message doesn't get a block, but the toggle UI is honest about what's currently visible.
- The toggle UI displays the *desired* state; the DOM reflects the *actual* state. They converge on every chunk arrival because each new chunk calls `renderSourcesBlock` which checks the cache.

**Trade-off**: New messages streamed while the toggle is OFF won't have a sources block even after the user toggles back ON. To recover, the user can re-send the question (creating a new message that calls `renderSourcesBlock` with the toggle now ON). For v1 this is acceptable.

### 1.10 Single Allowlist Single Source (FR-25.2 / NFR-11)

**Choice**: `ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())` defined in `backend/rag/loaders/__init__.py`. Imported by `routes.py` for the upload endpoint's extension check, by the frontend's `<input accept="...">` (must be kept in sync manually but trivially), and by tests.

**Rationale**:
- Adding `.docx` / `.csv` doesn't require editing `routes.py`'s allowlist — the loader module's `@register(".docx")` decorator does it.
- Frontend `<input accept>` is the one place that's not auto-synced (it's a browser-level hint, not server-enforced). The server-side `ALLOWED_EXTENSIONS` is the authoritative check.
- Tests assert against `ALLOWED_EXTENSIONS` directly — no string duplication in test fixtures.

**Trade-off**: One more import dependency. Worth it for the consistency gain.

---

## 2. Module Layout

### 2.1 New Files

```
backend/rag/loaders/
├── __init__.py          # RawDocument, REGISTRY, register, load, UnsupportedFormatError, ALLOWED_EXTENSIONS
├── text.py              # .txt, .md — utf-8 read
├── pdf.py               # .pdf — pypdf page-by-page
├── html.py              # .html — BeautifulSoup, strip <script>/<style>
├── docx.py              # .docx — python-docx paragraphs
└── csv.py               # .csv — csv.reader rows; row_number, headers

backend/tests/rag/fixtures/
├── sample.md            # H1 + H2 + fenced code + numbered list
├── sample.txt           # plain text
├── sample.pdf           # 3 pages, page 2 sparse
├── sample.html          # <title>, <script>, <style>, nested <div>
├── sample.docx          # 2 paragraphs, one with Heading 2 style
└── sample.csv           # 3 rows + header

backend/tests/rag/test_loaders.py    # NEW — one assertion per loader + edge cases
```

### 2.2 Per-File Responsibilities

**`backend/rag/loaders/__init__.py`** — `RawDocument` dataclass, `UnsupportedFormatError`, `REGISTRY` dict, `register(extension)` decorator, `load(path, source)` dispatcher, `ALLOWED_EXTENSIONS` derived from REGISTRY.

**`backend/rag/loaders/text.py`** — `@register(".txt")` and `@register(".md")`. Read UTF-8, yield one RawDocument.

**`backend/rag/loaders/pdf.py`** — `@register(".pdf")`. `pypdf.PdfReader`, iterate pages, yield one RawDocument per page with `page_number` and `total_pages` metadata.

**`backend/rag/loaders/html.py`** — `@register(".html")`. `BeautifulSoup(html, "html.parser")`, decompose `<script>` / `<style>`, get_text with `\n` separator and strip. Capture `<title>` as metadata.

**`backend/rag/loaders/docx.py`** — `@register(".docx")`. `docx.Document`, iterate `paragraphs`, yield one RawDocument per non-empty paragraph with `paragraph_number` and `style` metadata.

**`backend/rag/loaders/csv.py`** — `@register(".csv")`. `csv.reader`, infer headers from row 1, yield one RawDocument per data row formatted as `"header1: value1\nheader2: value2\n..."` with `row_number` and `headers` metadata.

**`backend/rag/splitter.py`** — `pick_splitter(extension)` (MarkdownTextSplitter for `.md`, RecursiveCharacterTextSplitter otherwise), `_md_header_path(full_text, offset)`, `split_into_documents(path, source_type, conversation_id, chunk_size, chunk_overlap)` (the new entry point).

**`backend/rag/service.py`** — `RagService.__init__` creates `library_dir`; `ingest_file` and `reindex_library` use `split_into_documents`; new methods `list_library_files`, `save_library_file` (atomic write + auto-reindex), `delete_library_file` (auto-reindex). `_walk_library` moves here from `splitter.py` (it's a service concern).

**`backend/rag/routes.py`** — `ALLOWED_EXTENSIONS` imported from `loaders`. Three new endpoints (`library_files`, `library_upload`, `library_file_delete`). Stats response gains `library_files`.

### 2.3 Modified Files

| File | Change |
|---|---|
| `backend/rag/loaders/__init__.py` | NEW |
| `backend/rag/loaders/text.py` | NEW |
| `backend/rag/loaders/pdf.py` | NEW |
| `backend/rag/loaders/html.py` | NEW |
| `backend/rag/loaders/docx.py` | NEW |
| `backend/rag/loaders/csv.py` | NEW |
| `backend/rag/splitter.py` | Add `pick_splitter`, `split_into_documents`, `_md_header_path`. Keep `make_splitter` and `_read_text` and `_walk_library` as deprecated thin wrappers that delegate to the new code (removed once nothing imports them). |
| `backend/rag/service.py` | `ingest_file` and `reindex_library` rewritten to use `split_into_documents`. New methods (`list_library_files`, `save_library_file`, `delete_library_file`). `_walk_library` moves here. `__init__` calls `library_dir.mkdir`. Add `if docs:` guard around `add_documents` (fixes empty-list crash bug). |
| `backend/rag/routes.py` | Import `ALLOWED_EXTENSIONS` from `loaders`. Add 3 new endpoints. Stats gains `library_files`. |
| `requirements.txt` | Add `python-docx`, `beautifulsoup4`. |
| `.env.example` | Document `RAG_*` env vars (cosmetic). |
| `backend/tests/rag/test_splitter.py` | Add format-aware tests (MarkdownTextSplitter for `.md`, RecursiveCharacterTextSplitter otherwise). |
| `backend/tests/rag/test_service.py` | Add tests for new methods (`list_library_files`, `save_library_file` atomic + auto-reindex, `delete_library_file`). Update `chunk_id` assertion to new formula. Add mixed-format reindex test. |
| `backend/tests/rag/test_routes.py` | Add tests for new endpoints (200/400/404/409 matrix). |
| `backend/tests/test_chat_rag_integration.py` | Add assertions that DOCX/CSV chunks have `format` and per-format metadata. |
| `backend/tests/rag/test_loaders.py` | NEW — one assertion per loader + edge cases. |
| `frontend/index.html` | Sidebar tabs (`[Conversations]` / `[Library]`); `<div id="libraryView" hidden>`; RAG column header gains `<input type="checkbox" id="showSourcesToggle">`; both columns' Upload `accept` attribute gains `.docx,.csv`. |
| `frontend/static/cache.js` | Add `KEYS.currentSidebarTab()`, `KEYS.showSources()`; `getCurrentSidebarTab / setCurrentSidebarTab`; `getShowSources / setShowSources`. |
| `frontend/static/app.js` | `renderActiveTab()` dispatcher; `renderLibraryView()` with Upload / Reindex / per-file delete / stats footer; upload/reindex/delete handlers; `applySourcesVisibility()`; show-sources toggle change handler. |

### 2.4 Unchanged Files

- `backend/chat/chain.py` — iter-7 chain, unmodified.
- `backend/chat/routes.py` — `ChatRequest`, `RetrievalConfig`, `UploadedFile` unchanged.
- `backend/chat/service.py` — `generate_background` unchanged. Sources SSE event payload format unchanged (still `{"sources": [{filename, excerpt, scope}]}`).
- `backend/main.py` — startup wiring unchanged. RAG routes already mounted under `/api/rag`; new endpoints register on the same router.
- `backend/storage/file_storage.py` — reuse its tmp+rename pattern; do not modify.

---

## 3. Component Skeletons

### 3.1 Loader Registry

```python
# backend/rag/loaders/__init__.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

LoaderFn = Callable[[Path, str], Iterator["RawDocument"]]

@dataclass
class RawDocument:
    text: str
    metadata: dict = field(default_factory=dict)

class UnsupportedFormatError(Exception):
    pass

REGISTRY: dict[str, LoaderFn] = {}

def register(extension: str):
    def _decorator(fn: LoaderFn) -> LoaderFn:
        REGISTRY[extension.lower()] = fn
        return fn
    return _decorator

def load(path: Path, source: str) -> Iterator[RawDocument]:
    ext = path.suffix.lower()
    loader = REGISTRY.get(ext)
    if loader is None:
        raise UnsupportedFormatError(ext)
    yield from loader(path, source)

# Self-registration: importing this module's children populates REGISTRY.
from backend.rag.loaders import text  # noqa: F401  (.txt, .md)

# Routes that want richer formats also import pdf, html, docx, csv.
ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())
```

### 3.2 Per-Format Loaders

**`backend/rag/loaders/text.py`**:
```python
from pathlib import Path
from typing import Iterator
from backend.rag.loaders import RawDocument, register

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

@register(".txt")
def load_txt(path: Path, source: str) -> Iterator[RawDocument]:
    yield RawDocument(text=_read(path), metadata={"format": ".txt"})

@register(".md")
def load_md(path: Path, source: str) -> Iterator[RawDocument]:
    yield RawDocument(text=_read(path), metadata={"format": ".md"})
```

**`backend/rag/loaders/pdf.py`**:
```python
from pathlib import Path
from typing import Iterator
from backend.rag.loaders import RawDocument, register

@register(".pdf")
def load_pdf(path: Path, source: str) -> Iterator[RawDocument]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield RawDocument(
            text=text,
            metadata={"format": ".pdf", "page_number": i, "total_pages": total},
        )
```

**`backend/rag/loaders/html.py`**:
```python
from pathlib import Path
from typing import Iterator
from backend.rag.loaders import RawDocument, register

@register(".html")
def load_html(path: Path, source: str) -> Iterator[RawDocument]:
    from bs4 import BeautifulSoup
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    title = (soup.title.string if soup.title else None) or ""
    yield RawDocument(
        text=soup.get_text(separator="\n", strip=True),
        metadata={"format": ".html", "title": title.strip()},
    )
```

**`backend/rag/loaders/docx.py`**:
```python
from pathlib import Path
from typing import Iterator
from backend.rag.loaders import RawDocument, register

@register(".docx")
def load_docx(path: Path, source: str) -> Iterator[RawDocument]:
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    for i, para in enumerate(doc.paragraphs, start=1):
        text = para.text
        if not text.strip():
            continue
        style = para.style.name if para.style else ""
        yield RawDocument(
            text=text,
            metadata={"format": ".docx", "paragraph_number": i, "style": style},
        )
```

**`backend/rag/loaders/csv.py`**:
```python
import csv
from pathlib import Path
from typing import Iterator
from backend.rag.loaders import RawDocument, register

@register(".csv")
def load_csv(path: Path, source: str) -> Iterator[RawDocument]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    headers = rows[0]
    for i, row in enumerate(rows[1:], start=1):
        cells = []
        for h, v in zip(headers, row):
            h = h.strip() or "(unnamed)"
            cells.append(f"{h}: {v.strip()}")
        yield RawDocument(
            text="\n".join(cells),
            metadata={"format": ".csv", "row_number": i, "headers": headers},
        )
```

### 3.3 Format-Aware Splitter

```python
# backend/rag/splitter.py
import hashlib
import re
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


def _md_header_path(full_text: str, offset: int) -> str:
    """Walk back from `offset` and return the most-recent header breadcrumb.
    E.g. "Intro / Setup / Install". Returns "" if no header precedes offset."""
    headers: dict[int, str] = {}
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", full_text[:offset], re.MULTILINE):
        level = len(m.group(1))
        headers[level] = m.group(2).strip()
        for deeper in list(headers.keys()):
            if deeper > level:
                headers.pop(deeper)
    if not headers:
        return ""
    return " / ".join(headers[k] for k in sorted(headers))


def split_into_documents(
    path: Path,
    source_type: str,        # "library" | "upload"
    conversation_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[Document]:
    ext = path.suffix.lower()
    splitter = pick_splitter(ext, chunk_size, chunk_overlap)
    full_text = path.read_text(encoding="utf-8") if ext == ".md" else ""

    for raw in registry_load(path, source_type):
        if not raw.text.strip():
            continue
        for chunk_text in splitter.split_text(raw.text):
            meta = dict(raw.metadata)
            meta["source"] = source_type              # iter-7 compat
            meta["source_type"] = source_type          # new explicit
            meta["filename"] = path.name
            meta["format"] = ext
            if conversation_id is not None:
                meta["conversation_id"] = conversation_id
            if ext == ".md":
                snippet = chunk_text[:80].strip()
                if snippet:
                    idx = full_text.find(snippet)
                    if idx >= 0:
                        meta["header_path"] = _md_header_path(full_text, idx)
            meta["chunk_id"] = hashlib.sha256(
                f"{path.name}:{chunk_text}".encode()
            ).hexdigest()[:16]
            yield Document(page_content=chunk_text, metadata=meta)


# ── Deprecated thin wrappers (kept for iter-7 tests; remove after migration) ──

def make_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    return pick_splitter(".txt", chunk_size, chunk_overlap)


def _read_text(path: Path) -> str:
    """Iter-7 compat: returns concatenated text from all RawDocuments."""
    return "\n".join(rd.text for rd in registry_load(path, "upload"))


def _walk_library(library_dir: Path) -> list[Path]:
    """Iter-7 compat: walks library_dir for allowlisted files. Implementation
    has moved to backend.rag.service._walk_library."""
    from backend.rag.loaders import ALLOWED_EXTENSIONS
    if not library_dir.exists():
        return []
    return sorted(
        p for p in library_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
```

### 3.4 RagService — Library Methods

```python
# backend/rag/service.py (additions)
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class RagService:
    def __init__(self, settings: RagSettings, embeddings: Embeddings):
        # ... existing init ...
        self.library_dir.mkdir(parents=True, exist_ok=True)  # ← NEW

    def list_library_files(self) -> list[dict]:
        from backend.rag.loaders import ALLOWED_EXTENSIONS
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
        """Atomically write content to library_dir/filename, then auto-reindex.
        Caller validates filename + extension."""
        from backend.rag.loaders import ALLOWED_EXTENSIONS
        self.library_dir.mkdir(parents=True, exist_ok=True)
        dest = self.library_dir / filename
        # Atomic write: tmp + os.replace
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.library_dir),
            prefix=f".{filename}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Auto-reindex (FR-27.6)
        try:
            self.reindex_library()
        except Exception:
            logger.exception("Auto-reindex after library upload failed for %s", filename)
        return dest

    def delete_library_file(self, filename: str) -> bool:
        target = self.library_dir / filename
        if not target.exists() or not target.is_file():
            return False
        target.unlink()
        # Auto-reindex (FR-27.7)
        try:
            self.reindex_library()
        except Exception:
            logger.exception("Auto-reindex after library delete failed for %s", filename)
        return True


def _walk_library(library_dir: Path) -> list[Path]:
    """Service-internal helper: walks library_dir for allowlisted files.
    Moved from splitter.py."""
    from backend.rag.loaders import ALLOWED_EXTENSIONS
    if not library_dir.exists():
        return []
    return sorted(
        p for p in library_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
```

### 3.5 Library Routes

```python
# backend/rag/routes.py (additions)

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
    content = file.file.read()
    try:
        path = svc.save_library_file(file.filename, content)
    except OSError as e:
        raise HTTPException(500, detail=f"Could not save file: {e}")
    return {"filename": file.filename, "size": path.stat().st_size, "saved": True}


@router.delete("/library/file/{filename}")
def library_file_delete(filename: str):
    svc = _service_or_503()
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, detail="Invalid filename")
    _check_extension(filename)
    if not svc.delete_library_file(filename):
        raise HTTPException(404, detail="Not found")
    return {"deleted": True, "filename": filename}
```

### 3.6 Stats Endpoint Update

```python
@router.get("/stats")
def stats():
    svc = _service_or_503()
    data = svc.stats()
    data["inline_context_threshold_bytes"] = RagSettings().rag_inline_context_threshold_bytes
    data["library_files"] = len(svc.list_library_files())   # ← NEW (FR-28.8)
    return data
```

---

## 4. Configuration

`requirements.txt` gains:

```
python-docx>=1.0.0
beautifulsoup4>=4.12.0
```

Both pure-Python, no native deps.

`.env.example` gains a `RAG_*` block documenting the existing env vars. Cosmetic — values stay at their defaults.

No new env vars in iteration 8. The `RAG_*` settings (`rag_chunk_size`, `rag_chunk_overlap`, `rag_inline_context_threshold_bytes`) continue to govern chunking and routing uniformly.

---

## 5. Error Handling

| Failure | Behavior |
|---|---|
| Library upload: extension not in allowlist | HTTP 400 (boundary check) |
| Library upload: filename already exists | HTTP 409 "delete first" |
| Library upload: filename contains `/`, `\`, or starts with `.` | HTTP 400 (path-traversal guard) |
| Library upload: write fails | HTTP 500, tmp file cleaned up, library dir unchanged |
| Library delete: file not found | HTTP 404 |
| Library delete: reindex fails after delete | File gone, next manual reindex recovers. Critical log. |
| Loader raises (corrupt DOCX, malformed CSV) | `errors` list captures, run continues with remaining files |
| Empty RawDocument (blank PDF page, empty CSV row) | Skipped — no empty Document emitted |
| `split_into_documents` called with extension not in REGISTRY | `UnsupportedFormatError` raised; caller surfaces as 400 |
| HTML: malformed markup | BeautifulSoup is permissive; degrades gracefully |
| Markdown chunk's text not found in full_text | `header_path` defaults to `""`; chunk is still emitted |
| Frontend library upload: backend non-2xx | Inline error in the library tab |
| Show-sources toggle: localStorage corrupted | Treated as default (ON); user re-toggle restores |

**Atomicity principle**: a per-file library upload is atomic via `tmp + os.replace`. Reindex is all-or-nothing at the FAISS index level — if any file fails to parse, the others still ingest.

---

## 6. Testing Strategy

### 6.1 Layers

| Layer | Files | Speed target |
|---|---|---|
| Loader unit | `test_loaders.py` (NEW) | <10 ms each |
| Splitter unit | `test_splitter.py` (extend) | <10 ms each |
| Service unit | `test_service.py` (extend) | <500 ms each |
| Route unit | `test_routes.py` (extend) | <200 ms each |
| Chain integration | `test_chat_rag_integration.py` (extend) | <1 s each |
| Manual UI | — | per checklist |

### 6.2 Key Test Cases

**Loader tests** (`test_loaders.py`):
- Each loader yields correct `RawDocument` count for a known fixture
- Empty PDF: yields no RawDocument
- CSV with quoted fields / multi-line cells: cells parse correctly
- HTML with `<script>` / `<style>`: content stripped from output
- DOCX with empty paragraphs: filtered out
- Markdown with H1 + H2 + fenced code: single RawDocument with full text
- Unsupported extension raises `UnsupportedFormatError`

**Splitter tests** (`test_splitter.py`):
- `pick_splitter(".md", ...)` returns `MarkdownTextSplitter`; others return `RecursiveCharacterTextSplitter`
- `split_into_documents` propagates RawDocument metadata to chunks
- Empty RawDocument filtered out
- `UnsupportedFormatError` raised for `.xyz`
- MD chunks contain `header_path` matching the section
- PDF chunks have `page_number` and `total_pages`
- `chunk_id` formula = `sha256(f"{path.name}:{chunk_text}")`

**Service tests** (`test_service.py`):
- `RagService()` constructor auto-creates `library_dir`
- `ingest_file` with each format produces Documents with correct `format` and `source`
- `ingest_file` with empty file (no chunks): no crash (`if docs:` guard)
- `reindex_library` with mixed-format fixtures produces correct chunks
- `list_library_files` returns sorted metadata, ignores non-allowlist files
- `save_library_file` writes atomically, auto-runs reindex
- `delete_library_file` removes the file and triggers reindex
- `save_library_file` writes partial file (simulate crash via mock): tmp file cleaned up

**Route tests** (`test_routes.py`):
- `POST /library/upload`: 200 happy path; 400 bad ext; 409 duplicate; 400 path-traversal; 400 dotfile
- `GET /library/files`: 200 with sorted list
- `DELETE /library/file/{name}`: 200 happy path; 404 missing; 400 bad name; 400 bad ext
- `/api/rag/stats` response includes `library_files: int`
- All endpoints 503 when RAG disabled

**Integration tests** (`test_chat_rag_integration.py`):
- Upload a DOCX; verify chunks have `format = ".docx"` and `paragraph_number`
- Upload a CSV; verify chunks have `format = ".csv"` and `row_number`
- Upload a PDF; verify sources event includes `page_number` per source
- Upload an HTML with `<script>` block; verify retrieval only sees visible text

### 6.3 Iter-7 Compatibility

- All iter-7 tests must pass unchanged (except those that assert the old `chunk_id` formula — those are updated to the new formula).
- `metadata["source"]` is preserved on every chunk.
- `metadata["source_type"]` is the new explicit name; both coexist.

### 6.4 Frontend — Manual v1

1. Sidebar tab switch works; Library tab shows "no files" empty state.
2. Upload `sample.md` to library → file appears with correct size; stats footer auto-updates.
3. Click Reindex → stats refresh.
4. Ask a question whose answer is in the library doc → RAG column answer references it.
5. Delete a library file → list refreshes; chunk count drops on next render.
6. Toggle "Show sources" off → sources blocks disappear. Toggle back on → reappear. Refresh page → state persists.
7. Upload `bad.xyz` → inline error; file not added.
8. Library tab hidden when `RAG_ENABLED=false`.
9. HTML quality: upload HTML with `<script>` block → retrieval references only visible text.
10. DOCX: upload `sample.docx` → chunks have `format = ".docx"`.
11. CSV: upload `sample.csv` → chunks have `format = ".csv"` and `row_number`.
12. Markdown `header_path`: upload `sample.md` → answer cites the right section.

---

## 7. Future Work (out of scope for iteration 8)

1. PPTX / XLSX / RTF / EPUB (loader registry is ready; add modules + register)
2. Per-file incremental reindex (currently whole-index on every upload)
3. Debounced auto-reindex (currently synchronous per upload)
4. Versioned library (undo, diff)
5. Inline-files persistence across server restart
6. Markdown link-target extraction, code-block language detection
7. OCR for images
8. Web URL ingestion
9. Cross-encoder reranking
10. Background ingestion (long uploads don't block the request)
11. Auth on library endpoints (when the app leaves local-only)
12. Chunk deduplication during reindex