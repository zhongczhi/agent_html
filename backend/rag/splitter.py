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


# ── New (iter-8) public surface ─────────────────────────────────────────────

def pick_splitter(extension: str, chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """Pick the splitter appropriate for an extension. Markdown uses
    MarkdownTextSplitter so splits respect header / code-block / list
    boundaries; everything else uses RecursiveCharacterTextSplitter."""
    if extension.lower() == ".md":
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _md_header_path(full_text: str, offset: int) -> str:
    """Return the markdown header breadcrumb at `offset`, e.g.
    "Intro / Setup / Install". Walks back from `offset` collecting the
    most recent header at each level; deeper levels drop out when a
    shallower one is found. Returns "" if no header precedes the chunk."""
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
    """Single entry point for the ingestion pipeline. Dispatches to the
    registered loader, runs the format-appropriate splitter on each
    RawDocument, and yields chunk Documents with propagated metadata.

    Used by RagService.ingest_file and RagService.reindex_library.

    Metadata fields guaranteed on every chunk:
        source         — "library" | "upload" (iter-7 compat)
        source_type    — same value as `source` (iter-8 explicit name)
        filename       — basename of the file
        format         — file extension including leading dot
        chunk_id       — sha256(f"{path.name}:{chunk_text}") prefix
        conversation_id — present only when conversation_id is not None

    Per-format metadata propagated from RawDocument:
        .md  → header_path
        .pdf → page_number, total_pages
        .html → title
        .docx → paragraph_number, style
        .csv → row_number, headers
    """
    ext = path.suffix.lower()
    splitter = pick_splitter(ext, chunk_size, chunk_overlap)

    # Markdown chunks need a header breadcrumb derived from the file's
    # original text. Read it once up front.
    full_text = path.read_text(encoding="utf-8") if ext == ".md" else ""

    for raw in registry_load(path, source_type):
        if not raw.text.strip():
            continue
        for chunk_text in splitter.split_text(raw.text):
            meta = dict(raw.metadata)
            meta["source"] = source_type
            meta["source_type"] = source_type
            meta["filename"] = path.name
            meta["format"] = ext
            if conversation_id is not None:
                meta["conversation_id"] = conversation_id
            if ext == ".md":
                # Find this chunk's offset in the original file text and
                # derive the header breadcrumb. Use first-occurrence find
                # (no cursor) so repeated phrases and overlap don't
                # misalign the offset.
                snippet = chunk_text[:80].strip()
                if snippet:
                    idx = full_text.find(snippet)
                    if idx >= 0:
                        meta["header_path"] = _md_header_path(full_text, idx)
            meta["chunk_id"] = hashlib.sha256(
                f"{path.name}:{chunk_text}".encode()
            ).hexdigest()[:16]
            yield Document(page_content=chunk_text, metadata=meta)


# ── Deprecated (iter-7) wrappers — kept for backward compat ────────────────
# These delegate to the new loader registry where possible, preserving the
# behavior that iter-7 tests assert. They will be removed once service.py
# migrates to split_into_documents.

def make_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """Iter-7 compat: returns RecursiveCharacterTextSplitter regardless of
    extension. Service.py uses this directly for the legacy ingest path."""
    return pick_splitter(".txt", chunk_size, chunk_overlap)


def _read_text(path: Path) -> str:
    """Iter-7 compat: returns concatenated text from a file. Delegates to
    the registered loader (which yields one RawDocument per page for
    PDFs, one for HTML/txt/md, etc.)."""
    return "\n".join(rd.text for rd in registry_load(path, "upload"))


def _walk_library(library_dir: Path) -> list[Path]:
    """Iter-7 compat: walks library_dir, returns allowlisted files sorted
    alphabetically. Implementation moved to backend.rag.service in
    iter-8 Phase D; this re-exports for backward compat with iter-7 tests."""
    from backend.rag.service import _walk_library as _service_walk_library
    return _service_walk_library(library_dir)