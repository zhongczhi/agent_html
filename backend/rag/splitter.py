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

    Used by RagService.ingest_file and RagService.reindex_library."""
    ext = path.suffix.lower()
    splitter = pick_splitter(ext, chunk_size, chunk_overlap)
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
            meta["chunk_id"] = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            yield Document(page_content=chunk_text, metadata=meta)


# ── Deprecated (iter-7) wrappers — kept for backward compat ────────────────
# These delegate to the new loader registry where possible, preserving the
# behavior that iter-7 tests assert. They will be removed once service.py
# migrates to split_into_documents in iter-8 Phase B.

def make_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """Iter-7 compat: returns RecursiveCharacterTextSplitter regardless of
    extension. Service.py uses this directly for the legacy ingest path."""
    return pick_splitter(".txt", chunk_size, chunk_overlap)


def _read_text(path: Path) -> str:
    """Iter-7 compat: returns concatenated text from a file. PDF parsing
    stays inline (the loaders/pdf module is added in Phase B); .txt/.md
    delegate to the registered loaders."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return "\n".join(rd.text for rd in registry_load(path, "upload"))


def _walk_library(library_dir: Path) -> list[Path]:
    """Iter-7 compat: walks library_dir, returns allowlisted files sorted
    alphabetically. Imports ALLOWED_EXTENSIONS from the loaders package."""
    from backend.rag.loaders import ALLOWED_EXTENSIONS
    if not library_dir.exists():
        return []
    return sorted(
        p for p in library_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )