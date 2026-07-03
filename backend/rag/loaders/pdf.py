"""Loader for .pdf files. Phase A stub: concatenates all pages into a
single RawDocument. Phase B replaces this with a page-by-page loader
that emits one RawDocument per page with `page_number` and `total_pages`
metadata.
"""
from pathlib import Path
from typing import Iterator

from backend.rag.loaders import RawDocument, register


@register(".pdf")
def load_pdf(path: Path, source: str) -> Iterator[RawDocument]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    yield RawDocument(text=text, metadata={"format": ".pdf"})