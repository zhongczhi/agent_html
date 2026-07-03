"""Loader for .pdf files. Emits one RawDocument per page so each chunk
carries `page_number` and `total_pages` metadata. A blank page yields
no RawDocument (the splitter filters empty text).
"""
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
            metadata={
                "format": ".pdf",
                "page_number": i,
                "total_pages": total,
            },
        )