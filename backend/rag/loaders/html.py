"""Loader for .html files. Phase A stub: reads as raw UTF-8. Phase B
replaces this with a BeautifulSoup-based loader that strips <script> /
<style> tags and captures <title> as metadata.
"""
from pathlib import Path
from typing import Iterator

from backend.rag.loaders import RawDocument, register


@register(".html")
def load_html(path: Path, source: str) -> Iterator[RawDocument]:
    yield RawDocument(
        text=path.read_text(encoding="utf-8"),
        metadata={"format": ".html"},
    )