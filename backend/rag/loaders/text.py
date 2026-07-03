"""Loader for .txt and .md files. Both formats are read as UTF-8 text and
yielded as a single RawDocument. The Markdown-specific metadata (header
breadcrumbs) is attached by the splitter, not here, because it depends on
chunk boundaries established after splitting.
"""
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