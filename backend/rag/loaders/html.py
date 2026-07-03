"""Loader for .html files. Strips <script> and <style> blocks, then
extracts visible text via BeautifulSoup. Captures the <title> element
as metadata so downstream code can show "from page X titled Y" in
the sources block.
"""
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