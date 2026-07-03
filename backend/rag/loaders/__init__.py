"""Format-aware document loaders for the RAG ingestion pipeline.

Each loader yields RawDocument objects containing the extracted text plus
format-specific metadata (page_number for PDFs, header_path for Markdown,
title for HTML, row_number for CSV, etc.). The splitter reads these and
chunks them, propagating the metadata to each chunk.

Adding a new format means:
  1. Write a `load(path, source) -> Iterator[RawDocument]` in a new module.
  2. Register it in REGISTRY below.

`ALLOWED_EXTENSIONS` is the union of REGISTRY keys — used by the HTTP
routes to reject unsupported extensions at the boundary.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

LoaderFn = Callable[[Path, str], Iterator["RawDocument"]]


@dataclass
class RawDocument:
    """One unit of extracted text from a file. For most formats this is the
    whole file; for paginated formats (PDF) or row-oriented formats (CSV)
    each unit is a single page or row."""
    text: str
    metadata: dict = field(default_factory=dict)


class UnsupportedFormatError(Exception):
    """Raised when the registry has no loader for an extension."""


# Populated below; loaders self-register when their module is imported.
REGISTRY: dict[str, LoaderFn] = {}


def register(extension: str):
    """Decorator for loader functions. Stores `fn` in REGISTRY under `extension`."""
    def _decorator(fn: LoaderFn) -> LoaderFn:
        REGISTRY[extension.lower()] = fn
        return fn
    return _decorator


def load(path: Path, source: str) -> Iterator[RawDocument]:
    """Dispatch to the loader registered for `path`'s extension. Raises
    UnsupportedFormatError if no loader is registered."""
    ext = path.suffix.lower()
    loader = REGISTRY.get(ext)
    if loader is None:
        raise UnsupportedFormatError(ext)
    yield from loader(path, source)


# Self-registration: import the loader modules after the public API above
# is defined so they can `from backend.rag.loaders import RawDocument, register`
# without hitting a partial-module circular import.
from backend.rag.loaders import text  # noqa: E402, F401  (.txt, .md)
from backend.rag.loaders import pdf  # noqa: E402, F401  (.pdf) — Phase A stub; Phase B replaces
from backend.rag.loaders import html  # noqa: E402, F401  (.html) — Phase A stub; Phase B replaces

# Routes that want richer formats also import docx, csv (Phase C).
ALLOWED_EXTENSIONS = frozenset(REGISTRY.keys())