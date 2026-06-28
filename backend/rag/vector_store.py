from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

_PLACEHOLDER_KEY = "_placeholder"


def _placeholder_doc() -> Document:
    return Document(page_content="", metadata={_PLACEHOLDER_KEY: True})


def _is_placeholder(doc: Document) -> bool:
    return bool(doc.metadata.get(_PLACEHOLDER_KEY))


def load_or_init(path: Path, embeddings: Embeddings) -> FAISS:
    """Load an existing FAISS index from path, or create an empty one with
    a single placeholder doc. FAISS requires at least one vector to be
    constructible; the placeholder is filtered out at query time.
    """
    if (path / "index.faiss").exists():
        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True,
        )
    return FAISS.from_documents([_placeholder_doc()], embeddings)


def save(index: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))


def rebuild_filtered(
    index: FAISS,
    embeddings: Embeddings,
    keep,
) -> FAISS:
    """Rebuild the index keeping only docs where keep(doc) is True.
    Placeholder docs are never carried over. If nothing survives, returns
    a fresh placeholder-only index (still a valid FAISS instance)."""
    surviving = [
        d for d in index.docstore._dict.values()
        if not _is_placeholder(d) and keep(d)
    ]
    if not surviving:
        return FAISS.from_documents([_placeholder_doc()], embeddings)
    return FAISS.from_documents(surviving, embeddings)
