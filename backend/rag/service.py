import hashlib
import logging
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from backend.rag.config import RagSettings
from backend.rag.retriever import ScopedRetriever
from backend.rag.vector_store import load_or_init, save, rebuild_filtered
from backend.rag.splitter import make_splitter, _read_text, _walk_library
from backend.rag.embeddings import make_embeddings

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, settings: RagSettings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.splitter = make_splitter(settings.rag_chunk_size, settings.rag_chunk_overlap)

        # Anchor paths to backend/ — same pattern as backend/storage/file_storage.py
        # which uses Path(__file__).parent.parent.parent / "storage". This way uvicorn
        # can be started from any directory and paths still resolve correctly.
        backend_root = Path(__file__).parent.parent
        self.library_dir = (backend_root / settings.rag_library_dir).resolve()
        self.uploads_dir = (backend_root / settings.rag_uploads_dir).resolve()
        self.rag_dir = (backend_root / settings.rag_index_dir).resolve()

        # Index files are tagged with the embedding backend name. This prevents
        # the silent-failure mode where switching EMBEDDING_BACKEND loads a stale
        # index built with a different embedding model.
        self.backend_tag = settings.rag_embedding_backend
        self.library_index = load_or_init(self._index_path("library_index"), self.embeddings)
        self.uploads_index = load_or_init(self._index_path("uploads_index"), self.embeddings)

    def _index_path(self, name: str) -> Path:
        return self.rag_dir / f"{name}.{self.backend_tag}"

    @classmethod
    def from_settings(cls) -> "RagService":
        settings = RagSettings()
        embeddings = make_embeddings(settings.rag_embedding_backend)
        return cls(settings=settings, embeddings=embeddings)

    # ── Write paths ──────────────────────────────────────────────────

    def ingest_file(self, conversation_id: str, file_path: Path) -> list[str]:
        conv_uploads = self.uploads_dir / conversation_id
        conv_uploads.mkdir(parents=True, exist_ok=True)
        dest = conv_uploads / file_path.name
        if file_path.resolve() != dest.resolve():
            shutil.copy2(file_path, dest)

        text = _read_text(dest)
        chunks = self.splitter.split_text(text)
        docs = []
        for chunk_text in chunks:
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": "upload",
                    "conversation_id": conversation_id,
                    "filename": file_path.name,
                    "chunk_id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                },
            ))
        # Filter out placeholder if it exists, then re-add (FAISS doesn't
        # gracefully handle add_documents on an empty placeholder index)
        self.uploads_index = rebuild_filtered(
            self.uploads_index, self.embeddings, keep=lambda d: True,
        )
        self.uploads_index.add_documents(docs)
        save(self.uploads_index, self._index_path("uploads_index"))
        return [d.metadata["chunk_id"] for d in docs]

    def reindex_library(self) -> dict:
        files = _walk_library(self.library_dir)
        errors: list[str] = []
        all_docs: list[Document] = []
        for path in files:
            try:
                text = _read_text(path)
                chunks = self.splitter.split_text(text)
                for chunk_text in chunks:
                    all_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": "library",
                            "filename": str(path.relative_to(self.library_dir)),
                            "chunk_id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                        },
                    ))
            except Exception as e:
                errors.append(f"{path}: {e}")

        # Empty case: no files in library. FAISS requires ≥1 doc, so use the
        # placeholder. load_or_init would do the same, but here we want to
        # overwrite the existing index (not just init it on first start).
        if not all_docs:
            self.library_index = FAISS.from_documents([Document(page_content="", metadata={"_placeholder": True})], self.embeddings)
        else:
            self.library_index = FAISS.from_documents(all_docs, self.embeddings)
        save(self.library_index, self._index_path("library_index"))
        return {"files_processed": len(files), "chunks_added": len(all_docs), "errors": errors}

    def purge_uploads(self, conversation_id: str) -> None:
        conv_uploads = self.uploads_dir / conversation_id
        if conv_uploads.exists():
            shutil.rmtree(conv_uploads)
        self.uploads_index = rebuild_filtered(
            self.uploads_index, self.embeddings,
            keep=lambda doc: doc.metadata.get("conversation_id") != conversation_id,
        )
        save(self.uploads_index, self._index_path("uploads_index"))

    # ── Read path ────────────────────────────────────────────────────

    def make_scoped_retriever(self, conversation_id: str, top_k: int) -> ScopedRetriever:
        return ScopedRetriever(
            retrievers=[
                (self.library_index.as_retriever(search_kwargs={"k": top_k}), False),
                (self.uploads_index.as_retriever(search_kwargs={"k": top_k}), True),
            ],
            conversation_id=conversation_id,
        )

    # ── Misc ─────────────────────────────────────────────────────────

    def persist_all(self) -> None:
        save(self.library_index, self._index_path("library_index"))
        save(self.uploads_index, self._index_path("uploads_index"))

    def stats(self) -> dict:
        # NOTE: `docstore._dict` is a leading-underscore (private) attribute.
        # Stable in practice but technically internal to langchain_community.
        library_dict = self.library_index.docstore._dict
        uploads_dict = self.uploads_index.docstore._dict
        return {
            "enabled": True,
            "embedding_backend": self.settings.rag_embedding_backend,
            "library_chunks": sum(1 for d in library_dict.values() if not d.metadata.get("_placeholder")),
            "uploads_chunks": sum(1 for d in uploads_dict.values() if not d.metadata.get("_placeholder")),
            "uploads_conversations": sorted({
                d.metadata["conversation_id"]
                for d in uploads_dict.values()
                if d.metadata.get("conversation_id")
            }),
        }
