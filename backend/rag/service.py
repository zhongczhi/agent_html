import logging
import os
import shutil
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from backend.rag.config import RagSettings
from backend.rag.retriever import ScopedRetriever
from backend.rag.splitter import split_into_documents
from backend.rag.vector_store import load_or_init, rebuild_filtered, save
from backend.rag.embeddings import make_embeddings
from backend.rag.loaders import ALLOWED_EXTENSIONS, UnsupportedFormatError

logger = logging.getLogger(__name__)


def _walk_library(library_dir: Path) -> list[Path]:
    """Walks library_dir for allowlisted files, sorted alphabetically.
    Service-internal helper — moved from splitter.py in iter-8 Phase D."""
    if not library_dir.exists():
        return []
    return sorted(
        p for p in library_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


class RagService:
    def __init__(self, settings: RagSettings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings

        # Anchor paths to <repo>/backend/. Default RagSettings values
        # ("storage/library", "storage/uploads", "storage/rag") resolve to
        # <repo>/backend/storage/{library,uploads,rag}, keeping all RAG
        # runtime state under one tree owned by the backend package. The
        # legacy conversation storage at <repo>/storage/conversations.json
        # (via file_storage.py) is unchanged.
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

        # Auto-create the library dir so the API can list/save into it
        # without first requiring an admin to mkdir on disk.
        self.library_dir.mkdir(parents=True, exist_ok=True)

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

        docs = list(split_into_documents(
            dest,
            source_type="upload",
            conversation_id=conversation_id,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        ))

        # Filter out placeholder if it exists, then re-add (FAISS doesn't
        # gracefully handle add_documents on an empty placeholder index).
        self.uploads_index = rebuild_filtered(self.uploads_index, self.embeddings, keep=lambda d: True)
        # Guard against empty docs (e.g., a DOCX whose paragraphs are all
        # empty). FAISS.add_documents([]) raises on some versions.
        if docs:
            self.uploads_index.add_documents(docs)
        save(self.uploads_index, self._index_path("uploads_index"))
        return [d.metadata["chunk_id"] for d in docs]

    def reindex_library(self) -> dict:
        files = _walk_library(self.library_dir)
        errors: list[str] = []
        all_docs: list[Document] = []
        for path in files:
            try:
                all_docs.extend(split_into_documents(
                    path,
                    source_type="library",
                    conversation_id=None,
                    chunk_size=self.settings.rag_chunk_size,
                    chunk_overlap=self.settings.rag_chunk_overlap,
                ))
            except UnsupportedFormatError:
                # _walk_library already filters by extension; this is defensive
                continue
            except Exception as e:
                errors.append(f"{path}: {e}")

        # Empty case: no files in library. FAISS requires ≥1 doc, so use the
        # placeholder. load_or_init would do the same, but here we want to
        # overwrite the existing index (not just init it on first start).
        if not all_docs:
            self.library_index = FAISS.from_documents(
                [Document(page_content="", metadata={"_placeholder": True})],
                self.embeddings,
            )
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

    # ── Library management (iter-8 Phase D) ──────────────────────────

    def list_library_files(self) -> list[dict]:
        """Return metadata for every allowlisted file in library_dir,
        sorted alphabetically. Used by the library sidebar tab."""
        if not self.library_dir.exists():
            return []
        files: list[dict] = []
        for p in self.library_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            stat = p.stat()
            files.append({
                "filename": p.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            })
        return files

    def save_library_file(self, filename: str, content: bytes) -> Path:
        """Atomically write `content` to library_dir/<filename>, then auto-
        reindex so the file is queryable immediately. Caller must validate
        filename + extension (routes do this)."""
        self.library_dir.mkdir(parents=True, exist_ok=True)
        dest = self.library_dir / filename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.library_dir),
            prefix=f".{filename}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Auto-reindex so the uploaded file is queryable right away.
        # Log on failure but don't roll back the write — the file is on
        # disk and the manual reindex endpoint can recover.
        try:
            self.reindex_library()
        except Exception:
            logger.exception("Auto-reindex after library upload failed for %s", filename)
        return dest

    def delete_library_file(self, filename: str) -> bool:
        """Delete library_dir/<filename> and auto-reindex. Returns True if
        the file existed."""
        target = self.library_dir / filename
        if not target.exists() or not target.is_file():
            return False
        target.unlink()
        try:
            self.reindex_library()
        except Exception:
            logger.exception("Auto-reindex after library delete failed for %s", filename)
        return True

    # ── Read path ────────────────────────────────────────────────────

    def make_scoped_retriever(self, conversation_id: str, top_k: int) -> ScopedRetriever:
        return ScopedRetriever(
            retrievers=[
                (self.library_index.as_retriever(search_kwargs={"k": top_k}), False),
                (self.uploads_index.as_retriever(search_kwargs={"k": top_k}), True),
            ],
            conversation_id=conversation_id,
        )

    def retrieve_by_scope(self, conversation_id: str, query: str, top_k: int) -> dict[str, list]:
        """Search each scope independently and return hits grouped by scope.

        Unlike make_scoped_retriever (which merges hits across scopes),
        this preserves per-scope emptiness so the chat service can emit a
        sources event even when every scope returned zero hits — letting
        the frontend render an explicit "library: 0 / uploads: 0" empty
        state instead of silently dropping the sources block.
        """
        library_hits = [
            d for d in self.library_index.as_retriever(search_kwargs={"k": top_k}).invoke(query)
            if not d.metadata.get("_placeholder")
        ]
        uploads_hits = [
            d for d in self.uploads_index.as_retriever(search_kwargs={"k": top_k}).invoke(query)
            if not d.metadata.get("_placeholder")
            and d.metadata.get("conversation_id") == conversation_id
        ]
        return {"library": library_hits, "uploads": uploads_hits}

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
            "library_files": len(self.list_library_files()),
        }