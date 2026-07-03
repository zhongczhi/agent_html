import hashlib
from pathlib import Path
import pytest
from langchain_core.embeddings import FakeEmbeddings

from backend.rag.service import RagService
from backend.rag.config import RagSettings


def _make_settings(tmp_path: Path, **overrides) -> RagSettings:
    """Build a RagSettings rooted in tmp_path so the test never touches the
    real backend/storage tree."""
    defaults = dict(
        rag_enabled=True,
        rag_embedding_backend="fake",
        rag_library_dir=str(tmp_path / "library"),
        rag_uploads_dir=str(tmp_path / "uploads"),
        rag_index_dir=str(tmp_path / "rag"),
        rag_chunk_size=200,
        rag_chunk_overlap=20,
        rag_top_k=4,
    )
    defaults.update(overrides)
    return RagSettings(**defaults)


def _service(tmp_path: Path) -> RagService:
    return RagService(
        settings=_make_settings(tmp_path),
        embeddings=FakeEmbeddings(size=8),
    )


def _chunk_id(filename: str, text: str) -> str:
    # iter-8 formula: content-addressed within a file, so identical text
    # in different files gets distinct IDs.
    return hashlib.sha256(f"{filename}:{text}".encode()).hexdigest()[:16]


def test_ingest_file_saves_to_uploads_dir_and_indexes_chunks(tmp_path: Path):
    svc = _service(tmp_path)

    src = tmp_path / "source.txt"
    src.write_text("hello world from rag", encoding="utf-8")
    ids = svc.ingest_file("c1", src)

    assert len(ids) > 0
    # File saved to storage/uploads/c1/
    assert (tmp_path / "uploads" / "c1" / "source.txt").exists()
    # Index has the chunk with proper metadata
    upload_docs = [
        d for d in svc.uploads_index.docstore._dict.values()
        if not d.metadata.get("_placeholder")
    ]
    assert len(upload_docs) == len(ids)
    for d in upload_docs:
        assert d.metadata["source"] == "upload"
        assert d.metadata["conversation_id"] == "c1"
        assert d.metadata["filename"] == "source.txt"
        assert d.metadata["chunk_id"] == _chunk_id(d.metadata["filename"], d.page_content)


def test_ingest_file_persists_to_disk(tmp_path: Path):
    svc = _service(tmp_path)
    src = tmp_path / "x.txt"
    src.write_text("persist me", encoding="utf-8")
    svc.ingest_file("c1", src)

    # New service should be able to load the persisted index
    svc2 = RagService(
        settings=_make_settings(tmp_path),
        embeddings=FakeEmbeddings(size=8),
    )
    docs = [d for d in svc2.uploads_index.docstore._dict.values()
            if not d.metadata.get("_placeholder")]
    assert len(docs) >= 1


def test_purge_uploads_removes_only_target_conversation(tmp_path: Path):
    svc = _service(tmp_path)
    (tmp_path / "a.txt").write_text("a-content")
    (tmp_path / "b.txt").write_text("b-content")
    svc.ingest_file("c1", tmp_path / "a.txt")
    svc.ingest_file("c2", tmp_path / "b.txt")

    svc.purge_uploads("c1")

    remaining = [
        d for d in svc.uploads_index.docstore._dict.values()
        if not d.metadata.get("_placeholder")
    ]
    assert {d.metadata["conversation_id"] for d in remaining} == {"c2"}
    # Files for c1 also gone
    assert not (tmp_path / "uploads" / "c1").exists()
    assert (tmp_path / "uploads" / "c2").exists()


def test_reindex_library_is_idempotent(tmp_path: Path):
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    (lib_dir / "doc1.md").write_text("library content one", encoding="utf-8")
    (lib_dir / "doc2.txt").write_text("library content two", encoding="utf-8")

    svc = _service(tmp_path)
    n1 = svc.reindex_library()
    n2 = svc.reindex_library()
    assert n1["chunks_added"] == n2["chunks_added"]
    assert n1["files_processed"] == 2

    docs = [d for d in svc.library_index.docstore._dict.values()
            if not d.metadata.get("_placeholder")]
    for d in docs:
        assert d.metadata["source"] == "library"
        # Library chunks must NOT have conversation_id (asymmetric metadata)
        assert "conversation_id" not in d.metadata


def test_make_scoped_retriever_returns_scoped_with_topk(tmp_path: Path):
    svc = _service(tmp_path)
    # Add a library file in the actual library dir, not tmp_path/lib.md.
    (tmp_path / "library").mkdir(parents=True, exist_ok=True)
    (tmp_path / "library" / "lib.md").write_text("library-doc", encoding="utf-8")
    (tmp_path / "u.txt").write_text("upload-doc", encoding="utf-8")
    svc.reindex_library()
    svc.ingest_file("c1", tmp_path / "u.txt")

    scoped = svc.make_scoped_retriever("c1", top_k=2)
    assert scoped.conversation_id == "c1"
    # library retriever + uploads retriever
    assert len(scoped.retrievers) == 2


def test_scoped_retriever_integration(tmp_path: Path):
    """End-to-end: service creates indexes → scoped retriever returns chunks."""
    svc = _service(tmp_path)
    (tmp_path / "u.txt").write_text("the answer is forty two")
    svc.ingest_file("c1", tmp_path / "u.txt")

    scoped = svc.make_scoped_retriever("c1", top_k=4)
    hits = scoped.invoke("what is the answer")
    assert any("forty two" in d.page_content for d in hits)
    # Library empty, so only upload hits
    assert all(d.metadata.get("source") == "upload" for d in hits)


def test_reindex_library_handles_mixed_formats(tmp_path: Path):
    """Library reindex ingests every supported format and emits chunks with
    the correct per-format metadata."""
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "notes.md").write_text("# Heading\n\nSome markdown content.")
    (lib / "data.txt").write_text("Plain text content here.")
    (lib / "page.html").write_text("<html><body><h1>Hi</h1><p>Visible.</p></body></html>")

    svc = _service(tmp_path)
    result = svc.reindex_library()
    assert result["files_processed"] == 3
    assert result["chunks_added"] >= 3
    assert result["errors"] == []

    docs = [d for d in svc.library_index.docstore._dict.values()
            if not d.metadata.get("_placeholder")]
    formats = {d.metadata["format"] for d in docs}
    assert formats == {".md", ".txt", ".html"}
    for d in docs:
        assert d.metadata["source"] == "library"
        assert d.metadata["source_type"] == "library"  # iter-8 explicit
        assert "conversation_id" not in d.metadata
        # All chunks from this run have the new chunk_id formula.
        assert d.metadata["chunk_id"] == _chunk_id(d.metadata["filename"], d.page_content)
