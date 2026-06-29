import io
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings

import backend.storage.file_storage as file_storage
import backend.rag.routes as rag_routes
from backend.rag.config import RagSettings
from backend.rag.service import RagService
from backend.rag.routes import router as rag_router


@pytest.fixture
def rag_client(monkeypatch, tmp_path):
    """Build a FastAPI app with RAG routes mounted and RagService installed.
    Returns a (client, rag_service) pair for test introspection."""
    settings = RagSettings(
        rag_enabled=True,
        rag_embedding_backend="fake",
        rag_library_dir=str(tmp_path / "library"),
        rag_uploads_dir=str(tmp_path / "uploads"),
        rag_index_dir=str(tmp_path / "rag"),
        rag_chunk_size=200,
        rag_chunk_overlap=20,
        rag_top_k=4,
    )
    monkeypatch.setattr(file_storage, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(file_storage, "CONVERSATIONS_FILE", tmp_path / "storage" / "conversations.json")
    (tmp_path / "storage").mkdir()

    rag = RagService(settings=settings, embeddings=FakeEmbeddings(size=8))
    monkeypatch.setattr(rag_routes, "get_rag_service", lambda: rag)

    app = FastAPI()
    app.include_router(rag_router)
    return TestClient(app), rag


def test_stats_endpoint_returns_counts(rag_client):
    client, rag = rag_client
    resp = client.get("/api/rag/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["embedding_backend"] == "fake"
    assert "library_chunks" in body
    assert "uploads_chunks" in body
    assert "uploads_conversations" in body


def test_upload_small_file_takes_inline_path(rag_client, tmp_path):
    """Files ≤ rag_inline_context_threshold_bytes are returned inline
    (no FAISS). The client will send the content with its next chat
    request via the uploaded_files field. (FR-12.2)"""
    client, rag = rag_client
    upload_file = tmp_path / "doc.txt"
    upload_file.write_text("hello rag", encoding="utf-8")

    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c1"},
        files={"file": ("doc.txt", upload_file.read_bytes(), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "doc.txt"
    assert body["mode"] == "inline"
    assert body["bytes"] == len("hello rag")
    assert body["content"] == "hello rag"

    # No chunks should have been added to FAISS
    assert len(rag.uploads_index.docstore._dict) == 0 or all(
        d.metadata.get("_placeholder") for d in rag.uploads_index.docstore._dict.values()
    )

    # The conversation is still visible in file_storage (idempotent create)
    conv = file_storage.get_conversation("c1")
    assert conv is not None
    assert conv["conversation_id"] == "c1"


def test_upload_large_file_takes_indexed_path(rag_client, tmp_path):
    """Files > rag_inline_context_threshold_bytes are embedded into FAISS.
    (FR-12.3)"""
    client, rag = rag_client
    # Pad with enough text to exceed the default 8192-byte threshold.
    big_text = "x" * (8192 + 100)
    upload_file = tmp_path / "big.txt"
    upload_file.write_text(big_text, encoding="utf-8")

    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c2"},
        files={"file": ("big.txt", upload_file.read_bytes(), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "big.txt"
    assert body["mode"] == "indexed"
    assert body["chunks_added"] >= 1
    assert body["bytes"] == len(big_text)

    # The FAISS uploads index now contains chunks
    real_chunks = [
        d for d in rag.uploads_index.docstore._dict.values()
        if not d.metadata.get("_placeholder")
    ]
    assert len(real_chunks) >= 1


def test_stats_endpoint_includes_threshold(rag_client):
    """The threshold is exposed so the client can apply the same boundary.
    (FR-12.5)"""
    client, _ = rag_client
    resp = client.get("/api/rag/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "inline_context_threshold_bytes" in body
    assert isinstance(body["inline_context_threshold_bytes"], int)


def test_upload_503_when_rag_disabled(monkeypatch, tmp_path):
    """Without get_rag_service returning a service, upload returns 503."""
    monkeypatch.setattr(rag_routes, "get_rag_service", lambda: None)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c1"},
        files={"file": ("doc.txt", b"x", "text/plain")},
    )
    assert resp.status_code == 503


def test_library_reindex_returns_processed_count(rag_client, tmp_path):
    client, rag = rag_client
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "doc.md").write_text("library content", encoding="utf-8")

    resp = client.post("/api/rag/library/reindex")
    assert resp.status_code == 200
    body = resp.json()
    assert body["files_processed"] == 1
    assert body["chunks_added"] >= 1
    assert body["errors"] == []


def test_upload_rejects_disallowed_extension(rag_client):
    """FR-12.1: file extensions outside the allowlist return 400. The
    rejection happens before any IO, so no FAISS ingestion or conversation
    creation occurs."""
    client, rag = rag_client
    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c1"},
        files={"file": ("virus.exe", b"x" * 100, "application/octet-stream")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "Unsupported file type" in body["detail"]
    assert ".exe" in body["detail"]

    # No conversation was created (rejection is pre-IO)
    assert file_storage.get_conversation("c1") is None
    # No FAISS ingestion
    real_chunks = [
        d for d in rag.uploads_index.docstore._dict.values()
        if not d.metadata.get("_placeholder")
    ]
    assert real_chunks == []


def test_upload_accepts_all_allowed_extensions(rag_client):
    """FR-12.1: every extension in the allowlist is accepted. .pdf and .html
    always take the indexed path (binary content can't be UTF-8 decoded)."""
    client, rag = rag_client
    for filename in ("doc.md", "doc.txt", "notes.pdf", "page.html"):
        resp = client.post(
            "/api/rag/upload",
            data={"conversation_id": f"c-{filename}"},
            files={"file": (filename, b"some content " * 100, "application/octet-stream")},
        )
        assert resp.status_code == 200, f"{filename}: {resp.text}"


def test_upload_extension_check_is_case_insensitive(rag_client):
    """FR-12.1: extension comparison normalizes case (".MD" == ".md")."""
    client, _ = rag_client
    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c1"},
        files={"file": ("README.MD", b"# Title\n", "text/markdown")},
    )
    assert resp.status_code == 200, resp.text


def test_upload_rejects_file_with_no_extension(rag_client):
    """FR-12.1: a file with no extension is rejected (empty suffix is
    outside the allowlist)."""
    client, _ = rag_client
    resp = client.post(
        "/api/rag/upload",
        data={"conversation_id": "c1"},
        files={"file": ("README", b"x", "text/plain")},
    )
    assert resp.status_code == 400
