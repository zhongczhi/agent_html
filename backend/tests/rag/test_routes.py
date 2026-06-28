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


def test_upload_indexes_file_and_creates_conversation(rag_client, tmp_path):
    client, rag = rag_client
    # Seed an upload file
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
    assert body["chunks_added"] >= 1

    # The conversation is visible in file_storage
    conv = file_storage.get_conversation("c1")
    assert conv is not None
    assert conv["conversation_id"] == "c1"


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
