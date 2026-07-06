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
    # The RagService fixture auto-creates the library dir, so use exist_ok=True.
    lib = tmp_path / "library"
    lib.mkdir(exist_ok=True)
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


# ── iter-8 library management ───────────────────────────────────────────────

def test_library_files_lists_allowlisted(rag_client, tmp_path):
    """GET /api/rag/library/files returns sorted metadata for allowlisted files."""
    client, _ = rag_client
    lib = tmp_path / "library"
    (lib / "b.md").write_text("b-content", encoding="utf-8")
    (lib / "a.txt").write_text("a-content", encoding="utf-8")
    (lib / "ignored.bin").write_bytes(b"nope")  # not in allowlist

    resp = client.get("/api/rag/library/files")
    assert resp.status_code == 200
    files = resp.json()["files"]
    names = [f["filename"] for f in files]
    assert names == ["a.txt", "b.md"]  # alphabetical
    assert all("size" in f and "modified_at" in f for f in files)


def test_library_upload_writes_and_reindexes(rag_client, tmp_path):
    """POST /api/rag/library/upload saves the file and triggers reindex."""
    client, rag = rag_client
    lib = tmp_path / "library"

    resp = client.post(
        "/api/rag/library/upload",
        files={"file": ("new.md", b"# New\n\nfresh content", "text/markdown")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "new.md"
    assert body["saved"] is True
    assert (lib / "new.md").exists()
    # Auto-reindex ran — library_index now has chunks from new.md
    chunks = [d for d in rag.library_index.docstore._dict.values()
              if not d.metadata.get("_placeholder")]
    assert any(d.metadata.get("filename") == "new.md" for d in chunks)


def test_library_upload_409_on_duplicate(rag_client, tmp_path):
    """Uploading a filename that already exists returns 409."""
    client, _ = rag_client
    lib = tmp_path / "library"
    (lib / "exists.md").write_text("already here", encoding="utf-8")

    resp = client.post(
        "/api/rag/library/upload",
        files={"file": ("exists.md", b"second", "text/markdown")},
    )
    assert resp.status_code == 409
    assert "already in the library" in resp.json()["detail"]


def test_library_upload_400_on_bad_extension(rag_client):
    client, _ = rag_client
    resp = client.post(
        "/api/rag/library/upload",
        files={"file": ("bad.exe", b"x", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_library_upload_400_on_path_traversal(rag_client):
    client, _ = rag_client
    resp = client.post(
        "/api/rag/library/upload",
        files={"file": ("../escape.md", b"x", "text/markdown")},
    )
    assert resp.status_code == 400


def test_library_upload_400_on_dotfile(rag_client):
    client, _ = rag_client
    resp = client.post(
        "/api/rag/library/upload",
        files={"file": (".hidden.md", b"x", "text/markdown")},
    )
    assert resp.status_code == 400


def test_library_file_delete_removes_and_reindexes(rag_client, tmp_path):
    """DELETE /api/rag/library/file/{name} removes the file and reindexes."""
    client, rag = rag_client
    lib = tmp_path / "library"
    (lib / "doomed.md").write_text("to be removed", encoding="utf-8")
    rag.reindex_library()
    chunks_before = [d for d in rag.library_index.docstore._dict.values()
                     if not d.metadata.get("_placeholder")]
    assert any(d.metadata["filename"] == "doomed.md" for d in chunks_before)

    resp = client.delete("/api/rag/library/file/doomed.md")
    assert resp.status_code == 200
    assert not (lib / "doomed.md").exists()
    chunks_after = [d for d in rag.library_index.docstore._dict.values()
                    if not d.metadata.get("_placeholder")]
    assert not any(d.metadata["filename"] == "doomed.md" for d in chunks_after)


def test_library_file_delete_404_when_missing(rag_client):
    client, _ = rag_client
    resp = client.delete("/api/rag/library/file/nonexistent.md")
    assert resp.status_code == 404


def test_library_file_delete_400_on_path_traversal(rag_client):
    client, _ = rag_client
    # FastAPI path matching prevents most traversal, but defense-in-depth
    resp = client.delete("/api/rag/library/file/..")
    assert resp.status_code in (400, 404)  # either is acceptable


def test_library_file_delete_400_on_bad_extension(rag_client):
    client, _ = rag_client
    resp = client.delete("/api/rag/library/file/bad.exe")
    assert resp.status_code == 400


def test_stats_includes_library_files_count(rag_client, tmp_path):
    """stats endpoint reports the library_files count."""
    client, _ = rag_client
    lib = tmp_path / "library"
    (lib / "a.md").write_text("a", encoding="utf-8")
    (lib / "b.txt").write_text("b", encoding="utf-8")

    resp = client.get("/api/rag/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["library_files"] == 2


# ── iter-9: recursive list + subpath delete via REST ─────────────────────────


def test_library_files_endpoint_is_recursive(rag_client, tmp_path):
    """Regression: GET /api/rag/library/files used list_library_files()
    which used iterdir() and hid subdirs. After iter-9 the endpoint must
    return relative paths for files at any depth."""
    client, _ = rag_client
    lib = tmp_path / "library"
    (lib / "top.md").write_text("top-level", encoding="utf-8")
    sub = lib / "hotpotqa"
    sub.mkdir(exist_ok=True)
    (sub / "qa1.md").write_text("nested", encoding="utf-8")

    # Force a fresh service from the populated dir
    rag_client[1].library_dir = lib  # type: ignore[attr-defined]

    resp = client.get("/api/rag/library/files")
    assert resp.status_code == 200
    names = {f["filename"] for f in resp.json()["files"]}
    assert "top.md" in names
    assert "hotpotqa/qa1.md" in names


def test_library_file_delete_accepts_subpath_filename(rag_client, tmp_path):
    """Regression: DELETE /api/rag/library/file/{filename} used to reject
    names with '/'. Iter-9 allows subpaths so users can remove files
    visible in the recursive listing."""
    client, _ = rag_client
    lib = tmp_path / "library"
    sub = lib / "hotpotqa"
    sub.mkdir(parents=True, exist_ok=True)
    target = sub / "qa42.md"
    target.write_text("to be removed", encoding="utf-8")
    rag_client[1].library_dir = lib  # type: ignore[attr-defined]

    resp = client.delete("/api/rag/library/file/hotpotqa/qa42.md")
    assert resp.status_code == 200
    assert not target.exists()


def test_library_file_delete_rejects_traversal_via_subpath(rag_client, tmp_path):
    """A traversal-shaped filename must be rejected with 400, never 5xx.
    Note: literal `..` in the URL is normalized by Starlette (route returns
    404 for `/api/rag/library/file/../escape.md` before reaching our handler),
    so we URL-encode the slash. `_safe_library_path` still sees the literal
    `..` and raises ValueError → the route translates to 400."""
    client, _ = rag_client
    lib = tmp_path / "library"
    lib.mkdir(exist_ok=True)
    rag_client[1].library_dir = lib  # type: ignore[attr-defined]

    # `..%2Fescape.md` decodes to `../escape.md`, passes through as a single
    # path segment under `{filename:path}`, then trips _safe_library_path.
    resp = client.delete("/api/rag/library/file/..%2Fescape.md")
    assert resp.status_code == 400
