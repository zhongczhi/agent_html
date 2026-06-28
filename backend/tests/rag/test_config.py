import os
from backend.rag.config import RagSettings


def test_default_settings_have_rag_disabled():
    """RAG must be disabled by default — the plugin property depends on this."""
    settings = RagSettings(_env_file=None)
    assert settings.rag_enabled is False


def test_settings_load_embedding_backend_from_env(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "minimax")
    settings = RagSettings()
    assert settings.rag_enabled is True
    assert settings.rag_embedding_backend == "minimax"


def test_settings_have_chunk_and_topk_defaults():
    settings = RagSettings(_env_file=None)
    assert settings.rag_chunk_size == 800
    assert settings.rag_chunk_overlap == 200
    assert settings.rag_top_k == 4


def test_settings_paths_relative_to_backend():
    """Paths are stored as relative strings; RagService.__init__ anchors them."""
    settings = RagSettings(_env_file=None)
    assert settings.rag_library_dir == "storage/library"
    assert settings.rag_uploads_dir == "storage/uploads"
    assert settings.rag_index_dir == "storage/rag"
