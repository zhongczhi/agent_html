from unittest.mock import patch, MagicMock
import pytest
from backend.rag.embeddings import make_embeddings, MiniMaxEmbeddings


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        make_embeddings("nonsense")


def test_sentence_transformers_returns_huggingface_instance():
    """Lazy-load HuggingFaceEmbeddings only inside the function so import-time
    cost is zero and tests don't need a downloaded model."""
    fake_hf = MagicMock(name="HuggingFaceEmbeddings_instance")
    with patch("backend.rag.embeddings._build_huggingface", return_value=fake_hf) as build:
        result = make_embeddings("sentence-transformers", model_name="all-MiniLM-L6-v2")
    assert result is fake_hf
    build.assert_called_once_with("all-MiniLM-L6-v2")


def test_minimax_returns_minimax_class_instance():
    fake_mm = MagicMock(name="MiniMaxEmbeddings_instance")
    with patch("backend.rag.embeddings._build_minimax", return_value=fake_mm) as build:
        result = make_embeddings("minimax", api_key="k", base_url="https://x")
    assert result is fake_mm
    build.assert_called_once_with("k", "https://x")


def test_minimax_class_is_langchain_embeddings_subclass():
    from langchain_core.embeddings import Embeddings
    assert issubclass(MiniMaxEmbeddings, Embeddings)
