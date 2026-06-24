import pytest
from unittest.mock import AsyncMock, MagicMock

import backend.storage.file_storage as file_storage


@pytest.fixture
def temp_storage_dir(monkeypatch, tmp_path):
    """Redirect file_storage writes to a per-test temporary directory."""
    monkeypatch.setattr(file_storage, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(file_storage, "CONVERSATIONS_FILE", tmp_path / "conversations.json")
    return file_storage, tmp_path


@pytest.fixture
def mock_chain():
    chain = MagicMock()
    chain.astream = AsyncMock()
    return chain
