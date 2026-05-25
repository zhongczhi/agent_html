# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def temp_storage_dir():
    temp_dir = tempfile.mkdtemp()
    original_storage_path = Path(__file__).parent.parent / "backend" / "storage" / "file_storage.py"
    import sys
    import importlib.util

    spec = importlib.util.spec_from_file_location("file_storage", original_storage_path)
    file_storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_storage)

    original_storagedir = file_storage.STORAGE_DIR
    original_conversations_file = file_storage.CONVERSATIONS_FILE

    file_storage.STORAGE_DIR = Path(temp_dir)
    file_storage.CONVERSATIONS_FILE = Path(temp_dir) / "conversations.json"

    import backend.storage.file_storage as original_module
    original_module.STORAGE_DIR = file_storage.STORAGE_DIR
    original_module.CONVERSATIONS_FILE = file_storage.CONVERSATIONS_FILE

    yield file_storage, Path(temp_dir)

    original_module.STORAGE_DIR = original_storagedir
    original_module.CONVERSATIONS_FILE = original_conversations_file

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_chain():
    chain = MagicMock()
    chain.astream = AsyncMock()
    return chain