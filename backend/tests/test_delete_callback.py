"""Tests for delete_conversation's on_delete callback parameter (RAG hook)."""
from unittest.mock import MagicMock
import backend.storage.file_storage as file_storage


def test_delete_conversation_calls_on_delete(temp_storage_dir):
    fs, _ = temp_storage_dir
    fs.save_conversation("c1", [{"role": "user", "content": "hi"}])

    hook = MagicMock()
    assert fs.delete_conversation("c1", on_delete=hook) is True
    hook.assert_called_once_with("c1")


def test_delete_conversation_skips_hook_when_none(temp_storage_dir):
    """Default behavior: on_delete=None — no hook called, no error."""
    fs, _ = temp_storage_dir
    fs.save_conversation("c1", [{"role": "user", "content": "hi"}])
    # No on_delete kwarg at all — must not raise
    assert fs.delete_conversation("c1") is True


def test_delete_conversation_hook_exception_does_not_fail_delete(temp_storage_dir):
    """If the hook raises, the JSON delete still succeeded — caller sees True
    and the hook error is logged but does not propagate."""
    fs, _ = temp_storage_dir
    fs.save_conversation("c1", [{"role": "user", "content": "hi"}])

    def bad_hook(conv_id):
        raise RuntimeError("rag service down")

    assert fs.delete_conversation("c1", on_delete=bad_hook) is True
    # JSON state is consistent — c1 is gone
    assert fs.get_conversation("c1") is None


def test_delete_conversation_hook_not_called_when_not_found(temp_storage_dir):
    """Missing conversation: no JSON delete, no hook call, returns False."""
    fs, _ = temp_storage_dir
    hook = MagicMock()
    assert fs.delete_conversation("nonexistent", on_delete=hook) is False
    hook.assert_not_called()
