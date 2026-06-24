# tests/test_storage.py
import json
import os
import threading

import pytest


class TestStorage:
    def test_save_and_load_conversation(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        conversation_id = "test-123"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        file_storage.save_conversation(conversation_id, messages)

        loaded = file_storage.get_conversation(conversation_id)
        assert loaded is not None
        assert loaded["conversation_id"] == conversation_id
        assert len(loaded["messages"]) == 2
        assert loaded["messages"][0]["content"] == "Hello"

    def test_append_message(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        conversation_id = "test-456"
        file_storage.append_message(conversation_id, "user", "First message")

        messages = file_storage.append_message(conversation_id, "assistant", "First response")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_nonexistent_conversation(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        result = file_storage.get_conversation("nonexistent-id")
        assert result is None

    def test_conversations_persist_to_file(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        conversation_id = "persist-test"
        messages = [{"role": "user", "content": "Test"}]

        file_storage.save_conversation(conversation_id, messages)

        assert (temp_dir / "conversations.json").exists()

    def test_get_conversation_handles_invalid_json(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        conversations_file = temp_dir / "conversations.json"
        conversations_file.write_text("{ invalid json content }")

        result = file_storage.get_conversation("any-id")
        assert result is None


class TestConversationList:
    def test_get_conversation_list_empty(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        result = file_storage.get_conversation_list()
        assert result == []

    def test_get_conversation_list_with_data(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        file_storage.save_conversation("conv1", [{"role": "user", "content": "hello"}])
        file_storage.save_conversation("conv2", [{"role": "user", "content": "hi"}])
        result = file_storage.get_conversation_list()
        assert len(result) == 2
        ids = [c["conversation_id"] for c in result]
        assert "conv1" in ids
        assert "conv2" in ids

    def test_get_conversation_list_sorted_by_updated_at(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        file_storage.save_conversation("older", [{"role": "user", "content": "older"}])
        file_storage.save_conversation("newer", [{"role": "user", "content": "newer"}])
        result = file_storage.get_conversation_list()
        # Should be sorted by updated_at descending (newer first)
        assert result[0]["conversation_id"] == "newer"
        assert result[1]["conversation_id"] == "older"

    def test_get_conversation_list_title_truncation(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        long_content = "A" * 100
        file_storage.save_conversation("long", [{"role": "user", "content": long_content}])
        result = file_storage.get_conversation_list()
        assert result[0]["title"] == "A" * 50 + "..."
        assert len(result[0]["title"]) == 53  # 50 + "..."


class TestDeleteConversation:
    def test_delete_conversation(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        file_storage.save_conversation("to-delete", [{"role": "user", "content": "hello"}])
        result = file_storage.delete_conversation("to-delete")
        assert result == True
        assert file_storage.get_conversation("to-delete") is None

    def test_delete_conversation_not_exists(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        result = file_storage.delete_conversation("non-existent")
        assert result == False

    def test_delete_conversation_clears_from_list(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir

        file_storage.save_conversation("to-delete", [{"role": "user", "content": "hello"}])
        file_storage.delete_conversation("to-delete")
        result = file_storage.get_conversation_list()
        assert len(result) == 0


class TestAtomicWrite:
    """Atomic write: the on-disk file is always fully-old or fully-new."""

    def test_atomic_write_replaces_file_on_success(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir
        path = temp_dir / "atomic.json"
        path.write_text('{"old": true}')

        file_storage._atomic_write_json(path, {"new": True})

        assert json.loads(path.read_text()) == {"new": True}
        # Temp file is gone (replaced)
        assert not (temp_dir / "atomic.json.tmp").exists()

    def test_atomic_write_leaves_original_when_replace_fails(self, temp_storage_dir, monkeypatch):
        file_storage, temp_dir = temp_storage_dir
        path = temp_dir / "atomic.json"
        path.write_text('{"original": true}')

        def fail_replace(*args, **kwargs):
            raise OSError("simulated crash mid-swap")

        monkeypatch.setattr(os, "replace", fail_replace)

        with pytest.raises(OSError, match="simulated crash"):
            file_storage._atomic_write_json(path, {"new": True})

        # Original is fully intact — never partial.
        assert json.loads(path.read_text()) == {"original": True}
        # Temp cleaned up.
        assert not (temp_dir / "atomic.json.tmp").exists()

    def test_atomic_write_cleans_tmp_when_write_fails(self, temp_storage_dir, monkeypatch):
        file_storage, temp_dir = temp_storage_dir
        path = temp_dir / "atomic.json"
        path.write_text('{"original": true}')

        def fail_dump(*args, **kwargs):
            raise ValueError("simulated write failure")

        monkeypatch.setattr("json.dump", fail_dump)

        with pytest.raises(ValueError, match="simulated write failure"):
            file_storage._atomic_write_json(path, {"new": True})

        assert json.loads(path.read_text()) == {"original": True}
        # No tmp left behind even on dump failure.
        assert not (temp_dir / "atomic.json.tmp").exists()


class TestWriteLock:
    """Threading lock: concurrent writers don't lose updates."""

    def test_concurrent_appends_preserve_all_messages(self, temp_storage_dir):
        file_storage, temp_dir = temp_storage_dir
        conv_id = "concurrent-append"
        file_storage.create_conversation(conv_id)

        n = 50
        threads = [
            threading.Thread(
                target=file_storage.append_message,
                args=(conv_id, "user", f"msg-{i}"),
            )
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conv = file_storage.get_conversation(conv_id)
        contents = [m["content"] for m in conv["messages"]]
        # Every thread's message must be present — no lost updates.
        assert len(contents) == n
        assert set(contents) == {f"msg-{i}" for i in range(n)}

    def test_concurrent_saves_are_serialized_with_no_corruption(self, temp_storage_dir):
        """Concurrent save_conversation calls are serialized by the lock.
        `save_conversation` is a *replace* (not a merge), so the last
        writer wins for the messages field — but the lock guarantees
        that all writes complete and the on-disk file is always fully
        valid JSON (no partial writes from interleaving)."""
        file_storage, temp_dir = temp_storage_dir
        conv_id = "concurrent-save"
        file_storage.create_conversation(conv_id)

        n = 20
        barrier = threading.Barrier(n)

        def save_with(i: int):
            barrier.wait()
            file_storage.save_conversation(conv_id, [
                {"role": "user", "content": f"from-{i}"},
            ])

        threads = [threading.Thread(target=save_with, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes completed (no deadlock), file is valid JSON.
        json.loads((temp_dir / "conversations.json").read_text())
        # Last writer wins — exactly one message is in storage.
        conv = file_storage.get_conversation(conv_id)
        assert len(conv["messages"]) == 1
        assert conv["messages"][0]["content"].startswith("from-")
        # No .tmp file is left behind.
        assert not (temp_dir / "conversations.json.tmp").exists()
