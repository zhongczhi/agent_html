# tests/test_storage.py
import pytest
from pathlib import Path


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
