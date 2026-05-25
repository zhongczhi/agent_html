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
