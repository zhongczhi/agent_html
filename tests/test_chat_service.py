# tests/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio


async def async_gen_from_list(items):
    for item in items:
        yield item


class TestChatService:
    @pytest.mark.asyncio
    async def test_generate_returns_expected_response(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService

        file_storage, _ = temp_storage_dir

        mock_chain.astream = MagicMock()
        mock_chain.astream.return_value = async_gen_from_list(["Hello", " ", "World"])

        service = ChatService(mock_chain)

        messages = []
        async for token in service.generate("Hi"):
            messages.append(token)

        assert "".join(messages) == "Hello World"

    @pytest.mark.asyncio
    async def test_generate_with_conversation_history(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService

        file_storage, _ = temp_storage_dir

        mock_chain.astream = MagicMock()
        mock_chain.astream.return_value = async_gen_from_list(["Response"])

        service = ChatService(mock_chain)

        conversation_id = "test-conv-123"

        async for _ in service.generate("First message", conversation_id):
            pass

        history = service.get_history(conversation_id)
        assert history is not None
        assert len(history["messages"]) == 2
        assert history["messages"][0]["role"] == "user"
        assert history["messages"][0]["content"] == "First message"
        assert history["messages"][1]["role"] == "assistant"

    def test_get_history_returns_none_for_missing_conversation(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService

        file_storage, _ = temp_storage_dir

        service = ChatService(mock_chain)
        history = service.get_history("nonexistent")
        assert history is None