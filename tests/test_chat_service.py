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


class TestStreamJobIntegration:
    @pytest.mark.asyncio
    async def test_generate_stores_tokens_in_job(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService
        from backend.chat.stream_manager import clear_job, get_job

        file_storage, _ = temp_storage_dir

        mock_chain.astream = MagicMock()
        mock_chain.astream.return_value = async_gen_from_list(["H", "i", "!"])

        service = ChatService(mock_chain)

        clear_job("stream-test")
        tokens = []
        async for token in service.generate("hello", "stream-test"):
            tokens.append(token)

        assert tokens == ["H", "i", "!"]
        job = get_job("stream-test")
        assert job is not None
        assert job.tokens == ["H", "i", "!"]
        # After full iteration, job should be completed
        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_generate_marks_completed(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService
        from backend.chat.stream_manager import clear_job, get_job

        file_storage, _ = temp_storage_dir

        mock_chain.astream = MagicMock()
        mock_chain.astream.return_value = async_gen_from_list(["done"])

        service = ChatService(mock_chain)

        clear_job("complete-test")
        async for _ in service.generate("hello", "complete-test"):
            pass

        job = get_job("complete-test")
        assert job is not None
        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_generate_marks_failed_on_error(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService
        from backend.chat.stream_manager import clear_job, get_job

        file_storage, _ = temp_storage_dir

        def error_gen(*args, **kwargs):
            raise Exception("Test error")

        mock_chain.astream = MagicMock()
        mock_chain.astream.side_effect = error_gen

        service = ChatService(mock_chain)

        clear_job("error-test")
        with pytest.raises(Exception):
            async for _ in service.generate("hello", "error-test"):
                pass

        job = get_job("error-test")
        assert job is not None
        assert job.status == "failed"
        assert job.error == "Test error"

    def test_get_stream_status_none(self, mock_chain, temp_storage_dir):
        from backend.chat.service import ChatService
        from backend.chat.stream_manager import clear_job

        file_storage, _ = temp_storage_dir

        clear_job("nonexistent-status")
        service = ChatService(mock_chain)
        status = service.get_stream_status("nonexistent-status")
        assert status["streaming"] == False
        assert status["status"] == "none"
        assert status["tokens_count"] == 0