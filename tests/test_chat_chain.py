# tests/test_chat_chain.py
import pytest
from unittest.mock import MagicMock, patch


class TestChatChain:
    def test_create_chain_returns_runnable(self):
        mock_llm_instance = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm_instance) as mock_anthropic:
            from backend.chat.chain import create_chain

            chain = create_chain()

            assert chain is not None
            mock_anthropic.assert_called_once()
            call_kwargs = mock_anthropic.call_args.kwargs
            assert call_kwargs["model"] == "minimax-2.7-highspeed"
            assert call_kwargs["max_tokens"] == 4096