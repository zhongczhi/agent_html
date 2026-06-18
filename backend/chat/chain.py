# backend/chat/chain.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.config import settings


def create_chain():
    # Set base URL via environment variable for custom endpoint (MiniMax)
    os.environ["ANTHROPIC_API_BASE"] = settings.anthropic_base_url

    llm = ChatAnthropic(
        model="minimax-3",
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )

    def convert_messages(messages: list) -> list:
        return [
            HumanMessage(content=m["content"]) if m["role"] == "user" else {"role": "assistant", "content": m["content"]}
            for m in messages
        ]

    chain = RunnableLambda(convert_messages) | llm
    return chain
