from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.config import settings


# RAG-specific system prompt. Sent only on turns that use per-turn context
# (RAG retrieval or inline file injection). Vanilla turns send no system
# prompt at all — the LLM uses its default behavior. chat.service decides
# which case applies per turn and prepends (or doesn't) a system message.
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user."
)


def convert_messages(messages: list) -> list:
    """Convert stored-message dicts into LangChain message objects.

    Handles all three roles — user, assistant, and system. The system
    role is included for RAG turns (chat.service prepends a RAG system
    message in-memory before calling the chain; it is NOT saved to disk,
    so subsequent-turn loads from storage won't carry a system message).

    Prior assistant `thinking` is fed back as a thinking block alongside
    the visible text so reasoning continues across turns.
    """
    result = []
    for m in messages:
        if m["role"] == "system":
            result.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            if m.get("thinking"):
                result.append(AIMessage(content=[
                    {"type": "thinking", "thinking": m["thinking"]},
                    {"type": "text", "text": m["content"]},
                ]))
            else:
                result.append(AIMessage(content=m["content"]))
    return result


def create_chain():
    llm = ChatAnthropic(
        model="minimax-3",
        anthropic_api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )

    # convert_messages now converts whatever it gets — including any
    # system message chat.service decides to prepend for RAG turns.
    chain = convert_messages | llm
    return chain