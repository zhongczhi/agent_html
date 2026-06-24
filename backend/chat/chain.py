from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.config import settings


def convert_messages(messages: list) -> list:
    # Prior assistant `thinking` is fed back to the model as a thinking block
    # alongside the visible text. Without it, the model emits no `thinking`
    # content on turns 2+; with it, the chain of reasoning continues across
    # the conversation.
    result = []
    for m in messages:
        if m["role"] == "user":
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

    chain = RunnableLambda(convert_messages) | llm
    return chain
