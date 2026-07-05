"""Tests for the chat chain's message conversion logic.

The chain does NOT prepend a system message automatically — chat.service
decides per turn whether to inject the RAG system prompt (only when
context was actually used). convert_messages just translates whatever
dict list it receives into LangChain message objects, handling all three
roles: system, user, assistant.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.chat.chain import RAG_SYSTEM_PROMPT, convert_messages


# ── convert_messages: per-role conversion ───────────────────────────────


def test_system_message_is_converted():
    out = convert_messages([{"role": "system", "content": "you are helpful"}])
    assert len(out) == 1
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "you are helpful"


def test_user_messages_become_human_message():
    out = convert_messages([{"role": "user", "content": "hi"}])
    assert len(out) == 1
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == "hi"


def test_assistant_without_thinking_becomes_plain_aimessage():
    out = convert_messages([{"role": "assistant", "content": "hello"}])
    assert len(out) == 1
    assert isinstance(out[0], AIMessage)
    assert out[0].content == "hello"


def test_assistant_with_thinking_becomes_block_list():
    """Prior thinking must be fed back so the model continues reasoning on
    subsequent turns. Without this, the LLM emits no `thinking` content on
    turn 2+."""
    out = convert_messages([{
        "role": "assistant",
        "content": "the answer",
        "thinking": "step by step reasoning",
    }])
    assert len(out) == 1
    assert isinstance(out[0], AIMessage)
    assert out[0].content == [
        {"type": "thinking", "thinking": "step by step reasoning"},
        {"type": "text", "text": "the answer"},
    ]


def test_rag_turn_shape_system_plus_user():
    """A RAG turn's chain input is [system (RAG prompt), user (with
    <context>...</context> tag)]. convert_messages must produce a
    SystemMessage + HumanMessage pair."""
    tagged = "<context>\n[doc.md]: fact\n</context>\n\nWhat does the doc say?"
    out = convert_messages([
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": tagged},
    ])
    assert len(out) == 2
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == RAG_SYSTEM_PROMPT
    assert isinstance(out[1], HumanMessage)
    # The context tag passes through to the HumanMessage — the LLM needs
    # to see it for grounding. Stripping is the backend's get_history
    # responsibility (frontend display), not the chain's.
    assert out[1].content == tagged
    assert "<context>" in out[1].content


def test_full_multi_turn_conversion():
    """Regression: a 2-turn conversation. The prior assistant's thinking
    must survive conversion so the LLM emits thinking on turn 2."""
    out = convert_messages([
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer", "thinking": "first reasoning"},
        {"role": "user", "content": "second question"},
    ])
    assert len(out) == 3
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == "first question"
    assert isinstance(out[1], AIMessage)
    # Thinking is preserved as a content block, not as additional_kwargs —
    # the thinking-enabled model consumes it from content blocks.
    assert out[1].content == [
        {"type": "thinking", "thinking": "first reasoning"},
        {"type": "text", "text": "first answer"},
    ]
    assert isinstance(out[2], HumanMessage)
    assert out[2].content == "second question"


# ── RAG_SYSTEM_PROMPT: only sent on context-using turns ─────────────────


def test_rag_system_prompt_instructs_llm_to_use_context_tags():
    """The RAG system prompt must tell the model how to interpret
    <context>...</context> tags — without this, the LLM might ignore
    the grounding or echo the tags to the user."""
    assert "<context>" in RAG_SYSTEM_PROMPT
    assert "prefer" in RAG_SYSTEM_PROMPT.lower()
    assert "not mention" in RAG_SYSTEM_PROMPT.lower() or "do not" in RAG_SYSTEM_PROMPT.lower()


# ── create_chain: structural sanity ────────────────────────────────────────


def test_create_chain_returns_callable():
    """Smoke test: create_chain returns a Runnable that can be invoked.
    We don't call the LLM (no API key in CI); just verify the chain
    object is wired up."""
    from backend.chat.chain import create_chain
    chain = create_chain()
    assert hasattr(chain, "invoke")
    assert hasattr(chain, "astream")