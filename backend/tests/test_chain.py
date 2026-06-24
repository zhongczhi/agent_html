from langchain_core.messages import AIMessage, HumanMessage

from backend.chat.chain import convert_messages


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


def test_full_multi_turn_conversion():
    """The regression scenario: a 2-turn conversation. The prior assistant's
    thinking must survive conversion so the LLM emits thinking on turn 2."""
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


def test_unknown_roles_are_dropped():
    out = convert_messages([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "should be ignored"},
        {"role": "assistant", "content": "hello"},
    ])
    assert len(out) == 2
    assert all(isinstance(m, (HumanMessage, AIMessage)) for m in out)
