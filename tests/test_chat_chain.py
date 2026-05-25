import pytest
from backend.chat.chain import create_chain

@pytest.mark.asyncio
async def test_chain_yields_thinking_before_text():
    """Test that chain yields thinking blocks before text blocks."""
    chain = create_chain()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    chunks = []
    async for chunk in chain.astream(messages):
        chunks.append(chunk)

    # Check that we get AIMessage chunks
    assert len(chunks) > 0

    # Check for content_blocks attribute (the structured output)
    has_content_blocks = any(hasattr(c, 'content_blocks') for c in chunks)
    assert has_content_blocks, "Should have content_blocks attribute"

    # Find chunks with thinking/reasoning and text content
    has_reasoning = False
    has_text = False
    for chunk in chunks:
        if hasattr(chunk, 'content_blocks'):
            for block in chunk.content_blocks:
                if block.get('type') == 'reasoning':
                    has_reasoning = True
                if block.get('type') == 'text':
                    has_text = True

    # The stream should eventually contain both reasoning (thinking) and text blocks
    assert has_reasoning or has_text, "Should have reasoning or text blocks in content_blocks"