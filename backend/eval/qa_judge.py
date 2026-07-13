"""End-to-end QA prompt builder + LLM caller for the iter-11 eval pipeline.

Isolation note: this module duplicates the RAG system prompt from
backend.chat.chain to preserve the FR-32 isolation rule that
scripts/eval_qa_hotpotqa.py must not import from backend.chat.*.
If the chat prompt changes, update RAG_SYSTEM_PROMPT_HERE in lockstep.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


# Mirror of backend.chat.chain.RAG_SYSTEM_PROMPT. See isolation note above.
# This is intentionally a separate string constant, not an import.
RAG_SYSTEM_PROMPT_HERE = (
    "You are a helpful assistant. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user."
)


def build_qa_prompt(
    question: str,
    context_docs: list[Document] | None,
) -> list[dict]:
    """Build the messages list for the LLM call.

    With context (FR-41.1):
        [system (RAG), user (<context>...</context> + question)]
    Without context (baseline mode):
        [user (question only)] — no system message.

    The chat chain's service uses the same prompt format
    (backend.chat.service._embed_context + RAG_SYSTEM_PROMPT prepended);
    we mirror it here so the eval reflects real chat behavior.
    """
    if not context_docs:
        return [{"role": "user", "content": question}]

    context_str = "\n\n".join(
        f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
    )
    user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT_HERE},
        {"role": "user", "content": user_content},
    ]


def _block_text(block) -> str:
    """Extract text from a content block regardless of TypedDict vs BaseModel.

    Anthropic SDK returns blocks as either Pydantic models (with .type / .text
    attributes) or plain dicts (with ['type'] / ['text'] keys), depending on
    version. This helper normalizes both.
    """
    if isinstance(block, dict):
        return block.get("text", "") or ""
    text = getattr(block, "text", None)
    return text or ""


def _block_type(block) -> str | None:
    """Extract type from a content block."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


async def ask_llm(
    client: "AsyncAnthropic",
    model: str,
    messages: list[dict],
    max_tokens: int = 200,
    thinking_budget: int | None = None,
) -> str:
    """One Anthropic call returning the answer text (FR-41.2).

    Skips thinking blocks (we only want the visible answer text for scoring).
    Joins multiple text blocks with newlines; trims whitespace.
    temperature=0 for determinism (NFR-21).

    If `thinking_budget` is set (positive int), enables Anthropic extended
    thinking mode with that many tokens of internal reasoning budget.
    `max_tokens` should be >= `thinking_budget` so the visible answer has
    room to render. Returns only the visible `text` blocks — reasoning
    is discarded (kept implicit in non-text blocks).
    """
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )
    if thinking_budget is not None and thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    response = await client.messages.create(**kwargs)
    parts: list[str] = []
    for block in response.content:
        if _block_type(block) == "text":
            text = _block_text(block)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()