# Anthropic Prompt Caching

## Overview

Add Anthropic-style prompt caching to the chat pipeline so multi-turn conversations benefit from server-side cached prefixes. The change is always-on, requires no new config, and degrades gracefully if the upstream gateway does not forward `cache_control` markers.

## Problem

Every turn of a conversation sends the full conversation history as input tokens. As conversations grow (or as users upload long-context files / retrieve large RAG context), the per-turn input token cost grows linearly. Anthropic's prompt-caching feature lets the upstream API cache prefixes and charge ~10% of base input price on cache hits, but only when callers explicitly mark cache breakpoints with `cache_control: {"type": "ephemeral"}`.

This project has zero caching today — every token is billed at full price on every turn.

## Verification of upstream support

Before designing, we confirmed empirically that the project's MiniMax Anthropic-compatible gateway (`https://api.minimaxi.com/anthropic/v1/messages`) forwards `cache_control` markers and returns Anthropic-shaped usage fields. Test script and result live in [`scripts/verify_minimax_caching.py`](../../scripts/verify_minimax_caching.py).

Result: usage response includes `cache_creation_input_tokens` and `cache_read_input_tokens` after a `cache_control` marker is sent. The marker is honored.

## Solution

Modify `convert_messages` in `backend/chat/chain.py` to emit a content-block list with `cache_control: ephemeral` on the second-to-last user message. On turn 1 (only one user message), no marker is emitted. After streaming completes, log cache stats from the final chunk's `response_metadata.usage`.

No new env flag. No frontend changes. No new dependencies.

## Breakpoint placement rule

For each request's message list:

- **Find all user-message indices**, take the second-to-last as `cache_marker_idx`.
- **If `cache_marker_idx` is None (turn 1)**: emit all messages as today (plain string content). No caching on the first turn. Acceptable — first turn is a cache miss by definition.
- **If `cache_marker_idx` is set**: emit that user message as `HumanMessage(content=[{"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}])`. Emit every other message as today.

Effect per turn N (N ≥ 2):

```
messages = [user_1, assistant_1, user_2, assistant_2, ..., user_(N-1), assistant_(N-1), user_N]
marker   = user_(N-1)
cached   = [user_1, assistant_1, ..., user_(N-1)]
fresh    = [assistant_(N-1), user_N]
```

For RAG-augmented turns, `service.py` injects a system message between `assistant_(N-1)` and `user_N`. The system message is not marked — it sits in the fresh portion. On turn N+1, the marker advances to `user_N`, and the prior turn's RAG system message becomes part of the cached prefix automatically.

## File-by-file changes

### 1. `backend/chat/chain.py` — modify `convert_messages`

Current shape:

```python
def convert_messages(messages: list) -> list:
    result = []
    for m in messages:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            ...
    return result
```

Updated shape (logic only, full code in plan):

```python
def convert_messages(messages: list) -> list:
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    cache_marker_idx = user_indices[-2] if len(user_indices) >= 2 else None

    result = []
    for i, m in enumerate(messages):
        if m["role"] == "user":
            if i == cache_marker_idx:
                result.append(HumanMessage(content=[
                    {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}},
                ]))
            else:
                result.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # existing thinking-block handling unchanged
            ...
    return result
```

The `create_chain` function is untouched. `ChatAnthropic` (already in deps as `langchain-anthropic>=0.1.0`) accepts content as a list of blocks and forwards `cache_control` fields unchanged.

### 2. `backend/chat/service.py` — log cache stats after streaming

In `ChatService.generate_background`, after the `async for chunk in self.chain.astream(messages)` loop completes successfully, extract usage from the last chunk and log:

```python
# Track the last chunk so we can read response_metadata after the loop.
last_chunk = None
async for chunk in self.chain.astream(messages):
    last_chunk = chunk
    # ... existing per-chunk handling unchanged ...

# After loop: emit cache stats. Defensive — missing fields log as zero.
usage = (last_chunk.response_metadata or {}).get("usage", {}) if last_chunk is not None else {}
created = usage.get("cache_creation_input_tokens", 0)
read = usage.get("cache_read_input_tokens", 0)
input_tokens = usage.get("input_tokens", 0)
turn_number = sum(1 for m in messages if m["role"] == "user")
logger.info(
    "cache_turn cid=%s turn=%s created=%s read=%s input=%s",
    conversation_id, turn_number, created, read, input_tokens,
)
```

`turn_number` is recoverable from the messages list — no new state needed. The log call must never raise: if `response_metadata` or `usage` is missing on the final chunk, log zeros and continue.

### 3. `backend/tests/chat/test_chain.py` — new test file

Cover:

- 1-turn conversation: no marker emitted (every `HumanMessage.content` is a `str`).
- 2-turn conversation: `user_1` content is a list-of-blocks dict with `cache_control`; `user_2` is a plain string.
- 3-turn conversation: `user_2` is the marked message; `user_1` and `user_3` are plain strings.
- RAG-augmented multi-turn: only the prior user message is marked; the injected system message is not marked (treated as a regular message).
- Assistant with thinking block: the marker logic only touches user messages, assistant handling unchanged.

## Edge cases

| Case | Behavior |
|------|----------|
| Turn 1 (one user message) | No marker. Cache miss for first turn. Acceptable. |
| All-user message is empty / whitespace | Marker still emitted with empty `text`. Cache write may be skipped by gateway on < 1024 token prefix — same as today. |
| Conversation history rewritten / branched | Each request is independent; cache prefix is whatever the request sent. If a prior turn was edited, the new prefix is a new cache entry. No stale-cache issue. |
| Different `conversation_id` same content | Each conversation is a separate cache key from the gateway's perspective. No cross-conversation leakage (each request hits a fresh prefix from the gateway's view). |
| Thinking blocks | Only user messages get the marker. Assistant `AIMessage` content (which may contain thinking blocks) is emitted as today. |

## Cost / latency expectation

Per Anthropic's published pricing (and assuming MiniMax passes through the same rates):

- Cache write: ~25% surcharge on cached tokens (one-time, when prefix first lands).
- Cache hit: ~10% of base input price on cached tokens.
- Cache TTL: ~5 minutes, refreshed on each hit.

Net for a 10-turn conversation with ~2000 tokens of prior context:

- Without caching: 10 × 2000 = 20000 tokens at full price (plus growing prefix cost).
- With caching: 1 × 2000 at write+25%, then 9 × 2000 at read+10%. Roughly **55–70% reduction** in input cost across the 10 turns, depending on how the gateway prices cache tokens.

These numbers are illustrative. Real savings depend on the actual rates MiniMax charges for cached tokens, which are not documented publicly. The log line in §2 makes this measurable in production.

## Out of scope (explicit YAGNI)

- A `PROMPT_CACHING_ENABLED` env flag. Decision: always-on is safe because the markers are zero-cost when not supported.
- A `cache_stats` SSE event to the frontend. Decision: log-only for v1. Add later if end-to-end verification is needed.
- Caching the RAG system message specifically. Decision: it enters the cache prefix naturally on the next turn via the prior-user-message marker. No special-casing.
- A second cache breakpoint on the system message. Decision: this project has no system prompt today, and adding a second breakpoint is a future optimization once we have data on hit rates from the single-breakpoint deployment.

## Testing strategy

- Unit tests in `backend/tests/chat/test_chain.py` cover the placement logic in isolation — no live API calls.
- Manual smoke test: run the server, send 3 turns in a conversation, inspect server logs for `cache_turn` lines. First turn should show `created=0 read=0`; turns 2+ should show non-zero `read` (if the gateway's caching actually activates at the prefix sizes the conversation produces).

## Rollback

Revert `convert_messages` to its prior shape (always emit `HumanMessage(content=str)`). No other state to undo. The `cache_turn` log line becomes dead code and can be removed in the same revert if desired.