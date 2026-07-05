import json
import logging
import re

from backend.storage import file_storage
from backend.chat.chain import RAG_SYSTEM_PROMPT
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


# Tag that brackets retrieved / inline-file context inside a user message.
# The LLM uses it as the boundary of "grounding material"; the backend's
# get_history strips it before sending the message to the frontend (the
# user only sees their original question, not the retrieved chunks).
_CONTEXT_TAG_RE = re.compile(r"<context>[\s\S]*?</context>\s*")


def _embed_context(user_content: str, context_text: str) -> str:
    """Prepend a <context>...</context> block to a user message.

    The tagged message is BOTH saved to disk AND sent to the LLM — the
    context is essential grounding material for the model, not
    presentational scaffolding to be hidden at the chat layer.
    """
    return f"<context>\n{context_text}\n</context>\n\n{user_content}"


def _strip_context_tags(user_content: str) -> str:
    """Strip <context>...</context> blocks from a user message before it
    leaves the backend. Disk keeps the tagged form (so subsequent LLM
    turns see their context); the wire-out form is clean.
    """
    if not user_content:
        return user_content
    return _CONTEXT_TAG_RE.sub("", user_content).strip()


def _replace_last_user_context(messages: list, context_text: str) -> None:
    """Find the last user message in the list and prepend a <context>...
    block to its content. Mutates `messages` in place. If no user message
    exists (defensive — shouldn't happen since the route appends first),
    the call is a no-op."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            messages[i] = {
                "role": "user",
                "content": _embed_context(messages[i]["content"], context_text),
            }
            return


class ChatService:
    def __init__(self, chain, rag_service=None):
        self.chain = chain
        self.rag_service = rag_service
        # Per-conversation list of small files uploaded via the inline path.
        # Cleared when the conversation is deleted (see clear_pending_inline_files).
        # In-memory only — a server restart drops the list. This is acceptable
        # for v1 because re-uploading a small file is cheap.
        self._pending_inline_files: dict[str, list[dict]] = {}

    def clear_pending_inline_files(self, conversation_id: str) -> None:
        """Drop the pending inline-files list for a conversation. Called from
        main.py's delete-conversation callback chain. Best-effort: any exception
        is swallowed by the caller (file_storage.delete_conversation)."""
        self._pending_inline_files.pop(conversation_id, None)

    async def generate_background(
        self,
        message: str,
        conversation_id: str,
        retrieval=None,
        uploaded_files=None,
    ) -> None:
        """
        Background task: fetches from LLM and stores in StreamJob.
        Does not yield to caller - runs independently.

        retrieval: optional RetrievalConfig. When set AND rag_service is set,
        the service retrieves chunks via RagService.retrieve_by_scope,
        pushes a 'sources' SSE chunk (per-scope dict, empty arrays included
        so the frontend can render an explicit "no matches" state), and
        embeds the retrieved context into the LAST user message wrapped in
        <context>...</context>. The chat prompt template prepends the
        global SYSTEM_PROMPT; per-turn context lives inside the user
        message where it actually reaches the LLM. When retrieval is None
        or rag_service is None, the chat runs identically to the pre-RAG
        version (the plugin-off guarantee).
        """
        job = get_or_create_job(conversation_id, [])

        # The user message is already appended to storage by routes.stream_chat
        # (so the conversation appears in the sidebar with the correct title
        # during streaming). Load it from storage here — no need to append
        # again, since storage is the single source of truth.
        history = file_storage.get_conversation(conversation_id)
        messages = history["messages"] if history else []
        job.messages = messages

        # Merge newly-uploaded inline files into the per-conversation pending
        # list, so subsequent turns in the same conversation see them without
        # re-sending. Deduplicate by filename — re-uploading the same file
        # replaces the prior content (in case the user edited the file).
        if uploaded_files:
            existing = self._pending_inline_files.setdefault(conversation_id, [])
            existing_filenames = {f["filename"] for f in existing}
            for uf in uploaded_files:
                if uf.filename in existing_filenames:
                    for i, prev in enumerate(existing):
                        if prev["filename"] == uf.filename:
                            existing[i] = {"filename": uf.filename, "content": uf.content}
                            break
                else:
                    existing.append({"filename": uf.filename, "content": uf.content})
                    existing_filenames.add(uf.filename)

        # ── Inline-files pre-processing block ─────────────────────────
        # Takes precedence over FAISS retrieval. The user has explicitly
        # attached a file to this turn (or the conversation has pending files);
        # FAISS results are skipped to avoid context bloat and double-grounding.
        used_context = False
        pending = self._pending_inline_files.get(conversation_id, [])
        if pending:
            try:
                sources_event = {
                    "sources": {
                        "uploads": [
                            {
                                "filename": f["filename"],
                                "excerpt": f["content"][:300],
                            }
                            for f in pending
                        ]
                    }
                }
                job.append_chunk("sources", json.dumps(sources_event))
                context_str = "\n\n".join(
                    f"[{f['filename']}]:\n{f['content']}" for f in pending
                )
                _replace_last_user_context(messages, context_str)
                used_context = True
            except Exception as e:
                logger.exception("Inline file injection failed; continuing without context: %s", e)
        # ──────────────────────────────────────────────────────────────

        # ── RAG (FAISS) pre-processing block ───────────────────────────
        # Only runs when no inline files were attached. Both paths emit a
        # sources event before tokens; the inline path takes precedence so
        # we don't double up. We use retrieve_by_scope (not
        # make_scoped_retriever) so the sources event preserves per-scope
        # emptiness — the frontend can then render "library: 0 / uploads: 0"
        # instead of silently dropping the sources block when nothing matched.
        elif retrieval is not None and self.rag_service is not None:
            try:
                hits_by_scope = self.rag_service.retrieve_by_scope(
                    conversation_id, message, retrieval.top_k,
                )
                sources_event = {
                    "sources": {
                        scope: [
                            {
                                "filename": h.metadata.get("filename"),
                                "excerpt": h.page_content[:300],
                            }
                            for h in hits
                        ]
                        for scope, hits in hits_by_scope.items()
                    }
                }
                job.append_chunk("sources", json.dumps(sources_event))
                all_hits = [h for hits in hits_by_scope.values() for h in hits]
                if all_hits:
                    context_str = "\n\n".join(
                        f"[{h.metadata.get('filename')}]: {h.page_content}" for h in all_hits
                    )
                    _replace_last_user_context(messages, context_str)
                    used_context = True
            except Exception as e:
                logger.exception("Retrieval failed; continuing without context: %s", e)
        # ──────────────────────────────────────────────────────────────

        # Prepend the RAG system prompt only on turns that actually used
        # per-turn context (inline files or RAG retrieval). Vanilla turns
        # send no system message at all — no RAG-specific instructions
        # to bias the model when there's no <context> tag in play. The
        # system message is transient (in-memory for the LLM call only);
        # storage stays clean with just user + assistant messages.
        #
        # `messages` stays as the saved-history list (no system message);
        # `llm_messages` is the augmented copy that goes to the LLM.
        llm_messages = messages
        if used_context:
            llm_messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}] + messages

        try:
            async for chunk in self.chain.astream(llm_messages):
                # If the user deleted the conversation mid-stream, stop early and
                # do NOT call mark_completed or save_conversation — that would
                # resurrect the deleted conversation in storage.
                if job.cancelled:
                    return

                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, "dict") and "content" in chunk:
                    content = chunk["content"]
                elif isinstance(chunk, str):
                    content = chunk

                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "thinking":
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                job.append_chunk("thinking", thinking_text)
                        elif block.get("type") == "text":
                            token = block.get("text", "")
                            if token:
                                job.append_chunk("token", token)
                elif isinstance(content, str) and content:
                    job.append_chunk("token", content)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            job.mark_failed(str(e))
            return

        # Defensive: if cancellation happened right as the LLM finished, do not save.
        if job.cancelled:
            return

        job.mark_completed()

        # Save to history
        full_content = "".join(c["chunk"] for c in job.chunks if c["type"] == "token")
        full_thinking = "".join(c["chunk"] for c in job.chunks if c["type"] == "thinking")
        # Sources are persisted alongside the message so they survive a page
        # reload. The frontend's renderMessagesFromCache reads msg.sources
        # and re-renders the block. None when RAG was off / inline files
        # path was used (no sources chunk) — `None` is the canonical
        # "no sources" value (vs. an empty dict, which means "searched but
        # found nothing").
        sources_obj = None
        sources_chunks = [c["chunk"] for c in job.chunks if c["type"] == "sources"]
        if sources_chunks:
            sources_obj = json.loads(sources_chunks[0]).get("sources")
        messages.append({
            "role": "assistant",
            "content": full_content,
            "thinking": full_thinking,
            "sources": sources_obj,
        })
        file_storage.save_conversation(conversation_id, messages)

    def get_history(self, conversation_id: str) -> dict | None:
        """Return the conversation history with <context>...</context>
        blocks stripped from user messages. Disk retains the tagged form
        (so subsequent LLM turns see their per-turn grounding); the
        frontend sees clean user-typed content only."""
        raw = file_storage.get_conversation(conversation_id)
        if raw is None:
            return None
        cleaned = []
        for m in raw.get("messages", []):
            if m.get("role") == "user":
                cleaned.append({**m, "content": _strip_context_tags(m.get("content", ""))})
            else:
                cleaned.append(m)
        return {**raw, "messages": cleaned}