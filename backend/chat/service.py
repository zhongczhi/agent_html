import json
import logging

from backend.storage import file_storage
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


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
        the service retrieves chunks via RagService.make_scoped_retriever,
        pushes a 'sources' SSE chunk, and augments the messages with a
        system message containing the retrieved context. When retrieval is
        None or rag_service is None, the chat runs identically to the
        pre-RAG version (the plugin-off guarantee).
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
        #
        # Filter out any prior-turn system messages we injected. They were
        # saved into storage along with the assistant reply (see save block
        # below) so they re-appear in `messages` next turn; without this
        # filter, every turn would add another stale system message.
        messages = [
            m for m in messages
            if not (isinstance(m, dict)
                    and m.get("role") == "system"
                    and isinstance(m.get("content"), str)
                    and (m["content"].startswith("Use this uploaded file as context:")
                         or m["content"].startswith("Use this retrieved context:")))
        ]

        pending = self._pending_inline_files.get(conversation_id, [])
        if pending:
            try:
                sources_event = {
                    "sources": [
                        {
                            "filename": f["filename"],
                            "excerpt": f["content"][:300],
                            "scope": "upload",
                        }
                        for f in pending
                    ]
                }
                job.append_chunk("sources", json.dumps(sources_event))
                context_str = "\n\n".join(
                    f"[{f['filename']}]:\n{f['content']}" for f in pending
                )
                messages = messages[:-1] + [
                    {"role": "system", "content": f"Use this uploaded file as context:\n\n{context_str}"},
                    messages[-1],
                ]
            except Exception as e:
                logger.exception("Inline file injection failed; continuing without context: %s", e)
        # ──────────────────────────────────────────────────────────────

        # ── RAG (FAISS) pre-processing block ───────────────────────────
        # Only runs when no inline files were attached. Both paths emit a
        # sources event before tokens; the inline path takes precedence so
        # we don't double up.
        elif retrieval is not None and self.rag_service is not None:
            try:
                scoped = self.rag_service.make_scoped_retriever(
                    conversation_id, retrieval.top_k,
                )
                hits = scoped.invoke(message)
                if hits:
                    sources_event = {
                        "sources": [
                            {
                                "filename": h.metadata.get("filename"),
                                "excerpt": h.page_content[:300],
                                "scope": h.metadata.get("source"),
                            }
                            for h in hits
                        ]
                    }
                    job.append_chunk("sources", json.dumps(sources_event))
                    context_str = "\n\n".join(
                        f"[{h.metadata.get('filename')}]: {h.page_content}" for h in hits
                    )
                    messages = messages[:-1] + [
                        {"role": "system", "content": f"Use this retrieved context:\n{context_str}"},
                        messages[-1],
                    ]
            except Exception as e:
                logger.exception("Retrieval failed; continuing without context: %s", e)
        # ──────────────────────────────────────────────────────────────

        try:
            async for chunk in self.chain.astream(messages):
                # If the user deleted the conversation mid-stream, stop early and
                # do NOT call mark_completed or save_conversation — that would
                # resurrect the deleted conversation in storage.
                if job.cancelled:
                    return

                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict) and "content" in chunk:
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
        messages.append({
            "role": "assistant",
            "content": full_content,
            "thinking": full_thinking
        })
        file_storage.save_conversation(conversation_id, messages)

    def get_history(self, conversation_id: str) -> dict | None:
        return file_storage.get_conversation(conversation_id)
