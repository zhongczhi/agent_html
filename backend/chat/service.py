import json
import logging

from backend.storage import file_storage
from backend.chat.stream_manager import get_or_create_job, get_job

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, chain, rag_service=None):
        self.chain = chain
        self.rag_service = rag_service

    async def generate_background(
        self,
        message: str,
        conversation_id: str,
        retrieval=None,
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

        # ── RAG pre-processing block ──────────────────────────────────
        if retrieval is not None and self.rag_service is not None:
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
