"""ChatService RAG integration tests — verify the plugin-off property and
the per-turn retrieval/sources/augmentation behavior."""
import asyncio
import json
from unittest.mock import MagicMock
from langchain_core.documents import Document

import backend.storage.file_storage as file_storage
from backend.chat.service import ChatService
from backend.chat.stream_manager import clear_job, get_or_create_job, get_job


class _StubRagService:
    """Stand-in for RagService used by the chat integration tests.
    Returns a fixed set of hits and records what was requested.

    Hits' metadata.source ("library" | "upload") determines which scope
    retrieve_by_scope returns them under — matches the production behavior
    where the chunk's source_type is set by the loader.
    """
    def __init__(self, hits: list[Document]):
        self.hits = hits
        self.calls: list[tuple[str, str, int]] = []  # (conv_id, query, top_k)

    def retrieve_by_scope(self, conversation_id: str, query: str, top_k: int):
        captured = self
        captured.calls.append((conversation_id, query, top_k))
        by_scope: dict[str, list[Document]] = {"library": [], "uploads": []}
        for h in captured.hits:
            src = h.metadata.get("source")
            if src == "library":
                by_scope["library"].append(h)
            elif src in ("upload", "uploads"):
                by_scope["uploads"].append(h)
        return by_scope

    def make_scoped_retriever(self, conversation_id: str, top_k: int):
        # Kept for tests that specifically exercise the legacy path or
        # want to raise on retrieval (graceful-failure test).
        captured = self

        class _Retriever:
            def invoke(self, query, **_):
                captured.calls.append((conversation_id, query, top_k))
                return list(captured.hits)

        return _Retriever()


def _make_chain():
    """A chain that yields a single token so we can observe the LLM call."""
    chain = MagicMock()

    class _It:
        def __aiter__(self): return self
        async def __anext__(self):
            raise StopAsyncIteration

    # No-op chain that records the messages it was called with
    chain._last_input = None

    async def _stream(messages):
        chain._last_input = list(messages)
        return
        yield  # unreachable, makes this a generator

    # The above doesn't work with async; use a manual async iterator
    class _AsyncIter:
        def __init__(self, msgs):
            self.msgs = msgs
            self.yielded = False
        def __aiter__(self): return self
        async def __anext__(self):
            if self.yielded:
                raise StopAsyncIteration
            self.yielded = True
            chunk = MagicMock()
            chunk.content = [{"type": "text", "text": "answer"}]
            return chunk

    def _astream(messages):
        # The chain is invoked positionally with the messages list. Snapshot
        # the list so later in-place mutation by ChatService (it appends
        # the assistant message after the LLM streams) doesn't leak into
        # what tests see.
        chain._last_input = list(messages)
        return _AsyncIter(messages)

    chain.astream = _astream
    return chain


def _run(coro):
    return asyncio.run(coro)


def test_no_rag_path_does_not_call_rag_service(temp_storage_dir):
    """Plugin-off guarantee: when retrieval is None, RagService is never
    consulted and the chain receives the original (un-augmented) messages.
    This is the load-bearing assertion for the 'selective plugin' property."""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    rag = MagicMock()
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    _run(service.generate_background("hi", "c1", retrieval=None))

    rag.retrieve_by_scope.assert_not_called()
    rag.make_scoped_retriever.assert_not_called()
    # Chain received the original messages (no augmentation)
    assert chain._last_input == [{"role": "user", "content": "hi"}]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_rag_path_pushes_sources_and_augments_messages(temp_storage_dir):
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    hits = [
        Document(
            page_content="RAG stands for retrieval-augmented generation.",
            metadata={"source": "library", "filename": "intro.md"},
        ),
        Document(
            page_content="It pulls chunks at query time.",
            metadata={"source": "upload", "conversation_id": "c1", "filename": "notes.md"},
        ),
    ]
    rag = _StubRagService(hits)
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background("hi", "c1", retrieval=RetrievalConfig(library=True, uploads=True, top_k=4)))

    job = get_job("c1")
    assert job is not None
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert len(sources_chunks) == 1
    payload = json.loads(sources_chunks[0]["chunk"])
    # Per-scope nested dict (library + uploads keys, both possibly empty)
    assert "sources" in payload
    assert set(payload["sources"].keys()) == {"library", "uploads"}
    assert len(payload["sources"]["library"]) == 1
    assert payload["sources"]["library"][0]["filename"] == "intro.md"
    assert len(payload["sources"]["uploads"]) == 1
    assert payload["sources"]["uploads"][0]["filename"] == "notes.md"

    # Rag service was called with the right conversation_id and top_k
    assert rag.calls == [("c1", "hi", 4)]

    # The chain received augmented messages: RAG context is now embedded
    # inside the last user message wrapped in <context>...</context>,
    # AND the RAG-specific system prompt is prepended (because context
    # was used this turn). Vanilla turns send NO system message.
    sent = chain._last_input
    assert len(sent) == 2  # system + user
    sys_msg, user_msg = sent
    assert sys_msg["role"] == "system"
    # The system prompt is the RAG one (mentions <context> tag handling)
    assert "<context>" in sys_msg["content"]
    assert "prefer" in sys_msg["content"].lower()
    assert user_msg["role"] == "user"
    # Context tags wrap the retrieved chunks; both per-scope hits appear
    assert "<context>" in user_msg["content"]
    assert "</context>" in user_msg["content"]
    assert "intro.md" in user_msg["content"]
    assert "notes.md" in user_msg["content"]
    assert "RAG stands for retrieval-augmented generation" in user_msg["content"]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_rag_path_emits_empty_sources_event_when_no_hits(temp_storage_dir):
    """When retrieval runs but returns no hits in either scope, the
    frontend still needs a sources event so it can render the empty-state
    block ("library: 0 / uploads: 0"). Otherwise the user gets no
    feedback about whether RAG was even consulted."""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "obscure query")

    rag = _StubRagService(hits=[])   # empty hits → both scopes return []
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background(
        "obscure query", "c1",
        retrieval=RetrievalConfig(library=True, uploads=True, top_k=4),
    ))

    job = get_job("c1")
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert len(sources_chunks) == 1
    payload = json.loads(sources_chunks[0]["chunk"])
    # Per-scope dict with both keys present and empty arrays
    assert set(payload["sources"].keys()) == {"library", "uploads"}
    assert payload["sources"]["library"] == []
    assert payload["sources"]["uploads"] == []

    # Retrieval was still called, but no hits → no <context>...</context>
    # tag was embedded in the user message. The RAG system prompt is
    # therefore NOT prepended (only sent when there's actual context to
    # ground on — sending it for empty retrievals would waste tokens and
    # bias the LLM toward a context-aware response when there's no
    # context to be aware of).
    assert rag.calls == [("c1", "obscure query", 4)]
    sent = chain._last_input
    assert sent == [{"role": "user", "content": "obscure query"}]
    sys_msgs = [m for m in sent if isinstance(m, dict) and m.get("role") == "system"]
    assert sys_msgs == []

    clear_job("c1")
    fs.delete_conversation("c1")


def test_sources_persisted_in_saved_assistant_message(temp_storage_dir):
    """The sources payload must end up in the persisted conversation so
    renderMessagesFromCache can re-render the block on reload. This is
    what closes the 'sources vanish on reload' gap — without this, the
    frontend cache and backend diverge and reload-from-cache silently
    strips the block."""
    from backend.storage import file_storage
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    hits = [
        Document(
            page_content="RAG stands for retrieval-augmented generation.",
            metadata={"source": "library", "filename": "intro.md"},
        ),
    ]
    rag = _StubRagService(hits)
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background(
        "hi", "c1", retrieval=RetrievalConfig(library=True, uploads=True, top_k=4),
    ))

    # Reload from disk (bypass any cache) — this is what the history
    # endpoint serves on reload, so it must include sources.
    persisted = file_storage.get_conversation("c1")
    assert persisted is not None
    msgs = persisted["messages"]
    # After the Pattern-A refactor, RAG context lives inside the user
    # message wrapped in <context>...</context> — no separate system
    # message is saved. Saved list is [user, assistant]. The user message
    # keeps its tag on disk so subsequent LLM turns see their grounding;
    # get_history strips the tag before sending to the frontend.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert "<context>" in msgs[0]["content"]
    assert "intro.md" in msgs[0]["content"]
    assistant = msgs[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"]  # token content joined
    # sources dict persisted with the right per-scope structure
    assert assistant["sources"] is not None
    assert set(assistant["sources"].keys()) == {"library", "uploads"}
    assert len(assistant["sources"]["library"]) == 1
    assert assistant["sources"]["library"][0]["filename"] == "intro.md"
    assert assistant["sources"]["uploads"] == []

    clear_job("c1")
    fs.delete_conversation("c1")


def test_sources_field_is_none_when_no_retrieval_ran(temp_storage_dir):
    """When retrieval=None (vanilla chat) or rag_service is None, no
    sources chunk is emitted, so the persisted message must have
    sources=None (not an empty dict, not missing). The frontend uses
    `if (msg.sources)` to decide whether to render — None and missing
    both skip, but explicit None is the documented contract."""
    from backend.storage import file_storage
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    chain = _make_chain()
    service = ChatService(chain, rag_service=None)

    _run(service.generate_background("hi", "c1", retrieval=None))

    persisted = file_storage.get_conversation("c1")
    assistant = persisted["messages"][-1]
    assert assistant["role"] == "assistant"
    # No retrieval → no sources chunk → sources key present with value None
    assert "sources" in assistant
    assert assistant["sources"] is None

    clear_job("c1")
    fs.delete_conversation("c1")


def test_get_history_strips_context_tags_from_user_messages(temp_storage_dir):
    """Disk keeps user messages with <context>...</context> tags intact so
    subsequent LLM turns see their grounding context. The frontend must
    never see those tags — ChatService.get_history strips them before
    returning. This closes the 'Use this retrieved context:' chat-bubble
    leak that surfaced on reload for any conversation that used RAG.
    """
    from backend.storage import file_storage
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    # Seed a tagged user message directly on disk (simulating what
    # generate_background would have saved).
    tagged = "<context>\n[doc.md]: some retrieved fact\n</context>\n\nwhat does the doc say?"
    fs.append_message("c1", "user", tagged)
    fs.append_message("c1", "assistant", "the doc says X")

    # get_history must strip the tag for the frontend
    chain = _make_chain()
    service = ChatService(chain, rag_service=None)
    view = service.get_history("c1")
    assert view is not None
    user_msg = view["messages"][0]
    assert user_msg["role"] == "user"
    assert "<context>" not in user_msg["content"]
    assert "</context>" not in user_msg["content"]
    assert "some retrieved fact" not in user_msg["content"]
    # The original user query remains
    assert "what does the doc say?" in user_msg["content"]
    # Assistant message passes through untouched
    assert view["messages"][1] == {"role": "assistant", "content": "the doc says X"}

    # But disk still has the tagged form (for LLM context on next turn)
    raw = file_storage.get_conversation("c1")
    assert "<context>" in raw["messages"][0]["content"]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_rag_path_with_no_rag_service_acts_like_vanilla(temp_storage_dir):
    """Defensive: if retrieval is set but rag_service is None (RAG_ENABLED
    false at startup but a client sent retrieval anyway), server falls
    through to vanilla. No crash, no retrieval, chain gets original messages."""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    chain = _make_chain()
    service = ChatService(chain, rag_service=None)

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background("hi", "c1", retrieval=RetrievalConfig()))

    job = get_job("c1")
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert sources_chunks == []
    assert chain._last_input == [{"role": "user", "content": "hi"}]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_rag_path_handles_retrieval_failure_gracefully(temp_storage_dir):
    """If RagService.retrieve_by_scope raises, the LLM call still runs
    with the original (un-augmented) messages — no crash."""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    class _BoomRag:
        def retrieve_by_scope(self, conv_id, query, top_k):
            raise RuntimeError("retrieval failed")

    chain = _make_chain()
    service = ChatService(chain, rag_service=_BoomRag())

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background("hi", "c1", retrieval=RetrievalConfig()))

    # No sources chunk pushed (the raise was caught and swallowed before emit)
    job = get_job("c1")
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert sources_chunks == []
    # Chain still ran with the un-augmented messages
    assert chain._last_input == [{"role": "user", "content": "hi"}]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_inline_files_path_injects_content_and_skips_retrieval(temp_storage_dir):
    """When uploaded_files is non-empty, ChatService injects file content as
    a system message and skips FAISS retrieval — the two paths are mutually
    exclusive per turn. (FR-12.6 / 12.8)"""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "summarize this")

    rag = MagicMock()
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    from backend.chat.routes import UploadedFile
    _run(service.generate_background(
        "summarize this",
        "c1",
        retrieval=None,
        uploaded_files=[UploadedFile(filename="notes.txt", content="important context")],
    ))

    # FAISS retrieval was NOT consulted
    rag.retrieve_by_scope.assert_not_called()
    rag.make_scoped_retriever.assert_not_called()

    job = get_job("c1")
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert len(sources_chunks) == 1
    payload = json.loads(sources_chunks[0]["chunk"])
    # Inline path emits dict format with only "uploads" populated
    assert set(payload["sources"].keys()) == {"uploads"}
    assert len(payload["sources"]["uploads"]) == 1
    assert payload["sources"]["uploads"][0]["filename"] == "notes.txt"

    sent = chain._last_input
    # Inline path also uses per-turn context, so the RAG system prompt
    # is prepended. Inline file content lives inside the last user
    # message wrapped in <context>...</context>.
    assert len(sent) == 2  # system + user
    sys_msg, user_msg = sent
    assert sys_msg["role"] == "system"
    assert "<context>" in sys_msg["content"]
    assert user_msg["role"] == "user"
    assert "<context>" in user_msg["content"]
    assert "notes.txt" in user_msg["content"]
    assert "important context" in user_msg["content"]
    # The original user query follows the context block
    assert "summarize this" in user_msg["content"]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_inline_files_persist_across_turns(temp_storage_dir):
    """Files uploaded in turn 1 stay available for turn 2 without re-sending.
    Subsequent turns without uploaded_files still see them. (FR-12.6)"""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "first")

    chain = _make_chain()
    service = ChatService(chain, rag_service=None)

    from backend.chat.routes import UploadedFile
    # Turn 1: send with the file
    _run(service.generate_background(
        "first", "c1",
        retrieval=None,
        uploaded_files=[UploadedFile(filename="x.txt", content="the data")],
    ))
    fs.append_message("c1", "assistant", "ok")

    # Turn 2: no uploaded_files, but the conversation's pending list still has x.txt
    _run(service.generate_background("second", "c1", retrieval=None, uploaded_files=None))

    sent = chain._last_input
    # Inline path uses per-turn context → RAG system prompt is prepended.
    # Turn 2's chain input: [system, user (turn 1, with context), assistant
    # (turn 1), assistant (manually appended between turns)]. The
    # snapshot in _make_chain captures the messages list at call time.
    assert sent[0]["role"] == "system"
    assert "<context>" in sent[0]["content"]
    user_msgs = [m for m in sent if m["role"] == "user"]
    assert user_msgs, "chain input should contain at least one user message"
    # The pending inline file (x.txt) is still available in turn 2 and is
    # embedded as <context>...</context> in the LAST USER message in the
    # chain input. Since this test bypasses the route (no user:"second"
    # appended to storage), the last user message is still turn-1's
    # "first" — context is embedded there because it's the only user
    # message the chain can attach to. Either way, the inline file
    # persisted across turns — that's the load-bearing assertion.
    last_user = user_msgs[-1]
    assert "<context>" in last_user["content"]
    assert "x.txt" in last_user["content"]
    assert "the data" in last_user["content"]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_inline_files_take_precedence_over_retrieval(temp_storage_dir):
    """When BOTH uploaded_files and retrieval are set, the inline path wins.
    FAISS retrieval is skipped to avoid double-grounding. (FR-12.8)"""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    rag = MagicMock()
    chain = _make_chain()
    service = ChatService(chain, rag_service=rag)

    from backend.chat.routes import RetrievalConfig, UploadedFile
    _run(service.generate_background(
        "hi", "c1",
        retrieval=RetrievalConfig(),
        uploaded_files=[UploadedFile(filename="x.txt", content="x")],
    ))

    rag.retrieve_by_scope.assert_not_called()
    rag.make_scoped_retriever.assert_not_called()
    sent = chain._last_input
    # Inline path uses context → RAG system prompt is prepended.
    assert len(sent) == 2
    sys_msg, user_msg = sent
    assert sys_msg["role"] == "system"
    assert "<context>" in sys_msg["content"]
    # Inline path: context block contains the uploaded file's content
    assert user_msg["role"] == "user"
    assert "<context>" in user_msg["content"]
    assert "x.txt" in user_msg["content"]

    clear_job("c1")
    fs.delete_conversation("c1")


def test_clear_pending_inline_files_drops_conversation_entry(temp_storage_dir):
    """clear_pending_inline_files(conversation_id) drops only that
    conversation's pending list; other conversations' lists are untouched."""
    fs, _ = temp_storage_dir
    chain = _make_chain()
    service = ChatService(chain, rag_service=None)

    from backend.chat.routes import UploadedFile
    # Seed two conversations with pending files
    for cid in ("c1", "c2"):
        fs.create_conversation(cid)
        fs.append_message(cid, "user", "hi")
        _run(service.generate_background("hi", cid, retrieval=None, uploaded_files=[
            UploadedFile(filename=f"{cid}.txt", content=f"data for {cid}"),
        ]))

    assert "c1" in service._pending_inline_files
    assert "c2" in service._pending_inline_files

    service.clear_pending_inline_files("c1")
    assert "c1" not in service._pending_inline_files
    assert "c2" in service._pending_inline_files
