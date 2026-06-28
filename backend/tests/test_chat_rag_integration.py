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
    Returns a fixed set of hits and records what was requested."""
    def __init__(self, hits: list[Document]):
        self.hits = hits
        self.calls: list[tuple[str, str, int]] = []  # (conv_id, query, top_k)

    def make_scoped_retriever(self, conversation_id: str, top_k: int):
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
    assert len(payload["sources"]) == 2
    assert payload["sources"][0]["filename"] == "intro.md"
    assert payload["sources"][0]["scope"] == "library"
    assert payload["sources"][1]["filename"] == "notes.md"
    assert payload["sources"][1]["scope"] == "upload"

    # Rag service was called with the right conversation_id and top_k
    assert rag.calls == [("c1", "hi", 4)]

    # The chain received augmented messages: original + system message before last user
    sent = chain._last_input
    # sent is a list of BaseMessage objects (we mock convert_messages implicitly;
    # here the chain is called with the raw dict list)
    # Filter to dict-like check
    sys_msgs = [m for m in sent if isinstance(m, dict) and m.get("role") == "system"]
    assert len(sys_msgs) == 1
    assert "retrieval-augmented" in sys_msgs[0]["content"]
    # Last message is still the user message
    assert sent[-1] == {"role": "user", "content": "hi"}

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
    """If RagService.make_scoped_retriever.invoke raises, the LLM call still
    runs with the original (un-augmented) messages — no crash."""
    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    class _BoomRag:
        def make_scoped_retriever(self, conv_id, top_k):
            class _R:
                def invoke(self, q, **_):
                    raise RuntimeError("retrieval failed")
            return _R()

    chain = _make_chain()
    service = ChatService(chain, rag_service=_BoomRag())

    from backend.chat.routes import RetrievalConfig
    _run(service.generate_background("hi", "c1", retrieval=RetrievalConfig()))

    # No sources chunk pushed
    job = get_job("c1")
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert sources_chunks == []
    # Chain still ran with the un-augmented messages
    assert chain._last_input == [{"role": "user", "content": "hi"}]

    clear_job("c1")
    fs.delete_conversation("c1")
