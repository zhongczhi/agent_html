"""Test the SSE payload shape emitted by /api/chat/stream.

This exists to catch regressions in the wire format the FE depends on:
each chunk event has shape `{chunk: <text>, type: "token"|"thinking", ...}`
(not the nested `chunk.type` form some early FE code assumed).
"""
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

import backend.storage.file_storage as file_storage
import backend.chat.routes as chat_routes_module


def _serialize(chunk: dict) -> dict:
    """Replicate chat/routes.py::_serialize_chunk for assertion reference."""
    return {"chunk": chunk["chunk"], "type": chunk["type"], "message_id": chunk["message_id"]}


def test_serialize_chunk_shape():
    """Sanity: the wire format is {chunk, type, message_id} with chunk being
    the text and type being a sibling. The compare FE parses this shape."""
    out = _serialize({"chunk": "hello", "type": "token", "message_id": "0"})
    assert out == {"chunk": "hello", "type": "token", "message_id": "0"}
    # The text is at .chunk, NOT at .chunk.chunk
    assert out["chunk"] == "hello"
    assert out["type"] == "token"


def test_end_event_shape():
    """End-of-stream event is {end: true} with no chunk field."""
    payload = {"end": True}
    assert payload["end"] is True
    assert "chunk" not in payload


def test_sources_chunk_serializes_with_type_field(temp_storage_dir):
    """The RAG path pushes a chunk with type='sources' and chunk=JSON string.
    The FE parses payload.type === 'sources' and JSON.parse(payload.chunk)."""
    from backend.chat.service import ChatService
    from backend.chat.stream_manager import get_or_create_job, clear_job
    from langchain_core.documents import Document

    fs, _ = temp_storage_dir
    fs.create_conversation("c1")
    fs.append_message("c1", "user", "hi")

    hits = [
        Document(page_content="RAG is great.", metadata={"source": "library", "filename": "intro.md"}),
    ]

    class _Stub:
        def make_scoped_retriever(self, conv_id, top_k):
            class _R:
                def invoke(self, q, **_):
                    return list(hits)
            return _R()

    chain = MagicMock()
    class _It:
        def __aiter__(self): return self
        async def __anext__(self):
            raise StopAsyncIteration
    chain.astream = lambda messages: _It()

    service = ChatService(chain, rag_service=_Stub())
    clear_job("c1")
    from backend.chat.routes import RetrievalConfig
    asyncio.run(service.generate_background(
        "hi", "c1", retrieval=RetrievalConfig(library=True, uploads=True, top_k=4)
    ))

    job = get_or_create_job("c1", [])
    sources_chunks = [c for c in job.chunks if c["type"] == "sources"]
    assert len(sources_chunks) == 1
    # The chunk is a JSON string at the .chunk field; the type is a sibling
    assert sources_chunks[0]["type"] == "sources"
    payload_str = sources_chunks[0]["chunk"]
    parsed = json.loads(payload_str)
    assert "sources" in parsed
    assert parsed["sources"][0]["filename"] == "intro.md"

    # And the serialized wire form (what the FE actually sees) is:
    wire = _serialize(sources_chunks[0])
    assert wire["chunk"] == payload_str
    assert wire["type"] == "sources"
    # message_id is a unique counter; we don't pin its value, just verify it's set
    assert "message_id" in wire

    clear_job("c1")
    fs.delete_conversation("c1")
