from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from backend.rag.vector_store import load_or_init, save, rebuild_filtered


def _docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"doc-{i}", metadata={"i": i, "source": "upload", "conversation_id": "c1"})
        for i in range(n)
    ]


def test_load_or_init_creates_empty_index_when_no_files(tmp_path: Path):
    path = tmp_path / "idx"
    embeddings = FakeEmbeddings(size=8)
    index = load_or_init(path, embeddings)
    assert isinstance(index, FAISS)
    # Placeholder present (FAISS requires ≥1 vector to be valid)
    docs = list(index.docstore._dict.values())
    assert len(docs) == 1
    assert docs[0].metadata.get("_placeholder") is True


def test_save_and_load_roundtrip_preserves_documents(tmp_path: Path):
    path = tmp_path / "idx"
    embeddings = FakeEmbeddings(size=8)
    index = FAISS.from_documents(_docs(3), embeddings)
    save(index, path)

    assert (path / "index.faiss").exists()

    loaded = load_or_init(path, embeddings)
    loaded_docs = sorted(
        d.page_content for d in loaded.docstore._dict.values()
        if not d.metadata.get("_placeholder")
    )
    assert loaded_docs == ["doc-0", "doc-1", "doc-2"]


def test_rebuild_filtered_keeps_only_matching_docs(tmp_path: Path):
    embeddings = FakeEmbeddings(size=8)
    docs = [
        Document(page_content="keep", metadata={"conversation_id": "c1"}),
        Document(page_content="drop", metadata={"conversation_id": "c2"}),
        Document(page_content="keep2", metadata={"conversation_id": "c1"}),
    ]
    index = FAISS.from_documents(docs, embeddings)
    rebuilt = rebuild_filtered(
        index, embeddings,
        keep=lambda d: d.metadata.get("conversation_id") == "c1",
    )
    remaining = sorted(d.page_content for d in rebuilt.docstore._dict.values())
    assert remaining == ["keep", "keep2"]


def test_rebuild_filtered_handles_empty_survivors(tmp_path: Path):
    """When no docs match the keep predicate, the result is still a valid
    (but empty/placeholder) FAISS index — not an exception."""
    embeddings = FakeEmbeddings(size=8)
    docs = [Document(page_content="drop-me", metadata={"k": 1})]
    index = FAISS.from_documents(docs, embeddings)
    rebuilt = rebuild_filtered(
        index, embeddings,
        keep=lambda d: d.metadata.get("k") == 999,
    )
    assert isinstance(rebuilt, FAISS)
    surviving = [d for d in rebuilt.docstore._dict.values() if not d.metadata.get("_placeholder")]
    assert surviving == []
