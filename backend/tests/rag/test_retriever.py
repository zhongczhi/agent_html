from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.rag.retriever import ScopedRetriever


class _FakeRetriever(BaseRetriever):
    """Returns a fixed list of Documents, ignoring the query."""
    docs: List[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return list(self.docs)


def _doc(content: str, *, source: str = "library", conversation_id: str | None = None) -> Document:
    md = {"source": source}
    if conversation_id is not None:
        md["conversation_id"] = conversation_id
    return Document(page_content=content, metadata=md)


def test_merges_two_scopes():
    lib = _FakeRetriever(docs=[_doc("lib-a"), _doc("lib-b")])
    upl = _FakeRetriever(docs=[_doc("upl-a", source="upload", conversation_id="c1")])
    r = ScopedRetriever(
        retrievers=[(lib, False), (upl, True)],
        conversation_id="c1",
    )
    hits = r.invoke("query")
    contents = [d.page_content for d in hits]
    assert "lib-a" in contents
    assert "lib-b" in contents
    assert "upl-a" in contents


def test_uploads_filtered_by_conversation_id():
    lib = _FakeRetriever(docs=[_doc("lib-a")])
    upl = _FakeRetriever(docs=[
        _doc("upl-mine", source="upload", conversation_id="c1"),
        _doc("upl-other", source="upload", conversation_id="c2"),
    ])
    r = ScopedRetriever(
        retrievers=[(lib, False), (upl, True)],
        conversation_id="c1",
    )
    hits = r.invoke("q")
    contents = [d.page_content for d in hits]
    assert "lib-a" in contents
    assert "upl-mine" in contents
    assert "upl-other" not in contents


def test_library_chunks_pass_through_unfiltered():
    """Library chunks have NO conversation_id field. The should_filter=False
    branch means they're never filtered. The asymmetric-metadata mechanism
    is what makes this work — see ScopedRetriever docstring."""
    lib = _FakeRetriever(docs=[
        _doc("lib-1"),  # no conversation_id
        _doc("lib-2"),
    ])
    r = ScopedRetriever(
        retrievers=[(lib, False)],
        conversation_id="any-conv-id",
    )
    hits = r.invoke("q")
    assert {d.page_content for d in hits} == {"lib-1", "lib-2"}


def test_no_merged_result_cap():
    """Per-scope cap is the underlying retrievers' responsibility (via
    search_kwargs). ScopedRetriever does NOT cap the merged result, so
    4 library + 4 uploads = 8 chunks reach the LLM."""
    lib = _FakeRetriever(docs=[_doc(f"lib-{i}") for i in range(4)])
    upl = _FakeRetriever(docs=[
        _doc(f"upl-{i}", source="upload", conversation_id="c1") for i in range(4)
    ])
    r = ScopedRetriever(
        retrievers=[(lib, False), (upl, True)],
        conversation_id="c1",
    )
    hits = r.invoke("q")
    assert len(hits) == 8


def test_works_with_single_scope():
    """Only one retriever in the list — other scopes disabled by config."""
    lib = _FakeRetriever(docs=[_doc("lib-a"), _doc("lib-b")])
    r = ScopedRetriever(
        retrievers=[(lib, False)],
        conversation_id="c1",
    )
    hits = r.invoke("q")
    assert {d.page_content for d in hits} == {"lib-a", "lib-b"}
