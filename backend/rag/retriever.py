from langchain_core.retrievers import BaseRetriever


class ScopedRetriever(BaseRetriever):
    """Wraps multiple retrievers, filtering specific ones by metadata.

    Per-scope cap is applied by the underlying retrievers (via
    search_kwargs={"k": k}) — ScopedRetriever does NOT cap the merged
    result, so the LLM can see top_k hits from each scope (e.g., 4 library
    + 4 uploads = 8 total when both scopes are enabled).

    Convention: library chunks are tagged with metadata.source == "library"
    and have NO "conversation_id" field. Upload chunks have both. The
    filter `metadata.get("conversation_id") == self.conversation_id` is
    False for library chunks, so they pass through unfiltered under the
    should_filter=False branch. Do not add a "conversation_id" field to
    library chunks; the asymmetric metadata is the mechanism that makes
    the "library is global, uploads are per-conversation" property work.
    """
    retrievers: list[tuple[BaseRetriever, bool]]   # (retriever, should_filter_by_conv)
    conversation_id: str

    def _get_relevant_documents(self, query, *, run_manager):
        hits = []
        for r, should_filter in self.retrievers:
            r_hits = r.invoke(query)
            r_hits = [d for d in r_hits if not d.metadata.get("_placeholder")]
            if should_filter:
                r_hits = [d for d in r_hits
                          if d.metadata.get("conversation_id") == self.conversation_id]
            hits.extend(r_hits)
        return hits
