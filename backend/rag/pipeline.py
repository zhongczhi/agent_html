"""Pluggable RAG pipeline.

A RAG pipeline is composed of four stages, each swappable independently:

    [embedder] -> [retriever] -> [reranker?] -> [prompt_builder] -> [LLM]

`PipelineConfig` names each stage's implementation. `build_pipeline(config)`
returns a `RagPipeline` ready to call `run(question, corpus)`. Preset
configurations live in `PRESETS`.

Adding a new pipeline variant (e.g., a different reranker or a new prompt
template) means adding one preset entry — no other code changes.

Design notes
------------
- The pipeline is intentionally a thin orchestration layer. Heavy lifting
  (FAISS index construction, LLM calls, prompt formatting) lives in the
  existing `backend.eval.*` and `backend.rag.*` modules. This module
  composes them.
- `corpus: list[Document]` is passed in by the caller. This makes the
  pipeline testable with synthetic corpora and reusable across the
  eval script (per-question FAISS) and the chat service (library index).
- The reranker slot is optional. When `reranker=None`, the retriever's
  output is sliced directly to `top_k`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ── Pipeline config ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    """Names each stage's implementation.

    Attributes:
        name: human-readable preset name (matches a key in `PRESETS`).
        embedding_backend: passed to `make_embeddings()`. One of
            'sentence-transformers' or 'minimax'.
        embedding_model: model name for sentence-transformers backend.
            Ignored when embedding_backend != 'sentence-transformers'.
        retriever: retriever kind. One of 'dense' (FAISS top-k).
            Future: 'hybrid_bm25_dense', 'multi_query', etc.
        reranker: reranker kind. None or 'cross_encoder'.
        rerank_top_k: how many candidates the retriever returns before
            the reranker narrows to `top_k`. Ignored when reranker is None.
        top_k: final number of chunks sent to the LLM.
        prompt_template: prompt template name. 'default' or 'extract_span'.
        llm_model: model name for the LLM call (passed to qa_judge.ask_llm).
    """

    name: str
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever: str = "dense"
    reranker: str | None = None
    rerank_top_k: int = 50
    top_k: int = 4
    prompt_template: str = "default"
    llm_model: str = "minimax-3"


# ── Component protocols (duck-typed; concrete impls in this module) ──────


class Retriever(Protocol):
    """Returns up to `k` candidate Documents for a query."""

    def retrieve(self, query: str, k: int) -> list[Document]: ...


class Reranker(Protocol):
    """Re-orders `candidates` for `query` and returns the top `top_k`."""

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]: ...


class PromptBuilder(Protocol):
    """Builds the messages list passed to the LLM."""

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]: ...


class LLM(Protocol):
    """One LLM call returning answer text."""

    async def ask(self, messages: list[dict], max_tokens: int = 200) -> str: ...


# ── Concrete implementations ─────────────────────────────────────────────


class DenseRetriever:
    """FAISS-based dense retriever over a corpus built once at construction.

    `corpus` is a list of (id, embedding) pairs OR a list of Documents with
    pre-computed embeddings via `embed_documents`. For simplicity here we
    take pre-built FAISS-like objects (anything with `.similarity_search`).
    """

    def __init__(self, vectorstore):
        self._vs = vectorstore

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self._vs.similarity_search(query, k=k)


class BM25Retriever:
    """Sparse keyword retriever over a corpus of Documents.

    Wraps rank_bm25.BM25Okapi. The corpus is tokenized once at construction
    (lowercase, alphanumeric-only). At query time, the query is tokenized
    the same way and scored against every doc.

    Pure-Python, deterministic, no model download. Slower than FAISS for
    large corpora but adequate for per-question eval workloads (10 docs each).
    """

    def __init__(self, docs: list[Document]):
        import re

        from rank_bm25 import BM25Okapi

        self._docs = list(docs)
        # Tokenize: lowercase, strip non-word chars (except spaces), split.
        # We keep numbers and punctuation-free tokens.
        self._tokenize = lambda s: [
            t for t in re.findall(r"[a-z0-9]+", s.lower()) if t
        ]
        if self._docs:
            tokenized_corpus = [self._tokenize(d.page_content) for d in self._docs]
            # BM25Okapi raises on empty docs (rare but possible if a paragraph
            # is just punctuation). Replace empty token lists with a placeholder.
            self._bm25 = BM25Okapi(
                [toks if toks else ["_empty_"] for toks in tokenized_corpus]
            )
        else:
            self._bm25 = None

    def retrieve(self, query: str, k: int) -> list[Document]:
        if not self._docs or self._bm25 is None:
            return []
        q_tokens = self._tokenize(query) or ["_empty_"]
        scores = self._bm25.get_scores(q_tokens)
        # Top-k by score (descending). We keep all docs by score; the
        # >0 filter used to drop zero-scored docs but with very small
        # corpora BM25 returns 0 for every term (each term appears in
        # every doc, IDF -> 0). For our 10-paragraph eval workload this
        # rarely matters, but the filter is too aggressive for tiny cases.
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._docs[i] for i in ranked_indices[:k]]


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) of dense + BM25 retrieval.

    For each query, get top-K from both retrievers, then fuse ranks with
    RRF: score(d) = sum(1 / (rrf_k + rank_in_list)). RRF doesn't require
    learning weights and is robust to score-scale differences between
    the two retrievers. The dense_retriever is expected to be a
    `DenseRetriever`; the bm25_retriever a `BM25Retriever`.

    Returns the top `k` documents by fused RRF score.
    """

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[Document]:
        # Pull more candidates than k from each retriever so fusion has
        # room to express itself. We use 4k as the candidate depth — the
        # RRF paper recommends ~3-5x the desired depth.
        cand_k = max(k * 4, 20)
        dense_hits = self._dense.retrieve(query, k=cand_k)
        bm25_hits = self._bm25.retrieve(query, k=cand_k)

        scores: dict[int, float] = {}
        docs_by_id: dict[int, Document] = {}

        for rank, doc in enumerate(dense_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc
        for rank, doc in enumerate(bm25_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [docs_by_id[doc_id] for doc_id, _ in ranked[:k]]


class NoOpReranker:
    """Returns the first `top_k` candidates unchanged."""

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker over (query, document) pairs.

    Uses sentence-transformers CrossEncoder. Falls back to the input order
    if the model fails to load — but logs a warning so operators see it.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # lazy-loaded on first call

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: WPS433
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        if not candidates:
            return []
        self._ensure_model()
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._model.predict(pairs)
        # Sort by score descending, take top_k.
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_k]]


class DefaultPromptBuilder:
    """Wraps backend.eval.qa_judge.build_qa_prompt (RAG context + question)."""

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        # Imported here to avoid a top-level import cycle (qa_judge.py
        # is in backend.eval.* and must not import from backend.rag.*).
        from backend.eval.qa_judge import build_qa_prompt
        return build_qa_prompt(question, context_docs)


class ExtractSpanPromptBuilder:
    """Same shape as Default, but instructs the LLM to extract the answer
    span verbatim from the context first.

    This typically increases answer_f1 (more literal spans) at the cost of
    sounding less conversational. Use when downstream eval is F1-based.
    """

    EXTRACT_INSTRUCTION = (
        "Read the <context>...</context> block carefully and extract the "
        "exact span that answers the question. Begin your response with the "
        "extracted span (in quotation marks if it is a phrase), then briefly "
        "explain. Do not paraphrase the answer — quote it verbatim."
    )

    def __init__(self):
        # Local mirror of RAG_SYSTEM_PROMPT — see backend.eval.qa_judge for
        # the isolation rationale. We append the extract instruction.
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.EXTRACT_INSTRUCTION

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class AnthropicLLM:
    """Async Anthropic client wrapped as an LLM protocol.

    Uses the same prompt shape as the eval pipeline (qa_judge.ask_llm) so
    pipelines and evals produce identical outputs given identical inputs.
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    async def ask(self, messages: list[dict], max_tokens: int = 200) -> str:
        from backend.eval.qa_judge import ask_llm
        return await ask_llm(self._client, self._model, messages, max_tokens=max_tokens)


# ── Factory ──────────────────────────────────────────────────────────────


def build_embedder(backend: str, model_name: str) -> Embeddings:
    """Dispatch to the named embedding backend.

    Thin wrapper over backend.rag.embeddings.make_embeddings that takes the
    model_name as a parameter (the upstream helper accepts it too but the
    default name is "all-MiniLM-L6-v2").
    """
    from backend.rag.embeddings import make_embeddings
    return make_embeddings(backend, model_name=model_name)


def build_retriever(config: PipelineConfig, vectorstore, corpus: list[Document] | None = None) -> Retriever:
    """Dispatch on config.retriever.

    For 'dense' and 'hybrid', `vectorstore` is the FAISS vectorstore built
    over the corpus. For 'hybrid', `corpus` is also required (must be the
    list of Documents that built the FAISS index, in the same order).
    """
    if config.retriever == "dense":
        return DenseRetriever(vectorstore)
    if config.retriever == "hybrid":
        if corpus is None:
            raise ValueError(
                "Hybrid retriever requires the raw corpus (list of Documents) "
                "to build the BM25 index. Pass corpus=... to build_pipeline."
            )
        return HybridRetriever(
            dense_retriever=DenseRetriever(vectorstore),
            bm25_retriever=BM25Retriever(corpus),
        )
    raise ValueError(f"Unknown retriever: {config.retriever!r}")


def build_reranker(config: PipelineConfig) -> Reranker | None:
    if config.reranker is None:
        return None
    if config.reranker == "cross_encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {config.reranker!r}")


def build_prompt_builder(config: PipelineConfig) -> PromptBuilder:
    if config.prompt_template == "default":
        return DefaultPromptBuilder()
    if config.prompt_template == "extract_span":
        return ExtractSpanPromptBuilder()
    raise ValueError(f"Unknown prompt_template: {config.prompt_template!r}")


def build_llm(config: PipelineConfig, client) -> LLM:
    return AnthropicLLM(client=client, model=config.llm_model)


# ── Pipeline orchestrator ────────────────────────────────────────────────


class RagPipeline:
    """Orchestrates retriever -> reranker -> prompt -> LLM for a query.

    The retriever and LLM are provided at construction time (so they can
    hold expensive state like FAISS indices or API clients). The reranker
    and prompt builder are stateless and built lazily on first run.
    """

    def __init__(
        self,
        config: PipelineConfig,
        retriever: Retriever,
        reranker: Reranker | None,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ):
        self.config = config
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm

    async def run(self, question: str) -> str:
        """One end-to-end RAG call.

        Step 1: retrieve up to `rerank_top_k` candidates (or `top_k` if
                no reranker).
        Step 2: optionally rerank to `top_k`.
        Step 3: build the prompt.
        Step 4: ask the LLM.
        """
        # Step 1 — retrieve.
        retrieve_k = (
            self.config.rerank_top_k if self._reranker is not None else self.config.top_k
        )
        candidates = self._retriever.retrieve(question, k=retrieve_k)

        # Step 2 — rerank (or slice).
        if self._reranker is not None:
            final_docs = self._reranker.rerank(question, candidates, top_k=self.config.top_k)
        else:
            final_docs = candidates[: self.config.top_k]

        # Step 3 — prompt.
        messages = self._prompt_builder.build(question, final_docs)

        # Step 4 — ask.
        return await self._llm.ask(messages)


def build_pipeline(
    config: PipelineConfig,
    vectorstore,
    llm_client,
    corpus: list[Document] | None = None,
) -> RagPipeline:
    """Top-level factory. The user-visible one-switch API.

    Example:
        config = PRESETS["naive_dense"]
        pipeline = build_pipeline(config, vectorstore=my_faiss, llm_client=client)
        answer = await pipeline.run("What year was X born?")

    For hybrid presets (`hybrid_bm25_dense`), `corpus` must be the list of
    Documents that built the FAISS index, in the same order. For dense
    presets, corpus is ignored.
    """
    return RagPipeline(
        config=config,
        retriever=build_retriever(config, vectorstore, corpus=corpus),
        reranker=build_reranker(config),
        prompt_builder=build_prompt_builder(config),
        llm=build_llm(config, llm_client),
    )


# ── Presets ──────────────────────────────────────────────────────────────


PRESETS: dict[str, PipelineConfig] = {
    # Existing baseline: small embedding model, no reranker, default prompt.
    "naive_dense": PipelineConfig(
        name="naive_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Bigger embedding model, no reranker — measure embedding-model lift alone.
    "large_dense": PipelineConfig(
        name="large_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Bigger embedding + cross-encoder reranker — full precision push.
    "dense_then_ce": PipelineConfig(
        name="dense_then_ce",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker="cross_encoder",
        rerank_top_k=50,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Prompt-only change: same retrieval, but ask LLM to extract spans
    # verbatim. Useful for boosting answer_f1 in benchmarks.
    "extract_span_prompt": PipelineConfig(
        name="extract_span_prompt",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # Combined: bigger context (top_k=8) + verbatim-span extraction. This
    # composes the two extraction-side wins from iter-14: more paragraphs
    # in context covers the long tail of gold-paragraphs-outside-top-4,
    # and verbatim-span guidance reduces LLM extraction error.
    "extract_span_k8": PipelineConfig(
        name="extract_span_k8",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=8,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # Same as extract_span_k8 but with the full 10-paragraph context.
    # Only marginally better than k=8 on HotpotQA but useful when the
    # supporting-fact set spans more paragraphs than top-k=8 covers.
    "extract_span_k10": PipelineConfig(
        name="extract_span_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # Hybrid BM25 + dense via Reciprocal Rank Fusion. Different lever than
    # embedding-model size: BM25 catches exact entity-name matches the
    # embedding model glosses over, dense catches paraphrase matches BM25
    # misses. RRF combines without learning weights.
    "hybrid_bm25_dense": PipelineConfig(
        name="hybrid_bm25_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="hybrid",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
}


def list_presets() -> list[str]:
    """Return the names of all available presets (sorted)."""
    return sorted(PRESETS.keys())