"""SHA-keyed per-question FAISS cache for the eval pipeline."""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from backend.eval.hotpotqa import HotpotQaItem
from backend.rag.vector_store import load_or_init, save

log = logging.getLogger(__name__)

# Default cache root. Tests monkeypatch this attribute to a tmp dir.
# Resolved relative to <repo>/backend/ so it sits under the same tree as
# the RAG library / uploads / index dirs (see backend/rag/service.py).
EVAL_CACHE_ROOT = Path(__file__).parent.parent / "storage" / "eval" / "hotpotqa" / "cache"


def _build_index(item: HotpotQaItem, embeddings: Embeddings) -> FAISS:
    """Construct a FAISS index over `item.context` as paragraph Documents.
    Skips paragraphs whose joined sentence text is empty. Does NOT use
    MarkdownTextSplitter — preserves paragraph granularity so the metric
    is stable across any chat-side chunking-config change."""
    docs: list[Document] = []
    for idx, (title, sentences) in enumerate(item.context):
        text = " ".join(sentences).strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "question_id": item.id,
                    "paragraph_idx": idx,
                    "title": title,
                    "source": "hotpotqa",
                    "type": item.type,
                    "level": item.level,
                },
            )
        )
    return FAISS.from_documents(docs, embeddings)


def embedding_tag(embeddings: Embeddings) -> str:
    """Best-effort stable identifier for an embedding model.

    Some embedders (HuggingFace sentence-transformers) carry a `model_name`
    attribute; others expose `model`; the rest fall back to the dimension
    of a single probe query. If even that fails, the class name is used.

    The output is intended to be a stable string that distinguishes between
    models that produce different vector spaces (and therefore must not
    share a FAISS index). It is not a cryptographic fingerprint.
    """
    # 1. HuggingFace-style model_name attribute (the common case).
    for attr in ("model_name", "model"):
        if hasattr(embeddings, attr):
            value = getattr(embeddings, attr)
            if isinstance(value, str) and value:
                return value.replace("/", "_").replace("\\", "_")
    # 2. Probe the embedder's output dimension. This distinguishes models
    # with different vector sizes (MiniLM 384 vs mpnet 768) but not two
    # models that happen to produce the same size.
    try:
        vec = embeddings.embed_query("embedding-tag-probe")
        return f"dim{len(vec)}"
    except Exception:
        pass
    # 3. Class name fallback. Two unrelated embedders that share a class
    # name (e.g., FakeEmbeddings in tests) will collide here — the test
    # suite uses no_cache=True to avoid the issue.
    return type(embeddings).__name__


def load_or_build(
    item: HotpotQaItem,
    dataset_sha: str,
    embeddings: Embeddings,
    no_cache: bool = False,
    embedding_tag_override: str | None = None,
) -> tuple[FAISS, bool]:
    """Returns (index, was_hit). `was_hit` is True if loaded from disk.

    Cache layout: EVAL_CACHE_ROOT / {dataset_sha}_{embedding_tag} / item.id /
    The embedding_tag distinguishes indices built with different models so
    a switch from MiniLM to mpnet does not silently reuse a stale 384-dim
    FAISS index. Pass `embedding_tag_override` to force a specific tag
    (e.g., 'mpnet' or 'fake64'); otherwise `embedding_tag(embeddings)` is
    called to derive one.

    - If `no_cache`, always build. Cache is overwritten on disk.
    - Otherwise: cache_path = EVAL_CACHE_ROOT / dataset_sha_tag / item.id /
      - if missing, build + save, was_hit=False.
      - if present, attempt load_or_init; on any failure, rmtree + build + save
        + WARNING log, was_hit=False.
    """
    tag = embedding_tag_override or embedding_tag(embeddings)
    cache_dir = EVAL_CACHE_ROOT / f"{dataset_sha}_{tag}" / item.id
    if no_cache or not cache_dir.exists():
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        return index, False
    try:
        index = load_or_init(cache_dir, embeddings)
        return index, True
    except Exception as e:
        log.warning("cache corrupt for %s (%s); rebuilding", item.id, e)
        shutil.rmtree(cache_dir, ignore_errors=True)
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        return index, False
