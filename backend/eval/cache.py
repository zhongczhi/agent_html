"""SHA-keyed per-question FAISS cache for the eval pipeline."""
from __future__ import annotations

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


def load_or_build(
    item: HotpotQaItem,
    dataset_sha: str,
    embeddings: Embeddings,
    no_cache: bool = False,
) -> tuple[FAISS, bool]:
    """Returns (index, was_hit). `was_hit` is True if loaded from disk.

    - If `no_cache`, always build. Cache is overwritten on disk.
    - Otherwise: cache_path = EVAL_CACHE_ROOT / dataset_sha / item.id /
      - if missing, build + save, was_hit=False.
      - if present, attempt load_or_init; on any failure, rmtree + build + save
        + WARNING log, was_hit=False.
    """
    cache_dir = EVAL_CACHE_ROOT / dataset_sha / item.id
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
