"""One-shot RAG CLI.

Run a single question through a named pipeline preset and print the answer.

This is the most direct way to test a pipeline variant without the overhead
of the full eval loop. Useful for debugging prompts, comparing embeddings,
or sanity-checking a reranker.

Examples:
    python scripts/run_rag.py --pipeline naive_dense \\
        --question "What year was John Smith born?" \\
        --library-dir storage/library

    python scripts/run_rag.py --pipeline dense_then_ce \\
        --question "What year was John Smith born?"

The library corpus is loaded once and held in memory. For large libraries
this can be slow — pass --no-index-cache to rebuild every time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anthropic import AsyncAnthropic  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

from backend.rag.config import RagSettings  # noqa: E402
from backend.rag.embeddings import make_embeddings  # noqa: E402
from backend.rag.pipeline import PRESETS, build_pipeline, list_presets  # noqa: E402
from backend.rag.splitter import split_into_documents  # noqa: E402

log = logging.getLogger("run_rag")

ALLOWED_EXTS = {".md", ".txt", ".pdf", ".html", ".docx", ".csv"}


def _walk_library(library_dir: Path):
    if not library_dir.exists():
        return []
    return sorted(
        p for p in library_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    )


def build_corpus_from_library(library_dir: Path, chunk_size: int, chunk_overlap: int):
    """Walk library_dir and split every file into Documents."""
    files = _walk_library(library_dir)
    docs = []
    for path in files:
        try:
            docs.extend(split_into_documents(
                path,
                source_type="library",
                conversation_id=None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ))
        except Exception as e:
            log.warning("Skipping %s: %s", path, e)
    return docs


async def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run a single question through a RAG pipeline preset.")
    parser.add_argument("--pipeline", help="Pipeline preset name (use --list to see options)")
    parser.add_argument("--question", help="The question to ask")
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("storage/library"),
        help="Directory to load corpus from (default: storage/library)",
    )
    parser.add_argument("--no-corpus", action="store_true", help="Skip corpus loading (test the pipeline with empty context)")
    parser.add_argument(
        "--list", action="store_true", help="List available pipeline presets and exit."
    )
    args = parser.parse_args(argv)

    if args.list:
        print("Available pipeline presets:")
        for name in list_presets():
            cfg = PRESETS[name]
            print(f"  {name:<24} embed={cfg.embedding_model:<24} rerank={cfg.reranker or 'none':<14} prompt={cfg.prompt_template}")
        return 0

    if not args.pipeline or not args.question:
        parser.error("--pipeline and --question are required (or use --list)")

    if args.pipeline not in PRESETS:
        print(f"Unknown pipeline: {args.pipeline!r}. Available: {sorted(PRESETS.keys())}", file=sys.stderr)
        return 1

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY env var is not set.", file=sys.stderr)
        return 1

    cfg = PRESETS[args.pipeline]
    print(f"[run_rag] pipeline={cfg.name} embed={cfg.embedding_model} rerank={cfg.reranker or 'none'} prompt={cfg.prompt_template}")

    # Build embeddings (using the preset's choice).
    embeddings = make_embeddings(cfg.embedding_backend, model_name=cfg.embedding_model)
    settings = RagSettings()

    # Build a corpus (or empty list).
    if args.no_corpus:
        from langchain_core.documents import Document
        corpus = [Document(page_content="(no corpus)", metadata={"_placeholder": True})]
    else:
        corpus = build_corpus_from_library(args.library_dir, settings.rag_chunk_size, settings.rag_chunk_overlap)
        if not corpus:
            print(f"No documents found in {args.library_dir}. Use --no-corpus to test without context.", file=sys.stderr)
            return 1
        print(f"[run_rag] loaded {len(corpus)} chunks from {args.library_dir}")

    # Build the FAISS vectorstore.
    vectorstore = FAISS.from_documents(corpus, embeddings)

    # Build the LLM client and the pipeline.
    async with AsyncAnthropic(api_key=api_key) as client:
        pipeline = build_pipeline(cfg, vectorstore=vectorstore, llm_client=client)
        answer = await pipeline.run(args.question)

    print()
    print(f"Q: {args.question}")
    print(f"A: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))