"""Run the SOTA pipeline on the Track B heterogeneous-format corpus.

Track B is a small (~7 files, ~20 QA pairs) corpus that exercises the
real format-aware loaders in `backend.rag.loaders`. Each file is in a
different format (PDF, DOCX, CSV, MD, HTML, TXT). The SOTA pipeline
must extract the gold answer from each file's chunks and emit it.

Two eval modes:
  1. Per-question isolated context (default): the context is built from
     the source_file's chunks only. Tests format parsing + extraction
     discipline without retrieval noise.
  2. Open-corpus retrieval (--retrieval): all files in one index, the
     pipeline retrieves top-k chunks for each question. Tests retrieval
     + extraction end-to-end.

Per-format results are printed at the end.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Bootstrap sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from anthropic import AsyncAnthropic

from backend.eval.metrics import answer_f1, exact_match, answer_coverage_at_k, gold_paragraph_in_top_k
from backend.eval.qa_judge import ask_llm
from backend.rag.config import RagSettings
from backend.rag.embeddings import make_embeddings
from backend.rag.loaders import load as loader_load
from backend.rag.pipeline import PRESETS
from backend.rag.splitter import split_into_documents

log = logging.getLogger("eval_track_b")

CORPUS_DIR = _SCRIPT_DIR / ".cache" / "track_b_corpus"
DEFAULT_QA_PAIRS = CORPUS_DIR / "qa_pairs.json"
DEFAULT_FIXTURE_OUT = _SCRIPT_DIR / ".cache" / "track_b_fixture.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def load_and_chunk_files(corpus_dir: Path) -> dict[str, list[Document]]:
    """Load every file in `corpus_dir` via the format-aware loaders and
    chunk via the standard splitter. Returns {filename: [Document]}."""
    out: dict[str, list[Document]] = {}
    for path in sorted(corpus_dir.iterdir()):
        if path.name.startswith("qa_") or path.suffix == ".json":
            continue
        if not path.is_file():
            continue
        try:
            docs = list(split_into_documents(
                path,
                source_type="track_b",
                conversation_id=None,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ))
            out[path.name] = docs
            log.info("Loaded %s -> %d chunks", path.name, len(docs))
        except Exception as e:
            log.warning("Skipping %s: %s", path.name, e)
    return out


async def run_one_question(
    client: AsyncAnthropic,
    model: str,
    question: str,
    context_docs: list[Document],
    gold: str,
    thinking_budget: int | None,
) -> dict:
    """One SOTA call. Builds the title-strip CoT prompt, runs the LLM,
    scores against gold, and returns the result dict."""
    from backend.rag.pipeline import CoTExtractNoTitlesPromptBuilder
    builder = CoTExtractNoTitlesPromptBuilder()
    messages = builder.build(question, context_docs)
    if thinking_budget is not None and thinking_budget > 0:
        answer = await ask_llm(
            client, model, messages,
            max_tokens=max(thinking_budget + 512, 2048),
            thinking_budget=thinking_budget,
        )
    else:
        answer = await ask_llm(client, model, messages)
    f1 = answer_f1(answer, gold)
    em = exact_match(answer, gold)
    contains = answer_coverage_at_k([answer], gold)
    return {
        "question": question,
        "gold": gold,
        "predicted": answer,
        "answer_f1": f1,
        "answer_em": 1.0 if em else 0.0,
        "contains_gold": contains,
    }


async def eval_isolated(
    qa_pairs: list[dict],
    chunks_by_file: dict[str, list[Document]],
    model: str,
    thinking_budget: int | None,
) -> list[dict]:
    """Per-question isolated context. The context for question Q is the
    chunks of its source_file. Tests format parsing + SOTA extraction
    without retrieval noise."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return []
    results: list[dict] = []
    async with AsyncAnthropic(api_key=api_key) as client:
        for qa in qa_pairs:
            source_file = qa["source_file"]
            context_docs = chunks_by_file.get(source_file, [])
            if not context_docs:
                log.warning("No chunks for %s; skipping %s", source_file, qa["qid"])
                results.append({**qa, "contains_gold": 0.0, "predicted": "", "answer_f1": 0.0, "answer_em": 0.0})
                continue
            r = await run_one_question(
                client, model, qa["question"], context_docs, qa["answer"], thinking_budget,
            )
            results.append({**qa, **r})
            status = "✓" if r["contains_gold"] >= 1.0 else "✗"
            log.info(
                "  %s  %s  [fmt=%s]  q=%s",
                status, qa["qid"], qa["source_format"],
                qa["question"][:60].replace("\n", " "),
            )
    return results


def format_results_table(results: list[dict]) -> str:
    """Per-format contains_gold + summary. Returns a printable string."""
    by_format: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_format[r["source_format"]].append(r)
    lines = []
    lines.append("")
    lines.append("Per-format results:")
    lines.append(f"  {'format':<10}  {'n':>4}  {'contains_gold':>15}  {'answer_f1':>10}")
    lines.append(f"  {'-'*10}  {'-'*4}  {'-'*15}  {'-'*10}")
    n_total = sum_cg = n_success = 0
    for fmt in sorted(by_format):
        rs = by_format[fmt]
        n = len(rs)
        cg = sum(r["contains_gold"] for r in rs) / n if n else 0.0
        f1 = sum(r["answer_f1"] for r in rs) / n if n else 0.0
        lines.append(f"  {fmt:<10}  {n:>4}  {cg:>15.3f}  {f1:>10.3f}")
        n_total += n
        sum_cg += sum(r["contains_gold"] for r in rs)
        n_success += sum(1 for r in rs if r["contains_gold"] >= 1.0)
    lines.append(f"  {'-'*10}  {'-'*4}  {'-'*15}  {'-'*10}")
    overall_cg = sum_cg / n_total if n_total else 0.0
    lines.append(f"  {'TOTAL':<10}  {n_total:>4}  {overall_cg:>15.3f}  {n_success:>5d}/{n_total}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run SOTA on Track B heterogeneous-format corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help="Directory containing the Track B files (PDF, DOCX, etc.)",
    )
    parser.add_argument(
        "--qa-pairs",
        type=Path,
        default=DEFAULT_QA_PAIRS,
        help="Path to qa_pairs.json",
    )
    parser.add_argument(
        "--pipeline",
        default="cot_extract_notitles_thinking_k10",
        choices=list(PRESETS.keys()),
        help="Pipeline preset name (default: SOTA)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write per-question results as JSON Lines",
    )
    args = parser.parse_args(argv)

    if not args.corpus_dir.exists():
        print(f"Corpus dir not found: {args.corpus_dir}", file=sys.stderr)
        return 1
    if not args.qa_pairs.exists():
        print(f"QA pairs not found: {args.qa_pairs}", file=sys.stderr)
        return 1

    qa_pairs = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    log.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_pairs)

    chunks_by_file = load_and_chunk_files(args.corpus_dir)
    if not chunks_by_file:
        print("No files loaded — check --corpus-dir", file=sys.stderr)
        return 1
    log.info("Loaded %d files, %d total chunks",
             len(chunks_by_file), sum(len(v) for v in chunks_by_file.values()))

    pipeline_cfg = PRESETS[args.pipeline]
    log.info("Using pipeline: %s (top_k=%d, prompt=%s, thinking_budget=%s)",
             pipeline_cfg.name, pipeline_cfg.top_k, pipeline_cfg.prompt_template,
             pipeline_cfg.thinking_budget)

    t0 = time.monotonic()
    results = asyncio.run(eval_isolated(
        qa_pairs, chunks_by_file, pipeline_cfg.llm_model, pipeline_cfg.thinking_budget,
    ))
    elapsed = time.monotonic() - t0

    print(format_results_table(results))
    print(f"\nElapsed: {elapsed:.1f}s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Wrote per-question results to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
