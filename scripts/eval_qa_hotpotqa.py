"""End-to-end QA accuracy eval for HotpotQA. CLI only.

Isolated from chat: imports nothing from backend.chat.*.
The RAG system prompt is duplicated in backend.eval.qa_judge to preserve
this isolation rule (FR-32, FR-44.3).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Bootstrap sys.path so `python scripts/eval_qa_hotpotqa.py` (any cwd) finds
# the `backend` package, exactly like the chat service does on startup.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from backend.eval import cache as ev_cache  # noqa: E402
from backend.eval import hotpotqa as hotpot  # noqa: E402
from backend.eval import metrics  # noqa: E402
from backend.eval.paraphrases import load_paraphrases  # noqa: E402
from backend.eval.qa_judge import ask_llm, build_qa_prompt  # noqa: E402
from backend.rag.config import RagSettings  # noqa: E402
from backend.rag.embeddings import make_embeddings  # noqa: E402

log = logging.getLogger("eval_qa_hotpotqa")

REPO_ROOT = _REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"

PACING_SECONDS = 1  # NFR-20: 1s between LLM calls.


async def _evaluate_one(
    client: AsyncAnthropic,
    model: str,
    item,
    retrieved_docs,
    question_text: str,
    variant_name: str,
    mode: str,
    prompt_template: str = "default",
    gold_in_top_k: bool | None = None,
    gold_titles: set[str] | None = None,
    retrieved_titles: list[str] | None = None,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
) -> dict:
    """One LLM call + scoring (FR-42).

    mode is "with_context" (uses retrieved_docs) or "without_context" (no context).
    Returns a dict with predicted answer, gold, and four scoring metrics:
      - answer_f1: HotpotQA-standard token F1 over the full prediction.
        Diluted by conversational wrappers; partial credit.
      - answer_em: token sets identical after normalization. Rare.
      - contains_gold: 1.0 if normalized gold appears as substring in
        normalized prediction. The most user-relevant metric — answers
        the question "did the user see the right answer?" rather than
        "did the model output exactly the gold string?"
      - contains_f1: token F1 weighted to favor coverage of gold tokens
        (precision * recall with recall dominating). Computed as
        2*P*R / (P+R) but with len(pred) clamped so long wrappers don't
        dilute the score. Implementation: token-level overlap divided by
        the max of (len(pred), len(gold)).

    gold_in_top_k: True iff at least one gold paragraph title appeared in
        the retrieved top-k. Used to localize the failure mode of failed
        questions: gold_in_top_k=1 + contains_gold=0 = extraction miss;
        gold_in_top_k=0 + contains_gold=0 = retrieval miss. None for
        without-context runs (retrieval doesn't apply).

    prompt_template: 'default' uses qa_judge.build_qa_prompt; 'extract_span'
    uses a custom builder that asks the LLM to extract verbatim spans.
    """
    await asyncio.sleep(PACING_SECONDS)
    # Build prompt according to the template.
    if prompt_template == "extract_span":
        from backend.rag.pipeline import ExtractSpanPromptBuilder
        builder = ExtractSpanPromptBuilder()
        prompt = builder.build(question_text, retrieved_docs if mode == "with_context" else None)
    elif prompt_template == "cot_extract":
        from backend.rag.pipeline import CoTExtractPromptBuilder
        builder = CoTExtractPromptBuilder()
        prompt = builder.build(question_text, retrieved_docs if mode == "with_context" else None)
    elif prompt_template == "cot_extract_v2":
        from backend.rag.pipeline import CoTExtractV2PromptBuilder
        builder = CoTExtractV2PromptBuilder()
        prompt = builder.build(question_text, retrieved_docs if mode == "with_context" else None)
    elif prompt_template == "cot_extract_no_titles":
        from backend.rag.pipeline import CoTExtractNoTitlesPromptBuilder
        builder = CoTExtractNoTitlesPromptBuilder()
        prompt = builder.build(question_text, retrieved_docs if mode == "with_context" else None)
    else:
        if mode == "with_context":
            prompt = build_qa_prompt(question_text, retrieved_docs)
        else:
            prompt = build_qa_prompt(question_text, None)
    # If the preset configures extended thinking, enable it on the LLM call.
    # max_tokens must be >= thinking_budget so the visible answer has room.
    if thinking_budget is not None and thinking_budget > 0:
        answer = await ask_llm(
            client,
            model,
            prompt,
            max_tokens=max_tokens or max(thinking_budget + 512, 2048),
            thinking_budget=thinking_budget,
        )
    else:
        answer = await ask_llm(client, model, prompt)
    f1 = metrics.answer_f1(answer, item.answer)
    em = metrics.exact_match(answer, item.answer)
    contains = metrics.answer_coverage_at_k([answer], item.answer)
    return {
        "qid": item.id,
        "question": question_text,
        "variant": variant_name,
        "mode": mode,
        "predicted": answer,
        "gold": item.answer,
        "answer_f1": f1,
        "answer_em": 1.0 if em else 0.0,
        "contains_gold": contains,
        "gold_in_top_k": gold_in_top_k,
        "gold_paragraph_titles": sorted(gold_titles) if gold_titles else [],
        "retrieved_titles": list(retrieved_titles) if retrieved_titles else [],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="HotpotQA end-to-end QA accuracy eval (with LLM judge)."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--subset",
        type=int,
        metavar="N",
        help="Stratified sample of N questions",
    )
    grp.add_argument("--full", action="store_true", help="Use all questions (default)")
    parser.add_argument("--k", type=int, default=4, help="Top-k to retrieve (default 4)")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force rebuild of every per-question FAISS index",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Read dataset from PATH (test hook)",
    )
    parser.add_argument(
        "--paraphrase-set",
        type=Path,
        default=None,
        help=(
            "Path to a paraphrase JSON file. If given, eval runs original + "
            "each available paraphrase style per question."
        ),
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help=(
            "Also run without-context (vanilla LLM, no retrieval) for each "
            "question. The output reports both modes and the retrieval lift delta."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("ANTHROPIC_MODEL", "minimax-3"),
        help="LLM model name (default: minimax-3)",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Pipeline preset name (see backend.rag.pipeline.PRESETS). "
            "If given, overrides --k and uses the preset's embedding model "
            "and prompt template. Run with no value to see available presets."
        ),
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="Print available pipeline presets and exit.",
    )
    parser.add_argument(
        "--dump-results",
        type=Path,
        default=None,
        help=(
            "Write per-question results as JSON Lines to this path. Each line "
            "is one result with qid, question, predicted, gold, contains_gold, "
            "answer_f1, answer_em, gold_in_top_k, gold_paragraph_titles, "
            "retrieved_titles, etc. Useful for failure-mode inspection."
        ),
    )
    args = parser.parse_args(argv)

    # Handle --list-pipelines early.
    if args.list_pipelines:
        from backend.rag.pipeline import list_presets
        print("Available pipeline presets:")
        for name in list_presets():
            from backend.rag.pipeline import PRESETS
            cfg = PRESETS[name]
            print(f"  {name:<24} embed={cfg.embedding_model:<24} rerank={cfg.reranker or 'none':<14} prompt={cfg.prompt_template}")
        return 0

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY env var is not set. Set it before running.",
            file=sys.stderr,
        )
        return 1

    dataset_path = args.fixture or DEFAULT_DATASET
    if not dataset_path.exists():
        print(
            f"Dataset not found at {dataset_path}.\n"
            "Run scripts/ingest_hotpotqa.py (downloads to scripts/.cache/), "
            "or pass --fixture PATH.",
            file=sys.stderr,
        )
        return 1
    try:
        items = hotpot.load(dataset_path)
    except json.JSONDecodeError as e:
        print(f"Dataset JSON is corrupt: {e}", file=sys.stderr)
        return 1
    if args.subset is not None:
        items = hotpot.sample(items, args.subset)

    d_sha = hotpot.dataset_sha(dataset_path)
    print(
        "Dataset: HotpotQA dev_distractor v1 "
        "(CC BY-SA 4.0 — https://hotpotqa.github.io/)"
    )

    paraphrases: dict[str, dict[str, str]] = {}
    if args.paraphrase_set:
        if not args.paraphrase_set.exists():
            print(f"Paraphrase set not found: {args.paraphrase_set}", file=sys.stderr)
            return 1
        try:
            paraphrases = load_paraphrases(args.paraphrase_set)
            log.info(
                "Loaded %d paraphrase entries from %s",
                len(paraphrases),
                args.paraphrase_set,
            )
        except json.JSONDecodeError as e:
            print(f"Paraphrase set is corrupt: {e}", file=sys.stderr)
            return 1

    # Resolve pipeline preset (if any) and use it to drive embedding + prompt.
    pipeline_cfg = None
    prompt_template = "default"
    if args.pipeline:
        from backend.rag.pipeline import PRESETS
        if args.pipeline not in PRESETS:
            print(
                f"Unknown pipeline: {args.pipeline!r}. "
                f"Available: {sorted(PRESETS.keys())}. "
                f"Use --list-pipelines to see details.",
                file=sys.stderr,
            )
            return 1
        pipeline_cfg = PRESETS[args.pipeline]
        prompt_template = pipeline_cfg.prompt_template
        log.info(
            "Using pipeline preset: %s (embed=%s, rerank=%s, prompt=%s, top_k=%d)",
            pipeline_cfg.name,
            pipeline_cfg.embedding_model,
            pipeline_cfg.reranker or "none",
            pipeline_cfg.prompt_template,
            pipeline_cfg.top_k,
        )
        # Override --k with the preset's top_k so all parts of the pipeline agree.
        args.k = pipeline_cfg.top_k

    settings = RagSettings()
    # When a pipeline preset is active, override the embedding model with the
    # preset's choice. Otherwise use the env-driven default.
    if pipeline_cfg is not None and pipeline_cfg.embedding_backend == "sentence-transformers":
        embeddings = make_embeddings(
            pipeline_cfg.embedding_backend,
            model_name=pipeline_cfg.embedding_model,
        )
    else:
        embeddings = make_embeddings(settings.rag_embedding_backend)

    per_q: list[dict] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()

    async def run() -> int:
        nonlocal cache_hits, cache_builds, errors
        async with AsyncAnthropic(api_key=api_key) as client:
            for item in items:
                try:
                    # Hybrid retrievers also need the raw corpus (to build
                    # the BM25 index). For dense-only, we skip it for speed.
                    needs_corpus = (
                        pipeline_cfg is not None
                        and pipeline_cfg.retriever == "hybrid"
                    )
                    if needs_corpus:
                        index, hit, corpus = ev_cache.load_or_build(
                            item, d_sha, embeddings,
                            no_cache=args.no_cache, with_corpus=True,
                        )
                    else:
                        index, hit = ev_cache.load_or_build(
                            item, d_sha, embeddings, no_cache=args.no_cache
                        )
                        corpus = None
                    if hit:
                        cache_hits += 1
                    else:
                        cache_builds += 1

                    # Build the (question_text, variant_name) list.
                    variants: list[tuple[str, str]] = [(item.question, "original")]
                    para_entry = paraphrases.get(item.id)
                    if para_entry:
                        for style in ("lexical", "structural", "casual"):
                            if style in para_entry.get("paraphrases", {}):
                                variants.append(
                                    (para_entry["paraphrases"][style], style)
                                )

                    # If the pipeline is hybrid, build the hybrid retriever
                    # once per question (it holds the BM25 index in memory).
                    hybrid_retriever = None
                    if pipeline_cfg is not None and pipeline_cfg.retriever == "hybrid":
                        from backend.rag.pipeline import (
                            BM25Retriever, DenseRetriever, HybridRetriever,
                        )
                        hybrid_retriever = HybridRetriever(
                            dense_retriever=DenseRetriever(index),
                            bm25_retriever=BM25Retriever(corpus),
                        )

                    for q_text, vname in variants:
                        # When the pipeline preset configures a reranker,
                        # retrieve more candidates and rerank to top_k.
                        # Otherwise a plain top_k retrieval is enough.
                        # Hybrid path uses the prebuilt HybridRetriever.
                        if hybrid_retriever is not None:
                            retrieved_docs = hybrid_retriever.retrieve(q_text, k=pipeline_cfg.top_k)
                        elif pipeline_cfg is not None and pipeline_cfg.reranker is not None:
                            from backend.rag.pipeline import build_reranker
                            reranker = build_reranker(pipeline_cfg)
                            candidates = index.similarity_search(
                                q_text, k=pipeline_cfg.rerank_top_k,
                            )
                            retrieved_docs = reranker.rerank(
                                q_text, candidates, top_k=pipeline_cfg.top_k,
                            )
                        else:
                            retrieved_docs = index.similarity_search(q_text, k=args.k)
                        # Localize failure modes: was the gold paragraph in the
                        # retrieved set, even if the LLM still missed the answer?
                        gold_titles = hotpot.gold_paragraph_titles(item)
                        retrieved_titles = [d.metadata.get("title", "") for d in retrieved_docs]
                        gold_hit = metrics.gold_paragraph_in_top_k(retrieved_titles, gold_titles)
                        # with-context mode
                        per_q.append(await _evaluate_one(
                            client, args.llm_model, item, retrieved_docs,
                            q_text, vname, "with_context",
                            prompt_template=prompt_template,
                            gold_in_top_k=gold_hit,
                            gold_titles=gold_titles,
                            retrieved_titles=retrieved_titles,
                            thinking_budget=pipeline_cfg.thinking_budget if pipeline_cfg else None,
                            max_tokens=(pipeline_cfg.thinking_budget + 512) if pipeline_cfg and pipeline_cfg.thinking_budget else None,
                        ))
                        if args.compare_baseline:
                            # without-context baseline (retrieval doesn't apply)
                            per_q.append(await _evaluate_one(
                                client, args.llm_model, item, None,
                                q_text, vname, "without_context",
                                prompt_template=prompt_template,
                                gold_in_top_k=None,
                                gold_titles=gold_titles,
                                retrieved_titles=retrieved_titles,
                                thinking_budget=None,  # baseline never uses thinking
                            ))
                except Exception as e:
                    log.warning("qid=%s error: %s", item.id, e)
                    errors += 1

        elapsed = time.monotonic() - t0

        def avg(predicate, key):
            relevant = [r[key] for r in per_q if predicate(r)]
            return (sum(relevant) / len(relevant)) if relevant else 0.0

        def fmt(x: float) -> str:
            return f"{x:.3f}"

        label = "full" if args.subset is None else str(args.subset)
        print(
            f"\nHotpotQA End-to-End QA Eval — subset={label}, k={args.k}, "
            f"dataset_sha={d_sha}"
        )

        with_n = sum(1 for r in per_q if r["mode"] == "with_context")
        print("  with_context:")
        print(
            f"    contains_gold: {fmt(avg(lambda r: r['mode'] == 'with_context', 'contains_gold'))}  (n={with_n})"
        )
        print(
            f"    answer_f1   : {fmt(avg(lambda r: r['mode'] == 'with_context', 'answer_f1'))}  (n={with_n})"
        )
        print(
            f"    answer_em   : {fmt(avg(lambda r: r['mode'] == 'with_context', 'answer_em'))}  (n={with_n})"
        )

        # Failure-mode breakdown: distinguishes retrieval misses from
        # extraction misses for failed questions (contains_gold=0).
        # See FR for explanation of the lever this informs.
        def _count(predicate) -> int:
            return sum(1 for r in per_q if predicate(r))

        succ = _count(lambda r: r["mode"] == "with_context" and r["contains_gold"] >= 1.0)
        ext_miss = _count(
            lambda r: r["mode"] == "with_context"
            and r["contains_gold"] < 1.0
            and r["gold_in_top_k"] is True
        )
        ret_miss = _count(
            lambda r: r["mode"] == "with_context"
            and r["contains_gold"] < 1.0
            and r["gold_in_top_k"] is False
        )
        unk = _count(
            lambda r: r["mode"] == "with_context"
            and r["contains_gold"] < 1.0
            and r["gold_in_top_k"] is None
        )
        if with_n > 0:
            print("  failure-mode breakdown (with_context):")
            print(
                f"    success         : {succ:>4}  ({fmt(succ / with_n)})"
            )
            print(
                f"    extraction miss : {ext_miss:>4}  "
                f"({fmt(ext_miss / with_n)}) — gold in top-k, LLM missed"
            )
            print(
                f"    retrieval miss  : {ret_miss:>4}  "
                f"({fmt(ret_miss / with_n)}) — gold NOT in top-k"
            )
            if unk:
                print(
                    f"    unknown         : {unk:>4}  "
                    f"({fmt(unk / with_n)}) — gold_in_top_k not recorded"
                )

        if args.compare_baseline:
            without_n = sum(1 for r in per_q if r["mode"] == "without_context")
            print("  without_context (baseline):")
            print(
                f"    contains_gold: {fmt(avg(lambda r: r['mode'] == 'without_context', 'contains_gold'))}  (n={without_n})"
            )
            print(
                f"    answer_f1   : {fmt(avg(lambda r: r['mode'] == 'without_context', 'answer_f1'))}  (n={without_n})"
            )
            print(
                f"    answer_em   : {fmt(avg(lambda r: r['mode'] == 'without_context', 'answer_em'))}  (n={without_n})"
            )
            with_f1 = avg(lambda r: r["mode"] == "with_context", "answer_f1")
            without_f1 = avg(lambda r: r["mode"] == "without_context", "answer_f1")
            with_em = avg(lambda r: r["mode"] == "with_context", "answer_em")
            without_em = avg(lambda r: r["mode"] == "without_context", "answer_em")
            with_cg = avg(lambda r: r["mode"] == "with_context", "contains_gold")
            without_cg = avg(lambda r: r["mode"] == "without_context", "contains_gold")
            print("  delta (retrieval helps):")
            print(
                f"    contains_gold: {with_cg - without_cg:+.3f}  "
                f"({fmt(with_cg)} - {fmt(without_cg)})"
            )
            print(
                f"    answer_f1   : {with_f1 - without_f1:+.3f}  "
                f"({fmt(with_f1)} - {fmt(without_f1)})"
            )
            print(
                f"    answer_em   : {with_em - without_em:+.3f}  "
                f"({fmt(with_em)} - {fmt(without_em)})"
            )

        if paraphrases:
            print("  -- by variant -- (with_context)")
            for variant in ("original", "lexical", "structural", "casual"):
                n = sum(
                    1
                    for r in per_q
                    if r["mode"] == "with_context" and r["variant"] == variant
                )
                if n == 0:
                    print(f"  {variant:<12} : (no data)")
                    continue
                cg = avg(
                    lambda r: r["mode"] == "with_context" and r["variant"] == variant,
                    "contains_gold",
                )
                f1 = avg(
                    lambda r: r["mode"] == "with_context" and r["variant"] == variant,
                    "answer_f1",
                )
                em = avg(
                    lambda r: r["mode"] == "with_context" and r["variant"] == variant,
                    "answer_em",
                )
                print(
                    f"  {variant:<12} : n={n:<4}  "
                    f"cg={fmt(cg)}  f1={fmt(f1)}  em={fmt(em)}"
                )

        # Footer.
        llm_calls = len(per_q)
        print(f"  LLM calls             : {llm_calls}")
        print(f"  cache hits / builds   : {cache_hits} / {cache_builds}")
        print(f"  errors                : {errors}")
        print(f"  elapsed               : {elapsed:.1f}s")

        # Optional dump of every per-question result to JSON Lines.
        # Used for failure-mode inspection: failures can be grepped out and
        # inspected offline. Quietly skip if the path isn't writable.
        if args.dump_results is not None:
            try:
                args.dump_results.parent.mkdir(parents=True, exist_ok=True)
                with args.dump_results.open("w", encoding="utf-8") as f:
                    for r in per_q:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  dumped per-q results  : {args.dump_results}  (n={len(per_q)})")
            except OSError as e:
                print(f"  WARNING: failed to dump results to {args.dump_results}: {e}", file=sys.stderr)

        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())