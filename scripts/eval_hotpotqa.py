"""Run the HotpotQA retrieval evaluation pipeline. CLI only.

Isolated from chat: this script and the backend/eval package import
nothing from the chat domain. The isolation is verified by a grep guard
(task 7 of the implementation plan).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Bootstrap sys.path so `python scripts/eval_hotpotqa.py` (any cwd) can find
# the `backend` package, exactly like the chat service does on startup.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.eval import cache as ev_cache  # noqa: E402  (import after path setup)
from backend.eval import hotpotqa as hotpot  # noqa: E402
from backend.eval import metrics  # noqa: E402
from backend.rag.config import RagSettings  # noqa: E402
from backend.rag.embeddings import make_embeddings  # noqa: E402

log = logging.getLogger("eval_hotpotqa")

REPO_ROOT = _REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="HotpotQA retrieval eval (paragraph-level)."
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
        help="Force rebuild of every per-question index",
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
            "Path to a paraphrase JSON file. Default: "
            "backend/storage/eval/hotpotqa/paraphrases/{dataset_sha}.json. "
            "If absent, eval runs in original-only mode with a WARNING."
        ),
    )
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

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
    settings = RagSettings()
    embeddings = make_embeddings(settings.rag_embedding_backend)

    from backend.eval.paraphrases import load_paraphrases

    paraphrase_path = args.paraphrase_set or (
        REPO_ROOT
        / "backend" / "storage" / "eval" / "hotpotqa" / "paraphrases"
        / f"{d_sha}.json"
    )
    paraphrases: dict[str, dict[str, str]] = {}
    if paraphrase_path.exists():
        try:
            paraphrases = load_paraphrases(paraphrase_path)
            log.info(
                "Loaded %d paraphrase entries from %s",
                len(paraphrases),
                paraphrase_path,
            )
        except json.JSONDecodeError as e:
            log.warning(
                "Paraphrase set at %s is corrupt (%s); running original-only",
                paraphrase_path,
                e,
            )
    else:
        log.warning(
            "Paraphrase set not found at %s; running original-only mode",
            paraphrase_path,
        )

    per_q: list[dict] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()
    for item in items:
        try:
            index, hit = ev_cache.load_or_build(
                item, d_sha, embeddings, no_cache=args.no_cache
            )
            if hit:
                cache_hits += 1
            else:
                cache_builds += 1

            # Build the list of (question_text, variant_name) pairs.
            variants: list[tuple[str, str]] = [(item.question, "original")]
            para_entry = paraphrases.get(item.id)
            if para_entry:
                para_styles = para_entry.get("paraphrases", {})
                for style in ("lexical", "structural", "casual"):
                    if style in para_styles:
                        variants.append((para_styles[style], style))

            for question_text, variant_name in variants:
                docs = index.similarity_search(question_text, k=args.k)
                retrieved_titles = [d.metadata.get("title", "") for d in docs]
                retrieved_texts = [d.page_content for d in docs]
                gold_titles = hotpot.gold_paragraph_titles(item)
                pr = metrics.paragraph_recall_at_k(retrieved_titles, gold_titles)
                sp, sr, sf_f1, em = metrics.supporting_fact_metrics(
                    retrieved_titles, gold_titles
                )
                ac = metrics.answer_coverage_at_k(retrieved_texts, item.answer)
                per_q.append({
                    "qid": item.id,
                    "variant": variant_name,
                    "type": item.type,
                    "level": item.level,
                    "paragraph_recall": pr,
                    "sf_precision": sp,
                    "sf_recall": sr,
                    "sf_f1": sf_f1,
                    "sf_em": em,
                    "answer_coverage": ac,
                })
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
    print(f"\nHotpotQA Eval — subset={label}, k={args.k}, dataset_sha={d_sha}")
    if paraphrases:
        n_paraphrase_evals = sum(
            1 for r in per_q if r["variant"] != "original"
        )
        print(
            f"  paraphrase_set      : {paraphrase_path} "
            f"({len(paraphrases)} entries; {n_paraphrase_evals} paraphrase evaluations)"
        )
    else:
        print(f"  paraphrase_set      : (none — original-only mode)")

    # Headline (preserved labels for the existing integration test).
    print(f"  paragraph_recall@{args.k}  : {fmt(avg(lambda _: True, 'paragraph_recall'))}")
    print(f"  sf_precision        : {fmt(avg(lambda _: True, 'sf_precision'))}")
    print(f"  sf_recall           : {fmt(avg(lambda _: True, 'sf_recall'))}")
    print(f"  sf_f1               : {fmt(avg(lambda _: True, 'sf_f1'))}")
    print(f"  sf_em               : {fmt(avg(lambda _: True, 'sf_em'))}")

    # By variant (new).
    print("  -- by variant --")
    for variant in ("original", "lexical", "structural", "casual"):
        n = sum(1 for r in per_q if r["variant"] == variant)
        if n == 0:
            print(f"  {variant:<12} : (no data)")
            continue
        print(
            f"  {variant:<12} : "
            f"n={n:<4}  "
            f"ans_cov={fmt(avg(lambda r: r['variant'] == variant, 'answer_coverage'))}  "
            f"sf_recall={fmt(avg(lambda r: r['variant'] == variant, 'sf_recall'))}  "
            f"para_recall={fmt(avg(lambda r: r['variant'] == variant, 'paragraph_recall'))}"
        )

    # Aggregate (new).
    if per_q:
        print("  -- aggregate --")
        print(f"  mean_ans_cov@k    : {fmt(avg(lambda _: True, 'answer_coverage'))}")

        # Robustness@4: fraction of qids where all 4 variants had ans_cov=1.
        from collections import defaultdict
        by_qid: dict[str, dict[str, float]] = defaultdict(dict)
        for r in per_q:
            by_qid[r["qid"]][r["variant"]] = r["answer_coverage"]
        robust_count = sum(
            1
            for qid, vs in by_qid.items()
            if vs.get("original") == 1.0
            and vs.get("lexical") == 1.0
            and vs.get("structural") == 1.0
            and vs.get("casual") == 1.0
        )
        robust_total = sum(
            1
            for qid, vs in by_qid.items()
            if all(s in vs for s in ("original", "lexical", "structural", "casual"))
        )
        if robust_total:
            print(
                f"  robustness@4      : {robust_count / robust_total:.3f} "
                f"({robust_count}/{robust_total} qids with all 4 variants ans_cov=1)"
            )
        else:
            print("  robustness@4      : (no qid had all 4 variants)")

        # Per (type, level)
        print("  -- by (type, level) --")
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in per_q:
            buckets[(r["type"], r["level"])].append(r)
        for key in sorted(buckets):
            rows = buckets[key]
            n = len(rows)
            ac = sum(r["answer_coverage"] for r in rows) / n
            print(f"  {key[0]}/{key[1]:<8} : ans_cov={fmt(ac)}  (n={n})")

    # Footer (preserved labels for the existing integration test).
    print(
        f"  questions successfully evaluated : {len(per_q)} "
        f"(out of {len(items)} attempted)"
    )
    print(f"  cache hits / builds  : {cache_hits} / {cache_builds}")
    print(f"  errors               : {errors}")
    print(f"  elapsed              : {elapsed:.1f}s")

    if errors:
        log.warning(
            "%d questions errored (skipped, not counted in metrics above)", errors
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
