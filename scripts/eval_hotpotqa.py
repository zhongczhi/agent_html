"""Run the HotpotQA retrieval evaluation pipeline. CLI only.

Strictly isolated from chat: this script and backend/eval/* import nothing
from backend/chat/. The isolation is verified by a grep guard in task 7 of
the implementation plan.
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

    per_q: list[tuple[float, float, float, float, float]] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()
    for item in items:
        try:
            index, hit = ev_cache.load_or_build(
                item, d_sha, embeddings, no_cache=args.no_cache
            )
            docs = index.similarity_search(item.question, k=args.k)
            retrieved_titles = [d.metadata.get("title", "") for d in docs]
            gold_titles = hotpot.gold_paragraph_titles(item)
            pr = metrics.paragraph_recall_at_k(retrieved_titles, gold_titles)
            sp, sr, sf_f1, em = metrics.supporting_fact_metrics(
                retrieved_titles, gold_titles
            )
            per_q.append((pr, sp, sr, sf_f1, em))
            if hit:
                cache_hits += 1
            else:
                cache_builds += 1
        except Exception as e:
            log.warning("qid=%s error: %s", item.id, e)
            errors += 1

    elapsed = time.monotonic() - t0

    def avg(i: int) -> float:
        return (sum(q[i] for q in per_q) / len(per_q)) if per_q else 0.0

    label = "full" if args.subset is None else str(args.subset)
    print(f"\nHotpotQA Eval — subset={label}, k={args.k}, dataset_sha={d_sha}")
    print(f"  paragraph_recall@{args.k}   : {avg(0):.3f}")
    print(f"  sf_precision         : {avg(1):.3f}")
    print(f"  sf_recall            : {avg(2):.3f}")
    print(f"  sf_f1                : {avg(3):.3f}")
    print(f"  sf_em                : {avg(4):.3f}")
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
