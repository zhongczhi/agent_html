"""Build the iter-29 smoke-test subset: 100 SOTA-failure cases + 100 random cases.

Purpose: the iter-22 SOTA (cot_extract_notitles_thinking_k10) on MultiHop-RAG
n=2556 produced 538 dump fails. 301 of those are `null` questions where the
gold requires the model to refuse and the SOTA has no refusal path — those
are guaranteed 0% regardless of preset and would dilute the signal. The
remaining 237 are real SOTA extraction/reasoning failures:

  comparison: 113 (18.7% fail rate on 604 completed)
  inference:    7 (0.9% fail rate on 815 completed)
  temporal:   117 (20.1% fail rate on 582 completed)

We sample 100 of those 237 proportionally by type, then 100 random from
the full 2302 dump records (excluding the failure set). The combined 200
is written as a new fixture file with the same shape as the n=2556
fixture so `scripts/eval_qa_hotpotqa.py` can run on it directly.

Output: scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json

Reproducibility: random.Random(42) for the random set; the failure set
is sampled deterministically by sorting qids within each type bucket
(by hash) and taking the first N — no randomness.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("build_iter29_smoke_subset")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "scripts" / ".cache"
SOURCE_FIXTURE = FIXTURE_DIR / "multihop_rag_fixture_2556.json"
SOURCE_DUMP = REPO_ROOT / "docs" / "eval-results" / "iter26-multihop-rag-sota-k10-full-dump.jsonl"
DEFAULT_OUT = FIXTURE_DIR / "multihop_rag_fixture_iter29_smoke_200.json"

# Per-type sample sizes for the 100-qid failure set. Proportional to the
# 237 real SOTA failures (113 comparison + 7 inference + 117 temporal).
# Null (301 fails) is excluded by design.
FAILURE_TARGETS = {
    "comparison": 48,
    "temporal": 49,
    "inference": 3,
}
assert sum(FAILURE_TARGETS.values()) == 100

RANDOM_SEED = 42
RANDOM_N = 100


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-fixture", type=Path, default=SOURCE_FIXTURE)
    parser.add_argument("--source-dump", type=Path, default=SOURCE_DUMP)
    args = parser.parse_args(argv)

    log.info("Loading source fixture: %s", args.source_fixture)
    fixture = json.load(args.source_fixture.open(encoding="utf-8"))
    qid_to_record = {item["_id"]: item for item in fixture}
    log.info("  %d fixture records", len(qid_to_record))

    log.info("Loading iter-22 SOTA dump: %s", args.source_dump)
    dump = [json.loads(l) for l in args.source_dump.open(encoding="utf-8")]
    log.info("  %d dump records", len(dump))

    # Group dump records by qid -> (type, contains_gold)
    dump_by_qid = {r["qid"]: r for r in dump}

    # Group qids by (type, pass/fail)
    by_type_fail: dict[str, list[str]] = defaultdict(list)
    by_type_pass: dict[str, list[str]] = defaultdict(list)
    for qid, r in dump_by_qid.items():
        t = qid_to_record.get(qid, {}).get("type")
        if t is None:
            continue
        if r["contains_gold"] == 0:
            by_type_fail[t].append(qid)
        else:
            by_type_pass[t].append(qid)

    # 1. Build the 100-qid failure set (real SOTA failures, no null).
    failure_set: list[str] = []
    for t, n in FAILURE_TARGETS.items():
        bucket = by_type_fail[t]
        # Sort deterministically by qid so re-runs pick the same qids.
        bucket_sorted = sorted(bucket)
        picked = bucket_sorted[:n]
        if len(picked) < n:
            log.warning("Only %d %s fails available, requested %d", len(bucket), t, n)
        failure_set.extend(picked)
        log.info("  failure[%s]: %d requested, %d picked", t, n, len(picked))

    failure_set_set = set(failure_set)
    assert len(failure_set_set) == 100, f"failure set has duplicates: {len(failure_set)} -> {len(failure_set_set)}"

    # 2. Build the 100-qid random set from all 2302 completed records,
    #    excluding the failure set. Sample proportionally to natural type
    #    distribution (no stratification — random.Random(42) gives a
    #    natural mix, which is what we want for a smoke test).
    pool = [qid for qid in dump_by_qid if qid not in failure_set_set]
    rng = random.Random(RANDOM_SEED)
    random_set = rng.sample(pool, RANDOM_N)
    log.info("  random: %d picked from pool of %d", len(random_set), len(pool))

    selected_qids = failure_set + random_set
    assert len(set(selected_qids)) == 200, "qid collision between failure and random sets"

    # 3. Write the new fixture: keep the full per-question record (so
    #    eval_qa_hotpotqa.py sees context + supporting_facts and behaves
    #    exactly as it did on the n=2556 fixture).
    out_records = [qid_to_record[qid] for qid in selected_qids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, args.out.open("w", encoding="utf-8"), ensure_ascii=False)
    log.info("Wrote %d records to %s", len(out_records), args.out)

    # Per-type summary for verification.
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"fail_set": 0, "rand_set": 0})
    for qid in failure_set:
        t = qid_to_record[qid]["type"]
        type_counts[t]["fail_set"] += 1
    for qid in random_set:
        t = qid_to_record[qid]["type"]
        type_counts[t]["rand_set"] += 1

    print()
    print(f"{'type':<12} {'fail_set':>10} {'rand_set':>10} {'total':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for t in sorted(type_counts):
        c = type_counts[t]
        print(f"{t:<12} {c['fail_set']:>10} {c['rand_set']:>10} {c['fail_set']+c['rand_set']:>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    total_fail = sum(c["fail_set"] for c in type_counts.values())
    total_rand = sum(c["rand_set"] for c in type_counts.values())
    print(f"{'TOTAL':<12} {total_fail:>10} {total_rand:>10} {total_fail+total_rand:>10}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())