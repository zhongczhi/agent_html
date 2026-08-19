"""v19f validation: re-score v19b-soft's predictions with the v19f normalizer.

The v19b-soft dump has `predicted_raw` (raw LLM output) and `predicted`
(post v19a normalization). Apply the v19f normalizer (C2 'Both' prefix
+ T1 'I can confirm' prefix) to the raw outputs and recompute contains_gold.

This avoids re-running the LLM calls. ~5 min, no API cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DUMP = REPO / "docs/eval-results/iter35-v19c-soft-r1.jsonl"
FIXTURE = REPO / "scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json"


def main():
    from backend.eval.normalizer import normalize_answer
    from backend.eval import metrics

    with FIXTURE.open(encoding="utf-8") as f:
        items = json.load(f)
    qid_type = {it["_id"]: it["type"] for it in items}

    with DUMP.open(encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]

    old_pass = 0  # v19b-soft (current SOTA)
    new_pass = 0  # v19b-soft + v19f normalizer (proposed)
    flips = []

    for r in recs:
        qid = r["qid"]
        qtype = qid_type.get(qid)
        gold = r["gold"]
        raw = r.get("predicted_raw") or r["predicted"]
        old_cg = r["contains_gold"] >= 1.0  # v19b-soft (v19a normalizer)
        new_pred = normalize_answer(raw, qtype=qtype)
        new_cg = metrics.answer_coverage_at_k([new_pred], gold) >= 1.0
        if old_cg:
            old_pass += 1
        if new_cg:
            new_pass += 1
        if old_cg != new_cg:
            flips.append({
                "qid": qid,
                "type": qtype,
                "gold": gold,
                "raw": raw[:80],
                "old_norm": r["predicted"][:80],
                "new_norm": new_pred[:80],
                "direction": "↑" if new_cg else "↓",
            })

    n = len(recs)
    print("=" * 70)
    print("V19F VALIDATION — re-score v19b-soft predictions with v19f normalizer")
    print("=" * 70)
    print()
    print(f"v19b-soft (current SOTA, v19a normalizer): {old_pass}/{n} = {old_pass/n:.3f}")
    print(f"v19b-soft + v19f normalizer               : {new_pass}/{n} = {new_pass/n:.3f}")
    print(f"Lift: +{new_pass - old_pass} = +{(new_pass - old_pass)/n*100:.1f}pp")
    print()

    # Per-type
    print("Per-type:")
    for t in ["comparison", "inference", "null", "temporal"]:
        old_t = sum(1 for r in recs if qid_type.get(r["qid"]) == t and r["contains_gold"] >= 1.0)
        new_t = sum(1 for r in recs
                    if qid_type.get(r["qid"]) == t
                    and metrics.answer_coverage_at_k(
                        [normalize_answer(r.get("predicted_raw") or r["predicted"], qtype=t)],
                        r["gold"]) >= 1.0)
        n_t = sum(1 for r in recs if qid_type.get(r["qid"]) == t)
        print(f"  {t:<13}: {old_t}/{n_t}  →  {new_t}/{n_t}  Δ={new_t-old_t:+d}")

    print()
    print(f"Flips ({len(flips)}):")
    for f in flips:
        print(f"  {f['direction']} {f['qid']} ({f['type']}) gold={f['gold']}")
        print(f"    raw    : {f['raw']}")
        print(f"    old    : {f['old_norm']}")
        print(f"    new    : {f['new_norm']}")


if __name__ == "__main__":
    main()