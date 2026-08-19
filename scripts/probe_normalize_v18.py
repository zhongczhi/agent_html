"""Probe: apply normalizer to v18 dump predictions, recompute scores.

Pure post-processing — no LLM calls. Validates whether the
normalization layer would lift v18's pass rate.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.eval.normalizer import normalize_answer

V18_DUMP = Path("docs/eval-results/iter35-smoke-v18-dump.jsonl")
V17_DUMP = Path("docs/eval-results/iter35-smoke-v17-k5-dump.jsonl")
FIXTURE = Path("scripts/.cache/multihop_rag_fixture_iter29_smoke_200.json")
OUT_PATH = Path("docs/eval-results/v18-normalized.jsonl")


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_gold(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False
    return normalize(gold) in normalize(predicted)


def main():
    fixture = {q["_id"]: q for q in json.loads(FIXTURE.read_text(encoding="utf-8"))}
    dump = [json.loads(l) for l in V18_DUMP.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(dump)} v18 results")

    out_rows = []
    raw_pass = 0
    norm_pass = 0
    flips = []
    by_type = defaultdict(lambda: [0, 0, 0, 0])  # raw_pass, norm_pass, raw_total, norm_total

    for r in dump:
        qid = r["qid"]
        predicted_raw = r.get("predicted") or ""
        gold = fixture[qid]["answer"]
        qtype = fixture[qid]["type"]

        # Raw pass (v18 baseline)
        raw_ok = contains_gold(predicted_raw, gold)
        # Normalized pass
        predicted_norm = normalize_answer(predicted_raw, qtype=qtype)
        norm_ok = contains_gold(predicted_norm, gold)

        if raw_ok:
            raw_pass += 1
        if norm_ok:
            norm_pass += 1
        if norm_ok and not raw_ok:
            flips.append({"qid": qid, "type": qtype, "gold": gold, "raw": predicted_raw[:120], "norm": predicted_norm[:120]})
        if not norm_ok and raw_ok:
            flips.append({"qid": qid, "type": qtype, "direction": "regression", "gold": gold, "raw": predicted_raw[:120], "norm": predicted_norm[:120]})

        by_type[qtype][0] += int(raw_ok)
        by_type[qtype][1] += int(norm_ok)
        by_type[qtype][2] += 1
        by_type[qtype][3] += 1

        out_rows.append({
            "qid": qid,
            "type": qtype,
            "gold": gold,
            "raw_predicted": predicted_raw,
            "norm_predicted": predicted_norm,
            "raw_pass": raw_ok,
            "norm_pass": norm_ok,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} rows to {OUT_PATH}")

    n = len(dump)
    print(f"\n=== v18 baseline (no normalization): {raw_pass}/{n} = {raw_pass/n:.3f}")
    print(f"=== v18 + normalization:             {norm_pass}/{n} = {norm_pass/n:.3f}")
    print(f"=== Lift:                             +{norm_pass-raw_pass} questions ({(norm_pass-raw_pass)/n*100:.1f}pp)")
    print()
    print("Per-type (raw / norm / n):")
    for t in sorted(by_type):
        rp, np_, rt, nt = by_type[t]
        print(f"  {t:>12}: raw={rp}/{nt}  norm={np_}/{nt}  delta={np_-rp:+d}")

    if flips:
        print(f"\n=== Flips ({len(flips)}) ===")
        for f in flips:
            dir = "↑" if f.get("direction") != "regression" else "↓"
            print(f"  {dir} {f['qid']} type={f['type']} gold={f['gold']!r}")
            print(f"    raw:  {f['raw']!r}")
            print(f"    norm: {f['norm']!r}")


if __name__ == "__main__":
    main()