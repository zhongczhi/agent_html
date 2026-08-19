"""Per-category stability analysis for the 25 iter-35 v18 comparison failures.

For each of the 25 qids that failed in v18 (temp=0, post-normalization),
load the 3 temperature=0.3 runs and report:
  - pass/fail for each of the 3 runs
  - how many runs (0/1/2/3) recovered
  - per-category aggregates (A=premise, B=verdict-buried, C=vocab)

Also reports v18 baseline (temp=0) for comparison.

Usage:
  python scripts/analyze_comparison_stability.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "docs" / "eval-results"
FIXTURE = REPO / "scripts" / ".cache" / "multihop_rag_fixture_iter29_smoke_200.json"

# v18 baseline (temp=0, post-normalization)
V18_DUMP = EVAL_DIR / "iter35-smoke-v18-dump.jsonl"

# 3 temperature=0.3 runs (capture-thinking enabled)
T03_RUNS = [
    EVAL_DIR / "iter35-t03-r1.jsonl",
    EVAL_DIR / "iter35-t03-r2.jsonl",
    EVAL_DIR / "iter35-t03-r3.jsonl",
]


# Failure category assignment per qid, derived from
# docs/eval-results/2026-08-01-rag-iteration-summary.md Section "Persistent
# failure modes" + the per-type breakdown in iter-35 v18.
#
# A = Premise-disagreement (5): model reverses verdict, says "No" when gold is Yes
# B = Verdict-buried / preamble (13): model writes substantive answer but doesn't
#     lead with verdict word
# C = Verdict-vocabulary mismatch (7): gold uses True/Different/Similar but
#     model says Yes/No
CATEGORY = {
    # Category A — premise-disagreement / wrong verdict (5)
    "mhrag_0c69b8fd": "A",
    "mhrag_39d3acb4": "A",
    "mhrag_3f3a1eff": "A",
    "mhrag_56d1f35e": "A",
    "mhrag_96f230ba": "A",
    # Category B — verdict-buried / preamble (13)
    "mhrag_1291bbe8": "B",
    "mhrag_1c6b36dd": "B",
    "mhrag_28d16fd4": "B",
    "mhrag_10cbd523": "B",
    "mhrag_14a3933e": "B",
    "mhrag_1aebcf0d": "B",
    "mhrag_253cf807": "B",
    "mhrag_42d704e0": "B",
    "mhrag_580b6de0": "B",
    "mhrag_5931848a": "B",
    "mhrag_5d4c3829": "B",
    "mhrag_2b8acb60": "B",
    "mhrag_ca18edbe": "B",
    # Category C — verdict-vocabulary mismatch (7)
    "mhrag_1388f62e": "C",
    "mhrag_2db51a4d": "C",
    "mhrag_34f651af": "C",
    "mhrag_351a3d54": "C",
    "mhrag_433b16f8": "C",
    "mhrag_595a561a": "C",
    "mhrag_8c07cbf7": "C",
}
assert len(CATEGORY) == 25, f"Expected 25, got {len(CATEGORY)}"


def load_dump(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {json.loads(l)["qid"]: json.loads(l) for l in f}


def main():
    # Load v18 baseline and confirm the 25 comparison failures are the ones
    # in CATEGORY.
    v18 = load_dump(V18_DUMP)
    if not v18:
        print(f"WARNING: v18 dump not found at {V18_DUMP}")
    # Restrict mismatch check to comparison-type failures only.
    with FIXTURE.open(encoding="utf-8") as f:
        items = json.load(f)
    qid_type = {it["_id"]: it["type"] for it in items}
    v18_comp_fail_qids = {
        qid for qid, r in v18.items()
        if qid_type.get(qid) == "comparison" and r.get("contains_gold", 0) < 1.0
    }
    cat_qids = set(CATEGORY.keys())
    missing_in_v18 = cat_qids - v18_comp_fail_qids
    extra_in_v18 = v18_comp_fail_qids - cat_qids
    if missing_in_v18 or extra_in_v18:
        print("MISMATCH between CATEGORY and v18 comparison failures:")
        print(f"  in CATEGORY but passed in v18: {sorted(missing_in_v18)}")
        print(f"  comparison-failed in v18 but not in CATEGORY: {sorted(extra_in_v18)}")
        print()

    # Load the 3 temp=0.3 runs.
    runs = [load_dump(p) for p in T03_RUNS]
    available_runs = [(i + 1, r) for i, r in enumerate(runs) if r]
    if not available_runs:
        print("No temp=0.3 runs found yet.")
        return
    print(f"Loaded {len(available_runs)} of 3 temp=0.3 runs.")
    print()

    # Per-qid stability table.
    print("=" * 78)
    print("PER-QID STABILITY (pass/fail across 3 temp=0.3 runs)")
    print("=" * 78)
    header = f'{"qid":<18} {"cat":<3} {"v18(t=0)":<9} {"R1":<5} {"R2":<5} {"R3":<5} {"recov":<5}'
    print(header)
    print("-" * 78)

    cat_recovery = {"A": [0, 0, 0, 0], "B": [0, 0, 0, 0], "C": [0, 0, 0, 0]}  # 0/1/2/3 passes
    per_qid_rows = []
    for qid in sorted(CATEGORY.keys(), key=lambda q: (CATEGORY[q], q)):
        cat = CATEGORY[qid]
        v18_pass = v18.get(qid, {}).get("contains_gold", 0) >= 1.0
        run_passes = []
        for _, run_dict in available_runs:
            p = run_dict.get(qid, {}).get("contains_gold", 0) >= 1.0
            run_passes.append(p)
        # Pad if not all 3 runs available
        while len(run_passes) < 3:
            run_passes.append(None)
        recovery = sum(p for p in run_passes if p)
        cat_recovery[cat][recovery] += 1
        per_qid_rows.append((qid, cat, v18_pass, run_passes, recovery))
        def fmt(b): return ("PASS" if b else ("FAIL" if b is not None else "----"))
        print(
            f"{qid:<18} {cat:<3} "
            f"{(fmt(v18_pass)):<9} "
            f"{fmt(run_passes[0]):<5} {fmt(run_passes[1]):<5} {fmt(run_passes[2]):<5} "
            f"{recovery}/3"
        )

    print()
    print("=" * 78)
    print("PER-CATEGORY RECOVERY (how many of the 3 runs passed each qid)")
    print("=" * 78)
    cat_size = {cat: sum(1 for c in CATEGORY.values() if c == cat) for cat in "ABC"}
    print(f'{"cat":<3} {"n":<3} {"0/3 passes":<11} {"1/3":<5} {"2/3":<5} {"3/3":<5} {"avg recovery":<12}')
    print("-" * 78)
    overall_recov = []
    for cat in "ABC":
        n = cat_size[cat]
        counts = cat_recovery[cat]
        avg = sum(i * c for i, c in enumerate(counts)) / n if n else 0
        overall_recov.append((cat, n, counts, avg))
        print(f"{cat:<3} {n:<3} {counts[0]:<11} {counts[1]:<5} {counts[2]:<5} {counts[3]:<5} {avg:.2f}/3")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    print("If a category's avg recovery is high (>= 2.0/3), the v18 failure was")
    print("sampling noise — the v18 SOTA was unlucky on those qids. No prompt")
    print("change is needed; just re-run.")
    print()
    print("If a category's avg recovery is low (<= 1.0/3), the v18 failure is")
    print("structural — the prompt consistently fails on those qids. v19 should")
    print("target the specific failure mode (e.g., lead-with-verdict directive for")
    print("Category B, verdict-vocab normalizer for Category C).")
    print()
    print("Category A is the premise-disagreement mode (model reverses verdict).")
    print("This is documented as untouchable across 9 prior attempts — a high")
    print("recovery here would actually be surprising.")


if __name__ == "__main__":
    main()
