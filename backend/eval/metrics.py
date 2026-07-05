"""Retrieval-only metrics for HotpotQA evaluation.

All functions are pure: no I/O, no FAISS, no embeddings. The unit tests in
backend/tests/eval/test_metrics.py don't need any fixtures.
"""

from __future__ import annotations


def paragraph_recall_at_k(
    retrieved: list[str],
    gold: set[str],
) -> float:
    """Fraction of gold paragraphs appearing in the top-k retrieved list.

    Vacuously returns 1.0 when gold is empty. Capped at 1.0 (when k exceeds
    the number of gold paragraphs, "more hits than gold" is clamped).
    """
    if not gold:
        return 1.0
    hits = sum(1 for t in retrieved if t in gold)
    return min(hits, len(gold)) / len(gold)


def supporting_fact_metrics(
    retrieved: list[str],
    gold: set[str],
) -> tuple[float, float, float, float]:
    """Returns (precision, recall, f1, em).

    Edge cases (per SPEC_focus FR-31.11):
      - empty gold AND empty retrieved : (1, 1, 1, 1)
      - empty gold AND non-empty       : (0, 1, 0, 0)   (vacuous recall)
      - non-empty gold AND empty       : (0, 0, 0, 0)
      - both non-empty                 : standard formulas over set(predicted) vs set(gold)
    """
    if not gold and not retrieved:
        return (1.0, 1.0, 1.0, 1.0)
    if not retrieved:
        return (0.0, 0.0, 0.0, 0.0)
    if not gold:
        return (0.0, 1.0, 0.0, 0.0)
    pred = set(retrieved)
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    em = 1.0 if pred == gold else 0.0
    return (precision, recall, f1, em)
