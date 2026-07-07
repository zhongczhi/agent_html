"""Retrieval-only metrics for HotpotQA evaluation.

All functions are pure: no I/O, no FAISS, no embeddings. The unit tests in
backend/tests/eval/test_metrics.py don't need any fixtures.
"""

from __future__ import annotations

import re

# Pre-compile at module load — the character class is constant and the
# regex runs once per query.
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


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


def _normalize_for_coverage(text: str) -> str:
    """Lowercase, strip non-word/non-space characters, collapse whitespace.
    'Yes,' -> 'yes'; '  multi  word ' -> 'multi word'."""
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def answer_coverage_at_k(retrieved_texts: list[str], gold_answer: str) -> float:
    """Returns 1.0 if the normalized gold_answer is a substring of any
    normalized retrieved text (paragraphs joined with newlines). Else 0.0.

    Vacuous: empty gold_answer -> 1.0 (matches the convention used by
    paragraph_recall_at_k). Empty retrieved -> 0.0.

    Why substring (not token-set): HotpotQA gold answers are typically
    1-3 words ('yes', '1968', 'The Godfather'). Substring containment after
    normalization is the simplest heuristic that handles all three forms.
    """
    gold = _normalize_for_coverage(gold_answer)
    if not gold:
        return 1.0
    if not retrieved_texts:
        return 0.0
    blob = "\n".join(_normalize_for_coverage(t) for t in retrieved_texts)
    return 1.0 if gold in blob else 0.0
