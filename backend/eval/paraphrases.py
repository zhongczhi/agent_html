"""Paraphrase load + validate helpers. Pure: no I/O at import time, no LLM.

These are used by both scripts/generate_paraphrases_hotpotqa.py (validation
gate at generation time) and scripts/eval_hotpotqa.py (loading the JSON at
eval time). Keeping them in backend.eval.* lets both scripts import them
without violating the FR-32 isolation rule against backend.chat.* imports.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REQUIRED_STYLES: tuple[str, ...] = ("lexical", "structural", "casual")

# Tokens from the gold answer that appear in the paraphrase at this fraction
# of the gold-answer tokens -> reject. 0.80 means ">=4 of 5 gold tokens in
# the paraphrase -> the LLM is leaking".
_LEAK_OVERLAP_THRESHOLD = 0.80

_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def required_styles() -> tuple[str, ...]:
    """The three styles the generator must produce per question."""
    return _REQUIRED_STYLES


def _normalize(text: str) -> str:
    """Lowercase, strip non-word/non-space characters, collapse whitespace."""
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def validate_paraphrase(paraphrase: str, gold_answer: str) -> bool:
    """Return True iff the paraphrase is acceptable.

    Reject if >= 80% of the gold-answer tokens appear in the paraphrase
    (case-insensitive, punctuation-stripped). Returns False for empty
    paraphrase or empty gold answer — both are degenerate.
    """
    if not paraphrase or not gold_answer:
        return False
    gold_tokens = set(_normalize(gold_answer).split())
    para_tokens = set(_normalize(paraphrase).split())
    if not gold_tokens or not para_tokens:
        return False
    overlap = len(gold_tokens & para_tokens) / len(gold_tokens)
    return overlap < _LEAK_OVERLAP_THRESHOLD


def load_paraphrases(path: Path) -> dict[str, dict[str, str]]:
    """Load the JSON at `path` and return its `items` dict.

    Shape: {qid: {"paraphrases": {style: text, ...}}, ...}

    Raises FileNotFoundError if path missing; json.JSONDecodeError if the
    file is not valid JSON. A JSON object with no `items` key returns {}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, dict):
        return {}
    return items


def lookup(paraphrases: dict[str, dict[str, str]], qid: str) -> dict[str, str] | None:
    """Return the {style: text} dict for `qid`, or None if absent."""
    entry = paraphrases.get(qid)
    if not isinstance(entry, dict):
        return None
    paras = entry.get("paraphrases")
    if not isinstance(paras, dict):
        return None
    return paras