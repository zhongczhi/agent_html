"""HotpotQA loader, sampling, and gold-derivation helpers."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Canonical URL for the dev-distractor JSON (CC BY-SA 4.0). Pinned here so
# ingest + eval agree on the dataset version. If hotpotqa.github.io changes
# hosting, this is the one constant to update; the dataset_sha cache prefix
# then busts every cached per-question index automatically.
HOTPOTQA_DEV_DISTRACTOR_URL = (
    "https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
)


@dataclass(frozen=True)
class HotpotQaItem:
    id: str
    question: str
    answer: str
    type: str
    level: str
    context: list[tuple[str, list[str]]]
    supporting_facts: list[tuple[str, int]]


def load(path: Path) -> list[HotpotQaItem]:
    """Load every question from the HotpotQA JSON at `path`. Raises
    json.JSONDecodeError on a corrupt file (handled by the CLI as exit 1)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[HotpotQaItem] = []
    for entry in raw:
        ctx = [(title, sentences) for title, sentences in entry["context"]]
        sf = [(title, int(idx)) for title, idx in entry["supporting_facts"]]
        items.append(
            HotpotQaItem(
                id=entry["_id"],
                question=entry["question"],
                answer=entry["answer"],
                type=entry["type"],
                level=entry["level"],
                context=ctx,
                supporting_facts=sf,
            )
        )
    return items


def dataset_sha(path: Path) -> str:
    """First 16 hex chars of SHA-256 of the file. Used as the cache-invalidation
    prefix in backend.eval.cache."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def gold_paragraph_titles(item: HotpotQaItem) -> set[str]:
    """Distinct paragraph titles appearing in the question's gold supporting facts."""
    return {title for title, _ in item.supporting_facts}


def sample(
    items: list[HotpotQaItem],
    n: int,
    seed: int = 42,
) -> list[HotpotQaItem]:
    """Stratified sampling across the 6 (type, level) buckets, deterministic.

    - n >= len(items): returns a deterministic shuffle of `items` unchanged in size.
    - n <= 1: raises ValueError; caller (CLI argparse) should reject before calling.

    Per-bucket cap is `min(ceil(n / 6), len(bucket))`, so small buckets cannot
    be over-sampled. The sampled set is then deterministically shuffled.
    """
    if n <= 1:
        raise ValueError("sample n must be >= 2; the CLI rejects smaller values")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[HotpotQaItem]] = {}
    for it in items:
        buckets.setdefault((it.type, it.level), []).append(it)
    per_bucket = max(1, -(-n // 6))
    sampled: list[HotpotQaItem] = []
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)
        sampled.extend(bucket_items[: per_bucket])
    sampled = sampled[:n]
    rng.shuffle(sampled)
    return sampled
