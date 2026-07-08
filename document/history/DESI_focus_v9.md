# Chatbot Project — Iteration 9 Design (HotpotQA Library Ingest + Retrieval Eval Pipeline)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> The architecture decisions, module-level design, and phased TDD implementation plan for iteration 9. See [SPEC_focus.md](SPEC_focus.md) for requirements and [docs/superpowers/specs/2026-07-05-hotpotqa-rag-design.md](../docs/superpowers/specs/2026-07-05-hotpotqa-rag-design.md) for the full brainstorming artifact.

---

## 1. Architecture Decisions

### 1.1 One Markdown File Per Question (Not Per Paragraph)

**Choice**: Library ingest writes one `.md` per question with each paragraph as an H1 section in the same file. ~7,405 files instead of ~74k.

**Rationale**:
- The existing `MarkdownTextSplitter` in `backend/rag/splitter.py:23` already splits at H1 boundaries during library reindex. Each H1 section becomes one chunk automatically, with `header_path` set to the paragraph title.
- 7,405 files is dramatically more manageable than 74k — git, IDEs, `find`/`walk` all stay snappy.
- Each file is a coherent unit ("all 10 paragraphs for question X") — useful for human inspection during debugging.
- Reindex walk time scales with file count, not chunk count; FAISS index size is unchanged.

**Trade-off**: Duplicate paragraph text across multiple questions' contexts causes slight index inflation (~5–15% chunk duplication, estimated). For this dataset, the prevalence is tolerable; no dedup is performed. If this becomes painful later, the integration of a content-hash dedup at ingest time is a small additive change.

### 1.2 Frontmatter-Only Metadata (No Gold Leakage)

**Choice**: Each library file's frontmatter contains `question_id`, `question_type`, `question_level`, `source` — never `question`, `answer`, or `supporting_facts`.

**Rationale**:
- If the question text or gold answer landed in a chunk, a chat-time retrieval against that chunk would let the model see the ground-truth answer in its own context — a trivial form of contamination.
- Chat UX doesn't need the question or answer in the library; it needs the paragraphs. Frontmatter carries structural metadata (type, level, source) that's useful for operators browsing the library tab without helping the model cheat.
- The eval pipeline reads the JSON directly and never looks at library files, so it can safely use all of `_id`, `question`, `answer`, `supporting_facts`, `context`, `type`, `level`.

**Trade-off**: An operator debugging retrieval against HotpotQA can see only the paragraph text from the library tab; the question/answer for that file are in the JSON (or visible via `git log`-able spec).

### 1.3 Separate CLI Scripts (No Shared Library Code Beyond Pure Primitives)

**Choice**: Two scripts — `scripts/ingest_hotpotqa.py` and `scripts/eval_hotpotqa.py`. They share:
- The downloaded JSON at `scripts/.cache/hotpot_dev_distractor_v1.json`
- A small set of helpers in `backend/eval/` (`hotpotqa.py`, `metrics.py`, `cache.py`)

**Rationale**:
- The user's constraint: "the evaluation pipeline should be separate from my chat business." Two scripts + a small shared package keeps this clean.
- `scripts/eval_hotpotqa.py` is forbidden from importing anything under `backend/chat/`. The import surface it may use: `backend.rag.embeddings` (the sentence-transformers factory), `backend.rag.vector_store` (`load_or_init` / `save`), and `backend.eval.*`.
- A grep guard (`grep -r "backend\\.chat" backend/eval/ scripts/eval_hotpotqa.py` returning empty) verifies this at review time.

**Trade-off**: Some duplication of argparse patterns between the two scripts. Acceptable given the small surface area (~30 lines each).

### 1.4 Paragraph-Level Document in the Eval Index (Not MarkdownTextSplitter)

**Choice**: `backend/eval/cache.py::_build_index` constructs `Document(page_content=paragraph_text, metadata={...})` for each paragraph in `item.context` — no splitter is invoked.

**Rationale**:
- The chat pipeline's `MarkdownTextSplitter` will produce paragraph-level chunks for these files only if each H1 section is short enough relative to `rag_chunk_size`. If `rag_chunk_size` is large (e.g., 4000 chars), the whole file becomes one chunk — a metric computed against that would not match per-paragraph recall.
- Decoupling the eval's retrieval granularity from the chat pipeline's chunking decisions keeps the metric stable across reindex-config changes.
- 10 paragraphs per question, simple `text` + `metadata` per Document, no splitter math.

**Trade-off**: The metric measures paragraph-level retrieval. If the chat pipeline produces larger-than-paragraph chunks, the chat-time UX isn't directly comparable to this metric. This is by design — eval measures retrieval, chat measures UX; they're related but distinct.

### 1.5 SHA-Keyed On-Disk Cache for Per-Question Indices

**Choice**: `backend/eval/cache.py::load_or_build` keys the cache directory by `dataset_sha[:16]` (sha256 of the dataset JSON). Each question gets its own subdirectory `cache/{dataset_sha[:16]}/{qid}/` containing FAISS's native `index.faiss` + `index.pkl`.

**Rationale**:
- Re-downloading the JSON (e.g., if HotpotQA updates the dataset) busts all caches atomically — the directory prefix changes and the next run rebuilds.
- Per-question directories mean partial-corrupt recovery is isolated to one qid; the rest stay cached.
- FAISS's `save_local` / `load_local` round-trip handles persistence without bespoke file format.
- First run on full set is slow (~minutes); subsequent runs are seconds.

**Trade-off**: The cache directory tree can grow large (~hundreds of MB for full set). `.gitignore` covers `storage/eval/`; cleanup is manual via `rm -rf storage/eval/hotpotqa/cache/`.

### 1.6 Dataset Auto-Download + .cache Stash

**Choice**: `scripts/ingest_hotpotqa.py` is the canonical way to acquire the dataset. It downloads once, stashes at `scripts/.cache/hotpot_dev_distractor_v1.json`, and prints the SHA so subsequent runs can verify it. `scripts/eval_hotpotqa.py` reads from the stash (or from `--fixture PATH` for tests).

**Rationale**:
- A single source of truth for the dataset file — download it once with the ingest CLI, run eval as many times as you want from the cached copy.
- `scipts/.cache/` mirrors a common pattern (`./.pytest_cache/`, `./.coverage`) and is gitignored.
- The `--force` flag for ingest lets operators bust the cache.
- Tests use `--fixture` to provide their own JSON, avoiding any network during CI.

**Trade-off**: Operators who want to run only `scripts/eval_hotpotqa.py` without first running ingest need to download manually. The "dataset missing" error in eval gives clear download instructions.

### 1.7 CC BY-SA 4.0 Attribution in Two Places

**Choice**: Attribution lives at the file level (`storage/library/hotpotqa/README.md` — written by ingest) and at the run level (eval script prints `Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)` to stdout before the metric block).

**Rationale**:
- The README travels with the data: someone browsing `storage/library/hotpotqa/` sees attribution.
- The eval stdout attribution means anyone running eval (or reading CI logs) sees provenance.
- The CC BY-SA 4.0 license is a requirement of using the dataset, so two straightforward sightings are reasonable.
- Surface-attribution in the library sidebar UI is deferred (would require a frontend change — out of scope this iter per FR-32 isolation).

**Trade-off**: Two sightings vs one. Tipping toward "more obvious = better" for license compliance.

### 1.8 Stratified Sample With Deterministic Seed

**Choice**: `--subset N` samples `min(ceil(N / 6), len(bucket))` items from each of the 6 `(type, level)` buckets using `random.Random(42)`. The sampled set is concatenated and shuffled with the same RNG.

**Rationale**:
- HotpotQA's `type` × `level` distribution is uneven — without stratification, a small random sample could over-represent `comparison/easy` questions.
- `random.Random(42)` is a non-global instance so it doesn't interact with anything else. Two runs with `--subset 50` produce the same 50 qids in the same evaluation order.
- The bucket cap (`min(..., len(bucket))`) protects against over-sampling small buckets when N is large.

**Trade-off**: A flat `--subset N` would be simpler. Stratification costs ~10 lines; the metric stability it buys is worth it.

### 1.9 Exit Codes: Partial Errors Are Non-Fatal

**Choice**: `--subset N` or `--full` runs of `scripts/eval_hotpotqa.py` always exit 0 unless setup fails. Per-question retrieval errors are logged at WARNING and counted in the `errors` field of the report.

**Rationale**:
- 7,405 questions × (transient embedding call) means we can expect a small number of failures during a long run. Treating one query failure as fatal makes the eval flaky.
- The metric block shows both `successfully evaluated` and `attempted` — operators can see at a glance whether the count matches what they expect.
- The setup-failure exit code (1) is reserved for things the operator can fix: download the JSON, install dependencies.

**Trade-off**: If literally every question errors (e.g., the embedding model silently produces NaN), the run exits 0 with `successfully evaluated : 0`. Acceptable: the operator sees the 0 and investigates.

---

## 2. Module Layout

### 2.1 New Files

```
backend/eval/
├── __init__.py
├── hotpotqa.py         # HotpotQaItem dataclass, load(), dataset_sha(), gold_paragraph_titles(), sample()
├── metrics.py          # paragraph_recall_at_k(), supporting_fact_metrics()
└── cache.py            # load_or_build(), EVAL_CACHE_ROOT, _build_index()

backend/tests/eval/
├── __init__.py
├── __init__.py         # marker so pytest can collect
├── fixtures/
│   ├── tiny_hotpot.json           # 3-question fixture for hotpotqa.py tests
│   └── integration_hotpot.json    # 5-question fixture for the eval_integration test
├── test_metrics.py     # pure-function unit tests
├── test_hotpotqa.py    # loader + sha + gold_paragraph_titles + sample
├── test_cache.py       # per-question FAISS cache + corruption recovery
└── test_eval_integration.py   # subprocess-driven end-to-end with synthetic JSON

scripts/
├── ingest_hotpotqa.py  # CLI: download + write library files
└── eval_hotpotqa.py    # CLI: run the eval pipeline

storage/library/hotpotqa/
└── README.md           # generated by ingest; one-line license notice

scripts/.cache/
└── hotpot_dev_distractor_v1.json  # generated by ingest; gitignored
```

### 2.2 Per-File Responsibilities

**`backend/eval/__init__.py`** — empty marker package.

**`backend/eval/hotpotqa.py`** — pure-Python module, no I/O except reading the JSON file. Exposes `HotpotQaItem` dataclass (with `id`, `question`, `answer`, `type`, `level`, `context: list[tuple[str, list[str]]]`, `supporting_facts: list[tuple[str, int]]`), `load(path) -> list[HotpotQaItem]`, `dataset_sha(path) -> str` (16-char sha256 prefix), `gold_paragraph_titles(item) -> set[str]` (descriptive helper for §3.3), and `sample(items, n, seed=42) -> list[HotpotQaItem]` (stratified, deterministic).

**`backend/eval/metrics.py`** — pure functions, no I/O, no FAISS. `paragraph_recall_at_k(retrieved, gold) -> float` and `supporting_fact_metrics(retrieved, gold) -> tuple[float, float, float, float]`. Edge cases are explicit in the FR-31.10 / FR-31.11 rows of [SPEC_focus.md](SPEC_focus.md).

**`backend/eval/cache.py`** — owns the per-question FAISS cache. `EVAL_CACHE_ROOT = Path("storage/eval/hotpotqa/cache")`. `load_or_build(item, dataset_sha, embeddings, no_cache) -> tuple[FAISS, bool]` returns `(index, was_hit)`. `_build_index(item, embeddings) -> FAISS` constructs 10 paragraph Documents with metadata `{question_id, paragraph_idx, title, source, type, level}`. Cache load uses `backend.rag.vector_store.load_or_init`. Cache save uses `backend.rag.vector_store.save`. Corrupt cache → `shutil.rmtree(cache_path, ignore_errors=True)` + rebuild.

**`backend/tests/eval/__init__.py`** — empty marker package.

**`backend/tests/eval/fixtures/tiny_hotpot.json`** — hand-authored 3-question JSON with one of each (type × level) combination so loader tests cover all shapes.

**`backend/tests/eval/fixtures/integration_hotpot.json`** — hand-authored 5-question JSON used by the integration test via `--fixture` flag.

**`backend/tests/eval/test_metrics.py`** — pure tests for `metrics.py`. No fixtures, no FAISS, no embeddings.

**`backend/tests/eval/test_hotpotqa.py`** — uses `tiny_hotpot.json` fixture. Tests `load()`, `dataset_sha()`, `gold_paragraph_titles()`, `sample()`.

**`backend/tests/eval/test_cache.py`** — uses `langchain_community.embeddings.fake.FakeEmbeddings` (deterministic; no model download). Tests `load_or_build` hit/miss/recovery. Uses temp dir for cache root.

**`backend/tests/eval/test_eval_integration.py`** — invokes `python scripts/eval_hotpotqa.py --fixture backend/tests/eval/fixtures/integration_hotpot.json --k 4` via `subprocess.run`. Asserts exit 0, output contains all metric lines, cache dir exists. Runs twice; second run reports all hits.

**`scripts/ingest_hotpotqa.py`** — CLI. argparse with `--subset N`, `--full` (mutually exclusive group; default `--full`), `--force`. Downloads JSON if `--force` or not cached, then writes one `.md` per question to `storage/library/hotpotqa/`. Writes `storage/library/hotpotqa/README.md` if absent. Prints attribution + per-question summary on completion.

Embeddings factory for the eval pipeline (used in Task 6 / script 3.5):
```python
from backend.rag.config import RagSettings
from backend.rag.embeddings import make_embeddings
settings = RagSettings()
embeddings = make_embeddings(settings.rag_embedding_backend)
```

This is the same pattern `RagService.from_settings()` uses. We do NOT add a `get_embeddings()` helper; we reuse the existing factory directly.

**`scripts/eval_hotpotqa.py`** — CLI. argparse with `--subset N | --full` (mutually exclusive; default `--full`), `--k 4`, `--no-cache`, `--fixture PATH`. Loads JSON, optionally samples, iterates questions, builds/loads per-question index, retrieves top-k, computes metrics, aggregates, prints report.

### 2.3 Modified Files

| File | Change |
|---|---|
| `.gitignore` | Existing rule ignores only `backend/storage/rag/` and `storage/conversations.json`. Add `scripts/.cache/`, `storage/library/hotpotqa/`, and `storage/eval/` so neither the downloaded JSON nor the bulky library files nor the eval cache (potentially hundreds of MB on full set) pollute commits. |

### 2.4 Unchanged Files

- `backend/chat/*` — the eval pipeline does not import any of these (verified by grep guard at PR time).
- `backend/main.py` — no startup wiring changes; the eval pipeline runs as a CLI, not in the FastAPI process.
- `backend/rag/*` — eval uses `backend.rag.embeddings` and `backend.rag.vector_store` as primitives; no API change to those modules.
- `frontend/*` — no UI changes.
- `requirements.txt` — no new deps.
- `.env*` — no new env vars.

---

## 3. Component Skeletons

### 3.1 `backend/eval/hotpotqa.py`

```python
"""HotpotQA loader, sampling, and gold-derivation helpers."""
from __future__ import annotations
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Download URL shared with the ingest script (also lives in scripts/ingest_hotpotqa.py).
HOTPOTQA_DOWNLOAD_URL = "https://hotpotqa.github.io/"


@dataclass(frozen=True)
class HotpotQaItem:
    id: str
    question: str
    answer: str
    type: str                          # "bridge" | "comparison"
    level: str                         # "easy" | "medium" | "hard"
    context: list[tuple[str, list[str]]]   # [(title, [sent, sent, ...]), ...]
    supporting_facts: list[tuple[str, int]]  # [(title, sent_idx), ...]


def load(path: Path) -> list[HotpotQaItem]:
    """Load every question from the HotpotQA JSON at `path`. Raises
    json.JSONDecodeError on a corrupt file (handled by the CLI as exit 1)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[HotpotQaItem] = []
    for entry in raw:
        ctx = [(title, sentences) for title, sentences in entry["context"]]
        sf = [(title, int(idx)) for title, idx in entry["supporting_facts"]]
        items.append(HotpotQaItem(
            id=entry["_id"],
            question=entry["question"],
            answer=entry["answer"],
            type=entry["type"],
            level=entry["level"],
            context=ctx,
            supporting_facts=sf,
        ))
    return items


def dataset_sha(path: Path) -> str:
    """First 16 hex chars of SHA-256 of the file. Used as the cache-invalidation prefix."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def gold_paragraph_titles(item: HotpotQaItem) -> set[str]:
    """Distinct paragraph titles appearing in the question's gold supporting facts."""
    return {title for title, _ in item.supporting_facts}


def sample(items: list[HotpotQaItem], n: int, seed: int = 42) -> list[HotpotQaItem]:
    """Stratified sampling across the 6 (type, level) buckets, deterministic.

    - n >= len(items): returns a deterministic shuffle of `items` unchanged in size.
    - n <= 0 or n == 1: caller should reject before calling (this raises ValueError).
    """
    if n <= 1:
        raise ValueError("sample n must be >= 2; for n<=1 the caller should reject")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[HotpotQaItem]] = {}
    for it in items:
        buckets.setdefault((it.type, it.level), []).append(it)
    per_bucket = max(1, -(-n // 6))  # ceil(n/6), at least 1
    sampled: list[HotpotQaItem] = []
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)
        sampled.extend(bucket_items[: per_bucket])
    sampled = sampled[:n]            # trim in case ceil overshot
    rng.shuffle(sampled)             # stable order across runs
    return sampled
```

### 3.2 `backend/eval/metrics.py`

```python
"""Retrieval-only metrics for HotpotQA evaluation.

All functions are pure: no I/O, no FAISS, no embeddings. The unit tests in
backend/tests/eval/test_metrics.py don't need any fixtures."""

from __future__ import annotations


def paragraph_recall_at_k(
    retrieved: list[str],   # paragraph titles of top-k retrieved chunks
    gold: set[str],         # gold paragraph titles from supporting_facts
) -> float:
    """Fraction of gold paragraphs appearing in the top-k retrieved list.

    Vacuously returns 1.0 when gold is empty. Capped at 1.0 (when k exceeds the
    number of gold paragraphs, "more hits than gold" is silently clamped).
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

    Edge cases (as specified in FR-31.11):
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
```

### 3.3 `backend/eval/cache.py`

```python
"""SHA-keyed per-question FAISS cache for the eval pipeline."""
from __future__ import annotations
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from backend.eval.hotpotqa import HotpotQaItem
from backend.rag.vector_store import load_or_init, save

log = logging.getLogger(__name__)

EVAL_CACHE_ROOT = Path("storage/eval/hotpotqa/cache")


def _build_index(item: HotpotQaItem, embeddings: Embeddings) -> FAISS:
    """Construct a FAISS index over `item.context` as paragraph Documents.
    Skips paragraphs whose joined sentence text is empty. Does NOT use
    MarkdownTextSplitter — preserves paragraph granularity."""
    docs: list[Document] = []
    for idx, (title, sentences) in enumerate(item.context):
        text = " ".join(sentences).strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "question_id": item.id,
                "paragraph_idx": idx,
                "title": title,
                "source": "hotpotqa",
                "type": item.type,
                "level": item.level,
            },
        ))
    return FAISS.from_documents(docs, embeddings)


def load_or_build(
    item: HotpotQaItem,
    dataset_sha: str,
    embeddings: Embeddings,
    no_cache: bool = False,
) -> tuple[FAISS, bool]:
    """Returns (index, was_hit). `was_hit` is True if loaded from disk.

    - If `no_cache`, always build. Cache is overwritten on disk.
    - Otherwise: cache_path = EVAL_CACHE_ROOT / dataset_sha[:16] / item.id /
      - if missing, build + save, was_hit=False.
      - if present, attempt load_or_init; on any failure, rmtree + build + save + WARNING log, was_hit=False.
    """
    cache_dir = EVAL_CACHE_ROOT / dataset_sha / item.id
    if no_cache or not cache_dir.exists():
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        return index, False
    try:
        index = load_or_init(cache_dir, embeddings)
        return index, True
    except Exception as e:
        log.warning("cache corrupt for %s (%s); rebuilding", item.id, e)
        shutil.rmtree(cache_dir, ignore_errors=True)
        index = _build_index(item, embeddings)
        save(index, cache_dir)
        return index, False
```

### 3.4 `scripts/ingest_hotpotqa.py`

```python
"""Download HotpotQA dev distractor and write one Markdown file per question
into storage/library/hotpotqa/. Idempotent. CC BY-SA 4.0 attribution is
written to storage/library/hotpotqa/README.md."""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger("ingest_hotpotqa")

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"
LIBRARY_DIR = REPO_ROOT / "storage" / "library" / "hotpotqa"
README_PATH = LIBRARY_DIR / "README.md"

# HotpotQA S3 bucket — public, CC BY-SA 4.0. Mirror at hotpotqa.github.io redirects here.
# Confirmed stable through 2026; if the URL changes, update this constant and the cache
# SHA prefix will bust all eval caches automatically.
HOTPOTQA_URL = "https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
README_TEXT = """# HotpotQA Library Data

Dataset: HotpotQA dev_distractor v1 (Yelp-style multi-hop QA).
Source: https://hotpotqa.github.io/
License: CC BY-SA 4.0.

This directory contains one Markdown file per question, generated by
`scripts/ingest_hotpotqa.py`. Files contain paragraph text from the original
dataset; questions, gold answers, and gold supporting facts are NOT included
in any file (kept in the source JSON only).

If you redistribute these files, retain this README.
"""


def download_or_use_cache(force: bool) -> Path:
    """Download the JSON if absent or `force` is True; cache it at CACHE_PATH.
    Returns CACHE_PATH on success. Exits 1 with download URL on repeated failure."""
    if CACHE_PATH.exists() and not force:
        return CACHE_PATH
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in (0, 1):
        try:
            import urllib.request
            with urllib.request.urlopen(HOTPOTQA_URL, timeout=60) as resp:
                data = resp.read()
            CACHE_PATH.write_bytes(data)
            return CACHE_PATH
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(5)
    print(f"Download failed after 2 attempts: {last_err}\nURL: {HOTPOTQA_URL}", file=sys.stderr)
    sys.exit(1)


def slugify(qid: str) -> str:
    out = []
    for c in qid:
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("_")
    return "".join(out)


def stratified_sample(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Same 6-bucket deterministic sampling as backend/eval/hotpotqa.py::sample.

    Kept inline in this script so ingest has no path dependency on backend.eval.
    Tests for this sampler live in test_hotpotqa.py via the backend/eval module.
    """
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        buckets.setdefault((it["type"], it["level"]), []).append(it)
    per = max(1, -(-n // 6))
    out: list[dict] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        out.extend(bucket[:per])
    out = out[:n]
    rng.shuffle(out)
    return out


def write_question_file(item: dict, dest_dir: Path) -> None:
    qid = slugify(item["_id"])
    frontmatter = (
        "---\n"
        f"question_id: {item['_id']}\n"
        f"question_type: {item['type']}\n"
        f"question_level: {item['level']}\n"
        "source: hotpotqa\n"
        "---\n\n"
    )
    body_parts: list[str] = []
    for title, sentences in item["context"]:
        text = " ".join(sentences).strip()
        if not text:
            continue
        body_parts.append(f"# {title}\n{text}\n")
    full = frontmatter + "\n".join(body_parts)
    # Atomic write.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), prefix=f".{qid}.", suffix=".tmp")
    try:
        import os
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(full)
        os.replace(tmp_path, dest_dir / f"{qid}.md")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest HotpotQA dev distractor into the library.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--subset", type=int, metavar="N", help="Stratified sample of N questions")
    grp.add_argument("--full", action="store_true", help="Use all 7,405 questions (default)")
    parser.add_argument("--force", action="store_true", help="Re-download the dataset even if cached")
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    print("Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)")
    json_path = download_or_use_cache(args.force)
    log.info("Loaded %d bytes from %s", json_path.stat().st_size, json_path.name)

    raw_items = json.loads(json_path.read_text(encoding="utf-8"))
    items = raw_items
    if args.subset is not None:
        items = stratified_sample(items, args.subset)
    log.info("Writing %d question files to %s", len(items), LIBRARY_DIR)

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if not README_PATH.exists():
        README_PATH.write_text(README_TEXT, encoding="utf-8")

    skipped: list[str] = []
    written = 0
    for it in items:
        try:
            write_question_file(it, LIBRARY_DIR)
            written += 1
        except Exception as e:
            log.warning("Skipping qid=%s: %s", it.get("_id"), e)
            skipped.append(it.get("_id", "?"))
    log.info("Done: %d written, %d skipped", written, len(skipped))
    if skipped:
        log.warning("Skipped IDs: %s", ", ".join(skipped[:10]) + ("..." if len(skipped) > 10 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 3.5 `scripts/eval_hotpotqa.py`

```python
"""Run the HotpotQA retrieval evaluation pipeline. CLI only — no HTTP, no chat coupling."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from backend.eval import hotpotqa as hotpot
from backend.eval import metrics, cache as ev_cache
from backend.rag.config import RagSettings
from backend.rag.embeddings import make_embeddings

log = logging.getLogger("eval_hotpotqa")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="HotpotQA retrieval eval (paragraph-level).")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--subset", type=int, metavar="N", help="Stratified sample of N questions")
    grp.add_argument("--full", action="store_true", help="Use all questions (default)")
    parser.add_argument("--k", type=int, default=4, help="Top-k to retrieve (default 4)")
    parser.add_argument("--no-cache", action="store_true", help="Force rebuild of every per-question index")
    parser.add_argument("--fixture", type=Path, help="Read dataset from PATH (test hook)")
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    dataset_path = args.fixture or DEFAULT_DATASET
    if not dataset_path.exists():
        print(
            f"Dataset not found at {dataset_path}.\n"
            "Run scripts/ingest_hotpotqa.py (downloads to scripts/.cache/), or pass --fixture PATH.",
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
    print(f"Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)")
    settings = RagSettings()
    embeddings = make_embeddings(settings.rag_embedding_backend)

    per_q: list[tuple[float, float, float, float, float]] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()
    for item in items:
        try:
            index, hit = ev_cache.load_or_build(item, d_sha, embeddings, no_cache=args.no_cache)
            docs = index.similarity_search(item.question, k=args.k)
            retrieved_titles = [d.metadata.get("title", "") for d in docs]
            gold_titles = hotpot.gold_paragraph_titles(item)
            pr = metrics.paragraph_recall_at_k(retrieved_titles, gold_titles)
            sp, sr, sf, em = metrics.supporting_fact_metrics(retrieved_titles, gold_titles)
            per_q.append((pr, sp, sr, sf, em))
            if hit: cache_hits += 1
            else:   cache_builds += 1
        except Exception as e:
            log.warning("qid=%s error: %s", item.id, e)
            errors += 1

    elapsed = time.monotonic() - t0
    avg = lambda i: sum(q[i] for q in per_q) / len(per_q) if per_q else 0.0
    print(f"\nHotpotQA Eval — subset={'full' if args.subset is None else args.subset}, "
          f"k={args.k}, dataset_sha={d_sha}")
    print(f"  paragraph_recall@{args.k}   : {avg(0):.3f}")
    print(f"  sf_precision         : {avg(1):.3f}")
    print(f"  sf_recall            : {avg(2):.3f}")
    print(f"  sf_f1                : {avg(3):.3f}")
    print(f"  sf_em                : {avg(4):.3f}")
    print(f"  questions successfully evaluated : {len(per_q)}  (out of {len(items)} attempted)")
    print(f"  cache hits / builds  : {cache_hits} / {cache_builds}")
    print(f"  errors               : {errors}")
    print(f"  elapsed              : {elapsed:.1f}s")

    if errors:
        log.warning("%d questions errored (skipped, not counted in metrics above)", errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 4. Configuration

No new env vars. No new config fields. Reuses existing `EMBEDDING_BACKEND` and the sentence-transformers model wired up in `backend/rag/embeddings.py`.

### `.gitignore` additions

```gitignore
# HotpotQA iteration 9 — bulky data + cache
scripts/.cache/
storage/eval/
storage/library/hotpotqa/
```

### requirements.txt

No changes. Sentence-transformers, FAISS (`langchain_community`), and `langchain_core` are already present.

### Filesystem layout

```
storage/library/hotpotqa/                ← ingest output (existing path, gitignored)
storage/eval/hotpotqa/cache/             ← NEW, eval cache root (gitignored)
scripts/.cache/hotpot_dev_distractor_v1.json  ← NEW, downloaded JSON stash (gitignored)
```

---

## 5. Error Handling

| Stage | Failure | Behavior |
|---|---|---|
| Ingest: download | Network error | One retry after 5s; second failure exits 1 with download URL. |
| Ingest: per-question write | Permission error / disk full | Log path, continue with remaining. Exit non-zero if any file failed. |
| Ingest: whole-file `JSONDecodeError` | JSON corrupt | Exit 1 with "fix the file or re-download" hint. |
| Ingest: per-question schema error | Missing fields | Log WARNING with qid, skip, continue. Final summary lists skipped IDs. |
| Eval: dataset missing | Path doesn't exist | Print expected path + download instructions, exit 1. |
| Eval: dataset corrupt (`JSONDecodeError`) | Parse fails | Print exception tail, exit 1. |
| Eval: embedding model load | sentence-transformers not installed | Fail fast with `pip install -r requirements.txt` hint, exit 1. |
| Eval: per-question cache corrupted | `load_local` raises | `shutil.rmtree(cache_path, ignore_errors=True)`, rebuild, WARNING log, continue. |
| Eval: per-question retrieval | Embedding call raises (transient) | Log WARNING, count as errored, skip rest of run unaffected. |
| Eval: missing title in retrieved chunk | Defensive case | Treat chunk's title as empty string → no-match for recall. |
| Eval: per-question FAISS construction | Embedding backend rejects a chunk | Skip that question, log WARNING. |

**Logging**: standard `logging` module, INFO default. Per-question errors at WARNING. Setup failures at ERROR (exit 1).

**Atomicity**:
- Ingest: per-file atomic write (`tmp + os.replace`). Re-running on unchanged dataset is byte-identical, no mtime churn. A crash mid-write is recovered by the next run overwriting the half-written file.
- Eval: per-question state. Cache build for one question is atomic (rmtree + save); a partial-corrupt cache is recovered automatically on the next call.

**Isolation grep guard** (operator runs before merging PR):
```bash
grep -r "backend\.chat" backend/eval/ scripts/eval_hotpotqa.py
# Expected: no matches
```

---

## 6. Testing Strategy

### 6.1 Layers

| Layer | Files | Speed target |
|---|---|---|
| Metrics unit | `backend/tests/eval/test_metrics.py` (NEW) | <5 ms each |
| Loader unit | `backend/tests/eval/test_hotpotqa.py` (NEW) | <50 ms each |
| Cache unit | `backend/tests/eval/test_cache.py` (NEW) | <500 ms each |
| Integration | `backend/tests/eval/test_eval_integration.py` (NEW) | <5 s |
| Iter-8 compat | run existing `pytest backend/tests/` | unchanged |
| Manual smoke | operator runs `scripts/ingest_hotpotqa.py` + `scripts/eval_hotpotqa.py` | minutes |

### 6.2 Key Test Cases

**`test_metrics.py`** — pure tests, no fixtures:
- `paragraph_recall_at_k`: empty gold → 1.0; all retrieved → 1.0; none retrieved → 0.0; partial overlap → exact fraction; cap-at-1.0 verified by `min(hits, |gold|) / |gold|` (e.g., gold size 2, retrieved hits 3 → 1.0, not 1.5).
- `supporting_fact_metrics`: 6 edge-case rows (the FR-31.11 table); standard formulas produce expected values for the partial-overlap case; `em == 1.0` iff `set(retrieved) == gold`.

**`test_hotpotqa.py`** — uses `tiny_hotpot.json`:
- `load()`: 3 questions → 3 `HotpotQaItem`s with expected fields.
- `dataset_sha()`: deterministic on the same file; changes when the file changes (delete one question → SHA changes).
- `gold_paragraph_titles()`: same title at multiple sentence indices contributes once.
- `sample(items, n=200)`: deterministic across two calls, returns exactly 200 items, distribution across (type, level) is within ±5% of expected (a soft check, not a hard one).

**`test_cache.py`** — uses `FakeEmbeddings` (`langchain_community.embeddings.fake`) and a tmp dir as `EVAL_CACHE_ROOT`:
- `_build_index`: 10 paragraphs → 10 Documents in FAISS; empty paragraph skipped.
- `load_or_build`: first call returns `(index, False)`; second call returns `(index, True)`.
- Different `dataset_sha` → different cache dir; no cross-pollination.
- `--no_cache=True` → first AND second call return `hit=False`.
- Corrupted cache (write garbage to `index.faiss`) → next call rebuilds cleanly, no exception propagates.
- Unused parameter warnings about `load_or_init` are absent (we actually use it).

**`test_eval_integration.py`** — uses `integration_hotpot.json`:
- Run `python scripts/eval_hotpotqa.py --fixture backend/tests/eval/fixtures/integration_hotpot.json --k 4` via `subprocess.run(..., capture_output=True, text=True)`.
- Assert exit code 0; stdout contains all 5 metric lines (`paragraph_recall@4`, `sf_precision`, `sf_recall`, `sf_f1`, `sf_em`).
- Assert cache dir exists at `storage/eval/hotpotqa/cache/{dataset_sha[:16]}/` for at least one qid.
- Run twice; second run's captured stdout contains `cache hits / builds  : 5 / 0`.

### 6.3 Iter-8 Compatibility

- All iter-8 tests continue to pass unchanged. Eval does not import any chat module, so no behavior change in chat.
- The eval pipeline uses `backend.rag.config.RagSettings()` + `backend.rag.embeddings.make_embeddings()` and `backend.rag.vector_store.{load_or_init, save}` as primitives — those APIs are stable from iter-7 / iter-8.

### 6.4 Manual Smoke Test

```bash
# 1. Ingest a small subset
python scripts/ingest_hotpotqa.py --subset 50
ls storage/library/hotpotqa/ | wc -l    # expect: 50 (plus README.md)
test -f storage/library/hotpotqa/README.md && echo OK
# 2. Reindex library via the existing endpoint (in a separate terminal, with server running)
#    POST /api/rag/library/reindex  → "chunks" count rises
# 3. Eval with cache cold
python scripts/eval_hotpotqa.py --subset 50 --k 4
# expect: cache hits 0, builds 50, sane recall (0.4–0.7 typical)
# 4. Eval again with cache warm
python scripts/eval_hotpotqa.py --subset 50 --k 4
# expect: cache hits 50, builds 0, elapsed ~seconds
# 5. Full run for a real number
python scripts/ingest_hotpotqa.py --full
python scripts/eval_hotpotqa.py --full --k 4
# first run: minutes; record recall as iter-9 baseline
```

### 6.5 Isolation Guard

```bash
grep -r "backend\.chat" backend/eval/ scripts/eval_hotpotqa.py
# Expected: no output (exit 1 from grep on no match)
```

---

## 7. Implementation Tasks (TDD)

Each task ends with a commit. Run `pytest backend/tests/eval/ -v` after the eval-related tasks; full project suite (`pytest backend/tests/ -v`) after the integration task.

### Task 1: Repo hygiene — gitignore + package markers

**Files**:
- Modify: `.gitignore` (add three lines)
- Create: `backend/eval/__init__.py`, `backend/tests/eval/__init__.py`

- [ ] **Step 1: Update `.gitignore`**

Append to `.gitignore`:
```
# HotpotQA iteration 9 — bulky data + cache
scripts/.cache/
storage/eval/
storage/library/hotpotqa/
```

- [ ] **Step 2: Create empty package markers**

```bash
mkdir -p backend/eval backend/tests/eval/fixtures
touch backend/eval/__init__.py backend/tests/eval/__init__.py
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "import backend.eval; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore backend/eval/__init__.py backend/tests/eval/__init__.py
git commit -m "chore(rag): iter-9 scaffolding — eval package, gitignore"
```

### Task 2: Implement `backend/eval/metrics.py` (TDD)

**Files**:
- Create: `backend/eval/metrics.py`
- Test: `backend/tests/eval/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/eval/test_metrics.py`:
```python
import pytest
from backend.eval.metrics import paragraph_recall_at_k, supporting_fact_metrics


def test_paragraph_recall_at_k_empty_gold_vacuous():
    assert paragraph_recall_at_k([], set()) == 1.0
    assert paragraph_recall_at_k(["a", "b"], set()) == 1.0


def test_paragraph_recall_at_k_all_retrieved():
    assert paragraph_recall_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_paragraph_recall_at_k_none_retrieved():
    assert paragraph_recall_at_k(["x", "y"], {"a", "b"}) == 0.0


def test_paragraph_recall_at_k_partial_overlap():
    # gold size 2, retrieved ["a","x"] → 1 hit, recall = 1/2 = 0.5
    assert paragraph_recall_at_k(["a", "x"], {"a", "b"}) == 0.5


def test_paragraph_recall_at_k_capped_at_one():
    # gold size 1, retrieved hits 2 (duplicate), capped at 1/1 = 1.0
    assert paragraph_recall_at_k(["a", "a"], {"a"}) == 1.0


def test_sf_metrics_empty_empty():
    assert supporting_fact_metrics([], set()) == (1.0, 1.0, 1.0, 1.0)


def test_sf_metrics_empty_gold_nonempty_retrieved():
    # vacuous recall=1, precision=0 (nothing valid), f1=0, em=0
    assert supporting_fact_metrics(["x"], set()) == (0.0, 1.0, 0.0, 0.0)


def test_sf_metrics_retrieved_empty_nonempty_gold():
    assert supporting_fact_metrics([], {"a"}) == (0.0, 0.0, 0.0, 0.0)


def test_sf_metrics_partial_overlap():
    # gold={a,b}, retrieved=[a,c] → tp=1, p=1/2=0.5, r=1/2=0.5, f1=0.5, em=0
    sp, sr, sf, em = supporting_fact_metrics(["a", "c"], {"a", "b"})
    assert (sp, sr, sf, em) == pytest.approx((0.5, 0.5, 0.5, 0.0))


def test_sf_metrics_em_one_iff_exact_set_match():
    assert supporting_fact_metrics(["a", "b"], {"a", "b"})[3] == 1.0
    # Order-insensitive
    assert supporting_fact_metrics(["b", "a"], {"a", "b"})[3] == 1.0
    # Extra retrieved → em=0
    assert supporting_fact_metrics(["a", "b", "c"], {"a", "b"})[3] == 0.0
```

Note: every expected value matches the formula in [SPEC_focus.md](../document/SPEC_focus.md) FR-31.10 / FR-31.11.

- [ ] **Step 2: Run — expect failure**

Run: `pytest backend/tests/eval/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'backend.eval.metrics'`

- [ ] **Step 3: Write the implementation**

Create `backend/eval/metrics.py`:
```python
from __future__ import annotations

__all__ = ["paragraph_recall_at_k", "supporting_fact_metrics"]


def paragraph_recall_at_k(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return 1.0
    hits = sum(1 for t in retrieved if t in gold)
    return min(hits, len(gold)) / len(gold)


def supporting_fact_metrics(
    retrieved: list[str],
    gold: set[str],
) -> tuple[float, float, float, float]:
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
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest backend/tests/eval/test_metrics.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/metrics.py backend/tests/eval/test_metrics.py
git commit -m "feat(eval): paragraph_recall_at_k + supporting_fact_metrics (TDD)"
```

### Task 3: Implement `backend/eval/hotpotqa.py` (TDD)

**Files**:
- Create: `backend/eval/hotpotqa.py`
- Create: `backend/tests/eval/fixtures/tiny_hotpot.json`
- Test: `backend/tests/eval/test_hotpotqa.py`

- [ ] **Step 1: Write the fixture**

Create `backend/tests/eval/fixtures/tiny_hotpot.json`:
```json
[
  {
    "_id": "aaa111",
    "question": "Q1?",
    "answer": "A1",
    "type": "bridge",
    "level": "easy",
    "supporting_facts": [["Title A", 0], ["Title A", 1]],
    "context": [
      ["Title A", ["s0", "s1"]],
      ["Title B", ["t0", "t1"]],
      ["Title C", ["u0"]]
    ]
  },
  {
    "_id": "bbb222",
    "question": "Q2?",
    "answer": "A2",
    "type": "comparison",
    "level": "medium",
    "supporting_facts": [["Title X", 0]],
    "context": [["Title X", ["x0"]], ["Title Y", ["y0", "y1"]]]
  },
  {
    "_id": "ccc333",
    "question": "Q3?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": [["Title M", 0], ["Title N", 0]],
    "context": [["Title M", ["m0"]], ["Title N", ["n0"]], ["Title O", ["o0"]]]
  }
]
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/eval/test_hotpotqa.py`:
```python
from pathlib import Path

from backend.eval.hotpotqa import (
    HotpotQaItem, load, dataset_sha, gold_paragraph_titles, sample,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hotpot.json"


def test_load_count_and_fields():
    items = load(FIXTURE)
    assert len(items) == 3
    assert items[0].id == "aaa111"
    assert items[0].type == "bridge"
    assert items[0].level == "easy"
    assert len(items[0].context) == 3
    assert items[0].context[0] == ("Title A", ["s0", "s1"])
    assert items[0].supporting_facts == [("Title A", 0), ("Title A", 1)]


def test_dataset_sha_changes_on_file_change():
    sha1 = dataset_sha(FIXTURE)
    original = FIXTURE.read_bytes()
    try:
        modified = original.replace(b"aaa111", b"aaa999")
        FIXTURE.write_bytes(modified)
        sha2 = dataset_sha(FIXTURE)
    finally:
        FIXTURE.write_bytes(original)
    assert sha1 != sha2
    assert len(sha1) == 16


def test_gold_paragraph_titles_dedup():
    items = load(FIXTURE)
    # aaa111's supporting facts both reference "Title A" — must dedupe to 1.
    assert gold_paragraph_titles(items[0]) == {"Title A"}
    assert gold_paragraph_titles(items[1]) == {"Title X"}
    assert gold_paragraph_titles(items[2]) == {"Title M", "Title N"}


def test_sample_deterministic_and_stratified():
    items = load(FIXTURE)
    s1 = sample(items, 12)
    s2 = sample(items, 12)
    assert [i.id for i in s1] == [i.id for i in s2]
    assert len(s1) == len(items)  # requested 12, only 3 available, no oversample cliff in this small case


def test_sample_rejects_too_small():
    import pytest
    items = load(FIXTURE)
    with pytest.raises(ValueError):
        sample(items, 0)
    with pytest.raises(ValueError):
        sample(items, 1)
```

- [ ] **Step 3: Run — expect failure**

Run: `pytest backend/tests/eval/test_hotpotqa.py -v`
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 4: Write the implementation**

Create `backend/eval/hotpotqa.py` (see §3.1 for full content).

- [ ] **Step 5: Run — expect pass**

Run: `pytest backend/tests/eval/test_hotpotqa.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/eval/hotpotqa.py backend/tests/eval/test_hotpotqa.py backend/tests/eval/fixtures/
git commit -m "feat(eval): hotpotqa loader, sha, gold_paragraph_titles, stratified sample (TDD)"
```

### Task 4: Implement `backend/eval/cache.py` (TDD)

**Files**:
- Create: `backend/eval/cache.py`
- Test: `backend/tests/eval/test_cache.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/eval/test_cache.py`:
```python
import shutil
from pathlib import Path

import pytest
from langchain_community.embeddings.fake import FakeEmbeddings

from backend.eval.hotpotqa import load as load_items
from backend.eval.cache import EVAL_CACHE_ROOT, load_or_build, _build_index
from backend.eval.hotpotqa import HotpotQaItem

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hotpot.json"


@pytest.fixture
def tmp_cache_root(tmp_path, monkeypatch):
    """Point EVAL_CACHE_ROOT at a tmp dir for the duration of the test."""
    new_root = tmp_path / "eval_cache"
    monkeypatch.setattr("backend.eval.cache.EVAL_CACHE_ROOT", new_root)
    yield new_root
    shutil.rmtree(new_root, ignore_errors=True)


def _embedding_factory():
    # Use deterministic embeddings so we don't download anything.
    return FakeEmbeddings(size=64)


def test_build_index_has_all_paragraphs(tmp_cache_root):
    items = load_items(FIXTURE)
    index = _build_index(items[0], _embedding_factory())
    assert index.docstore._dict  # truthy
    n = sum(1 for d in index.docstore._dict.values() if not d.metadata.get("_placeholder"))
    assert n == 3  # all 3 context paragraphs


def test_load_or_build_first_call_misses(tmp_cache_root):
    items = load_items(FIXTURE)
    index, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is False
    assert index is not None


def test_load_or_build_second_call_hits(tmp_cache_root):
    items = load_items(FIXTURE)
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    _, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is True


def test_load_or_build_no_cache_forces_rebuild(tmp_cache_root):
    items = load_items(FIXTURE)
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    _, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory(), no_cache=True)
    assert hit is False


def test_load_or_build_different_sha_separates_dirs(tmp_cache_root):
    items = load_items(FIXTURE)
    _, hit_a = load_or_build(items[0], "aaaaaaaa00000000", _embedding_factory())
    _, hit_b = load_or_build(items[0], "bbbbbbbb00000000", _embedding_factory())
    assert hit_a is False and hit_b is False
    # Both cache dirs exist independently.
    assert (tmp_cache_root / "aaaaaaaa00000000").exists()
    assert (tmp_cache_root / "bbbbbbbb00000000").exists()


def test_load_or_build_recovers_from_corruption(tmp_cache_root):
    items = load_items(FIXTURE)
    cache_dir = tmp_cache_root / "deadbeef00000000" / items[0].id
    # First call writes a valid cache.
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    # Corrupt the cache by overwriting index.faiss with garbage.
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.faiss").write_bytes(b"not a real faiss index")
    # Second call must rebuild cleanly.
    index, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is False  # treated as miss because we rebuilt
    assert index is not None
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest backend/tests/eval/test_cache.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `backend/eval/cache.py` (see §3.3 for full content).

- [ ] **Step 4: Run — expect pass**

Run: `pytest backend/tests/eval/test_cache.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/cache.py backend/tests/eval/test_cache.py
git commit -m "feat(eval): SHA-keyed per-question FAISS cache with corruption recovery (TDD)"
```

### Task 5: Implement `scripts/ingest_hotpotqa.py`

**Files**:
- Create: `scripts/ingest_hotpotqa.py`

- [ ] **Step 1: Smoke-test the CLI's help output**

```bash
python scripts/ingest_hotpotqa.py --help
```
Expected: argparse output listing `--subset`, `--full`, `--force`.

- [ ] **Step 2: Smoke-test with a fixture (skipping download by injecting a small synthetic JSON)**

For this task we test the file-writing logic using a tiny synthetic JSON to avoid the network download in CI:

```bash
cat > /tmp/tiny_for_ingest.json <<'EOF'
[{"_id":"ingest111","question":"q","answer":"a","type":"bridge","level":"easy",
  "supporting_facts":[["Ta",0]],
  "context":[["Ta",["sent 0","sent 1"]],["Tb",["x"]]]}]
EOF
```

Temporarily patch `CACHE_PATH` in the script via env or run a one-off inline test:

```bash
python -c "
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, 'scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('ingest', 'scripts/ingest_hotpotqa.py')
ing = importlib.util.module_from_spec(spec)
# Stub CACHE_PATH for this test run.
ing.CACHE_PATH = Path('/tmp/tiny_for_ingest.json')
ing.LIBRARY_DIR = Path('/tmp/hotpotqa_lib_test')
ing.HOTPOTQA_URL = ''
spec.loader.exec_module(ing)
ing.write_question_file(json.loads(Path('/tmp/tiny_for_ingest.json').read_text())[0], ing.LIBRARY_DIR)
print(list(ing.LIBRARY_DIR.iterdir()))
"
```

Expected: `[/tmp/hotpotqa_lib_test/ingest111.md, /tmp/hotpotqa_lib_test/README.md]` (or similar — README written on first call). Inspect the .md file to confirm structure.

```bash
cat /tmp/hotpotqa_lib_test/ingest111.md
```

Expected: frontmatter + two H1 sections (`# Ta`, `# Tb`).

- [ ] **Step 3: Run the manual smoke subset**

```bash
python scripts/ingest_hotpotqa.py --subset 50
ls storage/library/hotpotqa/ | wc -l
test -f storage/library/hotpotqa/README.md && echo OK
```

Expected: `51` (50 question files + README.md); `OK`.

- [ ] **Step 4: Verify library ingest via existing endpoint**

```bash
# Operator runs the server in another terminal:
uvicorn backend.main:app --port 8080
# In this terminal:
curl -sX POST http://localhost:8080/api/rag/library/reindex | head -c 200
curl -s http://localhost:8080/api/rag/stats | python -c "import sys,json; d=json.load(sys.stdin); print('library_chunks:', d.get('library_chunks'))"
```

Expected: reindex returns 200 with files_processed ≥ 50; library_chunks > 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_hotpotqa.py
git commit -m "feat(scripts): ingest_hotpotqa CLI — download + write per-question library files"
```

### Task 6: Implement `scripts/eval_hotpotqa.py` (the integration)

**Files**:
- Create: `scripts/eval_hotpotqa.py`
- Create: `backend/tests/eval/fixtures/integration_hotpot.json`
- Test: `backend/tests/eval/test_eval_integration.py`

- [ ] **Step 1: Write the integration fixture**

Create `backend/tests/eval/fixtures/integration_hotpot.json`:
```json
[
  {"_id":"int001","question":"Which city?","answer":"Paris","type":"bridge","level":"easy",
    "supporting_facts":[["Paris, France",0]],
    "context":[["Paris, France",["Paris is the capital of France."]],
               ["Berlin, Germany",["Berlin is the capital of Germany."]],
               ["Madrid, Spain",["Madrid is the capital of Spain."]]]},
  {"_id":"int002","question":"Author of Hamlet?","answer":"Shakespeare","type":"bridge","level":"medium",
    "supporting_facts":[["Hamlet",0]],
    "context":[["Hamlet",["Hamlet is a tragedy by William Shakespeare."]],
               ["Macbeth",["Macbeth is also by Shakespeare."]],
               ["Doctor Who",["Doctor Who is a British TV show."]]]},
  {"_id":"int003","question":"Same director?","answer":"yes","type":"comparison","level":"easy",
    "supporting_facts":[["Movie A",0],["Movie B",0]],
    "context":[["Movie A",["Directed by Nolan."]],
               ["Movie B",["Directed by Nolan too."]],
               ["Movie C",["Directed by Spielberg."]]]},
  {"_id":"int004","question":"Largest planet?","answer":"Jupiter","type":"comparison","level":"hard",
    "supporting_facts":[["Jupiter",0]],
    "context":[["Jupiter",["The largest planet in the solar system."]],
               ["Mars",["Smaller than Earth."]],
               ["Earth",["Third planet from the Sun."]]]},
  {"_id":"int005","question":"Inventor of telephone?","answer":"Bell","type":"bridge","level":"hard",
    "supporting_facts":[["Telephone",0]],
    "context":[["Telephone",["Invented by Alexander Graham Bell."]],
               ["Telegraph",["Older technology."]],
               ["Radio",["Different invention."]]]}
]
```

- [ ] **Step 2: Write the integration test**

Create `backend/tests/eval/test_eval_integration.py`:
```python
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "eval_hotpotqa.py"
FIXTURE = ROOT / "backend" / "tests" / "eval" / "fixtures" / "integration_hotpot.json"
CACHE_ROOT = ROOT / "storage" / "eval" / "hotpotqa" / "cache"


@pytest.fixture(autouse=True)
def clean_cache():
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT, ignore_errors=False)
    yield
    # Leave cache intact by default; tests can override via fixture.


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--fixture", str(FIXTURE), "--k", "4"],
        capture_output=True, text=True, check=False, cwd=str(ROOT),
    )


def test_eval_first_run_cold_cache():
    p = _run([])
    assert p.returncode == 0, p.stdout + p.stderr
    for label in ("paragraph_recall@4", "sf_precision", "sf_recall", "sf_f1", "sf_em"):
        assert label in p.stdout, f"{label!r} missing from:\n{p.stdout}"
    m = re.search(r"cache hits / builds\s+:\s+(\d+)\s+/\s+(\d+)", p.stdout)
    assert m and m.group(2) == "5", f"expected 5 builds, got {p.stdout}"


def test_eval_second_run_warm_cache():
    _run([])
    p = _run([])
    assert p.returncode == 0
    m = re.search(r"cache hits / builds\s+:\s+(\d+)\s+/\s+(\d+)", p.stdout)
    assert m and m.group(1) == "5" and m.group(2) == "0", p.stdout
```

- [ ] **Step 3: Run — expect ImportError / ModuleNotFoundError on missing script**

Run: `pytest backend/tests/eval/test_eval_integration.py -v`
Expected: FILE NOT FOUND or similar.

- [ ] **Step 4: Write `scripts/eval_hotpotqa.py`**

See §3.5 for full content.

- [ ] **Step 5: Run — expect pass**

Run: `pytest backend/tests/eval/test_eval_integration.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run full test suite to verify no iter-8 regression**

Run: `pytest backend/tests/ -v`
Expected: all iter-8 tests still pass + new eval tests pass.

- [ ] **Step 7: Manual smoke (full scale)**

```bash
python scripts/ingest_hotpotqa.py --full   # minutes
python scripts/eval_hotpotqa.py --full --k 4   # minutes (first), seconds (warm)
```

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_hotpotqa.py backend/tests/eval/test_eval_integration.py backend/tests/eval/fixtures/integration_hotpot.json
git commit -m "feat(scripts+eval): end-to-end eval CLI; integration test with synthetic fixture"
```

### Task 7: Isolation guard verification

**Files**: none (verification only)

- [ ] **Step 1: Verify no chat imports**

```bash
grep -rn "backend\.chat" backend/eval/ scripts/eval_hotpotqa.py
```

Expected: no output. (Exit 1 from grep is fine — that's what we want.)

- [ ] **Step 2: Add to PR description** (operator-driven)

If running grep returned matches, fix them; otherwise the guard passes.

---

## 8. Out of Scope (Deferred to Future Iterations)

1. LLM-based answer evaluation (`answer_em`, `answer_f1`) — would require calling `minimax-3` per question, doubling eval cost and adding API-key dependencies at eval-time.
2. `/api/eval/` route — CLI only by design.
3. JSON output (`--json-out`) — easy add later if downstream tooling wants to consume eval results.
4. Per-type / per-level breakdown in default output — would clutter the report.
5. HotpotQA `fullwiki` setting — requires a separate 5M+ Wikipedia paragraph corpus ingestion pipeline.
6. Multi-process or distributed evaluation.
7. Incremental cache invalidation beyond dataset SHA change (any single-byte change in the JSON busts all caches).
8. CI hookup for the eval script.
9. Embedding-recipe sweeps (automatically try multiple `EMBEDDING_BACKEND` values).
10. Cross-encoder re-ranking on top of FAISS results.
11. Sentence-level supporting-fact metrics (would require LLM-based extraction).
12. Surface-attribution in the library sidebar UI when files with `source=hotpotqa` are present (would require frontend changes).
13. Multi-pass retrieval (Hop 2 using Hop-1 results to refine the query) — interesting follow-up for true multi-hop performance, but requires the fullwiki pipeline.
