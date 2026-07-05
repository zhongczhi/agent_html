# HotpotQA RAG Integration — Iteration 9 Design Spec

**Date**: 2026-07-05
**Status**: Draft, pending user review
**Iteration goal**: Extend the iter-8 RAG module in two ways: (1) load the HotpotQA dev-distractor dataset into `storage/library/` as live, queryable library data; (2) add a CLI-only evaluation pipeline (no HTTP routes, no chat coupling) that measures paragraph retrieval recall and paragraph-level supporting-fact precision/recall/F1/EM against the same library conventions.

This iteration does **not** modify any chat code. The two sub-features share data: the eval pipeline reads the same HotpotQA JSON that ingest writes library files from, but it builds its own per-question indices — it does not read the library directory at all. Sharing happens at the dataset, not at the index.

---

## 1. Goals & Non-Goals

### Goals

1. **Library ingestion of HotpotQA**: a CLI script downloads `hotpot_dev_distractor_v1.json` and writes one `.md` file per question into `storage/library/hotpotqa/`. Each file uses an H1-per-paragraph structure so the existing `MarkdownTextSplitter` produces paragraph-granular chunks automatically during library reindex.
2. **Subsetting**: a `--subset N` flag produces a stratified-by-`(type, level)` deterministic sample (seed=42). Default = full dev distractor (7,405 Q).
3. **Eval pipeline as a separate CLI**: `scripts/eval_hotpotqa.py` runs the HotpotQA dev set through the project's sentence-transformers + FAISS retrieval, builds a transient per-question index, retrieves top-k, and reports retrieval metrics. No LLM calls. No HTTP routes. No coupling to `backend/chat/`.
4. **Per-question index cache**: SHA-keyed by `(dataset_sha, qid)`, persisted under `storage/eval/hotpotqa/cache/`. First run builds all; subsequent runs are seconds. The `--no-cache` flag forces rebuild.
5. **Retrieval-only metrics**: `paragraph_recall@k`, `sf_precision`, `sf_recall`, `sf_f1`, `sf_em`. Aggregate over the subset, terminal output only (no JSON, no per-type breakdown).

### Non-Goals (v1)

- No LLM-based answer evaluation (no answer EM/F1, no supporting-fact-extraction prompting).
- No `/api/eval/` route — CLI only, keeping eval out of chat business.
- No JSON output, no `--json-out` flag.
- No per-type / per-level breakdown in the default output.
- No incremental cache invalidation beyond dataset SHA change.
- No parallel/distributed evaluation.
- No CI hookup; eval is operator-run.
- No editing of `backend/chat/*`, `backend/main.py`, or any frontend file.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Scripts (CLI, NO chat dependencies)                   │
│                                                                         │
│   scripts/ingest_hotpotqa.py          scripts/eval_hotpotqa.py          │
│      │                                    │                            │
│      │ download                            │ load same JSON             │
│      ▼                                     ▼                            │
│   hotpot_dev_distractor_v1.json (cached in scripts/.cache/)             │
│      │                                     │                            │
└──────┼─────────────────────────────────────┼────────────────────────────┘
       │                                     │
       ▼                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│   storage/library/hotpotqa/      │   │   backend/eval/   (NEW)          │
│     <qid>.md                    │   │     ├─ hotpotqa.py               │
│       H1 paragraph 1 text       │   │     ├─ metrics.py                │
│       H1 paragraph 2 text       │   │     └─ cache.py                  │
│       ...                       │   │                                  │
└─────────────┬───────────────────┘   └──────────────────┬───────────────┘
              │ read by library loader                   │
              ▼                                          ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│  backend/rag/ (UNCHANGED)        │   │  for each question Q:            │
│    rag service reindex           │   │    docs = paragraphs(Q.context) │
│    rag retriever (chat-time)     │   │    FAISS(docs).save() → cache/   │
│                                  │   │    embed Q.question              │
│  ⚠ library may contain           │   │    top_k = retrieve(k)           │
│  cross-question paragraphs       │   │    metrics(top_k, Q.gold)        │
│  (expected, library UX is fine)  │   │                                  │
└─────────────────────────────────┘   └──────────────────┬───────────────┘
                                                         │
                                                         ▼
                                            Terminal: aggregate metrics
```

### Isolation guarantees

- `scripts/eval_hotpotqa.py` imports **nothing** from `backend/chat/`.
- Eval imports only `backend.rag.embeddings`, `backend.rag.vector_store`, and `backend.eval.*`. These are pure primitives with no chat coupling.
- Eval builds per-question FAISS indices on the fly (one cache entry per question). It never touches the global library index.
- Library ingestion writes only paragraph text + structural metadata. Gold answers and gold supporting facts live in the JSON only — they never appear in any library file.

---

## 3. Component Details

### 3.1 Dataset: `hotpot_dev_distractor_v1.json`

Source: `https://hotpotqa.github.io/` (CC BY-SA 4.0). 7,405 questions, ~74k paragraphs across them. Each item:

```json
{
  "_id": "5a8a7a9a55429937a7665480",
  "question": "What film directed by ...",
  "answer": "Some Movie Title",
  "supporting_facts": [["Movie Title", 3], ["Some Movie", 0]],
  "context": [["Movie Title", ["sent 0", "sent 1", ...]], ...],
  "type": "bridge",
  "level": "medium"
}
```

### 3.2 Library ingest (`scripts/ingest_hotpotqa.py`)

**Output layout** (one file per question):
```
storage/library/hotpotqa/<qid>.md
```

**File body** (paragraphs as H1 sections, joined text follows):
```markdown
---
question_id: 5a8a7a9a55429937a7665480
question_type: bridge
question_level: medium
source: hotpotqa
---

# Paragraph Title 1
sentence 0 sentence 1 sentence 2 ...

# Paragraph Title 2
sentence 0 sentence 1 ...
```

The existing `MarkdownTextSplitter` in `backend/rag/splitter.py:23` splits at H1 boundaries, attaching each chunk's metadata via `_md_header_path` (the breadcrumb path at the chunk's offset). For our per-question files the breadcrumb is exactly the paragraph title. Each H1 section therefore becomes one chunk during library reindex.

**Frontmatter fields:**
- `question_id` — hotpot `_id` (also embedded in filename for sanity check)
- `question_type` — `bridge` or `comparison`
- `question_level` — `easy`, `medium`, or `hard`
- `source` — literal `"hotpotqa"`, so users browsing the library tab can identify dataset origins

**What is NOT in the file**: the question text, the gold answer, the gold `supporting_facts`. Including any of these would leak contamination into the retrieval corpus. The eval pipeline reads those fields directly from the JSON for its own use.

**Filename slugification**: replace any non-`[a-zA-Z0-9_-]` in the `_id` with `_`. HotpotQA `_id`s are typically hex hashes that already conform.

**CLI**:
```bash
python scripts/ingest_hotpotqa.py [--subset N] [--full] [--force]
# default: --full (7,405 Q)
# --subset N : stratified sample; --full is the default (mutually exclusive)
# --force    : re-download even if cached
```

`argparse` makes `--subset N` and `--full` mutually exclusive (the latter is the default). If `--subset N > 7405`, sample is bounded to total. `--subset 0` and `--subset 1` are noops and exit with a usage error rather than producing an empty/garbage sample.

**Stratified sampling** (`backend/eval/hotpotqa.py::sample`):
1. Bucket all 7,405 questions by `(type, level)`.
2. For each bucket: sample `min(ceil(N / 6), len(bucket))` deterministically using `random.Random(42)`. Bucket cap ensures a small bucket isn't over-sampled.
3. Concatenate; shuffle with same seed (stable order across runs).
4. Returns `list[HotpotQaItem]`, length ≤ min(N, total questions).

**Download cache**: file is fetched once and stashed at `scripts/.cache/hotpot_dev_distractor_v1.json`. SHA-256 of the file is the `dataset_sha` cache key.

**Idempotence**: a successful first run produces a set of well-formed files. Re-running the script on an unchanged dataset is a no-op — already-present files are byte-identical to what would be written, so on-disk content is unchanged and directory mtimes don't churn. A second run that starts after a partial first run simply fills in the missing files; per-file atomicity means a half-written file from a crashed prior run is replaced in full on the next attempt.

### 3.3 Eval pipeline (`scripts/eval_hotpotqa.py`)

**Per-item accessor** (used in §3.5 metric calls):

```python
def gold_paragraph_titles(item: HotpotQaItem) -> set[str]:
    """Returns the set of paragraph titles that appear in the question's
    gold supporting_facts. Each (title, sent_idx) entry contributes its title;
    duplicates collapse."""
    return {title for title, _ in item.supporting_facts}
```

**Top-level flow:**

```python
def main():
    args = parse_args()                       # mutually-exclusive --subset N | --full, --k, --no-cache
    items = hotpotqa.load(dataset_path)       # all questions
    if args.subset is not None:
        items = hotpotqa.sample(items, args.subset)
    embeddings = get_embeddings()             # reuse backend.rag.embeddings factory
    per_q = []                                # list of (paragraph_recall, sf_p, sf_r, sf_f1, sf_em)
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()
    for item in items:
        try:
            index, hit = cache.load_or_build(item, dataset_sha, embeddings, no_cache=args.no_cache)
            docs = retrieve(index, item.question, k=args.k)
            retrieved_titles = [d.metadata.get("title", "") for d in docs]
            gold_titles = gold_paragraph_titles(item)
            pr = paragraph_recall_at_k(retrieved_titles, gold_titles)
            sp, sr, sf, em = supporting_fact_metrics(retrieved_titles, gold_titles)
            per_q.append((pr, sp, sr, sf, em))
            if hit: cache_hits += 1
            else:   cache_builds += 1
        except Exception as e:
            log.warning("qid=%s error: %s", item.id, e)
            errors += 1
    elapsed = time.monotonic() - t0
    print_report(per_q, len(items), cache_hits, cache_builds, errors, elapsed)
    sys.exit(0)   # partial per-question errors are logged, not fatal; only setup failures exit 1
```

**Per-question index construction (custom; does NOT use MarkdownTextSplitter):**

```python
def _build_index(item, embeddings):
    docs = []
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
```

The eval intentionally builds paragraph-granular Documents (not split via `MarkdownTextSplitter`). This decouples the eval's retrieval granularity from the chat pipeline's chunking decisions — important because the chat's `MarkdownTextSplitter` may or may not split on each H1 depending on `chunk_size` configuration, and we want a stable, predictable metric.

### 3.4 Per-question cache (`backend/eval/cache.py`)

**Cache directory layout** (one subdirectory per question, dataset SHA as a
top-level invalidation prefix):

```
storage/eval/hotpotqa/cache/{dataset_sha[:16]}/{item.id}/
  index.faiss          (FAISS native)
  index.pkl            (FAISS docstore)
```

**`cache.load_or_build(item, dataset_sha, embeddings, no_cache)`** algorithm:
```
cache_path = EVAL_CACHE_ROOT / dataset_sha[:16] / item.id
if no_cache or not cache_path.exists():
    index = _build_index(item, embeddings)
    save(index, cache_path)
    return index, hit=False
try:
    index = load_or_init(cache_path, embeddings)  # reuses backend.rag.vector_store
    return index, hit=True
except Exception as e:
    log.warning("cache corrupt for %s, rebuilding: %s", item.id, e)
    shutil.rmtree(cache_path, ignore_errors=True)
    index = _build_index(item, embeddings)
    save(index, cache_path)
    return index, hit=False
```

**Invalidation**: the `dataset_sha` directory prefix means any change to the JSON file (re-download, schema update) busts all caches atomically. There's no need for per-file checksums beyond that.

### 3.5 Metrics (`backend/eval/metrics.py`)

All metrics operate at paragraph-title level (not sentence-level), since we don't run an LLM.

```python
def paragraph_recall_at_k(
    retrieved: list[str],   # paragraph titles of top-k retrieved chunks
    gold: set[str],         # gold paragraph titles from supporting_facts
) -> float:
    """Fraction of gold paragraphs appearing in the top-k retrieved list.
    Capped at 1.0 (more hits than gold is possible when k > len(gold)).
    Returns 1.0 vacuously when gold is empty."""
    if not gold:
        return 1.0
    hits = sum(1 for t in retrieved if t in gold)
    return min(hits, len(gold)) / len(gold)


def supporting_fact_metrics(
    retrieved: list[str],
    gold: set[str],
) -> tuple[float, float, float, float]:
    """Returns (precision, recall, f1, em)."""
    if not gold and not retrieved:
        return (1.0, 1.0, 1.0, 1.0)
    if not retrieved:
        return (0.0, 0.0, 0.0, 0.0)
    if not gold:
        return (0.0, 1.0, 0.0, 0.0)   # retrieved things nobody asked for
    retrieved_set = set(retrieved)
    tp = len(retrieved_set & gold)
    precision = tp / len(retrieved_set)
    recall = tp / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    em = 1.0 if retrieved_set == gold else 0.0
    return (precision, recall, f1, em)
```

### 3.6 Output format (terminal only)

```
HotpotQA Eval — subset=full, k=4, dataset_sha=a1b2c3d4
  paragraph_recall@4   : 0.642
  sf_precision         : 0.354
  sf_recall            : 0.642
  sf_f1                : 0.453
  sf_em                : 0.085
  questions successfully evaluated : 7405  (out of 7405 attempted)
  cache hits / builds  : 7380 / 25
  errors               : 0
  elapsed              : 47s
```

`questions successfully evaluated` is `len(per_q)`, i.e. `len(items) - errors`. `attempts` is `len(items)` (= `subset` count after sampling).

If errors > 0, a final block is printed:
```
WARN: 3 questions errored (skipped, not counted in metrics above):
  - qid=...: <reason>
  ...
```

### 3.7 CLI surface

```bash
python scripts/eval_hotpotqa.py [--subset N | --full] [--k 4] [--no-cache] [--fixture PATH]
# default: --full
# --k: top-k to retrieve (default 4, matching FR-25.6)
# --no-cache: ignore and rebuild all per-question indices
# --fixture PATH: read the dataset from PATH instead of scripts/.cache/hotpot_dev_distractor_v1.json.
#                 Used by the integration test; not normally needed by an operator.
```

Exit codes:
- `0`: ran to completion (errors in individual questions are non-fatal)
- `1`: setup failure (dataset missing, embedding model load failed, JSON parse error)

Source-of-truth priority for the dataset path:
1. `--fixture PATH` (explicit, used by tests)
2. `scripts/.cache/hotpot_dev_distractor_v1.json` (auto-downloaded by `ingest_hotpotqa.py` or by running the eval once with the network reachable)
3. Fail fast with download instructions.

---

## 4. Configuration

No new env vars, no new config fields. Reuses:
- `EMBEDDING_BACKEND` (already in `backend/rag/config.py`)
- sentence-transformers model name (already configured)
- FAISS machinery (already wrapped in `backend/rag/vector_store.py`)

### Filesystem locations

```
storage/library/hotpotqa/                ← ingest output (existing storage path)
storage/eval/hotpotqa/cache/             ← NEW, eval cache root
scripts/.cache/hotpot_dev_distractor_v1.json  ← NEW, downloaded JSON stash
```

`scripts/.cache/` and `storage/eval/` should be gitignored (existing `.gitignore` already ignores `storage/` plus `__pycache__` — add `storage/eval/...` if not implicit, and `scripts/.cache/`).

### requirements.txt

No new deps. sentence-transformers, FAISS, langchain_core are all already present.

### CC BY-SA 4.0 attribution

The HotpotQA dataset is CC BY-SA 4.0. We ship attribution in two places:

1. `storage/library/hotpotqa/README.md` — written by the ingest script if missing. Contains the dataset name, source URL (`https://hotpotqa.github.io/`), license text (or a pointer to `LICENSE-hotpotqa`), and a one-line "shipped with respect to CC BY-SA 4.0" note.
2. The eval script's terminal header — prints `Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)` once before the metric block so anyone running the script sees the provenance.

Surface-attribution in the library tab UI when files with `source=hotpotqa` are present is **deferred** (would require a frontend change, out of scope per §1).

---

## 5. Error Handling

| Stage | Failure | Behavior |
|-------|---------|----------|
| Ingest: download | Network error | Retry once after 5s. After 2 failures, exit 1 with download URL. |
| Ingest: write | Permission error / disk full | Log path, continue with remaining. Exit non-zero if any file failed. |
| Ingest: corrupt JSON (whole file) | `json.JSONDecodeError` | Exit 1 with the file path and a clear "download again or fix manually" hint. Cannot proceed without a parsable dataset. |
| Ingest: per-question schema error | Missing `_id`, `context`, etc. | Log WARNING with qid, skip that question, continue. Final summary lists skipped IDs. |
| Eval: dataset missing | JSON not at cache path | Print expected path + download instructions, exit 1. |
| Eval: dataset corrupt (whole file) | `json.JSONDecodeError` | Print the exception tail, exit 1. Cannot proceed. |
| Eval: embedding model load | sentence-transformers not installed | Fail fast with `pip install -r requirements.txt` hint. Exit 1. |
| Eval: per-question cache corrupted | `index.faiss` read fails | Delete cache dir for that qid, rebuild from scratch, log WARNING. Continue with rest. |
| Eval: per-question retrieval | Embedding call raises (transient) | Log WARNING, count as errored, skip rest of run unaffected. |
| Eval: missing title in retrieved chunk | Defensive case (should never happen) | Treat chunk's title as empty string → no-match for recall/f1. Question still counted. |
| Eval: per-question FAISS construction | Library embeddings reject a chunk | Skip that question, log WARNING. |

**Logging**: standard `logging` module, INFO default. Per-question errors at WARNING. Setup errors at ERROR (with non-zero exit).

**Atomicity**:
- Ingest: per-file write; partial directories are tolerated on restart (re-running cleans them up).
- Eval: per-question state; cache build for one question is atomic (`shutil.rmtree` + `save`), but a partial-corrupt cache is recovered automatically on the next run.

---

## 6. Testing

### Unit tests (no I/O, fast — CI-friendly)

**`backend/tests/eval/test_metrics.py`** (NEW)
- `paragraph_recall_at_k`: empty gold → 1.0 (vacuous); all retrieved → 1.0; none retrieved → 0.0; partial overlap → correct fraction.
- `supporting_fact_metrics`:
  - perfect match → (1, 1, 1, 1)
  - empty gold + empty retrieved → (1, 1, 1, 1)
  - empty gold + non-empty retrieved → (0, 1, 0, 0)
  - retrieved = empty ∩ gold ≠ empty → (0, 0, 0, 0)
  - partial overlap → exact precision/recall/f1 formula values
  - em is 1.0 if and only if sets equal (regardless of size)

**`backend/tests/eval/test_hotpotqa.py`** (NEW)
- `load()` with fixture JSON containing 3 questions → expected `HotpotQaItem` count + fields.
- `dataset_sha()`: deterministic, changes when file changes.
- `gold_paragraph_titles(item)` correctly extracts `set(title for title, _ in item.supporting_facts)`, deduplicates when one title appears at multiple sentence indices.
- `sample(items, n)`: deterministic across runs (seed=42), stratified, returns ≤ n items.

**`backend/tests/eval/test_cache.py`** (NEW)
- `cache.load_or_build`: first call builds, returns hit=False; second call returns hit=True.
- Different `dataset_sha` → different cache dir, no cross-pollination.
- `--no-cache` flag forces rebuild on every call.
- Corrupted cache (write garbage to `index.faiss`) → next call rebuilds cleanly (asserts no exception raised).
- Empty paragraphs in `context` are skipped (don't create empty Documents).
- Uses real `FakeEmbeddings` from langchain (deterministic vectors) so no model download required.

**`backend/tests/eval/test_eval_integration.py`** (NEW, slower)
- Uses a synthetic 5-question JSON fixture shipped at `backend/tests/eval/fixtures/integration_hotpot.json` (hermetic — no network, no real dataset download). `FakeEmbeddings` from `langchain_community`.
- Invokes `python scripts/eval_hotpotqa.py --fixture backend/tests/eval/fixtures/integration_hotpot.json --k 4` via `subprocess.run`.
- Asserts: exit code 0, output contains all 5 metric lines, cache dir created at the expected SHA-derived path.
- Runs the eval twice in succession; second run must report `cache hits / builds : 5 / 0`.

### Manual smoke test (operator runs)

```bash
# 1. Ingest a small subset
python scripts/ingest_hotpotqa.py --subset 50
ls storage/library/hotpotqa/ | wc -l    # expect: 50

# 2. Reindex library via existing endpoint
# (operator starts server, hits POST /api/rag/library/reindex via curl or UI)
# Check stats: should now include the 50 new files

# 3. Eval with cache cold
python scripts/eval_hotpotqa.py --subset 50 --k 4
# expect: cache_hits=0, builds=50, sane recall (0.4–0.7 typical for sentence-transformers)

# 4. Eval again with cache warm
python scripts/eval_hotpotqa.py --subset 50 --k 4
# expect: cache_hits=50, builds=0, elapsed ~seconds

# 5. Full run for a real number
python scripts/ingest_hotpotqa.py --full
python scripts/eval_hotpotqa.py --full --k 4
# first run: minutes; record recall number as the iter-9 baseline
```

---

## 7. Module Layout

### New files

```
scripts/
├── ingest_hotpotqa.py              # CLI: downloads + writes library files
└── eval_hotpotqa.py                # CLI: runs eval pipeline

backend/eval/
├── __init__.py
├── hotpotqa.py                     # load(), sample(), dataset_sha(), HotpotQaItem
├── metrics.py                      # paragraph_recall_at_k, supporting_fact_metrics
└── cache.py                        # load_or_build(), per-question FAISS cache

backend/tests/eval/
├── __init__.py
├── fixtures/
│   └── tiny_hotpot.json            # 3-question fixture
├── test_metrics.py
├── test_hotpotqa.py
├── test_cache.py
└── test_eval_integration.py
```

### Modified files

| File | Change |
|------|--------|
| `.gitignore` | Existing rule ignores only `backend/storage/rag/` and `storage/conversations.json`. Add `scripts/.cache/`, `storage/library/hotpotqa/`, and `storage/eval/` so neither the downloaded JSON nor the eval cache (which can be hundreds of MB on full set) nor the bulky library files (tens of MB) pollute commits. |

### Unchanged files

- `backend/chat/*` — no chat coupling.
- `backend/main.py` — no startup wiring changes.
- `backend/rag/*` — eval imports existing modules as primitives, no API change.
- `frontend/*` — no UI changes.

---

## 8. Migration & Rollback

- **No migrations**: no DB, no schema, no env vars to add.
- **Rollback**: delete `scripts/ingest_hotpotqa.py`, `scripts/eval_hotpotqa.py`, `backend/eval/`, `backend/tests/eval/`, the `.gitignore` line, and `storage/library/hotpotqa/`. Nothing else.
- **Existing indexes**: `storage/rag/library_index.*` is unaffected. Running the existing library reindex after HotpotQA ingest simply adds the new files to the index.
- **Dataset provenance**: ingest is reproducible — re-running on the same HotpotQA release yields identical files. The `dataset_sha` cache key makes that explicit.

---

## 9. Out of Scope (deferred)

- LLM-based answer evaluation (`answer_em`, `answer_f1`) — would require calling minimax-3 per question, doubling eval cost. Easy to add as a separate script later.
- Per-type / per-level breakdown (`--breakdown`) — would clutter default output.
- JSON output (`--json-out`) — deferred; easy add.
- `/api/eval/` route — CLI only by design.
- Incremental cache invalidation beyond dataset SHA change.
- CI hookup for the eval command.
- Multi-process eval / parallel question processing.
- HotpotQA `fullwiki` setting (would require the 5M+ paragraph Wikipedia corpus as a separate ingestion pipeline).
- Embedding-recipe sweeps (try multiple `EMBEDDING_BACKEND` values automatically).
- Cross-encoder re-ranking on top of FAISS results.
- Attribution UI in the library tab for files with `source=hotpotqa`.

---

## 10. Open Questions

None. The four pre-design decisions in the brainstorm round (evaluation-scope clarification, dataset scope with subset flag, retrieval-only metrics, two-script + cache shape) were all resolved during brainstorming.
