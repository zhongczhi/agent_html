# Chatbot Project — Iteration 9 Spec (HotpotQA Library Data + Retrieval Eval Pipeline)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Covers the HotpotQA library ingest CLI and the separate, retrieval-only evaluation pipeline.

## Overview

Iteration 8 shipped the multi-format document loader pipeline, the library management API + sidebar tab, and the show-sources toggle — closing the major gaps in the iter-7 RAG module. The iter-8 ingest path now correctly handles `.md` / `.txt` / `.pdf` / `.html` / `.docx` / `.csv`, the library can be populated from inside the UI, and RAG panel sources can be hidden without affecting retrieval.

Iteration 9 extends the library concept with a real dataset and adds a CLI-only evaluation pipeline so we can measure how well retrieval actually works. There are two sub-features:

1. **HotpotQA library data**: a CLI script downloads `hotpot_dev_distractor_v1.json` and writes one `.md` file per question into `storage/library/hotpotqa/`. Each file uses an H1-per-paragraph structure so the existing `MarkdownTextSplitter` (introduced in iter-8) chunks them one-per-paragraph automatically during the standard library reindex. The result: the chat-time library contains ~7,405 questions × 10 paragraphs of Wikipedia-grounded content ready for retrieval.

2. **Separation of concerns for evaluation**: a second CLI script `scripts/eval_hotpotqa.py` runs the same dev set through the project's sentence-transformers + FAISS retrieval, but **does NOT** read the library directory. It builds its own transient per-question FAISS index from the question's 10 distractor paragraphs, retrieves top-k, and reports retrieval metrics. No LLM calls, no HTTP routes, no coupling to `backend/chat/`.

The chat core, the existing library reindex path, the chat-time RAG chain, and the frontend are **all unchanged**. This iteration sits orthogonally next to chat: ingest populates `storage/library/` (consumed by chat exactly the same way as a manually-uploaded file), and the eval pipeline is a CLI tool with its own state.

**Iteration 9 Highlights:**
- New CLI: `scripts/ingest_hotpotqa.py` — downloads HotpotQA dev distractor, writes one Markdown file per question with H1-per-paragraph structure.
- New CLI: `scripts/eval_hotpotqa.py` — runs the dataset, computes paragraph recall@k + supporting-fact precision/recall/F1/EM at the paragraph level.
- New module: `backend/eval/` — `hotpotqa.py` (loader + sampling), `metrics.py` (pure functions), `cache.py` (SHA-keyed per-question FAISS cache).
- Per-question indices cached on disk at `storage/eval/hotpotqa/cache/{dataset_sha[:16]}/{qid}/` — first run is slow, subsequent runs are seconds.
- Sample-vs-full via `--subset N | --full` (mutually exclusive; default `--full`); stratified sampling with seed=42 for determinism.
- Library ingest contaminates nothing: gold answers, gold supporting facts, and the question text are **never** written into any library file. Eval reads those fields directly from the JSON.
- HotpotQA is CC BY-SA 4.0; attribution lives in `storage/library/hotpotqa/README.md` (written by the ingest script) and as a header in the eval script's terminal output.
- Zero new Python dependencies; zero new env vars; zero edits to `backend/chat/`, `backend/main.py`, or the frontend.

---

## Functional Requirements

### FR-30: HotpotQA Library Ingestion

| ID | Requirement |
|----|-------------|
| FR-30.1 | `scripts/ingest_hotpotqa.py` is a CLI that downloads `hotpot_dev_distractor_v1.json` from `https://hotpotqa.github.io/` (CC BY-SA 4.0) and caches the file at `scripts/.cache/hotpot_dev_distractor_v1.json`. SHA-256 of the file is the `dataset_sha` cache key referenced by FR-31.7. |
| FR-30.2 | The script writes one `.md` file per question into `storage/library/hotpotqa/<qid>.md` where `<qid>` is the hotpot `_id` slugified to `[a-zA-Z0-9_-]+` (non-conforming characters replaced with `_`). |
| FR-30.3 | Each file's body has YAML-style frontmatter with the four fields: `question_id`, `question_type` (`"bridge"` \| `"comparison"`), `question_level` (`"easy"` \| `"medium"` \| `"hard"`), and `source: hotpotqa`. |
| FR-30.4 | Each paragraph from the question's `context` becomes an H1 section (`# Title`) followed by the joined paragraph text (sentences joined with a single space, no period added). The number of H1 sections in the file equals the number of non-empty paragraphs in `context`. |
| FR-30.5 | The library file body **never** contains the question text, the gold `answer`, or any gold `supporting_facts` — those fields stay in the JSON only. |
| FR-30.6 | The CLI exposes a `--subset N` flag and a `--full` flag in a mutually-exclusive group; `--full` is the default. `--subset 0` and `--subset 1` are rejected with a usage error (not silently noop). |
| FR-30.7 | The `--subset N` option produces a stratified sample by `(type, level)` bucket (6 buckets). For each bucket, sample `min(ceil(N / 6), len(bucket))` items deterministically via `random.Random(42)`. Bucket cap prevents over-sampling when N is large. |
| FR-30.8 | The CLI exposes a `--force` flag that re-downloads the dataset even if the cached copy exists. |
| FR-30.9 | The CLI is idempotent: re-running on an unchanged dataset is a no-op for files whose content matches what would be written. Per-question file writes are atomic via `tmp + os.replace`. A half-written file from a crashed prior run is replaced on the next attempt. |
| FR-30.10 | The CLI emits the dataset attribution (`Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)`) to stdout once at the start of a run. |
| FR-30.11 | The CLI writes `storage/library/hotpotqa/README.md` on first run (or whenever missing), containing the dataset name, source URL, and a one-line license notice. |
| FR-30.12 | Network errors during download trigger one retry after a 5-second wait; the second failure exits non-zero with a download URL printed. |
| FR-30.13 | Per-question schema errors (missing `_id`, `context`, etc.) are logged as WARNING and the question is skipped; the final summary lists skipped IDs. Whole-file `JSONDecodeError` exits non-zero with a "fix the file or re-download" hint. |

### FR-31: Eval Pipeline (CLI Only)

| ID | Requirement |
|----|-------------|
| FR-31.1 | `scripts/eval_hotpetqa.py` is a standalone CLI. It does **not** register any HTTP route. It does **not** import anything from `backend/chat/`. |
| FR-31.2 | The CLI exposes a `--subset N | --full` mutually-exclusive group; `--full` is the default. The semantics match FR-30.7. |
| FR-31.3 | The CLI exposes a `--k N` flag for retrieval depth. Default is 4 (matches the FR-25 `top_k` default). |
| FR-31.4 | The CLI exposes a `--no-cache` flag that forces rebuild of every per-question FAISS index. |
| FR-31.5 | The CLI exposes a `--fixture PATH` flag for tests; the dataset path is taken from `--fixture` first, otherwise from `scripts/.cache/hotpot_dev_distractor_v1.json`, otherwise the CLI exits with download instructions. |
| FR-31.6 | For each question in the chosen subset, the script builds a transient FAISS index from the question's 10 distractor paragraphs as Documents (one paragraph per Document). The chunking pipeline used by chat (`MarkdownTextSplitter`) is NOT used here — paragraph granularity is preserved verbatim to keep the metric stable across reindex-config changes. |
| FR-31.7 | Each per-question index is persisted at `storage/eval/hotpotqa/cache/{dataset_sha[:16]}/{qid}/` (containing FAISS's `index.faiss` and `index.pkl`). The path directory `dataset_sha[:16]` is the cache-invalidation prefix: any change to the dataset JSON destroys all cached indices atomically. |
| FR-31.8 | On each cache lookup: if `--no-cache` is set OR the cache directory doesn't exist, build + save and report `hit=False`; otherwise load and report `hit=True`. If loading raises (corrupted cache), `shutil.rmtree` the cache dir, rebuild + save, log WARNING, report `hit=False`. |
| FR-31.9 | `backend/eval/hotpotqa.py` exposes `gold_paragraph_titles(item)` returning the set of distinct paragraph titles appearing in the item's `supporting_facts`. Titles are deduplicated (the same title at multiple sentence indices contributes once). |
| FR-31.10 | `backend/eval/metrics.py` exposes `paragraph_recall_at_k(retrieved_titles, gold_titles) -> float`. Returns 1.0 vacuously when `gold_titles` is empty. Otherwise returns `min(hits, len(gold_titles)) / len(gold_titles)` where `hits` is the number of `retrieved_titles` entries (with duplicates counted) that are in `gold_titles`. |
| FR-31.11 | `backend/eval/metrics.py` exposes `supporting_fact_metrics(retrieved_titles, gold_titles) -> tuple[precision, recall, f1, em]` with the standard set-comparison formulas. Edge cases: empty gold + empty retrieved → all four are 1.0; empty gold + non-empty retrieved → `(0, 1, 0, 0)`; non-empty gold + empty retrieved → all four are 0; non-empty gold + non-empty retrieved → standard formulas. |
| FR-31.12 | The terminal output reports the 5 metrics from FR-31.10–31.11 plus `paragraph_recall@{args.k}` averaged over the successfully-evaluated subset, the count of successfully evaluated questions (out of attempted), the cache hit / build split, the error count, and elapsed seconds. Format is the block in §3.6 of the brainstorming spec. Attribution header (`Dataset: HotpotQA ... CC BY-SA 4.0 ...`) is printed once before the metric block. |
| FR-31.13 | The CLI exits 0 when the run completes (per-question errors during retrieval are logged at WARNING and counted toward the `errors` field; they do not affect exit code). The CLI exits 1 only on setup failure (dataset missing, dataset `JSONDecodeError`, embedding model load failure). |
| FR-31.14 | Partial-result runs (some questions errored, some succeeded) are valid: the metric block shows both `successfully evaluated` and `attempted`, and exits 0. |

### FR-32: Isolation Guarantees

| ID | Requirement |
|----|-------------|
| FR-32.1 | `scripts/eval_hotpotqa.py` imports nothing from `backend/chat/` (verified by `grep -r "backend.chat" scripts/eval_hotpotqa.py backend/eval/` returning no results). |
| FR-32.2 | The eval pipeline may import: `backend.rag.embeddings` (sentence-transformers factory), `backend.rag.vector_store` (`load_or_init` / `save`), and `backend.eval.*`. These are pure primitives. |
| FR-32.3 | The eval pipeline never reads the `storage/library/` directory; it does not depend on the global library FAISS index existing. |
| FR-32.4 | The ingest pipeline never touches the FAISS indexes. Library reindex remains the existing `POST /api/rag/library/reindex` flow. |

### FR-33: HotpotQA Attribution

| ID | Requirement |
|----|-------------|
| FR-33.1 | A `storage/library/hotpotqa/README.md` file documents the dataset name, source URL, and CC BY-SA 4.0 license. Created by the ingest script when absent. |
| FR-33.2 | The eval script prints a one-line attribution header to stdout before the metric block. |

---

## Non-Functional Requirements

### NFR-12: No chat coupling

- The eval pipeline (`scripts/eval_hotpotqa.py` + `backend/eval/`) imports nothing from `backend/chat/`. A grep check is added to the manual smoke test (`grep -r "backend\\.chat" backend/eval/`).

### NFR-13: Cache safety

- The per-question cache directory is wiped before rebuild when loading fails. The disk format follows FAISS's native `save_local` / `load_local` round-trip. A corrupt cache never blocks the run — the rebuild path is exercised automatically.

### NFR-14: Deterministic subset sampling

- `random.Random(42)` is the only source of randomness in the sampling path. Two runs with `--subset N` (and the same dataset version) produce the same set of qids and the same evaluation order.

### NFR-15: No new dependencies

- No additions to `requirements.txt`. `sentence-transformers`, `FAISS` (via `langchain_community`), `langchain_core`, and stdlib `json` / `hashlib` / `argparse` are sufficient.

---

## Out of Scope (deferred to future iterations)

- LLM-based answer evaluation (calling `minimax-3` to score answers — would require per-question LLM calls, doubling eval cost and adding API-key dependencies)
- `/api/eval/` route (CLI only by design; UI integration deferred)
- JSON output for downstream tooling (`--json-out` flag deferred)
- Per-type / per-level breakdown in default output
- HotpotQA `fullwiki` setting (requires a 5M+ Wikipedia paragraph corpus as a separate ingestion pipeline)
- Multi-process or distributed evaluation
- Incremental cache invalidation beyond dataset SHA change (any single-byte change in the JSON busts all caches)
- CI hookup for the eval script
- Embedding-recipe sweeps (automatically try multiple `EMBEDDING_BACKEND` values)
- Cross-encoder re-ranking on top of FAISS results
- Surface-attribution in the library sidebar UI when files with `source=hotpotqa` are present (would require a frontend change; out of scope per FR-32 isolation guarantees)
- Multi-pass retrieval (Hop 2 using Hop-1 results to refine the query) — interesting follow-up for true multi-hop, but fullwiki pipeline required
- Sentence-level supporting-fact metrics (would require LLM-based extraction)
