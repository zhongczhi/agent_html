# Chatbot Project — Iteration 11 Spec (End-to-End QA Accuracy Eval)

> **Working document for the current iteration.** Will be merged into [SPEC.md](SPEC.md) on completion.
> Adds an end-to-end QA accuracy eval that measures what users actually care about: does the LLM produce an answer matching the gold answer when given retrieved context?

## Overview

Iteration 10 fixed the paraphrase generator's validation-gate coverage problem (35% → 0% zero-coverage). But every metric in iter-9 and iter-10 measures **retrieval** (does the gold answer *appear* in top-k?), not **answer quality** (does the user get the right *answer*?).

A retriever with high recall can still produce a low-quality answer if:
- The retrieved context is noisy (low precision → LLM distracted by irrelevant chunks)
- The LLM hallucinates even when gold context is present
- The LLM extracts the wrong span from a long paragraph

Iteration 11 closes this gap with an end-to-end QA eval: for each HotpotQA question, retrieve top-k (same FAISS pipeline), feed the context to the LLM in the same prompt format the chat chain uses, extract the answer from the response, and score against the gold answer using HotpotQA's official answer-F1 metric.

The eval runs **two modes per question**:
1. **With context**: retrieved top-k embedded in `<context>...</context>` block (real chat behavior)
2. **Without context**: vanilla LLM call (no retrieval)

The delta between modes measures how much retrieval actually helps. If without-context ≥ with-context, our retrieval is hurting.

## Functional Requirements

### FR-40: HotpotQA Standard Answer F1

| ID | Requirement |
|----|-------------|
| FR-40.1 | `backend.eval.metrics.answer_f1(predicted: str, gold: str) -> float` computes the SQuAD-style token-F1 that HotpotQA's official eval script uses: lowercase, strip punctuation, remove articles (`a`, `an`, `the`), tokenize on whitespace, compute precision/recall/F1 over token sets. Returns 0.0 for empty predicted or gold. |
| FR-40.2 | `backend.eval.metrics.exact_match(predicted: str, gold: str) -> bool` returns True iff `answer_f1 == 1.0` (token sets are identical after normalization). Returns False for empty predicted or gold. |
| FR-40.3 | Both functions are pure: no I/O, no LLM, no Anthropic imports. |

### FR-41: QA Judge Module

| ID | Requirement |
|----|-------------|
| FR-41.1 | New module `backend.eval.qa_judge` exposes `build_qa_prompt(question: str, context_chunks: list[Document] | None) -> list[dict]`. With context: returns `[system_msg (RAG), user_msg (<context>...</context> + question)]`. Without context: returns `[user_msg (question only)]`. The system message is the same `RAG_SYSTEM_PROMPT` used by `backend.chat.chain`. |
| FR-41.2 | `qa_judge.ask_llm(client, model, prompt, max_tokens=200) -> str` calls the LLM once with `temperature=0`, extracts the text content (skipping thinking blocks), and returns the cleaned answer string. |
| FR-41.3 | `qa_judge.ask_llm` takes an `AsyncAnthropic` client (matches the paraphrase generator's pattern) and is fully async. |

### FR-42: End-to-End Eval CLI

| ID | Requirement |
|----|-------------|
| FR-42.1 | `scripts/eval_qa_hotpotqa.py` is a standalone CLI. It does **not** register any HTTP route. It does **not** import from `backend.chat.*` (same isolation rule as FR-32 for the retrieval eval). |
| FR-42.2 | The CLI exposes `--subset N | --full` mutually-exclusive group; `--full` is the default. Semantics match FR-31.2. |
| FR-42.3 | The CLI exposes `--k N` for retrieval depth. Default is 4 (matches FR-31.3). |
| FR-42.4 | The CLI exposes `--no-cache` flag (matches FR-31.4). |
| FR-42.5 | The CLI exposes `--fixture PATH` flag (matches FR-31.5). |
| FR-42.6 | The CLI exposes `--paraphrase-set PATH` flag. If absent, eval runs in original-only mode (one question variant per item). If present, eval runs original + each available paraphrase style. |
| FR-42.7 | The CLI exposes `--compare-baseline` flag. When set, each question is evaluated twice: with retrieved context AND without context. When absent, only with-context mode runs. |
| FR-42.8 | The CLI exposes `--llm-model NAME` flag. Default: `minimax-3` (override with `--llm-model` or `$ANTHROPIC_MODEL`). |
| FR-42.9 | The CLI reuses the existing per-question FAISS cache from `backend.eval.cache`. Cache key is `(dataset_sha, qid)` — same as the retrieval eval. |
| FR-42.10 | The CLI applies 1-second pacing between consecutive LLM calls within a question's flow (with-context + optional without-context + optional paraphrase variants). |
| FR-42.11 | The CLI exits 0 on completion. Per-question LLM errors are logged at WARNING and counted toward an `errors` field; they do not affect exit code. The CLI exits 1 only on setup failure (dataset missing, JSONDecodeError, embedding model load, missing API key). |

### FR-43: Output Format

| ID | Requirement |
|----|-------------|
| FR-43.1 | The CLI prints a header line: `HotpotQA End-to-End QA Eval — subset={...}, k={...}, dataset_sha={...}`. |
| FR-43.2 | The CLI prints `with_context:` section with `answer_f1` and `answer_em` averages. |
| FR-43.3 | If `--compare-baseline` is set, the CLI prints `without_context:` section with the same two metrics, then `delta (retrieval helps):` showing the signed difference for each. |
| FR-43.4 | If `--paraphrase-set` is given, the CLI prints `-- by variant --` section showing `n`, `f1`, `em` for `original`, `lexical`, `structural`, `casual`. |
| FR-43.5 | The CLI prints a footer with: total LLM calls, cache hits / builds, errors, elapsed seconds. |
| FR-43.6 | The CLI prints `Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)` attribution once before the metric block. |

### FR-44: Backward Compatibility

| ID | Requirement |
|----|-------------|
| FR-44.1 | `backend.eval.metrics.answer_coverage_at_k` is unchanged. `paragraph_recall_at_k` and `supporting_fact_metrics` are unchanged. |
| FR-44.2 | `scripts/eval_hotpotqa.py` (the retrieval-only eval) is unchanged. It continues to work standalone with `--paraphrase-set`. |
| FR-44.3 | The eval CLI imports from `backend.eval.*` (new) + `anthropic` (new). It does NOT import from `backend.chat.*` (preserves FR-32 isolation). |

## Non-Functional Requirements

### NFR-19: Cost ceiling

The `--subset 100` run with `--compare-baseline --paraphrase-set` issues at most 800 LLM calls (100 questions × 4 variants × 2 modes). At `minimax-3` pricing with `thinking.enabled` budget=10000, expect $5-10.

The `--subset 1000 --compare-baseline` run (no paraphrases) issues ~2000 LLM calls. Expect $15-25.

### NFR-20: Latency target

Per-question end-to-end latency: < 8 seconds (LLM call dominates; FAISS retrieval is ~50ms; answer extraction is negligible). With 1-second pacing, wall-clock for `--subset 1000` is ~17 minutes per mode (35 min with `--compare-baseline`).

### NFR-21: Reproducibility

LLM calls use `temperature=0` for determinism. Cache hits for FAISS indices make retrieval deterministic. MiniMax endpoint is not 100% reproducible (network jitter, server-side variance) but `temperature=0` provides best-effort determinism for answer text.

### NFR-22: Test isolation

All new code in `backend.eval.metrics`, `backend.eval.qa_judge`, and `scripts.eval_qa_hotpotqa` is testable without real LLM calls:
- Metrics tests use string inputs.
- `qa_judge.build_qa_prompt` is pure (no I/O).
- `qa_judge.ask_llm` is tested via `AsyncMock` for `AsyncAnthropic`.
- The CLI tests use subprocess + a small fixture JSON + mocked LLM via env-var stub.

## Out of Scope (deferred to future iterations)

- **Cross-encoder reranking** before LLM call (the iter-10 retrieval-only eval would benefit from this; the QA eval would inherit the gain).
- **Larger embedding model** (`all-mpnet-base-v2`) — same rationale.
- **Hybrid BM25 + dense** — same rationale.
- **Thinking budget tuning** — we use the same 10000 as chat. A lower budget (e.g., 2000) would speed up eval 5× but might change answer quality.
- **Multi-shot prompting** — HotpotQA is zero-shot in our setup; few-shot is a known win for QA but adds prompt complexity.
- **Sentence-level supporting-fact scoring** at the QA level (would require the LLM to emit supporting-fact lists, not just answers).
- **Calibration metrics** (does the LLM's confidence match its correctness?) — orthogonal to this eval.
- **Per-(type, level) breakdown** for the QA eval — the sample-size for `comparison/hard` in our 334-question subset is small; deferring until full 7405-question run.
- **RAGAS-style metrics** (faithfulness, answer relevance) — these need a separate reference answer or LLM-as-judge pipeline; out of scope.