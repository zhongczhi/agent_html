# Chatbot Project — Iteration 11 Design (End-to-End QA Accuracy Eval)

> **Working document for the current iteration.** Will be merged into [DESI.md](DESI.md) on completion.
> See [SPEC_focus.md](SPEC_focus.md) for requirements. This document covers the architectural choices and module changes.

---

## 1. Architecture Decisions

### 1.1 Reuse the Retrieval Pipeline, Add an LLM Call

**Choice**: The new eval pipeline (`scripts/eval_qa_hotpotqa.py`) reuses the same per-question FAISS indices built by `backend.eval.cache`. The only new step is calling the LLM with the retrieved context.

**Rationale**:
- Decoupling retrieval and answer evaluation lets us measure retrieval *and* answer quality independently.
- The cache is dataset-SHA-keyed, so a re-run with the same dataset hits cache and runs in seconds.
- Adding `--no-cache` forces a rebuild (same as the retrieval eval).

**Trade-off**: The new CLI doesn't share CLI arguments with `eval_hotpotqa.py`. Some duplication of argparse setup (~15 lines). Acceptable given the different goals (retrieval-only vs end-to-end).

### 1.2 Use the Same Prompt Format as the Chat Chain

**Choice**: `qa_judge.build_qa_prompt` produces the exact same prompt format that `backend.chat.service._embed_context` produces. System message is `RAG_SYSTEM_PROMPT` from `backend.chat.chain`. User message embeds retrieved chunks in `<context>...</context>` followed by the question.

**Rationale**:
- We want the eval to reflect real chat behavior. If the eval uses a different prompt than production, the numbers don't mean what we want them to mean.
- The chat chain already imports `RAG_SYSTEM_PROMPT` from `backend.chat.chain`. To preserve FR-32 isolation, the QA eval can't import from `backend.chat.*` — so `qa_judge` defines its own copy of the prompt template. We document this duplication.

**Trade-off**: Two copies of the RAG system prompt string. If the chat prompt changes, both must be updated. The duplication is enforced (intentional) by the FR-32 isolation rule.

### 1.3 HotpotQA Standard Answer F1 (SQuAD-style)

**Choice**: Token-overlap F1 with HotpotQA's standard normalization: lowercase, strip punctuation, remove articles (`a`, `an`, `the`), tokenize on whitespace.

**Rationale**:
- This is what the HotpotQA paper reports, so our numbers are directly comparable.
- It's the official metric used by HotpotQA's eval script (`hotpot_evaluate_v1.py`).
- It handles short answers (`yes`, `1968`), multi-word answers (`The Lord of the Rings`), and yes/no uniformly.

**Trade-off**: A multi-line answer like "The Lord of the Rings: The Fellowship of the Ring" loses precision on the colon-separated variant. Acceptable: we only score against the single gold answer, not variants.

### 1.4 Compare-With-Baseline Mode

**Choice**: `--compare-baseline` flag runs each question twice: with-context and without-context. The output reports both and the delta.

**Rationale**:
- The without-context mode is the baseline: vanilla LLM with no retrieval.
- The delta (`with - without`) tells us how much retrieval *actually* helps. If it's negative, our retrieval is hurting.
- Costs 2× the LLM calls. Worth it for the diagnostic signal.

**Trade-off**: 2× the API cost. For `--subset 1000`, that's ~2000 calls instead of 1000. At `minimax-3` pricing with thinking enabled, ~$15-25 instead of ~$8-12.

### 1.5 One-Second Pacing (Not Five)

**Choice**: 1-second pacing between LLM calls (not the 5 seconds used in the paraphrase generator).

**Rationale**:
- LLM calls are heavier than paraphrase calls; rate-limit (429) is more sensitive.
- But the QA eval has fewer total calls (1000-4000) than the paraphrase generator (1300+), so the absolute wait time matters less.
- 1s is enough to keep the burst rate low without doubling wall-clock.

**Trade-off**: Slightly higher 429 risk than the paraphrase generator. The Anthropic SDK retries transparently so we don't see hard failures.

### 1.6 Skip the LLM Call When Gold Is Already in Top-k

**Choice**: If `answer_coverage_at_k(retrieved_texts, gold)` is 1.0 (gold already in top-k), still call the LLM. Don't optimize the "trivial" case.

**Rationale**:
- The LLM might extract the wrong span even when the gold is present (the "right context, wrong answer" failure mode).
- Skipping the LLM call would conflate "retrieval hit" with "LLM answered correctly" — they're different failure modes.

**Trade-off**: More LLM calls. Acceptable: the eval is about end-to-end accuracy, not retrieval-only recall.

### 1.7 Subset Defaults

**Choice**: Default subset is 100 (not 1000). `--full` runs all questions (334 in dev_distractor's effective 2-bucket dataset).

**Rationale**:
- 100 questions × 4 modes (with, without, original, paraphrase variants) ≈ 800 LLM calls ≈ ~$5. Cheap.
- 100 questions gives a reasonable signal for "is the eval pipeline working?" without burning budget.
- Operators can scale to `--subset 1000 --compare-baseline` (no paraphrases) for the full signal at ~$15.

**Trade-off**: Subset 100 has noisy per-variant metrics (only ~25 per bucket). Subset 1000 is the standard for publication-quality numbers.

---

## 2. Module Layout

### 2.1 New Files

```
backend/eval/
└── qa_judge.py             # NEW: build_qa_prompt, ask_llm

backend/tests/eval/
└── test_qa_judge.py        # NEW: tests for prompt construction + mock LLM

scripts/
└── eval_qa_hotpotqa.py     # NEW: end-to-end QA eval CLI
```

### 2.2 Modified Files

| File | Change |
|---|---|
| `backend/eval/metrics.py` | Add `answer_f1(predicted, gold) -> float` and `exact_match(predicted, gold) -> bool`. Both pure, no I/O. |
| `backend/tests/eval/test_metrics.py` | Add tests for `answer_f1` and `exact_match`. |
| `document/SPEC.md` | Add FR-40..FR-44 + NFR-19..NFR-22 as section 17. |
| `document/DESI.md` | Add this design as section 17. |

### 2.3 Unchanged Files

- `backend/eval/cache.py` — reused as-is.
- `backend/eval/hotpotqa.py` — reused as-is.
- `backend/eval/paraphrases.py` — reused as-is (paraphrase set is loaded if `--paraphrase-set` is given).
- `backend/chat/*` — `RAG_SYSTEM_PROMPT` is duplicated in `qa_judge.py` to preserve FR-32 isolation. The string is a constant; a sync failure would be a bug.
- `scripts/eval_hotpotqa.py` — unchanged retrieval eval.
- `scripts/generate_paraphrases_hotpotqa.py` — unchanged.

---

## 3. Component Skeletons

### 3.1 `backend/eval/metrics.py` — additions

```python
def _normalize_for_answer(text: str) -> str:
    """HotpotQA standard normalization: lowercase, strip punctuation,
    remove articles, collapse whitespace, tokenize.
    Returns list of tokens."""
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)            # strip punctuation
    text = _WHITESPACE_RE.sub(" ", text).strip()  # collapse whitespace
    # Remove articles per HotpotQA convention.
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.split() if text else []


def answer_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token F1 after HotpotQA normalization.
    Returns 0.0 for empty predicted or gold."""
    pred_tokens = _normalize_for_answer(predicted)
    gold_tokens = _normalize_for_answer(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    # Count token occurrences (not just set membership) to handle duplicates.
    from collections import Counter
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    num_same = sum(min(pred_counter[t], gold_counter[t]) for t in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, gold: str) -> bool:
    """Returns True iff token sets are identical after normalization."""
    return (
        _normalize_for_answer(predicted) == _normalize_for_answer(gold)
        and bool(_normalize_for_answer(predicted))
    )
```

### 3.2 `backend/eval/qa_judge.py` — new module

```python
"""End-to-end QA prompt builder + LLM caller for the iter-11 eval pipeline.

Isolation note: this module duplicates the RAG system prompt from
backend.chat.chain to preserve the FR-32 isolation rule that
scripts/eval_qa_hotpotqa.py must not import from backend.chat.*.
If the chat prompt changes, update RAG_SYSTEM_PROMPT_HERE in lockstep.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

# Mirror of backend.chat.chain.RAG_SYSTEM_PROMPT. See isolation note above.
RAG_SYSTEM_PROMPT_HERE = (
    "You are a helpful assistant. When the user's message contains a "
    "<context>...</context> block, treat the contents as grounding material: "
    "prefer it over your general knowledge when answering the question that "
    "follows the block. Do not mention the tag itself or the retrieval "
    "mechanism to the user."
)


def build_qa_prompt(
    question: str,
    context_docs: list[Document] | None,
) -> list[dict]:
    """Build the messages list for the LLM call.

    With context: [system (RAG), user (<context>...</context> + question)].
    Without context: [user (question only)].
    """
    if context_docs is None or not context_docs:
        return [{"role": "user", "content": question}]

    context_str = "\n\n".join(
        f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
    )
    user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT_HERE},
        {"role": "user", "content": user_content},
    ]


async def ask_llm(
    client: "AsyncAnthropic",
    model: str,
    messages: list[dict],
    max_tokens: int = 200,
) -> str:
    """One Anthropic call returning the answer text (no thinking blocks)."""
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )
    # Extract text blocks, skip thinking blocks.
    parts: list[str] = []
    for block in response.content:
        # Block may be a TypedDict or a BaseModel; access via getattr.
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()
```

### 3.3 `scripts/eval_qa_hotpotqa.py` — new CLI

```python
"""End-to-end QA accuracy eval for HotpotQA. CLI only.

Isolated from chat: imports nothing from backend.chat.*.
The RAG system prompt is duplicated in backend.eval.qa_judge to preserve
this isolation rule (FR-32).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Bootstrap sys.path so `python scripts/eval_qa_hotpotqa.py` (any cwd) finds
# the `backend` package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from backend.eval import cache as ev_cache  # noqa: E402
from backend.eval import hotpotqa as hotpot  # noqa: E402
from backend.eval import metrics  # noqa: E402
from backend.eval.qa_judge import ask_llm, build_qa_prompt  # noqa: E402
from backend.eval.paraphrases import load_paraphrases  # noqa: E402
from backend.rag.config import RagSettings  # noqa: E402
from backend.rag.embeddings import make_embeddings  # noqa: E402

log = logging.getLogger("eval_qa_hotpotqa")

REPO_ROOT = _REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"

PACING_SECONDS = 1  # NFR-20: 1s between LLM calls.


async def _evaluate_question(
    client: AsyncAnthropic,
    model: str,
    item,
    retrieved_docs: list,  # langchain Document list (or None for baseline)
    question_text: str,
    variant_name: str,
    mode: str,  # "with_context" | "without_context"
) -> dict:
    """One LLM call + scoring."""
    await asyncio.sleep(PACING_SECONDS)
    if mode == "with_context":
        prompt = build_qa_prompt(question_text, retrieved_docs)
    else:
        prompt = build_qa_prompt(question_text, None)
    answer = await ask_llm(client, model, prompt)
    f1 = metrics.answer_f1(answer, item.answer)
    em = metrics.exact_match(answer, item.answer)
    return {
        "qid": item.id,
        "variant": variant_name,
        "mode": mode,
        "predicted": answer,
        "gold": item.answer,
        "answer_f1": f1,
        "answer_em": 1.0 if em else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="HotpotQA end-to-end QA accuracy eval."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--subset", type=int, metavar="N")
    grp.add_argument("--full", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--paraphrase-set", type=Path, default=None)
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run without-context (vanilla LLM) for retrieval-lift measurement.",
    )
    parser.add_argument("--llm-model", default=os.environ.get("ANTHROPIC_MODEL", "minimax-3"))
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY env var is not set.", file=sys.stderr)
        return 1

    dataset_path = args.fixture or DEFAULT_DATASET
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}.", file=sys.stderr)
        return 1
    try:
        items = hotpot.load(dataset_path)
    except json.JSONDecodeError as e:
        print(f"Dataset JSON is corrupt: {e}", file=sys.stderr)
        return 1
    if args.subset is not None:
        items = hotpot.sample(items, args.subset)

    d_sha = hotpot.dataset_sha(dataset_path)
    print("Dataset: HotpotQA dev_distractor v1 (CC BY-SA 4.0 — https://hotpotqa.github.io/)")

    paraphrases: dict[str, dict[str, str]] = {}
    if args.paraphrase_set:
        if not args.paraphrase_set.exists():
            print(f"Paraphrase set not found: {args.paraphrase_set}", file=sys.stderr)
            return 1
        try:
            paraphrases = load_paraphrases(args.paraphrase_set)
            log.info("Loaded %d paraphrase entries from %s", len(paraphrases), args.paraphrase_set)
        except json.JSONDecodeError as e:
            print(f"Paraphrase set is corrupt: {e}", file=sys.stderr)
            return 1

    settings = RagSettings()
    embeddings = make_embeddings(settings.rag_embedding_backend)

    per_q: list[dict] = []
    cache_hits = cache_builds = errors = 0
    t0 = time.monotonic()

    async def run() -> int:
        nonlocal cache_hits, cache_builds, errors
        async with AsyncAnthropic(api_key=api_key) as client:
            for item in items:
                try:
                    index, hit = ev_cache.load_or_build(
                        item, d_sha, embeddings, no_cache=args.no_cache
                    )
                    if hit:
                        cache_hits += 1
                    else:
                        cache_builds += 1

                    # Build (question_text, variant_name) list.
                    variants: list[tuple[str, str]] = [(item.question, "original")]
                    para_entry = paraphrases.get(item.id)
                    if para_entry:
                        for style in ("lexical", "structural", "casual"):
                            if style in para_entry.get("paraphrases", {}):
                                variants.append((para_entry["paraphrases"][style], style))

                    for q_text, vname in variants:
                        retrieved_docs = index.similarity_search(q_text, k=args.k)
                        # With-context mode
                        per_q.append(await _evaluate_question(
                            client, args.llm_model, item, retrieved_docs,
                            q_text, vname, "with_context",
                        ))
                        if args.compare_baseline:
                            # Without-context mode (no retrieval)
                            per_q.append(await _evaluate_question(
                                client, args.llm_model, item, None,
                                q_text, vname, "without_context",
                            ))
                except Exception as e:
                    log.warning("qid=%s error: %s", item.id, e)
                    errors += 1

        elapsed = time.monotonic() - t0

        def avg(predicate, key):
            relevant = [r[key] for r in per_q if predicate(r)]
            return (sum(relevant) / len(relevant)) if relevant else 0.0

        def fmt(x: float) -> str:
            return f"{x:.3f}"

        label = "full" if args.subset is None else str(args.subset)
        print(f"\nHotpotQA End-to-End QA Eval — subset={label}, k={args.k}, dataset_sha={d_sha}")

        # with_context
        with_n = sum(1 for r in per_q if r["mode"] == "with_context")
        print("  with_context:")
        print(f"    answer_f1   : {fmt(avg(lambda r: r['mode'] == 'with_context', 'answer_f1'))}  (n={with_n})")
        print(f"    answer_em   : {fmt(avg(lambda r: r['mode'] == 'with_context', 'answer_em'))}  (n={with_n})")

        if args.compare_baseline:
            without_n = sum(1 for r in per_q if r["mode"] == "without_context")
            print("  without_context (baseline):")
            print(f"    answer_f1   : {fmt(avg(lambda r: r['mode'] == 'without_context', 'answer_f1'))}  (n={without_n})")
            print(f"    answer_em   : {fmt(avg(lambda r: r['mode'] == 'without_context', 'answer_em'))}  (n={without_n})")
            with_f1 = avg(lambda r: r["mode"] == "with_context", "answer_f1")
            without_f1 = avg(lambda r: r["mode"] == "without_context", "answer_f1")
            with_em = avg(lambda r: r["mode"] == "with_context", "answer_em")
            without_em = avg(lambda r: r["mode"] == "without_context", "answer_em")
            print("  delta (retrieval helps):")
            print(f"    answer_f1   : {fmt(with_f1 - without_f1):>7s}  ({fmt(with_f1)} - {fmt(without_f1)})")
            print(f"    answer_em   : {fmt(with_em - without_em):>7s}  ({fmt(with_em)} - {fmt(without_em)})")

        # by variant (if paraphrases)
        if paraphrases:
            print("  -- by variant -- (with_context)")
            for variant in ("original", "lexical", "structural", "casual"):
                n = sum(1 for r in per_q if r["mode"] == "with_context" and r["variant"] == variant)
                if n == 0:
                    print(f"  {variant:<12} : (no data)")
                    continue
                f1 = avg(lambda r: r["mode"] == "with_context" and r["variant"] == variant, "answer_f1")
                em = avg(lambda r: r["mode"] == "with_context" and r["variant"] == variant, "answer_em")
                print(f"  {variant:<12} : n={n:<4}  f1={fmt(f1)}  em={fmt(em)}")

        # Footer
        llm_calls = len(per_q)
        if args.compare_baseline:
            llm_calls = sum(1 for _ in per_q)  # all calls are LLM
        print(f"  LLM calls             : {llm_calls}")
        print(f"  cache hits / builds   : {cache_hits} / {cache_builds}")
        print(f"  errors                : {errors}")
        print(f"  elapsed               : {elapsed:.1f}s")
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
```

### 3.4 `backend/tests/eval/test_metrics.py` — additions

```python
from backend.eval.metrics import answer_f1, exact_match


def test_answer_f1_exact_match():
    assert answer_f1("The Godfather", "The Godfather") == 1.0


def test_answer_f1_case_insensitive():
    assert answer_f1("the godfather", "The Godfather") == 1.0


def test_answer_f1_strips_punctuation():
    assert answer_f1("The Godfather.", "The Godfather") == 1.0


def test_answer_f1_strips_articles():
    # 'The' is removed; 'cat' vs 'cat' matches.
    assert answer_f1("a cat", "cat") == 1.0


def test_answer_f1_partial_overlap():
    # pred="The Godfather Part II", gold="The Godfather" -> 2/3 P, 2/2 R, F1=0.8
    assert answer_f1("The Godfather Part II", "The Godfather") == pytest.approx(0.8)


def test_answer_f1_empty_predicted():
    assert answer_f1("", "yes") == 0.0


def test_answer_f1_empty_gold():
    assert answer_f1("anything", "") == 0.0


def test_answer_f1_no_overlap():
    assert answer_f1("apple", "banana") == 0.0


def test_exact_match_identical_tokens():
    assert exact_match("The Godfather", "The Godfather") is True


def test_exact_match_after_normalization():
    assert exact_match("the godfather", "The Godfather") is True


def test_exact_match_partial():
    assert exact_match("The Godfather II", "The Godfather") is False


def test_exact_match_empty():
    assert exact_match("", "yes") is False
    assert exact_match("anything", "") is False
```

### 3.5 `backend/tests/eval/test_qa_judge.py` — new file

```python
"""Tests for backend.eval.qa_judge. No real LLM calls."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.eval.qa_judge import ask_llm, build_qa_prompt


class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def test_build_qa_prompt_without_context():
    msgs = build_qa_prompt("What year?", None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What year?"


def test_build_qa_prompt_with_empty_context():
    msgs = build_qa_prompt("What year?", [])
    assert len(msgs) == 1  # no system msg when no context


def test_build_qa_prompt_with_context():
    docs = [
        FakeDoc("Born in 1968 in Berlin.", {"title": "John Smith"}),
        FakeDoc("A composer.", {"title": "John Smith Bio"}),
    ]
    msgs = build_qa_prompt("When was John Smith born?", docs)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "<context>" in msgs[0]["content"] or "<context>" in msgs[1]["content"]
    # Check the user message contains both chunks and the question.
    user_content = msgs[1]["content"]
    assert "Born in 1968" in user_content
    assert "When was John Smith born?" in user_content
    assert "John Smith" in user_content  # title appears as prefix


def _mock_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return MagicMock(content=[block])


def _mock_thinking_then_text(thinking: str, text: str) -> MagicMock:
    t = MagicMock(); t.type = "thinking"; t.thinking = thinking
    txt = MagicMock(); txt.type = "text"; txt.text = text
    return MagicMock(content=[t, txt])


@pytest.mark.asyncio
async def test_ask_llm_returns_text():
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_text_response("1968"))
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    assert out == "1968"


@pytest.mark.asyncio
async def test_ask_llm_skips_thinking_blocks():
    client = MagicMock()
    client.messages.create = AsyncMock(
        return_value=_mock_thinking_then_text("internal reasoning...", "1968")
    )
    out = await ask_llm(client, "minimax-3", [{"role": "user", "content": "Q"}])
    assert out == "1968"
    assert "internal reasoning" not in out
```

---

## 4. Configuration

No new env vars. Reuses `ANTHROPIC_API_KEY` (already required for the chat server). Reuses `ANTHROPIC_MODEL` if set (defaults to `minimax-3`).

## 5. Error Handling

| Stage | Failure | Behavior |
|---|---|---|
| Eval: missing `ANTHROPIC_API_KEY` | env var unset | Exit 1 with hint to set it. |
| Eval: dataset missing | path doesn't exist | Exit 1 with download instructions. |
| Eval: dataset corrupt | JSON parse fails | Exit 1. |
| Eval: embedding model load | sentence-transformers not installed | Exit 1. |
| Eval: per-question cache corrupted | `load_local` raises | `shutil.rmtree`, rebuild, WARNING log, continue. |
| Eval: per-question LLM call fails | API error / network error | Log WARNING, count as errored, skip rest of run unaffected. |
| Eval: per-question LLM call rate-limited (429) | Anthropic SDK retries internally | Transparent. |
| Eval: LLM returns empty content | text extraction yields "" | `answer_f1("", gold)` returns 0.0; counted normally. |

## 6. Testing Strategy

### 6.1 Layers

| Layer | Files | Speed |
|---|---|---|
| Metrics unit | `backend/tests/eval/test_metrics.py` (NEW: answer_f1, exact_match) | <10 ms each |
| QA judge unit | `backend/tests/eval/test_qa_judge.py` (NEW) | <100 ms each |
| Retrieval eval integration | `backend/tests/eval/test_eval_integration.py` (unchanged) | <5 s |
| Manual smoke | operator runs `python scripts/eval_qa_hotpotqa.py --subset 10` | ~30 s |

### 6.2 Manual Smoke Test

```bash
# 1. Small smoke test (10 questions, no baseline, no paraphrases)
python scripts/eval_qa_hotpotqa.py --subset 10
# expect: 10 LLM calls, sensible f1/em numbers

# 2. Medium run with baseline + paraphrases
python scripts/eval_qa_hotpotqa.py --subset 100 \
    --compare-baseline \
    --paraphrase-set backend/storage/eval/hotpotqa/paraphrases/4e9ecb5c8d3b719f.json
# expect: ~800 LLM calls, ~5-10 min wall-clock, sensible f1/em numbers

# 3. Full run (334 effective questions, baseline only)
python scripts/eval_qa_hotpotqa.py --subset 1000 --compare-baseline
# expect: ~2000 LLM calls, ~30-40 min wall-clock
```

### 6.3 Isolation Guard

```bash
grep -rn "backend\.chat" backend/eval/qa_judge.py scripts/eval_qa_hotpotqa.py
# Expected: no matches (qa_judge intentionally duplicates the prompt as a string constant;
# scripts/eval_qa_hotpotqa.py imports from backend.eval.* only).
```

---

## 7. Implementation Tasks (TDD)

### Task 1: Add `answer_f1` + `exact_match` to metrics.py

**Files**:
- Modify: `backend/eval/metrics.py`
- Modify: `backend/tests/eval/test_metrics.py`

- [ ] Step 1: Write the failing tests in `test_metrics.py` (see §3.4)
- [ ] Step 2: Run — expect failure
- [ ] Step 3: Add `_normalize_for_answer`, `answer_f1`, `exact_match` to `metrics.py` (see §3.1)
- [ ] Step 4: Run — expect pass
- [ ] Step 5: Commit `feat(eval): add answer_f1 + exact_match metrics (HotpotQA standard)`

### Task 2: Build `backend/eval/qa_judge.py`

**Files**:
- Create: `backend/eval/qa_judge.py`
- Create: `backend/tests/eval/test_qa_judge.py`

- [ ] Step 1: Write the failing tests in `test_qa_judge.py` (see §3.5)
- [ ] Step 2: Run — expect failure (module not found)
- [ ] Step 3: Create `qa_judge.py` with `build_qa_prompt` + `ask_llm` (see §3.2)
- [ ] Step 4: Run — expect pass
- [ ] Step 5: Commit `feat(eval): qa_judge module — prompt builder + LLM caller`

### Task 3: Build `scripts/eval_qa_hotpotqa.py` CLI

**Files**:
- Create: `scripts/eval_qa_hotpotqa.py`

- [ ] Step 1: Write the script (see §3.3)
- [ ] Step 2: Manual smoke: `--subset 10` — verify it produces sensible f1/em numbers
- [ ] Step 3: Commit `feat(scripts): end-to-end QA accuracy eval CLI`

### Task 4: Run + report

- [ ] Step 1: `--subset 100 --compare-baseline --paraphrase-set ...` (~5-10 min, ~$5)
- [ ] Step 2: `--subset 1000 --compare-baseline` (~30-40 min, ~$15)
- [ ] Step 3: Write `docs/eval-results/2026-07-XX-end-to-end-qa-eval.md`
- [ ] Step 4: Commit `docs(eval-results): end-to-end QA eval report`

### Task 5: Push

- [ ] Step 1: Push all commits to origin/master

---

## 8. Out of Scope (Deferred to Future Iterations)

1. **Cross-encoder reranking** before LLM call.
2. **Larger embedding model** (`all-mpnet-base-v2`).
3. **Hybrid BM25 + dense** retrieval.
4. **Thinking budget tuning** — we use the same 10000 as chat.
5. **Multi-shot prompting** for QA.
6. **Sentence-level supporting-fact scoring** at the QA level.
7. **Calibration metrics** (does confidence match correctness?).
8. **Per-(type, level) breakdown** for the QA eval (deferred to full-dataset run).
9. **RAGAS-style metrics** (faithfulness, answer relevance).
10. **LLM-as-judge** for subjective quality (e.g., "is the answer helpful?").