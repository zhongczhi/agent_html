"""Generate 3 styled paraphrases per HotpotQA question via 3 concurrent LLM
calls (one per style). Validates each paraphrase against the gold answer
(token-overlap gate, rejects >=80% overlap). Persists to disk as
{dataset_sha}.json under storage/eval/hotpotqa/paraphrases/. Idempotent:
re-running on an unchanged dataset skips qids already in the JSON.

Isolation: this script may import from backend.eval.* and from `anthropic`.
It does NOT import from backend.chat.* — that boundary is preserved for
scripts/eval_hotpotqa.py (the actual eval script), which must stay
LLM-call-free per FR-32.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Bootstrap sys.path so `python scripts/generate_paraphrases_hotpotqa.py`
# finds the `backend` package from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from backend.eval.hotpotqa import dataset_sha, load as load_dataset, sample  # noqa: E402
from backend.eval.paraphrases import (  # noqa: E402
    load_paraphrases,
    required_styles,
    validate_paraphrase,
)

log = logging.getLogger("generate_paraphrases_hotpotqa")

REPO_ROOT = _REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "scripts" / ".cache" / "hotpot_dev_distractor_v1.json"
PARAPHRASES_DIR = REPO_ROOT / "backend" / "storage" / "eval" / "hotpotqa" / "paraphrases"

# Style-specific system prompts. Each steers the LLM toward a distinct
# surface variation of the original question. The "HARD RULE" block is
# front-loaded so the model attends to it before style-specific instructions
# (FR-35.1). Each prompt also names the style (lexical / structural / casual)
# so the style is detectable from the system prompt text alone (useful for
# downstream auditing and for tests that route mock responses by style).
STYLE_PROMPTS: dict[str, str] = {
    "lexical": (
        "You are a lexical paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question using synonym swaps only "
        "(e.g., 'In which year' -> 'What year'). Keep the exact "
        "sentence structure.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
    "structural": (
        "You are a structural paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question by reordering clauses "
        "(e.g., active -> passive, 'X was born in Y' -> 'In which year "
        "was X born, given that Y is associated with X?'). Keep all "
        "the original entities and facts.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
    "casual": (
        "You are a casual paraphraser.\n\n"
        "HARD RULE: Do NOT include the answer to the question in your "
        "paraphrase. The answer is supplied below. If your paraphrase "
        "contains the answer, it is invalid and will be rejected.\n\n"
        "Task: paraphrase the question in an informal, conversational "
        "tone as if a real user typed it quickly in a chat: use "
        "contractions, drop articles where natural, allow lowercase.\n\n"
        "Output: ONLY the paraphrase, one sentence, no preamble."
    ),
}


def _default_paraphrase_path(dataset_path: Path) -> Path:
    return PARAPHRASES_DIR / f"{dataset_sha(dataset_path)}.json"


def _user_prompt(question: str, gold_answer: str) -> str:
    # We DO include the gold answer in the prompt so the model knows what to
    # avoid — but the validation gate then rejects anything that leaks.
    # Without this, the model has no signal that "Paris" is the answer to
    # avoid using in "When was X born?" paraphrases.
    # FR-35.4: gold answer is the FIRST line after a HARD RULE label, so the
    # model sees "what to avoid" before "what to paraphrase".
    return (
        f"Question to paraphrase: {question}\n"
        f"Answer to AVOID in your paraphrase (HARD RULE): {gold_answer}\n"
        f"Output ONLY the paraphrase, one sentence."
    )


# FR-36 + FR-38: temperature schedule + 5s pacing between API calls.
PACING_SECONDS = 5


def _retry_temperature_for(attempt: int) -> float:
    """Return the temperature for a given attempt number (1, 2, or 3).

    FR-36: attempt 1 = 0.3 (low variance, fast), attempt 2 = 0.7 (medium
    variance, real escape from attempt 1's local minimum), attempt 3 = 1.0
    (high variance, last shot before skip).
    """
    if attempt == 1:
        return 0.3
    if attempt == 2:
        return 0.7
    return 1.0  # attempt == 3


async def _generate_one_style(
    client: AsyncAnthropic,
    model: str,
    style: str,
    question: str,
    gold_answer: str,
    attempt: int,
) -> str:
    """One Anthropic call returning the paraphrase text for one style.

    `attempt` is 1-indexed (1 = first attempt, 2 = first retry, 3 = second
    retry). Temperature is determined by `_retry_temperature_for(attempt)`.

    FR-38.1: sleep `PACING_SECONDS` before each call so concurrent calls
    within a question are spaced 5s apart, reducing rate-limit (429) hits.
    """
    await asyncio.sleep(PACING_SECONDS)
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=_retry_temperature_for(attempt),
        system=STYLE_PROMPTS[style],
        messages=[
            {"role": "user", "content": _user_prompt(question, gold_answer)},
        ],
    )
    # Response has one text block (we asked for one sentence).
    return response.content[0].text.strip()


async def _generate_for_question(
    client: AsyncAnthropic,
    model: str,
    question: str,
    gold_answer: str,
) -> dict[str, str]:
    """Generate all 3 styles in parallel; validate; retry up to 3 attempts.

    Each style runs as its own task. Attempt 1 fires concurrently for all 3
    styles via asyncio.gather. Each task then validates and conditionally
    fires retry attempts 2 and 3 with progressively higher temperatures.
    The 3 tasks run concurrently throughout, so wall-clock per question is
    roughly 3× a single attempt's latency (worst case: all 3 attempts leak).

    Returns {style: text} for styles that passed validation (possibly fewer
    than 3 if all attempts leaked for that style).
    """
    styles = required_styles()

    async def gen_with_retry(style: str) -> tuple[str, str | None]:
        for attempt in (1, 2, 3):
            text = await _generate_one_style(
                client, model, style, question, gold_answer, attempt,
            )
            if validate_paraphrase(text, gold_answer):
                log.info(
                    "qid=? style=%s attempt=%d temp=%.1f accepted",
                    style, attempt, _retry_temperature_for(attempt),
                )
                return style, text
            log.warning(
                "qid=? style=%s attempt=%d temp=%.1f leaked; %s",
                style, attempt, _retry_temperature_for(attempt),
                "retrying" if attempt < 3 else "skipping",
            )
        return style, None

    results = await asyncio.gather(*[gen_with_retry(s) for s in styles])
    return {s: t for s, t in results if t is not None}


def _write_output(
    output_path: Path,
    dataset_sha_hex: str,
    items: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_sha": dataset_sha_hex,
        "schema_version": 1,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate HotpotQA paraphrase set via 3 concurrent LLM calls per question."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--subset",
        type=int,
        metavar="N",
        help="Stratified sample of N questions (mutually exclusive with --full)",
    )
    grp.add_argument(
        "--full",
        action="store_true",
        help="Use all questions (default)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Read dataset from PATH (test hook)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", "minimax-3"),
        help="Anthropic model name (default: minimax-3, override with --model or $ANTHROPIC_MODEL)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate paraphrases even for qids already in the output file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: storage/eval/hotpotqa/paraphrases/{dataset_sha}.json)",
    )
    args = parser.parse_args(argv)

    if args.subset is not None and args.subset <= 1:
        parser.error("--subset must be >= 2")

    dataset_path = args.fixture or DEFAULT_DATASET
    if not dataset_path.exists():
        print(
            f"Dataset not found at {dataset_path}.\n"
            "Run scripts/ingest_hotpotqa.py first, or pass --fixture PATH.",
            file=sys.stderr,
        )
        return 1
    try:
        items_all = load_dataset(dataset_path)
    except json.JSONDecodeError as e:
        print(f"Dataset JSON is corrupt: {e}", file=sys.stderr)
        return 1
    if args.subset is not None:
        items_all = sample(items_all, args.subset)

    d_sha = dataset_sha(dataset_path)
    output_path = args.output or _default_paraphrase_path(dataset_path)

    existing: dict[str, dict[str, str]] = {}
    if output_path.exists() and not args.force:
        try:
            existing = load_paraphrases(output_path)
        except json.JSONDecodeError:
            log.warning("Existing paraphrase file at %s is corrupt; regenerating", output_path)
            existing = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY env var is not set. Set it before running.",
            file=sys.stderr,
        )
        return 1

    async def run() -> dict[str, dict[str, str]]:
        # Preserve existing entries (unless --force) and add new ones.
        merged = dict(existing) if not args.force else {}
        async with AsyncAnthropic(api_key=api_key) as client:
            for idx, item in enumerate(items_all):
                if item.id in merged and not args.force:
                    log.info("Skipping qid=%s (already in JSON)", item.id)
                    continue
                try:
                    paraphrases = await _generate_for_question(
                        client, args.model, item.question, item.answer
                    )
                except Exception as e:
                    log.warning("qid=%s generation failed: %s", item.id, e)
                    continue
                if paraphrases:
                    merged[item.id] = {"paraphrases": paraphrases}
                    log.info(
                        "qid=%s generated %d/%d styles",
                        item.id,
                        len(paraphrases),
                        len(required_styles()),
                    )
                # FR-38.2: pace 5s between questions (skip after the last).
                if idx < len(items_all) - 1:
                    await asyncio.sleep(PACING_SECONDS)
        return merged

    all_items = asyncio.run(run())
    _write_output(output_path, d_sha, all_items)
    log.info(
        "Wrote %d paraphrase entries to %s",
        len(all_items),
        output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())