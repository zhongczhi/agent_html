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
# surface variation of the original question. Each prompt also names the
# style (lexical / structural / casual) so the style is detectable from
# the system prompt text alone (useful for downstream auditing and for
# tests that route mock responses by style).
STYLE_PROMPTS: dict[str, str] = {
    "lexical": (
        "You are a lexical paraphraser. You paraphrase questions. "
        "Output ONLY the paraphrase, no preamble. "
        "Keep the exact sentence structure of the original but substitute "
        "synonyms and minor word choices (e.g. 'In which year' -> 'What year'). "
        "Do NOT include the answer in your paraphrase. Output one sentence."
    ),
    "structural": (
        "You are a structural paraphraser. You paraphrase questions. "
        "Output ONLY the paraphrase, no preamble. "
        "Keep all the original entities and facts but reorder the clauses "
        "(e.g. active -> passive, 'X was born in Y' -> 'In which year was X "
        "born, given that Y is associated with X?'). Do NOT include the answer "
        "in your paraphrase. Output one sentence."
    ),
    "casual": (
        "You are a casual paraphraser. You paraphrase questions. "
        "Output ONLY the paraphrase, no preamble. "
        "Make the question informal and conversational, as if a real user "
        "typed it quickly in a chat: use contractions, drop articles where "
        "natural, allow lowercase. Do NOT include the answer in your "
        "paraphrase. Output one sentence."
    ),
}


def _default_paraphrase_path(dataset_path: Path) -> Path:
    return PARAPHRASES_DIR / f"{dataset_sha(dataset_path)}.json"


def _user_prompt(question: str, gold_answer: str) -> str:
    # We DO include the gold answer in the prompt so the model knows what to
    # avoid — but the validation gate then rejects anything that leaks.
    # Without this, the model has no signal that "Paris" is the answer to
    # avoid using in "When was X born?" paraphrases.
    return (
        f"Original question: {question}\n"
        f"Do NOT include this answer in your paraphrase: {gold_answer}"
    )


async def _generate_one_style(
    client: AsyncAnthropic,
    model: str,
    style: str,
    question: str,
    gold_answer: str,
) -> str:
    """One Anthropic call returning the paraphrase text for one style."""
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
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
    """Generate all 3 styles in parallel; validate; retry failures once.

    Each style runs as its own task: first-pass call, validate, and (if the
    first-pass leaked the answer) a single retry. The 3 tasks are gathered
    together so the 3 first-pass calls fire concurrently — keeping the
    wall-clock cost per question to roughly 2x a single call's latency.

    Returns {style: text} for styles that passed validation (possibly fewer
    than 3 if some failed twice).
    """
    styles = required_styles()

    async def gen_with_retry(style: str) -> tuple[str, str | None]:
        text1 = await _generate_one_style(
            client, model, style, question, gold_answer
        )
        if validate_paraphrase(text1, gold_answer):
            log.info("qid=? style=%s validated on first attempt", style)
            return style, text1
        log.warning("qid=? style=%s leaked answer; retrying", style)
        text2 = await _generate_one_style(
            client, model, style, question, gold_answer
        )
        if validate_paraphrase(text2, gold_answer):
            log.info("qid=? style=%s retry succeeded", style)
            return style, text2
        log.warning("qid=? style=%s failed validation twice; skipping", style)
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
            for item in items_all:
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