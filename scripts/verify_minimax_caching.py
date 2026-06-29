"""
Verification script: does the MiniMax Anthropic-protocol gateway forward
cache_control markers, and does it return Anthropic-shaped usage fields?

Strategy:
  1. Build a long system prompt (>1024 tokens so caching can activate).
  2. Mark it with cache_control: {"type": "ephemeral"}.
  3. POST to /v1/messages twice within seconds.
  4. On a working Anthropic-compatible gateway with prompt caching:
     - Call 1 usage shows `cache_creation_input_tokens` > 0
     - Call 2 usage shows `cache_read_input_tokens` > 0
  5. Print the raw usage dict from each call so we can see the field shape.

API key is read from .env (ANTHROPIC_API_KEY). Never printed.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimax.chat/v1")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "minimax-3"

if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
    sys.exit(1)

# Anthropic-format endpoint. Strip trailing /v1 because the path is /v1/messages.
root = BASE_URL.rstrip("/")
if root.endswith("/v1"):
    root = root[:-3]
url = f"{root}/v1/messages"

# A long filler system prompt to exceed the 1024-token caching threshold.
# Anthropic caches prefixes of >= 1024 tokens; we aim for ~1200 tokens to be safe.
filler = (
    "The following is reference material for the assistant. "
    "Treat it as authoritative context for any user question.\n\n"
) + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200)


def build_payload(user_text: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 64,
        "system": [
            {
                "type": "text",
                "text": filler,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": user_text},
        ],
    }


def call(payload: dict, label: str) -> dict:
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"\n--- Call: {label} ---")
    print(f"HTTP status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Body (first 500 chars): {resp.text[:500]}")
        sys.exit(2)
    data = resp.json()
    usage = data.get("usage", {})
    print(f"usage keys: {sorted(usage.keys())}")
    print(f"usage full: {json.dumps(usage, indent=2)}")
    # Surface any cache-related fields at top level too (some gateways
    # surface them differently).
    for k, v in data.items():
        if "cache" in k.lower():
            print(f"top-level cache field {k!r}: {v}")
    return data


def main() -> int:
    print(f"Endpoint: {url}")
    print(f"Model:    {MODEL}")
    print(f"System prompt length (chars): {len(filler)} (estimated tokens: {len(filler)//4})")

    # First call: expected to CREATE cache (cache_creation_input_tokens > 0)
    call(build_payload("Summarize the reference material in one sentence."), "first (cache write expected)")

    # Wait briefly so the two requests land close together but distinctively.
    time.sleep(2)

    # Second call: expected to HIT cache (cache_read_input_tokens > 0)
    call(build_payload("Now list three key entities from it."), "second (cache read expected)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())