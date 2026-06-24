import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
CONVERSATIONS_FILE = STORAGE_DIR / "conversations.json"

# Serializes concurrent writers within the same process so load → modify →
# save cycles never interleave. Without this, two writers reading the same
# baseline would each write back their own version, losing one change.
# Crash-safety is handled separately by _atomic_write_json below. This
# lock is per-process; a multi-worker deployment would additionally need a
# file-level lock (fcntl/msvcrt) — out of scope for D2.
_write_lock = threading.Lock()


def _ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_conversations() -> dict:
    # No lock: reads are concurrent. Combined with atomic writes, a read
    # sees either the fully-old or fully-new file — never partial.
    _ensure_storage_dir()
    if not CONVERSATIONS_FILE.exists():
        return {"conversations": {}}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Move the unreadable file aside so the user can recover it, then
        # start fresh. Overwriting in-place would silently destroy history.
        backup = CONVERSATIONS_FILE.with_suffix(".json.corrupt")
        CONVERSATIONS_FILE.rename(backup)
        logger.warning(
            "Failed to decode conversations.json, moved to %s and starting fresh",
            backup,
        )
        return {"conversations": {}}


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to <path>.tmp in the same directory,
    then os.replace() to swap. Atomic on POSIX and on Windows when source
    and destination are on the same volume (they are). A crash before the
    replace leaves the original file intact; after the replace the new
    file is in place. No reader ever sees a partial write."""
    _ensure_storage_dir()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def get_conversation(conversation_id: str) -> dict | None:
    data = _load_conversations()
    conversations = data.get("conversations", {})
    return conversations.get(conversation_id)


def create_conversation(conversation_id: str) -> None:
    """Create an empty conversation entry so it appears in the conversation list."""
    with _write_lock:
        data = _load_conversations()
        if "conversations" not in data:
            data["conversations"] = {}
        if conversation_id not in data["conversations"]:
            data["conversations"][conversation_id] = {
                "conversation_id": conversation_id,
                "messages": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            _atomic_write_json(CONVERSATIONS_FILE, data)


def save_conversation(conversation_id: str, messages: list) -> None:
    with _write_lock:
        data = _load_conversations()
        if "conversations" not in data:
            data["conversations"] = {}

        existing = data["conversations"].get(conversation_id)
        data["conversations"][conversation_id] = {
            "conversation_id": conversation_id,
            "messages": messages,
            "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        _atomic_write_json(CONVERSATIONS_FILE, data)


def append_message(conversation_id: str, role: str, content: str) -> list:
    with _write_lock:
        data = _load_conversations()
        if "conversations" not in data:
            data["conversations"] = {}
        if conversation_id not in data["conversations"]:
            data["conversations"][conversation_id] = {
                "conversation_id": conversation_id,
                "messages": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        data["conversations"][conversation_id]["messages"].append({"role": role, "content": content})
        data["conversations"][conversation_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(CONVERSATIONS_FILE, data)
        return data["conversations"][conversation_id]["messages"]


def get_conversation_list() -> List[Dict[str, Any]]:
    """Get list of all conversations with metadata, sorted by updated_at desc."""
    data = _load_conversations()
    conversations = data.get("conversations", {})

    result = []
    for conv_id, conv_data in conversations.items():
        messages = conv_data.get("messages", [])
        first_msg = messages[0]["content"] if messages else ""
        title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg

        result.append({
            "conversation_id": conv_id,
            "title": title or "New conversation",
            "message_count": len(messages),
            "updated_at": conv_data.get("updated_at")
        })

    # Sort by updated_at descending. ISO-8601 strings sort lexicographically
    # in the same order as the underlying timestamps, so no datetime parse needed.
    result.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return result


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation. Returns True if deleted, False if not found."""
    with _write_lock:
        data = _load_conversations()
        if "conversations" not in data:
            return False
        if conversation_id not in data["conversations"]:
            return False

        del data["conversations"][conversation_id]
        _atomic_write_json(CONVERSATIONS_FILE, data)
        return True
