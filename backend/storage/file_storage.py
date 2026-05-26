import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
CONVERSATIONS_FILE = STORAGE_DIR / "conversations.json"


def _ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_conversations() -> dict:
    _ensure_storage_dir()
    if not CONVERSATIONS_FILE.exists():
        return {"conversations": {}}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Failed to decode conversations.json, starting fresh")
        return {"conversations": {}}


def _save_conversations(data: dict) -> None:
    _ensure_storage_dir()
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_conversation(conversation_id: str) -> Optional[dict]:
    data = _load_conversations()
    conversations = data.get("conversations", {})
    return conversations.get(conversation_id)


def create_conversation(conversation_id: str) -> None:
    """Create an empty conversation entry so it appears in the conversation list."""
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
        _save_conversations(data)


def save_conversation(conversation_id: str, messages: list) -> None:
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
    _save_conversations(data)


def append_message(conversation_id: str, role: str, content: str) -> list:
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
    _save_conversations(data)
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

    # Sort by updated_at descending
    result.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return result


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation. Returns True if deleted, False if not found."""
    data = _load_conversations()
    if "conversations" not in data:
        return False
    if conversation_id not in data["conversations"]:
        return False

    del data["conversations"][conversation_id]
    _save_conversations(data)
    return True