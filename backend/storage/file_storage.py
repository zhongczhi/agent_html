# backend/storage/file_storage.py
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
CONVERSATIONS_FILE = STORAGE_DIR / "conversations.json"


def _ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_conversations() -> dict:
    _ensure_storage_dir()
    if not CONVERSATIONS_FILE.exists():
        return {}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Failed to decode conversations.json, starting fresh")
        return {}


def _save_conversations(data: dict) -> None:
    _ensure_storage_dir()
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_conversation(conversation_id: str) -> Optional[dict]:
    data = _load_conversations()
    return data.get(conversation_id)


def save_conversation(conversation_id: str, messages: list) -> None:
    data = _load_conversations()
    data[conversation_id] = {"conversation_id": conversation_id, "messages": messages}
    _save_conversations(data)


def append_message(conversation_id: str, role: str, content: str) -> list:
    data = _load_conversations()
    if conversation_id not in data:
        data[conversation_id] = {"conversation_id": conversation_id, "messages": []}
    data[conversation_id]["messages"].append({"role": role, "content": content})
    _save_conversations(data)
    return data[conversation_id]["messages"]
