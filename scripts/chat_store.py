"""Small durable, local chat-history store for the Streamlit application."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils import _PROJECT_ROOT


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ChatStore:
    """Autosave conversations atomically and retain a compact local backup set."""

    def __init__(self, project_root: Optional[Path] = None, backup_limit: int = 30) -> None:
        root = Path(project_root or _PROJECT_ROOT).resolve()
        self.data_root = root / "data" / "cleaned_data"
        self.path = self.data_root / "chat_history.json"
        self.backup_root = self.data_root / "chat_backups"
        self.backup_limit = backup_limit

    def _empty(self) -> dict[str, Any]:
        return {"version": 1, "chats": {}}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        with self.path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("version") != 1 or not isinstance(state.get("chats"), dict):
            raise ValueError("Chat history has an unsupported format.")
        return state

    def _backup(self) -> None:
        if not self.path.exists():
            return
        self.backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(self.path, self.backup_root / f"chat_history_{stamp}.json")
        backups = sorted(self.backup_root.glob("chat_history_*.json"))
        for obsolete in backups[:-self.backup_limit]:
            obsolete.unlink()

    def _write(self, state: dict[str, Any], backup: bool = True) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        if backup:
            self._backup()
        descriptor, temp_path = tempfile.mkstemp(prefix="chat_history_", suffix=".tmp", dir=self.data_root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, self.path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def create(self, knowledge_bases: Optional[list[str]] = None) -> dict[str, Any]:
        state = self._load()
        chat_id = str(uuid.uuid4())
        chat = {
            "id": chat_id,
            "title": "New conversation",
            "created_at": _now(),
            "updated_at": _now(),
            "knowledge_bases": knowledge_bases or [],
            "messages": [],
        }
        state["chats"][chat_id] = chat
        self._write(state)
        return chat

    def get(self, chat_id: str) -> dict[str, Any]:
        try:
            return self._load()["chats"][chat_id]
        except KeyError as error:
            raise ValueError("Conversation no longer exists.") from error

    def list(self) -> list[dict[str, Any]]:
        return sorted(
            self._load()["chats"].values(),
            key=lambda chat: chat["updated_at"],
            reverse=True,
        )

    def update_knowledge_bases(self, chat_id: str, knowledge_bases: list[str]) -> None:
        state = self._load()
        chat = state["chats"][chat_id]
        chat["knowledge_bases"] = knowledge_bases
        chat["updated_at"] = _now()
        self._write(state)

    def append(self, chat_id: str, message: dict[str, Any]) -> None:
        state = self._load()
        chat = state["chats"][chat_id]
        chat["messages"].append(message)
        if message.get("role") == "user" and chat["title"] == "New conversation":
            chat["title"] = message.get("content", "New conversation")[:56]
        chat["updated_at"] = _now()
        self._write(state)

    def clear(self, chat_id: str) -> None:
        state = self._load()
        chat = state["chats"][chat_id]
        chat["messages"] = []
        chat["title"] = "New conversation"
        chat["updated_at"] = _now()
        self._write(state)

    def delete(self, chat_id: str) -> None:
        state = self._load()
        state["chats"].pop(chat_id, None)
        self._write(state)

    def create_backup(self) -> None:
        self._backup()
