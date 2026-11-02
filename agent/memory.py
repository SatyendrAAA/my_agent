"""Conversation memory: an in-RAM message list that can be saved to JSON."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Session:
    """One conversation thread."""

    session_id: str
    system_prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.updated_at = time.time()

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = time.time()

    def history(self) -> list[dict[str, Any]]:
        """Messages in the wire format, system prompt first."""
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=str(data.get("session_id") or new_session_id()),
            system_prompt=str(data.get("system_prompt") or ""),
            messages=list(data.get("messages") or []),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class SessionStore:
    """Reads and writes sessions as `<session_dir>/<id>.json`."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()

    def path_for(self, session_id: str) -> Path:
        if any(sep in session_id for sep in ("/", "\\", "..")):
            raise ValueError(f"invalid session id: {session_id!r}")
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
        if not safe:
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.directory / f"{safe}.json"

    def save(self, session: Session) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"no session {session_id!r} in {self.directory}")
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def latest(self) -> Session | None:
        items = self.list()
        return items[0] if items else None

    def list(self) -> list[Session]:
        if not self.directory.exists():
            return []
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                sessions.append(Session.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
