from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY_DB = PROJECT_ROOT / "agent_history.sqlite3"
VALID_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str

    def as_chat_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    def __init__(self, db_path: Path = DEFAULT_HISTORY_DB) -> None:
        self.db_path = db_path
        self.initialize()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages (session_id, id)
                """
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session_id = validate_session_id(session_id)
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid message role: {role}")
        if not content:
            raise ValueError("message content must not be empty")

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, created_at),
            )

    def load_recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        session_id = validate_session_id(session_id)
        limit = max(0, min(limit, 50))
        if limit == 0:
            return []

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()

        return [StoredMessage(role=row[0], content=row[1]).as_chat_message() for row in rows]


def validate_session_id(session_id: str) -> str:
    session_id = session_id.strip()
    if not session_id:
        raise ValueError("session_id must not be empty")
    if len(session_id) > 120:
        raise ValueError("session_id must be 120 characters or fewer")
    return session_id
