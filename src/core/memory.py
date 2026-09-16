"""
PeopleQuery AI - Persistent Conversation Store & Memory Layer
Provides SQLite-backed thread-safe persistence for chat sessions, message histories, and follow-up contextual recall.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import get_settings


@dataclass
class ConversationMessage:
    id: str
    conversation_id: str
    role: str  # "user" | "assistant"
    content: str
    source: Optional[str] = None  # "sql" | "rag" | "master"
    category: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def metadata(self) -> Dict[str, Any]:
        if not self.metadata_json:
            return {}
        try:
            return json.loads(self.metadata_json)
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = self.metadata
        return d


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationStore:
    """
    SQLite-backed conversation memory store.
    Guarantees strict isolation across conversations and provides persistent storage
    for message histories and follow-ups.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            settings = get_settings()
            raw_url = settings.DATABASE_URL
            if raw_url.startswith("sqlite:///"):
                db_path = raw_url.replace("sqlite:///", "")
            else:
                db_path = "./data/hr_database.sqlite"

        self.db_path = str(db_path)
        # Ensure parent directory exists if using a file path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create conversation and message tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    category TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_id
                ON conversation_messages(conversation_id, created_at);
                """
            )
            conn.commit()

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation with a unique ID."""
        conv_id = conversation_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conv_title = title or "New Conversation"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?);
                """,
                (conv_id, conv_title, now, now),
            )
            conn.commit()

        return Conversation(
            id=conv_id,
            title=conv_title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Fetch conversation by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN conversation_messages m ON c.id = m.conversation_id
                WHERE c.id = ?
                GROUP BY c.id;
                """,
                (conversation_id,),
            ).fetchone()

            if not row:
                return None

            return Conversation(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )

    def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
        default_title: Optional[str] = None,
    ) -> Conversation:
        """Fetch existing conversation or create a new one."""
        if conversation_id:
            conv = self.get_conversation(conversation_id)
            if conv:
                return conv

        return self.create_conversation(
            conversation_id=conversation_id,
            title=default_title or "New Conversation",
        )

    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List recent conversations sorted by updated_at descending."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN conversation_messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

            return [
                Conversation(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=row["message_count"],
                )
                for row in rows
            ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and cascade delete its messages."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ?;",
                (conversation_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?;
                """,
                (title.strip(), now, conversation_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        source: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Store a message in the conversation history and update conversation timestamp."""
        # Ensure conversation exists
        self.get_or_create_conversation(conversation_id)

        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages
                (id, conversation_id, role, content, source, category, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (msg_id, conversation_id, role, content, source, category, meta_json, now),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?;
                """,
                (now, conversation_id),
            )
            conn.commit()

        # If this is the first user message, update title if title is generic
        if role == "user":
            conv = self.get_conversation(conversation_id)
            if conv and conv.title in ("New Conversation", "New Chat", "Conversation"):
                # Derive title from user query (max 40 chars)
                new_title = content.strip().replace("\n", " ")
                if len(new_title) > 40:
                    new_title = new_title[:37].rsplit(" ", 1)[0] + "..."
                if new_title:
                    self.update_conversation_title(conversation_id, new_title)

        return ConversationMessage(
            id=msg_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            source=source,
            category=category,
            metadata_json=meta_json,
            created_at=now,
        )

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """Fetch all messages for a given conversation in chronological order."""
        query = """
            SELECT id, conversation_id, role, content, source, category, metadata_json, created_at
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """
        params: list[Any] = [conversation_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                ConversationMessage(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    source=row["source"],
                    category=row["category"],
                    metadata_json=row["metadata_json"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def get_history_for_orchestrator(
        self,
        conversation_id: str,
        max_turns: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Format recent conversation turns for the MasterOrchestrator and Router.
        Returns up to `max_turns` recent messages.
        """
        messages = self.get_messages(conversation_id)
        if not messages:
            return []

        recent = messages[-max_turns:]
        return [
            {
                "role": m.role,
                "content": m.content,
                "source": m.source,
                "category": m.category,
            }
            for m in recent
        ]


# Default singleton instance
_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
