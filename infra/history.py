"""Chatverlauf in SQLite, mit Zeitstempel pro Nachricht.

Ersetzt `SQLChatMessageHistory` aus langchain-community. Die Klasse kennt keine Zeitstempel,
und das Paket ist als abgekündigt markiert.
"""

import json
from datetime import datetime
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, delete, insert, select
from sqlalchemy.engine import Engine

_metadata = MetaData()

chat_messages = Table(
    "chat_messages",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(255), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False, default=datetime.now),
    Column("message", Text, nullable=False),
)


class SQLiteChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, engine: Engine):
        self.session_id = session_id
        self.engine = engine
        _metadata.create_all(engine)  # legt die Tabelle beim ersten Start an, sonst passiert nichts

    def _rows(self):
        stmt = (
            select(chat_messages.c.message, chat_messages.c.created_at)
            .where(chat_messages.c.session_id == self.session_id)
            .order_by(chat_messages.c.id)
        )
        with self.engine.connect() as conn:
            return conn.execute(stmt).fetchall()

    @property
    def messages(self) -> list[BaseMessage]:
        return messages_from_dict([json.loads(row.message) for row in self._rows()])

    def timestamped_messages(self) -> list[tuple[BaseMessage, datetime]]:
        """Nachrichten zusammen mit dem Zeitpunkt, an dem sie gespeichert wurden (für die Anzeige)."""
        rows = self._rows()
        msgs = messages_from_dict([json.loads(row.message) for row in rows])
        return [(msg, row.created_at) for msg, row in zip(msgs, rows)]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        now = datetime.now()
        with self.engine.begin() as conn:
            conn.execute(
                insert(chat_messages),
                [
                    {
                        "session_id": self.session_id,
                        "created_at": now,
                        "message": json.dumps(message_to_dict(m), ensure_ascii=False),
                    }
                    for m in messages
                ],
            )

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(chat_messages).where(chat_messages.c.session_id == self.session_id))
