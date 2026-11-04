"""SQLite storage for notes and the operation log."""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from .contracts import MemoryOp, Note

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    timestamp TEXT,
    keywords TEXT,
    tags TEXT,
    context TEXT,
    links TEXT
);
CREATE TABLE IF NOT EXISTS ops (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    body TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.commit()
        self.conn.close()

    # ---- notes ----

    def upsert_note(self, note: Note):
        self.conn.execute(
            "INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (note.id, note.content, note.timestamp, json.dumps(note.keywords),
             json.dumps(note.tags), note.context, json.dumps(note.links)),
        )

    def get_note(self, note_id: str) -> Optional[Note]:
        row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._note(row) if row else None

    def list_notes(self) -> List[Note]:
        return [self._note(r) for r in self.conn.execute("SELECT * FROM notes")]

    @staticmethod
    def _note(row) -> Note:
        return Note(id=row["id"], content=row["content"], timestamp=row["timestamp"],
                    keywords=json.loads(row["keywords"]), tags=json.loads(row["tags"]),
                    context=row["context"], links=json.loads(row["links"]))

    # ---- operation log ----

    def log_op(self, op: MemoryOp):
        self.conn.execute("INSERT INTO ops (body) VALUES (?)", (op.model_dump_json(),))

    def ops(self) -> List[MemoryOp]:
        return [MemoryOp.model_validate_json(r[0])
                for r in self.conn.execute("SELECT body FROM ops ORDER BY seq")]
