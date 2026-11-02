"""SQLite storage for notes, facts and the operation log."""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from .contracts import Fact, MemoryOp, Note

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
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL COLLATE NOCASE,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    supersedes TEXT,
    source_note_id TEXT,
    status TEXT NOT NULL,
    conflict_with TEXT
);
CREATE INDEX IF NOT EXISTS facts_sp ON facts (subject, predicate);
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

    # ---- facts ----

    def add_fact(self, fact: Fact):
        self.conn.execute(
            "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact.id, fact.subject, fact.predicate, fact.object, fact.valid_from,
             fact.valid_to, fact.supersedes, fact.source_note_id, fact.status,
             fact.conflict_with),
        )

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return Fact(**dict(row)) if row else None

    # Subjects compare case-insensitively (COLLATE NOCASE) but are stored as given.
    # Facts come back in insertion order (rowid).

    def active_facts(self, subject: str, predicate: Optional[str] = None) -> List[Fact]:
        sql = "SELECT * FROM facts WHERE subject = ? AND status = 'active'"
        args = [subject]
        if predicate:
            sql += " AND predicate = ?"
            args.append(predicate)
        sql += " ORDER BY rowid"
        return [Fact(**dict(r)) for r in self.conn.execute(sql, args)]

    def fact_history(self, subject: str, predicate: Optional[str] = None) -> List[Fact]:
        """Every fact for a subject, current and retired, in insertion order."""
        sql = "SELECT * FROM facts WHERE subject = ?"
        args = [subject]
        if predicate:
            sql += " AND predicate = ?"
            args.append(predicate)
        sql += " ORDER BY rowid"
        return [Fact(**dict(r)) for r in self.conn.execute(sql, args)]

    def supersede(self, old_id: str, new_fact: Fact):
        """Retire an old fact and store the one that replaces it. Nothing is deleted."""
        self.conn.execute(
            "UPDATE facts SET status = 'retired', valid_to = ? WHERE id = ?",
            (new_fact.valid_from, old_id),
        )
        new_fact.supersedes = old_id
        self.add_fact(new_fact)

    def retire(self, fact_id: str, valid_to: str):
        """Retire a fact without a replacement (e.g. "I no longer like tea"). Nothing is deleted."""
        self.conn.execute(
            "UPDATE facts SET status = 'retired', valid_to = ? WHERE id = ?", (valid_to, fact_id),
        )

    def set_source_note(self, fact_ids: List[str], note_id: str):
        self.conn.executemany(
            "UPDATE facts SET source_note_id = ? WHERE id = ?", [(note_id, f) for f in fact_ids],
        )

    def revert_supersession(self, fact_id: str) -> Fact:
        """Human control: undo a supersession. The newer fact is rejected and the old one is current again."""
        new = self.get_fact(fact_id)
        if new is None or new.supersedes is None:
            raise ValueError(f"fact {fact_id} did not supersede anything")
        self.conn.execute("UPDATE facts SET status = 'rejected' WHERE id = ?", (fact_id,))
        self.conn.execute(
            "UPDATE facts SET status = 'active', valid_to = NULL WHERE id = ?", (new.supersedes,)
        )
        return self.get_fact(new.supersedes)

    def subjects(self) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT subject FROM facts GROUP BY subject ORDER BY MIN(rowid)")]

    # ---- operation log ----

    def log_op(self, op: MemoryOp):
        self.conn.execute("INSERT INTO ops (body) VALUES (?)", (op.model_dump_json(),))

    def ops(self) -> List[MemoryOp]:
        return [MemoryOp.model_validate_json(r[0])
                for r in self.conn.execute("SELECT body FROM ops ORDER BY seq")]
