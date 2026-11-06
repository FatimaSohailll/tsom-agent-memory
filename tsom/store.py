"""SQLite storage for notes and the operation log."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from .contracts import Fact, MemoryOp, Note
from .temporal.normalize import norm

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
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    note_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    attr_key TEXT NOT NULL,
    observed_at INTEGER NOT NULL,
    observed_time TEXT,
    valid_from TEXT,
    valid_until TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    operation TEXT NOT NULL DEFAULT 'ADD',
    supersedes TEXT NOT NULL DEFAULT '[]',
    superseded_by TEXT,
    observation_count INTEGER NOT NULL DEFAULT 1,
    temporal_status TEXT NOT NULL DEFAULT 'UNKNOWN',
    raw_attribute TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts (sample_id, entity_key, attr_key);
CREATE INDEX IF NOT EXISTS idx_facts_note ON facts (sample_id, note_id);
CREATE TABLE IF NOT EXISTS note_versions (
    note_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    observed_at INTEGER,
    context TEXT,
    tags_json TEXT,
    links_json TEXT,
    changed_by_note_id TEXT,
    PRIMARY KEY (note_id, version)
);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Databases written before a facts column existed get it added."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(facts)")}
        for name, ddl in (("temporal_status", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
                          ("raw_attribute", "TEXT")):
            if name not in cols:
                self.conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {ddl}")

    @contextmanager
    def atomic(self):
        """Undo every write made inside the block if it raises (SQLite savepoint)."""
        self.conn.execute("SAVEPOINT atomic_op")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK TO atomic_op")
            self.conn.execute("RELEASE atomic_op")
            raise
        self.conn.execute("RELEASE atomic_op")

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

    # ---- facts (append-only) ----

    def add_fact(self, fact: Fact):
        """Insert a new fact row. Plain INSERT: an existing fact_id raises."""
        self.conn.execute(
            "INSERT INTO facts (fact_id, sample_id, note_id, entity, attribute, value,"
            " entity_key, attr_key, observed_at, observed_time, valid_from, valid_until,"
            " status, operation, supersedes, superseded_by, observation_count, temporal_status,"
            " raw_attribute)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact.fact_id, fact.sample_id, fact.note_id, fact.entity, fact.attribute,
             fact.value, norm(fact.entity), norm(fact.attribute), fact.observed_at,
             fact.observed_time, fact.valid_from, fact.valid_until, fact.status,
             fact.operation, json.dumps(fact.supersedes), fact.superseded_by,
             fact.observation_count, fact.temporal_status, fact.raw_attribute),
        )

    def update_fact_status(self, fact_id: str, status: str,
                           valid_until: Optional[str] = None,
                           superseded_by: Optional[str] = None,
                           observation_count: Optional[int] = None):
        """The only allowed fact update. Never touches entity/attribute/value.

        valid_until, superseded_by and observation_count are left unchanged
        when None.
        """
        if status not in ("active", "superseded", "retracted"):
            raise ValueError(f"invalid fact status: {status!r}")
        sets, args = ["status = ?"], [status]
        if valid_until is not None:
            sets.append("valid_until = ?")
            args.append(valid_until)
        if superseded_by is not None:
            sets.append("superseded_by = ?")
            args.append(superseded_by)
        if observation_count is not None:
            sets.append("observation_count = ?")
            args.append(observation_count)
        args.append(fact_id)
        cur = self.conn.execute(f"UPDATE facts SET {', '.join(sets)} WHERE fact_id = ?", args)
        if cur.rowcount == 0:
            raise KeyError(f"unknown fact_id: {fact_id}")

    def revert_supersession(self, fact_id: str) -> List[Fact]:
        """Undo the REPLACE or CORRECT that created `fact_id`: its old facts are current
        again and it is retracted. No row is deleted. Returns the restored facts.

        Only a fact that is still active can be undone. Reverting anything older would
        bring back a value that a later step has already moved on from.
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            raise KeyError(f"unknown fact_id: {fact_id}")
        if not fact.supersedes:
            raise ValueError(f"{fact_id} replaced nothing, so there is nothing to revert")
        if fact.status != "active":
            raise ValueError(f"{fact_id} is no longer active; revert the newest step first")
        with self.atomic():
            for old_id in fact.supersedes:
                self.conn.execute(
                    "UPDATE facts SET status = 'active', superseded_by = NULL, valid_until ="
                    " CASE WHEN valid_until = ? THEN NULL ELSE valid_until END WHERE fact_id = ?",
                    (fact.valid_from, old_id))
            self.update_fact_status(fact_id, "retracted", superseded_by=fact.supersedes[0])
        return [self.get_fact(old_id) for old_id in fact.supersedes]

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        row = self.conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        return self._fact(row) if row else None

    def active_facts(self, sample_id: str, entity: str, attribute: str) -> List[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE sample_id = ? AND entity_key = ? AND attr_key = ?"
            " AND status = 'active' ORDER BY observed_at, rowid",
            (sample_id, norm(entity), norm(attribute)))
        return [self._fact(r) for r in rows]

    def fact_history(self, sample_id: str, entity: str, attribute: str) -> List[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE sample_id = ? AND entity_key = ? AND attr_key = ?"
            " ORDER BY observed_at, rowid",
            (sample_id, norm(entity), norm(attribute)))
        return [self._fact(r) for r in rows]

    def facts_for_notes(self, sample_id: str, note_ids: List[str]) -> List[Fact]:
        if not note_ids:
            return []
        marks = ", ".join("?" * len(note_ids))
        rows = self.conn.execute(
            f"SELECT * FROM facts WHERE sample_id = ? AND note_id IN ({marks})"
            " ORDER BY observed_at, rowid", [sample_id, *note_ids])
        return [self._fact(r) for r in rows]

    def entity_keys(self, sample_id: str) -> List[str]:
        """Distinct normalized entity keys that have at least one fact."""
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT entity_key FROM facts WHERE sample_id = ?", (sample_id,))]

    def attribute_counts(self, sample_id: str, entity: Optional[str] = None) -> List[tuple]:
        """(attribute, number of facts) per distinct attribute key, most used first.
        The attribute is spelled as first stored. Limited to one entity if given."""
        where, args = "sample_id = ?", [sample_id]
        if entity is not None:
            where, args = where + " AND entity_key = ?", args + [norm(entity)]
        rows = self.conn.execute(
            "SELECT f.attribute, c.n FROM facts f JOIN ("
            f" SELECT COUNT(*) AS n, MIN(rowid) AS r FROM facts WHERE {where} GROUP BY attr_key"
            ") c ON f.rowid = c.r ORDER BY c.n DESC, c.r", args)
        return [(r[0], r[1]) for r in rows]

    def entity_facts(self, sample_id: str, entity: str) -> List[Fact]:
        """Every fact (all attributes, all statuses) about an entity."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE sample_id = ? AND entity_key = ?"
            " ORDER BY observed_at, rowid", (sample_id, norm(entity)))
        return [self._fact(r) for r in rows]

    @staticmethod
    def _fact(row) -> Fact:
        return Fact(
            fact_id=row["fact_id"], sample_id=row["sample_id"], note_id=row["note_id"],
            entity=row["entity"], attribute=row["attribute"], value=row["value"],
            observed_at=row["observed_at"], observed_time=row["observed_time"],
            valid_from=row["valid_from"], valid_until=row["valid_until"],
            status=row["status"], operation=row["operation"],
            supersedes=json.loads(row["supersedes"]), superseded_by=row["superseded_by"],
            observation_count=row["observation_count"], temporal_status=row["temporal_status"],
            raw_attribute=row["raw_attribute"])

    # ---- note versions ----

    def add_note_version(self, note_id: str, version: int, observed_at: Optional[int],
                         context: str, tags: List[str], links: List[str],
                         changed_by_note_id: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO note_versions (note_id, version, observed_at, context, tags_json,"
            " links_json, changed_by_note_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (note_id, version, observed_at, context, json.dumps(tags),
             json.dumps(links), changed_by_note_id))

    def note_versions(self, note_id: str) -> List[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM note_versions WHERE note_id = ? ORDER BY version", (note_id,)))
