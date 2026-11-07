"""Inspection script for humans.

    python scripts/inspect_memory.py results/<run>/memory_sample_0.sqlite --notes 5
    python scripts/inspect_memory.py <db> --entity Caroline
    python scripts/inspect_memory.py <db> --revert <fact_id>

Prints a finished run's notes, or an entity's current and past facts, or undoes
the REPLACE or CORRECT that created a fact. --revert changes the database.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsom.store import Store  # noqa: E402


def show_entity(store: Store, entity: str):
    rows = store.conn.execute("SELECT DISTINCT sample_id FROM facts").fetchall()
    for sample_id, in rows:
        for f in store.entity_facts(sample_id, entity):
            print(f"{f.status:10} {f.entity} | {f.attribute} | {f.value}  step {f.observed_at}, "
                  f"{f.valid_from or '?'} to {f.valid_until or '?'}, {f.operation}, id={f.fact_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("--notes", type=int, default=5, metavar="N", help="print the first N notes")
    parser.add_argument("--entity", help="print this entity's facts instead of notes")
    parser.add_argument("--revert", metavar="FACT_ID", help="undo the replacement that created this fact")
    args = parser.parse_args()
    store = Store(args.db)
    try:
        if args.revert:
            for fact in store.revert_supersession(args.revert):
                print(f"current again: {fact.entity} | {fact.attribute} | {fact.value}")
        elif args.entity:
            show_entity(store, args.entity)
        else:
            for note in store.list_notes()[: args.notes]:
                print(f"[{note.timestamp}] {note.content}\n  keywords: {note.keywords}\n  "
                      f"tags: {note.tags}\n  context: {note.context}\n  links: {len(note.links)}")
    except (KeyError, ValueError) as e:
        print(f"not reverted: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
