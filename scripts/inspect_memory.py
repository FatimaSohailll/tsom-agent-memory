"""Inspection script for humans.

    python scripts/inspect_memory.py results/<run>/memory_sample_0.sqlite --notes 5

Prints the first N notes of a finished run's memory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsom.store import Store  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("--notes", type=int, default=5, metavar="N", help="print the first N notes")
    args = parser.parse_args()
    store = Store(args.db)
    for note in store.list_notes()[: args.notes]:
        print(f"[{note.timestamp}] {note.content}\n  keywords: {note.keywords}\n  "
              f"tags: {note.tags}\n  context: {note.context}\n  links: {len(note.links)}")
    store.close()


if __name__ == "__main__":
    main()
